"""Prompt pools of the single-turn domains (RL teachers, teacher data generation, MOPD).

  python -m immopd.data.prep_pools --out ${DATA_DIR}/pools [--only med,law,fin,if]

  med  MedQA train, four- and five-option prompts (written to med_candidates.jsonl; the MOPD pool
       med_train.jsonl holds the prompts of the Medical SFT data, see immopd.training.filter_samples)
  law  CaseHOLD train and bar-exam prompts
  fin  FinQA train and TAT-QA train (arithmetic / count) numeric-answer prompts
  if   instruction-following prompts of the rlvr1 blend of Nemotron-RL-Ultra-Training-Blends

Rows are {id, domain, input, output, source} plus ``n_options`` (med, law) or ``instruction_id_list`` /
``kwargs`` (if). Prompts that overlap a benchmark test prompt or a validation prompt are removed. For fin and
if, 100 prompts are written to ``<domain>_val.jsonl`` for validation during GRPO.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Set

from immopd.common.io import repo_root, write_jsonl
from immopd.eval.benchmarks import (CASEHOLD_QUESTION, NUM_SUFFIX, RAW, finqa_context, letters, load_casehold,
                                    load_finqa, load_medqa, mcqa_prompt)

_BOILER = re.compile(r"(please reason step by step.*$|^[A-J]\.\s.*$)", re.IGNORECASE | re.MULTILINE)


def grams(text: str, n: int = 13) -> Set[str]:
    toks = re.findall(r"\w+", _BOILER.sub(" ", text).lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def test_grams(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    g: Set[str] = set()
    for r in rows:
        g |= grams(r["prompt"])
    return g


def med_rows() -> Iterable[Dict[str, Any]]:
    four = [json.loads(l) for l in open(os.path.join(RAW, "domains/med_medqa/phrases_no_exclude_train.jsonl"))]
    for i, r in enumerate(four):
        opts = [r["options"][k] for k in sorted(r["options"])]
        yield {"id": f"med4-{i:05d}", "input": mcqa_prompt(r["question"], opts), "output": r["answer_idx"],
               "n_options": 4, "source": "MedQA-4opt"}
    import pyarrow.parquet as pq
    index = {r["question"].strip()[:200]: i for i, r in enumerate(four)}
    for f in sorted(glob.glob(os.path.join(RAW, "domains/med_medqa_5opt/**/train-*.parquet"), recursive=True)):
        for r in pq.read_table(f).to_pylist():
            i = index.get(r["question"].strip()[:200])
            if i is None:
                continue
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            yield {"id": f"med5-{i:05d}", "input": mcqa_prompt(r["question"], [o["value"] for o in opts]),
                   "output": r["answer_idx"].strip().upper(), "n_options": 5, "source": "MedQA-5opt"}


def law_rows() -> Iterable[Dict[str, Any]]:
    f = sorted(glob.glob(os.path.join(RAW, "domains/law_casehold/data/*/train.csv")))[0]
    with open(f) as fh:
        rd = csv.reader(fh)
        next(rd)
        for rec in rd:
            yield {"id": f"casehold-{rec[0]}", "input": mcqa_prompt(CASEHOLD_QUESTION, rec[2:7], context=rec[1]),
                   "output": letters(5)[int(float(rec[-1]))], "n_options": 5, "source": "CaseHOLD-train"}
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(RAW, "domains/law_barexam_qa/**/*.parquet"), recursive=True)) or \
        sorted(glob.glob(os.path.join(RAW, "domains/law_barexam_qa/**/*.csv"), recursive=True))
    df = pd.read_parquet(fs[0]) if fs[0].endswith(".parquet") else pd.read_csv(fs[0])
    cols = {c.lower(): c for c in df.columns}
    q, ans = cols.get("question") or cols["prompt"], cols.get("answer") or cols["gold"]
    choices = [cols[k] for k in ("choice_a", "choice_b", "choice_c", "choice_d")]
    for j, r in df.iterrows():
        gold = str(r[ans]).strip().upper()[:1]
        if gold in "ABCD":
            yield {"id": f"barexam-{j}", "input": mcqa_prompt(str(r[q]), [str(r[c]) for c in choices]),
                   "output": gold, "n_options": 4, "source": "barexam_qa"}


def fin_rows() -> Iterable[Dict[str, Any]]:
    for r in json.load(open(os.path.join(RAW, "third_party/FinQA/dataset/train.json"))):
        qa = r["qa"]
        yield {"id": f"finqa-{r['id']}", "input": finqa_context(r) + f"\n\nQuestion: {qa['question']}" + NUM_SUFFIX,
               "output": str(qa.get("exe_ans", qa.get("answer"))), "source": "FinQA-train"}
    for doc in json.load(open(os.path.join(RAW, "third_party/TAT-QA/dataset_raw/tatqa_dataset_train.json"))):
        table = "\n".join(" | ".join(str(c) for c in row) for row in doc["table"]["table"])
        paras = "\n".join(p["text"] for p in doc["paragraphs"])
        for q in doc["questions"]:
            ans = q["answer"][0] if isinstance(q["answer"], list) and q["answer"] else q["answer"]
            if q.get("answer_type") in ("arithmetic", "count") and ans is not None and ans != []:
                yield {"id": f"tatqa-{q['uid']}", "input": f"Table:\n{table}\n\n{paras}\n\nQuestion: {q['question']}" + NUM_SUFFIX,
                       "output": str(ans), "source": "TATQA-train"}


def if_rows() -> Iterable[Dict[str, Any]]:
    src = glob.glob(os.path.join(RAW, "hf/nvidia__Nemotron-RL-Ultra-Training-Blends/**/rlvr1*.jsonl"), recursive=True)[0]
    for i, line in enumerate(open(src)):
        r = json.loads(line)
        if (r.get("agent_ref") or {}).get("name") != "instruction_following_simple_agent":
            continue
        inp = (r.get("responses_create_params") or {}).get("input") or []
        prompt = (inp[0].get("content") if inp else "") or r.get("prompt")
        yield {"id": f"if-{i:06d}", "input": prompt, "output": "", "source": "rlvr1",
               "instruction_id_list": r.get("instruction_id_list") or [], "kwargs": r.get("kwargs") or [],
               "messages": [{"role": "user", "content": prompt}]}


BUILDERS = {"med": (med_rows, load_medqa), "law": (law_rows, load_casehold), "fin": (fin_rows, load_finqa),
            "if": (if_rows, None)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="med,law,fin,if")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    val = {json.loads(l)["input"].strip() for l in open(os.path.join(repo_root(), "data", "validation", "pool.jsonl"))}
    for domain in args.only.split(","):
        build, load_test = BUILDERS[domain]
        held = test_grams(load_test()) if load_test else set()
        rows: List[Dict[str, Any]] = []
        for r in build():
            if r["input"].strip() in val or (held and grams(r["input"]) & held):
                continue
            rows.append(dict(r, domain=domain))
        if domain != "if":
            random.Random(args.seed).shuffle(rows)
        if domain in ("fin", "if"):
            write_jsonl(os.path.join(args.out, f"{domain}_val.jsonl"), rows[-100:])
            rows = rows[:-100]
        path = os.path.join(args.out, "med_candidates.jsonl" if domain == "med" else f"{domain}_train.jsonl")
        print(domain, write_jsonl(path, rows), "->", path)


if __name__ == "__main__":
    main()
