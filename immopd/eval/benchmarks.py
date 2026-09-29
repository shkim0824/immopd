"""Single-turn benchmarks of the paper: MedQA, CaseHOLD, FinQA, IFBench and IFEval.

Evaluation uses temperature 1, top-p 1 and no top-k. MedQA / CaseHOLD / FinQA use a 26,624-token completion
cap, IFBench and IFEval a 32,768-token cap. Raw benchmark files are read from ``${DATA_DIR}/raw``
(``immopd.data.download``); IFEval and IFBench prompts ship in ``data/eval``.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from immopd.common.io import data_root, iter_jsonl, repo_root
from immopd.eval import graders as G
from immopd.eval import if_grader

SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0)
RAW = os.environ.get("RAW_DIR", os.path.join(data_root(), "raw"))

MCQA_SUFFIX = ("\n\nPlease reason step by step, then give your final answer on the "
               "last line in the form \"Answer: <letter>\".")
NUM_SUFFIX = ("\n\nPlease reason step by step, then give the final numeric answer on "
              "the last line in the form \"Answer: <number>\".")
CASEHOLD_QUESTION = "Which holding statement best completes the CITATION in the following excerpt?"


def letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def mcqa_prompt(question: str, options: List[str], context: str = "") -> str:
    body = (context + "\n\n" if context else "") + question.strip() + "\n\n"
    body += "\n".join(f"{l}. {o}" for l, o in zip(letters(len(options)), options))
    return body + MCQA_SUFFIX


def finqa_context(r: Dict[str, Any]) -> str:
    pre = " ".join(r.get("pre_text", []))
    post = " ".join(r.get("post_text", []))
    table = "\n".join(" | ".join(str(c) for c in row) for row in r.get("table", []))
    return f"{pre}\n\nTable:\n{table}\n\n{post}"


def load_medqa(split: str = "test") -> List[Dict[str, Any]]:
    rows = []
    for i, r in enumerate(iter_jsonl(os.path.join(RAW, f"domains/med_medqa/phrases_no_exclude_{split}.jsonl"))):
        opts = [r["options"][k] for k in sorted(r["options"])]
        rows.append({"id": f"medqa_{i}", "kind": "mcqa", "n_options": len(opts), "gold": r["answer_idx"],
                     "prompt": mcqa_prompt(r["question"], opts)})
    return rows


def load_casehold(split: str = "test") -> List[Dict[str, Any]]:
    f = sorted(glob.glob(os.path.join(RAW, f"domains/law_casehold/data/*/{split}.csv")))[0]
    rows = []
    with open(f) as fh:
        rd = csv.reader(fh)
        next(rd)
        for rec in rd:
            ctx, holds, label = rec[1], rec[2:7], rec[-1]
            rows.append({"id": f"casehold_{rec[0]}", "kind": "mcqa", "n_options": 5,
                         "gold": letters(5)[int(float(label))],
                         "prompt": mcqa_prompt(CASEHOLD_QUESTION, holds, context=ctx)})
    return rows


def load_finqa(split: str = "test") -> List[Dict[str, Any]]:
    rows = []
    for r in json.load(open(os.path.join(RAW, f"third_party/FinQA/dataset/{split}.json"))):
        qa = r["qa"]
        rows.append({"id": f"finqa_{r['id']}", "kind": "numeric", "gold": qa.get("exe_ans", qa.get("answer")),
                     "prompt": finqa_context(r) + f"\n\nQuestion: {qa['question']}" + NUM_SUFFIX})
    return rows


def load_if(name: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(os.path.join(repo_root(), "data", "eval", name, "test.jsonl")))


def score_answers(rows: Sequence[Dict[str, Any]], samples: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """avg@k accuracy: the mean over the k samples of the accuracy over rows."""
    k = len(samples[0])
    per = []
    for j in range(k):
        ok = 0
        for r, outs in zip(rows, samples):
            if r["kind"] == "mcqa":
                ok += G.grade_mcqa(outs[j], r["gold"], r["n_options"])["correct"]
            else:
                ok += G.grade_fin_numeric(outs[j], r["gold"])["correct"]
        per.append(100.0 * ok / len(rows))
    return {"score": sum(per) / k, "metric": f"accuracy avg@{k}", "n": len(rows), "score_per_sample": per}


def score_if(bench: str) -> Callable:
    head = "prompt_level_strict_acc" if bench == "ifeval" else "prompt_level_loose_acc"

    def _score(rows, samples):
        k = len(samples[0])
        res = [if_grader.score_benchmark(bench, rows, [s[j] for s in samples]) for j in range(k)]
        out = {m: sum(r[m] for r in res) / k for m in
               ("prompt_level_strict_acc", "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc")}
        out.update(score=out[head], metric=f"{head} avg@{k}", n=len(rows), score_per_sample=[r[head] for r in res])
        return out
    return _score


class Bench:
    def __init__(self, name: str, load: Callable, score: Callable, max_tokens: int):
        self.name, self.load, self.score, self.max_tokens = name, load, score, max_tokens

    def messages(self, row: Dict[str, Any]) -> List[Dict[str, str]]:
        return [{"role": "user", "content": row["prompt"]}]


BENCHMARKS: Dict[str, Bench] = {
    "medqa": Bench("medqa", load_medqa, score_answers, 26624),
    "casehold": Bench("casehold", load_casehold, score_answers, 26624),
    "finqa": Bench("finqa", load_finqa, score_answers, 26624),
    "ifbench": Bench("ifbench", lambda: load_if("ifbench"), score_if("ifbench"), 32768),
    "ifeval": Bench("ifeval", lambda: load_if("ifeval"), score_if("ifeval"), 32768),
}
DEFAULT = ["medqa", "casehold", "finqa", "ifbench"]


def get(name: str, limit: Optional[int] = None):
    if name not in BENCHMARKS:
        raise SystemExit("unknown benchmark %r; known: %s" % (name, sorted(BENCHMARKS)))
    b = BENCHMARKS[name]
    rows = b.load()
    return b, rows[:limit] if limit else rows
