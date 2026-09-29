"""Keep correct, deduplicated teacher samples as SFT rows (Medical, Law, Finance, IF).

  python -m immopd.training.filter_samples --domain med --gen-dir <dir> --pool <pool.jsonl> --out <sft.jsonl> --keep 3

A sample is kept if it finished, has one well-formed thinking block and is graded correct. Per prompt the
``--keep`` shortest samples are kept, skipping near-duplicates of an already kept sample. ``--pool-out``
writes the prompts that kept at least one sample.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
from collections import Counter

from immopd.eval import graders as G
from immopd.eval import if_grader

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def split_thinking(text: str):
    t = text.strip()
    if t.count("</think>") != 1:
        return None
    if t.startswith("<think>"):
        t = t[len("<think>"):]
    reasoning, answer = (x.strip() for x in t.split("</think>", 1))
    if len(reasoning) < 50 or len(answer) < 3:
        return None
    return reasoning, answer


def correct(domain: str, text: str, answer: str, row: dict) -> bool:
    if domain in ("med", "law"):
        return G.extract_letter(text, n_options=row["n_options"]) == str(row["output"]).strip().upper()
    if domain == "fin":
        return bool(G.grade_fin_numeric(answer, row["output"])["correct"])
    e = if_grader.evaluate_example("ifevalg", "", row["instruction_id_list"], row["kwargs"], text)
    return bool(e["prompt_strict"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["med", "law", "fin", "if"], required=True)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--sim", type=float, default=0.85)
    p.add_argument("--max-chars", type=int, default=70000)
    p.add_argument("--pool-out", default="")
    a = p.parse_args()

    pool = {}
    for i, l in enumerate(open(a.pool)):
        r = json.loads(l)
        pool[r.get("id", str(i))] = r
    stats = Counter()
    kept_prompts = []
    with open(a.out, "w") as fo:
        for sh in sorted(glob.glob(os.path.join(a.gen_dir, "shard*.jsonl"))):
            for line in open(sh):
                g = json.loads(line)
                row = pool[g["id"]]
                cjk_prompt = bool(CJK.search(row["input"]))
                stats["prompts"] += 1
                cands = []
                for s in g["samples"]:
                    txt = s["text"]
                    parts = split_thinking(txt)
                    if s.get("finish") != "stop" or len(txt) > a.max_chars or parts is None:
                        continue
                    reasoning, answer = parts
                    if not correct(a.domain, txt, answer, row):
                        continue
                    if CJK.search(reasoning + answer) and not (a.domain == "if" and cjk_prompt):
                        continue
                    if a.domain != "if" and "```" in reasoning + answer:
                        continue
                    cands.append((len(txt), reasoning, answer))
                cands.sort(key=lambda c: c[0])
                chosen = []
                for c in cands:
                    if all(difflib.SequenceMatcher(None, c[1][:2000], k[1][:2000]).ratio() < a.sim for k in chosen):
                        chosen.append(c)
                    if len(chosen) >= a.keep:
                        break
                stats["prompts_kept"] += bool(chosen)
                if chosen:
                    kept_prompts.append(row)
                for _, reasoning, answer in chosen:
                    fo.write(json.dumps({"messages": [{"role": "user", "content": row["input"]},
                                                      {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}],
                                         "meta": {"id": g["id"], "domain": a.domain}}, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
    if a.pool_out:
        with open(a.pool_out, "w") as f:
            for row in kept_prompts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
