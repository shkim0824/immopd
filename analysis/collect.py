"""Scores and average normalized score of evaluated checkpoints.

  python analysis/collect.py --eval-root ${RUN_DIR}/eval --base <tag> \
      --teachers med=<tag>,law=<tag>,fin=<tag>,if=<tag>,tau=<tag> --tags <tag>,...

``<eval-root>/<tag>/metrics.json`` holds the single-turn benchmarks (``scripts/eval.sh``) and
``<eval-root>/tau2-<tag>/metrics.json`` tau2 (``scripts/eval_tau2.sh``).
"""
import argparse
import json
import os

COLS = ["medqa", "casehold", "finqa", "ifbench", "tau2_telecom"]
DOMAIN = {"medqa": "med", "casehold": "law", "finqa": "fin", "ifbench": "if", "ifeval": "if", "tau2_telecom": "tau"}


def scores_of(root: str, tag: str) -> dict:
    out = {}
    for p in (os.path.join(root, tag, "metrics.json"), os.path.join(root, "tau2-" + tag, "metrics.json")):
        if os.path.exists(p):
            out.update({b: float(v["score"]) for b, v in json.load(open(p))["benchmarks"].items()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--teachers", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--cols", default=",".join(COLS))
    a = p.parse_args()
    cols = a.cols.split(",")
    base = scores_of(a.eval_root, a.base)
    tmap = dict(kv.split("=", 1) for kv in a.teachers.split(","))
    teacher = {c: scores_of(a.eval_root, tmap[DOMAIN[c]])[c] for c in cols}
    for t in a.tags.split(","):
        s = scores_of(a.eval_root, t)
        norm = sum(100.0 * (s[c] - base[c]) / (teacher[c] - base[c]) for c in cols) / len(cols)
        print(json.dumps({"tag": t, "scores": {c: round(s[c], 1) for c in cols}, "norm": round(norm, 1)}))


if __name__ == "__main__":
    main()
