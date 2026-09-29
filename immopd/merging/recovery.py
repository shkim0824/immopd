"""Merge rule of IM-MOPD (Algorithm 1): merge every domain whose validation recovery is at or below gamma.

    r_i = (s_i(theta) - s_i(theta_ref)) / (s_i(phi_i) - s_i(theta_ref))        M = {i | r_i <= gamma}

  python -m immopd.merging.recovery refs --base-acc <acc.json> --base-tau2 <metrics.json> \
      --teacher-acc med=<acc.json>,law=...,fin=...,if=... --teacher-tau2 <metrics.json> --out refs.json
  python -m immopd.merging.recovery plan --refs refs.json --acc <acc.json> --tau2 <metrics.json> \
      --step 50 --gamma 0.6 --delta 0.3 --out PLAN_50.json
"""
from __future__ import annotations

import argparse
import json
import os

DOMAINS = ["tau", "med", "law", "fin", "if"]


def tau2_score(metrics_json: str) -> float:
    b = json.load(open(metrics_json))["benchmarks"]
    return float(b[next(k for k in b if k.startswith("tau2"))]["score"])


def validation_scores(acc_json: str, tau2_metrics: str | None = None, domains=None) -> dict:
    acc = json.load(open(acc_json))["domains"]
    out = {d: float(v["acc"]) for d, v in acc.items() if domains is None or d in domains}
    if tau2_metrics:
        out["tau"] = tau2_score(tau2_metrics)
    return out


def recovery(refs: dict, scores: dict) -> dict:
    return {d: (scores[d] - refs["base"][d]) / (refs["teacher"][d] - refs["base"][d])
            for d in DOMAINS if d in scores and d in refs["base"]}


def decide(refs: dict, scores: dict, gamma: float) -> tuple:
    r = recovery(refs, scores)
    return r, [d for d in r if r[d] <= gamma]


def _write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def _pairs(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.split(",") if kv)


def cmd_refs(a):
    base = validation_scores(a.base_acc, a.base_tau2)
    teacher = {d: validation_scores(p, domains=(d,))[d] for d, p in _pairs(a.teacher_acc).items()}
    if a.teacher_tau2:
        teacher["tau"] = tau2_score(a.teacher_tau2)
    for d in base:
        assert teacher[d] > base[d], (d, teacher[d], base[d])
    _write(a.out, {"base": base, "teacher": {d: teacher[d] for d in base}})


def cmd_plan(a):
    scores = validation_scores(a.acc, a.tau2)
    r, picked = decide(json.load(open(a.refs)), scores, a.gamma)
    adds = ",".join("%s=%s" % (d, a.delta) for d in picked)
    _write(a.out, {"step": a.step, "gamma": a.gamma, "delta": a.delta, "scores": scores,
                   "recovery": {d: round(v, 4) for d, v in r.items()}, "picked": picked, "adds": adds})
    print("[recovery] step %d: %s -> merge %s" % (a.step, {d: round(v, 3) for d, v in r.items()}, picked or "none"), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--base-acc", required=True)
    r.add_argument("--base-tau2")
    r.add_argument("--teacher-acc", required=True)
    r.add_argument("--teacher-tau2")
    r.add_argument("--out", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--refs", required=True)
    q.add_argument("--acc", required=True)
    q.add_argument("--tau2")
    q.add_argument("--step", type=int, required=True)
    q.add_argument("--gamma", type=float, required=True)
    q.add_argument("--delta", type=float, required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"refs": cmd_refs, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    main()
