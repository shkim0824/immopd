"""Score a tau2-bench results file: pass@1 over all tasks, with errors counted as failures.

  python -m immopd.eval.tau2_metrics --results <results.json> --out <dir>      (tau2-bench python)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--split", default="base")
    a = ap.parse_args(argv)

    from tau2.data_model.simulation import Results
    from tau2.metrics.agent_metrics import is_successful

    res = Results.load(Path(a.results))
    sims = res.simulations
    task_ids = [t.id for t in res.tasks]
    trials = max(collections.Counter(s.task_id for s in sims).values()) if sims else 1
    wins = collections.Counter()
    for s in sims:
        if s.reward_info is not None and is_successful(float(s.reward_info.reward)):
            wins[s.task_id] += 1
    score = 100.0 * sum(wins[t] / trials for t in task_ids) / max(len(task_ids), 1)
    term = collections.Counter(str(getattr(s.termination_reason, "value", s.termination_reason)) for s in sims)
    bench = "tau2_%s" % a.domain
    metrics = {
        "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {bench: {"score": score, "metric": "pass@1", "split": a.split, "n_tasks": len(task_ids),
                               "n_trials": trials, "n_sims": len(sims), "termination_reasons": dict(term)}},
        "scores": {bench: score},
    }
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "metrics.json")
    json.dump(metrics, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    print(json.dumps(metrics["scores"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
