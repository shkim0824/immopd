"""Validation accuracy for the merge rule of IM-MOPD.

The validation set holds 256 prompts per single-turn domain, held out from the training pools
(``data/validation/pool.jsonl``). Tool Use is validated with tau2 on a task list
(``immopd.eval.tau2_run --task-ids-file data/validation/tau2_tasks.json``).

  python -m immopd.eval.validation run --model <checkpoint> --out <dir> --gpus 0-7      -> <dir>/acc.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_POOL = os.path.join(REPO, "data", "validation", "pool.jsonl")
DOMS = ["med", "law", "fin", "if"]
MAX_TOKENS = {"med": 24576, "law": 16384, "fin": 32768, "if": 32768}
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
CHUNK = 256


def load(pool_path: str) -> dict:
    rows = {d: [] for d in DOMS}
    for l in open(pool_path):
        r = json.loads(l)
        if r["domain"] in rows:
            rows[r["domain"]].append(r)
    return rows


def grade(dom: str, rows: list, texts: list) -> list:
    from immopd.eval import graders as G
    from immopd.eval import if_grader
    if dom in ("med", "law"):
        return [bool(G.grade_mcqa(t, r["output"], r["n_options"])["correct"]) for t, r in zip(texts, rows)]
    if dom == "fin":
        return [bool(G.grade_fin_numeric(t, r["output"])["correct"]) for t, r in zip(texts, rows)]
    irows = [{"prompt": r["input"], "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]} for r in rows]
    return [bool(e["prompt_strict"]) for e in if_grader.score_benchmark("ifevalg", irows, texts)["per_example"]]


def gen(a) -> None:
    rows = load(a.pool)
    doms = [d for d in a.doms.split(",") if d]
    tasks = [(d, i, k) for d in doms for i in range(len(rows[d])) for k in range(a.n)]
    mine = tasks[a.worker::a.workers]
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, f"part_{a.worker}.jsonl")
    done = set()
    if os.path.exists(fp):
        for l in open(fp):
            try:
                x = json.loads(l)
                done.add((x["d"], x["i"], x["k"]))
            except json.JSONDecodeError:
                pass
    todo = [t for t in mine if t not in done]
    if not todo:
        return
    from immopd.eval.generate import VllmGenerator
    g = VllmGenerator(a.model, max_model_len=40960, gpu_mem_util=a.gpu_mem_util, seed=0)
    with open(fp, "a") as f:
        for c in range(0, len(todo), CHUNK):
            part = todo[c:c + CHUNK]
            items = [{"messages": [{"role": "user", "content": rows[d][i]["input"]}], "n": 1,
                      "max_tokens": MAX_TOKENS[d], "sampling": SAMPLING, "seed": k} for d, i, k in part]
            for (d, i, k), o in zip(part, g.generate(items)):
                f.write(json.dumps({"d": d, "i": i, "k": k, "text": o[0]["text"], "n_tokens": o[0]["n_tokens"],
                                    "finish": o[0]["finish_reason"]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[val] worker {a.worker}: {c + len(part)}/{len(todo)}", flush=True)


def grade_stage(a) -> None:
    rows = load(a.pool)
    got = {}
    for fn in sorted(os.listdir(a.out)):
        if fn.startswith("part_") and fn.endswith(".jsonl"):
            for l in open(os.path.join(a.out, fn)):
                x = json.loads(l)
                got[(x["d"], x["i"], x["k"])] = x
    doms = sorted({d for d, _, _ in got}, key=DOMS.index)
    ks = sorted({k for _, _, k in got})
    res = {"model": a.model, "pool": a.pool, "n_samples": len(ks), "domains": {}}
    for d in doms:
        n = len(rows[d])
        assert all((d, i, k) in got for i in range(n) for k in ks), (d, "missing generations")
        acc_k = [100.0 * sum(grade(d, rows[d], [got[(d, i, k)]["text"] for i in range(n)])) / n for k in ks]
        res["domains"][d] = {"acc": sum(acc_k) / len(acc_k), "acc_by_sample": acc_k, "n_rows": n}
        print(f"[val] {d}: {res['domains'][d]['acc']:.2f}", flush=True)
    json.dump(res, open(os.path.join(a.out, "acc.json"), "w"), indent=1)


def run_stage(a) -> None:
    from immopd.eval.run_all import parse_gpus
    gpus = parse_gpus(a.gpus)
    os.makedirs(a.out, exist_ok=True)
    procs = []
    for w, g in enumerate(gpus):
        cache = os.path.join(a.out, "_cache", str(w))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   VLLM_CACHE_ROOT=cache + "/vllm", TORCHINDUCTOR_CACHE_DIR=cache + "/inductor",
                   TRITON_CACHE_DIR=cache + "/triton")
        cmd = [sys.executable, "-m", "immopd.eval.validation", "gen", "--model", a.model, "--out", a.out,
               "--pool", a.pool, "--doms", a.doms, "--n", str(a.n), "--worker", str(w), "--workers", str(len(gpus)),
               "--gpu-mem-util", str(a.gpu_mem_util)]
        logf = open(os.path.join(a.out, "worker_%d.log" % w), "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT))
    rc = [pr.wait() for pr in procs]
    assert all(x == 0 for x in rc), "validation workers failed (see %s/worker_*.log)" % a.out
    grade_stage(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "gen", "grade"])
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--doms", default=",".join(DOMS))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    a = ap.parse_args(argv)
    {"run": run_stage, "gen": gen, "grade": grade_stage}[a.stage](a)


if __name__ == "__main__":
    main()
