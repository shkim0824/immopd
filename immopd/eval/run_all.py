"""Single-turn evaluation: one vLLM engine per GPU serves all requested benchmarks.

  python -m immopd.eval.run_all --model <checkpoint> --out <dir> --gpus 0-7 \
      --benchmarks medqa,casehold,finqa,ifbench --n 3 --seed 42
  python -m immopd.eval.run_all --aggregate-only --model <checkpoint> --out <dir> --benchmarks ...

avg@n draws sample j of every prompt with seed ``seed + j``. Results are written to ``<out>/metrics.json``.
An interrupted run resumes from the generations already on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence

from immopd.common.io import read_jsonl, write_json, write_jsonl
from immopd.eval.benchmarks import DEFAULT, SAMPLING, get

PASS = ["model", "tokenizer", "out", "benchmarks", "max_model_len", "gpu_mem_util", "seed", "seqs_per_call", "limit", "n"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--benchmarks", default=",".join(DEFAULT))
    p.add_argument("--gpus", default="0-7")
    p.add_argument("--n", type=int, default=3, help="samples per prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--seqs-per-call", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="first K rows of every benchmark")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shard", default=None)
    p.add_argument("--out-shard", default=None)
    return p


def parse_gpus(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def load(args) -> Dict[str, Any]:
    out = {}
    for name in [b for b in args.benchmarks.split(",") if b]:
        out[name] = get(name, args.limit)
        print("[eval] %-10s %6d rows" % (name, len(out[name][1])), flush=True)
    return out


def shard_tasks(args, benches: Dict[str, Any], n_shards: int) -> List[List[Dict[str, Any]]]:
    tasks = [{"b": name, "i": i, "si": si, "w": b.max_tokens}
             for name, (b, rows) in benches.items() for i in range(len(rows)) for si in range(args.n)]
    tasks.sort(key=lambda t: (-t["w"], t["b"], t["i"], t["si"]))
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(n_shards)]
    load_ = [0.0] * n_shards
    for t in tasks:
        j = min(range(n_shards), key=lambda x: load_[x])
        shards[j].append({k: v for k, v in t.items() if k != "w"})
        load_[j] += t["w"]
    return shards


def key(t: Dict[str, Any]):
    return (t["b"], t["i"], t["si"])


def run_worker(args) -> None:
    from immopd.eval.generate import VllmGenerator

    tasks = read_jsonl(args.shard)
    open(args.out_shard, "a").close()
    done = {key(r) for r in read_jsonl(args.out_shard)}
    todo = [t for t in tasks if key(t) not in done]
    print("[worker] %d tasks, %d done" % (len(tasks), len(done)), flush=True)
    if not todo:
        return
    benches = load(args)
    gen = VllmGenerator(args.model, args.tokenizer, max_model_len=args.max_model_len,
                        gpu_mem_util=args.gpu_mem_util, seed=args.seed)
    for pos in range(0, len(todo), args.seqs_per_call):
        chunk = todo[pos:pos + args.seqs_per_call]
        items = []
        for t in chunk:
            b, rows = benches[t["b"]]
            items.append({"messages": b.messages(rows[t["i"]]), "n": 1, "max_tokens": b.max_tokens,
                          "sampling": SAMPLING, "seed": args.seed + t["si"]})
        outs = gen.generate(items)
        with open(args.out_shard, "a") as f:
            for t, o in zip(chunk, outs):
                f.write(json.dumps(dict(t, text=o[0]["text"], n_tokens=o[0]["n_tokens"],
                                        finish_reason=o[0]["finish_reason"]), ensure_ascii=False) + "\n")
        print("[worker] %d/%d" % (pos + len(chunk), len(todo)), flush=True)


def merge(args, benches: Dict[str, Any], n_shards: int) -> None:
    got: Dict[str, Dict[int, Dict[int, Any]]] = {b: {} for b in benches}
    for gi in range(n_shards):
        for r in read_jsonl(os.path.join(args.out, "_shards", "out_%d.jsonl" % gi)):
            if r["b"] in got:
                got[r["b"]].setdefault(r["i"], {})[r["si"]] = r
    for name, (b, rows) in benches.items():
        out = []
        for i, r in enumerate(rows):
            slot = got[name].get(i, {})
            assert sorted(slot) == list(range(args.n)), "%s[%d]: samples %s" % (name, i, sorted(slot))
            out.append({"idx": i, "id": r.get("id") or r.get("key"),
                        "samples": [{k: slot[j][k] for k in ("text", "n_tokens", "finish_reason")} for j in range(args.n)]})
        write_jsonl(os.path.join(args.out, "%s.gen.jsonl" % name), out)


def grade(args, benches: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"model": args.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "protocol": dict(SAMPLING, n=args.n, seed=args.seed), "benchmarks": {}}
    for name, (b, rows) in benches.items():
        gens = read_jsonl(os.path.join(args.out, "%s.gen.jsonl" % name))
        assert len(gens) == len(rows), "%s: %d generations for %d rows" % (name, len(gens), len(rows))
        res = b.score(rows, [[s["text"] for s in g["samples"]] for g in gens])
        flat = [s for g in gens for s in g["samples"]]
        res.update(max_tokens=b.max_tokens, avg_tokens=sum(s["n_tokens"] for s in flat) / len(flat),
                   truncated_frac=sum(s["finish_reason"] == "length" for s in flat) / len(flat))
        metrics["benchmarks"][name] = res
        print("[eval] %-10s %s = %.2f" % (name, res["metric"], res["score"]), flush=True)
    metrics["scores"] = {b: r["score"] for b, r in metrics["benchmarks"].items()}
    write_json(os.path.join(args.out, "metrics.json"), metrics)
    return metrics


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        run_worker(args)
        return 0
    os.makedirs(args.out, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    benches = load(args)
    shard_dir = os.path.join(args.out, "_shards")
    if not args.aggregate_only:
        os.makedirs(shard_dir, exist_ok=True)
        shards = shard_tasks(args, benches, len(gpus))
        procs = []
        for gi, g in enumerate(gpus):
            sp = os.path.join(shard_dir, "shard_%d.jsonl" % gi)
            op = os.path.join(shard_dir, "out_%d.jsonl" % gi)
            if not os.path.exists(sp):
                write_jsonl(sp, shards[gi])
            cache = os.path.join(args.out, "_cache", str(gi))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), VLLM_CACHE_ROOT=cache + "/vllm",
                       TORCHINDUCTOR_CACHE_DIR=cache + "/inductor", TRITON_CACHE_DIR=cache + "/triton")
            cmd = [sys.executable, "-m", "immopd.eval.run_all", "--worker", "--shard", sp, "--out-shard", op]
            for k in PASS:
                v = getattr(args, k)
                if v is not None:
                    cmd += ["--%s" % k.replace("_", "-"), str(v)]
            log = open(os.path.join(shard_dir, "worker_%d.log" % gi), "a")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        failed = 0
        for p, log in procs:
            failed += p.wait() != 0
            log.close()
        if failed:
            raise SystemExit("%d worker(s) failed; see %s/worker_*.log" % (failed, shard_dir))
    if not all(os.path.exists(os.path.join(args.out, "%s.gen.jsonl" % b)) for b in benches):
        merge(args, benches, len(gpus))
    print(json.dumps(grade(args, benches)["scores"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
