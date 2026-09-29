"""Student-prefix teacher-continuation on MedQA.

For every recorded student response of length L (``medqa.gen.jsonl`` of ``immopd.eval.run_all``), the teacher
continues after the first floor(alpha * L) student tokens; prefix + continuation is graded as a MedQA answer.

  python -m immopd.eval.prefix.medical --students name=<dir>/medqa.gen.jsonl,... --teacher <dir> --out <dir> --gpus 0-7
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from immopd.eval.benchmarks import SAMPLING, get


def load_items(students, alphas, tok):
    items = []
    for name, path in students:
        for line in open(path):
            r = json.loads(line)
            for j, s in enumerate(r["samples"]):
                ids = tok(s["text"], add_special_tokens=False)["input_ids"]
                for al in alphas:
                    k = min(int(math.floor(al * s["n_tokens"])), len(ids))
                    items.append({"student": name, "alpha": al, "idx": r["idx"], "sample": j, "k": k, "prefix_ids": ids[:k]})
    return items


def key(d):
    return (d["student"], d["alpha"], d["idx"], d["sample"])


def worker(a, students, alphas):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from immopd.common import chat

    bench, rows = get("medqa")
    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True))
    path = os.path.join(a.out, "shards", f"shard_{a.shard}.jsonl")
    done = {key(json.loads(l)) for l in open(path)} if os.path.exists(path) else set()
    items = [it for it in load_items(students, alphas, tok)[a.shard::a.nshards] if key(it) not in done]
    if not items:
        return
    max_len = min(a.max_model_len, int(json.load(open(os.path.join(a.teacher, "config.json")))
                                       .get("max_position_embeddings", a.max_model_len)))
    llm = LLM(model=a.teacher, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=a.seed)
    stop = chat.eos_token_ids(tok)
    prompts, sps = [], []
    for it in items:
        p = chat.render_prompt_ids(tok, bench.messages(rows[it["idx"]]), thinking=True) + it["prefix_ids"]
        budget = max(16, min(bench.max_tokens - it["k"], max_len - len(p)))
        prompts.append(TokensPrompt(prompt_token_ids=p))
        sps.append(SamplingParams(n=1, max_tokens=budget, seed=a.seed, stop_token_ids=stop, skip_special_tokens=True,
                                  **SAMPLING))
    outs = llm.generate(prompts, sps, use_tqdm=True)
    with open(path, "a") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({**{k: it[k] for k in ("student", "alpha", "idx", "sample", "k")},
                                "text": tok.decode(it["prefix_ids"], skip_special_tokens=True) + o.outputs[0].text}) + "\n")


def grade(a, students, alphas):
    bench, rows = get("medqa")
    recs = {}
    for fn in sorted(os.listdir(os.path.join(a.out, "shards"))):
        for l in open(os.path.join(a.out, "shards", fn)):
            d = json.loads(l)
            recs[key(d)] = d["text"]
    out = {"teacher": a.teacher, "students": {}}
    for name, path in students:
        gens = [json.loads(l) for l in open(path)]
        n = len(gens[0]["samples"])
        res = {"1.0": bench.score(rows, [[s["text"] for s in g["samples"]] for g in gens])["score"]}
        for al in alphas:
            texts = [[recs[(name, al, i, j)] for j in range(n)] for i in range(len(rows))]
            res[str(al)] = bench.score(rows, texts)["score"]
        out["students"][name] = res
        print(name, res, flush=True)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", required=True, help="name=<medqa.gen.jsonl>,...")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--alphas", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    students = [tuple(x.split("=", 1)) for x in a.students.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    os.makedirs(os.path.join(a.out, "shards"), exist_ok=True)
    if a.worker:
        return worker(a, students, alphas)
    from immopd.eval.run_all import parse_gpus
    gpus = parse_gpus(a.gpus)
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, "-m", "immopd.eval.prefix.medical", "--worker", "--shard", str(i), "--nshards", str(len(gpus)),
               "--students", a.students, "--teacher", a.teacher, "--alphas", a.alphas, "--out", a.out,
               "--seed", str(a.seed), "--max-model-len", str(a.max_model_len)]
        log = open(os.path.join(a.out, "shards", f"worker_{i}.log"), "a")
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), stdout=log,
                                      stderr=subprocess.STDOUT))
    assert all(p.wait() == 0 for p in procs), "workers failed; see %s/shards/worker_*.log" % a.out
    grade(a, students, alphas)


if __name__ == "__main__":
    sys.exit(main())
