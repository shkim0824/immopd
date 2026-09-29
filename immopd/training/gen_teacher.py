"""Sample k completions per prompt with vLLM (SFT teacher data and SFT warm-up data).

  python -m immopd.training.gen_teacher --model <dir> --pool <pool.jsonl> --out <dir> --k 4 \
      --shard-rank 0 --num-shards 8

Pool rows are {id, input, output} or pre-rendered {id, prompt_text}. One process per GPU; every process
writes ``<out>/shardNNN.jsonl`` with rows {id, gold, meta, samples: [{text, finish}]}.
"""
from __future__ import annotations

import argparse
import json
import os

META = ("sub_domain", "source_dialog_id", "turn_index")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rank = args.shard_rank

    rows = [json.loads(l) for l in open(args.pool)]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i))
    mine = rows[rank::args.num_shards]
    out_f = os.path.join(args.out, f"shard{rank:03d}.jsonl")
    tmp = out_f + ".tmp"
    if os.path.exists(out_f):
        return
    done = sum(1 for _ in open(tmp)) if os.path.exists(tmp) else 0
    todo = mine[done:]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              trust_remote_code=True, enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed + 1000 * rank)
    limit = args.max_model_len - 2048
    with open(tmp, "a") as f:
        for s in range(0, len(todo), args.chunk):
            part = todo[s:s + args.chunk]
            prompts = [r.get("prompt_text") or tok.apply_chat_template(
                [{"role": "user", "content": r["input"]}], add_generation_prompt=True, tokenize=False,
                enable_thinking=True) for r in part]
            fits = [len(tok(t, add_special_tokens=False)["input_ids"]) <= limit for t in prompts]
            outs = iter(llm.generate([t for t, ok in zip(prompts, fits) if ok], sp))
            for r, ok in zip(part, fits):
                samples = [{"text": c.text, "finish": c.finish_reason} for c in next(outs).outputs] if ok else []
                meta = r.get("meta") or {k: r[k] for k in META if k in r}
                f.write(json.dumps({"id": r["id"], "gold": r.get("output"), "meta": meta, "samples": samples},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[gen {rank}] {done + s + len(part)}/{len(mine)}", flush=True)
    os.replace(tmp, out_f)


if __name__ == "__main__":
    main()
