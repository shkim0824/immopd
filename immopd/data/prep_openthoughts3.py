"""Tokenize and pack OpenThoughts3-1.2M into a memory-mapped SFT cache (CPU only).

  python -m immopd.data.prep_openthoughts3 --src ${DATA_DIR}/raw/hf/open-thoughts__OpenThoughts3-1.2M \
      --out ${DATA_DIR}/ot3_cache_len16384 --model ${MODEL_DIR}/Qwen3-4B-Base --max-len 16384

The prompt is rendered with the chat template; loss covers the last assistant turn. Completions longer than
``--max-len`` are truncated on the right and stored without the end-of-turn token.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

from immopd.training.ot3_cache import (BIN_OFFSETS, EX_PROMPT_LENS, EX_STARTS, STATS_JSON, TOKENS_BIN,
                                pack_bfd)

ROLE = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}

STAT_KEYS = ("n_in", "n_bad_roles", "n_multi_turn", "n_missing_think_close",
             "n_truncated_completion", "n_dropped_prompt_too_long", "n_kept")

_W: Dict[str, Any] = {}


def _init_worker(model_path: str, max_len: int, tmp_dir: str):
    from transformers import AutoTokenizer

    from immopd.common import chat
    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
    _W.update(tok=tok, eos=tok.convert_tokens_to_ids(chat.IM_END), max_len=max_len, tmp_dir=tmp_dir)


def _do_shard(task: Tuple[int, str]) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    from immopd.common import chat
    k, path = task
    tok, eos, max_len, tmp_dir = _W["tok"], _W["eos"], _W["max_len"], _W["tmp_dir"]
    st: Dict[str, Any] = {key: 0 for key in STAT_KEYS}
    st["domain"], st["source"] = Counter(), Counter()
    lens: List[int] = []
    plens: List[int] = []
    samples: List[Dict[str, Any]] = []
    tmp_bin = os.path.join(tmp_dir, f"shard_{k:03d}.bin.tmp")
    with open(tmp_bin, "wb") as fout:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                st["n_in"] += 1
                conv = row["conversations"] or []
                msgs = [{"role": ROLE.get(m["from"], "?"), "content": m["value"]} for m in conv]
                if not msgs or any(m["role"] == "?" for m in msgs) or msgs[-1]["role"] != "assistant":
                    st["n_bad_roles"] += 1
                    continue
                if len(msgs) > 2:
                    st["n_multi_turn"] += 1
                comp = msgs[-1]["content"]
                if chat.THINK_OPEN in comp and chat.THINK_CLOSE not in comp:
                    st["n_missing_think_close"] += 1
                prompt_ids = tok(tok.apply_chat_template(msgs[:-1], tokenize=False,
                                                         add_generation_prompt=True,
                                                         enable_thinking=True),
                                 add_special_tokens=False)["input_ids"]
                budget = max_len - len(prompt_ids)
                if budget <= 0:
                    st["n_dropped_prompt_too_long"] += 1
                    continue
                comp_ids = tok(comp, add_special_tokens=False)["input_ids"]
                if len(comp_ids) + 1 > budget:
                    comp_ids = comp_ids[:budget]
                    st["n_truncated_completion"] += 1
                else:
                    comp_ids = comp_ids + [eos]
                ids = np.asarray(prompt_ids + comp_ids, dtype=np.int32)
                fout.write(ids.tobytes())
                lens.append(len(ids))
                plens.append(len(prompt_ids))
                st["n_kept"] += 1
                st["domain"][row["domain"]] += 1
                st["source"][row["source"]] += 1
                if len(samples) < 2:
                    samples.append({"difficulty": row["difficulty"], "source": row["source"],
                                    "domain": row["domain"], "messages": msgs})
    os.replace(tmp_bin, os.path.join(tmp_dir, f"shard_{k:03d}.bin"))
    np.save(os.path.join(tmp_dir, f"shard_{k:03d}.lens.npy"), np.asarray(lens, dtype=np.int64))
    np.save(os.path.join(tmp_dir, f"shard_{k:03d}.plens.npy"), np.asarray(plens, dtype=np.int32))
    st["domain"], st["source"] = dict(st["domain"]), dict(st["source"])
    return {"k": k, "stats": st, "samples": samples}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="OpenThoughts3-1.2M snapshot dir (contains data/*.parquet)")
    p.add_argument("--out", required=True, help="cache dir to create, e.g. data/train/ot3_cache_len16384")
    p.add_argument("--model", required=True, help="tokenizer source, e.g. models/Qwen3-4B-Base")
    p.add_argument("--check-tokenizer", default=None, help="second model path; assert identical tokenizer")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--procs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-shards", type=int, default=0, help="debug: only first N shards")
    args = p.parse_args(argv)

    if args.check_tokenizer:
        from transformers import AutoTokenizer
        a = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        b = AutoTokenizer.from_pretrained(args.check_tokenizer, trust_remote_code=True)
        assert a.get_vocab() == b.get_vocab() and a.chat_template == b.chat_template, \
            "tokenizers differ; one cache cannot serve both models"
        print("[prep] tokenizer identity check OK", flush=True)

    shards = sorted(glob.glob(os.path.join(args.src, "data", "*.parquet")))
    assert shards, f"no parquet shards under {args.src}/data"
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    os.makedirs(args.out, exist_ok=True)
    tmp_dir = os.path.join(args.out, "tmp_shards")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[prep] {len(shards)} shards, procs={args.procs}, max_len={args.max_len}", flush=True)

    t0 = time.time()
    agg: Dict[str, Any] = {key: 0 for key in STAT_KEYS}
    agg["domain"], agg["source"] = Counter(), Counter()
    all_samples: List[Dict[str, Any]] = []
    done = 0
    todo = []
    for k in range(len(shards)):
        if os.path.exists(os.path.join(tmp_dir, f"shard_{k:03d}.bin")) and \
                os.path.exists(os.path.join(tmp_dir, f"shard_{k:03d}.stats.json")):
            done += 1
        else:
            todo.append((k, shards[k]))
    if done:
        print(f"[prep] resuming: {done} shards already tokenised", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=args.procs, initializer=_init_worker,
                                 initargs=(args.model, args.max_len, tmp_dir)) as ex:
            for r in ex.map(_do_shard, todo):
                with open(os.path.join(tmp_dir, f"shard_{r['k']:03d}.stats.json"), "w") as f:
                    json.dump({"stats": r["stats"], "samples": r["samples"]}, f, ensure_ascii=False)
                done += 1
                if done % 10 == 0:
                    print(f"[prep] {done}/{len(shards)} shards ({time.time() - t0:.0f}s)", flush=True)
    for k in range(len(shards)):
        with open(os.path.join(tmp_dir, f"shard_{k:03d}.stats.json")) as f:
            r = json.load(f)
        for key in STAT_KEYS:
            agg[key] += r["stats"][key]
        agg["domain"].update(r["stats"]["domain"])
        agg["source"].update(r["stats"]["source"])
        all_samples.extend(r["samples"])
    agg["domain"], agg["source"] = dict(agg["domain"]), dict(agg["source"])
    print(f"[prep] tokenised {agg['n_kept']} examples ({time.time() - t0:.0f}s); packing...", flush=True)

    shard_lens = [np.load(os.path.join(tmp_dir, f"shard_{k:03d}.lens.npy")) for k in range(len(shards))]
    shard_plens = [np.load(os.path.join(tmp_dir, f"shard_{k:03d}.plens.npy")) for k in range(len(shards))]
    shard_off = [np.concatenate(([0], np.cumsum(l))) for l in shard_lens]
    counts = [len(l) for l in shard_lens]
    cum = np.concatenate(([0], np.cumsum(counts)))
    flat_lens = np.concatenate(shard_lens) if shard_lens else np.zeros(0, np.int64)
    assert int(flat_lens.max(initial=0)) <= args.max_len
    bins = pack_bfd([int(x) for x in flat_lens], args.max_len)
    random.Random(args.seed).shuffle(bins)
    mms = [np.memmap(os.path.join(tmp_dir, f"shard_{k:03d}.bin"), dtype=np.int32, mode="r")
           for k in range(len(shards))]
    for k, mm in enumerate(mms):
        assert len(mm) == int(shard_off[k][-1]), f"shard {k} bin size mismatch"

    n_ex = int(cum[-1])
    total = int(flat_lens.sum())
    bin_offsets = np.zeros(len(bins) + 1, dtype=np.int64)
    ex_starts = np.zeros(n_ex, dtype=np.int64)
    ex_plens_out = np.zeros(n_ex, dtype=np.int32)
    tmp_tokens = os.path.join(args.out, TOKENS_BIN + ".tmp")
    pos = 0
    j = 0
    with open(tmp_tokens, "wb") as f:
        for bi, ex_idx in enumerate(bins):
            bin_offsets[bi] = pos
            for i in ex_idx:
                sh = int(np.searchsorted(cum, i, side="right")) - 1
                lo = i - int(cum[sh])
                a, b = int(shard_off[sh][lo]), int(shard_off[sh][lo + 1])
                f.write(np.ascontiguousarray(mms[sh][a:b]).tobytes())
                ex_starts[j] = pos
                ex_plens_out[j] = int(shard_plens[sh][lo])
                pos += b - a
                j += 1
        bin_offsets[len(bins)] = pos
    assert pos == total and j == n_ex, (pos, total, j, n_ex)
    os.replace(tmp_tokens, os.path.join(args.out, TOKENS_BIN))
    np.save(os.path.join(args.out, BIN_OFFSETS), bin_offsets)
    np.save(os.path.join(args.out, EX_STARTS), ex_starts)
    np.save(os.path.join(args.out, EX_PROMPT_LENS), ex_plens_out)
    agg.update(src=os.path.abspath(args.src), model=os.path.abspath(args.model), thinking=True,
               max_len=args.max_len, tokens=total, bins=len(bins),
               fill=total / max(1, len(bins) * args.max_len), n_examples=n_ex, pack_seed=args.seed,
               seconds=round(time.time() - t0))
    with open(os.path.join(args.out, STATS_JSON), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump(all_samples[:40], f, ensure_ascii=False, indent=1)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[prep] DONE " + json.dumps({k: v for k, v in agg.items() if k not in ("domain", "source")}),
          flush=True)


if __name__ == "__main__":
    main()
