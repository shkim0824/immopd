"""Task-vector increments: DELTA = sum_i c_i (phi_i - theta_ref), stored in fp32.

  python -m immopd.merging.task_vector build --base <ref> --teachers tau=<dir>,med=<dir>,... \
      --adds tau=0.3,med=0.3 --out <delta dir>
  python -m immopd.merging.task_vector apply --model <checkpoint> --delta <delta dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from immopd.merging.merge_teachers import ShardWriter, copy_side_files, weight_map

SHARD = 4_000_000_000


def parse_adds(s: str) -> dict:
    return {kv.split("=")[0]: float(kv.split("=")[1]) for kv in s.split(",") if kv}


def build(base: str, teachers: dict, adds: dict, out: str, limit: int = 0) -> str:
    if os.path.exists(os.path.join(out, "DELTA_OK")):
        have = json.load(open(os.path.join(out, "delta.index.json")))["adds"]
        assert have == adds, ("an increment for other coefficients already exists here", out, have, adds)
        print("[delta] exists:", out)
        return out
    assert adds and all(n in teachers for n in adds), (adds, teachers)
    tmp = out + ".building"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    bmap = weight_map(base); tmaps = {n: weight_map(teachers[n]) for n in adds}
    keys = [k for k in bmap if k != "lm_head.weight"]
    for n in adds:
        miss = [k for k in keys if k not in tmaps[n]]
        assert not miss, ("teacher lacks base tensors", n, miss[:3])
    if limit:
        keys = keys[:limit]
    H = {}

    def get(path, key):
        if path not in H:
            H[path] = safe_open(path, framework="pt", device="cpu")
        return H[path].get_tensor(key)

    shard, nb, sid, index, n2 = {}, 0, 1, {}, 0.0

    def flush():
        nonlocal shard, nb, sid
        if shard:
            fn = "delta-%05d.safetensors" % sid; save_file(shard, os.path.join(tmp, fn), metadata={"format": "pt"})
            for k in shard:
                index[k] = fn
            shard, nb, sid = {}, 0, sid + 1

    for i, k in enumerate(keys):
        b = get(bmap[k], k).to(torch.float32); acc = torch.zeros_like(b)
        for n, lam in adds.items():
            acc += lam * (get(tmaps[n][k], k).to(torch.float32) - b)
        n2 += float((acc.double() ** 2).sum()); shard[k] = acc.contiguous(); nb += acc.numel() * 4
        if nb >= SHARD:
            flush()
        if i % 100 == 0:
            print("[delta] %d/%d %s" % (i, len(keys), k), flush=True)
    flush()
    json.dump({"adds": adds, "base": base, "teachers": {n: teachers[n] for n in adds}, "dtype": "float32", "norm": n2 ** 0.5,
               "weight_map": index}, open(os.path.join(tmp, "delta.index.json"), "w"), indent=1)
    open(os.path.join(tmp, "DELTA_OK"), "w").write("ok")
    shutil.rmtree(out, ignore_errors=True); os.replace(tmp, out)
    print("[delta] done: %d tensors, |delta| = %.4f -> %s" % (len(index), n2 ** 0.5, out), flush=True)
    return out


def apply(model: str, delta: str, out: str, save_dtype: str = "bfloat16") -> str:
    assert os.path.exists(os.path.join(delta, "DELTA_OK")), delta
    meta = json.load(open(os.path.join(delta, "delta.index.json")))
    dmap = {k: os.path.join(delta, v) for k, v in meta["weight_map"].items()}
    mmap = weight_map(model)
    H = {}

    def get(path, key):
        if path not in H:
            H[path] = safe_open(path, framework="pt", device="cpu")
        return H[path].get_tensor(key)

    tmp = out + ".building"; shutil.rmtree(tmp, ignore_errors=True)
    w = ShardWriter(tmp)
    moved = 0.0
    for i, k in enumerate(mmap):
        t = get(mmap[k], k).to(torch.float32)
        if k in dmap:
            d = get(dmap[k], k).to(torch.float32)
            assert d.shape == t.shape, (k, d.shape, t.shape)
            t = t + d
            moved += float((d.double() ** 2).sum())
        w.add(k, t.to(getattr(torch, save_dtype)))
        if i % 100 == 0:
            print("[apply] %d/%d %s" % (i, len(mmap), k), flush=True)
    w.close()
    copy_side_files(model, tmp)
    cfg = json.load(open(os.path.join(tmp, "config.json"))); cfg["torch_dtype"] = save_dtype; cfg.update({"dtype": save_dtype} if "dtype" in cfg else {})
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"), indent=2)
    json.dump({"model": model, "delta": delta, "adds": meta["adds"], "dtype": save_dtype},
              open(os.path.join(tmp, "merge_info.json"), "w"), indent=2)
    shutil.rmtree(out, ignore_errors=True); os.replace(tmp, out)
    print("[apply] done -> %s (|delta| %.4f)" % (out, moved ** 0.5), flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", required=True); b.add_argument("--teachers", required=True, help="domain=path,...")
    b.add_argument("--adds", required=True, help="domain=lam,..."); b.add_argument("--out", required=True)
    q = sub.add_parser("apply")
    q.add_argument("--model", required=True); q.add_argument("--delta", required=True); q.add_argument("--out", required=True)
    q.add_argument("--save-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = p.parse_args(argv)
    if a.cmd == "build":
        build(a.base, dict(kv.split("=", 1) for kv in a.teachers.split(",")), parse_adds(a.adds), a.out)
    else:
        apply(a.model, a.delta, a.out, a.save_dtype)


if __name__ == "__main__":
    main()
