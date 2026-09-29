"""Merge initialization by task arithmetic: theta_0 = theta_ref + sum_i c_i (phi_i - theta_ref).

  python -m immopd.merging.merge_teachers --base ${MODEL_DIR}/Qwen3-4B-OT3 \
      --teachers tau=<dir>,med=<dir>,law=<dir>,fin=<dir>,if=<dir> \
      --weights tau=0.2,med=0.2,law=0.2,fin=0.2,if=0.2 --out ${MODEL_DIR}/merge-uniform
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_LIMIT = 5_000_000_000


def weight_map(d: str):
    """{tensor name: file} of a safetensors checkpoint (sharded or single-file)."""
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(d, v) for k, v in json.load(open(idx))["weight_map"].items()}
    single = os.path.join(d, "model.safetensors")
    assert os.path.exists(single), f"no model.safetensors(.index.json) in {d}"
    with safe_open(single, framework="pt", device="cpu") as f:
        return {k: single for k in f.keys()}


def copy_side_files(src: str, dst: str) -> None:
    for f in os.listdir(src):
        if f.endswith((".json", ".jinja", ".txt")) and f != "model.safetensors.index.json":
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))


class ShardWriter:

    def __init__(self, out: str, limit: int = SHARD_LIMIT):
        self.out, self.limit = out, limit
        self.shard, self.bytes, self.sid, self.weight_map, self.total = {}, 0, 1, {}, 0
        os.makedirs(out, exist_ok=True)

    def add(self, key: str, t: torch.Tensor) -> None:
        t = t.contiguous()
        self.shard[key] = t
        self.bytes += t.numel() * t.element_size()
        self.total += t.numel() * t.element_size()
        if self.bytes >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.shard:
            return
        fn = f"model-{self.sid:05d}.safetensors"
        save_file(self.shard, os.path.join(self.out, fn), metadata={"format": "pt"})
        for k in self.shard:
            self.weight_map[k] = fn
        self.shard, self.bytes, self.sid = {}, 0, self.sid + 1

    def close(self) -> None:
        self.flush()
        n_sh = self.sid - 1
        renamed = {}
        for j in range(1, n_sh + 1):
            old = f"model-{j:05d}.safetensors"; new = f"model-{j:05d}-of-{n_sh:05d}.safetensors"
            os.replace(os.path.join(self.out, old), os.path.join(self.out, new)); renamed[old] = new
        self.weight_map = {k: renamed[v] for k, v in self.weight_map.items()}
        json.dump({"metadata": {"total_size": self.total}, "weight_map": self.weight_map},
                  open(os.path.join(self.out, "model.safetensors.index.json"), "w"), indent=2)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", required=True, help="name=path,name=path,...")
    p.add_argument("--weights", default="", help="name=w,... (default: uniform)")
    p.add_argument("--base", required=True, help="reference model")
    p.add_argument("--out", required=True)
    p.add_argument("--save-dtype", default="float32", choices=["bfloat16", "float32"])
    a = p.parse_args(argv)
    teachers = dict(kv.split("=", 1) for kv in a.teachers.split(","))
    weights = {n: 1.0 / len(teachers) for n in teachers}
    if a.weights:
        weights = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.weights.split(","))}
    assert set(weights) == set(teachers), (weights, teachers)
    maps = {n: weight_map(d) for n, d in teachers.items()}
    base_map = weight_map(a.base)
    keys = [k for k in base_map if k != "lm_head.weight"]
    for n, m in maps.items():
        missing = [k for k in keys if k not in m]
        assert not missing, (n, missing[:5])
    handles = {}

    def get(path, key):
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    w = ShardWriter(a.out)
    for i, k in enumerate(keys):
        base = get(base_map[k], k).to(torch.float32)
        acc = base.clone()
        for n in teachers:
            acc += weights[n] * (get(maps[n][k], k).to(torch.float32) - base)
        w.add(k, acc.to(getattr(torch, a.save_dtype)))
        if i % 50 == 0:
            print(f"{i}/{len(keys)} {k}", flush=True)
    w.close()
    copy_side_files(a.base, a.out)
    cfg = json.load(open(os.path.join(a.out, "config.json"))); cfg["torch_dtype"] = a.save_dtype; cfg.update({"dtype": a.save_dtype} if "dtype" in cfg else {})
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump({"base": a.base, "teachers": teachers, "weights": weights, "dtype": a.save_dtype},
              open(os.path.join(a.out, "merge_info.json"), "w"), indent=2)
    print("merged", len(w.weight_map), "tensors ->", a.out)


if __name__ == "__main__":
    main()
