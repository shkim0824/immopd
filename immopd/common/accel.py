"""Write the accelerate / DeepSpeed config of one launch.

  python -m immopd.common.accel --config <train config> --gpus 0-7 --out <accelerate.yaml>

Gradient accumulation is ``train.batch_size / (train.per_device_bs * number of GPUs)``.
"""
from __future__ import annotations

import argparse
import os

import yaml

from immopd.common.config import load_config
from immopd.common.io import repo_root


def write_accelerate_config(template: str, out_path: str, num_processes: int, grad_accum: int) -> str:
    cfg = yaml.safe_load(open(template))
    cfg["num_processes"] = int(num_processes)
    cfg.setdefault("deepspeed_config", {})["gradient_accumulation_steps"] = int(grad_accum)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpus", type=int, required=True, help="number of GPUs")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=os.path.join(repo_root(), "configs", "accelerate", "zero2.yaml"))
    a = p.parse_args(argv)
    t = load_config(a.config).train
    bs, pd = int(t.get("batch_size", 128)), int(t.get("per_device_bs", 1))
    assert bs % (pd * a.gpus) == 0, f"batch_size {bs} is not divisible by per_device_bs x GPUs = {pd * a.gpus}"
    write_accelerate_config(a.template, a.out, a.gpus, bs // (pd * a.gpus))


if __name__ == "__main__":
    main()
