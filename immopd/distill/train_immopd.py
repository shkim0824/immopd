"""IM-MOPD training process: MOPD with the iterative-merge callback. Launched by ``immopd.distill.driver``.

  immopd:
    steps: [25, 50, 75]        merge points (multiples of train.save_steps, below train.max_steps)
    gamma: 0.6                 recovery threshold
    delta: 0.3                 correction scale
    base: <reference model>
    teachers: {domain: path}
    plan: {50: "tau=0.3"}      optional fixed schedule; replaces the recovery rule
"""
from __future__ import annotations

import argparse
import copy
import os
import time

from immopd.common.config import add_config_args, dump, load_config
from immopd.common.resume import find_resume_checkpoint
from immopd.distill import train_mopd as T
from immopd.distill.merge_callback import IterativeMergeCallback
from immopd.distill.runtime import DEFAULTS as MOPD_DEFAULTS, is_main, patch_runtime
from immopd.merging.task_vector import build, parse_adds

DEFAULTS = copy.deepcopy(MOPD_DEFAULTS)
DEFAULTS["immopd"] = {
    "steps": [25, 50, 75], "gamma": 0.6, "delta": 0.3, "base": "", "teachers": {}, "plan": {},
    "refs": "", "decision_dir": "", "stop_at_decision": True, "wait_hours": 4.0,
    "validation": {"pool": "data/validation/pool.jsonl", "tau2_tasks": "data/validation/tau2_tasks.json",
                   "domains": ["med", "law", "fin", "if"], "n_samples": 4},
}


def decision_steps(cfg) -> list:
    steps = sorted(int(s) for s in cfg.immopd.steps if 0 < int(s) < int(cfg.train.max_steps))
    for s in steps:
        assert s % int(cfg.train.save_steps) == 0, f"merge step {s} must be a multiple of train.save_steps"
    return steps


def decision_dir(cfg) -> str:
    return cfg.immopd.decision_dir or os.path.join(cfg.train.output, "decisions")


def fixed_plan(cfg, steps) -> dict:
    """{step: increment dir} of a fixed schedule; rank 0 builds the increments."""
    plan = {int(k): v for k, v in dict(cfg.immopd.plan).items()}
    assert set(plan) == set(steps), (sorted(plan), steps)
    out = {}
    for s in steps:
        d = os.path.join(cfg.train.output, f"merge_delta_{s}")
        if is_main():
            build(cfg.immopd.base, dict(cfg.immopd.teachers), parse_adds(plan[s]), d)
        else:
            t0 = time.time()
            while not os.path.exists(os.path.join(d, "DELTA_OK")):
                if time.time() - t0 > 3 * 3600:
                    raise RuntimeError(f"increment {d} was not built")
                time.sleep(15)
        out[s] = d
    return out


def main(argv=None):
    patch_runtime()
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[immopd] config:\n" + dump(cfg), flush=True)
    steps = decision_steps(cfg)
    plan = fixed_plan(cfg, steps) if dict(cfg.immopd.plan) else None
    cb = IterativeMergeCallback(cfg.train.output, steps, decision_dir(cfg), plan=plan,
                                wait_s=float(cfg.immopd.wait_hours) * 3600,
                                stop_at_decision=bool(cfg.immopd.stop_at_decision))
    trainer, info = T.build_trainer(cfg, extra_callbacks=[cb])
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[immopd] merge steps={steps} prompts={info['counts']} resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    if trainer.state.global_step >= int(cfg.train.max_steps):
        T.finish(cfg, trainer, info, t0, done_name="immopd_done.json")
    elif is_main():
        print(f"[immopd] paused after step {trainer.state.global_step}", flush=True)


if __name__ == "__main__":
    main()
