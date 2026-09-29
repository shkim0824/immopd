"""IM-MOPD driver (Algorithm 1) on one host.

  python -m immopd.distill.driver --config configs/immopd/qwen3-4b.yaml --gpus 0-7 \
      --teacher-endpoints ${RUN_DIR}/teachers/teacher_endpoints.json [--dry-run]

For every merge step s: train to s, evaluate checkpoint-s on the validation set, select the domains with
recovery <= gamma, build the increment delta * sum of their task vectors and write DECISION_s.json; training
resumes from checkpoint-s and applies it. ``--mode inline`` serves a trainer that was started separately and
waits at each merge step; the evaluation then runs on ``--eval-gpus``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from immopd.common.accel import write_accelerate_config
from immopd.common.config import add_config_args, dump, load_config
from immopd.distill.serve_teachers import parse_gpus
from immopd.distill.train_immopd import DEFAULTS, decision_dir, decision_steps
from immopd.merging.task_vector import build as build_delta, parse_adds

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S") + " [driver] " + " ".join(str(x) for x in a), flush=True)


def checkpoint_complete(ck: str) -> bool:
    return os.path.isfile(os.path.join(ck, "trainer_state.json")) and (
        os.path.isfile(os.path.join(ck, "model.safetensors"))
        or os.path.isfile(os.path.join(ck, "model.safetensors.index.json")))


class Driver:
    def __init__(self, a):
        self.a = a
        self.cfg = load_config(a.config, a.override, base=DEFAULTS)
        self.out = self.cfg.train.output
        self.dir = decision_dir(self.cfg)
        self.steps = decision_steps(self.cfg)
        self.gpus = parse_gpus(a.gpus)
        self.eval_gpus = parse_gpus(a.eval_gpus) if a.eval_gpus else self.gpus
        self.py = sys.executable
        os.makedirs(self.dir, exist_ok=True)

    def run(self, cmd, env=None, log_path=None) -> int:
        log("run:", " ".join(shlex.quote(c) for c in cmd))
        if self.a.dry_run:
            return 0
        e = dict(os.environ)
        e.update(env or {})
        e["PYTHONPATH"] = REPO + os.pathsep + e.get("PYTHONPATH", "")
        if not log_path:
            return subprocess.call(cmd, env=e, cwd=REPO)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            return subprocess.call(cmd, env=e, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)

    def train_segment(self, stop_at_decision: bool) -> int:
        cfg, world = self.cfg, len(self.gpus)
        bs, pd = int(cfg.train.batch_size), int(cfg.train.per_device_bs)
        assert bs % (pd * world) == 0, f"batch_size {bs} is not divisible by {pd * world}"
        accel = os.path.join(self.out, "accelerate.yaml")
        if not self.a.dry_run:
            write_accelerate_config(self.a.accelerate_template, accel, world, bs // (pd * world))
        ov = ["teachers.endpoints_file=%s" % self.a.teacher_endpoints,
              "immopd.stop_at_decision=%s" % ("true" if stop_at_decision else "false")] + list(self.a.override)
        cmd = ["accelerate", "launch", "--config_file", accel, "--num_processes", str(world),
               "--main_process_port", str(self.a.accel_port), "-m", "immopd.distill.train_immopd",
               "--config", self.a.config]
        for o in ov:
            cmd += ["--override", o]
        return self.run(cmd, env={"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus))},
                        log_path=os.path.join(self.out, "train.log"))

    def evaluate(self, s: int, ck: str, gpus, mem_util: float) -> dict:
        v = self.cfg.immopd.validation
        gs = ",".join(map(str, gpus))
        acc_dir = os.path.join(self.dir, "val", "ck%d" % s)
        if not os.path.exists(os.path.join(acc_dir, "acc.json")):
            self.run([self.py, "-m", "immopd.eval.validation", "run", "--model", ck, "--out", acc_dir, "--gpus", gs,
                      "--pool", v.pool, "--doms", ",".join(v.domains), "--n", str(v.n_samples),
                      "--gpu-mem-util", str(mem_util)], log_path=os.path.join(acc_dir, "run.log"))
        paths = {"acc": os.path.join(acc_dir, "acc.json")}
        if v.get("tau2_tasks"):
            tau_dir = os.path.join(self.dir, "tau2", "ck%d" % s)
            if not os.path.exists(os.path.join(tau_dir, "metrics.json")):
                self.run([self.py, "-m", "immopd.eval.tau2_run", "--model", ck, "--out", tau_dir, "--gpus", gs,
                          "--split", "full", "--task-ids-file", v.tau2_tasks, "--max-retries", "0",
                          "--gpu-mem-util", str(mem_util),
                          "--tau2-python", self.a.tau2_python or self.py],
                         log_path=os.path.join(tau_dir, "run.log"))
            paths["tau2"] = os.path.join(tau_dir, "metrics.json")
        return paths

    def decide_step(self, s: int, gpus, mem_util: float) -> None:
        v = self.cfg.immopd
        dec_path = os.path.join(self.dir, "DECISION_%d.json" % s)
        if os.path.exists(dec_path):
            return
        ck = os.path.join(self.out, "checkpoint-%d" % s)
        assert self.a.dry_run or checkpoint_complete(ck), "checkpoint-%d is incomplete" % s
        paths = self.evaluate(s, ck, gpus, mem_util)
        plan_path = os.path.join(self.dir, "PLAN_%d.json" % s)
        cmd = [self.py, "-m", "immopd.merging.recovery", "plan", "--refs", v.refs, "--acc", paths["acc"],
               "--step", str(s), "--gamma", str(v.gamma), "--delta", str(v.delta), "--out", plan_path]
        if "tau2" in paths:
            cmd += ["--tau2", paths["tau2"]]
        rc = self.run(cmd)
        if self.a.dry_run:
            return
        assert rc == 0, "decision failed for step %d" % s
        plan = json.load(open(plan_path))
        plan["delta_dir"] = None
        if plan["adds"]:
            plan["delta_dir"] = build_delta(v.base, dict(v.teachers), parse_adds(plan["adds"]),
                                            os.path.join(self.out, "merge_delta_%d" % s))
        tmp = dec_path + ".tmp"
        json.dump(plan, open(tmp, "w"), indent=1)
        os.replace(tmp, dec_path)
        log("step %d: merge %s" % (s, plan["picked"] or "none"))

    def restart(self) -> None:
        cfg = self.cfg
        log("config:\n" + dump(cfg))
        if dict(cfg.immopd.plan):
            assert self.train_segment(stop_at_decision=False) == 0, "training failed"
            return
        mx = int(cfg.train.max_steps)
        for target in self.steps + [mx]:
            ck = os.path.join(self.out, "checkpoint-%d" % target)
            done = os.path.exists(os.path.join(self.out, "immopd_done.json"))
            if not (checkpoint_complete(ck) or done):
                log("training to step %d" % target)
                assert self.train_segment(stop_at_decision=(target != mx)) == 0, "training failed"
                assert self.a.dry_run or checkpoint_complete(ck), "checkpoint-%d was not written" % target
            if target != mx:
                self.decide_step(target, self.gpus, self.a.gpu_mem_util)
        log("done:", os.path.join(self.out, "final"))

    def inline(self) -> None:
        pending = list(self.steps)
        while pending:
            for s in list(pending):
                if os.path.exists(os.path.join(self.dir, "DECISION_%d.json" % s)):
                    pending.remove(s)
                elif os.path.exists(os.path.join(self.dir, "REQUEST_%d.json" % s)):
                    self.decide_step(s, self.eval_gpus, self.a.eval_gpu_mem_util)
                    pending.remove(s)
            if self.a.dry_run:
                break
            time.sleep(20)
        log("all decisions written")


def main(argv=None):
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--mode", default="restart", choices=["restart", "inline"])
    p.add_argument("--gpus", default="0-7", help="training GPUs")
    p.add_argument("--teacher-endpoints", required=True)
    p.add_argument("--accelerate-template", default=os.path.join(REPO, "configs", "accelerate", "zero2.yaml"))
    p.add_argument("--accel-port", type=int, default=29500)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--eval-gpus", default="", help="inline mode: evaluation GPUs")
    p.add_argument("--eval-gpu-mem-util", type=float, default=0.30)
    p.add_argument("--tau2-python", default="", help="python of the tau2-bench environment")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    d = Driver(a)
    d.restart() if a.mode == "restart" else d.inline()


if __name__ == "__main__":
    main()
