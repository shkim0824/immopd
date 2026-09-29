"""tau2-bench evaluation with the official harness (v1.0.1).

  python -m immopd.eval.tau2_run --model <checkpoint> --out <dir> --gpus 0-7 [--tau2-python <python>]

Telecom, 114 base-split tasks, 32,768-token context, 200-step limit, agent temperature 1 / top-p 1, user
simulator gpt-4.1-2025-04-14 at temperature 0 (``OPENAI_API_KEY``). ``--task-ids-file`` restricts the run to a
task list (validation). The score is pass@1 over all tasks; errors count as failures.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def parse_gpus(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def run(cmd, log=None, dry=False):
    print("[tau2] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    e = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if not log:
        return subprocess.call(cmd, env=e)
    with open(log, "a") as f:
        return subprocess.call(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)


def data_dir(tau2_python):
    return subprocess.run([tau2_python, "-c", "from tau2.utils.utils import DATA_DIR; print(DATA_DIR)"],
                          capture_output=True, text=True).stdout.strip()


def find_results(dd, name):
    for c in (os.path.join(dd, "simulations", name, "results.json"), os.path.join(dd, "simulations", name + ".json"),
              os.path.join(dd, "tau2", "simulations", name, "results.json")):
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(dd, "**", "simulations", "*" + name + "*", "results.json"), recursive=True)
    return hits[0] if hits else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", default="0-7")
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--domain", default="telecom")
    p.add_argument("--split", default="base")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--conc", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    p.add_argument("--user-args", default='{"temperature":0.0}')
    p.add_argument("--agent", default="llm_agent", help="llm_agent | llm_agent_prefix (immopd.eval.prefix)")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--task-ids-file", default="", help='JSON with "task_ids"')
    p.add_argument("--tau2-python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    results = os.path.join(a.out, "tau2_%s_%s.json" % (a.domain, a.split))
    agent_args = {"api_base": "http://127.0.0.1:%d/v1" % a.port, "api_key": "EMPTY", "temperature": 1.0,
                  "top_p": 1.0, "extra_body": {"top_k": -1}}
    if a.dry_run or not os.path.exists(results):
        if not a.dry_run and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("[tau2] OPENAI_API_KEY is not set (user simulator)")
        gpus = parse_gpus(a.gpus)
        cache = os.path.join(a.out, "_cache")
        server = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", os.path.abspath(a.model),
                  "--served-model-name", "student", "--host", "127.0.0.1", "--port", str(a.port),
                  "--tensor-parallel-size", "1", "--data-parallel-size", str(len(gpus)),
                  "--max-model-len", str(a.max_model_len), "--gpu-memory-utilization", str(a.gpu_mem_util),
                  "--dtype", "bfloat16", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                  "--reasoning-parser", "qwen3", "--generation-config", "vllm", "--trust-remote-code",
                  "--disable-log-requests"]
        print("[tau2] server: " + " ".join(shlex.quote(c) for c in server), flush=True)
        sp = None
        if not a.dry_run:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), VLLM_CACHE_ROOT=cache + "/vllm",
                       TORCHINDUCTOR_CACHE_DIR=cache + "/inductor", TRITON_CACHE_DIR=cache + "/triton")
            sp = subprocess.Popen(server, env=env, stdout=open(os.path.join(a.out, "vllm.log"), "a"),
                                  stderr=subprocess.STDOUT)
            for _ in range(120):
                if http_ok("http://127.0.0.1:%d/v1/models" % a.port):
                    break
                if sp.poll() is not None:
                    raise SystemExit("[tau2] vLLM server exited; see %s/vllm.log" % a.out)
                time.sleep(10)
            else:
                sp.kill()
                raise SystemExit("[tau2] vLLM server did not come up")
        name = "%s_%s_%s" % (os.path.basename(a.out.rstrip("/")), a.domain, a.split)
        cmd = [a.tau2_python, "-m", "immopd.eval.tau2_cli", "run", "--domain", a.domain, "--task-split-name", a.split,
               "--agent", a.agent, "--user", "user_simulator", "--agent-llm", "openai/student",
               "--agent-llm-args", json.dumps(agent_args), "--user-llm", a.user_llm, "--user-llm-args", a.user_args,
               "--num-trials", str(a.trials), "--max-concurrency", str(a.conc), "--max-steps", str(a.max_steps),
               "--seed", str(a.seed), "--save-to", name, "--auto-resume", "--max-retries", str(a.max_retries),
               "--log-level", "WARNING"]
        if a.task_ids_file:
            cmd += ["--task-ids"] + json.load(open(a.task_ids_file))["task_ids"]
        rc = run(cmd, log=os.path.join(a.out, "tau2_run.log"), dry=a.dry_run)
        if sp is not None:
            sp.terminate()
            try:
                sp.wait(timeout=60)
            except subprocess.TimeoutExpired:
                sp.kill()
        if not a.dry_run:
            res = find_results(data_dir(a.tau2_python), name)
            if res is None:
                raise SystemExit("[tau2] harness exit %d and no results file for %s" % (rc, name))
            shutil.copy(res, results)
    rc = run([a.tau2_python, "-m", "immopd.eval.tau2_metrics", "--results", results, "--out", a.out,
              "--model", a.model, "--domain", a.domain, "--split", a.split], dry=a.dry_run)
    assert rc == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
