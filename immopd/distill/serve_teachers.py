"""Serve the frozen domain teachers with vLLM and write ``teacher_endpoints.json``.

  python -m immopd.distill.serve_teachers --gpus 0-5 --out ${RUN_DIR}/teachers \
      --teachers tau=<dir>,med=<dir>,law=<dir>,fin=<dir>,if=<dir>

Every teacher gets one GPU; remaining GPUs become extra replicas. The process serves until it is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request


def parse_gpus(s: str) -> list:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def place(gpus: list, teachers: dict, base_port: int = 8100, host: str = "127.0.0.1") -> dict:
    """One server per GPU, teachers assigned round-robin."""
    names = list(teachers)
    assert len(gpus) >= len(names), f"{len(names)} teachers need at least {len(names)} GPUs, got {len(gpus)}"
    servers = [{"idx": i, "domain": names[i % len(names)], "model": teachers[names[i % len(names)]],
                "served_name": names[i % len(names)], "gpu": g, "port": base_port + i,
                "url": f"http://{host}:{base_port + i}"} for i, g in enumerate(gpus)]
    endpoints = {}
    for sv in servers:
        endpoints.setdefault(sv["domain"], []).append(sv["url"])
    return {"servers": servers, "endpoints": endpoints}


def wait_http(urls, timeout=1800, interval=10) -> bool:
    t0 = time.time()
    left = set(urls)
    while left and time.time() - t0 < timeout:
        for u in list(left):
            try:
                with urllib.request.urlopen(u + "/v1/models", timeout=5) as r:
                    if r.status == 200:
                        left.discard(u)
            except Exception:
                pass
        if left:
            time.sleep(interval)
    return not left


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", required=True, help="domain=path,...")
    p.add_argument("--gpus", default="0-5")
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--base-port", type=int, default=8100)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.55)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    teachers = dict(kv.split("=", 1) for kv in a.teachers.split(",") if kv)
    plan = place(parse_gpus(a.gpus), teachers, a.base_port, a.host)
    os.makedirs(a.out, exist_ok=True)
    procs = []
    for sv in plan["servers"]:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", os.path.abspath(sv["model"]),
               "--served-model-name", sv["served_name"], "--host", a.host, "--port", str(sv["port"]),
               "--max-model-len", str(a.max_model_len), "--gpu-memory-utilization", str(a.gpu_mem_util),
               "--dtype", "bfloat16", "--logprobs-mode", "raw_logprobs", "--enable-prefix-caching",
               "--max-num-batched-tokens", "4096", "--trust-remote-code"]
        print("[serve] %-4s gpu=%d port=%d: %s" % (sv["domain"], sv["gpu"], sv["port"],
                                                    " ".join(shlex.quote(c) for c in cmd)), flush=True)
        if a.dry_run:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sv["gpu"]))
        logf = open(os.path.join(a.out, "server_%d.log" % sv["idx"]), "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT))
    path = os.path.join(a.out, "teacher_endpoints.json")
    json.dump(plan, open(path, "w"), indent=2)
    print("[serve] endpoints -> %s" % path, flush=True)
    if a.dry_run:
        return 0

    def stop(*_):
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pr.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not wait_http([sv["url"] for sv in plan["servers"]]):
        print("[serve] servers did not come up; see %s/server_*.log" % a.out, flush=True)
        stop()
    print("[serve] all teachers ready", flush=True)
    while True:
        for pr in procs:
            if pr.poll() is not None:
                print("[serve] a server exited (rc %s)" % pr.returncode, flush=True)
                stop()
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
