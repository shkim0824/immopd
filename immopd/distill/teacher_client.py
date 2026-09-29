"""Teacher client: log-probabilities of a student rollout under a frozen teacher served by vLLM.

The full token sequence (prompt + completion) is sent as the prompt of a one-token completion request with
``prompt_logprobs``; the response carries the teacher log-probability of every token of the sequence.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


class TeacherPool:
    def __init__(self, endpoints: Dict[str, List[str]], served_names: Optional[Dict[str, str]] = None,
                 max_workers: int = 64, timeout: float = 600.0, retries: int = 3):
        self.endpoints = {d: list(u) for d, u in endpoints.items()}
        self.names = served_names or {d: d for d in endpoints}
        self.timeout = timeout
        self.retries = retries
        self._rr = {d: itertools.count() for d in endpoints}
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._session = threading.local()
        self.stats = {"requests": 0, "failures": 0, "seconds": 0.0, "tokens": 0}
        self._lock = threading.Lock()

    def _sess(self) -> requests.Session:
        s = getattr(self._session, "s", None)
        if s is None:
            s = requests.Session()
            self._session.s = s
        return s

    def healthcheck(self) -> Dict[str, bool]:
        out = {}
        for urls in self.endpoints.values():
            for u in urls:
                try:
                    out[u] = self._sess().get(f"{u}/v1/models", timeout=10).status_code == 200
                except Exception:
                    out[u] = False
        return out

    def wait_ready(self, timeout: float = 3600.0, interval: float = 10.0):
        t0 = time.time()
        while True:
            h = self.healthcheck()
            if all(h.values()):
                return h
            if time.time() - t0 > timeout:
                raise RuntimeError(f"teacher servers not ready: {h}")
            time.sleep(interval)

    def _url(self, domain: str) -> str:
        urls = self.endpoints[domain]
        return urls[next(self._rr[domain]) % len(urls)]

    def _post(self, domain: str, token_ids: Sequence[int]) -> List[Optional[Dict[str, Any]]]:
        body = {"model": self.names[domain], "prompt": list(map(int, token_ids)), "max_tokens": 1,
                "temperature": 1.0, "prompt_logprobs": 0, "logprobs": 0, "seed": 0}
        last = None
        for attempt in range(self.retries):
            url = self._url(domain)
            t0 = time.time()
            try:
                r = self._sess().post(f"{url}/v1/completions", json=body, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"{url} -> {r.status_code}: {r.text[:300]}")
                pl = r.json()["choices"][0]["prompt_logprobs"]
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["seconds"] += time.time() - t0
                    self.stats["tokens"] += len(token_ids)
                return pl
            except Exception as e:  # noqa: BLE001
                last = e
                with self._lock:
                    self.stats["failures"] += 1
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"teacher request failed for domain={domain} after {self.retries} tries: {last}")

    @staticmethod
    def _parse(pl: List[Optional[Dict[str, Any]]], token_ids: Sequence[int], start: int) -> List[float]:
        logps: List[float] = []
        for pos in range(start, len(token_ids)):
            entry = pl[pos] if pos < len(pl) else None
            if not entry:
                logps.append(0.0)
                continue
            actual = entry.get(str(int(token_ids[pos])))
            logps.append(float(actual["logprob"]) if actual is not None else -1e4)
        return logps

    def prefill_one(self, domain: str, token_ids: Sequence[int], completion_start: int) -> List[float]:
        return self._parse(self._post(domain, token_ids), token_ids, completion_start)

    def prefill_batch(self, domains: Sequence[str], token_id_lists: Sequence[Sequence[int]],
                      completion_starts: Sequence[int]) -> List[List[float]]:
        futs = [self._pool.submit(self.prefill_one, d, ids, s)
                for d, ids, s in zip(domains, token_id_lists, completion_starts)]
        return [f.result() for f in futs]

    def flush_stats(self) -> Dict[str, float]:
        with self._lock:
            s = dict(self.stats)
            self.stats = {"requests": 0, "failures": 0, "seconds": 0.0, "tokens": 0}
        return {"teacher/requests": s["requests"], "teacher/failures": s["failures"],
                "teacher/sec_per_request": s["seconds"] / max(s["requests"], 1),
                "teacher/tokens_per_request": s["tokens"] / max(s["requests"], 1)}


def load_endpoints(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    p = json.load(open(path))
    names = {s["domain"]: s["served_name"] for s in p.get("servers", [])}
    return p["endpoints"], names
