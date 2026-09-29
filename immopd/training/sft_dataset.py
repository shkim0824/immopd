"""SFT dataset: chat-template prompts with the loss on the assistant turn.

With ``packing="flatten"`` examples are concatenated into bins of at most ``max_len`` tokens; position ids
restart for every example, so flash-attention treats them as separate sequences (no padding).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from immopd.common import chat
from immopd.common.io import read_jsonl

IGNORE_INDEX = -100


def build_mixture(sources: Dict[str, str], weights: Optional[Dict[str, float]], max_rows: Optional[int],
                  seed: int) -> List[Dict[str, Any]]:
    pools = {d: read_jsonl(p) for d, p in sources.items()}
    rng = random.Random(seed)
    for rows in pools.values():
        rng.shuffle(rows)
    if not weights:
        out = [r for rows in pools.values() for r in rows]
        rng.shuffle(out)
        return out[:max_rows] if max_rows else out
    total = max_rows or sum(len(v) for v in pools.values())
    wsum = sum(weights.values())
    out = []
    for d, w in weights.items():
        k = min(int(round(total * w / wsum)), len(pools[d]))
        out.extend(pools[d][:k])
    rng.shuffle(out)
    return out


_W = {}


def _init_worker(tok, max_len, thinking):
    _W["ds"] = SFTDataset.__new__(SFTDataset)
    _W["ds"].tok, _W["ds"].max_len, _W["ds"].thinking = tok, max_len, thinking
    _W["ds"].eos = tok.convert_tokens_to_ids(chat.IM_END)
    _W["ds"].stats = {"dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0}


def _encode_chunk(rows):
    ds = _W["ds"]
    ds.stats = {"dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0}
    out = []
    for r in rows:
        ex = ds.encode_row(r)
        if ex is not None:
            out.append(ex)
            ds.stats["tokens"] += len(ex["input_ids"])
    return out, dict(ds.stats)


def _as_mode(packing) -> str:
    if packing is True:
        return "concat"
    if packing in (False, None, "", "none"):
        return "none"
    return str(packing)


class SFTDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tok, max_len: int, thinking: bool = True,
                 packing="none", seed: int = 0, cache_path: Optional[str] = None):
        self.tok = tok
        if cache_path:
            rank = int(os.environ.get("RANK", "0") or 0)
            if rank != 0 or os.path.exists(cache_path):
                for _ in range(720):
                    if os.path.exists(cache_path):
                        break
                    time.sleep(10)
                d = torch.load(cache_path, weights_only=False)
                self.max_len, self.thinking, self.mode = max_len, thinking, _as_mode(packing)
                self.eos = tok.convert_tokens_to_ids(chat.IM_END)
                self.examples, self.stats = d["examples"], d["stats"]
                return
        self.max_len = max_len
        self.thinking = thinking
        self.mode = _as_mode(packing)
        self.eos = tok.convert_tokens_to_ids(chat.IM_END)
        self.examples: List[Dict[str, List[int]]] = []
        self.stats = {"n_in": len(rows), "dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0, "packing": self.mode}
        nproc = int(os.environ.get("TOKENIZE_PROCS", "0")) or max(1, min(32, (os.cpu_count() or 4) - 2))
        if len(rows) >= 2000 and nproc > 1:
            from concurrent.futures import ProcessPoolExecutor
            chunks = [list(rows[i::nproc]) for i in range(nproc)]
            with ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker, initargs=(tok, max_len, thinking)) as ex:
                results = list(ex.map(_encode_chunk, chunks))
            for exs, st in results:
                self.examples.extend(exs)
                for k in ("dropped_long_completion", "prompt_truncated", "tokens"):
                    self.stats[k] += st[k]
        else:
            for r in rows:
                ex = self.encode_row(r)
                if ex is not None:
                    self.examples.append(ex)
                    self.stats["tokens"] += len(ex["input_ids"])
        if self.mode in ("concat", "flatten"):
            self.examples = self._pack(self.examples, seed, with_positions=(self.mode == "flatten"))
            self.stats["bins"] = len(self.examples)
            self.stats["fill"] = self.stats["tokens"] / max(1, len(self.examples) * max_len)
        self.stats["n_out"] = len(self.examples)
        if cache_path:
            tmp = cache_path + ".tmp"
            torch.save({"examples": self.examples, "stats": self.stats}, tmp)
            os.replace(tmp, cache_path)

    def encode_row(self, r: Dict[str, Any]):
        if "prompt_text" in r:
            return self._encode_text(r["prompt_text"], r["completion"])
        return self._encode(r["messages"])

    def _encode_text(self, prompt_text: str, completion: str):
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        comp_ids = self.tok(completion, add_special_tokens=False)["input_ids"] + [self.eos]
        return self._finish(prompt_ids, comp_ids)

    def _encode(self, messages: List[Dict[str, str]]):
        assert messages[-1]["role"] == "assistant"
        prompt_ids = chat.render_prompt_ids(self.tok, messages[:-1], thinking=self.thinking)
        comp = messages[-1]["content"]
        if not self.thinking:
            comp = chat.split_thinking(comp)["answer"]
        comp_ids = self.tok(comp, add_special_tokens=False)["input_ids"] + [self.eos]
        return self._finish(prompt_ids, comp_ids)

    def _finish(self, prompt_ids, comp_ids):
        if len(comp_ids) >= self.max_len:
            self.stats["dropped_long_completion"] += 1
            return None
        overflow = len(prompt_ids) + len(comp_ids) - self.max_len
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]
            self.stats["prompt_truncated"] += 1
        return {"input_ids": np.asarray(prompt_ids + comp_ids, dtype=np.int32),
                "labels": np.asarray([IGNORE_INDEX] * len(prompt_ids) + comp_ids, dtype=np.int32)}

    def _pack(self, exs, seed, with_positions: bool):
        """Best-fit-decreasing bin packing into bins of at most ``max_len`` tokens."""
        import bisect
        rng = random.Random(seed)
        exs = sorted(exs, key=lambda e: -len(e["input_ids"]))
        caps: List[int] = []
        cap_bins: List[int] = []
        parts: List[List[Dict[str, np.ndarray]]] = []
        for e in exs:
            n = len(e["input_ids"])
            k = bisect.bisect_left(caps, n)
            if k < len(caps):
                bi = cap_bins.pop(k)
                rem = caps.pop(k) - n
                parts[bi].append(e)
            else:
                parts.append([e])
                bi = len(parts) - 1
                rem = self.max_len - n
            j = bisect.bisect_left(caps, rem)
            caps.insert(j, rem)
            cap_bins.insert(j, bi)
        bins = []
        for ps in parts:
            b = {"input_ids": np.concatenate([e["input_ids"] for e in ps]),
                 "labels": np.concatenate([e["labels"] for e in ps])}
            if with_positions:
                b["position_ids"] = np.concatenate([np.arange(len(e["input_ids"]), dtype=np.int32) for e in ps])
            bins.append(b)
        rng.shuffle(bins)
        return bins

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if all("position_ids" in b for b in batch):
            ids = np.concatenate([np.asarray(b["input_ids"]) for b in batch])
            lab = np.concatenate([np.asarray(b["labels"]) for b in batch])
            pos = np.concatenate([np.asarray(b["position_ids"]) for b in batch])
            return {"input_ids": torch.from_numpy(ids.astype(np.int64))[None], "labels": torch.from_numpy(lab.astype(np.int64))[None],
                    "position_ids": torch.from_numpy(pos.astype(np.int64))[None]}
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), IGNORE_INDEX, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.as_tensor(np.asarray(b["input_ids"], dtype=np.int64))
            lab[i, :n] = torch.as_tensor(np.asarray(b["labels"], dtype=np.int64))
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
