"""Packed SFT cache as flat memory-mapped arrays, shared read-only by all training processes.

  tokens.bin          int32 token stream (bins of packed examples)
  bin_offsets.npy     start of every bin
  ex_starts.npy       start of every example
  ex_prompt_lens.npy  prompt length of every example (masked in the loss)
  stats.json          build statistics
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

IGNORE_INDEX = -100

TOKENS_BIN = "tokens.bin"
BIN_OFFSETS = "bin_offsets.npy"
EX_STARTS = "ex_starts.npy"
EX_PROMPT_LENS = "ex_prompt_lens.npy"
STATS_JSON = "stats.json"


def pack_bfd(lengths: List[int], max_len: int) -> List[List[int]]:
    """Best-fit-decreasing bin packing; returns bins as lists of example indices."""
    import bisect

    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    caps: List[int] = []
    cap_bins: List[int] = []
    bins: List[List[int]] = []
    for i in order:
        n = lengths[i]
        k = bisect.bisect_left(caps, n)
        if k < len(caps):
            bi = cap_bins.pop(k)
            rem = caps.pop(k) - n
            bins[bi].append(i)
        else:
            bins.append([i])
            bi = len(bins) - 1
            rem = max_len - n
        j = bisect.bisect_left(caps, rem)
        caps.insert(j, rem)
        cap_bins.insert(j, bi)
    return bins


def write_cache(out_dir: str, examples: List[Tuple[np.ndarray, int]], max_len: int,
                stats: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    import random

    os.makedirs(out_dir, exist_ok=True)
    lengths = [len(ids) for ids, _ in examples]
    assert all(l <= max_len for l in lengths), "example longer than max_len reached the packer"
    bins = pack_bfd(lengths, max_len)
    random.Random(seed).shuffle(bins)

    total = sum(lengths)
    bin_offsets = np.zeros(len(bins) + 1, dtype=np.int64)
    ex_starts = np.zeros(len(examples), dtype=np.int64)
    ex_plens = np.zeros(len(examples), dtype=np.int32)
    tmp = os.path.join(out_dir, TOKENS_BIN + ".tmp")
    pos = 0
    j = 0
    with open(tmp, "wb") as f:
        for bi, ex_idx in enumerate(bins):
            bin_offsets[bi] = pos
            for i in ex_idx:
                ids, plen = examples[i]
                f.write(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
                ex_starts[j] = pos
                ex_plens[j] = plen
                pos += len(ids)
                j += 1
        bin_offsets[len(bins)] = pos
    assert pos == total and j == len(examples)
    os.replace(tmp, os.path.join(out_dir, TOKENS_BIN))
    np.save(os.path.join(out_dir, BIN_OFFSETS), bin_offsets)
    np.save(os.path.join(out_dir, EX_STARTS), ex_starts)
    np.save(os.path.join(out_dir, EX_PROMPT_LENS), ex_plens)
    stats = dict(stats)
    stats.update(max_len=max_len, tokens=int(total), bins=len(bins),
                 fill=total / max(1, len(bins) * max_len), n_examples=len(examples), pack_seed=seed)
    with open(os.path.join(out_dir, STATS_JSON), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore


class MemmapSFTDataset(_TorchDataset):

    mode = "flatten"

    def __init__(self, cache_dir: str):
        for f in (TOKENS_BIN, BIN_OFFSETS, EX_STARTS, EX_PROMPT_LENS, STATS_JSON):
            assert os.path.exists(os.path.join(cache_dir, f)), f"cache incomplete: {cache_dir}/{f}"
        self.tokens = np.memmap(os.path.join(cache_dir, TOKENS_BIN), dtype=np.int32, mode="r")
        self.bin_offsets = np.load(os.path.join(cache_dir, BIN_OFFSETS))
        self.ex_starts = np.load(os.path.join(cache_dir, EX_STARTS))
        self.ex_prompt_lens = np.load(os.path.join(cache_dir, EX_PROMPT_LENS))
        self.stats = json.load(open(os.path.join(cache_dir, STATS_JSON)))
        assert int(self.bin_offsets[-1]) == len(self.tokens), "tokens.bin length mismatch"

    def __len__(self) -> int:
        return len(self.bin_offsets) - 1

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        s, e = int(self.bin_offsets[i]), int(self.bin_offsets[i + 1])
        ids = np.array(self.tokens[s:e], dtype=np.int32)
        labels = ids.copy()
        pos = np.empty(e - s, dtype=np.int32)
        j0 = int(np.searchsorted(self.ex_starts, s))
        j1 = int(np.searchsorted(self.ex_starts, e))
        for j in range(j0, j1):
            a = int(self.ex_starts[j]) - s
            b = (int(self.ex_starts[j + 1]) - s) if j + 1 < len(self.ex_starts) else e - s
            labels[a: a + int(self.ex_prompt_lens[j])] = IGNORE_INDEX
            pos[a:b] = np.arange(b - a, dtype=np.int32)
        return {"input_ids": ids, "labels": labels, "position_ids": pos}
