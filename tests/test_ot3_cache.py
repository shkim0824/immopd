"""CPU test: round trip of the packed SFT cache."""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import tempfile

import numpy as np

from immopd.training.ot3_cache import IGNORE_INDEX, MemmapSFTDataset, pack_bfd, write_cache

MAX_LEN = 64


def _fake_examples(n=200, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        total = rng.randint(4, MAX_LEN)
        plen = rng.randint(1, total - 1)
        ids = np.arange(i * 1000, i * 1000 + total, dtype=np.int32)
        out.append((ids, plen))
    return out


def test_pack_bfd():
    lens = [len(ids) for ids, _ in _fake_examples()]
    bins = pack_bfd(lens, MAX_LEN)
    assert sorted(i for b in bins for i in b) == list(range(len(lens)))
    assert all(sum(lens[i] for i in b) <= MAX_LEN for b in bins)
    print("pack ok:", len(bins), "bins, fill", round(sum(lens) / (len(bins) * MAX_LEN), 3))


def test_roundtrip():
    exs = _fake_examples()
    d = tempfile.mkdtemp(prefix="ot3cache_")
    try:
        stats = write_cache(d, exs, MAX_LEN, {"n_kept": len(exs)}, seed=7)
        assert stats["tokens"] == sum(len(ids) for ids, _ in exs)
        ds = MemmapSFTDataset(d)
        assert len(ds) == stats["bins"]
        by_first = {int(ids[0]): (ids, plen) for ids, plen in exs}
        seen = 0
        for i in range(len(ds)):
            b = ds[i]
            ids, lab, pos = b["input_ids"], b["labels"], b["position_ids"]
            assert len(ids) == len(lab) == len(pos) <= MAX_LEN
            starts = [0] + [k for k in range(1, len(pos)) if pos[k] == 0] + [len(pos)]
            for a, e in zip(starts[:-1], starts[1:]):
                orig, plen = by_first[int(ids[a])]
                assert np.array_equal(ids[a:e], orig)
                assert np.all(lab[a: a + plen] == IGNORE_INDEX)
                assert np.array_equal(lab[a + plen: e], orig[plen:])
                assert np.array_equal(pos[a:e], np.arange(e - a))
                seen += 1
        assert seen == len(exs)
        print("roundtrip ok:", seen, "examples across", len(ds), "bins")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_pack_bfd()
    test_roundtrip()
    print("ALL OK")
