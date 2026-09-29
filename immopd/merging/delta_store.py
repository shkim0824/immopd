"""Reads a task-vector increment tensor by tensor."""
from __future__ import annotations

import json
import os
import struct

import torch

DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


class DeltaStore:
    def __init__(self, d: str):
        assert os.path.exists(os.path.join(d, "DELTA_OK")), f"increment not complete: {d}"
        self.d = d
        meta = json.load(open(os.path.join(d, "delta.index.json")))
        self.map = meta["weight_map"]
        self.info = {k: v for k, v in meta.items() if k != "weight_map"}
        self.files = {}

    def _file(self, fn):
        if fn not in self.files:
            p = os.path.join(self.d, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
            self.files[fn] = (os.open(p, os.O_RDONLY), 8 + n, hdr)
        return self.files[fn]

    def shape(self, key):
        return tuple(self._file(self.map[key])[2][key]["shape"])

    def get(self, key) -> torch.Tensor:
        fd, off, hdr = self._file(self.map[key]); m = hdr[key]; s, e = m["data_offsets"]
        buf = bytearray(e - s); mv = memoryview(buf); got = 0
        while got < len(buf):
            r = os.preadv(fd, [mv[got:]], off + s + got)
            if r <= 0:
                raise IOError("short read " + key)
            got += r
        return torch.frombuffer(buf, dtype=DT[m["dtype"]]).reshape(m["shape"])

    def check_model(self, model) -> None:
        miss = [n for n, _ in model.named_parameters() if n not in self.map]
        bad = [n for n, p in model.named_parameters() if n in self.map and tuple(p.shape) != self.shape(n)]
        assert not miss and not bad, ("increment does not match the model", self.d, miss[:3], bad[:3])
