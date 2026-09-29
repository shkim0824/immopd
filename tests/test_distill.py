"""CPU tests: policy-gradient advantages, teacher response parsing, batch composition, chat helpers."""
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import torch

from immopd.common import chat as C
from immopd.common.config import load_config
from immopd.distill import loss as L
from immopd.distill.teacher_client import TeacherPool


def test_pg_advantages():
    t = torch.tensor([[-1.0, -2.0, -9.0]])
    s = torch.tensor([[-1.5, -1.0, -1.0]])
    a = L.pg_advantages(t, s, torch.tensor([[1.0, 1.0, 0.0]]), a_max=5.0)
    assert torch.allclose(a, torch.tensor([[0.5, -1.0, 0.0]]))
    assert L.pg_advantages(t, s, torch.ones(1, 3), a_max=5.0)[0, 2].item() == -5.0


def test_teacher_parse():
    ids = [11, 22, 33, 44]
    pl = [None, {"22": {"logprob": -0.5}}, {"9": {"logprob": -0.1}, "33": {"logprob": -3.0}}, {"44": {"logprob": -0.2}}]
    assert TeacherPool._parse(pl, ids, 2) == [-3.0, -0.2]
    assert TeacherPool._parse(pl, ids, 0) == [0.0, -0.5, -3.0, -0.2]


def test_equal_blocks():
    try:
        from immopd.distill.train_mopd import equal_blocks
    except ImportError:
        print("skip (trl not installed)")
        return
    doms = ["tau", "med", "law", "fin", "if"]
    per = {d: [{"domain": d, "i": i} for i in range(40)] for d in doms}
    rows = equal_blocks(per, doms, 128, 10, random.Random(0))
    for b in range(10):
        n = {d: sum(r["domain"] == d for r in rows[b * 128:(b + 1) * 128]) for d in doms}
        assert sum(n.values()) == 128 and set(n.values()) <= {25, 26}, n


def test_config_and_chat():
    cfg = load_config(None, ["train.lr=5e-7", "flag=true"], base={"train": {"lr": 1e-6}, "flag": False})
    assert cfg.train.lr == 5e-7 and cfg.flag is True
    assert C.normalize_assistant("ans", "why") == "<think>\nwhy\n</think>\n\nans"
    assert C.split_thinking("<think>\nr\n</think>\n\nA") == {"reasoning": "r", "answer": "A", "finished_thinking": True}


if __name__ == "__main__":
    for f in (test_pg_advantages, test_teacher_parse, test_equal_blocks, test_config_and_chat):
        f()
        print(f.__name__, "ok")
