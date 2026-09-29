"""CPU tests: merge rule, task vectors, teacher placement, configs, driver dry run."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from immopd.common.config import load_config
from immopd.distill.serve_teachers import place
from immopd.merging import recovery as R
from immopd.merging import task_vector as TV
from immopd.merging.delta_store import DeltaStore
from immopd.merging.merge_teachers import weight_map

KEY = "model.layers.0.self_attn.q_proj.weight"


def test_recovery_rule():
    refs = {"base": {"tau": 5.0, "med": 60.0, "law": 60.0, "fin": 60.0, "if": 25.0},
            "teacher": {"tau": 65.0, "med": 80.0, "law": 74.0, "fin": 75.0, "if": 57.0}}
    scores = {"tau": 5.0 + 0.5 * 60, "med": 60 + 0.7 * 20, "law": 60 + 0.2 * 14, "fin": 75.0, "if": 25 + 0.55 * 32}
    r, picked = R.decide(refs, scores, 0.6)
    assert picked == ["tau", "law", "if"], (r, picked)
    assert abs(r["med"] - 0.7) < 1e-9 and abs(r["fin"] - 1.0) < 1e-9
    assert R.decide(refs, scores, 0.4)[1] == ["law"]


def fake_checkpoint(d, scale):
    os.makedirs(d, exist_ok=True)
    sd = {KEY: torch.full((4, 3), float(scale)), "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2) * scale,
          "lm_head.weight": torch.zeros(3, 2)}
    save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"architectures": ["Fake"], "torch_dtype": "float32"}, open(os.path.join(d, "config.json"), "w"))


def test_task_vector():
    with tempfile.TemporaryDirectory() as t:
        p = lambda n: os.path.join(t, n)
        fake_checkpoint(p("base"), 1.0)
        fake_checkpoint(p("a"), 3.0)
        fake_checkpoint(p("b"), 0.0)
        out = TV.build(p("base"), {"a": p("a"), "b": p("b")}, {"a": 0.5, "b": 0.25}, p("delta"))
        st = DeltaStore(out)
        assert "lm_head.weight" not in st.map and st.info["adds"] == {"a": 0.5, "b": 0.25}
        assert torch.allclose(st.get(KEY), torch.full((4, 3), 0.75))
        TV.apply(p("base"), out, p("merged"), save_dtype="float32")
        got = safe_open(weight_map(p("merged"))[KEY], "pt").get_tensor(KEY)
        assert torch.allclose(got, torch.full((4, 3), 1.75))
        rc = subprocess.call([sys.executable, "-m", "immopd.merging.merge_teachers", "--base", p("base"), "--teachers",
                              "a=%s,b=%s" % (p("a"), p("b")), "--weights", "a=0.5,b=0.5", "--out", p("init")],
                             cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), stdout=subprocess.DEVNULL)
        assert rc == 0
        got = safe_open(weight_map(p("init"))[KEY], "pt").get_tensor(KEY)
        assert torch.allclose(got, torch.full((4, 3), 1.5))


def test_place():
    teachers = {d: "/m/" + d for d in ("tau", "med", "law", "fin", "if")}
    plan = place(list(range(6)), teachers)
    assert {d: len(u) for d, u in plan["endpoints"].items()} == {"tau": 2, "med": 1, "law": 1, "fin": 1, "if": 1}


def test_configs():
    os.environ.update(MODEL_DIR="/m", DATA_DIR="/d", RUN_DIR="/r")
    c = load_config(os.path.join(ROOT, "configs/immopd/qwen3-1p7b.yaml"))
    assert c.model.student == "/m/merge-1p7b-uniform" and c.immopd.gamma == 0.4 and c.immopd.delta == 0.3
    assert c.train.max_steps == 100 and c.train.batch_size == 128 and list(c.data.paths) == ["tau", "med", "law", "fin", "if"]
    c = load_config(os.path.join(ROOT, "configs/immopd/qwen3-4b.yaml"))
    assert c.immopd.gamma == 0.6 and c.immopd.teachers.med == "/m/teacher-med-4b"
    c = load_config(os.path.join(ROOT, "configs/mopd/2dom_medif_r60.yaml"))
    assert c.model.student == "/m/merge-medif-r60" and c.train.batch_size == 64 and list(c.data.paths) == ["med", "if"]
    for root, _, files in os.walk(os.path.join(ROOT, "configs")):
        for f in files:
            if root.endswith(("sft", "mopd", "immopd")):
                load_config(os.path.join(root, f))


def test_driver_dry_run():
    env = dict(os.environ, PYTHONPATH=ROOT, MODEL_DIR="/tmp/M", DATA_DIR="/tmp/D", RUN_DIR=tempfile.mkdtemp())
    def dry(cfg):
        r = subprocess.run([sys.executable, "-m", "immopd.distill.driver", "--config", cfg, "--gpus", "0-7",
                            "--teacher-endpoints", "/tmp/e.json", "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return r.stdout
    out = dry("configs/immopd/qwen3-4b.yaml")
    assert out.count("immopd.distill.train_immopd") == 4 and out.count("immopd.eval.validation run") == 3
    assert out.count("immopd.eval.tau2_run") == 3 and out.count("immopd.merging.recovery plan") == 3
    assert dry("configs/immopd/intervention_4b.yaml").count("immopd.distill.train_immopd") == 1


if __name__ == "__main__":
    os.chdir(ROOT)
    for f in (test_recovery_rule, test_task_vector, test_place, test_configs, test_driver_dry_run):
        f()
        print(f.__name__, "ok")
