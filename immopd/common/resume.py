"""Find the latest complete checkpoint of a run."""
from __future__ import annotations

import glob
import os
import re


def _weights_complete(d: str) -> bool:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        import json
        try:
            shards = set(json.load(open(idx))["weight_map"].values())
        except Exception:
            return False
        return all(os.path.isfile(os.path.join(d, f)) for f in shards)
    return os.path.isfile(os.path.join(d, "model.safetensors"))


def _complete(d: str, allow_model_only: bool = False) -> bool:
    if not os.path.isfile(os.path.join(d, "trainer_state.json")):
        return False
    if allow_model_only and not glob.glob(os.path.join(d, "global_step*")) and not os.path.isfile(os.path.join(d, "optimizer.pt")) \
            and _weights_complete(d):
        return True
    if glob.glob(os.path.join(d, "global_step*")):
        return bool(glob.glob(os.path.join(d, "global_step*", "*model_states.pt"))) or \
            bool(glob.glob(os.path.join(d, "global_step*", "*optim_states.pt")))
    return os.path.isfile(os.path.join(d, "optimizer.pt")) or os.path.isfile(os.path.join(d, "scheduler.pt"))


def list_checkpoints(output_dir: str):
    out = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            out.append((int(m.group(1)), d))
    return sorted(out)


def find_resume_checkpoint(output_dir: str, mode: str = "auto", allow_model_only: bool = False):
    if mode in (None, "", "none", "false", "False"):
        return None
    if mode != "auto":
        assert os.path.isdir(mode), f"resume checkpoint not found: {mode}"
        return mode
    latest = None
    for step, d in list_checkpoints(output_dir):
        if _complete(d, allow_model_only):
            latest = d
        elif _complete(d, True):
            continue
        else:
            if os.environ.get("RANK", "0") in ("0", ""):
                try:
                    os.rename(d, d + ".partial")
                except OSError:
                    pass
    return latest
