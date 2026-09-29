"""Defaults and process helpers shared by the MOPD and IM-MOPD trainers."""
from __future__ import annotations

import os
from datetime import timedelta

DEFAULTS = {
    "model": {"student": "", "dtype": "float32"},
    "teachers": {"endpoints_file": "", "max_workers": 64, "timeout": 900},
    "data": {"paths": {}, "seed": 42, "thinking": True, "max_prompt_chars": 24000, "max_prompt_tokens": 15872},
    "distill": {"a_max": 5.0},
    "train": {"lr": 1e-6, "batch_size": 128, "max_completion_length": 16384, "max_steps": 100,
              "per_device_bs": 1, "temperature": 1.0, "top_p": 1.0, "scheduler": "constant",
              "warmup_ratio": 0.05, "max_grad_norm": 1.0, "mask_truncated_completions": True,
              "save_steps": 25, "save_total_limit": None, "save_only_model": False, "resume": "auto",
              "logging_steps": 1, "vllm_gpu_memory_utilization": 0.25, "ddp_timeout": 7200,
              "output": "${RUN_DIR}/mopd"},
}


def is_main() -> bool:
    return os.environ.get("RANK", "0") in ("0", "")


def patch_runtime() -> None:
    """Disables vLLM's custom all-reduce in the colocated engine and raises the process-group timeout."""
    try:
        from vllm.entrypoints.llm import LLM as _LLM
        _orig = _LLM.__init__

        def _init(self, *a, **k):
            k["disable_custom_all_reduce"] = True
            _orig(self, *a, **k)
        _LLM.__init__ = _init
    except Exception:
        pass
    import torch.distributed as dist
    _o = dist.init_process_group

    def _pg(*a, **k):
        k.setdefault("timeout", timedelta(hours=2))
        return _o(*a, **k)
    dist.init_process_group = _pg
