"""NeMo-RL environment with an instruction-constraint reward (IF teacher)."""
from __future__ import annotations

import itertools
import os
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


class IFEnvConfig(TypedDict):
    num_workers: int
    worker_py_executable: NotRequired[str]
    worker_pythonpath: str
    nltk_data: str


class IFEnvMetadata(TypedDict):
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any] | None]
    prompt: NotRequired[str]


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class IFEnvironment(EnvironmentInterface):
    def __init__(self, cfg: IFEnvConfig):
        from immopd.training.rl_nemo.envs.if_verify_worker import IFVerifyWorker
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self._worker_counter = itertools.count()
        env_vars = {
            "PYTHONPATH": cfg["worker_pythonpath"] + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "NLTK_DATA": cfg["nltk_data"],
        }
        runtime_env = {"py_executable": PY_EXECUTABLES.SYSTEM, "env_vars": env_vars}
        if cfg.get("worker_py_executable"):
            runtime_env["py_executable"] = cfg["worker_py_executable"]
            env_vars["RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL"] = "minor"
        self.workers = [
            IFVerifyWorker.options(runtime_env=runtime_env).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for w in self.workers:
            ray.kill(w)

    def step(self, message_log_batch: list[LLMMessageLogType],
             metadata: list[IFEnvMetadata]) -> EnvironmentReturn:
        items = []
        for conv, meta in zip(message_log_batch, metadata):
            resp = "".join(str(m["content"]) for m in conv if m["role"] == "assistant")
            items.append({"response": resp,
                          "instruction_id_list": meta.get("instruction_id_list") or [],
                          "kwargs": meta.get("kwargs") or [],
                          "prompt": meta.get("prompt") or ""})
        widx = next(self._worker_counter) % self.num_workers
        chunks = chunk_list_to_workers(items, self.num_workers)
        futures = [self.workers[(widx + i) % self.num_workers].verify.remote(c)
                   for i, c in enumerate(chunks)]
        rewards: list[float] = []
        for r in ray.get(futures):
            rewards.extend(r)
        observations = [{"role": "environment",
                         "content": "Environment: correct" if r > 0.5 else "Environment: incorrect"}
                        for r in rewards]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_t,
            terminateds=torch.ones_like(rewards_t, dtype=torch.bool).cpu(),
            answers=[None] * len(message_log_batch),
        )

    def global_post_process_and_metrics(self, batch) -> tuple[Any, dict[str, float | int]]:
        rewards = batch["rewards"] * batch["is_end"]
        return batch, {
            "accuracy": rewards.mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], rewards),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
        }
