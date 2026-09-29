"""NeMo-RL GRPO entry point with the Finance and IF environments registered."""
import os
import runpy
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

from nemo_rl.data.processors import register_processor
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import ENV_REGISTRY

from immopd.training.rl_nemo.envs.processors import if_processor

_ENVS = {
    "fin_numeric": "immopd.training.rl_nemo.envs.fin_environment.FinEnvironment",
    "if_constraints": "immopd.training.rl_nemo.envs.if_environment.IFEnvironment",
}
for name, fqn in _ENVS.items():
    ENV_REGISTRY[name] = {"actor_class_fqn": fqn}
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM

register_processor("if_processor", if_processor)

NEMO_RL_REPO = os.environ.get("NEMO_RL_REPO", "/opt/nemo-rl")
runpy.run_path(os.path.join(NEMO_RL_REPO, "examples", "run_grpo.py"), run_name="__main__")
