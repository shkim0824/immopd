#!/bin/bash
# train_teacher_rl.sh <config> [overrides ...]     Finance / IF teachers with NeMo-RL GRPO
#   SIZE=4B NEMO_RL_REPO=<NeMo-RL checkout> scripts/train_teacher_rl.sh configs/rl/grpo_fin.yaml
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]] && usage
CONFIG="$1"; shift
export SIZE="${SIZE:-4B}" NEMO_RL_REPO="${NEMO_RL_REPO:-/opt/nemo-rl}" NEMO_EXTRAS="${NEMO_EXTRAS:-${REPO}/third_party}" NLTK_DATA="${NLTK_DATA:-${HOME}/nltk_data}"
RESOLVED="${RUN_DIR}/rl/$(basename "${CONFIG}" .yaml)_${SIZE}.yaml"; mkdir -p "$(dirname "${RESOLVED}")"
${PY} - "${CONFIG}" "${RESOLVED}" <<'PY'
import os, re, sys
s = open(sys.argv[1]).read()
s = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), s)
open(sys.argv[2], "w").write(s)
PY
run "${NEMO_PYTHON:-python}" "${REPO}/immopd/training/rl_nemo/run_grpo.py" --config "${RESOLVED}" "$@"
