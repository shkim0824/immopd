#!/bin/bash
# eval_tau2.sh <model> <out dir>     tau2-bench telecom, 114 base-split tasks, pass@1
#   OPENAI_API_KEY=... TAU2_PYTHON=<tau2-bench python> GPUS=0-7 scripts/eval_tau2.sh <model> ${RUN_DIR}/eval/tau2-immopd-4b
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]] && usage
run ${PY} -m immopd.eval.tau2_run --model "$1" --out "$2" --gpus "${GPUS}" --tau2-python "${TAU2_PYTHON:-${PY}}" \
    $([[ "${DRY_RUN:-0}" == 1 ]] && echo --dry-run)
