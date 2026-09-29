#!/bin/bash
# train_immopd.sh <config> [driver args ...]     IM-MOPD; the teachers must be served (serve_teachers.sh)
#   OPENAI_API_KEY=... TAU2_PYTHON=<tau2-bench python> GPUS=8-15 scripts/train_immopd.sh configs/immopd/qwen3-4b.yaml
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]] && usage
CONFIG="$1"; shift
run ${PY} -m immopd.distill.driver --config "${CONFIG}" --gpus "${GPUS}" --tau2-python "${TAU2_PYTHON:-${PY}}" \
    --teacher-endpoints "${TEACHER_ENDPOINTS:-${RUN_DIR}/teachers/teacher_endpoints.json}" $([[ "${DRY_RUN:-0}" == 1 ]] && echo --dry-run) "$@"
