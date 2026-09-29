#!/bin/bash
# serve_teachers.sh [size=4b] [out=${RUN_DIR}/teachers]     vLLM servers of the five teachers
#   GPUS=0-5 scripts/serve_teachers.sh 4b
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage
SIZE="${1:-4b}"; OUT="${2:-${RUN_DIR}/teachers}"; T=""
for d in ${DOMAINS:-tau med law fin if}; do T="${T:+${T},}${d}=${MODEL_DIR}/teacher-${d}-${SIZE}"; done
run ${PY} -m immopd.distill.serve_teachers --teachers "${T}" --gpus "${GPUS}" --out "${OUT}" $([[ "${DRY_RUN:-0}" == 1 ]] && echo --dry-run)
