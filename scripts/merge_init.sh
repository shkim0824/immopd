#!/bin/bash
# merge_init.sh <out> <coefficients> [size=4b]     merge initialization by task arithmetic
#   scripts/merge_init.sh ${MODEL_DIR}/merge-4b-uniform tau=0.2,med=0.2,law=0.2,fin=0.2,if=0.2
#   scripts/merge_init.sh ${MODEL_DIR}/merge-4b-scaled  tau=0.8,med=0.5,law=0.3,fin=0.2,if=0.3
#   scripts/merge_init.sh ${MODEL_DIR}/merge-medif-r40  med=0.4,if=0.6
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]] && usage
OUT="$1"; W="$2"; SIZE="${3:-4b}"; T=""
[[ "${SIZE}" == "4b" ]] && BASE="${MODEL_DIR}/Qwen3-4B-OT3" || BASE="${MODEL_DIR}/Qwen3-1.7B-OT3"
for d in $(echo "${W}" | tr ',' '\n' | cut -d= -f1); do T="${T:+${T},}${d}=${MODEL_DIR}/teacher-${d}-${SIZE}"; done
run ${PY} -m immopd.merging.merge_teachers --base "${BASE}" --teachers "${T}" --weights "${W}" --out "${OUT}"
