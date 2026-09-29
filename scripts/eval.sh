#!/bin/bash
# eval.sh <model> <out dir> [benchmarks=medqa,casehold,finqa,ifbench]     single-turn benchmarks, avg@3
#   GPUS=0-7 scripts/eval.sh ${RUN_DIR}/immopd/qwen3-4b/checkpoint-100 ${RUN_DIR}/eval/immopd-4b
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]] && usage
run ${PY} -m immopd.eval.run_all --model "$1" --out "$2" --benchmarks "${3:-medqa,casehold,finqa,ifbench}" --gpus "${GPUS}" \
    --n "${N:-3}" --seed "${SEED:-42}"
