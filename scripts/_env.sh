#!/bin/bash
# Shared settings: MODEL_DIR, DATA_DIR, RUN_DIR, GPUS (e.g. 0-7), PYTHON. DRY_RUN=1 prints commands only.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export REPO PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_DIR="${MODEL_DIR:-${REPO}/models}" DATA_DIR="${DATA_DIR:-${REPO}/data}" RUN_DIR="${RUN_DIR:-${REPO}/runs}"
GPUS="${GPUS:-0-7}"; PY="${PYTHON:-python}"
gpu_list() { ${PY} -c "import sys; print(','.join(str(g) for p in sys.argv[1].split(',') for g in (range(int(p.split('-')[0]), int(p.split('-')[1]) + 1) if '-' in p else [int(p)])))" "$1"; }
GPU_LIST="$(gpu_list "${GPUS}")"; NGPU="$(echo "${GPU_LIST}" | tr ',' '\n' | wc -l | tr -d ' ')"
run() { echo "+ $*"; [[ "${DRY_RUN:-0}" == 1 ]] || "$@"; }
usage() { sed -n '2,/^source/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0; }
launch() {  # launch <module> <config> [args ...]
  local module="$1" config="$2"; shift 2
  local out; out="$(${PY} -c "import sys; from immopd.common.config import load_config; print(load_config(sys.argv[1]).train.output)" "${config}")"
  run ${PY} -m immopd.common.accel --config "${config}" --gpus "${NGPU}" --out "${out}/accelerate.yaml"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" run accelerate launch --config_file "${out}/accelerate.yaml" --num_processes "${NGPU}" \
      -m "${module}" --config "${config}" "$@"
}
