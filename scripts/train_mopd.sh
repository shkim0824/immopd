#!/bin/bash
# train_mopd.sh <config> [--override key=value ...]     MOPD; the teachers must be served (serve_teachers.sh)
#   GPUS=8-15 scripts/train_mopd.sh configs/mopd/qwen3-4b_uniform-merge.yaml
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]] && usage
CONFIG="$1"; shift
launch immopd.distill.train_mopd "${CONFIG}" --override "teachers.endpoints_file=${TEACHER_ENDPOINTS:-${RUN_DIR}/teachers/teacher_endpoints.json}" "$@"
