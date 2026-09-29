#!/bin/bash
# train_sft.sh <config> [--override key=value ...]     reference models, SFT teachers, SFT warm-up
#   GPUS=0-7 scripts/train_sft.sh configs/sft/teacher_med_4b.yaml
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]] && usage
launch immopd.training.train_sft "$@"
