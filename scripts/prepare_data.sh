#!/bin/bash
# prepare_data.sh <stage>
#   download   models, datasets and benchmark repositories -> ${MODEL_DIR}, ${DATA_DIR}/raw
#   ot3        OpenThoughts3 caches for the reference models -> ${DATA_DIR}/ot3_cache_len{16384,18432}
#   pools      Medical / Law / Finance / IF prompt pools -> ${DATA_DIR}/pools
#   tau2       Tool Use SFT data and prompt pool -> ${DATA_DIR}/sft/tau2_sft.jsonl, ${DATA_DIR}/pools/tau_train.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]] && usage
RAW="${DATA_DIR}/raw"
case "$1" in
  download) run ${PY} -m immopd.data.download --dest "${RAW}" --models "${MODEL_DIR}" ${ONLY:+--only "${ONLY}"} ;;
  ot3) for len in 16384 18432; do
         run ${PY} -m immopd.data.prep_openthoughts3 --src "${RAW}/hf/open-thoughts__OpenThoughts3-1.2M" \
             --out "${DATA_DIR}/ot3_cache_len${len}" --model "${MODEL_DIR}/Qwen3-4B-Base" --max-len "${len}" --procs "${PROCS:-32}"
       done ;;
  pools) run ${PY} -m immopd.data.prep_pools --out "${DATA_DIR}/pools" ;;
  tau2) run ${PY} -m immopd.data.prep_tau2 sft --out "${DATA_DIR}/sft/tau2_sft.jsonl"
        run ${PY} -m immopd.data.prep_tau2 pool --out "${DATA_DIR}/pools/tau_train.jsonl" ;;
  *) usage ;;
esac
