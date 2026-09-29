#!/bin/bash
# gen_teacher.sh <model> <pool.jsonl> <out dir> [k=4]     teacher samples, one vLLM process per GPU
#   GPUS=0-7 scripts/gen_teacher.sh ${MODEL_DIR}/teacher-med-4b ${DATA_DIR}/pools/med_train.jsonl ${RUN_DIR}/gen/med-4b 4
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"; [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]] && usage
MODEL="$1"; POOL="$2"; OUT="$3"; K="${4:-4}"
mkdir -p "${OUT}"; pids=(); i=0
for g in ${GPU_LIST//,/ }; do
  echo "+ shard ${i}/${NGPU} on GPU ${g}"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    CUDA_VISIBLE_DEVICES="${g}" ${PY} -m immopd.training.gen_teacher --model "${MODEL}" --pool "${POOL}" --out "${OUT}" --k "${K}" \
        --num-shards "${NGPU}" --shard-rank "${i}" > "${OUT}/shard${i}.log" 2>&1 &
    pids+=($!)
  fi
  i=$((i + 1))
done
rc=0; for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done; exit ${rc}
