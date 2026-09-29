# Reproducing the paper

```bash
export MODEL_DIR=/path/to/models DATA_DIR=/path/to/data RUN_DIR=/path/to/runs
export TAU2_PYTHON=/path/to/tau2/bin/python OPENAI_API_KEY=...
```

`GPUS` selects the GPUs of a command (`GPUS=0-7`). The batch size of a config must be divisible by the number
of GPUs. Commands are shown for Qwen3-4B; the 1.7B configs end in `_1p7b` / `-1p7b`.

## 1. Data

```bash
scripts/prepare_data.sh download
scripts/prepare_data.sh ot3
scripts/prepare_data.sh pools
scripts/prepare_data.sh tau2
```

## 2. Reference models

```bash
scripts/train_sft.sh configs/sft/ot3_4b.yaml          # -> $MODEL_DIR/Qwen3-4B-OT3
scripts/train_sft.sh configs/sft/ot3_1p7b.yaml        # -> $MODEL_DIR/Qwen3-1.7B-OT3
```

## 3. Teachers (Tables 4 and 5)

Medical and Law: samples of Qwen3.6-35B-A3B on the prompt pools, filtered to correct, deduplicated samples.

```bash
scripts/gen_teacher.sh <Qwen3.6-35B-A3B> $DATA_DIR/pools/med_candidates.jsonl $RUN_DIR/gen/med
python -m immopd.training.filter_samples --domain med --gen-dir $RUN_DIR/gen/med \
    --pool $DATA_DIR/pools/med_candidates.jsonl --out $DATA_DIR/sft/med_sft.jsonl \
    --pool-out $DATA_DIR/pools/med_train.jsonl
scripts/gen_teacher.sh <Qwen3.6-35B-A3B> $DATA_DIR/pools/law_train.jsonl $RUN_DIR/gen/law
python -m immopd.training.filter_samples --domain law --gen-dir $RUN_DIR/gen/law \
    --pool $DATA_DIR/pools/law_train.jsonl --out $DATA_DIR/sft/law_sft.jsonl

scripts/train_sft.sh configs/sft/teacher_med_4b.yaml
scripts/train_sft.sh configs/sft/teacher_law_4b.yaml
scripts/train_sft.sh configs/sft/teacher_tau_4b.yaml
```

Finance and IF: GRPO with NeMo-RL. The teacher is the checkpoint of step 200; place it at
`$MODEL_DIR/teacher-fin-4b` and `$MODEL_DIR/teacher-if-4b`.

```bash
SIZE=4B NEMO_RL_REPO=<NeMo-RL> scripts/train_teacher_rl.sh configs/rl/grpo_fin.yaml
SIZE=4B NEMO_RL_REPO=<NeMo-RL> scripts/train_teacher_rl.sh configs/rl/grpo_if.yaml
```

## 4. Student initializations

```bash
scripts/merge_init.sh $MODEL_DIR/merge-4b-uniform tau=0.2,med=0.2,law=0.2,fin=0.2,if=0.2
scripts/merge_init.sh $MODEL_DIR/merge-4b-scaled  tau=0.8,med=0.5,law=0.3,fin=0.2,if=0.3
```

SFT warm-up (Table 6): samples of the five teachers on their own pools, mixed by token budget.

```bash
for d in med law fin if tau; do
  scripts/gen_teacher.sh $MODEL_DIR/teacher-$d-4b $DATA_DIR/pools/${d}_train.jsonl $RUN_DIR/gen/warmup-$d
done
for d in med law fin if; do
  python -m immopd.training.filter_samples --domain $d --gen-dir $RUN_DIR/gen/warmup-$d \
      --pool $DATA_DIR/pools/${d}_train.jsonl --out $DATA_DIR/sft/warmup_$d.jsonl
done
python -m immopd.training.tau2_filter --gen-dir $RUN_DIR/gen/warmup-tau --pool $DATA_DIR/pools/tau_train.jsonl \
    --ref $DATA_DIR/sft/tau2_sft.jsonl --out $DATA_DIR/sft/warmup_tau.jsonl
python -m immopd.data.build_warmup_mix --out $DATA_DIR/sft/warmup_4b.jsonl \
    --law 2.2 --med 62 --tau 100 --fin 0.36 --if 0.14 \
    --law-pool $DATA_DIR/sft/warmup_law.jsonl --med-pool $DATA_DIR/sft/warmup_med.jsonl \
    --tau-pool $DATA_DIR/sft/warmup_tau.jsonl --fin-pool $DATA_DIR/sft/warmup_fin.jsonl \
    --if-pool $DATA_DIR/sft/warmup_if.jsonl
scripts/train_sft.sh configs/sft/warmup_4b.yaml
```

## 5. MOPD and IM-MOPD (Table 1)

The teachers are served on their own GPUs while the student trains.

```bash
GPUS=0-7  scripts/serve_teachers.sh 4b
GPUS=8-15 scripts/train_mopd.sh   configs/mopd/qwen3-4b_base.yaml
GPUS=8-15 scripts/train_mopd.sh   configs/mopd/qwen3-4b_sft-warmup.yaml
GPUS=8-15 scripts/train_mopd.sh   configs/mopd/qwen3-4b_uniform-merge.yaml
GPUS=8-15 scripts/train_immopd.sh configs/immopd/qwen3-4b.yaml
```

IM-MOPD uses the validation set in `data/validation` and the reference scores `refs_4b.json` /
`refs_1p7b.json` (reference model and teachers on the validation set). For other models, measure them with

```bash
python -m immopd.eval.validation run --model <model> --out <dir> --gpus 0-7
python -m immopd.eval.tau2_run --model <model> --out <dir> --gpus 0-7 --split full \
    --task-ids-file data/validation/tau2_tasks.json --tau2-python $TAU2_PYTHON
python -m immopd.merging.recovery refs --base-acc ... --base-tau2 ... --teacher-acc med=...,law=...,fin=...,if=... \
    --teacher-tau2 ... --out refs.json
```

## 6. Evaluation

```bash
scripts/eval.sh      <checkpoint> $RUN_DIR/eval/<tag>          # MedQA, CaseHOLD, FinQA, IFBench: avg@3
scripts/eval_tau2.sh <checkpoint> $RUN_DIR/eval/tau2-<tag>     # tau2-Telecom: pass@1
python analysis/collect.py --eval-root $RUN_DIR/eval --base <tag> --teachers med=<tag>,... --tags <tag>,...
```

## 7. Ablations

| Result | Command |
|---|---|
| Table 2, all at initialization | `scripts/merge_init.sh $MODEL_DIR/merge-4b-at-init tau=1.1,med=0.5,law=0.5,fin=0.2,if=0.5`, then `configs/mopd/qwen3-4b_merge-at-init.yaml` |
| Table 2, all after MOPD | `python -m immopd.merging.task_vector build --adds tau=1.1,med=0.5,law=0.5,fin=0.2,if=0.5 ...`, then `apply` to `checkpoint-100` of `configs/mopd/qwen3-4b_base.yaml` |
| Tables 3, 9, 10 | `scripts/train_immopd.sh <config> --override immopd.gamma=<gamma> --override immopd.delta=<delta> --override train.output=<dir>` |
| Figure 1, Table 8 | `configs/mopd/qwen3-4b_{base,sft-warmup,uniform-merge,merge-scaled}.yaml`, `configs/immopd/qwen3-4b.yaml`; IF with `ifeval` |
| Figure 3a | `configs/immopd/intervention_4b.yaml` |
| Figure 3b, Table 11 | `scripts/merge_init.sh $MODEL_DIR/merge-medif-r40 med=0.4,if=0.6`, `configs/mopd/2dom_medif_r*.yaml`, `configs/immopd/2dom_medif.yaml` |
| Figure 3c | `configs/mopd/qwen3-4b_{base,uniform-merge,merge-scaled,merge-scaled-uniform,merge-normalized}.yaml` |
| Figure 4 | checkpoints 25 / 50 / 75 / 100 and `postmerge-*` of the Table 1 runs |
| Figure 5a | `python -m immopd.eval.prefix.medical --students <name>=<dir>/medqa.gen.jsonl,... --teacher $MODEL_DIR/teacher-med-4b --out <dir>` |
| Figure 5b | `python -m immopd.eval.prefix.token_stats`, then `PFX_N=<N> PFX_TEACHER_BASE=<teacher url> python -m immopd.eval.tau2_run --agent llm_agent_prefix ...` |

## 8. Tables and figures

```bash
python analysis/make_tables.py
python analysis/plot_fig1.py        # also plot_fig3a, plot_fig3b, plot_fig3c, plot_fig4, plot_fig5
```
