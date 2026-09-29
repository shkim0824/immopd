"""Multi-teacher on-policy distillation (MOPD).

  accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes 8 \
      -m immopd.distill.train_mopd --config configs/mopd/qwen3-4b_uniform-merge.yaml \
      --override teachers.endpoints_file=${RUN_DIR}/teachers/teacher_endpoints.json

The teachers are served by ``immopd.distill.serve_teachers``. Every batch holds the same number of prompts of
every domain.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig

from immopd.common import chat
from immopd.common.config import add_config_args, dump, load_config
from immopd.common.io import read_jsonl
from immopd.common.resume import find_resume_checkpoint
from immopd.distill.runtime import DEFAULTS, is_main, patch_runtime
from immopd.distill.teacher_client import TeacherPool, load_endpoints
from immopd.distill.trainer import MOPDTrainer


class StepTimer(TrainerCallback):
    def __init__(self):
        self.t0 = None

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is not None and is_main():
            print(f"[mopd] step {state.global_step} wall {time.time() - self.t0:.1f}s", flush=True)


def equal_blocks(per: dict, doms: list, B: int, n_blocks: int, rng: random.Random) -> list:
    """Batches of B prompts with B // D prompts of every domain; the B mod D remaining slots go to
    randomly drawn domains in every batch."""
    base, rem = divmod(B, len(doms))
    assert base > 0, (B, doms)
    ptr = {d: 0 for d in doms}
    out = []
    for _ in range(n_blocks):
        extra = set(rng.sample(doms, rem)) if rem else set()
        for d in doms:
            for _k in range(base + (1 if d in extra else 0)):
                if ptr[d] >= len(per[d]):
                    rng.shuffle(per[d])
                    ptr[d] = 0
                out.append(per[d][ptr[d]])
                ptr[d] += 1
    return out


def render_prompt(tok, messages: list, thinking: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def load_pools(cfg, tok) -> dict:
    """{domain: rows}. Pool rows are {input, output} (one user turn, rendered here) or pre-rendered
    {prompt_text, n_prompt_tokens} (Tool Use contexts, ``immopd.data.prep_tau2``)."""
    per = {}
    for d, p in dict(cfg.data.paths).items():
        if not p:
            continue
        rows = []
        for i, r in enumerate(read_jsonl(p)):
            if "prompt_text" in r:
                if int(r.get("n_prompt_tokens") or 0) > int(cfg.data.max_prompt_tokens):
                    continue
                text = r["prompt_text"]
            else:
                if len(r["input"]) > cfg.data.max_prompt_chars:
                    continue
                text = render_prompt(tok, [{"role": "user", "content": r["input"]}], cfg.data.thinking)
            rows.append({"prompt": text, "domain": d, "id": str(r.get("id") or f"{d}-{i}")})
        per[d] = rows
        if is_main():
            print(f"[mopd] {d}: {len(rows)} prompts", flush=True)
    return per


def build_dataset(cfg, tok) -> tuple[Dataset, dict]:
    rng = random.Random(cfg.data.seed)
    per = load_pools(cfg, tok)
    doms = list(per)
    for d in doms:
        rng.shuffle(per[d])
    n_blocks = int(cfg.train.max_steps * 1.05) + 2
    rows = equal_blocks(per, doms, int(cfg.train.batch_size), n_blocks, rng)
    return Dataset.from_list(rows), {d: len(per[d]) for d in doms}


def build_trainer(cfg, extra_callbacks=()):
    assert cfg.teachers.endpoints_file, "teachers.endpoints_file is required (immopd.distill.serve_teachers)"
    endpoints, names = load_endpoints(cfg.teachers.endpoints_file)
    for d, pth in dict(cfg.data.paths).items():
        assert not pth or d in endpoints, f"no teacher endpoint for domain {d}"
    pool = TeacherPool(endpoints, names, max_workers=cfg.teachers.max_workers, timeout=cfg.teachers.timeout)
    if is_main():
        print("[mopd] teachers:", pool.wait_ready(timeout=600), flush=True)
    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.student, trust_remote_code=True))
    ds, counts = build_dataset(cfg, tok)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gen_bs = cfg.train.batch_size
    assert gen_bs % (cfg.train.per_device_bs * world) == 0, (gen_bs, cfg.train.per_device_bs, world)
    grad_accum = gen_bs // (cfg.train.per_device_bs * world)
    targs = GRPOConfig(
        output_dir=cfg.train.output,
        model_init_kwargs={"dtype": cfg.model.dtype, "attn_implementation": "sdpa"},
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        generation_batch_size=gen_bs,
        num_generations=2,
        max_completion_length=cfg.train.max_completion_length,
        learning_rate=cfg.train.lr, lr_scheduler_type=cfg.train.scheduler, warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm, weight_decay=0.0,
        beta=0.0, loss_type="grpo", scale_rewards="none", epsilon=0.2, epsilon_high=0.2,
        mask_truncated_completions=cfg.train.mask_truncated_completions,
        temperature=cfg.train.temperature, top_p=cfg.train.top_p, num_iterations=1,
        save_steps=cfg.train.save_steps, save_strategy="steps", save_only_model=cfg.train.save_only_model,
        save_total_limit=cfg.train.save_total_limit, logging_steps=cfg.train.logging_steps,
        report_to="tensorboard", logging_dir=os.path.join(cfg.train.output, "tb"),
        bf16=True, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=cfg.train.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_importance_sampling_correction=True, vllm_importance_sampling_mode="token_truncate",
        vllm_enable_sleep_mode=False,
        chat_template_kwargs={"enable_thinking": cfg.data.thinking},
        remove_unused_columns=False, ddp_timeout=cfg.train.ddp_timeout, seed=cfg.data.seed,
        shuffle_dataset=False,
        log_completions=True, num_completions_to_print=2,
    )
    trainer = MOPDTrainer(model=cfg.model.student, args=targs, train_dataset=ds, processing_class=tok,
                          teacher_pool=pool, a_max=cfg.distill.a_max, num_generations_override=1,
                          callbacks=[StepTimer()] + list(extra_callbacks))
    info = {"counts": counts, "grad_accum": grad_accum, "world": world, "tok": tok, "n_rows": len(ds)}
    return trainer, info


def finish(cfg, trainer, info, t0, done_name="mopd_done.json"):
    final = os.path.join(cfg.train.output, "final")
    trainer.save_model(final)
    if is_main():
        info["tok"].save_pretrained(final)
        with open(os.path.join(final, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.train.output, done_name), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "counts": info["counts"],
                       "seconds": time.time() - t0}, f, indent=2)
        print("[mopd] done ->", final, flush=True)


def main(argv=None):
    patch_runtime()
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=copy.deepcopy(DEFAULTS))
    if is_main():
        print("[mopd] config:\n" + dump(cfg), flush=True)
    trainer, info = build_trainer(cfg)
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[mopd] prompts={info['counts']} world={info['world']} grad_accum={info['grad_accum']} "
              f"resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    finish(cfg, trainer, info, t0)


if __name__ == "__main__":
    main()
