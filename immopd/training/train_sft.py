"""Supervised fine-tuning: reference models (OpenThoughts3), SFT teachers and the SFT warm-up baseline.

  accelerate launch --config_file configs/accelerate/zero2.yaml --num_processes 8 \
      -m immopd.training.train_sft --config configs/sft/ot3_4b.yaml

Loss is taken on the assistant turn only. Sequences are packed without padding (flash-attention).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    import liger_kernel.transformers  # noqa: F401
except Exception:
    pass

from immopd.common import chat
from immopd.common.config import add_config_args, dump, load_config
from immopd.common.resume import find_resume_checkpoint
from immopd.training.sft_dataset import SFTCollator, SFTDataset, build_mixture

DEFAULTS = {
    "model": {"base": "", "attn": "flash_attention_2"},
    "data": {"sources": {}, "cache_dir": None, "weights": None, "max_rows": None, "max_len": 32768,
             "thinking": True, "packing": "flatten", "seed": 42},
    "train": {"output": "${RUN_DIR}/sft", "epochs": 1.0, "lr": 1e-5, "scheduler": "cosine",
              "scheduler_kwargs": None, "warmup_ratio": 0.05, "weight_decay": 0.0, "max_grad_norm": 1.0,
              "batch_size": 32, "per_device_bs": 1, "logging_steps": 5, "save_strategy": "steps",
              "save_steps": 100, "save_total_limit": None, "save_only_model": False, "max_steps": -1,
              "resume": "auto", "seed": 42, "dataloader_workers": 2, "liger": True},
}


def is_main() -> bool:
    return os.environ.get("RANK", "0") in ("0", "") and os.environ.get("LOCAL_RANK", "0") in ("0", "")


def main(argv=None):
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[sft] config:\n" + dump(cfg), flush=True)

    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.base, trust_remote_code=True))
    os.makedirs(cfg.train.output, exist_ok=True)
    if cfg.data.get("cache_dir"):
        from immopd.training.ot3_cache import MemmapSFTDataset
        ds = MemmapSFTDataset(cfg.data.cache_dir)
        assert ds.stats["max_len"] == cfg.data.max_len, (ds.stats["max_len"], cfg.data.max_len)
    else:
        sources = dict(cfg.data.sources)
        missing = [p_ for p_ in sources.values() if not os.path.exists(p_)]
        assert not missing, f"missing data files: {missing}"
        rows = build_mixture(sources, cfg.data.weights, cfg.data.max_rows, cfg.data.seed)
        key = hashlib.sha1(json.dumps([sorted(sources.items()), cfg.data.weights, cfg.data.max_rows, cfg.data.max_len,
                                       cfg.data.thinking, cfg.data.packing, cfg.data.seed, cfg.model.base]).encode()).hexdigest()[:12]
        ds = SFTDataset(rows, tok, max_len=cfg.data.max_len, thinking=cfg.data.thinking, packing=cfg.data.packing,
                        seed=cfg.data.seed, cache_path=os.path.join(cfg.train.output, f"dataset_cache_{key}.pt"))
    if is_main():
        print(f"[sft] dataset: {json.dumps(ds.stats)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.base, dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation=cfg.model.attn)
    model.config.use_cache = False
    model.config.eos_token_id = tok.convert_tokens_to_ids(chat.IM_END)
    model.config.pad_token_id = tok.pad_token_id

    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert cfg.train.batch_size % (cfg.train.per_device_bs * world) == 0, (cfg.train.batch_size, cfg.train.per_device_bs, world)
    targs = TrainingArguments(
        output_dir=cfg.train.output,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=cfg.train.batch_size // (cfg.train.per_device_bs * world),
        learning_rate=cfg.train.lr, lr_scheduler_type=cfg.train.scheduler,
        lr_scheduler_kwargs=dict(cfg.train.get("scheduler_kwargs") or {}),
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay, max_grad_norm=cfg.train.max_grad_norm,
        seed=cfg.train.seed, bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=cfg.train.logging_steps,
        max_steps=cfg.train.max_steps,
        save_strategy=cfg.train.save_strategy, save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit, save_only_model=cfg.train.save_only_model,
        report_to="tensorboard", logging_dir=os.path.join(cfg.train.output, "tb"),
        remove_unused_columns=False, dataloader_num_workers=cfg.train.dataloader_workers,
        use_liger_kernel=bool(cfg.train.liger),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=SFTCollator(tok.pad_token_id),
                      processing_class=tok)
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[sft] resume = {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg.train.output)
    if is_main():
        tok.save_pretrained(cfg.train.output)
        with open(os.path.join(cfg.train.output, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.train.output, "sft_done.json"), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "dataset": ds.stats, "seconds": time.time() - t0}, f, indent=2)
        print("[sft] done", flush=True)


if __name__ == "__main__":
    main()
