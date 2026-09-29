"""Offline vLLM generation on one GPU. Prompts are rendered with the chat template used in training."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from immopd.common import chat


class VllmGenerator:
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, max_model_len: int = 40960,
                 gpu_mem_util: float = 0.90, seed: int = 0, enable_prefix_caching: Optional[bool] = None):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True))
        cfg = os.path.join(model_path, "config.json")
        if os.path.exists(cfg):
            max_model_len = min(max_model_len, int(json.load(open(cfg)).get("max_position_embeddings", max_model_len)))
        extra = {} if enable_prefix_caching is None else {"enable_prefix_caching": enable_prefix_caching}
        self.llm = LLM(model=model_path, tokenizer=tokenizer_path or model_path, dtype="bfloat16",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
                       trust_remote_code=True, seed=seed, **extra)
        self.stop_ids = chat.eos_token_ids(self.tok)
        self.max_model_len = max_model_len

    def generate(self, items: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """items: {messages, n, max_tokens, sampling, seed} -> per item n dicts {text, n_tokens, finish_reason}."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts, sps = [], []
        for it in items:
            ids = chat.render_prompt_ids(self.tok, it["messages"], thinking=it.get("thinking", True))
            s = dict(it.get("sampling") or {})
            budget = max(16, min(int(it.get("max_tokens", 32768)), self.max_model_len - len(ids)))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            sps.append(SamplingParams(n=int(it.get("n", 1)), max_tokens=budget,
                                      temperature=s.get("temperature", 1.0),
                                      top_p=s.get("top_p", 1.0), top_k=s.get("top_k", -1),
                                      presence_penalty=s.get("presence_penalty", 0.0),
                                      seed=it.get("seed", 0),
                                      stop_token_ids=self.stop_ids, skip_special_tokens=True))
        outs = self.llm.generate(prompts, sps, use_tqdm=True)
        return [[{"text": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                 for c in r.outputs] for r in outs]
