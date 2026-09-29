"""Chat format shared by training and evaluation (Qwen3 chat template, thinking mode).

Checkpoints are saved with ``<|im_end|>`` as the end-of-sequence token.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def prepare_tokenizer(tok):
    if tok.chat_template is None:
        raise ValueError("tokenizer has no chat_template; expected a Qwen3 tokenizer")
    if IM_END in tok.get_vocab():
        tok.eos_token = IM_END
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        tok.pad_token = ENDOFTEXT if ENDOFTEXT in tok.get_vocab() else tok.eos_token
    tok.padding_side = "right"
    return tok


def eos_token_ids(tok) -> List[int]:
    ids = []
    for t in (IM_END, ENDOFTEXT):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i not in ids:
            ids.append(i)
    return ids


def stop_strings() -> List[str]:
    return [IM_END, ENDOFTEXT]


def render_prompt(tok, messages: Sequence[Dict[str, str]], thinking: bool = True) -> str:
    return tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def render_prompt_ids(tok, messages: Sequence[Dict[str, str]], thinking: bool = True) -> List[int]:
    return tok(render_prompt(tok, messages, thinking), add_special_tokens=False)["input_ids"]


def normalize_assistant(content: str, reasoning: Optional[str] = None, thinking: bool = True) -> str:
    content = content or ""
    if reasoning is None and THINK_CLOSE in content:
        pre, post = content.split(THINK_CLOSE, 1)
        reasoning = pre.split(THINK_OPEN, 1)[-1]
        content = post
    reasoning = (reasoning or "").strip("\n")
    answer = content.lstrip("\n").rstrip()
    if not thinking:
        return answer
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{answer}"


def split_thinking(text: str) -> Dict[str, Any]:
    if THINK_CLOSE in text:
        pre, post = text.split(THINK_CLOSE, 1)
        return {"reasoning": pre.split(THINK_OPEN, 1)[-1].strip("\n"), "answer": post.lstrip("\n"),
                "finished_thinking": True}
    if THINK_OPEN in text:
        return {"reasoning": text.split(THINK_OPEN, 1)[1], "answer": "", "finished_thinking": False}
    return {"reasoning": "", "answer": text, "finished_thinking": True}


def generation_config_dict(thinking: bool = True) -> Dict[str, Any]:
    base = {"do_sample": True, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "top_k": 20}
    if thinking:
        base.update(temperature=0.6, top_p=0.95)
    else:
        base.update(temperature=0.7, top_p=0.8)
    return base


def save_model_with_contract(model, tok, out_dir: str, thinking: bool = True):
    import json
    import os

    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    gc = generation_config_dict(thinking)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg["eos_token_id"] = 151645
        cfg["pad_token_id"] = 151643
        json.dump(cfg, open(cfg_path, "w"), indent=2)
