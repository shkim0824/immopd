"""Tool Use data from inclusionAI/AReaL-tau2-data (retail, airline, telecom).

  python -m immopd.data.prep_tau2 sft  --out ${DATA_DIR}/sft/tau2_sft.jsonl       teacher SFT rows
  python -m immopd.data.prep_tau2 pool --out ${DATA_DIR}/pools/tau_train.jsonl    MOPD prompts

Every row of the source is one assistant turn with its dialogue context. ``sft`` renders context and target
turn with the chat template and the tool schemas of the sub-domain (``data/tau2/tools_<domain>.json``);
supervision covers the target turn, earlier thinking is removed from the context. ``pool`` renders the
context as the prompt for the next assistant turn. Overlength rows and telecom rows whose task id is in the
base or test split of tau2-bench or in the validation task list are removed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from immopd.common.io import data_root, repo_root, write_jsonl

IM_END = "<|im_end|>"
DOMAINS = ("airline", "retail", "telecom")


def sub_domain(md: dict) -> str:
    if "task_id" in md:
        return "telecom"
    if "seed_pattern_task_id" in md:
        return "airline"
    if "scenario_id" in md:
        return "retail"
    return "unknown"


def tool_call(t: dict, as_string: bool) -> dict:
    f = t.get("function") if isinstance(t.get("function"), dict) else t
    name, args = f.get("name"), f.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {"type": "function", "function": {"name": name, "arguments": args}}
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args) if as_string else args}}


def sft_context(msgs: list) -> list:
    out = []
    for m in msgs:
        m2 = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant" and m.get("tool_calls"):
            m2["tool_calls"] = [tool_call(t, True) for t in m["tool_calls"]]
        out.append(m2)
    return out


def sft_target(ans: dict) -> dict:
    m = {"role": "assistant", "content": ans.get("content") or ""}
    if ans.get("thinking"):
        m["reasoning_content"] = ans["thinking"]
    if ans.get("tool_calls"):
        m["tool_calls"] = [tool_call(t, True) for t in ans["tool_calls"]]
    return m


def pool_context(msgs: list) -> list:
    out = []
    for m in msgs:
        m2 = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            if m.get("reasoning"):
                m2["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                m2["tool_calls"] = [tool_call(t, False) for t in m["tool_calls"]]
        out.append(m2)
    return out


def sft_row(r, md, d, tok, tools, max_len):
    ctx, tgt = sft_context(r["messages"]), sft_target(r["answer"])
    if not (tgt["content"].strip() or tgt.get("tool_calls")):
        return None
    prompt = tok.apply_chat_template(ctx, tools=tools[d], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full = tok.apply_chat_template(ctx + [tgt], tools=tools[d], tokenize=False, add_generation_prompt=False, enable_thinking=True)
    if not full.startswith(prompt):
        return None
    comp = full[len(prompt):]
    for end in (IM_END + "\n", IM_END):
        if comp.endswith(end):
            comp = comp[: -len(end)]
            break
    if tgt.get("reasoning_content") and "</think>" not in comp:
        return None
    n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
    n_c = len(tok(comp, add_special_tokens=False)["input_ids"]) + 1
    if n_p + n_c > max_len:
        return None
    return {"prompt_text": prompt, "completion": comp, "sub_domain": d, "n_prompt_tokens": n_p,
            "source_dialog_id": md.get("source_dialog_id"), "turn_index": md.get("turn_index")}


def pool_row(i, r, md, d, tok, tools, max_prompt_tokens):
    text = tok.apply_chat_template(pool_context(r["messages"]), tools=tools[d], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    if n > max_prompt_tokens:
        return None
    return {"id": "tau-%s-%s-t%s-%d" % (d, md.get("source_dialog_id"), md.get("turn_index"), i), "domain": "tau",
            "sub_domain": d, "prompt_text": text, "n_prompt_tokens": n,
            "source_dialog_id": md.get("source_dialog_id"), "turn_index": md.get("turn_index")}


def main(argv=None) -> int:
    raw = os.path.join(data_root(), "raw")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sft", "pool"])
    ap.add_argument("--src", default=os.path.join(raw, "hf/inclusionAI__AReaL-tau2-data/tau2_sft_train.jsonl"))
    ap.add_argument("--tau2-telecom", default=os.path.join(raw, "third_party/tau2-bench/data/tau2/domains/telecom"))
    ap.add_argument("--tools-dir", default=os.path.join(repo_root(), "data", "tau2"))
    ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("MODEL_DIR", "models"), "Qwen3-4B-Base"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=32768, help="sft: prompt + completion tokens")
    ap.add_argument("--max-prompt-tokens", type=int, default=15872, help="pool: prompt tokens")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tools = {d: json.load(open(os.path.join(a.tools_dir, "tools_%s.json" % d))) for d in DOMAINS}
    splits = json.load(open(os.path.join(a.tau2_telecom, "split_tasks.json")))
    eval_ids = set(splits["base"]) | set(splits["test"])
    eval_ids |= set(json.load(open(os.path.join(repo_root(), "data", "validation", "tau2_tasks.json")))["task_ids"])

    out = []
    with open(a.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            md = r.get("metadata") or {}
            d = sub_domain(md)
            if d == "unknown" or (d == "telecom" and str(md.get("task_id")) in eval_ids):
                continue
            row = sft_row(r, md, d, tok, tools, a.max_len) if a.cmd == "sft" else \
                pool_row(i, r, md, d, tok, tools, a.max_prompt_tokens)
            if row is not None:
                out.append(row)
    random.Random(a.seed).shuffle(out)
    print("wrote %s (%d rows)" % (a.out, write_jsonl(a.out, out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
