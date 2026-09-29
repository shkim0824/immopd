"""Instruction-following grading with the official checkers (``third_party/``).

  ifeval   google-research instruction_following_eval    IFEval, prompt-level strict accuracy
  ifbench  allenai/IFBench                               IFBench, prompt-level loose accuracy
  ifevalg  allenai/open-instruct IFEvalG                 validation prompts of the IF domain
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REGISTRIES: Dict[str, Dict[str, Any]] = {}


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>", 1)[1]
    return out.strip()


def registry(name: str) -> Dict[str, Any]:
    if name in _REGISTRIES:
        return _REGISTRIES[name]
    if name == "ifeval":
        from third_party.ifeval_google import instructions_registry as r
    elif name == "ifbench":
        from third_party.ifbench import instructions_registry as r
    elif name == "ifevalg":
        from third_party.ifevalg import instructions_registry as r
    else:
        raise KeyError(name)
    _REGISTRIES[name] = r.INSTRUCTION_DICT
    return _REGISTRIES[name]


def _clean_kwargs(kw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (kw or {}).items() if v is not None}


def _loose_variants(response: str) -> List[str]:
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    revised = response.replace("*", "")
    return [
        response, revised, remove_first, remove_last, remove_both,
        remove_first.replace("*", ""), remove_last.replace("*", ""), remove_both.replace("*", ""),
    ]


def _check_list(reg: Dict[str, Any], prompt: str, ids: Sequence[str], kwargs_list: Sequence[Optional[Dict[str, Any]]],
                response: str, loose: bool) -> List[bool]:
    variants = _loose_variants(response) if loose else [response]
    out: List[bool] = []
    for index, iid in enumerate(ids):
        cls = reg[iid]
        inst = cls(iid)
        kw = _clean_kwargs(kwargs_list[index] if index < len(kwargs_list) else None)
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        ok = False
        for v in variants:
            if v.strip() and inst.check_following(v):
                ok = True
                break
        out.append(ok)
    return out


def evaluate_example(bench: str, prompt: str, instruction_id_list: Sequence[str],
                     kwargs_list: Sequence[Optional[Dict[str, Any]]], response: str,
                     strip_think: bool = True) -> Dict[str, Any]:
    reg = registry(bench)
    resp = strip_thinking(response) if strip_think else (response or "")
    strict = _check_list(reg, prompt, instruction_id_list, kwargs_list, resp, loose=False)
    loose = _check_list(reg, prompt, instruction_id_list, kwargs_list, resp, loose=True)
    return {"strict": strict, "loose": loose,
            "prompt_strict": all(strict), "prompt_loose": all(loose)}


def score_benchmark(bench: str, rows: Sequence[Dict[str, Any]], responses: Sequence[str]) -> Dict[str, Any]:
    assert len(rows) == len(responses)
    ps = pl = 0
    is_ = il = it = 0
    per = []
    for row, resp in zip(rows, responses):
        e = evaluate_example(bench, row["prompt"], row["instruction_id_list"], row["kwargs"], resp)
        ps += e["prompt_strict"]
        pl += e["prompt_loose"]
        is_ += sum(e["strict"])
        il += sum(e["loose"])
        it += len(e["strict"])
        per.append(e)
    n = max(len(rows), 1)
    return {
        "prompt_level_strict_acc": 100.0 * ps / n,
        "prompt_level_loose_acc": 100.0 * pl / n,
        "inst_level_strict_acc": 100.0 * is_ / max(it, 1),
        "inst_level_loose_acc": 100.0 * il / max(it, 1),
        "n": len(rows), "n_instructions": it, "per_example": per,
    }
