"""Answer extraction and grading for multiple-choice (MedQA, CaseHOLD) and numeric (FinQA) answers.

Numeric answers are compared with the DocMath-Eval rule (0.15% relative tolerance, percent / scale rescue).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> str:
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE)[-1]
    return text


_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANS_LINE = re.compile(r"(?:answer)\s*(?:is\s*)?(?:[:\-]\s*)?\(?([A-Ja-j])\)?(?![a-zA-Z])",
                       re.IGNORECASE)
_LONE_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def extract_letter(response: str, n_options: int = 10) -> Optional[str]:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    text = strip_thinking(response).strip()
    for m in reversed(_BOXED.findall(text)):
        cand = m.strip().strip("()").upper()
        if cand in valid:
            return cand
    for m in reversed(list(_ANS_LINE.finditer(text))):
        cand = m.group(1).upper()
        if cand in valid:
            return cand
    tail = text[-200:]
    cands = [c.upper() for c in _LONE_LETTER.findall(tail) if c.upper() in valid]
    if cands:
        return cands[-1]
    return None


_YNM = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def extract_yesno_maybe(response: str, allow_maybe: bool = True) -> Optional[str]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        w = m.strip().lower()
        if w in ("yes", "no") or (allow_maybe and w == "maybe"):
            return w
    hits = [w.lower() for w in _YNM.findall(text)]
    if not allow_maybe:
        hits = [w for w in hits if w != "maybe"]
    return hits[-1] if hits else None


_NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def _clean_number(tok: str) -> Optional[float]:
    tok = tok.strip().replace("$", "").replace(",", "")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    pct = tok.endswith("%")
    tok = tok.rstrip("%")
    try:
        v = float(tok)
    except ValueError:
        return None
    if neg:
        v = -v
    return v


def extract_number(response: str) -> Optional[float]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        for tok in reversed(_NUM.findall(m)):
            v = _clean_number(tok)
            if v is not None:
                return v
    ans = re.split(r"(?:answer|answer is|final answer)\s*[:\-]?", text,
                   flags=re.IGNORECASE)
    for seg in reversed(ans[1:] or []):
        for tok in _NUM.findall(seg[:120]):
            v = _clean_number(tok)
            if v is not None:
                return v
    toks = _NUM.findall(text[-300:])
    for tok in reversed(toks):
        v = _clean_number(tok)
        if v is not None:
            return v
    return None


def extract_text_answer(response: str) -> str:
    text = strip_thinking(response).strip()
    m = re.split(r"(?:answer|final answer)\s*[:\-]\s*", text, flags=re.IGNORECASE)
    if len(m) > 1:
        return m[-1].strip().split("\n")[0].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def grade_mcqa(response: str, gold_letter: str, n_options: int = 4) -> Dict[str, Any]:
    pred = extract_letter(response, n_options=n_options)
    return {"pred": pred, "gold": gold_letter.upper(),
            "correct": pred is not None and pred == gold_letter.upper()}

def _dm_within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def _dm_round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def _dm_compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True

    if v2 <= v1 / 50 and _dm_within_eps(pred=v2 * 100, gt=v1):
        return True
    elif v2 <= v1 / 500 and _dm_within_eps(pred=v2 * 1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and _dm_within_eps(pred=v2 * 100000, gt=v1):
        return True

    if _dm_round_up_to_decimal(v1, 3) == _dm_round_up_to_decimal(v2, 3):
        return True

    return _dm_within_eps(pred=p, gt=gt)


def fin_numbers_equal(pred: float, gold: float) -> bool:
    return bool(_dm_compare_two_numbers(pred, gold))


def grade_fin_numeric(response: str, gold: Any) -> Dict[str, Any]:
    gold_s = str(gold).strip().lower()
    if gold_s in ("yes", "no"):
        pred = extract_yesno_maybe(response, allow_maybe=False)
        return {"pred": pred, "gold": gold_s, "correct": pred == gold_s}
    gv = _clean_number(gold_s)
    if gv is None:
        pred_t = extract_text_answer(response).lower()
        return {"pred": pred_t, "gold": gold_s, "correct": pred_t == gold_s}
    pv = extract_number(response)
    ok = pv is not None and fin_numbers_equal(pv, gv)
    return {"pred": pv, "gold": gv, "correct": bool(ok)}
