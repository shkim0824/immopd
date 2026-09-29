from __future__ import annotations

from typing import Any, Optional

import ray


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


@ray.remote  # pragma: no cover
class IFVerifyWorker:
    """Reward 1 if the response satisfies all instruction constraints, else 0."""

    def __init__(self) -> None:
        from verifiable_instructions import instructions_registry
        self._registry = instructions_registry.INSTRUCTION_DICT

    def verify(self, items: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for it in items:
            response = _strip_thinking(str(it.get("response") or "")).strip()
            ids = it.get("instruction_id_list") or []
            kwargs = it.get("kwargs") or [None] * len(ids)
            try:
                ok = True
                for iid, kw in zip(ids, kwargs):
                    cls = self._registry.get(iid)
                    if cls is None:
                        ok = False
                        break
                    inst = cls(iid)
                    inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
                    args = inst.get_instruction_args()
                    if args and "prompt" in args:
                        inst.build_description(prompt=it.get("prompt") or "")
                    if not response or not inst.check_following(response):
                        ok = False
                        break
                out.append(1.0 if ok else 0.0)
            except Exception:
                out.append(0.0)
        return out
