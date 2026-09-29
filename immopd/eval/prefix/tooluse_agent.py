"""Student-prefix teacher-continuation agent for tau2-bench (Tool Use).

In every assistant turn the student generates at most ``PFX_N`` tokens; if the turn is not finished, the
teacher continues from the student's tokens. ``PFX_N = round(alpha * mean tokens per assistant turn)`` of
the student (``immopd.eval.prefix.token_stats``).

Serve the student as "student" and the teacher as "teacher" (``PFX_TEACHER_BASE``), then run
``immopd.eval.tau2_run --agent llm_agent_prefix``.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
import urllib.error
import urllib.request

from loguru import logger

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.registry import registry
from tau2.utils.llm_utils import to_litellm_messages, validate_message_history

TOOL_CALL_RX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
HTTP_TIMEOUT = float(os.environ.get("PFX_HTTP_TIMEOUT", "3600"))
CUT_N = int(os.environ.get("PFX_N", "0"))
TEACHER_BASE = os.environ.get("PFX_TEACHER_BASE", "http://127.0.0.1:8124/v1")


class ContextWindowExceeded(RuntimeError):
    pass


def _root(base: str) -> str:
    base = base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _post(url: str, payload: dict, retries: int = 3) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:600]
            low = body.lower()
            if e.code == 400 and ("context length" in low or "maximum context" in low or "too long" in low or "max_model_len" in low):
                raise ContextWindowExceeded("%s -> %d: %s" % (url, e.code, body))
            if e.code >= 500 and attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError("%s -> %d: %s" % (url, e.code, body))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError("%s -> %r" % (url, e))


def split_think(text: str):
    """Same split as vLLM's qwen3 reasoning parser."""
    if "<think>" not in text or "</think>" not in text:
        return None, text
    parts = text.partition("<think>")
    rest = parts[2] if parts[1] else parts[0]
    if "</think>" not in rest:
        return None, rest
    reasoning, _, content = rest.partition("</think>")
    return reasoning, (content or None)


def parse_tool_calls(content):
    """Same parsing as vLLM's hermes tool parser."""
    if content is None or "<tool_call>" not in content:
        return content, []
    try:
        raw = [json.loads(m[0] if m[0] else m[1]) for m in TOOL_CALL_RX.findall(content)]
        calls = [ToolCall(id="call_%s" % uuid.uuid4().hex[:12], name=fc["name"], arguments=json.loads(json.dumps(fc["arguments"], ensure_ascii=False)))
                 for fc in raw]
    except Exception as e:                                           # noqa: BLE001 -- mirror vLLM: log + return content
        logger.warning("tool call parse failed (%r); returning raw content" % (e,))
        return content, []
    head = content[: content.find("<tool_call>")]
    return (head if head else None), calls


class PrefixAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        a = dict(llm_args or {})
        self.student_root = _root(a.get("api_base", "http://127.0.0.1:8123/v1"))
        self.teacher_root = _root(TEACHER_BASE)
        self.temperature = float(a.get("temperature", 1.0))
        self.top_p = float(a.get("top_p", 1.0))
        self.top_k = int((a.get("extra_body") or {}).get("top_k", -1))
        self.tools_schema = [t.openai_schema for t in self.tools] if self.tools else None

    def _messages(self, messages):
        out = []
        for m in to_litellm_messages(messages):
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"id": tc["id"], "type": "function", "function": tc["function"]} for tc in m["tool_calls"]]
            out.append(m)
        return out

    def _render(self, messages):
        r = _post(self.student_root + "/tokenize", {"model": "student", "messages": self._messages(messages), "tools": self.tools_schema,
                                                     "add_generation_prompt": True})
        return r["tokens"], int(r.get("max_model_len") or 0)

    def _complete(self, root, model, prompt_ids, max_tokens):
        p = {"model": model, "prompt": prompt_ids, "temperature": self.temperature, "top_p": self.top_p, "logprobs": 1, "return_tokens_as_token_ids": True,
             "max_tokens": max(1, int(max_tokens))}
        if self.top_k > 0:
            p["top_k"] = self.top_k
        r = _post(root + "/v1/completions", p)
        ch = r["choices"][0]
        toks = (ch.get("logprobs") or {}).get("tokens") or []
        ids = [int(t.split(":", 1)[1]) for t in toks if isinstance(t, str) and t.startswith("token_id:")]
        n = (r.get("usage") or {}).get("completion_tokens")
        return ch.get("text") or "", ids, ch.get("finish_reason"), (int(n) if n is not None else len(ids))

    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        validate_message_history(messages)
        turn_idx = sum(1 for m in state.messages if isinstance(m, AssistantMessage))
        t0 = time.perf_counter()
        prompt_ids, max_len = self._render(messages)
        max_len = max_len or 32768
        room = max_len - len(prompt_ids)
        if room <= 0:
            raise ContextWindowExceeded("prompt %d tokens >= max_model_len %d" % (len(prompt_ids), max_len))
        s_text = t_text = ""
        s_tok = t_tok = 0
        finished_by, finish, s_ids = "student", None, []
        budget = min(CUT_N, room)
        if budget > 0:
            s_text, s_ids, finish, s_tok = self._complete(self.student_root, "student", prompt_ids, budget)
        if budget <= 0 or finish == "length":
            left = room - s_tok
            if left <= 0:
                raise ContextWindowExceeded("no room after the student prefix")
            if s_ids and len(s_ids) == s_tok:
                cont = prompt_ids + s_ids
            else:
                cont = _post(self.student_root + "/detokenize", {"model": "student", "tokens": prompt_ids})["prompt"] + s_text
            t_text, _, finish, t_tok = self._complete(self.teacher_root, "teacher", cont, left)
            finished_by = "teacher"
        raw = s_text + t_text
        if finish == "length":
            logger.warning("Output might be incomplete due to token limit!")
        reasoning, content = split_think(raw)
        content, tool_calls = parse_tool_calls(content)
        usage = {"completion_tokens": s_tok + t_tok, "prompt_tokens": len(prompt_ids), "student_tokens": s_tok,
                 "teacher_tokens": t_tok, "finished_by": finished_by, "turn_idx": turn_idx, "cut": CUT_N,
                 "finish_reason": finish}
        return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls or None, cost=0.0, usage=usage,
                                raw_data={"student_text": s_text, "teacher_text": t_text, "reasoning": reasoning},
                                generation_time_seconds=time.perf_counter() - t0)


def create_prefix_agent(tools, domain_policy, **kwargs):
    return PrefixAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))


registry.register_agent_factory(create_prefix_agent, "llm_agent_prefix")
