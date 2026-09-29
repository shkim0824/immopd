"""NeMo-RL data processor for instruction-following prompts."""
from __future__ import annotations

from typing import Any

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec


def _build(datum_dict: dict[str, Any], task_data_spec: TaskDataSpec, tokenizer,
           max_seq_length: int, idx: int, extra_env_info: dict[str, Any]) -> DatumSpec:
    problem = datum_dict["messages"][0]["content"]
    message_list = []
    if task_data_spec.system_prompt:
        message_list.append({"role": "system", "content": task_data_spec.system_prompt})
    content = task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    message_list.append({"role": "user", "content": content})
    message: str = tokenizer.apply_chat_template(
        message_list, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
    token_ids = tokenizer(message, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    message_log: LLMMessageLogType = [{"role": "user", "content": message, "token_ids": token_ids}]
    length = len(token_ids)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        for m in message_log:
            m["token_ids"] = m["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0
    return {"message_log": message_log, "length": length, "extra_env_info": extra_env_info,
            "loss_multiplier": loss_multiplier, "idx": idx, "task_name": datum_dict["task_name"]}


def if_processor(datum_dict: dict[str, Any], task_data_spec: TaskDataSpec, tokenizer,
                      max_seq_length: int, idx: int) -> DatumSpec:
    return _build(datum_dict, task_data_spec, tokenizer, max_seq_length, idx,
                  {"instruction_id_list": datum_dict["instruction_id_list"],
                   "kwargs": datum_dict["kwargs"],
                   "prompt": datum_dict["messages"][0]["content"]})

