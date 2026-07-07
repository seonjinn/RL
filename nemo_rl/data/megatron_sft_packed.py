# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron-LM SFT packed JSONL preprocessing helpers."""

from typing import Any

import torch

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec

IGNORE_INDEX = -100
IDENTITY_TEMPLATE = (
    """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
)
NEMOTRON_H_ALIGNED_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}"""
NEMOTRON_NANO_V2_TEMPLATE = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"""


def split_megatron_sft_conversations(
    merged_messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    conversations: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in merged_messages:
        if message["role"] == "system":
            if current:
                conversations.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        conversations.append(current)
    return conversations


def _megatron_chat_template(prompt_format: str | None) -> str | None:
    if prompt_format == "identity":
        return IDENTITY_TEMPLATE
    if prompt_format == "nemotron-nano-v2":
        return NEMOTRON_NANO_V2_TEMPLATE
    if prompt_format == "nemotron-h-aligned":
        return NEMOTRON_H_ALIGNED_TEMPLATE
    return None


def _apply_chat_template_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    prompt_format: str | None,
    add_generation_prompt: bool = False,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    chat_template = _megatron_chat_template(prompt_format)
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(token_ids, "get") and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def _tokenize_megatron_sft_conversation(
    conversation: list[dict[str, Any]],
    tokenizer,
    prompt_format: str | None,
    assistant_prefix_len: int,
    add_generation_prompt: bool,
) -> tuple[list[int], list[int]]:
    tokens = _apply_chat_template_ids(
        tokenizer,
        conversation,
        prompt_format,
        add_generation_prompt=add_generation_prompt,
    )
    targets = list(tokens)

    if prompt_format == "identity":
        return tokens, targets

    idx = 0
    for turn_idx, turn in enumerate(conversation):
        role = turn["role"].lower()
        if role == "assistant":
            assert conversation[turn_idx - 1]["role"].lower() == "user"
        turn_tokens = _apply_chat_template_ids(tokenizer, [turn], prompt_format)
        turn_len = len(turn_tokens)
        if role in ("system", "user"):
            targets[idx : idx + turn_len] = [IGNORE_INDEX] * turn_len
        elif role == "assistant":
            prefix_len = min(assistant_prefix_len, turn_len)
            targets[idx : idx + prefix_len] = [IGNORE_INDEX] * prefix_len
        else:
            raise ValueError(f"Unsupported role in Megatron SFT record: {role}")

        assert tokens[idx : idx + turn_len] == turn_tokens, (
            "Turn tokenization did not match full-conversation tokenization. "
            "Set data.default.megatron_sft_prompt_format to the Megatron prompt "
            "format used to create the packed input."
        )
        idx += turn_len

    assert idx == len(tokens), "Conversation target mask length mismatch"
    return tokens, targets


def _resolve_pad_token_id(tokenizer, pad_token: str | None) -> int:
    if pad_token is not None:
        return int(tokenizer.convert_tokens_to_ids(pad_token))
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def megatron_sft_packed_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int | None,
    idx: int,
    prompt_format: str | None = None,
    pad_token: str | None = None,
    assistant_prefix_len: int = 0,
    context_parallel_size: int = 1,
    add_generation_prompt: bool = False,
    **_unused_kwargs,
) -> DatumSpec:
    """Build Megatron-LM SFTDataset-style tensors for one packed JSONL record."""
    del task_data_spec
    if max_seq_length is None:
        raise ValueError("max_seq_length is required for Megatron SFT packed data")
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be >= 1")
    pack_length = max_seq_length
    pad = _resolve_pad_token_id(tokenizer, pad_token)
    conversations = split_megatron_sft_conversations(datum_dict["packed_messages"])

    pack_tokens: list[int] = []
    pack_targets: list[int] = []
    pack_positions: list[int] = []
    cu_seqlens = [0]
    processed_token_count = 0

    def _extend_with_padding(pad_len: int) -> None:
        if pad_len <= 0:
            return
        start_position = pack_positions[-1] + 1 if pack_positions else 0
        pack_tokens.extend([pad] * pad_len)
        pack_targets.extend([pad] * pad_len)
        pack_positions.extend(range(start_position, start_position + pad_len))

    for conversation in conversations:
        tokens, targets = _tokenize_megatron_sft_conversation(
            conversation,
            tokenizer,
            prompt_format,
            assistant_prefix_len,
            add_generation_prompt,
        )

        remaining_input_capacity = max(pack_length - len(pack_tokens), 0)
        processed_token_count += min(len(tokens), remaining_input_capacity)
        pack_tokens.extend(tokens)
        pack_targets.extend(targets)
        pack_positions.extend(range(len(tokens)))

        if context_parallel_size > 1:
            pad_granularity = context_parallel_size * 2
            mod_token_count = len(pack_tokens) % pad_granularity
            if mod_token_count != 0:
                _extend_with_padding(pad_granularity - mod_token_count)

        cu_seqlens.append(len(pack_tokens))

        if len(pack_tokens) >= pack_length + 1:
            pack_tokens = pack_tokens[:pack_length]
            pack_targets = pack_targets[:pack_length]
            pack_tokens.append(pad)
            pack_targets.append(pad)
            pack_positions = pack_positions[: pack_length + 1]
            cu_seqlens[-1] = len(pack_tokens) - 1
            break

    if len(pack_tokens) < pack_length + 1:
        _extend_with_padding(pack_length + 1 - len(pack_tokens))
        cu_seqlens[-1] = len(pack_tokens) - 1

    if not (
        len(pack_tokens) == len(pack_targets) == len(pack_positions) == pack_length + 1
    ):
        raise ValueError(
            "Megatron SFT packed preprocessing produced inconsistent pack lengths: "
            f"tokens={len(pack_tokens)} targets={len(pack_targets)} "
            f"positions={len(pack_positions)} expected={pack_length + 1}"
        )

    input_ids_tensor = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
    target_ids_tensor = torch.tensor(pack_targets[1:], dtype=torch.int64)
    position_ids_tensor = torch.tensor(pack_positions[:-1], dtype=torch.int64)
    token_mask_tensor = torch.ones(pack_length, dtype=torch.float32)
    token_mask_tensor[target_ids_tensor == pad] = 0.0
    token_mask_tensor[target_ids_tensor == IGNORE_INDEX] = 0.0

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
    if context_parallel_size > 1:
        segment_lengths = cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]
        cp_granularity = 2 * context_parallel_size
        if bool((segment_lengths % cp_granularity != 0).any().item()):
            raise ValueError(
                "Megatron SFT packed segment lengths must be divisible by "
                f"{cp_granularity} when context_parallel_size={context_parallel_size}"
            )
    max_seqlen = int((cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item())

    return {
        "message_log": [],
        "input_ids": input_ids_tensor,
        "target_ids": target_ids_tensor,
        "token_mask": token_mask_tensor,
        "position_ids": position_ids_tensor,
        "packed_cu_seqlens": cu_seqlens_tensor,
        "packed_max_seqlen": max_seqlen,
        "packed_context_parallel_size": int(context_parallel_size),
        "processed_token_count": processed_token_count,
        "length": pack_length,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "megatron_sft_packed"),
    }
