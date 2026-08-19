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

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import torch
from torch import Tensor

_MAX_TTT_STEPS = 4


@dataclass(frozen=True, slots=True)
class EagleTTTAttentionPlan:
    pass_index: int
    sequence_length: int
    max_steps: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= _MAX_TTT_STEPS:
            raise ValueError(
                f"max_steps must be between 1 and {_MAX_TTT_STEPS}, got {self.max_steps}"
            )
        if not 0 <= self.pass_index < self.max_steps:
            raise ValueError(
                f"pass_index must be in [0, {self.max_steps}), got {self.pass_index}"
            )
        if self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )

    @property
    def teacher_offset(self) -> int:
        return self.pass_index + 1

    def rope_positions(self, *, device: torch.device | None = None) -> Tensor:
        return torch.arange(
            self.pass_index,
            self.sequence_length + self.pass_index,
            device=device,
        )

    def visibility_mask(self, *, device: torch.device | None = None) -> Tensor:
        query_positions = torch.arange(self.sequence_length, device=device)
        trunk_positions = torch.arange(self.sequence_length, device=device)
        trunk_visible = trunk_positions[None, :] <= query_positions[:, None]
        if self.pass_index == 0:
            return trunk_visible
        branch_visible = torch.eye(
            self.sequence_length, dtype=torch.bool, device=device
        ).repeat(1, self.pass_index)
        return torch.cat((trunk_visible, branch_visible), dim=1)


@dataclass(frozen=True, slots=True)
class EagleTTTKVCache:
    max_steps: int
    trunk_key: Tensor | None = None
    trunk_value: Tensor | None = None
    branches_key: tuple[Tensor, ...] = ()
    branches_value: tuple[Tensor, ...] = ()

    @classmethod
    def empty(cls, *, max_steps: int) -> EagleTTTKVCache:
        if not 1 <= max_steps <= _MAX_TTT_STEPS:
            raise ValueError(
                f"max_steps must be between 1 and {_MAX_TTT_STEPS}, got {max_steps}"
            )
        return cls(max_steps=max_steps)

    def with_trunk(self, key: Tensor, value: Tensor) -> EagleTTTKVCache:
        if self.trunk_key is not None or self.trunk_value is not None:
            raise ValueError("the causal trunk is already populated")
        _validate_kv_pair(key, value, name="trunk")
        return replace(self, trunk_key=key, trunk_value=value)

    def append_branch(self, key: Tensor, value: Tensor) -> EagleTTTKVCache:
        if self.trunk_key is None or self.trunk_value is None:
            raise ValueError("populate the causal trunk before appending a branch")
        if len(self.branches_key) >= self.max_steps - 1:
            raise ValueError(
                f"at most {self.max_steps - 1} branch entries are supported"
            )
        _validate_kv_pair(key, value, name="branch")
        if key.shape != self.trunk_key.shape:
            raise ValueError(
                "branch key/value shapes must match the causal trunk shape"
            )
        return replace(
            self,
            branches_key=(*self.branches_key, key),
            branches_value=(*self.branches_value, value),
        )


def _validate_kv_pair(key: Tensor, value: Tensor, *, name: str) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{name} key and value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError(f"{name} key and value shapes must match")
    if key.device != value.device:
        raise ValueError(f"{name} key and value must be on the same device")
    if key.dtype != value.dtype:
        raise ValueError(f"{name} key and value must have the same dtype")


def _expand_gqa(tensor: Tensor, *, query_heads: int) -> Tensor:
    repeats = query_heads // tensor.shape[1]
    return tensor.repeat_interleave(repeats, dim=1)


def _dense_causal_trunk(
    query: Tensor, key: Tensor, value: Tensor, *, scale: float
) -> tuple[Tensor, Tensor]:
    compute_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    expanded_key = _expand_gqa(key, query_heads=query.shape[1])
    expanded_value = _expand_gqa(value, query_heads=query.shape[1])
    scores = (
        torch.einsum(
            "bhqd,bhkd->bhqk", query.to(compute_dtype), expanded_key.to(compute_dtype)
        )
        * scale
    )
    sequence_length = query.shape[2]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril_()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum(
        "bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), expanded_value
    )
    return output, lse


def _causal_mask(
    _batch: Tensor, _head: Tensor, query_index: Tensor, key_index: Tensor
) -> Tensor:
    return key_index <= query_index


@lru_cache(maxsize=32)
def _causal_block_mask(sequence_length: int, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        _causal_mask,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
    )


def _flex_causal_trunk(
    query: Tensor, key: Tensor, value: Tensor, *, scale: float
) -> tuple[Tensor, Tensor]:
    from torch.nn.attention.flex_attention import (
        AuxRequest,
        FlexKernelOptions,
        flex_attention,
    )

    kernel_options: FlexKernelOptions = {
        "ROWS_GUARANTEED_SAFE": True,
        "BLOCKS_ARE_CONTIGUOUS": True,
    }
    result = flex_attention(
        query,
        key,
        value,
        block_mask=_causal_block_mask(query.shape[2], query.device),
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options=kernel_options,
    )
    output, auxiliary = result
    lse = auxiliary.lse
    if not isinstance(lse, Tensor):
        raise RuntimeError("FlexAttention did not return the requested LSE")
    return output, lse


def _merge_branch(
    query: Tensor,
    output: Tensor,
    lse: Tensor,
    branch_key: Tensor,
    branch_value: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    batch, query_heads, sequence_length, head_dim = query.shape
    kv_heads = branch_key.shape[1]
    groups = query_heads // kv_heads
    compute_dtype = lse.dtype
    grouped_query = query.to(compute_dtype).reshape(
        batch, kv_heads, groups, sequence_length, head_dim
    )
    branch_scores = torch.einsum(
        "bhgsd,bhsd->bhgs", grouped_query, branch_key.to(compute_dtype)
    ).reshape(batch, query_heads, sequence_length)
    branch_scores = branch_scores * scale
    merged_lse = torch.logaddexp(lse, branch_scores)
    grouped_output = output.to(compute_dtype).reshape(
        batch, kv_heads, groups, sequence_length, head_dim
    )
    trunk_weight = torch.exp(lse - merged_lse).reshape(
        batch, kv_heads, groups, sequence_length, 1
    )
    branch_weight = torch.exp(branch_scores - merged_lse).reshape(
        batch, kv_heads, groups, sequence_length, 1
    )
    merged_output = grouped_output * trunk_weight
    merged_output = (
        merged_output + branch_value.to(compute_dtype).unsqueeze(2) * branch_weight
    )
    return (
        merged_output.reshape(batch, query_heads, sequence_length, head_dim),
        merged_lse,
    )


def eagle_ttt_attention(
    *,
    query: Tensor,
    cache: EagleTTTKVCache,
    plan: EagleTTTAttentionPlan,
    scale: float | None = None,
) -> Tensor:
    if cache.max_steps != plan.max_steps:
        raise ValueError("cache and attention plan max_steps must match")
    if cache.trunk_key is None or cache.trunk_value is None:
        raise ValueError("the causal trunk is not populated")
    if len(cache.branches_key) != plan.pass_index:
        names = ("zero", "one", "two", "three")
        raise ValueError(
            f"pass {plan.pass_index} requires {names[plan.pass_index]} branches"
        )
    if query.ndim != 4:
        raise ValueError("query must be a rank-4 tensor")
    if query.shape[0] != cache.trunk_key.shape[0]:
        raise ValueError("query and cache batch sizes must match")
    if query.shape[2] != plan.sequence_length:
        raise ValueError("query sequence length must match the attention plan")
    if query.shape[2:] != cache.trunk_key.shape[2:]:
        raise ValueError("query and cache sequence/head dimensions must match")
    if query.shape[1] % cache.trunk_key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if query.device != cache.trunk_key.device or query.dtype != cache.trunk_key.dtype:
        raise ValueError("query and cache must share device and dtype")

    attention_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if query.is_cuda and query.dtype != torch.float64:
        output, lse = _flex_causal_trunk(
            query, cache.trunk_key, cache.trunk_value, scale=attention_scale
        )
    else:
        output, lse = _dense_causal_trunk(
            query, cache.trunk_key, cache.trunk_value, scale=attention_scale
        )
    for branch_key, branch_value in zip(
        cache.branches_key, cache.branches_value, strict=True
    ):
        output, lse = _merge_branch(
            query,
            output,
            lse,
            branch_key,
            branch_value,
            scale=attention_scale,
        )
    return output.to(query.dtype)
