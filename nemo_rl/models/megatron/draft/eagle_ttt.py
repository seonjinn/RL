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
from typing import Any

import torch
from torch import Tensor


def _validate_pass_bounds(*, pass_count: int, max_passes: int) -> None:
    if max_passes < 1:
        raise ValueError(f"max_passes must be positive, got {max_passes}")
    if pass_count < 1:
        raise ValueError(f"pass_count must be positive, got {pass_count}")
    if pass_count > max_passes:
        raise ValueError(
            f"pass_count {pass_count} exceeds the configured maximum {max_passes}"
        )


@dataclass(frozen=True, slots=True)
class EagleTTTAttentionPlan:
    """Scalar-only metadata for one inference-aligned EAGLE training pass."""

    pass_index: int
    pass_count: int
    max_passes: int
    sequence_length: int

    def __post_init__(self) -> None:
        _validate_pass_bounds(
            pass_count=self.pass_count,
            max_passes=self.max_passes,
        )
        if not 0 <= self.pass_index < self.pass_count:
            raise ValueError(
                f"pass_index must be in [0, {self.pass_count}), got {self.pass_index}"
            )
        if self.sequence_length < 1:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )

    @property
    def teacher_offset(self) -> int:
        """Return the target-logit offset predicted by this pass."""
        return self.pass_index + 1

    def rope_positions(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize int64 positions without narrowing or position reuse."""
        return torch.arange(
            self.pass_index,
            self.sequence_length + self.pass_index,
            dtype=torch.int64,
            device=device,
        )

    def dense_visibility_mask(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize a small dense oracle mask; never retain it in pass state."""
        query_positions = torch.arange(self.sequence_length, device=device)
        trunk_positions = torch.arange(self.sequence_length, device=device)
        trunk_visible = trunk_positions[None, :] <= query_positions[:, None]
        if self.pass_index == 0:
            return trunk_visible
        branch_visible = torch.eye(
            self.sequence_length,
            dtype=torch.bool,
            device=device,
        ).repeat(1, self.pass_index)
        return torch.cat((trunk_visible, branch_visible), dim=1)


@dataclass(frozen=True, slots=True)
class EagleTTTStoragePlan:
    """Pre-allocation bound for append-only trunk and prior-branch K/V state."""

    batch_size: int
    kv_heads: int
    sequence_length: int
    head_dim: int
    dtype: torch.dtype
    pass_count: int
    max_passes: int
    activation_budget_bytes: int

    def __post_init__(self) -> None:
        _validate_pass_bounds(
            pass_count=self.pass_count,
            max_passes=self.max_passes,
        )
        dimensions = {
            "batch_size": self.batch_size,
            "kv_heads": self.kv_heads,
            "sequence_length": self.sequence_length,
            "head_dim": self.head_dim,
        }
        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.activation_budget_bytes < 1:
            raise ValueError(
                "activation_budget_bytes must be positive, "
                f"got {self.activation_budget_bytes}"
            )
        if self.retained_bytes > self.activation_budget_bytes:
            raise ValueError(
                "packed EAGLE TTT state exceeds the activation budget: "
                f"requires {self.retained_bytes} bytes for {self.pass_count} passes, "
                f"budget is {self.activation_budget_bytes} bytes"
            )

    @property
    def retained_bytes(self) -> int:
        """Return bytes for one trunk K/V pair plus one pair per prior pass."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        elements_per_tensor = (
            self.batch_size * self.kv_heads * self.sequence_length * self.head_dim
        )
        return self.pass_count * 2 * elements_per_tensor * element_size


@dataclass(frozen=True, slots=True)
class EagleTTTState:
    """Immutable references to one causal trunk and append-only prior branch K/V."""

    pass_count: int
    max_passes: int
    activation_budget_bytes: int
    trunk_key: Tensor
    trunk_value: Tensor
    branch_keys: tuple[Tensor, ...] = ()
    branch_values: tuple[Tensor, ...] = ()

    @classmethod
    def from_trunk(
        cls,
        *,
        trunk_key: Tensor,
        trunk_value: Tensor,
        pass_count: int,
        max_passes: int,
        activation_budget_bytes: int,
    ) -> EagleTTTState:
        """Validate the complete bound before retaining any supplied tensor."""
        _validate_kv_pair(trunk_key, trunk_value, name="trunk")
        EagleTTTStoragePlan(
            batch_size=trunk_key.shape[0],
            kv_heads=trunk_key.shape[1],
            sequence_length=trunk_key.shape[2],
            head_dim=trunk_key.shape[3],
            dtype=trunk_key.dtype,
            pass_count=pass_count,
            max_passes=max_passes,
            activation_budget_bytes=activation_budget_bytes,
        )
        return cls(
            pass_count=pass_count,
            max_passes=max_passes,
            activation_budget_bytes=activation_budget_bytes,
            trunk_key=trunk_key,
            trunk_value=trunk_value,
        )

    def append_branch(
        self,
        *,
        branch_key: Tensor,
        branch_value: Tensor,
    ) -> EagleTTTState:
        """Return a state retaining one additional same-anchor branch K/V pair."""
        if len(self.branch_keys) >= self.pass_count - 1:
            raise ValueError(
                f"pass_count {self.pass_count} retains at most "
                f"{self.pass_count - 1} prior branches"
            )
        _validate_kv_pair(branch_key, branch_value, name="branch")
        if branch_key.shape != self.trunk_key.shape:
            raise ValueError("branch K/V shape must match the causal trunk")
        if (
            branch_key.device != self.trunk_key.device
            or branch_key.dtype != self.trunk_key.dtype
        ):
            raise ValueError("branch K/V must share trunk device and dtype")
        return replace(
            self,
            branch_keys=(*self.branch_keys, branch_key),
            branch_values=(*self.branch_values, branch_value),
        )

    @property
    def retained_tensor_bytes(self) -> int:
        """Return the bytes represented by retained K/V tensor references."""
        tensors = (
            self.trunk_key,
            self.trunk_value,
            *self.branch_keys,
            *self.branch_values,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _validate_kv_pair(key: Tensor, value: Tensor, *, name: str) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{name} key and value must be rank-4 tensors")
    if key.shape != value.shape:
        raise ValueError(f"{name} key and value shapes must match")
    if key.device != value.device:
        raise ValueError(f"{name} key and value must share a device")
    if key.dtype != value.dtype:
        raise ValueError(f"{name} key and value must share a dtype")


def _expand_gqa(tensor: Tensor, *, query_heads: int) -> Tensor:
    if query_heads % tensor.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    return tensor.repeat_interleave(query_heads // tensor.shape[1], dim=1)


def _dense_causal_trunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    compute_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    expanded_key = _expand_gqa(key, query_heads=query.shape[1])
    expanded_value = _expand_gqa(value, query_heads=query.shape[1])
    scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        query.to(compute_dtype),
        expanded_key.to(compute_dtype),
    ).mul_(scale)
    sequence_length = query.shape[2]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril_()
    scores.masked_fill_(~causal, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(scores, dim=-1),
        expanded_value.to(compute_dtype),
    )
    return output, lse


def _causal_mask(
    _batch: Tensor,
    _head: Tensor,
    query_index: Tensor,
    key_index: Tensor,
) -> Tensor:
    return key_index <= query_index


@lru_cache(maxsize=32)
def _causal_block_mask(sequence_length: int, device: torch.device) -> Any:
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
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    from torch.nn.attention.flex_attention import (
        AuxRequest,  # pyrefly: ignore[missing-module-attribute]
        flex_attention,
    )

    flex_attention_call: Any = flex_attention
    output, auxiliary = flex_attention_call(
        query,
        key,
        value,
        block_mask=_causal_block_mask(query.shape[2], query.device),
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options={
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": True,
        },
    )
    lse = auxiliary.lse
    if not isinstance(lse, Tensor):
        raise RuntimeError("FlexAttention did not return the requested LSE")
    return output, lse


def _merge_branch(
    *,
    query: Tensor,
    output: Tensor,
    lse: Tensor,
    branch_key: Tensor,
    branch_value: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    batch, query_heads, sequence_length, head_dim = query.shape
    kv_heads = branch_key.shape[1]
    groups = query_heads // kv_heads
    compute_dtype = lse.dtype
    grouped_query = query.to(compute_dtype).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        head_dim,
    )
    branch_scores = torch.einsum(
        "bhgsd,bhsd->bhgs",
        grouped_query,
        branch_key.to(compute_dtype),
    ).reshape(batch, query_heads, sequence_length)
    branch_scores.mul_(scale)
    merged_lse = torch.logaddexp(lse, branch_scores)
    grouped_output = output.to(compute_dtype).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        head_dim,
    )
    trunk_weight = torch.exp(lse - merged_lse).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    branch_weight = torch.exp(branch_scores - merged_lse).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    merged_output = grouped_output * trunk_weight
    merged_output.add_(branch_value.to(compute_dtype).unsqueeze(2) * branch_weight)
    return (
        merged_output.reshape(batch, query_heads, sequence_length, head_dim),
        merged_lse,
    )


def eagle_ttt_attention(
    *,
    query: Tensor,
    state: EagleTTTState,
    plan: EagleTTTAttentionPlan,
    scale: float | None = None,
) -> Tensor:
    """Attend to one causal trunk and pointwise prior same-anchor branches."""
    if state.pass_count != plan.pass_count or state.max_passes != plan.max_passes:
        raise ValueError("state and attention plan pass bounds must match")
    if len(state.branch_keys) != plan.pass_index:
        raise ValueError(
            f"pass {plan.pass_index} requires {plan.pass_index} prior branches, "
            f"got {len(state.branch_keys)}"
        )
    if query.ndim != 4:
        raise ValueError("query must be a rank-4 tensor")
    if query.shape[0] != state.trunk_key.shape[0]:
        raise ValueError("query and trunk batch sizes must match")
    if query.shape[2:] != state.trunk_key.shape[2:]:
        raise ValueError("query and trunk sequence/head dimensions must match")
    if query.shape[2] != plan.sequence_length:
        raise ValueError("query sequence length must match the attention plan")
    if query.device != state.trunk_key.device or query.dtype != state.trunk_key.dtype:
        raise ValueError("query and retained state must share device and dtype")
    _expand_gqa(state.trunk_key, query_heads=query.shape[1])

    attention_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if query.is_cuda and query.dtype != torch.float64:
        output, lse = _flex_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
        )
    else:
        output, lse = _dense_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
        )
    for branch_key, branch_value in zip(
        state.branch_keys,
        state.branch_values,
        strict=True,
    ):
        output, lse = _merge_branch(
            query=query,
            output=output,
            lse=lse,
            branch_key=branch_key,
            branch_value=branch_value,
            scale=attention_scale,
        )
    return output.to(dtype=query.dtype)
