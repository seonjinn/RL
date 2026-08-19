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

_FLEX_BLOCK_SIZE = 128


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

    @property
    def student_slice(self) -> slice:
        """Return ModelOpt's student rows for this pass."""
        return slice(self.pass_index, -1)

    @property
    def teacher_slice(self) -> slice:
        """Return the target rows paired with :attr:`student_slice`."""
        return slice(self.pass_index + 1, None)

    def rope_positions(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize the one base position vector shared by every pass."""
        return torch.arange(
            self.sequence_length,
            dtype=torch.int64,
            device=device,
        )

    def dense_visibility_mask(self, *, device: torch.device | None = None) -> Tensor:
        """Materialize a small dense oracle mask; never retain it in pass state."""
        query_positions = torch.arange(self.sequence_length, device=device)
        trunk_positions = torch.arange(self.sequence_length, device=device)
        trunk_visible = (
            trunk_positions[None, :] <= query_positions[:, None] - self.pass_index
        )
        if self.pass_index == 0:
            return trunk_visible
        branch_visible = tuple(
            trunk_positions[None, :]
            == query_positions[:, None] - (self.pass_index - branch_index - 1)
            for branch_index in range(self.pass_index)
        )
        return torch.cat((trunk_visible, *branch_visible), dim=1)


@dataclass(frozen=True, slots=True)
class EagleTTTSequenceLayout:
    """Compact padding and packed-document metadata for one token sequence."""

    valid_tokens: Tensor
    document_ids: Tensor

    def __post_init__(self) -> None:
        if self.valid_tokens.ndim != 2 or self.document_ids.ndim != 2:
            raise ValueError(
                "valid_tokens and document_ids must have [batch, sequence] shape"
            )
        if self.valid_tokens.shape != self.document_ids.shape:
            raise ValueError("valid_tokens and document_ids must have the same shape")
        if self.valid_tokens.device != self.document_ids.device:
            raise ValueError("valid_tokens and document_ids must share a device")
        if self.valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must have Boolean dtype")
        if self.document_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("document_ids must have an integer dtype")
        if self.valid_tokens.shape[0] < 1 or self.valid_tokens.shape[1] < 1:
            raise ValueError("sequence layout dimensions must be positive")
        if torch.any(self.document_ids[self.valid_tokens] < 0):
            raise ValueError("valid tokens must have a nonnegative document id")
        if torch.any(self.document_ids[~self.valid_tokens] != -1):
            raise ValueError("invalid tokens must use the -1 document sentinel")

    @property
    def batch_size(self) -> int:
        return self.valid_tokens.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.valid_tokens.shape[1]

    @classmethod
    def unpacked(
        cls,
        *,
        batch_size: int,
        sequence_length: int,
        valid_lengths: Tensor | None = None,
        device: torch.device | None = None,
    ) -> EagleTTTSequenceLayout:
        """Create a layout for an unpacked batch with optional right padding."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if valid_lengths is None:
            lengths = torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.int64,
                device=device,
            )
        else:
            if valid_lengths.shape != (batch_size,):
                raise ValueError(
                    f"valid_lengths must have shape ({batch_size},), "
                    f"got {valid_lengths.shape}"
                )
            if valid_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("valid_lengths must have int32 or int64 dtype")
            lengths = valid_lengths.to(device=device, dtype=torch.int64)
            if torch.any(lengths < 0) or torch.any(lengths > sequence_length):
                raise ValueError("valid_lengths must lie within the sequence length")
        positions = torch.arange(sequence_length, device=lengths.device)
        valid_tokens = positions[None, :] < lengths[:, None]
        document_ids = torch.where(
            valid_tokens,
            torch.zeros((), dtype=torch.int64, device=lengths.device),
            torch.full((), -1, dtype=torch.int64, device=lengths.device),
        )
        return cls(valid_tokens=valid_tokens, document_ids=document_ids)

    @classmethod
    def from_cu_seqlens(
        cls,
        *,
        cu_seqlens: Tensor,
        sequence_length: int,
        device: torch.device | None = None,
    ) -> EagleTTTSequenceLayout:
        """Create a batch-one packed layout from MCore cumulative lengths."""
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "cu_seqlens must be a rank-1 tensor with at least two entries"
            )
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must have int32 or int64 dtype")
        cumulative = cu_seqlens.to(device=device, dtype=torch.int64)
        if cumulative[0].item() != 0:
            raise ValueError("cu_seqlens must start at zero")
        lengths = cumulative[1:] - cumulative[:-1]
        if torch.any(lengths < 0):
            raise ValueError("cu_seqlens must be monotonic")
        valid_length = cumulative[-1].item()
        if valid_length > sequence_length:
            raise ValueError("cu_seqlens may not exceed the sequence length")
        document_ids = torch.full(
            (sequence_length,),
            -1,
            dtype=torch.int64,
            device=cumulative.device,
        )
        document_ids[:valid_length] = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=cumulative.device),
            lengths,
        )
        valid_tokens = (
            torch.arange(
                sequence_length,
                device=cumulative.device,
            )
            < valid_length
        )
        return cls(
            valid_tokens=valid_tokens.unsqueeze(0),
            document_ids=document_ids.unsqueeze(0),
        )


@dataclass(frozen=True, slots=True)
class EagleTTTStoragePlan:
    """Upper bound for all additional state retained by multi-pass training."""

    batch_size: int
    kv_heads: int
    sequence_length: int
    head_dim: int
    dtype: torch.dtype
    pass_count: int
    max_passes: int
    activation_budget_bytes: int
    layer_count: int = 1
    hidden_size: int = 0
    rope_dim: int = 0

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
            "layer_count": self.layer_count,
        }
        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in {
            "hidden_size": self.hidden_size,
            "rope_dim": self.rope_dim,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
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
        """Return the conservative total retained activation bound."""
        return (
            self.kv_bytes
            + self.hidden_bytes
            + self.rope_bytes
            + self.mask_bytes
            + self.loss_bytes
        )

    @property
    def kv_bytes(self) -> int:
        """Return per-layer trunk and prior-branch K/V bytes."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        elements_per_tensor = (
            self.batch_size * self.kv_heads * self.sequence_length * self.head_dim
        )
        return (
            self.layer_count * self.pass_count * 2 * elements_per_tensor * element_size
        )

    @property
    def hidden_bytes(self) -> int:
        """Return full-sequence branch outputs retained for pass backward."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.pass_count
            * self.batch_size
            * self.sequence_length
            * self.hidden_size
            * element_size
        )

    @property
    def rope_bytes(self) -> int:
        """Return the single shared base RoPE table bound."""
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return self.sequence_length * self.rope_dim * element_size

    @property
    def mask_bytes(self) -> int:
        """Return cached Flex BlockMask metadata for every pass.

        PyTorch retains four int32 block-count vectors and four int32 block-index
        matrices (forward and transposed, partial and full blocks). This bound
        deliberately assumes every block-index matrix is full.
        """
        block_count = math.ceil(self.sequence_length / _FLEX_BLOCK_SIZE)
        return self.pass_count * 16 * (block_count + block_count * block_count)

    @property
    def loss_bytes(self) -> int:
        """Return projected-loss row indices, bins, and FP32 normalizer bytes."""
        if self.hidden_size == 0:
            return 0
        rows = sum(
            self.batch_size * max(self.sequence_length - pass_index - 1, 0)
            for pass_index in range(self.pass_count)
        )
        return rows * (2 * torch.int64.itemsize + 2 * torch.float32.itemsize)


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
    pass_index: int,
    layout: EagleTTTSequenceLayout | None,
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
    query_positions = torch.arange(sequence_length, device=query.device)[:, None]
    key_positions = torch.arange(sequence_length, device=query.device)[None, :]
    causal = key_positions <= query_positions - pass_index
    if layout is None:
        visible = causal[None].expand(query.shape[0], -1, -1)
    else:
        same_document = (
            layout.document_ids[:, :, None] == layout.document_ids[:, None, :]
        )
        valid_pair = layout.valid_tokens[:, :, None] & layout.valid_tokens[:, None, :]
        visible = causal[None] & same_document & valid_pair
    scores.masked_fill_(~visible[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    safe_rows = visible.any(dim=-1)
    safe_scores = torch.where(safe_rows[:, None, :, None], scores, 0.0)
    output = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(safe_scores, dim=-1),
        expanded_value.to(compute_dtype),
    )
    output.masked_fill_(~safe_rows[:, None, :, None], 0.0)
    return output, lse


def _causal_mask(
    _batch: Tensor,
    _head: Tensor,
    query_index: Tensor,
    key_index: Tensor,
    *,
    pass_index: int,
) -> Tensor:
    return key_index <= query_index - pass_index


@lru_cache(maxsize=64)
def _causal_block_mask(
    sequence_length: int,
    pass_index: int,
    device: torch.device,
) -> Any:
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        lambda batch, head, query, key: _causal_mask(
            batch,
            head,
            query,
            key,
            pass_index=pass_index,
        ),
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
    )


def _layout_block_mask(
    *,
    layout: EagleTTTSequenceLayout,
    pass_index: int,
) -> Any:
    from torch.nn.attention.flex_attention import create_block_mask

    def packed_causal_mask(
        batch: Tensor,
        _head: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tensor:
        return (
            layout.valid_tokens[batch, query]
            & layout.valid_tokens[batch, key]
            & (layout.document_ids[batch, query] == layout.document_ids[batch, key])
            & (key <= query - pass_index)
        )

    return create_block_mask(
        packed_causal_mask,
        B=layout.batch_size,
        H=None,
        Q_LEN=layout.sequence_length,
        KV_LEN=layout.sequence_length,
        device=layout.valid_tokens.device,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
    )


def reset_eagle_ttt_attention_state() -> None:
    """Release device-resident per-sequence mask state at session teardown."""
    _causal_block_mask.cache_clear()


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Any:
    """Compile FlexAttention once so CUDA never uses its eager score fallback."""
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _flex_causal_trunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float,
    pass_index: int,
    layout: EagleTTTSequenceLayout | None,
) -> tuple[Tensor, Tensor]:
    from torch.nn.attention.flex_attention import (
        AuxRequest,  # pyrefly: ignore[missing-module-attribute]
    )

    flex_attention_call = _compiled_flex_attention()
    output, auxiliary = flex_attention_call(
        query,
        key,
        value,
        block_mask=(
            _causal_block_mask(query.shape[2], pass_index, query.device)
            if layout is None
            else _layout_block_mask(layout=layout, pass_index=pass_index)
        ),
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options={
            # Let Flex guard padded rows in dynamic partial query blocks.
            "ROWS_GUARANTEED_SAFE": False,
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
    position_offset: int,
    scale: float,
    layout: EagleTTTSequenceLayout | None,
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
    if position_offset == 0:
        aligned_key = branch_key
        aligned_value = branch_value
    elif position_offset >= sequence_length:
        aligned_key = branch_key * 0
        aligned_value = branch_value * 0
    else:
        aligned_key = torch.nn.functional.pad(
            branch_key[:, :, : sequence_length - position_offset],
            (0, 0, position_offset, 0),
        )
        aligned_value = torch.nn.functional.pad(
            branch_value[:, :, : sequence_length - position_offset],
            (0, 0, position_offset, 0),
        )
    branch_scores = torch.einsum(
        "bhgsd,bhsd->bhgs",
        grouped_query,
        aligned_key.to(compute_dtype),
    ).reshape(batch, query_heads, sequence_length)
    branch_scores.mul_(scale)
    positions = torch.arange(sequence_length, device=query.device)
    valid_position = positions >= position_offset
    if layout is None:
        valid = valid_position[None].expand(batch, -1)
    else:
        source_positions = (positions - position_offset).clamp_min(0)
        valid = (
            valid_position[None]
            & layout.valid_tokens
            & layout.valid_tokens[:, source_positions]
            & (layout.document_ids == layout.document_ids[:, source_positions])
        )
    branch_scores.masked_fill_(~valid[:, None, :], float("-inf"))
    merged_lse = torch.logaddexp(lse, branch_scores)
    grouped_output = output.to(compute_dtype).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        head_dim,
    )
    trunk_weight = torch.where(
        torch.isfinite(lse),
        torch.exp(lse - merged_lse),
        0.0,
    ).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    branch_weight = torch.where(
        valid[:, None, :],
        torch.exp(branch_scores - merged_lse),
        0.0,
    ).reshape(
        batch,
        kv_heads,
        groups,
        sequence_length,
        1,
    )
    merged_output = grouped_output * trunk_weight
    merged_output.add_(aligned_value.to(compute_dtype).unsqueeze(2) * branch_weight)
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
    layout: EagleTTTSequenceLayout | None = None,
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
    if layout is not None:
        if layout.valid_tokens.shape != (query.shape[0], query.shape[2]):
            raise ValueError("sequence layout must match query batch and sequence axes")
        if layout.valid_tokens.device != query.device:
            raise ValueError("sequence layout and query must share a device")
    _expand_gqa(state.trunk_key, query_heads=query.shape[1])

    attention_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    if query.is_cuda and query.dtype != torch.float64:
        output, lse = _flex_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
            pass_index=plan.pass_index,
            layout=layout,
        )
    else:
        output, lse = _dense_causal_trunk(
            query,
            state.trunk_key,
            state.trunk_value,
            scale=attention_scale,
            pass_index=plan.pass_index,
            layout=layout,
        )
    for branch_index, (branch_key, branch_value) in enumerate(
        zip(state.branch_keys, state.branch_values, strict=True)
    ):
        output, lse = _merge_branch(
            query=query,
            output=output,
            lse=lse,
            branch_key=branch_key,
            branch_value=branch_value,
            position_offset=plan.pass_index - branch_index - 1,
            scale=attention_scale,
            layout=layout,
        )
    return output.to(dtype=query.dtype)
