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
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from nemo_rl.models.megatron.draft.block_plan import DFlashBatchPlan


_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)


def _validate_attention_inputs(
    *,
    plan: DFlashBatchPlan,
    trunk_q: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
) -> None:
    tensors = (trunk_q, trunk_k, trunk_v, block_q, block_k, block_v)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("DFlash attention tensors must have rank four")

    batch_size = plan.batch_size
    sequence_length = plan.sequence_length
    num_blocks = batch_size * plan.anchors_per_sample
    block_size = plan.block_size
    if trunk_q.shape[:2] != (batch_size, sequence_length):
        raise ValueError("trunk_q shape does not match the DFlash plan")
    if trunk_k.shape[:2] != (batch_size, sequence_length):
        raise ValueError("trunk_k shape does not match the DFlash plan")
    if trunk_v.shape[:2] != (batch_size, sequence_length):
        raise ValueError("trunk_v shape does not match the DFlash plan")
    if block_q.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_q shape does not match the DFlash plan")
    if block_k.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_k shape does not match the DFlash plan")
    if block_v.shape[:2] != (num_blocks, block_size):
        raise ValueError("block_v shape does not match the DFlash plan")

    device = trunk_q.device
    dtype = trunk_q.dtype
    plan_tensors = (
        plan.token_valid_mask,
        plan.sample_rows,
        plan.anchor_positions,
        plan.slot_valid,
    )
    if any(tensor.device != device for tensor in (*tensors, *plan_tensors)):
        raise ValueError("DFlash attention inputs and plan must share a device")
    if not dtype.is_floating_point:
        raise TypeError("DFlash attention inputs must use a floating dtype")
    if any(tensor.dtype != dtype for tensor in tensors):
        raise TypeError("DFlash attention inputs must share a dtype")

    num_query_heads = trunk_q.shape[2]
    num_kv_heads = trunk_k.shape[2]
    head_dim = trunk_q.shape[3]
    if num_query_heads == 0 or num_kv_heads == 0:
        raise ValueError("DFlash attention requires nonzero head counts")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    if trunk_k.shape[3] != head_dim or trunk_v.shape[3] != head_dim:
        raise ValueError("trunk Q/K/V head dimensions must match")
    if trunk_v.shape[2] != num_kv_heads:
        raise ValueError("trunk K/V head counts must match")
    if block_q.shape[2:] != (num_query_heads, head_dim):
        raise ValueError("trunk and block query shapes must match")
    if block_k.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError("trunk and block key shapes must match")
    if block_v.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError("trunk and block value shapes must match")


def _grouped_masked_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    visibility: Tensor,
    *,
    scale: float,
) -> Tensor:
    batch_size, query_length, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    heads_per_group = num_query_heads // num_kv_heads

    grouped_query = query.permute(0, 2, 1, 3).reshape(
        batch_size,
        num_kv_heads,
        heads_per_group,
        query_length,
        head_dim,
    )
    key_by_head = key.permute(0, 2, 1, 3)
    value_by_head = value.permute(0, 2, 1, 3)
    scores = torch.einsum(
        "bhgqd,bhkd->bhgqk",
        grouped_query,
        key_by_head,
    )
    scores = scores * scale

    expanded_visibility = visibility[:, None, None, :, :]
    row_valid = visibility.any(dim=-1)
    scores = scores.masked_fill(~expanded_visibility, -torch.inf)
    scores = torch.where(
        row_valid[:, None, None, :, None],
        scores,
        torch.zeros_like(scores),
    )
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    probabilities = torch.where(
        expanded_visibility,
        probabilities,
        torch.zeros_like(probabilities),
    )
    output = torch.einsum(
        "bhgqk,bhkd->bhgqd",
        probabilities,
        value_by_head,
    )
    return (
        output.reshape(batch_size, num_query_heads, query_length, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _trunk_visibility(plan: DFlashBatchPlan) -> Tensor:
    positions = torch.arange(
        plan.sequence_length,
        dtype=torch.int64,
        device=plan.token_valid_mask.device,
    )
    causal = positions[None, :] <= positions[:, None]
    return (
        plan.token_valid_mask[:, :, None]
        & plan.token_valid_mask[:, None, :]
        & causal[None, :, :]
    )


def _block_visibility(plan: DFlashBatchPlan) -> Tensor:
    num_blocks = plan.batch_size * plan.anchors_per_sample
    trunk_key_count = plan.batch_size * plan.sequence_length
    device = plan.token_valid_mask.device

    if plan.sequence_length == 0:
        visible_trunk = torch.zeros(
            (num_blocks, 0),
            dtype=torch.bool,
            device=device,
        )
    else:
        trunk_key_indices = torch.arange(
            trunk_key_count,
            dtype=torch.int64,
            device=device,
        )
        trunk_key_rows = torch.div(
            trunk_key_indices,
            plan.sequence_length,
            rounding_mode="floor",
        )
        trunk_key_positions = torch.remainder(
            trunk_key_indices,
            plan.sequence_length,
        )
        visible_trunk = (
            (trunk_key_rows[None, :] == plan.sample_rows[:, None])
            & (trunk_key_positions[None, :] < plan.anchor_positions[:, None])
            & plan.token_valid_mask.reshape(-1)[None, :]
        )

    block_key_count = num_blocks * plan.block_size
    block_key_indices = torch.arange(
        block_key_count,
        dtype=torch.int64,
        device=device,
    )
    block_key_rows = torch.div(
        block_key_indices,
        plan.block_size,
        rounding_mode="floor",
    )
    visible_block = (
        block_key_rows[None, :] == torch.arange(num_blocks, device=device)[:, None]
    ) & plan.slot_valid.reshape(-1)[None, :]

    visible_keys = torch.cat((visible_trunk, visible_block), dim=-1)
    return plan.slot_valid[:, :, None] & visible_keys[:, None, :]


def _create_trunk_block_mask(plan: DFlashBatchPlan) -> BlockMask:
    token_valid_mask = plan.token_valid_mask

    def trunk_mask(
        batch_index: Tensor,
        _head_index: Tensor,
        query_index: Tensor,
        key_index: Tensor,
    ) -> Tensor:
        return (
            token_valid_mask[batch_index, query_index]
            & token_valid_mask[batch_index, key_index]
            & (key_index <= query_index)
        )

    return create_block_mask(
        mask_mod=trunk_mask,
        B=plan.batch_size,
        H=None,
        Q_LEN=plan.sequence_length,
        KV_LEN=plan.sequence_length,
        device=token_valid_mask.device,
    )


def _create_global_block_mask(plan: DFlashBatchPlan) -> BlockMask:
    num_blocks = plan.batch_size * plan.anchors_per_sample
    trunk_key_count = plan.batch_size * plan.sequence_length
    block_key_count = num_blocks * plan.block_size
    token_valid_mask = plan.token_valid_mask.reshape(-1)
    slot_valid = plan.slot_valid.reshape(-1)

    if plan.sequence_length == 0:

        def global_mask(
            block_index: Tensor,
            _head_index: Tensor,
            query_index: Tensor,
            key_index: Tensor,
        ) -> Tensor:
            key_block_index = torch.div(
                key_index,
                plan.block_size,
                rounding_mode="floor",
            )
            return (
                plan.slot_valid[block_index, query_index]
                & (key_block_index == block_index)
                & slot_valid[key_index]
            )

    else:

        def global_mask(
            block_index: Tensor,
            _head_index: Tensor,
            query_index: Tensor,
            key_index: Tensor,
        ) -> Tensor:
            safe_trunk_index = torch.clamp(key_index, max=trunk_key_count - 1)
            trunk_row = torch.div(
                safe_trunk_index,
                plan.sequence_length,
                rounding_mode="floor",
            )
            trunk_position = torch.remainder(
                safe_trunk_index,
                plan.sequence_length,
            )
            visible_trunk = (
                (key_index < trunk_key_count)
                & (trunk_row == plan.sample_rows[block_index])
                & (trunk_position < plan.anchor_positions[block_index])
                & token_valid_mask[safe_trunk_index]
            )

            safe_block_index = torch.clamp(
                key_index - trunk_key_count,
                min=0,
                max=block_key_count - 1,
            )
            key_block_index = torch.div(
                safe_block_index,
                plan.block_size,
                rounding_mode="floor",
            )
            visible_block = (
                (key_index >= trunk_key_count)
                & (key_block_index == block_index)
                & slot_valid[safe_block_index]
            )
            return plan.slot_valid[block_index, query_index] & (
                visible_trunk | visible_block
            )

    return create_block_mask(
        mask_mod=global_mask,
        B=num_blocks,
        H=None,
        Q_LEN=plan.block_size,
        KV_LEN=trunk_key_count + block_key_count,
        device=plan.token_valid_mask.device,
    )


def _flex_attention_cuda(
    *,
    plan: DFlashBatchPlan,
    trunk_q: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    num_kv_heads = trunk_k.shape[2]
    head_dim = trunk_k.shape[3]
    enable_gqa = trunk_q.shape[2] != num_kv_heads

    if plan.sequence_length == 0:
        trunk_output = torch.zeros_like(trunk_q)
    else:
        trunk_output = (
            cast(
                Tensor,
                _COMPILED_FLEX_ATTENTION(
                    trunk_q.permute(0, 2, 1, 3),
                    trunk_k.permute(0, 2, 1, 3),
                    trunk_v.permute(0, 2, 1, 3),
                    block_mask=_create_trunk_block_mask(plan),
                    scale=scale,
                    enable_gqa=enable_gqa,
                ),
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    global_key = torch.cat(
        (
            trunk_k.reshape(1, -1, num_kv_heads, head_dim),
            block_k.reshape(1, -1, num_kv_heads, head_dim),
        ),
        dim=1,
    ).permute(0, 2, 1, 3)
    global_value = torch.cat(
        (
            trunk_v.reshape(1, -1, num_kv_heads, head_dim),
            block_v.reshape(1, -1, num_kv_heads, head_dim),
        ),
        dim=1,
    ).permute(0, 2, 1, 3)
    block_output = (
        cast(
            Tensor,
            _COMPILED_FLEX_ATTENTION(
                block_q.permute(0, 2, 1, 3),
                global_key,
                global_value,
                block_mask=_create_global_block_mask(plan),
                scale=scale,
                enable_gqa=enable_gqa,
            ),
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return trunk_output, block_output


def dflash_block_attention(
    *,
    plan: DFlashBatchPlan,
    trunk_q: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply causal trunk and bidirectional anchored-block attention."""
    _validate_attention_inputs(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )
    effective_scale = trunk_q.shape[-1] ** -0.5 if scale is None else scale
    if not math.isfinite(effective_scale):
        raise ValueError("attention scale must be finite")

    if trunk_q.device.type == "cuda":
        trunk_output, block_output = _flex_attention_cuda(
            plan=plan,
            trunk_q=trunk_q,
            trunk_k=trunk_k,
            trunk_v=trunk_v,
            block_q=block_q,
            block_k=block_k,
            block_v=block_v,
            scale=effective_scale,
        )
    else:
        trunk_output = _grouped_masked_attention(
            trunk_q,
            trunk_k,
            trunk_v,
            _trunk_visibility(plan),
            scale=effective_scale,
        )
        num_kv_heads = trunk_k.shape[2]
        head_dim = trunk_k.shape[3]
        global_key = torch.cat(
            (
                trunk_k.reshape(1, -1, num_kv_heads, head_dim),
                block_k.reshape(1, -1, num_kv_heads, head_dim),
            ),
            dim=1,
        )
        global_value = torch.cat(
            (
                trunk_v.reshape(1, -1, num_kv_heads, head_dim),
                block_v.reshape(1, -1, num_kv_heads, head_dim),
            ),
            dim=1,
        )
        block_output = _grouped_masked_attention(
            block_q,
            global_key,
            global_value,
            _block_visibility(plan),
            scale=effective_scale,
        )
    return (
        torch.where(
            plan.token_valid_mask[:, :, None, None],
            trunk_output,
            torch.zeros_like(trunk_output),
        ),
        torch.where(
            plan.slot_valid[:, :, None, None],
            block_output,
            torch.zeros_like(block_output),
        ),
    )
