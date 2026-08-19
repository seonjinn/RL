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

import torch
from torch import Tensor, nn

from nemo_rl.models.megatron.draft.block_attention import dflash_block_attention
from nemo_rl.models.megatron.draft.block_plan import DFlashBatchPlan


class _DFlashLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_query_groups: int,
        ffn_hidden_size: int,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.head_dim = hidden_size // num_attention_heads
        qkv_size = (num_attention_heads + 2 * num_query_groups) * self.head_dim

        self.attention_norm = nn.LayerNorm(hidden_size, bias=False)
        self.qkv_projection = nn.Linear(hidden_size, qkv_size, bias=False)
        self.attention_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.ffn_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.qkv_projection(self.attention_norm(hidden_states))
        query_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_query_groups * self.head_dim
        query, key, value = projected.split((query_size, kv_size, kv_size), dim=-1)
        return (
            query.unflatten(-1, (self.num_attention_heads, self.head_dim)),
            key.unflatten(-1, (self.num_query_groups, self.head_dim)),
            value.unflatten(-1, (self.num_query_groups, self.head_dim)),
        )

    def _apply_ffn(self, hidden_states: Tensor) -> Tensor:
        normalized = self.ffn_norm(hidden_states)
        return hidden_states + self.ffn_down(
            torch.nn.functional.silu(self.ffn_up(normalized))
        )

    def forward(
        self,
        *,
        plan: DFlashBatchPlan,
        trunk_hidden_states: Tensor,
        block_hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        trunk_q, trunk_k, trunk_v = self._project_qkv(trunk_hidden_states)
        block_q, block_k, block_v = self._project_qkv(block_hidden_states)
        trunk_attention, block_attention = dflash_block_attention(
            plan=plan,
            trunk_q=trunk_q,
            trunk_k=trunk_k,
            trunk_v=trunk_v,
            block_q=block_q,
            block_k=block_k,
            block_v=block_v,
        )
        trunk_hidden_states = trunk_hidden_states + self.attention_output(
            trunk_attention.flatten(-2)
        )
        block_hidden_states = block_hidden_states + self.attention_output(
            block_attention.flatten(-2)
        )
        return (
            self._apply_ffn(trunk_hidden_states),
            self._apply_ffn(block_hidden_states),
        )


class DFlashModel(nn.Module):
    """Draft-only DFlash transformer over caller-owned target features."""

    def __init__(
        self,
        *,
        target_hidden_size: int,
        draft_hidden_size: int,
        num_target_hidden_taps: int,
        num_layers: int,
        num_attention_heads: int,
        num_query_groups: int,
        ffn_hidden_size: int,
    ) -> None:
        super().__init__()
        if target_hidden_size <= 0 or draft_hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        if num_target_hidden_taps <= 0 or num_layers <= 0:
            raise ValueError("tap and layer counts must be positive")
        if num_attention_heads <= 0 or num_query_groups <= 0:
            raise ValueError("attention head counts must be positive")
        if draft_hidden_size % num_attention_heads != 0:
            raise ValueError("draft hidden size must be divisible by attention heads")
        if num_attention_heads % num_query_groups != 0:
            raise ValueError("attention heads must be divisible by query groups")
        if ffn_hidden_size <= 0:
            raise ValueError("FFN hidden size must be positive")

        self.target_hidden_size = target_hidden_size
        self.num_target_hidden_taps = num_target_hidden_taps
        self.trunk_projection = nn.Linear(
            target_hidden_size * num_target_hidden_taps,
            draft_hidden_size,
            bias=False,
        )
        self.block_projection = nn.Linear(
            target_hidden_size,
            draft_hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                _DFlashLayer(
                    hidden_size=draft_hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_query_groups=num_query_groups,
                    ffn_hidden_size=ffn_hidden_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(draft_hidden_size, bias=False)
        self.output_projection = nn.Linear(
            draft_hidden_size,
            target_hidden_size,
            bias=False,
        )

    def _validate_inputs(
        self,
        *,
        plan: DFlashBatchPlan,
        target_hidden_taps: Tensor,
        input_embeddings: Tensor,
    ) -> None:
        expected_trunk_shape = (
            self.num_target_hidden_taps,
            plan.batch_size,
            plan.sequence_length,
            self.target_hidden_size,
        )
        if target_hidden_taps.shape != expected_trunk_shape:
            raise ValueError(
                "target_hidden_taps must have shape "
                "[taps, batch, sequence, target_hidden_size]"
            )
        expected_block_shape = (
            plan.batch_size * plan.anchors_per_sample,
            plan.block_size,
            self.target_hidden_size,
        )
        if input_embeddings.shape != expected_block_shape:
            raise ValueError(
                "input_embeddings must have shape "
                "[blocks, block_size, target_hidden_size]"
            )
        if not target_hidden_taps.dtype.is_floating_point:
            raise TypeError("DFlash model inputs must use a floating dtype")
        if input_embeddings.dtype != target_hidden_taps.dtype:
            raise TypeError("DFlash model inputs must share a dtype")
        plan_tensors = (plan.token_valid_mask, plan.slot_valid)
        if any(
            tensor.device != target_hidden_taps.device
            for tensor in (*plan_tensors, input_embeddings)
        ):
            raise ValueError("DFlash model inputs and plan must share a device")

    def forward(
        self,
        *,
        plan: DFlashBatchPlan,
        target_hidden_taps: Tensor,
        input_embeddings: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            plan=plan,
            target_hidden_taps=target_hidden_taps,
            input_embeddings=input_embeddings,
        )
        trunk_hidden_states = self.trunk_projection(
            target_hidden_taps.permute(1, 2, 0, 3).flatten(-2)
        )
        block_hidden_states = self.block_projection(input_embeddings)
        for layer in self.layers:
            trunk_hidden_states, block_hidden_states = layer(
                plan=plan,
                trunk_hidden_states=trunk_hidden_states,
                block_hidden_states=block_hidden_states,
            )
        output = self.output_projection(self.output_norm(block_hidden_states))
        return torch.where(
            plan.slot_valid[..., None],
            output,
            torch.zeros_like(output),
        )
