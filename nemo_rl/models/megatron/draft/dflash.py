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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from nemo_rl.models.megatron.draft.block_attention import dflash_block_attention
from nemo_rl.models.megatron.draft.block_plan import DFlashBatchPlan

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict


@dataclass(frozen=True, slots=True)
class DFlashBodyConfig:
    """Checkpoint-compatible DFlash body dimensions."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_hidden_layers: int = 5
    num_target_taps: int = 5
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        integer_fields = (
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.num_hidden_layers,
            self.num_target_taps,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("DFlash dimensions must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")


class QwenRMSNorm(nn.Module):
    """Qwen RMS normalization with FP32 variance accumulation."""

    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(torch.float32)
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class _DFlashAttention(nn.Module):
    def __init__(self, config: DFlashBodyConfig) -> None:
        super().__init__()
        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(config.head_dim, eps=config.rms_norm_eps)


class _DFlashMLP(nn.Module):
    def __init__(self, config: DFlashBodyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


class _DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashBodyConfig) -> None:
        super().__init__()
        self.self_attn = _DFlashAttention(config)
        self.mlp = _DFlashMLP(config)
        self.input_layernorm = QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rope(
    hidden_states: Tensor,
    positions: Tensor,
    *,
    theta: float,
) -> Tensor:
    head_dim = hidden_states.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            / head_dim
        )
    )
    frequencies = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine = angles.cos().to(hidden_states.dtype).unsqueeze(-2)
    sine = angles.sin().to(hidden_states.dtype).unsqueeze(-2)
    return hidden_states * cosine + _rotate_half(hidden_states) * sine


class DFlashBody(nn.Module):
    """DFlash decoder body that shares target embeddings and output head."""

    def __init__(self, config: DFlashBodyConfig | None = None) -> None:
        super().__init__()
        self.config = DFlashBodyConfig() if config is None else config
        self.fc = nn.Linear(
            self.config.num_target_taps * self.config.hidden_size,
            self.config.hidden_size,
            bias=False,
        )
        self.hidden_norm = QwenRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            _DFlashDecoderLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        )
        self.norm = QwenRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _validate_inputs(
        self,
        *,
        target_taps: Tensor,
        block_embeddings: Tensor,
        plan: DFlashBatchPlan,
    ) -> None:
        expected_target_shape = (
            plan.batch_size,
            plan.sequence_length,
            self.config.num_target_taps,
            self.config.hidden_size,
        )
        if target_taps.shape != expected_target_shape:
            raise ValueError(
                "target_taps must have shape "
                f"{expected_target_shape}, got {tuple(target_taps.shape)}"
            )
        num_blocks = plan.batch_size * plan.anchors_per_sample
        expected_block_shape = (
            num_blocks,
            plan.block_size,
            self.config.hidden_size,
        )
        if block_embeddings.shape != expected_block_shape:
            raise ValueError(
                "block_embeddings must have shape "
                f"{expected_block_shape}, got {tuple(block_embeddings.shape)}"
            )
        plan_tensors = (
            plan.token_valid_mask,
            plan.sample_rows,
            plan.anchor_positions,
            plan.query_positions,
            plan.slot_valid,
        )
        if any(tensor.device != target_taps.device for tensor in plan_tensors):
            raise ValueError("DFlash plan and inputs must share a device")
        if block_embeddings.device != target_taps.device:
            raise ValueError("DFlash inputs must share a device")
        if block_embeddings.dtype != target_taps.dtype:
            raise TypeError("DFlash inputs must share a dtype")
        if not target_taps.dtype.is_floating_point:
            raise TypeError("DFlash inputs must use a floating dtype")
        if plan.query_positions.dtype != torch.int64:
            raise TypeError("DFlash query positions must use torch.int64")

    def forward(
        self,
        *,
        target_taps: Tensor,
        block_embeddings: Tensor,
        plan: DFlashBatchPlan,
    ) -> Tensor:
        """Update anchored block embeddings from target hidden-state taps."""
        self._validate_inputs(
            target_taps=target_taps,
            block_embeddings=block_embeddings,
            plan=plan,
        )
        config = self.config
        batch_size = plan.batch_size
        sequence_length = plan.sequence_length
        target_hidden = self.hidden_norm(self.fc(target_taps.flatten(start_dim=2)))
        hidden_states = torch.where(
            plan.slot_valid[..., None],
            block_embeddings,
            torch.zeros_like(block_embeddings),
        )
        trunk_positions = torch.arange(
            sequence_length,
            dtype=torch.int64,
            device=target_taps.device,
        ).expand(batch_size, -1)

        for layer in self.layers:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            trunk_key = layer.self_attn.k_norm(
                layer.self_attn.k_proj(target_hidden).view(
                    batch_size,
                    sequence_length,
                    config.num_key_value_heads,
                    config.head_dim,
                )
            )
            trunk_value = layer.self_attn.v_proj(target_hidden).view(
                batch_size,
                sequence_length,
                config.num_key_value_heads,
                config.head_dim,
            )
            block_query = layer.self_attn.q_norm(
                layer.self_attn.q_proj(normalized).view(
                    *normalized.shape[:2],
                    config.num_attention_heads,
                    config.head_dim,
                )
            )
            block_key = layer.self_attn.k_norm(
                layer.self_attn.k_proj(normalized).view(
                    *normalized.shape[:2],
                    config.num_key_value_heads,
                    config.head_dim,
                )
            )
            block_value = layer.self_attn.v_proj(normalized).view(
                *normalized.shape[:2],
                config.num_key_value_heads,
                config.head_dim,
            )
            trunk_key = _apply_rope(
                trunk_key,
                trunk_positions,
                theta=config.rope_theta,
            )
            block_query = _apply_rope(
                block_query,
                plan.query_positions,
                theta=config.rope_theta,
            )
            block_key = _apply_rope(
                block_key,
                plan.query_positions,
                theta=config.rope_theta,
            )
            trunk_query = torch.zeros(
                (
                    batch_size,
                    sequence_length,
                    config.num_attention_heads,
                    config.head_dim,
                ),
                dtype=target_taps.dtype,
                device=target_taps.device,
            )
            _, attention_output = dflash_block_attention(
                plan=plan,
                trunk_q=trunk_query,
                trunk_k=trunk_key,
                trunk_v=trunk_value,
                block_q=block_query,
                block_k=block_key,
                block_v=block_value,
            )
            hidden_states = residual + layer.self_attn.o_proj(
                attention_output.flatten(start_dim=2)
            )
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )

        output = self.norm(hidden_states)
        return torch.where(
            plan.slot_valid[..., None],
            output,
            torch.zeros_like(output),
        )

    def sharded_state_dict(self, prefix: str = "") -> ShardedStateDict:
        """Return replicated MCore sharded tensors with public checkpoint names."""
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        replica_id: int | tuple[int, int, int]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            replica_id = (0, torch.distributed.get_rank(), 0)
        else:
            replica_id = 0
        sharded: ShardedStateDict = {}
        for name, tensor in self.state_dict(keep_vars=True).items():
            key = f"{prefix}{name}"
            sharded[key] = ShardedTensor.from_rank_offsets(
                key,
                tensor,
                replica_id=replica_id,
            )
        return sharded


__all__ = ["DFlashBody", "DFlashBodyConfig", "QwenRMSNorm"]
