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
from typing import Protocol

import torch
from torch import Tensor

from nemo_rl.algorithms.loss.draft import DraftLossStats
from nemo_rl.models.megatron.draft.eagle_ttt import (
    EagleTTTAttentionPlan,
    EagleTTTStoragePlan,
)

__all__ = [
    "EaglePassLoss",
    "EaglePassRunner",
    "EagleTTTOutput",
    "EagleTTTProvider",
    "modelopt_static_rotary_table",
]


class EaglePassRunner(Protocol):
    """Execute one pass with the structured trunk-plus-branch attention backend."""

    def __call__(
        self,
        *,
        embeddings: Tensor,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
        rope_positions: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class EaglePassLoss(Protocol):
    """Project and reduce one pass without returning its vocabulary logits."""

    def __call__(
        self,
        *,
        hidden_states: Tensor,
        plan: EagleTTTAttentionPlan,
    ) -> DraftLossStats: ...


@dataclass(frozen=True, slots=True)
class EagleTTTOutput:
    """Additive pass statistics and the detached state for the next invocation."""

    stats: DraftLossStats
    final_branch_state: Tensor
    plans: tuple[EagleTTTAttentionPlan, ...]


def modelopt_static_rotary_table(
    *,
    base_rotary_pos_emb: Tensor,
    plan: EagleTTTAttentionPlan,
) -> Tensor:
    """Build the linear table required by ModelOpt's static MCore cache slices.

    ModelOpt advances ``StaticInferenceContext.sequence_len_offset`` by one full
    sequence per pass. Its public training path repeats the base RoPE table, so
    pass ``p`` needs ``p + 1`` copies for MCore's
    ``[p * sequence_length:(p + 1) * sequence_length]`` query slice.
    """
    if base_rotary_pos_emb.ndim < 1:
        raise ValueError("base_rotary_pos_emb must have a sequence dimension")
    if base_rotary_pos_emb.shape[0] != plan.sequence_length:
        raise ValueError(
            "base_rotary_pos_emb sequence length must match the attention plan, "
            f"got {base_rotary_pos_emb.shape[0]} and {plan.sequence_length}"
        )
    if plan.pass_index == 0:
        return base_rotary_pos_emb
    return torch.cat([base_rotary_pos_emb] * (plan.pass_index + 1), dim=0)


class EagleTTTProvider:
    """Bounded method provider over a structured EAGLE pass runner.

    The provider deliberately does not call ModelOpt's dense multi-step mask
    helper. A model adapter must supply ``EaglePassRunner`` using the structured
    attention backend, while a PR6-style projected loss callable consumes each
    pass immediately. This keeps both square masks and retained pass logits out
    of the provider contract.
    """

    def __init__(
        self,
        *,
        max_passes: int,
        activation_budget_bytes: int,
    ) -> None:
        if max_passes < 1:
            raise ValueError(f"max_passes must be positive, got {max_passes}")
        if activation_budget_bytes < 1:
            raise ValueError(
                "activation_budget_bytes must be positive, "
                f"got {activation_budget_bytes}"
            )
        self.max_passes = max_passes
        self.activation_budget_bytes = activation_budget_bytes

    def _storage_plan(
        self,
        *,
        target_trunk_states: Tensor,
        pass_count: int,
    ) -> EagleTTTStoragePlan:
        if target_trunk_states.ndim != 3:
            raise ValueError(
                "target_trunk_states must have [sequence, batch, hidden] shape, "
                f"got {target_trunk_states.shape}"
            )
        sequence_length, batch_size, hidden_size = target_trunk_states.shape
        return EagleTTTStoragePlan(
            batch_size=batch_size,
            kv_heads=1,
            sequence_length=sequence_length,
            head_dim=hidden_size,
            dtype=target_trunk_states.dtype,
            pass_count=pass_count,
            max_passes=self.max_passes,
            activation_budget_bytes=self.activation_budget_bytes,
        )

    def forward_loss_stats(
        self,
        *,
        pass_runner: EaglePassRunner,
        pass_loss: EaglePassLoss,
        target_trunk_states: Tensor,
        input_embeds: Tensor,
        pass_count: int,
        pass_weights: Tensor,
    ) -> EagleTTTOutput:
        """Run bounded passes and consume each projected loss before the next pass."""
        storage = self._storage_plan(
            target_trunk_states=target_trunk_states,
            pass_count=pass_count,
        )
        if input_embeds.shape != target_trunk_states.shape:
            raise ValueError(
                "input_embeds must match target_trunk_states, "
                f"got {input_embeds.shape} and {target_trunk_states.shape}"
            )
        if (
            input_embeds.device != target_trunk_states.device
            or input_embeds.dtype != target_trunk_states.dtype
        ):
            raise ValueError(
                "input_embeds and target_trunk_states must share device and dtype"
            )
        if pass_weights.shape != (pass_count,):
            raise ValueError(
                f"pass_weights must have shape ({pass_count},), got {pass_weights.shape}"
            )
        if pass_weights.device != target_trunk_states.device:
            raise ValueError("pass_weights and target_trunk_states must share a device")

        current_hidden = target_trunk_states
        numerators: list[Tensor] = []
        counts: list[Tensor] = []
        plans: list[EagleTTTAttentionPlan] = []
        for pass_index in range(pass_count):
            plan = EagleTTTAttentionPlan(
                pass_index=pass_index,
                pass_count=pass_count,
                max_passes=self.max_passes,
                sequence_length=storage.sequence_length,
            )
            hidden_states, next_hidden_states = pass_runner(
                embeddings=input_embeds,
                hidden_states=current_hidden,
                plan=plan,
                rope_positions=plan.rope_positions(device=current_hidden.device),
            )
            self._validate_pass_output(
                hidden_states=hidden_states,
                next_hidden_states=next_hidden_states,
                target_trunk_states=target_trunk_states,
            )
            pass_stats = pass_loss(hidden_states=hidden_states, plan=plan)
            if pass_stats.numerators.shape != (1,) or pass_stats.counts.shape != (1,):
                raise ValueError(
                    "pass_loss must return exactly one additive bin per pass"
                )
            if not (
                pass_stats.numerators.device
                == pass_stats.counts.device
                == target_trunk_states.device
            ):
                raise ValueError("pass loss statistics must share the model device")
            numerators.append(pass_stats.numerators)
            counts.append(pass_stats.counts)
            plans.append(plan)

            shifted_branch = torch.cat(
                (
                    torch.zeros_like(next_hidden_states[:1]),
                    next_hidden_states[:-1],
                ),
                dim=0,
            )
            # Pinned ModelOpt EAGLE-3 captures the next-pass input with
            # clone().detach(); enforce that contract for every runner.
            current_hidden = shifted_branch.detach()

        return EagleTTTOutput(
            stats=DraftLossStats(
                numerators=torch.cat(numerators),
                counts=torch.cat(counts),
                weights=pass_weights.detach().to(dtype=torch.float32),
            ),
            final_branch_state=current_hidden,
            plans=tuple(plans),
        )

    @staticmethod
    def _validate_pass_output(
        *,
        hidden_states: Tensor,
        next_hidden_states: Tensor,
        target_trunk_states: Tensor,
    ) -> None:
        for name, tensor in (
            ("hidden", hidden_states),
            ("branch", next_hidden_states),
        ):
            if tensor.shape != target_trunk_states.shape:
                raise ValueError(
                    f"EAGLE pass {name} output must match target trunk shape, "
                    f"got {tensor.shape} and {target_trunk_states.shape}"
                )
            if (
                tensor.device != target_trunk_states.device
                or tensor.dtype != target_trunk_states.dtype
            ):
                raise ValueError(
                    f"EAGLE pass {name} output must share target device and dtype"
                )
