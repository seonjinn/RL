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
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from nemo_rl.algorithms.loss.draft import (
    DraftLossStats,
    streaming_vocab_parallel_soft_ce,
)
from nemo_rl.models.megatron.draft.eagle_ttt import (
    EagleTTTAttentionPlan,
    EagleTTTStoragePlan,
)


class _InferenceContext(Protocol):
    sequence_len_offset: int
    max_batch_size: int
    max_sequence_length: int
    key_value_memory_dict: dict[Any, Any]


class _EagleModule(Protocol):
    def __call__(
        self,
        *,
        embeddings: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        rotary_pos_emb: Tensor,
        inference_context: _InferenceContext,
    ) -> tuple[Tensor, Tensor]: ...


class InferenceContextFactory(Protocol):
    def __call__(
        self,
        max_batch_size: int,
        max_sequence_length: int,
    ) -> _InferenceContext: ...


MultiStepMaskHelper = Callable[[Tensor, int], Tensor]
RotaryProvider = Callable[..., Tensor]
LogitProjector = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class EagleTTTOutput:
    """Per-pass logits and differentiable branch states from one TTT forward."""

    pass_logits: tuple[Tensor, ...]
    branch_states: tuple[Tensor, ...]
    plans: tuple[EagleTTTAttentionPlan, ...]


class EagleTTTProvider:
    """Provider-shaped bounded EAGLE adapter awaiting shared worker integration."""

    def __init__(
        self,
        *,
        max_passes: int,
        activation_budget_bytes: int,
        token_chunk_size: int,
        inference_context_factory: InferenceContextFactory | None = None,
        multi_step_mask_helper: MultiStepMaskHelper | None = None,
    ) -> None:
        if max_passes < 1:
            raise ValueError(f"max_passes must be positive, got {max_passes}")
        if activation_budget_bytes < 1:
            raise ValueError(
                "activation_budget_bytes must be positive, "
                f"got {activation_budget_bytes}"
            )
        if token_chunk_size < 1:
            raise ValueError(
                f"token_chunk_size must be positive, got {token_chunk_size}"
            )
        if inference_context_factory is None:
            from megatron.core.inference.contexts import StaticInferenceContext

            inference_context_factory = StaticInferenceContext
        if multi_step_mask_helper is None:
            from modelopt.torch.speculative.plugins.megatron_eagle import (
                set_multi_step_attention_mask,
            )

            multi_step_mask_helper = set_multi_step_attention_mask
        self.max_passes = max_passes
        self.activation_budget_bytes = activation_budget_bytes
        self.token_chunk_size = token_chunk_size
        self._inference_context_factory = inference_context_factory
        self._multi_step_mask_helper = multi_step_mask_helper

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

    def forward(
        self,
        *,
        eagle_module: _EagleModule,
        project_logits: LogitProjector,
        target_trunk_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Tensor | None,
        pass_count: int,
        rotary_provider: RotaryProvider,
    ) -> EagleTTTOutput:
        """Run sequential public-API EAGLE passes after pre-allocation validation."""
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
        if attention_mask is not None and attention_mask.shape[0] != storage.batch_size:
            raise ValueError("attention_mask batch size must match target trunk states")

        context = self._inference_context_factory(
            max_batch_size=storage.batch_size,
            max_sequence_length=storage.sequence_length * pass_count,
        )
        current_hidden = target_trunk_states
        logits_by_pass: list[Tensor] = []
        branch_states: list[Tensor] = []
        plans: list[EagleTTTAttentionPlan] = []
        for pass_index in range(pass_count):
            plan = EagleTTTAttentionPlan(
                pass_index=pass_index,
                pass_count=pass_count,
                max_passes=self.max_passes,
                sequence_length=storage.sequence_length,
            )
            pass_mask = (
                None
                if attention_mask is None
                else self._multi_step_mask_helper(attention_mask, pass_index)
            )
            rotary_pos_emb = rotary_provider(
                storage.sequence_length,
                offset=pass_index,
            )
            hidden_states, next_hidden_states = eagle_module(
                embeddings=input_embeds,
                hidden_states=current_hidden,
                attention_mask=pass_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_context=context,
            )
            if hidden_states.shape != target_trunk_states.shape:
                raise ValueError(
                    "EagleModule.forward hidden output must match target trunk shape, "
                    f"got {hidden_states.shape} and {target_trunk_states.shape}"
                )
            if next_hidden_states.shape != target_trunk_states.shape:
                raise ValueError(
                    "EagleModule.forward branch output must match target trunk shape, "
                    f"got {next_hidden_states.shape} and {target_trunk_states.shape}"
                )
            projected = project_logits(hidden_states)
            if projected.ndim != 3 or projected.shape[:2] != hidden_states.shape[:2]:
                raise ValueError(
                    "project_logits must return [sequence, batch, vocabulary], "
                    f"got {projected.shape}"
                )
            logits_by_pass.append(projected.transpose(0, 1).contiguous())
            shifted_branch = torch.cat(
                (
                    torch.zeros_like(next_hidden_states[:1]),
                    next_hidden_states[:-1],
                ),
                dim=0,
            )
            branch_states.append(shifted_branch)
            plans.append(plan)
            current_hidden = shifted_branch
            context.sequence_len_offset += storage.sequence_length

        return EagleTTTOutput(
            pass_logits=tuple(logits_by_pass),
            branch_states=tuple(branch_states),
            plans=tuple(plans),
        )

    def loss_stats(
        self,
        *,
        output: EagleTTTOutput,
        teacher_logits: Tensor,
        valid_mask: Tensor,
        pass_weights: Tensor,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        """Return additive soft-CE numerator and count bins for every pass."""
        pass_count = len(output.pass_logits)
        if pass_count < 1 or len(output.plans) != pass_count:
            raise ValueError("output must contain one plan for every pass logit")
        if teacher_logits.ndim != 3:
            raise ValueError(
                "teacher_logits must have [batch, sequence, vocabulary] shape"
            )
        if valid_mask.shape != teacher_logits.shape[:-1]:
            raise ValueError(
                "valid_mask must match teacher batch and sequence dimensions"
            )
        if pass_weights.shape != (pass_count,):
            raise ValueError(
                f"pass_weights must have shape ({pass_count},), got {pass_weights.shape}"
            )
        if not (teacher_logits.device == valid_mask.device == pass_weights.device):
            raise ValueError(
                "teacher_logits, valid_mask, and pass_weights must share a device"
            )

        numerators: list[Tensor] = []
        counts: list[Tensor] = []
        for logits, plan in zip(output.pass_logits, output.plans, strict=True):
            if logits.shape != teacher_logits.shape:
                raise ValueError(
                    "student and teacher logits must match before pass alignment, "
                    f"got {logits.shape} and {teacher_logits.shape}"
                )
            offset = plan.teacher_offset
            if offset >= teacher_logits.shape[1]:
                raise ValueError(
                    f"pass {plan.pass_index} has no valid teacher rows for "
                    f"sequence length {teacher_logits.shape[1]}"
                )
            pass_stats = streaming_vocab_parallel_soft_ce(
                student_logits=logits[:, :-offset],
                teacher_logits=teacher_logits[:, offset:],
                mask=valid_mask[:, offset:],
                token_chunk_size=self.token_chunk_size,
                tp_group=tp_group,
            )
            numerators.append(pass_stats.numerators)
            counts.append(pass_stats.counts)
        return DraftLossStats(
            numerators=torch.cat(numerators),
            counts=torch.cat(counts),
            weights=pass_weights.detach().to(dtype=torch.float32),
        )
