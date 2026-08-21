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
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import unwrap_model
import torch
from torch import Tensor

from nemo_rl.algorithms.loss.draft import (
    DraftLossStats,
    dflash_projected_vocab_parallel_soft_ce,
)
from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.draft.block_plan import (
    DFlashBatchPlan,
    DSparkBatchPlan,
    build_dflash_batch_plan,
    build_dspark_batch_plan,
)

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
from nemo_rl.models.megatron.draft.utils import (
    DRAFT_GRAD_NORM_GROUP,
    build_draft_model,
    export_dflash_weights_to_hf,
    export_dspark_heads_to_hf,
    export_eagle_weights_to_hf,
    get_policy_lm_head_weight,
    load_hf_weights_to_dflash,
    load_hf_weights_to_dspark,
    register_draft_grad_norm_group,
)
from nemo_rl.models.policy.draft_config import (
    DFlashDraftConfig,
    DSparkDraftConfig,
    DraftConfig,
    Eagle3DraftConfig,
)

if TYPE_CHECKING:
    from megatron.bridge.models.model_provider import ModelProviderMixin
    from nemo_rl.models.megatron.draft.hidden_capture import CapturedStates


@dataclass(frozen=True, slots=True)
class DFlashForwardOutput:
    """DFlash hidden output paired with the immutable plan used to produce it."""

    hidden: Tensor
    plan: DFlashBatchPlan
    output_weight: Tensor


@dataclass(frozen=True, slots=True)
class DSparkForwardOutput:
    """DSpark hidden states and immutable token-alignment metadata."""

    hidden: Tensor
    plan: DSparkBatchPlan
    output_weight: Tensor
    previous_token_ids: Tensor
    adapter: Any


class DraftTrainingProvider(Protocol):
    """Method-neutral model, objective, plan, and checkpoint training seam."""

    config: DraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the draft model for the policy chunk that owns it."""

    def capture_layer_ids(self) -> tuple[int, ...] | None:
        """Return explicit target layers required by this training method."""

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> DFlashBatchPlan | None:
        """Build a method-specific immutable plan when one is required."""

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> Tensor | None:
        """Return local full-batch objective counts for synchronous training."""

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
    ) -> None:
        """Run method-specific draft forward and attach its transient output."""

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        """Return raw method-specific objective bins."""
        ...

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        """Return logical draft-body weights for later runtime adapters."""
        ...


DraftSpeculator = DraftTrainingProvider


@dataclass(frozen=True)
class Eagle3Speculator:
    """Current EAGLE-3 draft speculator."""

    config: Eagle3DraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the EAGLE-3 draft model with the existing implementation."""
        return build_draft_model(
            model_provider=model_provider,
            draft_config=self.config,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )

    def capture_layer_ids(self) -> tuple[int, ...] | None:
        return (
            tuple(self.config.aux_layer_indices)
            if self.config.aux_layer_indices is not None
            else None
        )

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> None:
        del data, optimizer_step
        return None

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> None:
        del data, optimizer_step
        return None

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
    ) -> None:
        del policy_model, input_ids, optimizer_step
        from megatron.core.parallel_state import get_context_parallel_group
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        if captured_states.inputs_embeds is None:
            raise RuntimeError("EAGLE draft training did not capture target embeddings")
        shifted_input_embeds = roll_tensor(
            captured_states.inputs_embeds,
            shifts=-1,
            dims=0,
            cp_group=get_context_parallel_group(),
        )[0]
        data["student_logits"] = draft_model(
            hidden_states=captured_states.hidden_states,
            input_embeds=shifted_input_embeds,
            attention_mask=attention_mask,
        )

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        loss_fn = DraftCrossEntropyLossFn(
            vocab_parallel_group=vocab_parallel_group,
        )
        loss_input, prepared_data = prepare_fn(
            logits=target_logits,
            data=data,
            loss_fn=loss_fn,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )
        return loss_fn.loss_stats(data=prepared_data, **loss_input)

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        return export_eagle_weights_to_hf(model)


@dataclass(frozen=True)
class DFlashSpeculator:
    """DFlash body provider sharing embeddings and output head with the target."""

    config: DFlashDraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build or load the exact-schema public DFlash body."""
        del policy_model_chunk
        if not self.config.enabled:
            return None
        if model_provider.sequence_parallel:
            raise ValueError(
                "DFlash co-training does not support sequence_parallel=True"
            )
        invalid_taps = [
            layer_id
            for layer_id in self.config.target_hidden_state_layer_ids
            if layer_id >= int(model_provider.num_layers)
        ]
        if invalid_taps:
            raise ValueError(
                "DFlash target hidden-state layer IDs exceed the target model: "
                + ", ".join(str(layer_id) for layer_id in invalid_taps)
            )

        from megatron.core.model_parallel_config import ModelParallelConfig

        parallel_config = ModelParallelConfig(
            tensor_model_parallel_size=model_provider.tensor_model_parallel_size,
            use_cpu_initialization=model_provider.use_cpu_initialization,
            fp16=model_provider.fp16,
            bf16=model_provider.bf16,
            params_dtype=model_provider.params_dtype,
            sequence_parallel=False,
        )
        body = DFlashBody(
            DFlashBodyConfig(
                hidden_size=model_provider.hidden_size,
                intermediate_size=model_provider.ffn_hidden_size,
                num_attention_heads=model_provider.num_attention_heads,
                num_key_value_heads=model_provider.num_query_groups,
                head_dim=model_provider.kv_channels,
                num_hidden_layers=self.config.num_layers,
                num_target_taps=len(self.config.target_hidden_state_layer_ids),
                rope_theta=model_provider.rotary_base,
                rms_norm_eps=model_provider.layernorm_epsilon,
                initializer_range=model_provider.init_method_std,
            ),
            tp_group=getattr(pg_collection, "tp", None),
            parallel_config=parallel_config,
        )
        if self.config.model_name is not None:
            missing_keys, unexpected_keys = load_hf_weights_to_dflash(
                body,
                self.config.model_name,
            )
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "[draft] DFlash checkpoint schema mismatch: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}."
                )
        register_draft_grad_norm_group()
        for parameter in body.parameters():
            parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
        return cast(MegatronModule, body)

    def capture_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.config.target_hidden_state_layer_ids)

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> DFlashBatchPlan:
        required = ("input_ids", "input_lengths", "draft_sample_ids")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "DFlash training requires stable plan inputs: " + ", ".join(missing)
            )
        input_ids = data["input_ids"]
        sequence_positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        )
        token_valid_mask = sequence_positions.unsqueeze(0) < data["input_lengths"].to(
            device=input_ids.device
        ).unsqueeze(1)
        return build_dflash_batch_plan(
            token_valid_mask,
            data["draft_sample_ids"].to(device=input_ids.device, dtype=torch.int64),
            anchors_per_sample=self.config.anchors_per_sample,
            gamma=self.config.gamma,
            optimizer_step=optimizer_step,
            seed=self.config.seed,
        )

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
    ) -> None:
        del input_ids, attention_mask
        if captured_states.hidden_states is None:
            raise RuntimeError("DFlash training did not capture target hidden states")
        if captured_states.inputs_embeds is None:
            raise RuntimeError("DFlash training did not capture target embeddings")
        plan = self.prepare_batch(data, optimizer_step=optimizer_step)
        hidden_size = int(cast(DFlashBody, draft_model).config.hidden_size)
        target_taps = (
            captured_states.hidden_states.detach()
            .unflatten(-1, (len(self.capture_layer_ids()), hidden_size))
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        input_embeddings = captured_states.inputs_embeds.detach().transpose(0, 1)
        anchor_embeddings = input_embeddings[
            plan.sample_rows,
            plan.anchor_positions,
        ].unsqueeze(1)
        policy = unwrap_model(policy_model)
        word_embeddings = policy.embedding.word_embeddings
        mask_ids = torch.full(
            (plan.sample_rows.numel(), self.config.gamma),
            self.config.mask_token_id,
            dtype=torch.int64,
            device=input_embeddings.device,
        )
        with torch.no_grad():
            mask_embeddings = word_embeddings(mask_ids)
        block_embeddings = torch.cat((anchor_embeddings, mask_embeddings), dim=1)
        data["dflash_output"] = DFlashForwardOutput(
            hidden=draft_model(
                target_taps=target_taps,
                block_embeddings=block_embeddings,
                plan=plan,
            ),
            plan=plan,
            output_weight=get_policy_lm_head_weight(policy_model).detach(),
        )

    @staticmethod
    def _loss_mask(
        plan: DFlashBatchPlan,
        data: BatchedDataDict[Any],
    ) -> Tensor:
        loss_mask = plan.loss_mask
        if "token_mask" in data:
            response_mask = data["token_mask"].to(dtype=torch.bool)
            loss_mask = (
                loss_mask
                & response_mask[
                    plan.sample_rows[:, None],
                    plan.label_positions,
                ]
            )
        if "sample_mask" in data:
            loss_mask = (
                loss_mask
                & data["sample_mask"].to(dtype=torch.bool)[plan.sample_rows, None]
            )
        return loss_mask

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> Tensor:
        plan = self.prepare_batch(data, optimizer_step=optimizer_step)
        return self._loss_mask(plan, data)[:, 1:].sum(dim=0, dtype=torch.float32)

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        del prepare_fn, vocab_parallel_rank, context_parallel_group
        output = data.get("dflash_output")
        if not isinstance(output, DFlashForwardOutput):
            raise RuntimeError("DFlash forward output is unavailable")
        plan = output.plan
        loss_mask = self._loss_mask(plan, data)
        return dflash_projected_vocab_parallel_soft_ce(
            draft_hidden=output.hidden,
            output_weight=output.output_weight,
            teacher_logits=target_logits,
            sample_rows=plan.sample_rows,
            label_positions=plan.label_positions,
            loss_mask=loss_mask,
            position_decay=self.config.position_decay,
            token_chunk_size=self.config.vocab_tile_size,
            tp_group=vocab_parallel_group,
        )

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        return export_dflash_weights_to_hf(model)


@dataclass(frozen=True)
class DSparkSpeculator:
    """DSpark provider reusing the DFlash body with Markov/confidence heads."""

    config: DSparkDraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        if not self.config.enabled:
            return None
        body_config = DFlashDraftConfig(
            enabled=True,
            model_name=None,
            loss_weight=self.config.loss_weight,
            gamma=self.config.block_size - 1,
            anchors_per_sample=self.config.anchors_per_sample,
            mask_token_id=self.config.mask_token_id,
            target_hidden_state_layer_ids=self.config.target_hidden_state_layer_ids,
            num_layers=self.config.num_layers,
            seed=self.config.seed,
            vocab_tile_size=self.config.vocab_tile_size,
            optimizer=self.config.optimizer,
        )
        body = DFlashSpeculator(body_config).build_model(
            model_provider=model_provider,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )
        if body is None:
            raise RuntimeError("enabled DSpark provider did not build its body")

        from nemo_rl.models.megatron.draft.dspark_provider import (
            build_dspark_provider,
        )

        tp_group = getattr(pg_collection, "tp", None)
        target_vocab_size = int(model_provider.vocab_size)
        draft_vocab_size = self.config.draft_vocab_size or target_vocab_size
        tp_size = int(model_provider.tensor_model_parallel_size)
        tp_rank = 0
        if tp_group is not None and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(tp_group)
            tp_rank = torch.distributed.get_rank(tp_group)
        if draft_vocab_size % tp_size:
            raise ValueError("DSpark draft vocabulary must be divisible by TP size")
        local_vocab_size = draft_vocab_size // tp_size
        body_weight = next(body.parameters())
        adapter = build_dspark_provider(
            body=body,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=int(model_provider.hidden_size),
            markov_rank=self.config.markov_rank,
            confidence_enabled=self.config.confidence_enabled,
            confidence_with_markov=self.config.confidence_with_markov,
            draft_vocab_start_index=tp_rank * local_vocab_size,
            draft_vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tensor_parallel_group=tp_group,
            device=body_weight.device,
            dtype=body_weight.dtype,
        )
        if self.config.model_name is not None:
            missing_keys, unexpected_keys = load_hf_weights_to_dspark(
                adapter,
                self.config.model_name,
            )
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "[draft] DSpark checkpoint schema mismatch: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}."
                )
        register_draft_grad_norm_group()
        for parameter in adapter.parameters():
            parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
        return cast(MegatronModule, adapter)

    def capture_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.config.target_hidden_state_layer_ids)

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> DSparkBatchPlan:
        required = ("input_ids", "input_lengths", "draft_sample_ids")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "DSpark training requires stable plan inputs: " + ", ".join(missing)
            )
        input_ids = data["input_ids"]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        token_valid_mask = positions.unsqueeze(0) < data["input_lengths"].to(
            device=input_ids.device
        ).unsqueeze(1)
        return build_dspark_batch_plan(
            token_valid_mask,
            data["draft_sample_ids"].to(device=input_ids.device, dtype=torch.int64),
            anchors_per_sample=self.config.anchors_per_sample,
            block_size=self.config.block_size,
            optimizer_step=optimizer_step,
            seed=self.config.seed,
        )

    @staticmethod
    def _loss_mask(
        plan: DSparkBatchPlan,
        data: BatchedDataDict[Any],
    ) -> Tensor:
        loss_mask = plan.loss_mask
        if "token_mask" in data:
            response_mask = data["token_mask"].to(dtype=torch.bool)
            loss_mask = (
                loss_mask
                & response_mask[plan.sample_rows[:, None], plan.label_positions]
            )
        if "sample_mask" in data:
            loss_mask = (
                loss_mask
                & data["sample_mask"].to(dtype=torch.bool)[plan.sample_rows, None]
            )
        return loss_mask

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
    ) -> Tensor:
        plan = self.prepare_batch(data, optimizer_step=optimizer_step)
        return self._loss_mask(plan, data).sum(dim=0, dtype=torch.float32)

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
    ) -> None:
        del attention_mask
        if captured_states.hidden_states is None:
            raise RuntimeError("DSpark training did not capture target hidden states")
        if captured_states.inputs_embeds is None:
            raise RuntimeError("DSpark training did not capture target embeddings")
        plan = self.prepare_batch(data, optimizer_step=optimizer_step)
        adapter = cast(Any, unwrap_model(draft_model))
        hidden_size = int(adapter.body.config.hidden_size)
        target_taps = (
            captured_states.hidden_states.detach()
            .unflatten(-1, (len(self.capture_layer_ids()), hidden_size))
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        input_embeddings = captured_states.inputs_embeds.detach().transpose(0, 1)
        anchor_embeddings = input_embeddings[
            plan.sample_rows,
            plan.anchor_positions,
        ].unsqueeze(1)
        policy = unwrap_model(policy_model)
        mask_ids = torch.full(
            (plan.sample_rows.numel(), self.config.block_size - 1),
            self.config.mask_token_id,
            dtype=torch.int64,
            device=input_embeddings.device,
        )
        with torch.no_grad():
            mask_embeddings = policy.embedding.word_embeddings(mask_ids)
        block_embeddings = torch.cat((anchor_embeddings, mask_embeddings), dim=1)
        data["dspark_output"] = DSparkForwardOutput(
            hidden=adapter.body(
                target_taps=target_taps,
                block_embeddings=block_embeddings,
                plan=plan,
            ),
            plan=plan,
            output_weight=get_policy_lm_head_weight(policy_model).detach(),
            previous_token_ids=input_ids[
                plan.sample_rows[:, None],
                plan.query_positions,
            ],
            adapter=adapter,
        )

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        del prepare_fn, vocab_parallel_rank, context_parallel_group
        output = data.get("dspark_output")
        if not isinstance(output, DSparkForwardOutput):
            raise RuntimeError("DSpark forward output is unavailable")
        plan = output.plan
        selected_target_logits = target_logits[
            plan.sample_rows[:, None],
            plan.query_positions,
        ]
        slot_bins = torch.arange(
            self.config.block_size,
            dtype=torch.int64,
            device=target_logits.device,
        ).expand_as(plan.query_positions)
        slot_weights = torch.exp(
            -torch.arange(
                self.config.block_size,
                dtype=torch.float32,
                device=target_logits.device,
            )
            / self.config.loss_decay_gamma
        )
        stats = output.adapter.objective_stats(
            draft_hidden=output.hidden,
            target_output_weight=output.output_weight,
            target_logits=selected_target_logits,
            previous_token_ids=output.previous_token_ids,
            hard_labels=data["input_ids"][
                plan.sample_rows[:, None], plan.label_positions
            ],
            valid_mask=self._loss_mask(plan, data),
            slot_bins=slot_bins,
            slot_weights=slot_weights,
            loss_weights=(
                self.config.ce_loss_weight,
                self.config.tv_loss_weight,
                self.config.confidence_loss_weight,
            ),
            token_chunk_size=self.config.vocab_tile_size,
            tp_group=vocab_parallel_group,
        )
        return DraftLossStats(
            numerators=stats.combined.numerators,
            counts=stats.combined.counts,
            weights=stats.combined.weights,
        )

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        adapter = cast(Any, unwrap_model(model))
        return [
            *export_dflash_weights_to_hf(adapter.body),
            *export_dspark_heads_to_hf(adapter),
        ]


_SPECULATOR_FACTORIES: dict[
    str,
    type[Eagle3Speculator] | type[DFlashSpeculator] | type[DSparkSpeculator],
] = {
    "eagle3": Eagle3Speculator,
    "dflash": DFlashSpeculator,
    "dspark": DSparkSpeculator,
}


def resolve_draft_speculator(
    config: DraftConfig | None,
) -> DraftTrainingProvider | None:
    """Resolve an enabled draft configuration to its speculator."""
    if config is None or not config.enabled:
        return None
    return cast(
        DraftTrainingProvider,
        _SPECULATOR_FACTORIES[config.speculator_type](config),
    )
