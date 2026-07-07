# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import time
import warnings
from dataclasses import dataclass, fields
from functools import partial
from numbers import Real
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.sft_comparison_metrics import (
    SFTComparisonObservation,
    build_sft_comparison_metrics,
)
from nemo_rl.utils.timer import TimeoutChecker, Timer


@dataclass
class SFTSaveState:
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


@dataclass(frozen=True)
class _SFTValidationResult:
    val_metrics: dict[str, Any]
    timing_metrics: dict[str, Any]
    validation_loss_available: bool


def _initial_sft_save_state() -> SFTSaveState:
    return SFTSaveState(
        epoch=0, step=0, total_steps=0, consumed_samples=0, total_valid_tokens=0
    )


def _build_sft_collate_fn(policy_config: PolicyConfig):
    """Bind the policy CP size so packed-row mismatches fail during collation."""
    megatron_cfg = policy_config.get("megatron_cfg", {})
    context_parallel_size = (
        int(megatron_cfg.get("context_parallel_size", 1))
        if megatron_cfg.get("enabled", False)
        else None
    )
    return partial(
        rl_collate_fn,
        megatron_sft_context_parallel_size=context_parallel_size,
    )


def _iter_timed_batches(dataloader, timer: Timer, timing_label: str = "data_fetch"):
    """Yield batches while measuring the blocking dataloader fetch."""
    iterator = iter(dataloader)
    while True:
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        timer.record_elapsed(timing_label, time.perf_counter() - start)
        yield batch


def _add_e2e_step_timing(timing_metrics: dict[str, float]) -> None:
    """Add a step boundary comparable to Megatron-LM's iteration timer."""
    timing_metrics["e2e_step_time"] = timing_metrics.get(
        "total_step_time", 0.0
    ) + timing_metrics.get("data_fetch", 0.0)


def _optional_float(value: Any) -> float | None:
    """Convert CPU scalar metrics to float without synchronizing CUDA tensors."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if (
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.device.type == "cpu"
    ):
        scalar = value.item()
        if isinstance(scalar, Real) and not isinstance(scalar, bool):
            return float(scalar)
    return None


def _measure_loop_interval(
    previous_boundary: float | None, current_boundary: float
) -> tuple[float, float | None]:
    """Measure consecutive batch-arrival boundaries for matched E2E timing."""
    if previous_boundary is None:
        return current_boundary, None
    return current_boundary, current_boundary - previous_boundary


def _maybe_reorder_megatron_sft_dp_stride(
    batch: BatchedDataDict[Any],
    dp_size: int,
    enabled: bool,
) -> BatchedDataDict[Any]:
    """Match Megatron-LM SFT sampler order before NeMo-RL DP sharding."""
    if not enabled or "packed_cu_seqlens" not in batch:
        return batch

    total_batch_size = batch.size
    if total_batch_size % dp_size != 0:
        raise ValueError(
            f"Cannot apply Megatron SFT DP-strided order: batch size "
            f"{total_batch_size} is not divisible by DP size {dp_size}"
        )

    per_dp_batch_size = total_batch_size // dp_size
    order = [
        mb_idx * dp_size + dp_rank
        for dp_rank in range(dp_size)
        for mb_idx in range(per_dp_batch_size)
    ]
    order_tensor = torch.tensor(order, dtype=torch.long)

    reordered = BatchedDataDict()
    for key, value in batch.items():
        if torch.is_tensor(value):
            reordered[key] = value.index_select(0, order_tensor)
        elif isinstance(value, PackedTensor):
            reordered[key] = value.slice(order)
        else:
            reordered[key] = [value[i] for i in order]
    return reordered


class SFTConfig(BaseModel, extra="allow"):
    max_num_steps: int = 60
    max_num_epochs: int = 1
    val_period: int = 10
    val_batches: int = 8
    val_global_batch_size: int = 32
    val_micro_batch_size: int = 1
    val_at_start: bool = True
    # Run each validation batch separately or submit the full four-batch event once.
    validation_execution_mode: Literal["per_batch", "event_batch"] = "per_batch"
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool = False
    seed: int = 42
    only_unmask_final: bool = False


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    NLLLossFn,
    Logger,
    CheckpointManager,
    SFTSaveState,
    MasterConfig,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    set_seed(master_config.sft.seed)

    # Extract individual configs for easier access
    policy_config = master_config.policy
    data_config = master_config.data
    sft_config = master_config.sft
    logger_config = master_config.logger
    cluster_config = master_config.cluster
    checkpointing_config = master_config.checkpointing

    checkpointing_pretrained = checkpointing_config.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config.model_dump())

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(checkpointing_config)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    loaded_state = checkpointer.load_training_info(last_checkpoint_path)
    if loaded_state is not None:
        # Filter to only known SFTSaveState fields; checkpoints may carry
        # extra keys (e.g. validation metrics from previous runs).
        # Backcompat: checkpoints saved before total_valid_tokens was added.
        loaded_state.setdefault("total_valid_tokens", 0)
        known_fields = {f.name for f in fields(SFTSaveState)}
        sft_save_state = SFTSaveState(
            **{k: v for k, v in loaded_state.items() if k in known_fields}
        )
    else:
        sft_save_state = _initial_sft_save_state()

    # ==========================
    #           Data
    # ==========================
    sft_collate_fn = _build_sft_collate_fn(policy_config)
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=sft_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )

    if last_checkpoint_path is not None:
        load_dataloader_state(train_dataloader, last_checkpoint_path, data_config)

    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=sft_config.val_global_batch_size,
            shuffle=False,
            collate_fn=sft_collate_fn,
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
    else:
        val_dataloader = None

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    num_nodes = cluster_config["num_nodes"]
    segment_size = cluster_config.get("segment_size")
    node_resource_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    cluster = RayVirtualCluster(
        name="sft_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_resource_constraints,
    )
    print(f"  ✓ Ray cluster initialized with {num_nodes} nodes")

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            sft_config.max_num_steps,
            sft_config.max_num_epochs * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters
    # check if tokenizer is a processor (e.g. for VLMs)
    processor = None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        processor = tokenizer
        tokenizer = processor.tokenizer

    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )
    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    loss_fn = NLLLossFn(
        use_fused_linear_logprobs=policy_config["megatron_cfg"]["enabled"]
        and policy_config["megatron_cfg"]["use_fused_linear_logprobs"]
    )
    print("  ✓ Model initialized")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    )


# =======================================================
# Training & Validation
# =======================================================
_EVENT_VALIDATION_BATCH_COUNT = 4
_PACKED_VALIDATION_METADATA_KEYS = {
    "packed_cu_seqlens",
    "packed_cu_seqlens_lengths",
    "packed_max_seqlens",
}


def _validate_packed_validation_metadata(batch: BatchedDataDict[Any]) -> None:
    if "packed_cu_seqlens" not in batch:
        return

    missing_keys = _PACKED_VALIDATION_METADATA_KEYS - batch.keys()
    if missing_keys:
        raise ValueError(
            f"Packed validation batch is missing metadata keys: {sorted(missing_keys)}"
        )

    cu_seqlens = batch["packed_cu_seqlens"]
    cu_seqlens_lengths = batch["packed_cu_seqlens_lengths"]
    packed_max_seqlens = batch["packed_max_seqlens"]
    if not all(
        torch.is_tensor(value)
        for value in (cu_seqlens, cu_seqlens_lengths, packed_max_seqlens)
    ):
        raise ValueError("Packed validation metadata must contain tensors")
    if cu_seqlens.ndim != 2:
        raise ValueError("packed_cu_seqlens must be a 2D tensor")
    if cu_seqlens_lengths.ndim != 1 or packed_max_seqlens.ndim != 1:
        raise ValueError(
            "packed_cu_seqlens_lengths and packed_max_seqlens must be 1D tensors"
        )

    for row_idx in range(batch.size):
        metadata_length = int(cu_seqlens_lengths[row_idx].item())
        if metadata_length < 2 or metadata_length > cu_seqlens.shape[1]:
            raise ValueError(
                "packed_cu_seqlens_lengths contains an out-of-range value at "
                f"row {row_idx}: {metadata_length}"
            )
        row_cu_seqlens = cu_seqlens[row_idx, :metadata_length]
        if int(row_cu_seqlens[0].item()) != 0:
            raise ValueError(f"packed_cu_seqlens row {row_idx} must start at 0")
        if bool((row_cu_seqlens[1:] < row_cu_seqlens[:-1]).any().item()):
            raise ValueError(
                f"packed_cu_seqlens row {row_idx} must be monotonically non-decreasing"
            )
        if int(row_cu_seqlens[-1].item()) != int(
            batch["input_lengths"][row_idx].item()
        ):
            raise ValueError(
                f"packed_cu_seqlens row {row_idx} must end at input_lengths"
            )
        expected_max_seqlen = int(
            (row_cu_seqlens[1:] - row_cu_seqlens[:-1]).max().item()
        )
        if expected_max_seqlen != int(packed_max_seqlens[row_idx].item()):
            raise ValueError(
                f"packed_max_seqlens row {row_idx} is inconsistent with "
                "packed_cu_seqlens"
            )
        if metadata_length < cu_seqlens.shape[1] and not bool(
            (cu_seqlens[row_idx, metadata_length:] == -1).all().item()
        ):
            raise ValueError(
                f"packed_cu_seqlens row {row_idx} must use -1 metadata padding"
            )


def _combine_validation_event_batches(
    batches: Sequence[BatchedDataDict[Any]],
    *,
    global_batch_size: int,
    pad_token_id: int,
) -> BatchedDataDict[Any]:
    if len(batches) != _EVENT_VALIDATION_BATCH_COUNT:
        raise ValueError(
            "event_batch validation requires exactly 4 validation batches; "
            f"collected {len(batches)}"
        )
    if global_batch_size <= 0:
        raise ValueError(
            f"Validation global batch size must be positive, got {global_batch_size}"
        )

    reference_keys = set(batches[0].keys())
    reference_values = batches[0]
    for batch_idx, batch in enumerate(batches):
        if batch.size != global_batch_size:
            raise ValueError(
                f"Validation event batch {batch_idx} has size {batch.size}; "
                f"expected {global_batch_size}"
            )
        if set(batch.keys()) != reference_keys:
            missing_keys = sorted(reference_keys - batch.keys())
            extra_keys = sorted(batch.keys() - reference_keys)
            raise ValueError(
                f"Validation event batch {batch_idx} has inconsistent keys; "
                f"missing={missing_keys}, extra={extra_keys}"
            )

        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != global_batch_size:
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} has "
                        f"leading size {value.shape[0] if value.ndim else 0}; "
                        f"expected {global_batch_size}"
                    )
                reference_value = reference_values[key]
                if not torch.is_tensor(reference_value):
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} has "
                        "an inconsistent value type"
                    )
                if value.dtype != reference_value.dtype:
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} dtype "
                        f"{value.dtype} does not match {reference_value.dtype}"
                    )
                if value.device != reference_value.device:
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} device "
                        f"{value.device} does not match {reference_value.device}"
                    )
                if value.ndim != reference_value.ndim:
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} rank "
                        f"{value.ndim} does not match {reference_value.ndim}"
                    )
                if (
                    key != "packed_cu_seqlens"
                    and value.shape[1:] != reference_value.shape[1:]
                ):
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} shape "
                        f"{tuple(value.shape[1:])} does not match "
                        f"{tuple(reference_value.shape[1:])}"
                    )
            elif isinstance(value, PackedTensor):
                if len(value) != global_batch_size or not isinstance(
                    reference_values[key], PackedTensor
                ):
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} has "
                        "inconsistent packed metadata"
                    )
                if value.dim_to_pack != reference_values[key].dim_to_pack:
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} has "
                        "an inconsistent packed dimension"
                    )
            elif isinstance(value, list):
                if len(value) != global_batch_size or not isinstance(
                    reference_values[key], list
                ):
                    raise ValueError(
                        f"Validation event batch {batch_idx} key {key!r} has "
                        "inconsistent list metadata"
                    )
            else:
                raise ValueError(
                    f"Validation event batch {batch_idx} key {key!r} has "
                    f"unsupported type {type(value)}"
                )

        _validate_packed_validation_metadata(batch)

    combined = BatchedDataDict.from_batches(
        batches,
        pad_value_dict={
            "input_ids": pad_token_id,
            "packed_cu_seqlens": -1,
        },
    )
    expected_size = _EVENT_VALIDATION_BATCH_COUNT * global_batch_size
    if combined.size != expected_size:
        raise ValueError(
            f"Combined validation event has size {combined.size}; expected "
            f"{expected_size} for 4 global batches"
        )
    for key, value in combined.items():
        if torch.is_tensor(value) and value.shape[0] != expected_size:
            raise ValueError(
                f"Combined validation event key {key!r} has leading size "
                f"{value.shape[0]}; expected {expected_size}"
            )
        if isinstance(value, (list, PackedTensor)) and len(value) != expected_size:
            raise ValueError(
                f"Combined validation event key {key!r} has leading size "
                f"{len(value)}; expected {expected_size}"
            )
    return combined


def _event_validation_losses(
    val_results: dict[str, Any], expected_count: int, *, megatron_backend: bool
) -> torch.Tensor:
    losses = val_results["loss"]
    if not torch.is_tensor(losses):
        losses = torch.as_tensor(losses)
    losses = losses.reshape(-1)
    if losses.numel() != expected_count:
        raise ValueError(
            f"Event validation returned {losses.numel()} global-batch losses; "
            f"expected {expected_count}"
        )
    if megatron_backend:
        # Megatron normalizes each loss by num_global_batches inside one train call.
        losses = losses * expected_count
    return losses


def _validate_with_loss_availability(
    policy: PolicyInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    comparison_instrumentation_enabled: bool = False,
) -> _SFTValidationResult:
    """Run validation and retain whether the reported loss was measured."""
    if val_dataloader is None:
        assert master_config.sft.val_period <= 0, (
            "val_dataloader is None, so sft.val_period must be <= 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return _SFTValidationResult({}, {}, False)

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        val_metrics = {"val_loss": 0.0}
        sum_num_valid_tokens = 0
        validation_execution_mode = master_config.sft.validation_execution_mode
        event_batches: list[BatchedDataDict[Any]] = []
        event_num_valid_tokens: list[torch.Tensor] = []

        policy.prepare_for_training()
        validation_batches = (
            _iter_timed_batches(
                val_dataloader,
                timer,
                timing_label="data_fetch_s",
            )
            if comparison_instrumentation_enabled
            else val_dataloader
        )
        for batch_idx, val_batch in enumerate(validation_batches):
            data_processing_start = (
                time.perf_counter() if comparison_instrumentation_enabled else None
            )
            if "packed_cu_seqlens" in val_batch:
                val_data = val_batch
            else:
                ## add loss mask based on role to every message
                add_loss_mask_to_message_log(
                    val_batch["message_log"],
                    roles_to_train_on=["assistant"],
                    only_unmask_final=master_config.sft.only_unmask_final,
                )

                cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                    val_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config.policy[
                        "make_sequence_length_divisible_by"
                    ],
                )

                val_data: BatchedDataDict = BatchedDataDict(
                    {
                        "input_ids": cat_and_padded["token_ids"],
                        "input_lengths": input_lengths,
                        "token_mask": cat_and_padded["token_loss_mask"],
                        "sample_mask": val_batch["loss_multiplier"],
                    }
                )

                # update multimodal data
                val_data.update(cat_and_padded.get_multimodal_dict(as_tensors=False))
            # When running validation with drop_last=False, we might end up with a partial batch.
            # Check if we need to pad the final batch to make it divisible by micro_batch_size * dp_size.
            if val_data.size < val_batch_size:
                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                val_data = maybe_pad_last_batch(val_data, dp_size, val_mbs)
            if data_processing_start is not None:
                timer.record_elapsed(
                    "data_processing_s",
                    time.perf_counter() - data_processing_start,
                )

            timing_kwargs = (
                {"timer": timer} if comparison_instrumentation_enabled else {}
            )
            if validation_execution_mode == "event_batch":
                event_batches.append(val_data)
                event_num_valid_tokens.append(
                    (
                        val_data["sample_mask"].unsqueeze(-1) * val_data["token_mask"]
                    ).sum()
                )
            else:
                ## just run model fwd
                val_results = policy.train(
                    val_data,
                    loss_fn,
                    eval_mode=True,
                    gbs=val_data.size,
                    mbs=val_mbs,
                    **timing_kwargs,
                )
                if comparison_instrumentation_enabled:
                    for name, elapsed in val_results.get(
                        "evaluation_timings", {}
                    ).items():
                        timer.record_elapsed(name, float(elapsed))

                if len(val_results["all_mb_metrics"]) == 0:
                    warnings.warn(
                        "No validation metrics were collected for this batch."
                        " This is likely because there were no valid samples."
                    )
                else:
                    num_valid_tokens = (
                        val_data["sample_mask"].unsqueeze(-1) * val_data["token_mask"]
                    ).sum()
                    val_metrics["val_loss"] += (
                        float(val_results["loss"]) * num_valid_tokens
                    )
                    sum_num_valid_tokens += num_valid_tokens

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if validation_execution_mode == "event_batch":
            combined_val_data = _combine_validation_event_batches(
                event_batches,
                global_batch_size=val_batch_size,
                pad_token_id=tokenizer.pad_token_id,
            )
            val_results = policy.train(
                combined_val_data,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size,
                mbs=val_mbs,
                **timing_kwargs,
            )
            if comparison_instrumentation_enabled:
                for name, elapsed in val_results.get("evaluation_timings", {}).items():
                    timer.record_elapsed(name, float(elapsed))

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                megatron_backend = (
                    "megatron_cfg" in master_config.policy
                    and master_config.policy["megatron_cfg"]["enabled"]
                )
                losses = _event_validation_losses(
                    val_results,
                    _EVENT_VALIDATION_BATCH_COUNT,
                    megatron_backend=megatron_backend,
                )
                for loss, num_valid_tokens in zip(losses, event_num_valid_tokens):
                    if num_valid_tokens > 0:
                        val_metrics["val_loss"] += float(loss) * num_valid_tokens
                        sum_num_valid_tokens += num_valid_tokens

        validation_loss_available = bool(sum_num_valid_tokens > 0)
        if validation_loss_available:
            val_metrics["val_loss"] /= sum_num_valid_tokens
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if validation_loss_available:
        # Print summary of validation results
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")

        # Print timing information
        print("\n  ⏱️  Validation Timing:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return _SFTValidationResult(
        val_metrics=val_metrics,
        timing_metrics=timing_metrics,
        validation_loss_available=validation_loss_available,
    )


def validate(
    policy: PolicyInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    comparison_instrumentation_enabled: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation and return the public metrics/timings two-tuple."""
    result = _validate_with_loss_availability(
        policy,
        val_dataloader,
        tokenizer,
        loss_fn,
        step,
        master_config,
        val_batches,
        val_batch_size,
        val_mbs,
        comparison_instrumentation_enabled,
    )
    return result.val_metrics, result.timing_metrics


def sft_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    sft_save_state: SFTSaveState,
) -> None:
    # Run basic sft training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    current_epoch = sft_save_state.epoch
    current_step = sft_save_state.step
    total_steps = sft_save_state.total_steps
    total_valid_tokens = sft_save_state.total_valid_tokens
    previous_loop_boundary: float | None = None

    sft_config = master_config.sft
    # Validation configuration
    val_period = sft_config.val_period
    val_at_start = sft_config.val_at_start
    val_at_end = sft_config.val_at_end
    max_num_epochs = sft_config.max_num_epochs
    megatron_sft_dp_stride_order = master_config.data.get(
        "megatron_sft_dp_stride_order", False
    )

    if logger.comparison_metrics_enabled:
        logger.define_metric("comparison/step")
        logger.define_metric("performance/*", step_metric="comparison/step")
        logger.define_metric("accuracy/*", step_metric="comparison/step")
        logger.define_metric("context/*", step_metric="comparison/step")

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        validation_result = _validate_with_loss_availability(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=sft_config.val_batches,
            val_batch_size=sft_config.val_global_batch_size,
            val_mbs=sft_config.val_micro_batch_size,
            comparison_instrumentation_enabled=logger.comparison_metrics_enabled,
        )
        val_metrics = validation_result.val_metrics
        validation_timings = validation_result.timing_metrics

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs and total_steps < master_config.sft.max_num_steps
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in _iter_timed_batches(train_dataloader, timer):
            previous_loop_boundary, loop_interval_time = _measure_loop_interval(
                previous_loop_boundary, time.perf_counter()
            )
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config.sft.max_num_steps)} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None
            validation_loss_available = False

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    if "packed_cu_seqlens" in batch:
                        train_data = batch
                    else:
                        ## add loss mask based on role to every message
                        add_loss_mask_to_message_log(
                            batch["message_log"],
                            roles_to_train_on=["assistant"],
                            only_unmask_final=master_config.sft.only_unmask_final,
                        )

                        cat_and_padded, input_lengths = (
                            batched_message_log_to_flat_message(
                                batch["message_log"],
                                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                                make_sequence_length_divisible_by=master_config.policy[
                                    "make_sequence_length_divisible_by"
                                ],
                            )
                        )

                        train_data: BatchedDataDict = BatchedDataDict(
                            {
                                "input_ids": cat_and_padded["token_ids"],
                                "input_lengths": input_lengths,
                                "token_mask": cat_and_padded["token_loss_mask"],
                                "sample_mask": batch["loss_multiplier"],
                            }
                        )
                        train_data.update(
                            cat_and_padded.get_multimodal_dict(as_tensors=False)
                        )

                    dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                    train_data = _maybe_reorder_megatron_sft_dp_stride(
                        train_data,
                        dp_size,
                        megatron_sft_dp_stride_order,
                    )

                print("▶ Taking a training step...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = total_steps + 1 >= master_config.sft.max_num_steps or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step or last step with val_at_end
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    validation_result = _validate_with_loss_availability(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=sft_config.val_batches,
                        val_batch_size=sft_config.val_global_batch_size,
                        val_mbs=sft_config.val_micro_batch_size,
                        comparison_instrumentation_enabled=logger.comparison_metrics_enabled,
                    )
                    val_metrics = validation_result.val_metrics
                    validation_timings = validation_result.timing_metrics
                    validation_loss_available = (
                        validation_result.validation_loss_available
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                ## Checkpointing
                sft_save_state.consumed_samples += master_config.policy[
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    sft_save_state.step = (current_step + 1) % len(train_dataloader)
                    sft_save_state.total_steps = total_steps + 1
                    sft_save_state.epoch = current_epoch
                    sft_save_state.total_valid_tokens = total_valid_tokens

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_loss --> 'val:val_loss'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if hasattr(sft_save_state, full_metric_name):
                                delattr(sft_save_state, full_metric_name)
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            setattr(
                                sft_save_state,
                                full_metric_name,
                                metrics_source[metric_name],
                            )

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, vars(sft_save_state), master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            _add_e2e_step_timing(timing_metrics)
            if loop_interval_time is not None:
                timing_metrics["loop_interval_time"] = loop_interval_time

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops
            print("\n⏱️  Timing:")
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            e2e_time = timing_metrics.get("e2e_step_time", total_time)
            print(f"  • E2E step time including data fetch: {e2e_time:.2f}s")
            matched_e2e_time = timing_metrics.get("loop_interval_time", e2e_time)
            if loop_interval_time is not None:
                print(f"  • Matched loop interval time: {matched_e2e_time:.2f}s")

            # Display all other timing metrics (if any)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k not in {
                    "total_step_time",
                    "e2e_step_time",
                    "loop_interval_time",
                }:
                    percent = (v / e2e_time * 100) if e2e_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            if matched_e2e_time > 0:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics.get("global_valid_toks", 0)
                    / matched_e2e_time
                    / total_num_gpus
                )
            else:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = 0.0
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")
            if logger.comparison_metrics_enabled:
                comparison_metrics = build_sft_comparison_metrics(
                    SFTComparisonObservation(
                        step=total_steps + 1,
                        train_step_time_s=_optional_float(
                            timing_metrics.get("policy_training")
                        ),
                        e2e_step_time_s=_optional_float(
                            timing_metrics.get("e2e_step_time")
                        ),
                        validation_time_s=_optional_float(
                            validation_timings.get("total_validation_time")
                            if validation_timings is not None
                            else None,
                        ),
                        main_lm_loss=_optional_float(metrics.get("loss")),
                        validation_loss=(
                            _optional_float(val_metrics.get("val_loss"))
                            if val_metrics is not None and validation_loss_available
                            else None
                        ),
                        grad_norm=_optional_float(metrics.get("grad_norm")),
                        learning_rate=_optional_float(metrics.get("lr")),
                    )
                )
                logger.log_metrics(
                    comparison_metrics,
                    total_steps + 1,
                    step_metric="comparison/step",
                    step_finished=True,
                )

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= master_config.sft.max_num_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
