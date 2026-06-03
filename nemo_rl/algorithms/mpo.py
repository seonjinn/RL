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
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import NotRequired, Optional, TypedDict, cast

# Set NRL_DEBUG=1 to enable [DEBUG] prints
_DEBUG = os.environ.get("NRL_DEBUG", "0") == "1"

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    MPOLossFn,
)
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import preference_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class MPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    consumed_samples: int


def _default_mpo_save_state() -> MPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class MPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    preference_loss_weight: float
    sft_loss_weight: float
    bco_loss_weight: float
    quality_average_log_probs: bool
    reward_shift_momentum: float
    reward_shift: float


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    mpo: MPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


class MPOValMetrics(TypedDict):
    loss: float
    sft_loss: float
    preference_loss: float
    bco_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float
    global_valid_seqs: float
    global_valid_toks: float
    mean_token_logprob: NotRequired[float]
    mean_ref_logprob: NotRequired[float]
    mean_abs_logratio_per_token: NotRequired[float]


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    dict[str, StatefulDataLoader],
    MPOLossFn,
    Logger,
    CheckpointManager,
    MPOSaveState,
    MasterConfig,
]:
    """Main entry point for running MPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    # Make sure we are not using dynamic batching or sequence packing.
    # Anything that changes the order of data within a batch is currently incompatible with MPO.
    assert not master_config["policy"]["dynamic_batching"]["enabled"], (
        "Dynamic batching is currently not supported with MPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )
    assert not master_config["policy"]["sequence_packing"]["enabled"], (
        "Sequence packing is currently not supported with MPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )

    set_seed(master_config["mpo"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    mpo_config = master_config["mpo"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    mpo_save_state: Optional[MPOSaveState] = cast(
        Optional[MPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    # ==========================
    #           Data
    # ==========================
    ## TODO(@ashors) reduce boilerplate and move reused code into utils
    num_dataloader_workers = int(os.environ.get("NRL_DATALOADER_WORKERS", "0"))
    deduplicate_multimodal = mpo_config.get("deduplicate_multimodal_data", False)
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=partial(
            preference_collate_fn,
            tokenizer=tokenizer,
            make_sequence_length_divisible_by=policy_config[
                "make_sequence_length_divisible_by"
            ],
            add_loss_mask=True,
            deduplicate_multimodal_data=deduplicate_multimodal,
        ),
        drop_last=True,
        num_workers=num_dataloader_workers,
        prefetch_factor=2 if num_dataloader_workers > 0 else None,
        persistent_workers=num_dataloader_workers > 0,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = {
        k: StatefulDataLoader(
            v,
            batch_size=mpo_config["val_global_batch_size"],
            shuffle=False,
            collate_fn=partial(
                preference_collate_fn,
                tokenizer=tokenizer,
                make_sequence_length_divisible_by=policy_config[
                    "make_sequence_length_divisible_by"
                ],
                add_loss_mask=True,
                deduplicate_multimodal_data=deduplicate_multimodal,
            ),
            drop_last=False,
            num_workers=num_dataloader_workers,
            prefetch_factor=2 if num_dataloader_workers > 0 else None,
            persistent_workers=num_dataloader_workers > 0,
        )
        for k, v in val_dataset.items()
    }

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="mpo_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            mpo_config["max_num_steps"],
            mpo_config["max_num_epochs"] * len(train_dataloader),
        )
        ## NOTE: we double the train_iters because effective batch size is doubled
        ## for (chosen, rejected) pairs
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters * 2
        if "scheduler" in policy_config["megatron_cfg"]:
            for k in policy_config["megatron_cfg"]["scheduler"]:
                if "iters" in k:
                    policy_config["megatron_cfg"]["scheduler"][k] *= 2
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=True,
    )
    loss_fn = MPOLossFn(master_config["mpo"])
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
        mpo_save_state,
        master_config,
    )


def add_ref_logprobs_to_data(dataloader, policy, master_config, is_val=False, tokenizer=None):
    dataloader_iter = iter(dataloader)

    max_seq = master_config["policy"].get("max_total_sequence_length")
    _image_token_id = None
    _img_start_token_id = None
    _pad_token_id = 0
    _temporal_patch_size = 1
    if tokenizer is not None:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            try:
                _image_token_id = convert("<image>")
            except Exception:
                pass
            try:
                _img_start_token_id = convert("<img>")
            except Exception:
                pass
        _pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    _data_cfg = master_config.get("data", {})
    _temporal_patch_size = _data_cfg.get("video_temporal_patch_size", 1)
    if _temporal_patch_size == 1:
        _vision_cfg = getattr(getattr(tokenizer, "config", None), "vision_config", None)
        _temporal_patch_size = getattr(_vision_cfg, "video_temporal_patch_size", 1)

    while True:
        try:
            batch = next(dataloader_iter)

            micro_batch_size = (
                master_config["mpo"]["val_micro_batch_size"] * 2
                if is_val
                else master_config["policy"]["train_micro_batch_size"] * 2
            )

            # when running validation with drop_last=False, we might end up with a partial batch.
            # In this case, we pad the batch to the next multiple of micro_batch_size * dp_size.
            dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
            if batch.size % (dp_size * micro_batch_size) != 0:
                assert is_val, (
                    "Partial batches should only happen during validation, but got a partial batch during training."
                )
                batch = maybe_pad_last_batch(batch, dp_size, micro_batch_size)

            # Truncate sequences whose expanded (post-vision) length exceeds the
            # budget.  Without this, large images cause CUDA OOM during the
            # Megatron forward pass.
            if (
                max_seq is not None
                and _image_token_id is not None
                and "imgs_sizes" in batch
            ):
                from nemo_rl.models.megatron.multimodal import truncate_for_expanded_budget
                batch, vision_truncated = truncate_for_expanded_budget(
                    batch,
                    max_seq_length=max_seq,
                    pad_token_id=_pad_token_id,
                    image_token_id=_image_token_id,
                    img_start_token_id=_img_start_token_id,
                    temporal_patch_size=_temporal_patch_size,
                )
                if vision_truncated.any():
                    n_vt = int(vision_truncated.sum().item())
                    print(
                        f"✂ MPO vision-budget truncation: {n_vt}/{len(vision_truncated)} "
                        f"samples truncated to fit expanded budget of {max_seq}",
                        flush=True,
                    )

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=micro_batch_size,
            )["reference_logprobs"]
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: dict[str, StatefulDataLoader],
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    logger: Logger,
):
    val_metrics, validation_timings = {}, {}
    for val_dataset_name, v in val_dataloader.items():
        k_val_metrics, k_validation_timings = validate_one_dataset(
            policy=policy,
            val_dataloader=v,
            loss_fn=loss_fn,
            step=step,
            master_config=master_config,
            val_batches=val_batches,
            val_batch_size=val_batch_size,
            val_mbs=val_mbs,
            dataset_name=val_dataset_name,
            tokenizer=tokenizer,
        )
        prefix = f"validation-{val_dataset_name}"

        logger.log_metrics(k_val_metrics, step, prefix=prefix)
        logger.log_metrics(k_validation_timings, step, prefix=f"timing/{prefix}")

        for metric_name in MPOValMetrics.__annotations__.keys():
            if metric_name in k_val_metrics:
                val_metrics[f"{prefix}_{metric_name}"] = k_val_metrics[metric_name]
        validation_timings[prefix + "_total_validation_time"] = k_validation_timings[
            "total_validation_time"
        ]

    if len(validation_timings) > 0:
        total_validation_time = sum(validation_timings.values())
        logger.log_metrics(
            {"total_validation_time": total_validation_time},
            step,
            prefix="timing/validation",
        )
        validation_timings["total_validation_time"] = total_validation_time

    return val_metrics, validation_timings


def validate_one_dataset(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    dataset_name: str,
    tokenizer=None,
):
    """Run validation on one validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config["mpo"]["val_period"] == 0, (
            "val_dataloader is None, so mpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step} for `{dataset_name}` set..")

        val_metrics = defaultdict(list)
        num_valid_batches = 0
        for batch_idx, val_batch in enumerate(
            add_ref_logprobs_to_data(val_dataloader, policy, master_config, is_val=True, tokenizer=tokenizer)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch.size,
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                for metric_name in MPOValMetrics.__annotations__.keys():
                    if metric_name not in val_results["all_mb_metrics"]:
                        continue
                    reduction = (
                        np.mean
                        if metric_name in {"global_valid_seqs", "global_valid_toks"}
                        else sum
                    )
                    val_metrics[metric_name] += [
                        reduction(val_results["all_mb_metrics"][metric_name])
                    ]

                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if num_valid_batches > 0:
            sum_num_valid_samples = sum(val_metrics["num_valid_samples"])
            global_valid_toks = sum(val_metrics["global_valid_toks"])
            global_valid_seqs = sum(val_metrics["global_valid_seqs"])
            val_metrics = MPOValMetrics(
                num_valid_samples=sum_num_valid_samples,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                **{
                    metric_name: sum(
                        [
                            value * weight
                            for value, weight in zip(
                                val_metrics[metric_name],
                                val_metrics["num_valid_samples"],
                            )
                        ]
                    )
                    / sum_num_valid_samples
                    for metric_name in MPOValMetrics.__annotations__.keys()
                    if metric_name
                    not in {
                        "num_valid_samples",
                        "global_valid_seqs",
                        "global_valid_toks",
                    }
                    and len(val_metrics.get(metric_name, [])) > 0
                },
            )
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )
            val_metrics = MPOValMetrics(
                **{
                    metric_name: 0.0
                    for metric_name in MPOValMetrics.__annotations__.keys()
                }
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if len(val_metrics) == 0:
        warnings.warn(
            "No validation metrics were collected."
            " This is likely because there were no valid samples in the validation set."
        )

    else:
        # Print summary of validation results
        print(f"\n📊 Validation Results for `{dataset_name}` set:")
        for metric_name in MPOValMetrics.__annotations__.keys():
            if metric_name in val_metrics:
                print(f"    • Validation {metric_name}: {val_metrics[metric_name]:.4f}")

        # Print timing information
        print(f"\n  ⏱️  Validation Timing for `{dataset_name}` set:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def _debug_print_mpo_batch(
    batch,
    tokenizer,
    total_steps: int,
    num_pairs_to_show: int = 2,
) -> None:
    """Print chosen/rejected text pairs from an MPO batch when NRL_DEBUG=1.

    The batch interleaves chosen and rejected: [c0, r0, c1, r1, ...].
    Even indices are chosen, odd indices are rejected.
    """
    input_ids = batch["input_ids"]  # (batch_size, seq_len)
    input_lengths = batch["input_lengths"]  # (batch_size,)
    batch_size = input_ids.shape[0]
    num_pairs = batch_size // 2
    n_show = min(num_pairs_to_show, num_pairs)

    print(f"[DEBUG] === MPO Step {total_steps + 1} — Showing {n_show}/{num_pairs} preference pairs ===")
    for p in range(n_show):
        chosen_idx = p * 2
        rejected_idx = p * 2 + 1

        chosen_len = int(input_lengths[chosen_idx].item())
        rejected_len = int(input_lengths[rejected_idx].item())

        chosen_text = tokenizer.decode(
            input_ids[chosen_idx, :chosen_len], skip_special_tokens=False
        )
        rejected_text = tokenizer.decode(
            input_ids[rejected_idx, :rejected_len], skip_special_tokens=False
        )

        # Try to split context (prompt) from completion. The omni/Qwen chat
        # template uses "<|im_start|>assistant\n" as the final assistant turn
        # marker; we use the *last* occurrence so that the context includes any
        # earlier assistant turns.
        marker = "<|im_start|>assistant\n"
        c_split = chosen_text.rfind(marker)
        r_split = rejected_text.rfind(marker)

        if c_split >= 0:
            context_preview = chosen_text[: c_split + len(marker)]
            chosen_completion = chosen_text[c_split + len(marker) :]
        else:
            context_preview = ""
            chosen_completion = chosen_text

        if r_split >= 0:
            rejected_completion = rejected_text[r_split + len(marker) :]
        else:
            rejected_completion = rejected_text

        def _preview_completion(text: str, limit: int = 300) -> str:
            if len(text) <= limit * 2:
                return text.replace("\n", "\\n")
            return (
                text[:limit].replace("\n", "\\n")
                + " ... "
                + text[-limit:].replace("\n", "\\n")
            )

        # Truncate for readability
        ctx = context_preview[-400:].replace("\n", "\\n")
        ch = _preview_completion(chosen_completion)
        rj = _preview_completion(rejected_completion)

        print(f"[DEBUG] --- Pair {p} ---")
        print(f"[DEBUG]   Context (last 400 chars): ...{ctx}")
        print(f"[DEBUG]   Chosen  (len={chosen_len}): {ch}")
        print(f"[DEBUG]   Rejected(len={rejected_len}): {rj}")


def mpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    mpo_save_state: MPOSaveState,
) -> None:
    # Run mpo training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if mpo_save_state is None:
        mpo_save_state = _default_mpo_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = mpo_save_state["epoch"]
        current_step = mpo_save_state["step"]
        total_steps = mpo_save_state["total_steps"]

    mpo_config = master_config["mpo"]
    # Validation configuration
    val_period = mpo_config["val_period"]
    val_at_start = mpo_config["val_at_start"]
    max_num_epochs = mpo_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        validation_result = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=mpo_config["val_batches"],
            val_batch_size=mpo_config["val_global_batch_size"],
            val_mbs=mpo_config["val_micro_batch_size"],
            logger=logger,
        )
        if validation_result is not None:
            val_metrics, validation_timings = validation_result
        else:
            val_metrics, validation_timings = None, None

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["mpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in add_ref_logprobs_to_data(train_dataloader, policy, master_config, tokenizer=tokenizer):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['mpo']['max_num_steps'])} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                print("▶ Taking a training step...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        batch,
                        loss_fn,
                        eval_mode=False,
                        ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                        ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                    )

                is_last_step = total_steps + 1 >= master_config["mpo"][
                    "max_num_steps"
                ] or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    validation_result = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=mpo_config["val_batches"],
                        val_batch_size=mpo_config["val_global_batch_size"],
                        val_mbs=mpo_config["val_micro_batch_size"],
                        logger=logger,
                    )
                    if validation_result is not None:
                        val_metrics, validation_timings = validation_result
                    else:
                        val_metrics, validation_timings = None, None

                ## Checkpointing
                mpo_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    mpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    mpo_save_state["total_steps"] = total_steps + 1
                    mpo_save_state["epoch"] = current_epoch
                    # Remove outdated validation metrics
                    for key in list(mpo_save_state):
                        if (
                            key.startswith("val")
                            and any(
                                [
                                    key.endswith(f"_{metric_name}")
                                    for metric_name in MPOValMetrics.__annotations__.keys()
                                    if metric_name != "num_valid_samples"
                                ]
                            )
                            and (val_metrics is None or key not in val_metrics)
                        ):
                            del mpo_save_state[key]
                    if val_metrics is not None:
                        mpo_save_state.update(val_metrics)

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in mpo_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, mpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # --- [DEBUG] Print chosen/rejected pairs for a few samples -----------
            if _DEBUG:
                _debug_print_mpo_batch(batch, tokenizer, total_steps)

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            for metric_name in MPOValMetrics.__annotations__.keys():
                if metric_name in metrics:
                    print(f"  • {metric_name}: {float(metrics[metric_name]):.4f}")
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

            # Display all other timing metrics (if any)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")
            if "moe_routing_diagnostics" in train_results:
                logger.log_metrics(
                    train_results["moe_routing_diagnostics"],
                    total_steps + 1,
                    prefix="moe_routing",
                )

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                return
            if total_steps >= master_config["mpo"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
