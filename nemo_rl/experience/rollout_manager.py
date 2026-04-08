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

"""Over-provisioning and selection for generation rollouts.

APRIL Phase 1: request generation for more prompts than needed, then select
the target number of complete prompt groups for training.  Incomplete or
slow groups are discarded (Phase 2 would recycle them).

The over-provisioning is algorithm-agnostic and sits between the dataloader
and the training step.
"""

from typing import NotRequired, TypedDict

import torch

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class OverProvisioningConfig(TypedDict):
    """Configuration for APRIL Phase 1 over-provisioning.

    Attributes:
        enabled: Whether over-provisioning is active.
        ratio: How many times more prompts to request than needed.
            A ratio of 2.0 means request 2x prompts and keep the best N.
        selection_metric: Which metric to use for selecting prompt groups.
            "reward_std" (default) keeps groups with highest reward variance
            (most informative for GRPO-style training).
            "random" selects groups randomly.
    """

    enabled: bool
    ratio: NotRequired[float]
    selection_metric: NotRequired[str]


def get_over_provisioned_prompt_count(
    num_prompts: int,
    config: OverProvisioningConfig,
) -> int:
    """Return the number of prompts to request from the dataloader.

    When over-provisioning is disabled, returns num_prompts unchanged.
    """
    if not config.get("enabled", False):
        return num_prompts
    ratio = config.get("ratio", 2.0)
    return int(num_prompts * ratio)


def select_prompt_groups(
    repeated_batch: BatchedDataDict[DatumSpec],
    num_target_prompts: int,
    num_generations_per_prompt: int,
    config: OverProvisioningConfig,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, float]]:
    """Select the best prompt groups from an over-provisioned batch.

    The batch is expected to be in repeated-interleave format: consecutive
    blocks of ``num_generations_per_prompt`` rows belong to the same prompt.

    Args:
        repeated_batch: The full over-provisioned batch after generation and
            reward computation.  Must contain "total_reward".
        num_target_prompts: How many prompt groups to keep for training.
        num_generations_per_prompt: Number of generations per prompt group.
        config: Over-provisioning configuration.

    Returns:
        A tuple of (selected_batch, metrics) where selected_batch has exactly
        ``num_target_prompts * num_generations_per_prompt`` rows and metrics
        contains logging information about the selection.
    """
    total_rows = repeated_batch.size
    total_prompts = total_rows // num_generations_per_prompt
    assert total_rows % num_generations_per_prompt == 0, (
        f"Batch size {total_rows} not divisible by num_generations_per_prompt {num_generations_per_prompt}"
    )
    assert total_prompts >= num_target_prompts, (
        f"Not enough prompt groups ({total_prompts}) to select {num_target_prompts}"
    )

    if total_prompts == num_target_prompts:
        return repeated_batch, {}

    selection_metric = config.get("selection_metric", "reward_std")
    rewards = repeated_batch["total_reward"]

    # Reshape rewards to (num_prompts, num_generations_per_prompt)
    prompt_rewards = rewards.view(total_prompts, num_generations_per_prompt)

    if selection_metric == "reward_std":
        # Select prompts with highest reward standard deviation
        # (most informative for advantage estimation)
        scores = prompt_rewards.std(dim=1, correction=0)
    elif selection_metric == "random":
        scores = torch.rand(total_prompts)
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    # Select top-k prompt groups by score
    _, selected_prompt_indices = torch.topk(scores, num_target_prompts, sorted=False)
    selected_prompt_indices = selected_prompt_indices.sort().values

    # Convert prompt-level indices to row-level indices
    row_indices = []
    for pi in selected_prompt_indices.tolist():
        start = pi * num_generations_per_prompt
        row_indices.extend(range(start, start + num_generations_per_prompt))

    selected_batch = repeated_batch.select_indices(row_indices)

    # Compute metrics
    num_discarded = total_prompts - num_target_prompts
    selected_stds = scores[selected_prompt_indices]
    discarded_mask = torch.ones(total_prompts, dtype=torch.bool)
    discarded_mask[selected_prompt_indices] = False
    discarded_stds = scores[discarded_mask]

    metrics: dict[str, float] = {
        "over_provisioning/num_provisioned_prompts": float(total_prompts),
        "over_provisioning/num_selected_prompts": float(num_target_prompts),
        "over_provisioning/num_discarded_prompts": float(num_discarded),
        "over_provisioning/ratio": float(total_prompts) / float(num_target_prompts),
    }
    if selection_metric == "reward_std":
        metrics["over_provisioning/selected_mean_reward_std"] = (
            selected_stds.mean().item()
        )
        metrics["over_provisioning/discarded_mean_reward_std"] = (
            discarded_stds.mean().item() if discarded_stds.numel() > 0 else 0.0
        )
        metrics["over_provisioning/selected_mean_reward"] = (
            prompt_rewards[selected_prompt_indices].mean().item()
        )
        metrics["over_provisioning/discarded_mean_reward"] = (
            prompt_rewards[discarded_mask].mean().item()
            if discarded_mask.any()
            else 0.0
        )

    return selected_batch, metrics
