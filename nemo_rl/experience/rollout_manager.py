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
    use_buffer: NotRequired[bool]


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


def _compute_mean_response_lengths(
    repeated_batch: BatchedDataDict[DatumSpec],
    total_prompts: int,
    num_generations_per_prompt: int,
) -> torch.Tensor:
    """Compute mean response length per prompt group from message_log.

    Returns a (total_prompts,) tensor of mean assistant token counts.
    """
    message_logs = repeated_batch["message_log"]
    total_rows = total_prompts * num_generations_per_prompt
    assert len(message_logs) == total_rows

    per_sample_lengths = torch.zeros(total_rows, dtype=torch.float32)
    for i, messages in enumerate(message_logs):
        gen_tokens = sum(
            len(msg["token_ids"]) for msg in messages if msg["role"] == "assistant"
        )
        per_sample_lengths[i] = gen_tokens

    # Mean over generations per prompt
    return per_sample_lengths.reshape(total_prompts, num_generations_per_prompt).mean(
        dim=1
    )


class APRILState:
    """Continuation buffer for APRIL Phase 2.

    Stores discarded prompt groups from the previous step so they can be
    prepended to the next step's batch (skip generation, go straight to
    selection).  This amortizes over-provisioning cost across steps.
    """

    def __init__(self) -> None:
        self._buffer: BatchedDataDict[DatumSpec] | None = None
        self._buffer_prompt_count: int = 0
        self._reused_count: int = 0

    @property
    def has_buffer(self) -> bool:
        return self._buffer is not None and self._buffer_prompt_count > 0

    def get_buffered_batch(
        self,
        current_batch: BatchedDataDict[DatumSpec],
        num_generations_per_prompt: int,
    ) -> tuple[BatchedDataDict[DatumSpec], int]:
        """Prepend buffered prompt groups to current batch.

        Returns the merged batch and the number of buffered prompts included.
        """
        if not self.has_buffer:
            return current_batch, 0

        assert self._buffer is not None
        merged = BatchedDataDict.from_batches([self._buffer, current_batch])
        num_buffered = self._buffer_prompt_count
        self._reused_count += num_buffered
        # Clear buffer after use
        self._buffer = None
        self._buffer_prompt_count = 0
        return merged, num_buffered

    def store_discarded(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        selected_prompt_indices: torch.Tensor,
        total_prompts: int,
        num_generations_per_prompt: int,
        max_buffer_prompts: int | None = None,
    ) -> None:
        """Save non-selected prompt groups for reuse in the next step.

        Args:
            max_buffer_prompts: Cap on how many prompt groups to buffer.
                Defaults to num_target_prompts (= selected count) to prevent
                unbounded growth.  Only the first max_buffer_prompts discarded
                groups are kept (by their original order, not score).
        """
        G = num_generations_per_prompt
        discarded_mask = torch.ones(total_prompts, dtype=torch.bool)
        discarded_mask[selected_prompt_indices.cpu()] = False
        discarded_indices = discarded_mask.nonzero(as_tuple=True)[0]

        if discarded_indices.numel() == 0:
            self._buffer = None
            self._buffer_prompt_count = 0
            return

        # Cap buffer size to prevent unbounded growth
        if max_buffer_prompts is None:
            max_buffer_prompts = selected_prompt_indices.numel()
        if discarded_indices.numel() > max_buffer_prompts:
            discarded_indices = discarded_indices[:max_buffer_prompts]

        row_indices = (discarded_indices.unsqueeze(1) * G + torch.arange(G)).reshape(-1)
        self._buffer = repeated_batch.select_indices(row_indices)
        self._buffer_prompt_count = discarded_indices.numel()

    def get_metrics(self) -> dict[str, float]:
        return {
            "april_buffer/buffered_prompts": float(self._buffer_prompt_count),
            "april_buffer/total_reused": float(self._reused_count),
        }


def select_prompt_groups(
    repeated_batch: BatchedDataDict[DatumSpec],
    num_target_prompts: int,
    num_generations_per_prompt: int,
    config: OverProvisioningConfig,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, float], torch.Tensor | None]:
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
        A tuple of (selected_batch, metrics, selected_prompt_indices) where
        selected_batch has exactly ``num_target_prompts * num_generations_per_prompt``
        rows, metrics contains logging information, and selected_prompt_indices
        is the prompt-level indices that were kept (None if no selection needed).
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
        return repeated_batch, {}, None

    selection_metric = config.get("selection_metric", "reward_std")
    rewards = repeated_batch["total_reward"]

    # Reshape rewards to (num_prompts, num_generations_per_prompt)
    prompt_rewards = rewards.reshape(total_prompts, num_generations_per_prompt)

    if selection_metric == "reward_std":
        # Select prompts with highest reward standard deviation
        # (most informative for advantage estimation)
        scores = prompt_rewards.std(dim=1, correction=0)
    elif selection_metric == "response_length":
        # APRIL: select prompts with shortest mean response length.
        # Short completions = already answered, efficient for training.
        # Negate so topk picks shortest.
        response_lens = _compute_mean_response_lengths(
            repeated_batch, total_prompts, num_generations_per_prompt
        )
        scores = -response_lens.to(rewards.device).float()
    elif selection_metric == "random":
        # Use CPU RNG (seeded by grpo.seed via set_seed) so random selection
        # is reproducible; CUDA RNG state is advanced by unrelated kernels.
        scores = torch.rand(total_prompts).to(rewards.device)
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    # Select top-k prompt groups by score
    _, selected_prompt_indices = torch.topk(scores, num_target_prompts, sorted=False)
    selected_prompt_indices = selected_prompt_indices.sort().values

    # Convert prompt-level indices to row-level indices (vectorized, CPU for select_indices)
    G = num_generations_per_prompt
    sel_cpu = selected_prompt_indices.cpu()
    row_indices = (sel_cpu.unsqueeze(1) * G + torch.arange(G)).reshape(-1)

    selected_batch = repeated_batch.select_indices(row_indices)

    # Compute metrics (all on same device as scores)
    num_discarded = total_prompts - num_target_prompts
    selected_stds = scores[selected_prompt_indices]
    discarded_mask = torch.ones(total_prompts, dtype=torch.bool, device=scores.device)
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
    elif selection_metric == "response_length":
        # response_lens already computed above for scoring
        metrics["over_provisioning/selected_mean_response_length"] = (
            response_lens[selected_prompt_indices.cpu()].mean().item()
        )
        metrics["over_provisioning/discarded_mean_response_length"] = (
            response_lens[discarded_mask.cpu()].mean().item()
            if discarded_mask.any()
            else 0.0
        )

    return selected_batch, metrics, selected_prompt_indices
