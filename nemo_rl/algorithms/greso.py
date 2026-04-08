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

"""GRESO: GRoup Relative Efficiency through Selective rOllouts.

Pre-rollout prompt filtering that skips prompts predicted to have zero-variance
rewards (thus zero gradient contribution). Algorithm-agnostic filter layer
between the dataloader and the training algorithm.

Reference: https://arxiv.org/abs/2506.02177
Reference implementation: https://github.com/Infini-AI-Lab/GRESO/
"""

from __future__ import annotations

import random
from typing import Any, NotRequired, TypedDict

import torch

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class GRESOConfig(TypedDict):
    enabled: bool
    target_zero_variance: NotRequired[float]  # default 0.25
    delta_p: NotRequired[float]  # default 0.01
    p_easy: NotRequired[float]  # initial easy exploration base, default 0.5
    p_hard: NotRequired[float]  # initial hard exploration base, default 0.5
    min_p: NotRequired[float]  # floor for adaptive p, default 0.05
    max_p: NotRequired[float]  # ceiling for adaptive p, default 0.95
    easy_reward_threshold: NotRequired[float]  # avg reward >= this = easy, default 0.98
    hard_reward_threshold: NotRequired[float]  # avg reward <= this = hard, default 0.11


class DataProfiler:
    """Per-prompt reward history tracker and pre-rollout filter.

    Follows the reference implementation from Infini-AI-Lab/GRESO. Tracks
    per-prompt mean reward across epochs and uses consecutive zero-variance
    streaks to compute probabilistic skip decisions.
    """

    def __init__(
        self, easy_threshold: float = 0.98, hard_threshold: float = 0.11
    ) -> None:
        # prompt_id -> list of (step, mean_reward)
        self.data: dict[str, list[tuple[int, float]]] = {}
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    def add_reward(self, step: int, data_id: str, reward: float) -> None:
        if data_id not in self.data:
            self.data[data_id] = []
        self.data[data_id].append((step, reward))

    def add_reward_list(
        self, step: int, data_ids: list[str], rewards: list[float]
    ) -> None:
        for data_id, reward in zip(data_ids, rewards):
            self.add_reward(step, data_id, reward)

    @staticmethod
    def get_data_id_list(indices: list[int]) -> list[str]:
        return [str(idx) for idx in indices]

    def easy_probabilistic_skip(self, data_id: str, p: float) -> bool:
        """Skip prompts that have been consistently easy (all correct)."""
        reward_trace = self.data.get(data_id)
        if not reward_trace:
            return False

        # Count consecutive epochs from the end where avg reward >= easy_threshold
        count = 0
        for i in range(len(reward_trace) - 1, -1, -1):
            if reward_trace[i][1] < self.easy_threshold:
                break
            count += 1

        if count == 0:
            return False

        skip_probability = 1.0 - max(p**count, 0.01)
        return random.random() < skip_probability

    def hard_probabilistic_skip(self, data_id: str, p: float) -> bool:
        """Skip prompts that have been consistently hard (all wrong)."""
        reward_trace = self.data.get(data_id)
        if not reward_trace:
            return False

        # Most recent must also be hard
        if reward_trace[-1][1] > self.hard_threshold:
            return False

        # Count consecutive epochs from the end where avg reward <= hard_threshold
        count = 0
        for i in range(len(reward_trace) - 1, -1, -1):
            if reward_trace[i][1] > self.hard_threshold:
                break
            count += 1

        if count == 0:
            return False

        skip_probability = 1.0 - max(p**count, 0.01)
        return random.random() < skip_probability

    def filter_batch(
        self,
        batch: BatchedDataDict[DatumSpec],
        p_easy: float,
        p_hard: float,
    ) -> tuple[BatchedDataDict[DatumSpec], dict[str, int]]:
        """Pre-rollout filter: probabilistically skip easy/hard prompts.

        Returns:
            (filtered_batch, log_dict) where log_dict has skip counts.
        """
        indices = batch["idx"]
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        data_ids = self.get_data_id_list(indices)

        keep_indices: list[int] = []
        easy_skipped = 0
        hard_skipped = 0

        for i, data_id in enumerate(data_ids):
            if self.easy_probabilistic_skip(data_id, p_easy):
                easy_skipped += 1
                continue
            if self.hard_probabilistic_skip(data_id, p_hard):
                hard_skipped += 1
                continue
            keep_indices.append(i)

        # Never skip everything
        if len(keep_indices) == 0:
            keep_indices = list(range(len(data_ids)))
            easy_skipped = 0
            hard_skipped = 0

        filtered_batch = batch.select_indices(keep_indices)
        log = {
            "greso/easy_skipped": easy_skipped,
            "greso/hard_skipped": hard_skipped,
            "greso/total_skipped": easy_skipped + hard_skipped,
            "greso/kept": len(keep_indices),
            "greso/total_sampled": len(data_ids),
        }
        return filtered_batch, log

    def state_dict(self) -> dict[str, Any]:
        return {"data": dict(self.data)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.data = state["data"]


class GRESOState:
    """Top-level GRESO controller managing the DataProfiler and adaptive rates."""

    def __init__(self, config: GRESOConfig) -> None:
        self.config = config
        self.easy_threshold = config.get("easy_reward_threshold", 0.98)
        self.hard_threshold = config.get("hard_reward_threshold", 0.11)
        self.profiler = DataProfiler(self.easy_threshold, self.hard_threshold)

        self.p_easy = config.get("p_easy", 0.5)
        self.p_hard = config.get("p_hard", 0.5)
        self.min_p = config.get("min_p", 0.05)
        self.max_p = config.get("max_p", 0.95)
        self.delta_p = config.get("delta_p", 0.01)
        self.target_zero_variance = config.get("target_zero_variance", 0.25)
        # Split target: 1/3 easy, 2/3 hard (from reference)
        self.targeted_easy = self.target_zero_variance / 3.0
        self.targeted_hard = self.target_zero_variance * 2.0 / 3.0

        self._step_metrics: dict[str, float] = {}

    def filter_batch(
        self, batch: BatchedDataDict[DatumSpec]
    ) -> tuple[BatchedDataDict[DatumSpec], dict[str, int]]:
        return self.profiler.filter_batch(batch, self.p_easy, self.p_hard)

    def update_rewards(
        self,
        step: int,
        batch: BatchedDataDict[DatumSpec],
        num_generations_per_prompt: int,
    ) -> dict[str, float]:
        """Record per-prompt mean rewards and compute zero-variance stats."""
        G = num_generations_per_prompt
        rewards = batch["total_reward"]
        num_samples = rewards.shape[0]
        num_prompts = num_samples // G

        # Get prompt indices (one per prompt, from every G-th element)
        indices = batch["idx"]
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        prompt_indices = [indices[i * G] for i in range(num_prompts)]
        data_ids = DataProfiler.get_data_id_list(prompt_indices)

        # Compute per-prompt stats
        rewards_grouped = rewards.view(num_prompts, G)
        means = rewards_grouped.mean(dim=1)
        stds = rewards_grouped.std(dim=1, correction=0)

        # Record to profiler
        self.profiler.add_reward_list(step, data_ids, means.tolist())

        # Compute zero-variance breakdown
        easy_count = 0
        hard_count = 0
        zero_var_count = 0
        for i in range(num_prompts):
            is_zero_var = stds[i].item() == 0.0
            if is_zero_var:
                zero_var_count += 1
                avg = means[i].item()
                if avg >= self.easy_threshold:
                    easy_count += 1
                elif avg <= self.hard_threshold:
                    hard_count += 1

        easy_ratio = easy_count / max(num_prompts, 1)
        hard_ratio = hard_count / max(num_prompts, 1)
        zero_var_ratio = zero_var_count / max(num_prompts, 1)

        self._step_metrics = {
            "greso/zero_var_ratio": zero_var_ratio,
            "greso/easy_ratio": easy_ratio,
            "greso/hard_ratio": hard_ratio,
            "greso/zero_var_count": float(zero_var_count),
            "greso/easy_count": float(easy_count),
            "greso/hard_count": float(hard_count),
            "greso/num_prompts": float(num_prompts),
            "greso/tracked_prompts": float(len(self.profiler.data)),
        }
        return self._step_metrics

    def adapt_exploration_rates(self) -> dict[str, float]:
        """Adjust p_easy and p_hard independently based on observed ratios.

        Follows reference: if observed easy ratio >= target easy ratio,
        decrease p_easy (more aggressive filtering). Same for hard.
        """
        easy_ratio = self._step_metrics.get("greso/easy_ratio", 0.0)
        hard_ratio = self._step_metrics.get("greso/hard_ratio", 0.0)

        if easy_ratio >= self.targeted_easy:
            self.p_easy = max(self.p_easy - self.delta_p, self.min_p)
        else:
            self.p_easy = min(self.p_easy + self.delta_p, self.max_p)

        if hard_ratio >= self.targeted_hard:
            self.p_hard = max(self.p_hard - self.delta_p, self.min_p)
        else:
            self.p_hard = min(self.p_hard + self.delta_p, self.max_p)

        rates = {
            "greso/p_easy": self.p_easy,
            "greso/p_hard": self.p_hard,
        }
        self._step_metrics.update(rates)
        return rates

    def get_metrics(self) -> dict[str, float]:
        return dict(self._step_metrics)

    def state_dict(self) -> dict[str, Any]:
        return {
            "p_easy": self.p_easy,
            "p_hard": self.p_hard,
            "profiler": self.profiler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.p_easy = state["p_easy"]
        self.p_hard = state["p_hard"]
        self.profiler.load_state_dict(state["profiler"])


def filter_zero_advantage(
    repeated_batch: BatchedDataDict[DatumSpec],
    num_generations_per_prompt: int,
) -> BatchedDataDict[DatumSpec]:
    """Post-rollout filter: remove prompt groups with zero reward variance.

    This is the GRESO equivalent of DAPO's dynamic_sampling, but simpler:
    just removes zero-advantage groups entirely.
    """
    G = num_generations_per_prompt
    rewards = repeated_batch["total_reward"]
    num_prompts = rewards.shape[0] // G
    prompt_rewards = rewards.view(num_prompts, G)

    # Keep prompts where max != min (non-zero variance)
    non_zero_mask = prompt_rewards.max(dim=1).values != prompt_rewards.min(dim=1).values
    keep_prompt_indices = torch.arange(num_prompts)[non_zero_mask]

    if len(keep_prompt_indices) == 0:
        return repeated_batch

    # Expand to per-response indices
    row_indices = []
    for pi in keep_prompt_indices.tolist():
        start = pi * G
        row_indices.extend(range(start, start + G))

    return repeated_batch.select_indices(row_indices)
