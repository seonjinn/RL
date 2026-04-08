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

"""Two-phase generation with early stopping for zero-variance prompt groups.

Generates a pilot batch (K samples per prompt) first, computes rewards,
then only generates the remaining (G-K) samples for prompts that show
non-zero reward variance. Prompts where all K pilot samples get the same
reward are unlikely to benefit from more samples.

This complements GRESO (which uses historical data to skip prompts
pre-rollout) by detecting zero-variance prompts on their FIRST visit.
"""

from typing import Any, NotRequired, TypedDict

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import GenerationInterface


class EarlyStopGenerationConfig(TypedDict):
    """Configuration for two-phase early stop generation.

    Attributes:
        enabled: Whether early stop generation is active.
        pilot_ratio: Fraction of total generations to use as pilot (0 < r < 1).
            With G=8 and pilot_ratio=0.5, pilot generates 4 samples per prompt.
            Must produce at least 2 pilot samples (pilot_ratio * G >= 2).
        min_pilot_samples: Minimum pilot samples per prompt (default 2).
    """

    enabled: bool
    pilot_ratio: NotRequired[float]
    min_pilot_samples: NotRequired[int]


def compute_pilot_and_remainder(
    num_generations_per_prompt: int,
    config: EarlyStopGenerationConfig,
) -> tuple[int, int]:
    """Compute pilot_k and remainder generation counts.

    Returns:
        (pilot_k, remainder): Number of pilot and remaining samples per prompt.
    """
    pilot_ratio = config.get("pilot_ratio", 0.5)
    min_pilot = max(2, config.get("min_pilot_samples", 2))  # min 2 for valid std()

    pilot_k = max(min_pilot, int(num_generations_per_prompt * pilot_ratio))
    pilot_k = min(pilot_k, num_generations_per_prompt - 1)  # at least 1 remainder
    remainder = num_generations_per_prompt - pilot_k
    return pilot_k, remainder


def two_phase_rollout(
    policy_generation: GenerationInterface,
    batch: BatchedDataDict[DatumSpec],
    tokenizer: PreTrainedTokenizerBase,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    num_generations_per_prompt: int,
    config: EarlyStopGenerationConfig,
    max_rollout_turns: int = 999999,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any], dict[str, float]]:
    """Two-phase generation with early stopping for zero-variance groups.

    Phase 1: Generate pilot_k samples per prompt for all prompts.
    Phase 2: Generate remaining (G - pilot_k) samples only for prompts
             with non-zero reward variance in the pilot.

    For zero-variance prompts, pilot samples are duplicated to fill
    the full G slots, maintaining consistent batch shape for training.

    Args:
        policy_generation: The generation interface.
        batch: Input batch of N prompts (not yet repeated).
        tokenizer: Tokenizer.
        task_to_env: Task-to-environment mapping.
        max_seq_len: Maximum sequence length.
        num_generations_per_prompt: Total G (target generations per prompt).
        config: Early stop configuration.
        max_rollout_turns: Max rollout turns.

    Returns:
        (combined_batch, rollout_metrics, early_stop_metrics):
            combined_batch has N*G samples (same shape as normal generation).
            early_stop_metrics contains filtering statistics.
    """
    N = batch.size
    G = num_generations_per_prompt
    pilot_k, remainder = compute_pilot_and_remainder(G, config)

    # Phase 1: pilot generation
    pilot_repeated = batch.repeat_interleave(pilot_k)
    pilot_result, pilot_rollout_metrics = run_multi_turn_rollout(
        policy_generation=policy_generation,
        input_batch=pilot_repeated,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        greedy=False,
    )

    # Compute per-prompt reward variance from pilot
    pilot_rewards = pilot_result["total_reward"]  # shape: (N * pilot_k,)
    pilot_rewards_grouped = pilot_rewards.reshape(N, pilot_k)
    per_prompt_std = pilot_rewards_grouped.std(dim=1)  # shape: (N,)
    nonzero_mask = per_prompt_std > 0  # prompts worth generating more

    num_nonzero = int(nonzero_mask.sum().item())
    num_zero = N - num_nonzero

    early_stop_metrics: dict[str, float] = {
        "early_stop/num_prompts": float(N),
        "early_stop/pilot_k": float(pilot_k),
        "early_stop/remainder": float(remainder),
        "early_stop/zero_var_prompts": float(num_zero),
        "early_stop/nonzero_var_prompts": float(num_nonzero),
        "early_stop/skip_ratio": float(num_zero) / max(N, 1),
        "early_stop/gen_saved_ratio": float(num_zero * remainder) / max(N * G, 1),
    }

    if num_nonzero == N:
        # All prompts have non-zero variance, generate full remainder for all
        remainder_repeated = batch.repeat_interleave(remainder)
        remainder_result, remainder_metrics = run_multi_turn_rollout(
            policy_generation=policy_generation,
            input_batch=remainder_repeated,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=False,
        )
        combined = _merge_pilot_and_remainder_all(
            pilot_result, remainder_result, N, pilot_k, remainder
        )
        _merge_rollout_metrics(pilot_rollout_metrics, remainder_metrics)
        return combined, pilot_rollout_metrics, early_stop_metrics

    if num_nonzero == 0:
        # All prompts are zero-variance, duplicate pilot samples to fill G slots
        combined = _expand_pilot_to_full(pilot_result, N, pilot_k, G)
        return combined, pilot_rollout_metrics, early_stop_metrics

    # Mixed case: some zero-var, some non-zero-var
    # Phase 2: generate remainder only for non-zero-variance prompts
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
    nonzero_batch = batch.select_indices(nonzero_indices)
    remainder_repeated = nonzero_batch.repeat_interleave(remainder)

    remainder_result, remainder_metrics = run_multi_turn_rollout(
        policy_generation=policy_generation,
        input_batch=remainder_repeated,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        greedy=False,
    )

    # Combine: for each prompt, assemble G samples
    combined = _merge_pilot_and_remainder_mixed(
        pilot_result,
        remainder_result,
        N,
        pilot_k,
        remainder,
        G,
        nonzero_mask,
        nonzero_indices,
    )
    _merge_rollout_metrics(pilot_rollout_metrics, remainder_metrics)
    return combined, pilot_rollout_metrics, early_stop_metrics


def _merge_pilot_and_remainder_all(
    pilot: BatchedDataDict[DatumSpec],
    remainder: BatchedDataDict[DatumSpec],
    N: int,
    pilot_k: int,
    remainder_k: int,
) -> BatchedDataDict[DatumSpec]:
    """Merge when all prompts got both pilot and remainder generations.

    Interleaves pilot and remainder samples so prompt groups are contiguous:
    [p0_pilot0..pilot_k, p0_rem0..rem_k, p1_pilot0..pilot_k, p1_rem0..rem_k, ...]
    """
    G = pilot_k + remainder_k
    # Concatenate pilot and remainder per prompt group
    all_batches = []
    for i in range(N):
        pilot_slice = pilot.slice(i * pilot_k, (i + 1) * pilot_k)
        rem_slice = remainder.slice(i * remainder_k, (i + 1) * remainder_k)
        all_batches.append(pilot_slice)
        all_batches.append(rem_slice)

    return BatchedDataDict.from_batches(all_batches)


def _expand_pilot_to_full(
    pilot: BatchedDataDict[DatumSpec],
    N: int,
    pilot_k: int,
    G: int,
) -> BatchedDataDict[DatumSpec]:
    """Expand pilot-only results to full G per prompt by duplicating samples.

    For zero-variance groups, duplicating is safe since all samples have
    the same reward. Training will compute zero advantage for these.
    """
    all_batches = []
    for i in range(N):
        pilot_group = pilot.slice(i * pilot_k, (i + 1) * pilot_k)
        # Repeat pilot samples to fill G slots
        repeats_needed = G // pilot_k
        extra = G % pilot_k
        parts = [pilot_group] * repeats_needed
        if extra > 0:
            parts.append(pilot_group.slice(0, extra))
        all_batches.extend(parts)

    return BatchedDataDict.from_batches(all_batches)


def _merge_pilot_and_remainder_mixed(
    pilot: BatchedDataDict[DatumSpec],
    remainder: BatchedDataDict[DatumSpec],
    N: int,
    pilot_k: int,
    remainder_k: int,
    G: int,
    nonzero_mask: torch.Tensor,
    nonzero_indices: torch.Tensor,
) -> BatchedDataDict[DatumSpec]:
    """Merge pilot and remainder for mixed zero/non-zero variance case."""
    all_batches = []
    remainder_prompt_idx = 0

    for i in range(N):
        pilot_group = pilot.slice(i * pilot_k, (i + 1) * pilot_k)

        if nonzero_mask[i]:
            # This prompt has non-zero variance: use pilot + remainder
            rem_start = remainder_prompt_idx * remainder_k
            rem_end = rem_start + remainder_k
            rem_group = remainder.slice(rem_start, rem_end)
            all_batches.append(pilot_group)
            all_batches.append(rem_group)
            remainder_prompt_idx += 1
        else:
            # Zero variance: duplicate pilot to fill G slots
            repeats_needed = G // pilot_k
            extra = G % pilot_k
            parts = [pilot_group] * repeats_needed
            if extra > 0:
                parts.append(pilot_group.slice(0, extra))
            all_batches.extend(parts)

    return BatchedDataDict.from_batches(all_batches)


def _merge_rollout_metrics(primary: dict[str, Any], secondary: dict[str, Any]) -> None:
    """Merge secondary rollout metrics into primary (in-place).

    Numeric metrics are summed or averaged as appropriate.
    """
    for k, v in secondary.items():
        if k in primary:
            if isinstance(v, (int, float)) and isinstance(primary[k], (int, float)):
                # Average generation-related metrics
                if "mean" in k or "avg" in k or "fraction" in k:
                    primary[k] = (primary[k] + v) / 2
                else:
                    primary[k] = primary[k] + v
        else:
            primary[k] = v
