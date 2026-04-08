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

"""Prompt-length aware batching for generation efficiency.

Sorts prompts by tokenized input length before generation so that
similar-length prompts are grouped together. Benefits:

1. Training micro-batches with similar-length sequences have less padding.
2. Similar prompt lengths correlate with similar response lengths,
   enabling more uniform KV cache utilization in vLLM.
3. When combined with sequence packing, pre-sorted sequences pack
   more efficiently.
"""

from typing import NotRequired, TypedDict

import torch

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class LengthAwareBatchingConfig(TypedDict):
    """Configuration for prompt-length aware batching.

    Attributes:
        enabled: Whether length-aware batching is active.
        reverse: If True, sort descending (longest first). Default True.
            Longest-first helps bin-packing and matches vLLM's
            scheduling preference for longer sequences.
    """

    enabled: bool
    reverse: NotRequired[bool]


def sort_batch_by_prompt_length(
    batch: BatchedDataDict[DatumSpec],
    config: LengthAwareBatchingConfig,
) -> tuple[BatchedDataDict[DatumSpec], torch.Tensor, dict[str, float]]:
    """Sort prompts by tokenized input length.

    Args:
        batch: Input batch of N prompts.
        config: Length-aware batching configuration.

    Returns:
        (sorted_batch, sort_indices, metrics):
            sorted_batch: Batch reordered by prompt length.
            sort_indices: Original indices for un-sorting if needed.
            metrics: Batching statistics.
    """
    reverse = config.get("reverse", True)

    # Compute prompt lengths from message_log token_ids
    lengths = []
    for msg_log in batch["message_log"]:
        total_tokens = sum(len(msg["token_ids"]) for msg in msg_log)
        lengths.append(total_tokens)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    # Sort indices
    sort_indices = torch.argsort(lengths_tensor, descending=reverse)
    sorted_batch = batch.select_indices(sort_indices)

    sorted_lengths = lengths_tensor[sort_indices]
    metrics = {
        "length_batching/min_prompt_tokens": float(sorted_lengths.min().item()),
        "length_batching/max_prompt_tokens": float(sorted_lengths.max().item()),
        "length_batching/mean_prompt_tokens": float(
            sorted_lengths.float().mean().item()
        ),
        "length_batching/std_prompt_tokens": float(
            sorted_lengths.float().std(correction=0).item()
        ),
    }

    return sorted_batch, sort_indices, metrics
