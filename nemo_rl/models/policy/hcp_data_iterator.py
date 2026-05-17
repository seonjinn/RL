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

from copy import deepcopy
from typing import Iterator

import torch
from megatron.core import parallel_state

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


_HCP_BATCH_METADATA_KEYS = frozenset(
    {
        "sample_id_groups",
        "shard_sample_ids",
        "sample_sequence_lengths",
        "local_cp_sizes",
    }
)


class HCPGroupIterator:
    """Iterate over lockstep HCP groups as BatchedDataDict microbatches.

    Every HCP rank reconstructs the same global microbatch schedule. Ranks that
    do not participate in a given local-CP subgroup yield a zero-loss dummy
    batch instead of skipping the microbatch. This keeps static Megatron TP/MoE
    collectives in the same order across ranks.
    """

    def __init__(
        self,
        data: BatchedDataDict,
        max_tokens_per_microbatch: int | None = None,
        global_cp_size: int = 1,
        microbatch_budget_multiplier: float = 1.0,
        input_lengths_key: str = "input_lengths",
        sequence_length_pad_multiple: int = 1,
    ):
        if "sample_id_groups" not in data or "shard_sample_ids" not in data:
            raise ValueError(
                "HCPGroupIterator requires 'sample_id_groups' and 'shard_sample_ids' metadata"
            )

        self.data = data
        self.sample_id_groups = data["sample_id_groups"]
        self.shard_sample_ids = data["shard_sample_ids"]
        self.sample_sequence_lengths = data.get("sample_sequence_lengths")
        self.hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.global_cp_size = max(1, int(global_cp_size))
        self.microbatch_budget_multiplier = max(0.0, float(microbatch_budget_multiplier))
        self.input_lengths_key = input_lengths_key
        self.sequence_length_pad_multiple = max(1, sequence_length_pad_multiple)

        self._sample_local_indices = {
            sample_id: local_idx for local_idx, sample_id in enumerate(self.shard_sample_ids)
        }
        self._sample_cache = {
            sample_id: self._slice_local_sample(local_idx)
            for local_idx, sample_id in enumerate(self.shard_sample_ids)
        }
        self._groups = self._build_groups()
        self._idx = 0

    def _slice_local_sample(self, local_idx: int) -> BatchedDataDict:
        sample = self.data.slice(local_idx, local_idx + 1)
        for key in _HCP_BATCH_METADATA_KEYS:
            sample.pop(key, None)
        return sample

    @staticmethod
    def _participant_hdp_ranks(
        round_assignments: list[list[int]], sample_id: int
    ) -> tuple[int, ...]:
        participant_ranks = tuple(
            hdp_rank
            for hdp_rank, rank_samples in enumerate(round_assignments)
            if sample_id in rank_samples
        )
        if not participant_ranks:
            raise RuntimeError(
                f"Sample {sample_id} was not assigned to any HCP rank in the current round"
            )
        return participant_ranks

    def _get_padded_sequence_length(self, sample_id: int) -> int:
        if self.sample_sequence_lengths is not None:
            length = self.sample_sequence_lengths[sample_id]
            seq_len = int(length.item()) if torch.is_tensor(length) else int(length)
            pad_multiple = self.sequence_length_pad_multiple
            return ((seq_len + pad_multiple - 1) // pad_multiple) * pad_multiple

        if self.input_lengths_key not in self.data:
            raise ValueError(
                f"Cannot enforce HCP microbatch token budget because '{self.input_lengths_key}' is missing"
            )

        local_idx = self._sample_local_indices[sample_id]
        length = self.data[self.input_lengths_key][local_idx]
        seq_len = int(length.item()) if torch.is_tensor(length) else int(length)
        pad_multiple = self.sequence_length_pad_multiple
        return ((seq_len + pad_multiple - 1) // pad_multiple) * pad_multiple

    def _effective_max_tokens_per_microbatch(
        self, local_cp_size: int
    ) -> int | None:
        if self.max_tokens_per_microbatch is None:
            return None

        scaled_budget = (
            self.max_tokens_per_microbatch
            * max(1, int(local_cp_size))
            * self.microbatch_budget_multiplier
            / self.global_cp_size
        )
        return max(1, min(self.max_tokens_per_microbatch, int(scaled_budget)))

    def _split_by_token_budget(
        self,
        sample_ids: list[int],
        max_tokens_per_microbatch: int | None = None,
    ) -> list[list[int]]:
        budget = (
            self.max_tokens_per_microbatch
            if max_tokens_per_microbatch is None
            else max_tokens_per_microbatch
        )
        if budget is None:
            return [sample_ids]

        chunks: list[list[int]] = []
        current_chunk: list[int] = []
        current_tokens = 0
        for sample_id in sample_ids:
            sample_tokens = self._get_padded_sequence_length(sample_id)
            if (
                current_chunk
                and current_tokens + sample_tokens > budget
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sample_id)
            current_tokens += sample_tokens

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _make_dummy_batch(self) -> BatchedDataDict:
        dummy = BatchedDataDict()
        optional_mm_tensor_keys = set(BatchedDataDict.ADDITIONAL_OPTIONAL_KEY_TENSORS)
        source_idx = 0 if self.data.size > 0 else None

        for key, value in self.data.items():
            if key in _HCP_BATCH_METADATA_KEYS:
                continue
            if isinstance(value, PackedTensor) or key in optional_mm_tensor_keys:
                continue

            if torch.is_tensor(value):
                if value.shape[0] > 0:
                    row = value[source_idx : source_idx + 1].clone()
                    if torch.is_floating_point(row) or torch.is_complex(row):
                        row.zero_()
                    else:
                        row.fill_(0)
                else:
                    row = torch.zeros(
                        (1, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )

                if key == "input_lengths":
                    row.fill_(1)
                elif key in {"sample_mask", "token_mask"}:
                    row.zero_()
                dummy[key] = row
            elif isinstance(value, list) and value:
                dummy[key] = [deepcopy(value[0])]

        if "input_ids" not in dummy:
            raise RuntimeError("Cannot build HCP dummy batch because input_ids is missing")
        if "input_lengths" not in dummy:
            dummy["input_lengths"] = torch.ones(1, dtype=torch.int32)
        if "sample_mask" in self.data and "sample_mask" not in dummy:
            dummy["sample_mask"] = torch.zeros(1, dtype=torch.float32)

        return dummy

    def _make_group_batch(
        self,
        participant_ranks: tuple[int, ...],
        microbatch_sample_ids: list[int],
    ) -> BatchedDataDict:
        if self.hdp_rank in participant_ranks:
            sample_batches = [
                self._sample_cache[sample_id] for sample_id in microbatch_sample_ids
            ]
            group_batch = (
                sample_batches[0]
                if len(sample_batches) == 1
                else BatchedDataDict.from_batches(sample_batches)
            )
            group_batch["local_cp_size"] = torch.tensor(
                len(participant_ranks), dtype=torch.int32
            )
            group_batch["_hcp_sample_ids"] = microbatch_sample_ids
            group_batch["_hcp_is_dummy"] = False
        else:
            group_batch = self._make_dummy_batch()
            group_batch["local_cp_size"] = torch.tensor(1, dtype=torch.int32)
            group_batch["_hcp_sample_ids"] = []
            group_batch["_hcp_is_dummy"] = True

        group_batch["_hcp_hdp_ranks"] = list(participant_ranks)
        return group_batch

    def _make_coalesced_wave_batch(
        self,
        wave: list[tuple[tuple[int, ...], list[int]]],
    ) -> BatchedDataDict:
        local_entry: tuple[tuple[int, ...], list[int]] | None = None
        for participant_ranks, microbatch_sample_ids in wave:
            if self.hdp_rank not in participant_ranks:
                continue
            if local_entry is not None:
                raise RuntimeError(
                    f"HCP rank {self.hdp_rank} was assigned to multiple local-CP "
                    f"subgroups in one coalesced microbatch"
                )
            local_entry = (participant_ranks, microbatch_sample_ids)

        if local_entry is not None:
            participant_ranks, microbatch_sample_ids = local_entry
            return self._make_group_batch(participant_ranks, microbatch_sample_ids)

        group_batch = self._make_dummy_batch()
        group_batch["local_cp_size"] = torch.tensor(1, dtype=torch.int32)
        group_batch["_hcp_sample_ids"] = []
        group_batch["_hcp_is_dummy"] = True
        group_batch["_hcp_hdp_ranks"] = sorted(
            {
                hdp_rank
                for participant_ranks, _ in wave
                for hdp_rank in participant_ranks
            }
        )
        return group_batch

    @staticmethod
    def _coalesce_disjoint_subgroups(
        subgroup_chunks: list[tuple[tuple[int, ...], list[int]]],
    ) -> list[list[tuple[tuple[int, ...], list[int]]]]:
        waves: list[list[tuple[tuple[int, ...], list[int]]]] = []
        pending = list(subgroup_chunks)
        while pending:
            used_hdp_ranks: set[int] = set()
            wave: list[tuple[tuple[int, ...], list[int]]] = []
            next_pending: list[tuple[tuple[int, ...], list[int]]] = []

            for participant_ranks, microbatch_sample_ids in pending:
                participant_set = set(participant_ranks)
                if participant_set.isdisjoint(used_hdp_ranks):
                    wave.append((participant_ranks, microbatch_sample_ids))
                    used_hdp_ranks.update(participant_set)
                else:
                    next_pending.append((participant_ranks, microbatch_sample_ids))

            if not wave:
                raise RuntimeError("Failed to build a non-empty HCP coalesced wave")

            waves.append(wave)
            pending = next_pending

        return waves

    def _build_groups(self) -> list[BatchedDataDict]:
        groups: list[BatchedDataDict] = []
        for round_assignments in self.sample_id_groups:
            subgroup_to_sample_ids: dict[tuple[int, ...], list[int]] = {}
            assigned_sample_ids = sorted(
                {
                    sample_id
                    for rank_samples in round_assignments
                    for sample_id in rank_samples
                }
            )
            for sample_id in assigned_sample_ids:
                participant_ranks = self._participant_hdp_ranks(
                    round_assignments, sample_id
                )
                subgroup_to_sample_ids.setdefault(participant_ranks, []).append(sample_id)

            subgroup_chunks: list[tuple[tuple[int, ...], list[int]]] = []
            for participant_ranks in sorted(subgroup_to_sample_ids):
                subgroup_sample_ids = subgroup_to_sample_ids[participant_ranks]
                subgroup_token_budget = self._effective_max_tokens_per_microbatch(
                    len(participant_ranks)
                )
                for microbatch_sample_ids in self._split_by_token_budget(
                    subgroup_sample_ids,
                    max_tokens_per_microbatch=subgroup_token_budget,
                ):
                    subgroup_chunks.append((participant_ranks, microbatch_sample_ids))

            for wave in self._coalesce_disjoint_subgroups(subgroup_chunks):
                groups.append(self._make_coalesced_wave_batch(wave))

        if not groups:
            raise RuntimeError(
                f"HCP rank {self.hdp_rank} did not receive any groups to process"
            )
        return groups

    def __iter__(self) -> "HCPGroupIterator":
        return self

    def __len__(self) -> int:
        return len(self._groups)

    def __next__(self) -> BatchedDataDict:
        if self._idx >= len(self._groups):
            raise StopIteration
        group = self._groups[self._idx]
        self._idx += 1
        return group


def iter_hcp_group_batches(
    data: BatchedDataDict,
    max_tokens_per_microbatch: int | None = None,
    global_cp_size: int = 1,
    microbatch_budget_multiplier: float = 1.0,
    input_lengths_key: str = "input_lengths",
    sequence_length_pad_multiple: int = 1,
) -> Iterator[BatchedDataDict]:
    """Return a worker-local iterator over hybrid CP groups."""

    return HCPGroupIterator(
        data,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        global_cp_size=global_cp_size,
        microbatch_budget_multiplier=microbatch_budget_multiplier,
        input_lengths_key=input_lengths_key,
        sequence_length_pad_multiple=sequence_length_pad_multiple,
    )
