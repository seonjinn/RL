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

from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, List, Tuple

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.models.policy.hybrid_cp_config import HybridCPConfig


class BalancedCPScheduler:
    """Pure-python workload balancer for head-node hybrid CP scheduling."""

    def __init__(self, max_seq_len_per_rank: int, total_hdp_gpus: int):
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.total_hdp_gpus = total_hdp_gpus

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 1
        if self.max_seq_len_per_rank <= 0:
            raise ValueError(
                f"max_seq_len_per_rank must be positive, got {self.max_seq_len_per_rank}"
            )

        required = max(1, 2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)))
        if required > self.total_hdp_gpus:
            raise ValueError(
                f"Sequence length {seq_len} requires local CP size {required}, "
                f"but only {self.total_hdp_gpus} HCP ranks are available. "
                "Increase max_seqlen_per_dp_cp_rank, increase global CP/DP capacity, "
                "or reduce the maximum sequence length."
            )
        return required

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: int | None = None) -> float:
        cp_size = self.gpus_needed(seq_length) if cp_size is None else cp_size
        return (seq_length * seq_length) / cp_size

    def make_buckets_equal(
        self,
        sample_seqlens: List[Tuple[int, int]],
        compute_estimator: Callable[[int], float],
    ) -> List[deque]:
        seqlens = [seq_len for _, seq_len in sample_seqlens]
        k = len({self.gpus_needed(length) for length in seqlens})

        work = []
        for _, seq_len in sample_seqlens:
            cp_size = self.gpus_needed(seq_len)
            work.append(compute_estimator(seq_len, cp_size))

        total_work = sum(work)
        target = total_work / k
        buckets, cur, cur_work = [], [], 0.0
        remaining_k = k

        for idx, (sample_id, seq_len) in enumerate(sample_seqlens):
            projected = cur_work + compute_estimator(seq_len)
            if cur and (
                projected > target * 1.1
                or len(sample_seqlens) - idx <= remaining_k - len(buckets)
            ):
                buckets.append(deque(cur))
                cur, cur_work = [], 0.0
                remaining_k -= 1

            cur.append((sample_id, seq_len))
            cur_work += compute_estimator(seq_len)

        if cur:
            buckets.append(deque(cur))
        return buckets

    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],
        compute_estimator: Callable[[int], float],
        total_gpus: int,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )

        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)
        micro_batches = [[] for _ in range(total_gpus)]
        exec_times = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu = [[] for _ in range(total_gpus)]

        gpu_group_id = [None] * total_gpus
        group_members = {}
        group_size = {}
        next_gid = 0
        prev_needed = None
        check_balance = False

        while buckets:
            sample_seq_tuple = bucket_idx = None
            for idx in range(len(buckets)):
                if not buckets[idx]:
                    continue
                candidate = buckets[idx][0]
                needed = self.gpus_needed(candidate[1])
                candidate_gids = [gid for gid, size in group_size.items() if size == needed]
                free_ranks = [rank for rank, gid in enumerate(gpu_group_id) if gid is None]
                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_idx = candidate, idx
                    break

            if sample_seq_tuple is None:
                break

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if prev_needed is None:
                prev_needed = needed

            candidate_gids = [gid for gid, size in group_size.items() if size == needed]
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (gid, max(exec_times[rank] for rank in group_members[gid]))
                        for gid in candidate_gids
                    ),
                    key=lambda item: item[1],
                )
            else:
                best_gid, best_load = None, float("inf")

            free_ranks = [rank for rank, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda rank: exec_times[rank])
                new_members = free_sorted[:needed]
                new_load = exec_times[new_members[-1]]
                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                chosen_members = group_members[best_gid]

            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid] = needed
                for rank in chosen_members:
                    gpu_group_id[rank] = best_gid

            per_gpu_cost = compute_estimator(seq_len)
            for rank in chosen_members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)

            buckets[bucket_idx].popleft()
            while buckets and not buckets[0]:
                buckets.pop(0)

            next_needed = None
            for bucket in buckets:
                if bucket:
                    next_needed = self.gpus_needed(bucket[0][1])
                    break

            if prev_needed is not None and next_needed is not None and next_needed != prev_needed:
                check_balance = True
            prev_needed = next_needed

            if check_balance:
                non_empty = [value for value in exec_times if value > 0]
                if non_empty:
                    max_exec = max(non_empty)
                    min_exec = min(non_empty)
                    if max_exec > 0 and (max_exec - min_exec) / max_exec <= 0.05:
                        break
                check_balance = False

        leftovers = []
        for bucket in buckets:
            leftovers.extend(list(bucket))

        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    def get_groups_and_subsamples(
        self, sample_id_seqlens: List[Tuple[int, int]]
    ) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
        groups = []
        sample_id_groups = []
        remaining = sorted(sample_id_seqlens, key=lambda item: item[1], reverse=True)
        while remaining:
            mb, remaining, _, sample_ids = self.next_hdp_group(
                remaining, self.get_total_workload, self.total_hdp_gpus
            )
            groups.append(mb)
            if len(sample_ids) < self.total_hdp_gpus:
                sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
            sample_id_groups.append(sample_ids)
        return groups, sample_id_groups


class HeadNodeHCPScheduler:
    """Schedule sequence-packed samples across the DPxCP mesh on the head node."""

    def __init__(
        self,
        hcp_config: HybridCPConfig,
        dp_size: int,
        cp_size: int,
        max_seq_len: int,
    ):
        self.hcp_config = hcp_config
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.max_seq_len = max_seq_len
        self.hdp_size = dp_size * cp_size
        self.max_seqlen_per_dp_cp_rank = (
            hcp_config.max_seqlen_per_dp_cp_rank
            if hcp_config.max_seqlen_per_dp_cp_rank is not None
            else max_seq_len // cp_size
        )
        self.scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
            total_hdp_gpus=self.hdp_size,
        )

    def extract_sequence_lengths(
        self, data: BatchedDataDict, seq_length_key: str = "input_lengths"
    ) -> list[int]:
        if seq_length_key not in data:
            raise ValueError(f"seq_length_key '{seq_length_key}' not found in data")

        seq_lengths = data[seq_length_key]
        if torch.is_tensor(seq_lengths):
            return seq_lengths.cpu().tolist()
        return list(seq_lengths)

    def schedule_samples(self, seq_lengths: list[int]) -> tuple[list, list]:
        if self.hcp_config.force_full_cp:
            return self._schedule_full_cp_samples(seq_lengths)
        sample_id_seqlens = [(idx, seq_len) for idx, seq_len in enumerate(seq_lengths)]
        return self.scheduler.get_groups_and_subsamples(sample_id_seqlens)

    def _schedule_full_cp_samples(self, seq_lengths: list[int]) -> tuple[list, list]:
        """Diagnostic mode: run every sample on the full static CP group."""
        round_assignments = [[] for _ in range(self.hdp_size)]
        groups = [[] for _ in range(self.hdp_size)]

        for sample_id, seq_len in enumerate(seq_lengths):
            dp_rank = sample_id % self.dp_size
            first_hdp_rank = dp_rank * self.cp_size
            for cp_rank in range(self.cp_size):
                hdp_rank = first_hdp_rank + cp_rank
                round_assignments[hdp_rank].append(sample_id)
                groups[hdp_rank].append(seq_len)

        return [groups], [round_assignments]

    def _sample_local_cp_sizes(
        self, sample_id_groups: list[list[list[int]]], num_samples: int
    ) -> list[int]:
        sample_local_cp_size = [0] * num_samples
        for round_assignments in sample_id_groups:
            for sample_ids in round_assignments:
                for sample_id in sample_ids:
                    if sample_local_cp_size[sample_id] != 0:
                        continue
                    sample_local_cp_size[sample_id] = sum(
                        1 for rank_samples in round_assignments if sample_id in rank_samples
                    )

        missing = [idx for idx, cp_size in enumerate(sample_local_cp_size) if cp_size == 0]
        if missing:
            raise RuntimeError(
                f"HCP scheduling failed: {len(missing)} samples were not assigned to any rank"
            )
        return sample_local_cp_size

    def shard_data_by_hdp_rank(
        self,
        data: BatchedDataDict,
        sample_id_groups: list[list[list[int]]],
        sample_local_cp_size: list[int],
        sample_sequence_lengths: list[int],
    ) -> list[SlicedDataDict]:
        shards = [SlicedDataDict() for _ in range(self.hdp_size)]
        sample_cache = {
            sample_id: data.slice(sample_id, sample_id + 1) for sample_id in range(data.size)
        }

        shard_sample_ids = [[] for _ in range(self.hdp_size)]
        for round_assignments in sample_id_groups:
            for hdp_rank, sample_ids in enumerate(round_assignments):
                for sample_id in sample_ids:
                    if sample_id not in shard_sample_ids[hdp_rank]:
                        shard_sample_ids[hdp_rank].append(sample_id)

        for hdp_rank, sample_ids in enumerate(shard_sample_ids):
            if sample_ids:
                shards[hdp_rank] = SlicedDataDict.from_batches(
                    [sample_cache[sample_id] for sample_id in sample_ids]
                )
                local_cp_sizes = [
                    sample_local_cp_size[sample_id] for sample_id in sample_ids
                ]
            else:
                # Empty HCP ranks still need a tensor row so worker-side dummy
                # microbatches can participate in collectives without counting
                # toward loss or throughput normalization.
                shards[hdp_rank] = data.slice(0, 1)
                if "sample_mask" in shards[hdp_rank]:
                    shards[hdp_rank]["sample_mask"] = torch.zeros_like(
                        shards[hdp_rank]["sample_mask"]
                    )
                if "token_mask" in shards[hdp_rank]:
                    shards[hdp_rank]["token_mask"] = torch.zeros_like(
                        shards[hdp_rank]["token_mask"]
                    )
                if "input_lengths" in shards[hdp_rank]:
                    shards[hdp_rank]["input_lengths"] = torch.ones_like(
                        shards[hdp_rank]["input_lengths"]
                    )
                local_cp_sizes = [1]
            shards[hdp_rank]["sample_id_groups"] = sample_id_groups
            shards[hdp_rank]["shard_sample_ids"] = sample_ids
            shards[hdp_rank]["sample_sequence_lengths"] = sample_sequence_lengths
            shards[hdp_rank]["local_cp_sizes"] = local_cp_sizes

        return shards

    def schedule_and_shard(
        self,
        data: BatchedDataDict,
        seq_length_key: str = "input_lengths",
    ) -> list[SlicedDataDict]:
        seq_lengths = self.extract_sequence_lengths(data, seq_length_key)
        _, sample_id_groups = self.schedule_samples(seq_lengths)
        sample_local_cp_size = self._sample_local_cp_sizes(sample_id_groups, data.size)
        return self.shard_data_by_hdp_rank(
            data,
            sample_id_groups,
            sample_local_cp_size,
            seq_lengths,
        )
