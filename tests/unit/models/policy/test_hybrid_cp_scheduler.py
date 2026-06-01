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

import pytest
import torch
import numpy as np

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.policy.hybrid_cp_config import HybridCPConfig
from nemo_rl.models.policy.hybrid_cp_scheduler import BalancedCPScheduler, HeadNodeHCPScheduler
from nemo_rl.models.policy.lm_policy import Policy


class TestHybridCPConfig:
    def test_defaults(self):
        config = HybridCPConfig(enabled=True)
        assert config.enabled is True
        assert config.max_seqlen_per_dp_cp_rank is None
        assert config.scheduling_strategy == "dp"
        assert config.balance_slack == 0.05
        assert config.eps_bucket == 0.10
        assert config.force_full_cp is False

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="scheduling_strategy"):
            HybridCPConfig(enabled=True, scheduling_strategy="invalid")

    def test_pp_strategy_not_supported(self):
        with pytest.raises(NotImplementedError, match="Pipeline parallel"):
            HybridCPConfig(enabled=True, scheduling_strategy="pp")

    def test_invalid_balance_slack(self):
        with pytest.raises(ValueError, match="balance_slack"):
            HybridCPConfig(enabled=True, balance_slack=2.0)

    def test_invalid_eps_bucket(self):
        with pytest.raises(ValueError, match="eps_bucket"):
            HybridCPConfig(enabled=True, eps_bucket=-0.5)


class TestHeadNodeHCPScheduler:
    @pytest.fixture
    def scheduler(self):
        return HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(
                enabled=True,
                max_seqlen_per_dp_cp_rank=10_000,
            ),
            dp_size=2,
            cp_size=4,
            max_seq_len=40_960,
        )

    def test_extract_sequence_lengths_tensor(self, scheduler):
        data = BatchedDataDict(
            input_ids=torch.zeros((4, 16), dtype=torch.long),
            input_lengths=torch.tensor([100, 200, 300, 400], dtype=torch.int32),
        )
        assert scheduler.extract_sequence_lengths(data) == [100, 200, 300, 400]

    def test_extract_sequence_lengths_missing_key(self, scheduler):
        with pytest.raises(ValueError, match="not found in data"):
            scheduler.extract_sequence_lengths(BatchedDataDict(input_ids=torch.zeros((1, 4))))

    def test_schedule_and_shard_preserves_all_samples(self, scheduler):
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (8, 32), dtype=torch.long),
            input_lengths=torch.tensor(
                [5_000, 12_000, 8_000, 25_000, 15_000, 30_000, 6_000, 18_000],
                dtype=torch.int32,
            ),
        )

        shards = scheduler.schedule_and_shard(data, seq_length_key="input_lengths")
        assert len(shards) == 8
        assert sum(shard.size for shard in shards) >= data.size

        assigned = set()
        for shard in shards:
            assert "sample_id_groups" in shard
            assert "shard_sample_ids" in shard
            assert "local_cp_sizes" in shard
            assert shard["sample_sequence_lengths"] == [
                5_000,
                12_000,
                8_000,
                25_000,
                15_000,
                30_000,
                6_000,
                18_000,
            ]
            assigned.update(shard["shard_sample_ids"])

        assert assigned == set(range(data.size))

    def test_local_cp_sizes_are_powers_of_two(self, scheduler):
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (6, 32), dtype=torch.long),
            input_lengths=torch.tensor(
                [5_000, 15_000, 35_000, 7_500, 18_000, 28_000], dtype=torch.int32
            ),
        )

        shards = scheduler.schedule_and_shard(data)
        local_cp_sizes = []
        for shard in shards:
            local_cp_sizes.extend(shard["local_cp_sizes"])

        assert local_cp_sizes
        for cp_size in local_cp_sizes:
            assert cp_size > 0
            assert (cp_size & (cp_size - 1)) == 0

    def test_force_full_cp_assigns_every_sample_to_all_cp_ranks(self):
        scheduler = HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(
                enabled=True,
                max_seqlen_per_dp_cp_rank=2_048,
                force_full_cp=True,
            ),
            dp_size=1,
            cp_size=4,
            max_seq_len=8_192,
        )
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (3, 32), dtype=torch.long),
            input_lengths=torch.tensor([1_024, 4_096, 8_192], dtype=torch.int32),
        )

        shards = scheduler.schedule_and_shard(data)

        assert len(shards) == 4
        for shard in shards:
            assert shard["shard_sample_ids"] == [0, 1, 2]
            assert shard["local_cp_sizes"] == [4, 4, 4]

    def test_empty_hcp_ranks_receive_zero_mask_dummy_rows(self):
        scheduler = HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(
                enabled=True,
                max_seqlen_per_dp_cp_rank=4_096,
            ),
            dp_size=1,
            cp_size=4,
            max_seq_len=16_384,
        )
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (1, 16), dtype=torch.long),
            attention_mask=torch.ones((1, 16), dtype=torch.int64),
            input_lengths=torch.tensor([1_024], dtype=torch.int32),
            sample_mask=torch.ones(1, dtype=torch.float32),
            token_mask=torch.ones((1, 16), dtype=torch.float32),
            advantages=torch.ones((1, 16), dtype=torch.float32),
            generation_logprobs=torch.ones((1, 16), dtype=torch.float32),
            prev_logprobs=torch.ones((1, 16), dtype=torch.float32),
            custom_token_scores=torch.ones((1, 15), dtype=torch.float32),
            pixel_values=PackedTensor([torch.ones(1, 3, 16, 16)], dim_to_pack=0),
            imgs_sizes=torch.tensor([[16, 16]], dtype=torch.int32),
        )

        shards = scheduler.schedule_and_shard(data)

        real_shards = [shard for shard in shards if shard["shard_sample_ids"]]
        dummy_shards = [shard for shard in shards if not shard["shard_sample_ids"]]
        assert len(real_shards) == 1
        assert len(dummy_shards) == 3
        assert real_shards[0]["shard_sample_ids"] == [0]
        for shard in dummy_shards:
            assert shard.size == 1
            assert shard["local_cp_sizes"] == [1]
            assert shard["sample_mask"].item() == 0
            assert shard["token_mask"].sum().item() == 0
            assert shard["token_mask"].shape == (1, 1)
            assert shard["advantages"].shape == (1, 1)
            assert shard["advantages"].sum().item() == 0
            assert shard["generation_logprobs"].shape == (1, 1)
            assert shard["generation_logprobs"].sum().item() == 0
            assert shard["prev_logprobs"].shape == (1, 1)
            assert shard["prev_logprobs"].sum().item() == 0
            assert shard["input_lengths"].item() == 1
            assert shard["input_ids"].shape == (1, 1)
            assert shard["input_ids"].sum().item() == 0
            assert shard["attention_mask"].shape == (1, 1)
            assert shard["attention_mask"].sum().item() == 1
            assert shard["custom_token_scores"].shape == (1, 1)
            assert shard["custom_token_scores"].sum().item() == 0
            assert "pixel_values" not in shard
            assert "imgs_sizes" not in shard

    def test_raises_when_required_local_cp_exceeds_hdp_size(self):
        scheduler = HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(
                enabled=True,
                max_seqlen_per_dp_cp_rank=1_024,
            ),
            dp_size=1,
            cp_size=4,
            max_seq_len=8_192,
        )
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (1, 16), dtype=torch.long),
            input_lengths=torch.tensor([8_192], dtype=torch.int32),
        )

        with pytest.raises(ValueError, match="requires local CP size"):
            scheduler.schedule_and_shard(data)

    def test_default_threshold_is_clamped_for_short_sequences(self):
        scheduler = HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(enabled=True),
            dp_size=1,
            cp_size=8,
            max_seq_len=4,
        )
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (1, 4), dtype=torch.long),
            input_lengths=torch.tensor([4], dtype=torch.int32),
            sample_mask=torch.ones(1, dtype=torch.float32),
        )

        shards = scheduler.schedule_and_shard(data)

        assert scheduler.max_seqlen_per_dp_cp_rank == 1
        assert sum(1 for shard in shards if shard["shard_sample_ids"]) == 4

    def test_min_local_cp_size_is_rounded_up_for_moe_ep_sync(self):
        scheduler = HeadNodeHCPScheduler(
            hcp_config=HybridCPConfig(
                enabled=True,
                max_seqlen_per_dp_cp_rank=4_096,
            ),
            dp_size=1,
            cp_size=8,
            max_seq_len=32_768,
            min_local_cp_size=3,
        )
        data = BatchedDataDict(
            input_ids=torch.randint(0, 128, (1, 16), dtype=torch.long),
            input_lengths=torch.tensor([1_024], dtype=torch.int32),
            sample_mask=torch.ones(1, dtype=torch.float32),
        )

        shards = scheduler.schedule_and_shard(data)

        real_shards = [shard for shard in shards if shard["shard_sample_ids"]]
        dummy_shards = [shard for shard in shards if not shard["shard_sample_ids"]]
        assert len(real_shards) == 4
        assert len(dummy_shards) == 4
        for shard in real_shards:
            assert shard["shard_sample_ids"] == [0]
            assert shard["local_cp_sizes"] == [4]

    def test_raises_when_min_local_cp_size_exceeds_hdp_size(self):
        with pytest.raises(ValueError, match="Minimum local CP size"):
            HeadNodeHCPScheduler(
                hcp_config=HybridCPConfig(
                    enabled=True,
                    max_seqlen_per_dp_cp_rank=4_096,
                ),
                dp_size=1,
                cp_size=8,
                max_seq_len=32_768,
                min_local_cp_size=9,
            )

    def test_zero_length_sequence_uses_single_hcp_rank(self):
        scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=4_096,
            total_hdp_gpus=8,
            min_cp_size=4,
        )

        assert scheduler.gpus_needed(0) == 1


class TestHybridCPPolicySharding:
    def test_nest_hcp_shards_matches_dp_cp_layout(self):
        policy = Policy.__new__(Policy)
        policy.use_hybrid_cp = True
        policy.sharding_annotations = NamedSharding(
            layout=np.arange(8).reshape(1, 2, 4, 1),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

        shards = [BatchedDataDict(sample_id=torch.tensor([idx])) for idx in range(8)]
        nested = policy._nest_hcp_shards(shards)

        assert len(nested) == 2
        assert [shard["sample_id"].item() for shard in nested[0]] == [0, 1, 2, 3]
        assert [shard["sample_id"].item() for shard in nested[1]] == [4, 5, 6, 7]
