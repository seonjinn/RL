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

import torch

from megatron.core import parallel_state

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.hcp_dummy import make_empty_hcp_dummy_tensor
from nemo_rl.models.policy.hcp_data_iterator import HCPGroupIterator


def _build_batch() -> BatchedDataDict:
    batch = BatchedDataDict(
        input_ids=torch.tensor([[11, 0], [22, 0], [33, 0]], dtype=torch.int64),
        input_lengths=torch.tensor([1, 1, 1], dtype=torch.int32),
        sample_mask=torch.ones(3, dtype=torch.float32),
    )
    batch["shard_sample_ids"] = [0, 1, 2]
    return batch


def test_iterator_splits_local_samples_by_hcp_subgroup(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
    )

    batch = _build_batch()
    batch["sample_id_groups"] = [
        [
            [0, 1],
            [0, 2],
            [],
        ]
    ]

    groups = list(HCPGroupIterator(batch))

    assert len(groups) == 2

    assert groups[0]["_hcp_sample_ids"] == [0]
    assert groups[0]["_hcp_hdp_ranks"] == [0, 1]
    assert groups[0]["_hcp_is_dummy"] is False
    assert int(groups[0]["local_cp_size"].item()) == 2

    assert groups[1]["_hcp_sample_ids"] == [1]
    assert groups[1]["_hcp_hdp_ranks"] == [0]
    assert groups[1]["_hcp_is_dummy"] is False
    assert int(groups[1]["local_cp_size"].item()) == 1


def test_iterator_coalesces_disjoint_subgroups_for_other_rank(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 1,
    )

    batch = _build_batch()
    batch["sample_id_groups"] = [
        [
            [0, 1],
            [0, 2],
            [],
        ]
    ]

    groups = list(HCPGroupIterator(batch))

    assert len(groups) == 2
    assert groups[0]["_hcp_sample_ids"] == [0]
    assert groups[0]["_hcp_hdp_ranks"] == [0, 1]
    assert groups[0]["_hcp_is_dummy"] is False
    assert int(groups[0]["local_cp_size"].item()) == 2

    assert groups[1]["_hcp_sample_ids"] == [2]
    assert groups[1]["_hcp_hdp_ranks"] == [1]
    assert groups[1]["_hcp_is_dummy"] is False
    assert int(groups[1]["local_cp_size"].item()) == 1


def test_iterator_batches_samples_that_share_the_same_hcp_subgroup(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 1,
    )

    batch = _build_batch()
    batch["sample_id_groups"] = [
        [
            [0, 1],
            [0, 1],
            [2],
        ]
    ]

    groups = list(HCPGroupIterator(batch))

    assert len(groups) == 1
    assert groups[0].size == 2
    assert groups[0]["_hcp_sample_ids"] == [0, 1]
    assert groups[0]["_hcp_hdp_ranks"] == [0, 1]
    assert groups[0]["_hcp_is_dummy"] is False
    assert int(groups[0]["local_cp_size"].item()) == 2


def test_iterator_splits_hcp_subgroup_by_token_budget(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
    )

    batch = _build_batch()
    batch["input_lengths"] = torch.tensor([5, 7, 3], dtype=torch.int32)
    batch["sample_id_groups"] = [
        [
            [0, 1, 2],
            [0, 1, 2],
            [],
        ]
    ]

    groups = list(HCPGroupIterator(batch, max_tokens_per_microbatch=10))

    assert len(groups) == 2
    assert groups[0].size == 1
    assert groups[0]["_hcp_sample_ids"] == [0]
    assert groups[1].size == 2
    assert groups[1]["_hcp_sample_ids"] == [1, 2]
    assert all(group["_hcp_hdp_ranks"] == [0, 1] for group in groups)
    assert all(group["_hcp_is_dummy"] is False for group in groups)


def test_iterator_uses_padded_lengths_for_token_budget(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
    )

    batch = _build_batch()
    batch["input_lengths"] = torch.tensor([5, 5, 1], dtype=torch.int32)
    batch["sample_id_groups"] = [
        [
            [0, 1, 2],
            [0, 1, 2],
            [],
        ]
    ]

    groups = list(
        HCPGroupIterator(
            batch,
            max_tokens_per_microbatch=10,
            sequence_length_pad_multiple=8,
        )
    )

    assert [group["_hcp_sample_ids"] for group in groups] == [[0], [1], [2]]
    assert all(group["_hcp_is_dummy"] is False for group in groups)


def test_iterator_yields_dummy_batch_for_nonparticipant_rank(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 2,
    )

    batch = _build_batch()
    batch["attention_mask"] = torch.ones((3, 8), dtype=torch.int64)
    batch["token_mask"] = torch.ones((3, 8), dtype=torch.float32)
    batch["advantages"] = torch.ones((3, 8), dtype=torch.float32)
    batch["generation_logprobs"] = torch.ones((3, 8), dtype=torch.float32)
    batch["custom_token_scores"] = torch.ones((3, 7), dtype=torch.float32)
    batch["per_sample_features"] = torch.ones((3, 4), dtype=torch.float32)
    batch["sample_id_groups"] = [
        [
            [0],
            [0],
            [],
        ]
    ]

    groups = list(HCPGroupIterator(batch))

    assert len(groups) == 1
    assert groups[0]["_hcp_sample_ids"] == []
    assert groups[0]["_hcp_hdp_ranks"] == [0, 1]
    assert groups[0]["_hcp_is_dummy"] is True
    assert int(groups[0]["local_cp_size"].item()) == 1
    assert groups[0]["input_ids"].shape == (1, 1)
    assert groups[0]["input_ids"].sum().item() == 0
    assert groups[0]["attention_mask"].shape == (1, 1)
    assert groups[0]["attention_mask"].sum().item() == 1
    assert groups[0]["input_lengths"].item() == 1
    assert groups[0]["sample_mask"].item() == 0
    assert groups[0]["token_mask"].shape == (1, 1)
    assert groups[0]["token_mask"].sum().item() == 0
    assert groups[0]["advantages"].shape == (1, 1)
    assert groups[0]["advantages"].sum().item() == 0
    assert groups[0]["generation_logprobs"].shape == (1, 1)
    assert groups[0]["generation_logprobs"].sum().item() == 0
    assert groups[0]["custom_token_scores"].shape == (1, 1)
    assert groups[0]["custom_token_scores"].sum().item() == 0
    assert groups[0]["per_sample_features"].shape == (1, 4)
    assert groups[0]["per_sample_features"].sum().item() == 0


def test_dummy_attention_mask_4d_is_single_unmasked_token():
    attention_mask = torch.ones((1, 1, 8, 8), dtype=torch.bool)

    dummy = make_empty_hcp_dummy_tensor("attention_mask", attention_mask, 8)

    assert dummy.shape == (1, 1, 1, 1)
    assert dummy.all().item()


def test_dummy_known_sequence_tensor_trims_without_input_width():
    token_mask = torch.ones((1, 8), dtype=torch.float32)

    dummy = make_empty_hcp_dummy_tensor("token_mask", token_mask, None)

    assert dummy.shape == (1, 1)
    assert dummy.sum().item() == 0


def test_dummy_off_width_tensor_trims_to_avoid_shape_mismatch():
    shifted_policy_logp = torch.ones((1, 7), dtype=torch.float32)

    dummy = make_empty_hcp_dummy_tensor("policy_logp", shifted_policy_logp, 8)

    assert dummy.shape == (1, 1)
    assert dummy.sum().item() == 0


def test_iterator_dummy_from_empty_source_tensors_keeps_valid_lengths_and_masks(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 1,
    )

    batch = BatchedDataDict(
        input_ids=torch.zeros((0, 8), dtype=torch.int64),
        input_lengths=torch.zeros((0,), dtype=torch.int32),
        attention_mask=torch.zeros((0, 8), dtype=torch.int64),
        sample_mask=torch.zeros((0,), dtype=torch.float32),
    )
    batch["shard_sample_ids"] = []
    batch["sample_sequence_lengths"] = [0, 8]
    batch["sample_id_groups"] = [[[1], []]]

    group = list(HCPGroupIterator(batch))[0]

    assert group["_hcp_is_dummy"] is True
    assert group["input_ids"].shape == (1, 1)
    assert group["input_lengths"].item() == 1
    assert group["attention_mask"].shape == (1, 1)
    assert group["attention_mask"].sum().item() == 1
    assert group["sample_mask"].item() == 0
