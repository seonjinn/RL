# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import torch
import pytest

from nemo_rl.data.megatron_sft_packed import megatron_sft_packed_preprocessor


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        token_ids = []
        for message in messages:
            content = message["content"].strip()
            if content:
                token_ids.extend(int(piece) for piece in content.split())
        return token_ids

    def convert_tokens_to_ids(self, token):
        assert token == "<unk>"
        return 0


def _packed_datum(idx, context_parallel_size=1):
    return {
        "input_ids": torch.tensor([idx, idx + 1, idx + 2, 0]),
        "target_ids": torch.tensor([idx + 1, idx + 2, 0, 0]),
        "token_mask": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "position_ids": torch.tensor([0, 1, 2, 3]),
        "packed_cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "packed_max_seqlen": 4,
        "packed_context_parallel_size": context_parallel_size,
        "length": 4,
        "loss_multiplier": 1.0,
        "idx": idx,
    }


def test_megatron_sft_packed_dataset_is_registered():
    from nemo_rl.data.datasets.response_datasets import (
        DATASET_REGISTRY,
        MegatronSFTPackedDataset,
    )

    assert DATASET_REGISTRY["megatron_sft_packed"] is MegatronSFTPackedDataset


def test_megatron_sft_packed_matches_megatron_truncation_shift():
    row = {
        "packed_messages": [
            {"role": "system", "content": "10"},
            {"role": "user", "content": "20"},
            {"role": "assistant", "content": "30 40 50"},
        ]
    }

    processed = megatron_sft_packed_preprocessor(
        row,
        task_data_spec={},
        tokenizer=_DummyTokenizer(),
        max_seq_length=4,
        idx=7,
        prompt_format="identity",
        pad_token="<unk>",
    )

    assert torch.equal(processed["input_ids"], torch.tensor([10, 20, 30, 40]))
    assert torch.equal(processed["target_ids"], torch.tensor([20, 30, 40, 0]))
    assert torch.equal(processed["token_mask"], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(processed["position_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 4]))
    assert processed["packed_max_seqlen"] == 4
    assert processed["packed_context_parallel_size"] == 1


def test_megatron_sft_packed_global_shift_keeps_cp_padding_bridge():
    row = {
        "packed_messages": [
            {"role": "system", "content": "10"},
            {"role": "user", "content": "20"},
            {"role": "assistant", "content": "30"},
            {"role": "system", "content": "40"},
            {"role": "user", "content": "50"},
            {"role": "assistant", "content": "60"},
        ]
    }

    processed = megatron_sft_packed_preprocessor(
        row,
        task_data_spec={},
        tokenizer=_DummyTokenizer(),
        max_seq_length=8,
        idx=7,
        prompt_format="identity",
        pad_token="<unk>",
        context_parallel_size=2,
    )

    assert torch.equal(
        processed["input_ids"], torch.tensor([10, 20, 30, 0, 40, 50, 60, 0])
    )
    assert torch.equal(
        processed["target_ids"], torch.tensor([20, 30, 0, 40, 50, 60, 0, 0])
    )
    assert torch.equal(
        processed["token_mask"], torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert torch.equal(
        processed["position_ids"], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    )
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 4, 8]))
    assert processed["packed_max_seqlen"] == 4
    assert processed["packed_context_parallel_size"] == 2


def test_megatron_sft_packed_cp8_segments_are_partitionable():
    row = {
        "packed_messages": [
            {"role": "system", "content": "10"},
            {"role": "user", "content": "20"},
            {"role": "assistant", "content": "30"},
            {"role": "system", "content": "40"},
            {"role": "user", "content": "50"},
            {"role": "assistant", "content": "60"},
        ]
    }

    processed = megatron_sft_packed_preprocessor(
        row,
        task_data_spec={},
        tokenizer=_DummyTokenizer(),
        max_seq_length=32,
        idx=7,
        prompt_format="identity",
        pad_token="<unk>",
        context_parallel_size=8,
    )

    adjacent_diffs = (
        processed["packed_cu_seqlens"][1:] - processed["packed_cu_seqlens"][:-1]
    )
    assert torch.all(adjacent_diffs % 16 == 0)
    assert processed["input_ids"].shape == (32,)
    assert processed["target_ids"].shape == (32,)


def test_identity_accepts_tool_and_consecutive_assistant_turns():
    row = {
        "packed_messages": [
            {"role": "system", "content": "10"},
            {"role": "user", "content": "20"},
            {"role": "assistant", "content": "30"},
            {"role": "tool", "content": "40"},
            {"role": "assistant", "content": "50"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "60"},
        ]
    }

    processed = megatron_sft_packed_preprocessor(
        row,
        task_data_spec={},
        tokenizer=_DummyTokenizer(),
        max_seq_length=8,
        idx=7,
        prompt_format="identity",
        pad_token="<unk>",
    )

    assert torch.equal(
        processed["input_ids"], torch.tensor([10, 20, 30, 40, 50, 60, 0, 0])
    )
    assert torch.equal(
        processed["target_ids"], torch.tensor([20, 30, 40, 50, 60, 0, 0, 0])
    )
    assert torch.equal(
        processed["token_mask"], torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )


def test_collate_dp_stride_then_contiguous_sharding_matches_megatron_order():
    from nemo_rl.data.collate_fn import rl_collate_fn

    batch = rl_collate_fn(
        [_packed_datum(idx) for idx in range(8)],
        megatron_sft_dp_stride_size=4,
        megatron_sft_context_parallel_size=1,
    )

    assert batch["idx"] == [0, 4, 1, 5, 2, 6, 3, 7]
    shards = batch.shard_by_batch_size(4, batch_size=8)
    assert [shard["idx"] for shard in shards] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


def _sequence_packing_args(max_seq_length=4):
    return {
        "algorithm": "modified_first_fit_decreasing",
        "input_key": "input_ids",
        "input_lengths_key": "length",
        "max_tokens_per_microbatch": max_seq_length,
        "sequence_length_pad_multiple": max_seq_length,
    }


def _assert_sharded_results_equal(left, right):
    left_shards, left_indices = left
    right_shards, right_indices = right

    assert left_indices == right_indices
    assert len(left_shards) == len(right_shards)
    for left_shard, right_shard in zip(left_shards, right_shards):
        assert left_shard.micro_batch_indices == right_shard.micro_batch_indices
        assert left_shard.micro_batch_lengths == right_shard.micro_batch_lengths
        assert left_shard.elem_counts_per_gb == right_shard.elem_counts_per_gb
        assert left_shard.keys() == right_shard.keys()
        for key in left_shard:
            left_value = left_shard[key]
            right_value = right_shard[key]
            if torch.is_tensor(left_value):
                assert torch.equal(left_value, right_value), key
            else:
                assert left_value == right_value, key


def test_fast_prepacked_sharding_matches_mffd_for_full_rows(monkeypatch):
    from nemo_rl.data.collate_fn import rl_collate_fn

    batch = rl_collate_fn(
        [_packed_datum(idx) for idx in range(16)],
        megatron_sft_context_parallel_size=1,
    )
    sequence_packing_args = _sequence_packing_args()

    monkeypatch.delenv("NRL_FAST_PREPACKED_SHARDING", raising=False)
    baseline = batch.shard_by_batch_size(
        4,
        batch_size=8,
        sequence_packing_args=sequence_packing_args,
    )

    monkeypatch.setenv("NRL_FAST_PREPACKED_SHARDING", "1")
    fast = batch.shard_by_batch_size(
        4,
        batch_size=8,
        sequence_packing_args=sequence_packing_args,
    )

    _assert_sharded_results_equal(fast, baseline)


def test_fast_prepacked_sharding_rejects_partial_rows(monkeypatch):
    from nemo_rl.data.collate_fn import rl_collate_fn

    batch = rl_collate_fn(
        [_packed_datum(idx) for idx in range(8)],
        megatron_sft_context_parallel_size=1,
    )
    batch["length"][0] = 3
    monkeypatch.setenv("NRL_FAST_PREPACKED_SHARDING", "1")

    with pytest.raises(ValueError, match="every packed row"):
        batch.shard_by_batch_size(
            4,
            batch_size=8,
            sequence_packing_args=_sequence_packing_args(),
        )


def test_collate_rejects_context_parallel_mismatch():
    from nemo_rl.data.collate_fn import rl_collate_fn

    with pytest.raises(ValueError, match="prepared for context_parallel_size=2"):
        rl_collate_fn(
            [_packed_datum(0, context_parallel_size=2)],
            megatron_sft_context_parallel_size=1,
        )
