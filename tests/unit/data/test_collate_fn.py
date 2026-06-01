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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data.collate_fn import preference_collate_fn, rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _packed_datum(idx: int, cu_seqlens: list[int] | None = None) -> DatumSpec:
    pack_length = 4
    if cu_seqlens is None:
        cu_seqlens = [0, 2, pack_length]
    return DatumSpec(
        input_ids=torch.arange(idx * 10, idx * 10 + pack_length, dtype=torch.int64),
        target_ids=torch.arange(
            idx * 10 + 1, idx * 10 + pack_length + 1, dtype=torch.int64
        ),
        token_mask=torch.ones(pack_length, dtype=torch.float32),
        position_ids=torch.arange(pack_length, dtype=torch.int64),
        packed_cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        packed_max_seqlen=max(b - a for a, b in zip(cu_seqlens, cu_seqlens[1:])),
        length=pack_length,
        loss_multiplier=1.0,
        idx=idx,
        task_name="megatron_sft_packed",
    )


def test_rl_collate_fn_megatron_sft_packed_rows():
    data = [_packed_datum(0), _packed_datum(1, [0, 1, 3, 4])]

    batch = rl_collate_fn(data)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["target_ids"].shape == (2, 4)
    assert batch["token_mask"].shape == (2, 4)
    assert batch["packed_cu_seqlens"].shape == (2, 4)
    assert torch.equal(batch["packed_cu_seqlens"][0], torch.tensor([0, 2, 4, -1]))
    assert torch.equal(batch["packed_cu_seqlens_lengths"], torch.tensor([3, 4]))
    assert batch["idx"] == [0, 1]


def test_rl_collate_fn_rejects_mixed_packed_batch():
    data = [
        _packed_datum(0),
        DatumSpec(
            message_log=[],
            length=1,
            loss_multiplier=1.0,
            extra_env_info=None,
            idx=1,
        ),
    ]

    with pytest.raises(ValueError, match="mixed packed/non-packed"):
        rl_collate_fn(data)


def test_rl_collate_fn_rejects_invalid_packed_metadata():
    bad = _packed_datum(0, [0, 2, 3])

    with pytest.raises(ValueError, match="final value must equal input length"):
        rl_collate_fn([bad])


def test_rl_collate_fn_megatron_sft_dp_stride_order():
    data = [_packed_datum(idx) for idx in range(8)]

    batch = rl_collate_fn(data, megatron_sft_dp_stride_size=4)

    assert batch["idx"] == [0, 4, 1, 5, 2, 6, 3, 7]


def test_preference_collate_fn():
    """Test that preference_collate_fn correctly processes preference data."""
    # Create mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    # Create test data with varying sequence lengths
    data_batch = [
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "token_ids": torch.tensor([4, 5, 6, 7]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Bye",
                    "token_ids": torch.tensor([8, 9]),
                },
            ],
            length_chosen=7,
            length_rejected=5,
            loss_multiplier=1.0,
            idx=0,
            task_name="test_task",
        ),
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "I'm good",
                    "token_ids": torch.tensor([13, 14, 15]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "Not great",
                    "token_ids": torch.tensor([16, 17, 18, 19]),
                },
            ],
            length_chosen=6,
            length_rejected=7,
            loss_multiplier=0,
            idx=1,
            task_name="test_task",
        ),
    ]

    # Call preference_collate_fn
    train_data = preference_collate_fn(
        data_batch,
        mock_tokenizer,
        make_sequence_length_divisible_by=16,
        add_loss_mask=True,
    )

    # Verify the output structure
    assert isinstance(train_data, BatchedDataDict)
    assert "input_ids" in train_data
    assert "input_lengths" in train_data
    assert "token_mask" in train_data
    assert "sample_mask" in train_data

    # Verify batch size is doubled (chosen + rejected for each example)
    assert train_data["input_ids"].shape[0] == 4  # 2 examples * 2 (chosen + rejected)

    # Verify input_ids shape and padding
    max_length = 16  # max of all sequence lengths, padded to be divisible by 16
    assert train_data["input_ids"].shape == (4, max_length)

    # Verify input_lengths
    expected_lengths = [7, 5, 6, 7]  # chosen1, rejected1, chosen2, rejected2
    assert torch.equal(train_data["input_lengths"], torch.tensor(expected_lengths))

    # Verify token_mask
    assert train_data["token_mask"].shape == (4, max_length)
    # First example chosen (length 7)
    assert torch.all(train_data["token_mask"][0][0:3] == 0)
    assert torch.all(train_data["token_mask"][0][3:7] == 1)
    # First example rejected (length 5)
    assert torch.all(train_data["token_mask"][1][0:3] == 0)
    assert torch.all(train_data["token_mask"][1][3:5] == 1)
    assert torch.all(train_data["token_mask"][1][5:] == 0)

    # Verify sample_mask
    expected_sample_mask = [
        1.0,
        1.0,
        0.0,
        0.0,
    ]  # loss_multiplier repeated for chosen/rejected
    assert torch.equal(train_data["sample_mask"], torch.tensor(expected_sample_mask))

    # Verify message content is preserved
    # First example chosen
    assert torch.equal(train_data["input_ids"][0][0:3], torch.tensor([1, 2, 3]))  # user
    assert torch.equal(
        train_data["input_ids"][0][3:7], torch.tensor([4, 5, 6, 7])
    )  # assistant
    # First example rejected
    assert torch.equal(train_data["input_ids"][1][0:3], torch.tensor([1, 2, 3]))  # user
    assert torch.equal(
        train_data["input_ids"][1][3:5], torch.tensor([8, 9])
    )  # assistant
