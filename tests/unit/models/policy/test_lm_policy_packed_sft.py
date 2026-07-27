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

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


def _direct_batch(
    batch_size: int = 4, row_order: list[int] | None = None
) -> BatchedDataDict:
    if row_order is None:
        row_order = list(range(batch_size))
    row_ids = torch.tensor(row_order)
    return BatchedDataDict(
        {
            "input_ids": row_ids.reshape(batch_size, 1),
            "target_ids": row_ids.reshape(batch_size, 1),
            "token_mask": torch.ones(batch_size, 1),
            "position_ids": torch.zeros(batch_size, 1, dtype=torch.long),
            "input_lengths": torch.ones(batch_size, dtype=torch.long),
            "sample_mask": torch.ones(batch_size),
            "packed_cu_seqlens": torch.tensor([[0, 1]], dtype=torch.int32).repeat(
                batch_size, 1
            ),
            "packed_cu_seqlens_lengths": torch.full((batch_size,), 2),
            "packed_max_seqlen": torch.ones(batch_size, dtype=torch.long),
        }
    )


def _policy(
    *,
    dp_size: int = 2,
    mbs: int = 1,
    dynamic: bool = False,
    megatron: bool = True,
) -> Policy:
    policy = cast(Any, object.__new__(Policy))
    policy.cfg = {
        "train_global_batch_size": 4,
        "train_micro_batch_size": mbs,
        "dynamic_batching": {"enabled": dynamic, "train_mb_tokens": 8},
        "sequence_packing": {
            "enabled": True,
            "algorithm": "first_fit_decreasing",
            "train_mb_tokens": 8,
        },
        "megatron_cfg": {"enabled": megatron},
    }
    policy.use_dynamic_batches = dynamic
    policy.use_sequence_packing = True
    policy.dynamic_batching_args = {}
    policy.sequence_packing_args = {}
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = dp_size
    policy.flops_tracker = None
    policy.worker_group = MagicMock()
    policy.worker_group.run_all_workers_sharded_data.return_value = ["future"]
    filtered_results = [
        {
            "global_loss": torch.tensor(0.5),
            "grad_norm": torch.tensor(1.0),
            "all_mb_metrics": {},
        }
    ]
    policy.worker_group.get_all_worker_results_with_unfiltered.return_value = (
        filtered_results,
        filtered_results,
    )
    return cast(Policy, policy)


def test_direct_packed_rows_construct_dp_strides_from_source_order():
    policy = _policy()

    policy.train(_direct_batch(row_order=[0, 1, 2, 3]), MagicMock())

    call = policy.worker_group.run_all_workers_sharded_data.call_args
    assert {
        "rows_by_dp_shard": [
            shard["input_ids"].flatten().tolist() for shard in call.kwargs["data"]
        ],
        "gbs": call.kwargs["common_kwargs"]["gbs"],
        "mbs": call.kwargs["common_kwargs"]["mbs"],
    } == {
        "rows_by_dp_shard": [[0, 2], [1, 3]],
        "gbs": 4,
        "mbs": 1,
    }


def test_train_aggregates_worker_phase_timing_distribution():
    policy = _policy()
    filtered_results = [
        {
            "global_loss": torch.tensor(0.5),
            "grad_norm": torch.tensor(1.0),
            "all_mb_metrics": {},
            "train_phase_timings": {"forward_backward": 4.0, "optimizer": 2.0},
        },
        {
            "global_loss": torch.tensor(0.5),
            "grad_norm": torch.tensor(1.0),
            "all_mb_metrics": {},
            "train_phase_timings": {"forward_backward": 6.0, "optimizer": 1.0},
        },
    ]
    unfiltered_results = [
        {
            "rank": 8,
            "train_phase_timings": {
                "forward_backward": 4.0,
                "optimizer": 2.0,
                "worker_total": 8.0,
            },
        },
        {
            "rank": 511,
            "train_phase_timings": {
                "forward_backward": 8.0,
                "optimizer": 1.0,
                "worker_total": 12.0,
            },
        },
        {
            "rank": 120,
            "train_phase_timings": {
                "forward_backward": 6.0,
                "optimizer": 4.0,
                "worker_total": 14.0,
            },
        },
        {
            "rank": 42,
            "train_phase_timings": {
                "forward_backward": 10.0,
                "optimizer": 3.0,
                "worker_total": 15.0,
            },
        },
    ]
    policy.worker_group.get_all_worker_results_with_unfiltered.return_value = (
        filtered_results,
        unfiltered_results,
    )

    result = policy.train(_direct_batch(), MagicMock())

    assert result["train_phase_timings"] == {
        "forward_backward": {
            "min": pytest.approx(4.0),
            "mean": pytest.approx(7.0),
            "median": pytest.approx(7.0),
            "max": pytest.approx(10.0),
            "max_rank": 42,
            "critical_rank_value": pytest.approx(10.0),
        },
        "optimizer": {
            "min": pytest.approx(1.0),
            "mean": pytest.approx(2.5),
            "median": pytest.approx(2.5),
            "max": pytest.approx(4.0),
            "max_rank": 120,
            "critical_rank_value": pytest.approx(3.0),
        },
        "worker_total": {
            "min": pytest.approx(8.0),
            "mean": pytest.approx(12.25),
            "median": pytest.approx(13.0),
            "max": pytest.approx(15.0),
            "max_rank": 42,
            "critical_rank_value": pytest.approx(15.0),
        },
    }


@pytest.mark.parametrize(
    ("policy_kwargs", "gbs", "message"),
    [
        ({"mbs": 2}, None, "micro batch size 1"),
        ({"dynamic": True}, None, "dynamic batching.*disabled"),
        ({"dp_size": 3}, None, "divisible by data parallel size"),
        ({}, 2, "global batch size must equal the packed row count"),
        ({"megatron": False}, None, "require the Megatron backend"),
    ],
)
def test_direct_packed_rows_reject_incompatible_batching(
    policy_kwargs: dict[str, Any],
    gbs: int | None,
    message: str,
):
    policy = _policy(**policy_kwargs)

    with pytest.raises(ValueError, match=message):
        policy.train(_direct_batch(), MagicMock(), gbs=gbs)


def test_direct_packed_rows_reject_missing_target_aligned_field():
    policy = _policy()
    batch = _direct_batch()
    del batch["target_ids"]

    with pytest.raises(ValueError, match=r"missing required fields.*target_ids"):
        policy.train(batch, MagicMock())


def test_direct_packed_rows_reject_draft_training():
    policy = _policy()
    policy.cfg["draft"] = {"enabled": True}

    with pytest.raises(NotImplementedError, match="draft training"):
        policy.train(_direct_batch(), MagicMock())
