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

import gc
import weakref
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from pydantic import ValidationError
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss import NLLLossFn
from nemo_rl.algorithms.sft import (
    MasterConfig,
    SFTConfig,
    _add_e2e_step_timing,
    _build_sft_collate_fn,
    _get_processed_token_count,
    _initial_sft_save_state,
    _iter_timed_batches,
    _measure_loop_interval,
    _optional_float,
    _recursive_tensor_payload_bytes,
    sft_train,
    validate,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import Timer


@pytest.mark.parametrize(
    ("train_data", "expected_processed_tokens"),
    [
        pytest.param(
            BatchedDataDict(
                {
                    "input_ids": torch.empty((2, 8), device="meta"),
                    "input_lengths": torch.tensor([5, 7], device="cpu"),
                    "packed_cu_seqlens": torch.tensor(
                        [[0, 2, 5], [0, 3, 7]], device="cpu"
                    ),
                }
            ),
            12,
            id="packed",
        ),
        pytest.param(
            BatchedDataDict(
                {
                    "input_ids": torch.empty((3, 9), device="meta"),
                    "input_lengths": torch.tensor([4, 6, 9], device="cpu"),
                }
            ),
            19,
            id="ordinary",
        ),
    ],
)
def test_get_processed_token_count_sums_cpu_input_lengths(
    train_data: BatchedDataDict, expected_processed_tokens: int
) -> None:
    assert train_data["input_lengths"].device.type == "cpu"

    assert _get_processed_token_count(train_data) == expected_processed_tokens


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10]},
    }
    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [[{"token_ids": torch.tensor([1, 2, 3]), "role": "assistant"}]],
        "loss_multiplier": torch.tensor(1.0),
    }

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = NLLLossFn()
    logger = MagicMock()
    logger.comparison_metrics_enabled = False
    checkpointer = MagicMock()

    # Create mock master config
    master_config = MasterConfig.model_construct(
        sft=SFTConfig.model_construct(
            max_num_steps=5,
            max_num_epochs=2,
            val_period=100,
            val_batches=1,
            val_global_batch_size=1,
            val_micro_batch_size=1,
            val_at_start=False,
            val_at_end=False,
            only_unmask_final=False,
        ),
        policy={
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
        },
        data={},
        checkpointing={
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
        },
        cluster={
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
    )

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "master_config": master_config,
    }


def _validation_config(execution_mode: str, **overrides: object) -> MasterConfig:
    sft_config = {
        "val_period": 20,
        "val_batches": 4,
        "val_global_batch_size": 64,
        "val_micro_batch_size": 1,
        "val_at_start": False,
        "validation_execution_mode": execution_mode,
        "validation_event_max_payload_bytes": 1_000_000,
        "validation_event_verified_ray_object_store_available_bytes": 10_000_000,
        "validation_event_memory_safety_multiplier": 2.0,
    }
    sft_config.update(overrides)
    return MasterConfig.model_construct(
        sft=SFTConfig(**sft_config),
        policy={
            "make_sequence_length_divisible_by": 1,
            "megatron_cfg": {"enabled": True},
        },
        data={},
        logger={},
        cluster={},
        checkpointing={},
    )


def _packed_validation_batch(
    batch_idx: int,
    *,
    batch_size: int = 64,
    valid_rows: int = 64,
) -> BatchedDataDict:
    row_ids = torch.arange(
        batch_idx * 64,
        batch_idx * 64 + batch_size,
        dtype=torch.int64,
    )
    input_ids = torch.stack((row_ids, row_ids + 1000), dim=1)
    sample_mask = torch.zeros(batch_size, dtype=torch.float32)
    sample_mask[:valid_rows] = 1.0
    return BatchedDataDict(
        input_ids=input_ids,
        target_ids=input_ids + 1,
        token_mask=torch.ones((batch_size, 2), dtype=torch.float32),
        position_ids=torch.tensor([0, 1], dtype=torch.int64).repeat(batch_size, 1),
        input_lengths=torch.full((batch_size,), 2, dtype=torch.int64),
        sample_mask=sample_mask,
        packed_cu_seqlens=torch.tensor([[0, 2]], dtype=torch.int32).repeat(
            batch_size, 1
        ),
        packed_cu_seqlens_lengths=torch.full((batch_size,), 2, dtype=torch.int64),
        packed_max_seqlens=torch.full((batch_size,), 2, dtype=torch.int64),
        idx=row_ids.tolist(),
        task_name=["validation"] * batch_size,
    )


def _validation_policy(losses: list[float]) -> MagicMock:
    policy = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 64
    policy.train.return_value = {
        "loss": torch.tensor(losses),
        "grad_norm": torch.tensor(0.0),
        "all_mb_metrics": {"global_valid_toks": [1]},
    }
    return policy


def _run_validation(
    policy: MagicMock,
    batches: Iterable[BatchedDataDict],
    execution_mode: str,
    master_config: MasterConfig | None = None,
) -> dict[str, float]:
    metrics, _ = validate(
        policy=policy,
        val_dataloader=batches,
        tokenizer=MagicMock(pad_token_id=0),
        loss_fn=NLLLossFn(),
        step=20,
        master_config=master_config or _validation_config(execution_mode),
        val_batches=4,
        val_batch_size=64,
        val_mbs=1,
    )
    return metrics


def test_sft_validation_execution_mode_defaults_to_per_batch() -> None:
    assert SFTConfig().validation_execution_mode == "per_batch"

    with pytest.raises(ValidationError, match="validation_execution_mode"):
        SFTConfig(validation_execution_mode="unsupported")


@pytest.mark.parametrize(
    ("validation_args", "config_overrides", "message"),
    [
        ({"val_batches": 3}, {}, "val_batches=4"),
        ({"val_batch_size": 32}, {}, "val_global_batch_size=64"),
        ({"val_mbs": 2}, {}, "val_micro_batch_size=1"),
        ({}, {"validation_event_max_payload_bytes": None}, "payload byte budget"),
        (
            {},
            {"validation_event_verified_ray_object_store_available_bytes": None},
            "launcher-verified Ray object-store available bytes",
        ),
        (
            {},
            {"validation_event_memory_safety_multiplier": 1.5},
            "safety multiplier.*at least 2.0",
        ),
    ],
)
def test_validation_event_batch_rejects_invalid_config_before_setup_or_iteration(
    validation_args: dict[str, int],
    config_overrides: dict[str, object],
    message: str,
) -> None:
    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])
    dataloader = MagicMock()
    dataloader.__iter__.side_effect = AssertionError("dataloader was retained")
    arguments = {
        "val_batches": 4,
        "val_batch_size": 64,
        "val_mbs": 1,
        **validation_args,
    }

    with pytest.raises(ValueError, match=message):
        validate(
            policy=policy,
            val_dataloader=dataloader,
            tokenizer=MagicMock(pad_token_id=0),
            loss_fn=NLLLossFn(),
            step=20,
            master_config=_validation_config("event_batch", **config_overrides),
            **arguments,
        )

    policy.prepare_for_training.assert_not_called()
    dataloader.__iter__.assert_not_called()


def test_validation_event_batch_calls_policy_once_in_original_order() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])

    _run_validation(policy, batches, "event_batch")

    policy.train.assert_called_once()
    combined_data, _ = policy.train.call_args.args
    assert combined_data.size == 256
    assert combined_data.size // policy.train.call_args.kwargs["gbs"] == 4
    assert combined_data["idx"] == list(range(256))
    assert torch.equal(combined_data["input_ids"][:, 0], torch.arange(256))
    assert policy.train.call_args.kwargs == {
        "eval_mode": True,
        "gbs": 64,
        "mbs": 1,
    }


def test_validation_event_batch_records_recursive_payload_bytes() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])

    _, timing_metrics = validate(
        policy=policy,
        val_dataloader=batches,
        tokenizer=MagicMock(pad_token_id=0),
        loss_fn=NLLLossFn(),
        step=20,
        master_config=_validation_config("event_batch"),
        val_batches=4,
        val_batch_size=64,
        val_mbs=1,
    )
    combined_data, _ = policy.train.call_args.args

    assert _recursive_tensor_payload_bytes(combined_data) == 23_552
    assert timing_metrics["validation_event_payload_bytes"] == 23_552


def test_validation_event_batch_matches_per_batch_token_weighting() -> None:
    batches = [
        _packed_validation_batch(0, valid_rows=64),
        _packed_validation_batch(1, valid_rows=32),
        _packed_validation_batch(2, valid_rows=16),
        _packed_validation_batch(3, valid_rows=63),
    ]
    losses = [1.0, 2.0, 4.0, 8.0]
    per_batch_policy = _validation_policy([0.0])
    per_batch_policy.train.side_effect = [
        {
            "loss": torch.tensor(loss),
            "grad_norm": torch.tensor(0.0),
            "all_mb_metrics": {"global_valid_toks": [1]},
        }
        for loss in losses
    ]
    event_policy = _validation_policy([loss / 4 for loss in losses])

    per_batch_metrics = _run_validation(per_batch_policy, batches, "per_batch")
    event_metrics = _run_validation(event_policy, batches, "event_batch")

    assert float(event_metrics["val_loss"]) == pytest.approx(
        float(per_batch_metrics["val_loss"])
    )
    assert float(event_metrics["val_loss"]) == pytest.approx(
        (1.0 * 128 + 2.0 * 64 + 4.0 * 32 + 8.0 * 126) / (128 + 64 + 32 + 126)
    )


def test_validation_event_batch_preserves_zero_valid_token_behavior() -> None:
    batches = [
        _packed_validation_batch(batch_idx, valid_rows=0) for batch_idx in range(4)
    ]
    policy = _validation_policy([0.0, 0.0, 0.0, 0.0])
    policy.train.return_value["all_mb_metrics"] = {}

    with pytest.warns(UserWarning, match="No validation metrics were collected"):
        metrics = _run_validation(policy, batches, "event_batch")

    assert metrics == {"val_loss": 0.0}


def test_validation_event_batch_rejects_wrong_batch_count() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(3)]
    policy = _validation_policy([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="exactly 4 validation batches"):
        _run_validation(policy, batches, "event_batch")

    policy.train.assert_not_called()


def test_validation_event_batch_rejects_inconsistent_packed_metadata() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    batches[2]["packed_cu_seqlens"] = batches[2]["packed_cu_seqlens"].to(torch.int64)
    policy = _validation_policy([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="packed_cu_seqlens.*dtype"):
        _run_validation(policy, batches, "event_batch")

    policy.train.assert_not_called()


def test_event_batch_rejects_partial_packed_batch_before_mutation() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(3)]
    partial_batch = _packed_validation_batch(3, batch_size=63, valid_rows=63)
    batches.append(partial_batch)
    policy = _validation_policy([1.0, 2.0, 3.0, 4.0])
    policy.sharding_annotations.get_axis_size.side_effect = AssertionError(
        "generic padding was called"
    )

    with pytest.raises(
        ValueError,
        match="Packed event_batch validation requires batch size 64; got 63",
    ):
        _run_validation(policy, batches, "event_batch")

    assert all(
        len(value) == 63
        for value in partial_batch.values()
        if torch.is_tensor(value) or isinstance(value, list)
    )
    policy.train.assert_not_called()


def test_validation_event_batch_enforces_payload_budget_before_train() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])
    config = _validation_config(
        "event_batch", validation_event_max_payload_bytes=23_551
    )

    with pytest.raises(MemoryError, match="payload budget"):
        _run_validation(policy, batches, "event_batch", config)

    policy.train.assert_not_called()


@pytest.mark.parametrize(
    ("host_available", "ray_available", "message"),
    [
        (47_103, 1_000_000, "host available memory"),
        (1_000_000, 47_103, "Ray object-store available memory"),
    ],
)
def test_validation_event_batch_enforces_memory_headroom_before_train(
    host_available: int,
    ray_available: int,
    message: str,
) -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])
    config = _validation_config(
        "event_batch",
        validation_event_verified_ray_object_store_available_bytes=ray_available,
    )

    with (
        patch(
            "nemo_rl.algorithms.sft.psutil.virtual_memory",
            return_value=SimpleNamespace(available=host_available),
        ) as virtual_memory,
        pytest.raises(MemoryError, match=message),
    ):
        _run_validation(policy, batches, "event_batch", config)

    virtual_memory.assert_called_once_with()
    policy.train.assert_not_called()


def test_validation_event_batch_releases_source_batches_before_train() -> None:
    source_refs: list[weakref.ReferenceType[BatchedDataDict]] = []

    class NonRetainingIterator:
        def __init__(self) -> None:
            self.batch_idx = 0

        def __iter__(self) -> Iterator[BatchedDataDict]:
            return self

        def __next__(self) -> BatchedDataDict:
            if self.batch_idx == 4:
                raise StopIteration
            batch = _packed_validation_batch(self.batch_idx)
            self.batch_idx += 1
            source_refs.append(weakref.ref(batch))
            return batch

    policy = _validation_policy([0.25, 0.5, 0.75, 1.0])
    result = policy.train.return_value

    def assert_sources_released(*_args: object, **_kwargs: object) -> dict[str, object]:
        gc.collect()
        assert all(source_ref() is None for source_ref in source_refs)
        return result

    policy.train.side_effect = assert_sources_released

    _run_validation(policy, NonRetainingIterator(), "event_batch")

    assert len(source_refs) == 4


def test_validation_per_batch_keeps_legacy_policy_calls() -> None:
    batches = [_packed_validation_batch(batch_idx) for batch_idx in range(4)]
    policy = _validation_policy([0.0])
    policy.train.side_effect = [
        {
            "loss": torch.tensor(float(batch_idx + 1)),
            "grad_norm": torch.tensor(0.0),
            "all_mb_metrics": {"global_valid_toks": [1]},
        }
        for batch_idx in range(4)
    ]

    _run_validation(policy, batches, "per_batch")

    assert policy.train.call_count == 4
    for batch, call in zip(batches, policy.train.call_args_list):
        assert call.args[0] is batch
        assert call.kwargs == {"eval_mode": True, "gbs": 64, "mbs": 1}


def test_sft_collate_validates_policy_context_parallel_size():
    collate_fn = _build_sft_collate_fn(
        {
            "megatron_cfg": {
                "enabled": True,
                "context_parallel_size": 16,
            }
        }
    )

    assert collate_fn.func.__name__ == "rl_collate_fn"
    assert collate_fn.keywords == {"megatron_sft_context_parallel_size": 16}


def test_iter_timed_batches_records_each_data_fetch():
    timer = Timer()

    assert list(_iter_timed_batches(["first", "second"], timer)) == [
        "first",
        "second",
    ]
    assert len(timer.get_elapsed("data_fetch")) == 2


def test_add_e2e_step_timing_includes_data_fetch():
    timing_metrics = {"total_step_time": 5.0, "data_fetch": 2.0}

    _add_e2e_step_timing(timing_metrics)

    assert timing_metrics["e2e_step_time"] == 7.0


def test_measure_loop_interval_uses_consecutive_boundaries():
    current_boundary, interval = _measure_loop_interval(11.5, 18.0)

    assert current_boundary == 18.0
    assert interval == 6.5
    assert _measure_loop_interval(None, 11.5) == (11.5, None)


def test_optional_float_converts_scalar_tensor_and_omits_non_scalar() -> None:
    assert _optional_float(torch.tensor(2.5)) == 2.5
    assert _optional_float(torch.tensor([2.5])) is None


def test_optional_float_omits_cuda_tensor_without_synchronizing() -> None:
    cuda_tensor = MagicMock(spec=torch.Tensor)
    cuda_tensor.ndim = 0
    cuda_tensor.device = torch.device("cuda")

    assert _optional_float(cuda_tensor) is None
    cuda_tensor.item.assert_not_called()


def test_validate_preserves_synthetic_zero_and_returns_two_tuple(mock_components):
    val_batch = BatchedDataDict(
        {
            "packed_cu_seqlens": torch.tensor([[0, 1]]),
            "sample_mask": torch.zeros(1),
            "token_mask": torch.zeros((1, 1)),
        }
    )
    policy = mock_components["policy"]
    policy.train.return_value = {
        "loss": torch.tensor(0.0),
        "all_mb_metrics": {},
    }

    with pytest.warns(UserWarning, match="No validation metrics were collected"):
        result = validate(
            policy,
            [val_batch],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            step=1,
            master_config=mock_components["master_config"],
            val_batches=1,
            val_batch_size=1,
            val_mbs=1,
        )

    assert len(result) == 2
    val_metrics, _ = result
    assert val_metrics == {"val_loss": 0.0}


def test_validation_comparison_instrumentation_records_local_subtimings():
    val_data = BatchedDataDict(
        {
            "packed_cu_seqlens": torch.tensor([[0, 2]], dtype=torch.int32),
            "sample_mask": torch.ones(1),
            "token_mask": torch.ones(1, 2),
        }
    )
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(0.0),
        "all_mb_metrics": {"loss": [torch.tensor(0.5)]},
        "evaluation_timings": {
            "worker_state_transition_s": 0.1,
            "forward_s": 0.2,
            "metric_reduction_s": 0.3,
            "state_restore_s": 0.4,
        },
    }
    master_config = MasterConfig.model_construct(
        sft=SFTConfig.model_construct(val_period=20, only_unmask_final=False),
        policy={"make_sequence_length_divisible_by": 1},
    )

    _, timings = validate(
        policy,
        [val_data],
        MagicMock(pad_token_id=0),
        NLLLossFn(),
        step=20,
        master_config=master_config,
        val_batches=1,
        val_batch_size=1,
        val_mbs=1,
        comparison_instrumentation_enabled=True,
    )

    assert isinstance(policy.train.call_args.kwargs["timer"], Timer)
    assert timings["worker_state_transition_s"] == pytest.approx(0.1)
    assert timings["forward_s"] == pytest.approx(0.2)
    assert timings["metric_reduction_s"] == pytest.approx(0.3)
    assert timings["state_restore_s"] == pytest.approx(0.4)
    assert "data_fetch_s" in timings
    assert "data_processing_s" in timings


def test_validation_default_does_not_request_comparison_instrumentation():
    val_data = BatchedDataDict(
        {
            "packed_cu_seqlens": torch.tensor([[0, 2]], dtype=torch.int32),
            "sample_mask": torch.ones(1),
            "token_mask": torch.ones(1, 2),
        }
    )
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(0.0),
        "all_mb_metrics": {"loss": [torch.tensor(0.5)]},
    }
    master_config = MasterConfig.model_construct(
        sft=SFTConfig.model_construct(val_period=20, only_unmask_final=False),
        policy={"make_sequence_length_divisible_by": 1},
    )

    _, timings = validate(
        policy,
        [val_data],
        MagicMock(pad_token_id=0),
        NLLLossFn(),
        step=20,
        master_config=master_config,
        val_batches=1,
        val_batch_size=1,
        val_mbs=1,
    )

    assert "timer" not in policy.train.call_args.kwargs
    assert "data_fetch_s" not in timings
    assert "data_processing_s" not in timings


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"].sft.max_num_steps = 12

    sft_save_state = _initial_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we only trained for 12 steps.
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"].sft.max_num_epochs = 2
    mock_components["master_config"].sft.max_num_steps = 100

    sft_save_state = _initial_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches).
    assert mock_components["policy"].train.call_count == 20


def test_exit_on_timeout(mock_components, capsys):
    """Test that training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    mock_components["master_config"].sft.max_num_steps = 100
    mock_components["master_config"].sft.max_num_epochs = 10

    sft_save_state = _initial_sft_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.sft.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        sft_train(
            mock_components["policy"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            mock_components["logger"],
            mock_components["checkpointer"],
            sft_save_state,
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_components["policy"].train.call_count == 8

        # Verify the timeout message was printed and is near the end (not followed by more training)
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Verify no new epoch started after timeout (which would indicate a bug where break was used instead of return)
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )


def test_training_with_disabled_validation(mock_components):
    """Test that training works when validation is disabled (val_dataloader=None, val_period<=0)"""
    mock_components["master_config"].sft.val_period = 0
    mock_components["master_config"].sft.max_num_steps = 5
    mock_components["master_config"].sft.max_num_epochs = 1

    sft_save_state = _initial_sft_save_state()

    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        None,  # val_dataloader is None
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    assert mock_components["policy"].train.call_count == 5


def test_training_with_negative_val_period(mock_components):
    """Test that training works when val_period is negative (validation disabled)"""
    mock_components["master_config"].sft.val_period = -1
    mock_components["master_config"].sft.max_num_steps = 3
    mock_components["master_config"].sft.max_num_epochs = 1

    sft_save_state = _initial_sft_save_state()

    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        None,  # val_dataloader is None
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    assert mock_components["policy"].train.call_count == 3


@pytest.mark.parametrize(
    (
        "validation_timing",
        "has_valid_tokens",
        "expected_validation_time",
        "expected_validation_loss",
    ),
    [
        pytest.param(
            torch.tensor(126.99),
            True,
            pytest.approx(126.99),
            0.6,
            id="scalar-tensors",
        ),
        pytest.param(
            torch.tensor(126.99),
            False,
            pytest.approx(126.99),
            None,
            id="synthetic-zero-loss",
        ),
    ],
)
def test_training_logs_exact_comparison_payload_and_preserves_native_metrics(
    mock_components,
    validation_timing,
    has_valid_tokens,
    expected_validation_time,
    expected_validation_loss,
):
    """SFT normalizes one payload without changing native metric calls."""

    class FixedTimer:
        def __init__(self) -> None:
            self.labels: set[str] = set()

        @contextmanager
        def time(self, name: str) -> Generator[None, None, None]:
            self.labels.add(name)
            yield

        def record_elapsed(self, name: str, _elapsed: float) -> None:
            self.labels.add(name)

        def get_timing_metrics(
            self, reduction_op: str
        ) -> dict[str, float | torch.Tensor]:
            assert reduction_op == "sum"
            if "total_validation_time" in self.labels:
                return {"total_validation_time": validation_timing}
            return {
                "policy_training": 55.28,
                "total_step_time": 60.0,
                "data_fetch": 2.0,
            }

        def reset(self) -> None:
            return None

    mock_components["master_config"].sft.max_num_steps = 1
    mock_components["master_config"].sft.max_num_epochs = 1
    mock_components["master_config"].sft.val_period = 1
    policy = mock_components["policy"]
    train_result = policy.train.return_value
    train_result["all_mb_metrics"]["lr"] = [4.2e-7]
    validation_result = {
        "loss": torch.tensor(0.6),
        "all_mb_metrics": {"loss": [0.6]} if has_valid_tokens else {},
    }
    policy.train.side_effect = [train_result, validation_result]
    logger = mock_components["logger"]
    logger.comparison_metrics_enabled = True
    val_batch = BatchedDataDict(
        {
            "packed_cu_seqlens": torch.tensor([[0, 1]]),
            "sample_mask": torch.ones(1),
            "token_mask": torch.ones((1, 1)),
        }
    )

    warning_context = (
        nullcontext()
        if has_valid_tokens
        else pytest.warns(UserWarning, match="No validation metrics were collected")
    )
    with warning_context, patch("nemo_rl.algorithms.sft.Timer", FixedTimer):
        sft_train(
            policy,
            mock_components["train_dataloader"],
            [val_batch],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            logger,
            mock_components["checkpointer"],
            _initial_sft_save_state(),
        )

    assert logger.define_metric.call_args_list == [
        call("comparison/step"),
        call("performance/*", step_metric="comparison/step"),
        call("throughput/*", step_metric="comparison/step"),
        call("accuracy/*", step_metric="comparison/step"),
        call("context/*", step_metric="comparison/step"),
    ]

    log_calls = logger.log_metrics.call_args_list
    assert len(log_calls) == 5
    native_validation_timings = log_calls[0].args[0]
    assert native_validation_timings["total_validation_time"] is validation_timing
    assert log_calls[0].args[1] == 1
    assert log_calls[0].kwargs == {"prefix": "timing/validation"}
    native_validation_metrics = log_calls[1].args[0]
    if has_valid_tokens:
        assert native_validation_metrics["val_loss"].item() == pytest.approx(0.6)
    else:
        assert native_validation_metrics == {"val_loss": 0.0}
    assert log_calls[1].args[1] == 1
    assert log_calls[1].kwargs == {"prefix": "validation"}
    assert log_calls[2].args == (
        {
            "loss": 0.5,
            "grad_norm": 1.0,
            "global_valid_toks": 10.0,
            "lr": 4.2e-7,
        },
        1,
    )
    assert log_calls[2].kwargs == {"prefix": "train"}
    assert log_calls[3].args[1] == 1
    assert log_calls[3].kwargs == {"prefix": "timing/train"}

    expected_comparison_payload = {
        "comparison/step": 1,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": 62.0,
        "accuracy/main_lm_loss": 0.5,
        "accuracy/grad_norm": 1.0,
        "accuracy/learning_rate": 4.2e-7,
        "throughput/processed_tokens_per_second": pytest.approx(3 / 55.28),
        "throughput/processed_tokens_per_second_per_gpu": pytest.approx(3 / 55.28 / 2),
        "context/processed_tokens": 3,
        "context/num_gpus": 2,
        "context/is_validation_step": 1,
    }
    if expected_validation_time is not None:
        expected_comparison_payload["performance/validation_time_s"] = (
            expected_validation_time
        )
    if expected_validation_loss is not None:
        expected_comparison_payload["accuracy/validation_loss"] = (
            expected_validation_loss
        )

    comparison_payload = log_calls[4].args[0]
    assert comparison_payload == expected_comparison_payload
    assert all(isinstance(value, (float, int)) for value in comparison_payload.values())
    assert log_calls[4].args[1] == 1
    assert log_calls[4].kwargs == {
        "step_metric": "comparison/step",
        "step_finished": True,
    }
