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

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss import NLLLossFn
from nemo_rl.algorithms.sft import (
    MasterConfig,
    SFTConfig,
    _add_e2e_step_timing,
    _build_sft_collate_fn,
    _initial_sft_save_state,
    _iter_timed_batches,
    _measure_loop_interval,
    sft_train,
)
from nemo_rl.utils.sft_comparison_metrics import SFTComparisonObservation
from nemo_rl.utils.timer import Timer


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


def test_training_logs_one_comparison_payload_with_custom_axis(mock_components):
    """SFT emits one normalized comparison payload after validation completes."""

    class FixedTimer:
        @contextmanager
        def time(self, _name: str) -> Generator[None, None, None]:
            yield

        def record_elapsed(self, _name: str, _elapsed: float) -> None:
            return None

        def get_timing_metrics(self, reduction_op: str) -> dict[str, float]:
            assert reduction_op == "sum"
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
    mock_components["policy"].train.return_value["all_mb_metrics"]["lr"] = [4.2e-7]
    logger = mock_components["logger"]
    logger.comparison_metrics_enabled = True
    comparison_payload = {"comparison/step": 1, "context/is_validation_step": 1}

    with (
        patch("nemo_rl.algorithms.sft.Timer", FixedTimer),
        patch(
            "nemo_rl.algorithms.sft.validate",
            return_value=(
                {"val_loss": 0.6},
                {"total_validation_time": 126.99},
            ),
        ),
        patch(
            "nemo_rl.algorithms.sft.build_sft_comparison_metrics",
            return_value=comparison_payload,
        ) as build_metrics,
    ):
        sft_train(
            mock_components["policy"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            logger,
            mock_components["checkpointer"],
            _initial_sft_save_state(),
        )

    logger.define_metric.assert_has_calls(
        [
            call("comparison/step"),
            call("performance/*", step_metric="comparison/step"),
            call("accuracy/*", step_metric="comparison/step"),
            call("context/*", step_metric="comparison/step"),
        ]
    )
    build_metrics.assert_called_once_with(
        SFTComparisonObservation(
            step=1,
            train_step_time_s=55.28,
            e2e_step_time_s=62.0,
            validation_time_s=126.99,
            main_lm_loss=0.5,
            validation_loss=0.6,
            grad_norm=1.0,
            learning_rate=4.2e-7,
        )
    )
    logger.log_metrics.assert_any_call(
        comparison_payload,
        1,
        step_metric="comparison/step",
    )
