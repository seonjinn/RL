# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import importlib.util
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4]
    / "nemo_rl"
    / "models"
    / "policy"
    / "workers"
    / "train_timing.py"
)
SPEC = importlib.util.spec_from_file_location("train_timing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_TIMING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_TIMING)
TrainPhaseTimer = TRAIN_TIMING.TrainPhaseTimer
aggregate_train_phase_timings = TRAIN_TIMING.aggregate_train_phase_timings
flatten_train_phase_timings = TRAIN_TIMING.flatten_train_phase_timings
flatten_train_phase_metadata = TRAIN_TIMING.flatten_train_phase_metadata


def _clock(values: list[float]) -> tuple[Iterator[float], Callable[[], float]]:
    iterator = iter(values)
    return iterator, lambda: next(iterator)


def test_train_phase_timer_records_synchronized_elapsed_time():
    _, clock = _clock([10.0, 12.5])
    sync_calls: list[None] = []
    timer = TrainPhaseTimer(
        enabled=True,
        clock=clock,
        synchronize=lambda: sync_calls.append(None),
    )

    with timer.time("forward_backward", synchronize_cuda=True):
        pass

    assert timer.metrics == {"forward_backward": pytest.approx(2.5)}
    assert len(sync_calls) == 2


def test_train_phase_timer_accumulates_repeated_phases():
    _, clock = _clock([1.0, 2.5, 5.0, 7.0])
    timer = TrainPhaseTimer(enabled=True, clock=clock, synchronize=lambda: None)

    with timer.time("data_setup"):
        pass
    with timer.time("data_setup"):
        pass

    assert timer.metrics == {"data_setup": pytest.approx(3.5)}


def test_train_phase_timer_supports_explicit_phase_boundaries():
    _, clock = _clock([2.0, 5.5])
    timer = TrainPhaseTimer(enabled=True, clock=clock, synchronize=lambda: None)

    timer.start("optimizer")
    timer.stop("optimizer")

    assert timer.metrics == {"optimizer": pytest.approx(3.5)}


def test_disabled_train_phase_timer_has_zero_observer_effect():
    clock_calls: list[None] = []
    sync_calls: list[None] = []
    timer = TrainPhaseTimer(
        enabled=False,
        clock=lambda: clock_calls.append(None) or 0.0,
        synchronize=lambda: sync_calls.append(None),
    )

    with timer.time("forward_backward", synchronize_cuda=True):
        pass

    assert timer.metrics == {}
    assert clock_calls == []
    assert sync_calls == []


def test_disabled_explicit_phase_boundaries_have_zero_observer_effect():
    clock_calls: list[None] = []
    sync_calls: list[None] = []
    timer = TrainPhaseTimer(
        enabled=False,
        clock=lambda: clock_calls.append(None) or 0.0,
        synchronize=lambda: sync_calls.append(None),
    )

    timer.start("optimizer", synchronize_cuda=True)
    timer.stop("optimizer", synchronize_cuda=True)

    assert timer.metrics == {}
    assert clock_calls == []
    assert sync_calls == []


@pytest.mark.parametrize(
    ("value", "enabled"), [(None, False), ("0", False), ("1", True)]
)
def test_train_phase_timer_reads_opt_in_environment(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("NRL_MEGATRON_TRAIN_BREAKDOWN", raising=False)
    else:
        monkeypatch.setenv("NRL_MEGATRON_TRAIN_BREAKDOWN", value)

    timer = TrainPhaseTimer.from_env(synchronize=lambda: None)

    assert timer.enabled is enabled


def test_train_phase_timer_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("NRL_MEGATRON_TRAIN_BREAKDOWN", "yes")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        TrainPhaseTimer.from_env(synchronize=lambda: None)


def test_aggregate_train_phase_timings_reports_rank_distribution():
    results = [
        {
            "rank": 8,
            "train_phase_timings": {
                "forward_backward": 4.0,
                "optimizer": 2.0,
                "worker_total": 7.0,
            },
        },
        {
            "rank": 511,
            "train_phase_timings": {
                "forward_backward": 6.0,
                "optimizer": 1.0,
                "worker_total": 9.0,
            },
        },
        {
            "rank": 120,
            "train_phase_timings": {
                "forward_backward": 5.0,
                "optimizer": 3.0,
                "worker_total": 10.0,
            },
        },
    ]

    assert aggregate_train_phase_timings(results) == {
        "forward_backward": {
            "min": pytest.approx(4.0),
            "mean": pytest.approx(5.0),
            "median": pytest.approx(5.0),
            "max": pytest.approx(6.0),
            "max_rank": 511,
            "critical_rank_value": pytest.approx(5.0),
        },
        "optimizer": {
            "min": pytest.approx(1.0),
            "mean": pytest.approx(2.0),
            "median": pytest.approx(2.0),
            "max": pytest.approx(3.0),
            "max_rank": 120,
            "critical_rank_value": pytest.approx(3.0),
        },
        "worker_total": {
            "min": pytest.approx(7.0),
            "mean": pytest.approx(26.0 / 3.0),
            "median": pytest.approx(9.0),
            "max": pytest.approx(10.0),
            "max_rank": 120,
            "critical_rank_value": pytest.approx(10.0),
        },
    }


def test_aggregate_train_phase_timings_requires_same_keys_on_every_rank():
    with pytest.raises(ValueError, match="same phase keys"):
        aggregate_train_phase_timings(
            [
                {"train_phase_timings": {"forward_backward": 4.0}},
                {"train_phase_timings": {"optimizer": 2.0}},
            ]
        )


def test_flatten_train_phase_timings_creates_logger_safe_metrics():
    assert flatten_train_phase_timings(
        {
            "forward_backward": {
                "min": 4.0,
                "mean": 5.0,
                "median": 5.0,
                "max": 6.0,
                "max_rank": 511,
                "critical_rank_value": 4.5,
            }
        }
    ) == {
        "worker_train/forward_backward_min": 4.0,
        "worker_train/forward_backward_mean": 5.0,
        "worker_train/forward_backward_median": 5.0,
        "worker_train/forward_backward_max": 6.0,
        "worker_train/forward_backward_critical_rank_value": 4.5,
    }
    assert flatten_train_phase_metadata(
        {
            "forward_backward": {
                "min": 4.0,
                "mean": 5.0,
                "median": 5.0,
                "max": 6.0,
                "max_rank": 511,
                "critical_rank_value": 4.5,
            },
            "worker_total": {
                "min": 8.0,
                "mean": 9.0,
                "median": 9.0,
                "max": 10.0,
                "max_rank": 120,
                "critical_rank_value": 10.0,
            },
        }
    ) == {
        "worker_train/critical_rank": 120,
        "worker_train/forward_backward_max_rank": 511,
        "worker_train/worker_total_max_rank": 120,
    }


def test_megatron_worker_train_wires_required_phase_boundaries():
    source_path = (
        Path(__file__).parents[4]
        / "nemo_rl"
        / "models"
        / "policy"
        / "workers"
        / "megatron_policy_worker.py"
    )
    tree = ast.parse(source_path.read_text())
    train_method = next(
        node
        for class_node in tree.body
        if isinstance(class_node, ast.ClassDef)
        and class_node.name == "MegatronPolicyWorkerImpl"
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    labels = {
        call.args[0].value
        for call in ast.walk(train_method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"start", "stop"}
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    cuda_sync_calls = [
        call
        for call in ast.walk(train_method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "synchronize"
    ]
    barrier_calls = [
        call
        for call in ast.walk(train_method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "barrier"
    ]

    assert {
        "setup",
        "entry_barrier",
        "entry_cuda_sync",
        "batch_preparation",
        "zero_grad_setup",
        "forward_backward",
        "post_forward_backward",
        "optimizer",
        "model_parallel_reductions",
        "loss_metric_processing",
        "loss_metric_broadcast",
        "eval_state_restore",
        "exit_barrier",
        "exit_cuda_sync",
        "scheduler",
        "aggregate_statistics",
        "result_materialization",
    } <= labels
    assert len(cuda_sync_calls) == 2
    assert len(barrier_calls) == 2
    for label in labels - {"train"}:
        starts = [
            call.lineno
            for call in ast.walk(train_method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "start"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == label
        ]
        stops = [
            call.lineno
            for call in ast.walk(train_method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "stop"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == label
        ]
        assert len(starts) == len(stops) == 1
        assert starts[0] < stops[0]
