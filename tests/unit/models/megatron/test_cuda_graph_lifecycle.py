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

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from types import ModuleType
from typing import Any, NoReturn

import pytest

_MODULE_NAME = "nemo_rl.models.megatron.cuda_graph_lifecycle"


class _FakeBank:
    def __init__(
        self,
        name: str,
        *,
        events: list[str] | None = None,
        activate_error: Exception | None = None,
        reset_error: Exception | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.activate_calls = 0
        self.reset_calls = 0
        self.activate_error = activate_error
        self._events = events
        self._reset_error = reset_error
        self._on_reset = on_reset

    def activate(self) -> None:
        self.activate_calls += 1
        if self._events is not None:
            self._events.append(f"activate:{self.name}")
        if self.activate_error is not None:
            raise self.activate_error

    def reset(self) -> None:
        self.reset_calls += 1
        if self._events is not None:
            self._events.append(f"reset:{self.name}")
        if self._on_reset is not None:
            self._on_reset()
        if self._reset_error is not None:
            raise self._reset_error


def _get_lifecycle_module() -> ModuleType:
    module_spec = importlib.util.find_spec(_MODULE_NAME)
    assert module_spec is not None, f"{_MODULE_NAME} is not implemented"
    return importlib.import_module(_MODULE_NAME)


def _unexpected_capture() -> NoReturn:
    raise AssertionError("capture callback must not be called")


def _capture(bank: _FakeBank, calls: list[str] | None = None) -> _FakeBank:
    if calls is not None:
        calls.append(bank.name)
    return bank


def _make_key(schedule_key_type: Any, num_microbatches: int) -> Any:
    return schedule_key_type.from_runtime(
        pipeline_parallel_size=2,
        num_microbatches=num_microbatches,
        overlap_moe_expert_parallel_comm=False,
    )


def test_cuda_graph_lifecycle_api_is_discoverable() -> None:
    lifecycle_module = _get_lifecycle_module()

    assert callable(getattr(lifecycle_module, "TECudaGraphBankProtocol", None))
    assert callable(getattr(lifecycle_module, "TECudaGraphScheduleKey", None))
    assert callable(getattr(lifecycle_module, "TECudaGraphEnsureResult", None))
    assert callable(getattr(lifecycle_module, "CudaGraphStepMetrics", None))
    lifecycle_type = getattr(lifecycle_module, "TECudaGraphLifecycle", None)
    assert callable(lifecycle_type)
    lifecycle = lifecycle_type(capacity=2, warmup_steps=3)
    assert callable(getattr(lifecycle, "reset_banks", None))


def test_schedule_overlap_defaults_to_disabled() -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=1,
        num_microbatches=7,
    )

    assert key.num_microbatches == 1


@pytest.mark.parametrize("num_microbatches", [1, 2, 17])
def test_pp1_without_overlap_normalizes_every_positive_runtime_count(
    num_microbatches: int,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=1,
        num_microbatches=num_microbatches,
        overlap_moe_expert_parallel_comm=False,
    )

    assert key.num_microbatches == 1


@pytest.mark.parametrize("num_microbatches", [1, 2, 17])
def test_pp1_with_overlap_keeps_runtime_count(num_microbatches: int) -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=1,
        num_microbatches=num_microbatches,
        overlap_moe_expert_parallel_comm=True,
    )

    assert key.num_microbatches == num_microbatches


@pytest.mark.parametrize("overlap", [False, True])
def test_pipeline_parallel_schedule_keeps_runtime_count(overlap: bool) -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=4,
        num_microbatches=7,
        overlap_moe_expert_parallel_comm=overlap,
    )

    assert key.num_microbatches == 7


@pytest.mark.parametrize("overlap", [0, 1, None, "true"])
def test_schedule_key_rejects_nonboolean_overlap(overlap: object) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(TypeError, match="overlap_moe_expert_parallel_comm"):
        lifecycle_module.TECudaGraphScheduleKey.from_runtime(
            pipeline_parallel_size=1,
            num_microbatches=2,
            overlap_moe_expert_parallel_comm=overlap,
        )


@pytest.mark.parametrize(
    ("field_name", "pipeline_parallel_size", "num_microbatches"),
    [
        ("pipeline_parallel_size", 0, 1),
        ("pipeline_parallel_size", -1, 1),
        ("num_microbatches", 1, 0),
        ("num_microbatches", 1, -1),
    ],
)
def test_schedule_key_rejects_nonpositive_runtime_counts(
    field_name: str,
    pipeline_parallel_size: int,
    num_microbatches: int,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(ValueError, match=field_name):
        lifecycle_module.TECudaGraphScheduleKey.from_runtime(
            pipeline_parallel_size=pipeline_parallel_size,
            num_microbatches=num_microbatches,
            overlap_moe_expert_parallel_comm=False,
        )


@pytest.mark.parametrize(
    ("field_name", "pipeline_parallel_size", "num_microbatches"),
    [
        ("pipeline_parallel_size", True, 1),
        ("pipeline_parallel_size", 1.0, 1),
        ("num_microbatches", 1, False),
        ("num_microbatches", 1, 1.0),
    ],
)
def test_schedule_key_rejects_noninteger_runtime_counts(
    field_name: str,
    pipeline_parallel_size: object,
    num_microbatches: object,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(TypeError, match=field_name):
        lifecycle_module.TECudaGraphScheduleKey.from_runtime(
            pipeline_parallel_size=pipeline_parallel_size,
            num_microbatches=num_microbatches,
            overlap_moe_expert_parallel_comm=False,
        )


@pytest.mark.parametrize("num_microbatches", [0, -1])
def test_schedule_key_constructor_rejects_nonpositive_counts(
    num_microbatches: int,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(ValueError, match="num_microbatches"):
        lifecycle_module.TECudaGraphScheduleKey(num_microbatches)


@pytest.mark.parametrize("num_microbatches", [True, 1.0])
def test_schedule_key_constructor_rejects_noninteger_counts(
    num_microbatches: object,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(TypeError, match="num_microbatches"):
        lifecycle_module.TECudaGraphScheduleKey(num_microbatches)


def test_schedule_and_ensure_results_are_immutable() -> None:
    lifecycle_module = _get_lifecycle_module()
    key = _make_key(lifecycle_module.TECudaGraphScheduleKey, 2)
    result = lifecycle_module.TECudaGraphEnsureResult(
        key=key,
        status="warming",
        evicted_key=None,
    )

    with pytest.raises(FrozenInstanceError):
        key.num_microbatches = 3
    with pytest.raises(FrozenInstanceError):
        result.status = "captured"


def _metrics_values() -> dict[str, int]:
    return {
        "capture_count": 1,
        "replay_count": 2,
        "cache_hit_count": 3,
        "cache_miss_count": 2,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 5,
        "eligible_calls": 6,
        "logical_tokens": 7,
        "padded_tokens": 8,
        "capacity_tokens": 9,
    }


def test_step_metrics_preserve_raw_counts_and_are_immutable() -> None:
    lifecycle_module = _get_lifecycle_module()
    metrics = lifecycle_module.CudaGraphStepMetrics(**_metrics_values())

    assert {
        field_name: getattr(metrics, field_name) for field_name in _metrics_values()
    } == _metrics_values()
    with pytest.raises(FrozenInstanceError):
        metrics.graph_calls = 0


@pytest.mark.parametrize("field_name", list(_metrics_values()))
def test_step_metrics_reject_negative_counts(field_name: str) -> None:
    lifecycle_module = _get_lifecycle_module()
    values = _metrics_values()
    values[field_name] = -1

    with pytest.raises(ValueError, match=field_name):
        lifecycle_module.CudaGraphStepMetrics(**values)


@pytest.mark.parametrize("invalid_count", [True, 1.0])
@pytest.mark.parametrize("field_name", list(_metrics_values()))
def test_step_metrics_reject_noninteger_counts(
    field_name: str,
    invalid_count: object,
) -> None:
    lifecycle_module = _get_lifecycle_module()
    values: dict[str, object] = _metrics_values()
    values[field_name] = invalid_count

    with pytest.raises(TypeError, match=field_name):
        lifecycle_module.CudaGraphStepMetrics(**values)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"capture_count": 3, "cache_miss_count": 2}, "capture_count"),
        ({"capture_count": 1, "eviction_count": 2}, "eviction_count"),
        ({"graph_calls": 7, "eligible_calls": 6}, "graph_calls"),
        ({"logical_tokens": 9, "padded_tokens": 8}, "logical_tokens"),
        ({"padded_tokens": 10, "capacity_tokens": 9}, "padded_tokens"),
        ({"fallback_count": 1}, "fallback_count"),
    ],
)
def test_step_metrics_reject_impossible_cross_field_counts(
    overrides: dict[str, int],
    match: str,
) -> None:
    lifecycle_module = _get_lifecycle_module()
    values = _metrics_values()
    values.update(overrides)

    with pytest.raises(ValueError, match=match):
        lifecycle_module.CudaGraphStepMetrics(**values)


@pytest.mark.parametrize("capacity", [0, -1])
def test_lifecycle_rejects_nonpositive_capacity(capacity: int) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(ValueError, match="capacity"):
        lifecycle_module.TECudaGraphLifecycle(capacity=capacity)


@pytest.mark.parametrize("capacity", [True, 1.0])
def test_lifecycle_rejects_noninteger_capacity(capacity: object) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(TypeError, match="capacity"):
        lifecycle_module.TECudaGraphLifecycle(capacity=capacity)


def test_lifecycle_rejects_negative_warmup_steps() -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(ValueError, match="warmup_steps"):
        lifecycle_module.TECudaGraphLifecycle(capacity=1, warmup_steps=-1)


@pytest.mark.parametrize("warmup_steps", [True, 1.0])
def test_lifecycle_rejects_noninteger_warmup_steps(warmup_steps: object) -> None:
    lifecycle_module = _get_lifecycle_module()

    with pytest.raises(TypeError, match="warmup_steps"):
        lifecycle_module.TECudaGraphLifecycle(
            capacity=1,
            warmup_steps=warmup_steps,
        )


def test_record_optimizer_step_rejects_nonboolean_success() -> None:
    lifecycle_module = _get_lifecycle_module()
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=1)

    with pytest.raises(TypeError, match="successful"):
        lifecycle.record_optimizer_step(successful=1)


def test_three_global_successes_warm_first_key_then_new_key_captures_immediately() -> (
    None
):
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")
    capture_calls: list[str] = []

    for _ in range(3):
        result = lifecycle.ensure_active(
            first_key,
            lambda: _capture(first_bank, capture_calls),
        )
        assert result == lifecycle_module.TECudaGraphEnsureResult(
            key=first_key,
            status="warming",
            evicted_key=None,
        )
        lifecycle.record_optimizer_step(successful=True)

    first_capture = lifecycle.ensure_active(
        first_key,
        lambda: _capture(first_bank, capture_calls),
    )
    second_capture = lifecycle.ensure_active(
        second_key,
        lambda: _capture(second_bank, capture_calls),
    )

    assert first_capture.status == "captured"
    assert second_capture.status == "captured"
    assert lifecycle.successful_optimizer_steps == 3
    assert capture_calls == ["first", "second"]
    assert first_bank.activate_calls == 1
    assert second_bank.activate_calls == 1


def test_failed_optimizer_step_does_not_advance_warmup_or_mutate_cached_bank() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=1, warmup_steps=1)
    key = _make_key(key_type, 5)
    bank = _FakeBank("bank")
    capture_calls: list[str] = []

    first_result = lifecycle.ensure_active(key, lambda: _capture(bank, capture_calls))
    lifecycle.record_optimizer_step(successful=False)
    second_result = lifecycle.ensure_active(key, lambda: _capture(bank, capture_calls))

    assert first_result.status == "warming"
    assert second_result.status == "warming"
    assert lifecycle.successful_optimizer_steps == 0
    assert capture_calls == []
    assert bank.activate_calls == 0
    assert bank.reset_calls == 0

    lifecycle.record_optimizer_step(successful=True)
    captured = lifecycle.ensure_active(key, lambda: _capture(bank, capture_calls))
    calls_before_failure = (bank.activate_calls, bank.reset_calls)

    lifecycle.record_optimizer_step(successful=False)

    assert captured.status == "captured"
    assert (bank.activate_calls, bank.reset_calls) == calls_before_failure
    assert lifecycle.active_key == key
    assert lifecycle.cached_keys == (key,)


def test_cache_hit_activates_and_becomes_mru_without_capture() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")

    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)
    hit = lifecycle.ensure_active(first_key, _unexpected_capture)

    assert hit == lifecycle_module.TECudaGraphEnsureResult(
        key=first_key,
        status="hit",
        evicted_key=None,
    )
    assert first_bank.activate_calls == 2
    assert second_bank.activate_calls == 1
    assert lifecycle.active_key == first_key
    assert lifecycle.cached_keys == (second_key, first_key)


def test_cached_activation_failure_preserves_active_key_and_lru_order() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")

    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)
    first_bank.activate_error = RuntimeError("cached activation failed")

    with pytest.raises(RuntimeError, match="cached activation failed"):
        lifecycle.ensure_active(first_key, _unexpected_capture)

    assert lifecycle.active_key == second_key
    assert lifecycle.cached_keys == (first_key, second_key)
    assert first_bank.reset_calls == 0
    assert second_bank.reset_calls == 0


def test_capacity_two_evicts_only_lru_after_replacement_activates() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    key_5 = _make_key(key_type, 5)
    key_3 = _make_key(key_type, 3)
    key_7 = _make_key(key_type, 7)
    events: list[str] = []
    bank_5 = _FakeBank("five", events=events)
    bank_3 = _FakeBank("three", events=events)
    bank_7 = _FakeBank("seven", events=events)

    lifecycle.ensure_active(key_5, lambda: bank_5)
    lifecycle.ensure_active(key_3, lambda: bank_3)
    lifecycle.ensure_active(key_5, _unexpected_capture)
    events.clear()

    captured = lifecycle.ensure_active(key_7, lambda: bank_7)

    assert captured == lifecycle_module.TECudaGraphEnsureResult(
        key=key_7,
        status="captured",
        evicted_key=key_3,
    )
    assert events == ["activate:seven", "reset:three"]
    assert bank_3.reset_calls == 1
    assert bank_5.reset_calls == 0
    assert lifecycle.active_key == key_7
    assert lifecycle.cached_keys == (key_5, key_7)


def test_capacity_one_recaptures_alternating_keys() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=1, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")
    recaptured_first_bank = _FakeBank("first-again")
    capture_calls: list[str] = []

    first_result = lifecycle.ensure_active(
        first_key, lambda: _capture(first_bank, capture_calls)
    )
    second_result = lifecycle.ensure_active(
        second_key, lambda: _capture(second_bank, capture_calls)
    )
    third_result = lifecycle.ensure_active(
        first_key, lambda: _capture(recaptured_first_bank, capture_calls)
    )

    assert [first_result.status, second_result.status, third_result.status] == [
        "captured",
        "captured",
        "captured",
    ]
    assert second_result.evicted_key == first_key
    assert third_result.evicted_key == second_key
    assert capture_calls == ["first", "second", "first-again"]
    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
    assert recaptured_first_bank.reset_calls == 0


def test_capture_exception_preserves_cache_active_key_and_lru_order() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    failed_key = _make_key(key_type, 7)
    replacement_key = _make_key(key_type, 9)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")
    replacement_bank = _FakeBank("replacement")

    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)
    lifecycle.ensure_active(first_key, _unexpected_capture)
    expected_keys = lifecycle.cached_keys

    def fail_capture() -> NoReturn:
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        lifecycle.ensure_active(failed_key, fail_capture)

    assert lifecycle.active_key == first_key
    assert lifecycle.cached_keys == expected_keys
    assert first_bank.reset_calls == 0
    assert second_bank.reset_calls == 0

    replacement = lifecycle.ensure_active(replacement_key, lambda: replacement_bank)
    assert replacement.evicted_key == second_key


def test_activation_exception_resets_only_uncommitted_bank_and_preserves_state() -> (
    None
):
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    failed_key = _make_key(key_type, 7)
    replacement_key = _make_key(key_type, 9)
    first_bank = _FakeBank("first")
    second_bank = _FakeBank("second")
    failed_bank = _FakeBank(
        "failed",
        activate_error=RuntimeError("activation failed"),
    )
    replacement_bank = _FakeBank("replacement")

    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)
    lifecycle.ensure_active(first_key, _unexpected_capture)
    expected_keys = lifecycle.cached_keys

    with pytest.raises(RuntimeError, match="activation failed"):
        lifecycle.ensure_active(failed_key, lambda: failed_bank)

    assert failed_bank.activate_calls == 1
    assert failed_bank.reset_calls == 1
    assert lifecycle.active_key == first_key
    assert lifecycle.cached_keys == expected_keys
    assert first_bank.reset_calls == 0
    assert second_bank.reset_calls == 0

    replacement = lifecycle.ensure_active(replacement_key, lambda: replacement_bank)
    assert replacement.evicted_key == second_key


def test_activation_error_wins_when_uncommitted_cleanup_also_fails() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=1, warmup_steps=0)
    active_key = _make_key(key_type, 5)
    failed_key = _make_key(key_type, 3)
    active_bank = _FakeBank("active")
    failed_bank = _FakeBank(
        "failed",
        activate_error=RuntimeError("activation failed"),
        reset_error=RuntimeError("cleanup failed"),
    )

    lifecycle.ensure_active(active_key, lambda: active_bank)

    with pytest.raises(RuntimeError, match="activation failed"):
        lifecycle.ensure_active(failed_key, lambda: failed_bank)

    assert failed_bank.reset_calls == 1
    assert lifecycle.active_key == active_key
    assert lifecycle.cached_keys == (active_key,)
    assert active_bank.reset_calls == 0


def test_eviction_reset_exception_keeps_committed_replacement_active() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=1, warmup_steps=0)
    old_key = _make_key(key_type, 5)
    replacement_key = _make_key(key_type, 3)
    events: list[str] = []
    old_bank = _FakeBank(
        "old",
        events=events,
        reset_error=RuntimeError("reset failed"),
    )
    replacement_bank = _FakeBank("replacement", events=events)

    lifecycle.ensure_active(old_key, lambda: old_bank)
    events.clear()

    with pytest.raises(RuntimeError, match="reset failed"):
        lifecycle.ensure_active(replacement_key, lambda: replacement_bank)

    assert events == ["activate:replacement", "reset:old"]
    assert lifecycle.active_key == replacement_key
    assert lifecycle.cached_keys == (replacement_key,)
    lifecycle.ensure_active(replacement_key, _unexpected_capture)


def test_reset_banks_resets_distinct_banks_and_preserves_warmup() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=3, warmup_steps=3)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    third_key = _make_key(key_type, 7)
    shared_bank = _FakeBank("shared")
    other_bank = _FakeBank("other")

    for _ in range(3):
        lifecycle.record_optimizer_step(successful=True)
    lifecycle.ensure_active(first_key, lambda: shared_bank)
    lifecycle.ensure_active(second_key, lambda: shared_bank)
    lifecycle.ensure_active(third_key, lambda: other_bank)

    lifecycle.reset_banks()
    lifecycle.reset_banks()

    assert shared_bank.reset_calls == 1
    assert other_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()
    assert lifecycle.successful_optimizer_steps == 3

    replacement = _FakeBank("replacement")
    result = lifecycle.ensure_active(first_key, lambda: replacement)
    assert result.status == "captured"


def test_reset_banks_continues_after_error_and_clears_before_callbacks() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=1)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    observations: list[tuple[object, tuple[object, ...], int]] = []

    def observe_clear_state() -> None:
        observations.append(
            (
                lifecycle.active_key,
                lifecycle.cached_keys,
                lifecycle.successful_optimizer_steps,
            )
        )

    first_bank = _FakeBank(
        "first",
        reset_error=RuntimeError("first reset failed"),
        on_reset=observe_clear_state,
    )
    second_bank = _FakeBank("second", on_reset=observe_clear_state)
    lifecycle.record_optimizer_step(successful=True)
    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)

    with pytest.raises(RuntimeError, match="first reset failed"):
        lifecycle.reset_banks()

    assert observations == [(None, (), 1), (None, (), 1)]
    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()
    assert lifecycle.successful_optimizer_steps == 1

    lifecycle.reset_banks()
    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1


def test_close_clears_before_callbacks_and_is_terminal_and_idempotent() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=3, warmup_steps=1)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    third_key = _make_key(key_type, 7)
    observations: list[tuple[object, tuple[object, ...], int]] = []

    def observe_terminal_state() -> None:
        observations.append(
            (
                lifecycle.active_key,
                lifecycle.cached_keys,
                lifecycle.successful_optimizer_steps,
            )
        )
        with pytest.raises(RuntimeError, match="closed"):
            lifecycle.record_optimizer_step(successful=True)
        with pytest.raises(RuntimeError, match="closed"):
            lifecycle.ensure_active(first_key, _unexpected_capture)
        with pytest.raises(RuntimeError, match="closed"):
            lifecycle.reset_banks()

    shared_bank = _FakeBank("shared", on_reset=observe_terminal_state)
    other_bank = _FakeBank("other", on_reset=observe_terminal_state)
    lifecycle.record_optimizer_step(successful=True)
    lifecycle.ensure_active(first_key, lambda: shared_bank)
    lifecycle.ensure_active(second_key, lambda: shared_bank)
    lifecycle.ensure_active(third_key, lambda: other_bank)

    lifecycle.close()
    lifecycle.close()

    assert observations == [(None, (), 0), (None, (), 0)]
    assert shared_bank.reset_calls == 1
    assert other_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()
    assert lifecycle.successful_optimizer_steps == 0

    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.record_optimizer_step(successful=False)
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.ensure_active(first_key, _unexpected_capture)
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.reset_banks()


def test_close_continues_after_reset_errors_and_remains_terminal() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=3, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    third_key = _make_key(key_type, 7)
    first_bank = _FakeBank("first", reset_error=RuntimeError("first reset failed"))
    second_bank = _FakeBank("second", reset_error=RuntimeError("second reset failed"))
    third_bank = _FakeBank("third")
    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)
    lifecycle.ensure_active(third_key, lambda: third_bank)

    with pytest.raises(RuntimeError, match="first reset failed"):
        lifecycle.close()

    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
    assert third_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()
    assert lifecycle.successful_optimizer_steps == 0

    lifecycle.close()
    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
    assert third_bank.reset_calls == 1

    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.record_optimizer_step(successful=True)
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.ensure_active(first_key, _unexpected_capture)
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.reset_banks()
