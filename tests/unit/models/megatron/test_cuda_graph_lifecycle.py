import importlib
import importlib.util
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
    ) -> None:
        self.name = name
        self.activate_calls = 0
        self.reset_calls = 0
        self._events = events
        self._activate_error = activate_error
        self._reset_error = reset_error

    def activate(self) -> None:
        self.activate_calls += 1
        if self._events is not None:
            self._events.append(f"activate:{self.name}")
        if self._activate_error is not None:
            raise self._activate_error

    def reset(self) -> None:
        self.reset_calls += 1
        if self._events is not None:
            self._events.append(f"reset:{self.name}")
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
    )


def test_cuda_graph_lifecycle_api_is_discoverable() -> None:
    lifecycle_module = _get_lifecycle_module()

    assert callable(getattr(lifecycle_module, "TECudaGraphBankProtocol", None))
    assert callable(getattr(lifecycle_module, "TECudaGraphScheduleKey", None))
    assert callable(getattr(lifecycle_module, "TECudaGraphEnsureResult", None))
    assert callable(getattr(lifecycle_module, "TECudaGraphLifecycle", None))


@pytest.mark.parametrize("num_microbatches", [1, 2, 17])
def test_pp1_normalizes_every_positive_runtime_count(
    num_microbatches: int,
) -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=1,
        num_microbatches=num_microbatches,
    )

    assert key.num_microbatches == 1


def test_pipeline_parallel_schedule_keeps_runtime_count() -> None:
    lifecycle_module = _get_lifecycle_module()

    key = lifecycle_module.TECudaGraphScheduleKey.from_runtime(
        pipeline_parallel_size=4,
        num_microbatches=7,
    )

    assert key.num_microbatches == 7


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

    first_result = lifecycle.ensure_active(
        key,
        lambda: _capture(bank, capture_calls),
    )
    lifecycle.record_optimizer_step(successful=False)
    second_result = lifecycle.ensure_active(
        key,
        lambda: _capture(bank, capture_calls),
    )

    assert first_result.status == "warming"
    assert second_result.status == "warming"
    assert capture_calls == []
    assert bank.activate_calls == 0
    assert bank.reset_calls == 0

    lifecycle.record_optimizer_step(successful=True)
    captured = lifecycle.ensure_active(
        key,
        lambda: _capture(bank, capture_calls),
    )
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


def test_capacity_two_evicts_only_lru_inactive_bank_after_replacement_activates() -> (
    None
):
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
        first_key,
        lambda: _capture(first_bank, capture_calls),
    )
    second_result = lifecycle.ensure_active(
        second_key,
        lambda: _capture(second_bank, capture_calls),
    )
    third_result = lifecycle.ensure_active(
        first_key,
        lambda: _capture(recaptured_first_bank, capture_calls),
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


def test_activation_error_wins_when_uncommitted_bank_cleanup_also_fails() -> None:
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


def test_close_resets_every_distinct_cached_bank_once_and_is_idempotent() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=3, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    third_key = _make_key(key_type, 7)
    shared_bank = _FakeBank("shared")
    other_bank = _FakeBank("other")

    lifecycle.ensure_active(first_key, lambda: shared_bank)
    lifecycle.ensure_active(second_key, lambda: shared_bank)
    lifecycle.ensure_active(third_key, lambda: other_bank)

    lifecycle.close()
    lifecycle.close()

    assert shared_bank.reset_calls == 1
    assert other_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()


def test_close_continues_after_reset_error_and_leaves_lifecycle_clear() -> None:
    lifecycle_module = _get_lifecycle_module()
    key_type = lifecycle_module.TECudaGraphScheduleKey
    lifecycle = lifecycle_module.TECudaGraphLifecycle(capacity=2, warmup_steps=0)
    first_key = _make_key(key_type, 5)
    second_key = _make_key(key_type, 3)
    first_bank = _FakeBank("first", reset_error=RuntimeError("reset failed"))
    second_bank = _FakeBank("second")

    lifecycle.ensure_active(first_key, lambda: first_bank)
    lifecycle.ensure_active(second_key, lambda: second_bank)

    with pytest.raises(RuntimeError, match="reset failed"):
        lifecycle.close()

    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
    assert lifecycle.active_key is None
    assert lifecycle.cached_keys == ()

    lifecycle.close()
    assert first_bank.reset_calls == 1
    assert second_bank.reset_calls == 1
