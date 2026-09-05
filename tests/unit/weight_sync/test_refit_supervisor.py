import importlib.util
import math
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol

import pytest

TEST_TIMEOUT_S = 10.0


def _load_refit_supervisor() -> ModuleType:
    module_path = Path(__file__).parents[3] / "nemo_rl/weight_sync/refit_supervisor.py"
    spec = importlib.util.spec_from_file_location(
        "_refit_supervisor_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Ref:
    name: str


class _StrSubclass(str):
    pass


class _FloatSubclass(float):
    pass


class _IntSubclass(int):
    pass


class _Participant(Protocol):
    role: str


class _FakeRay:
    def __init__(
        self,
        *,
        completion_order: list[_Ref],
        results: dict[_Ref, object],
        on_wait: Callable[[float], None] | None = None,
        copy_ready_ref: bool = False,
    ) -> None:
        self._completion_order = completion_order
        self._results = results
        self._on_wait = on_wait
        self._copy_ready_ref = copy_ready_ref
        self.wait_calls: list[tuple[tuple[_Ref, ...], int, float]] = []
        self.wait_fetch_local_calls: list[bool | None] = []
        self.get_calls: list[object] = []
        self.get_timeouts: list[float | None] = []

    def wait(
        self,
        refs: list[_Ref],
        *,
        num_returns: int,
        timeout: float,
        fetch_local: bool | None = None,
    ) -> tuple[list[_Ref], list[_Ref]]:
        self.wait_calls.append((tuple(refs), num_returns, timeout))
        self.wait_fetch_local_calls.append(fetch_local)
        if self._on_wait is not None:
            self._on_wait(timeout)
        ready = [
            ref
            for ref in self._completion_order
            if any(ref is candidate for candidate in refs)
        ][:num_returns]
        if not ready:
            return [], refs
        for ref in ready:
            self._completion_order.remove(ref)
        returned_ready = (
            [_Ref(ref.name) for ref in ready] if self._copy_ready_ref else ready
        )
        ready_identities = {id(ref) for ref in ready}
        return returned_ready, [ref for ref in refs if id(ref) not in ready_identities]

    def get(self, ref: object, *, timeout: float | None = None) -> object:
        self.get_calls.append(ref)
        self.get_timeouts.append(timeout)
        refs = ref if type(ref) is list else [ref]
        results: list[object] = []
        for item in refs:
            result = self._results[item]
            if isinstance(result, BaseException):
                raise result
            results.append(result)
        return results if type(ref) is list else results[0]


class _ScriptedRay:
    def __init__(
        self,
        *,
        wait_responses: list[tuple[list[object], list[object]]],
        results: dict[object, object],
    ) -> None:
        self._wait_responses = wait_responses
        self._results = results
        self.wait_calls: list[tuple[tuple[object, ...], int, float]] = []
        self.wait_fetch_local_calls: list[bool | None] = []
        self.get_calls: list[object] = []
        self.get_timeouts: list[float | None] = []

    def wait(
        self,
        refs: list[object],
        *,
        num_returns: int,
        timeout: float,
        fetch_local: bool | None = None,
    ) -> tuple[list[object], list[object]]:
        self.wait_calls.append((tuple(refs), num_returns, timeout))
        self.wait_fetch_local_calls.append(fetch_local)
        return self._wait_responses.pop(0)

    def get(self, ref: object, *, timeout: float | None = None) -> object:
        self.get_calls.append(ref)
        self.get_timeouts.append(timeout)
        refs = ref if type(ref) is list else [ref]
        results: list[object] = []
        for item in refs:
            result = self._results[item]
            if isinstance(result, BaseException):
                raise result
            results.append(result)
        return results if type(ref) is list else results[0]


@dataclass
class _FakeClock:
    now: float

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_consumer_failure_is_observed_without_waiting_for_never_ready_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    producer = _Ref("producer-never-ready")
    consumer = _Ref("consumer-failed")
    fake_ray = _FakeRay(
        completion_order=[consumer],
        results={consumer: False},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="sync-weights",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    error = exc_info.value
    assert error.operation == "sync-weights"
    assert error.participant.role == "consumer"
    assert error.participant.index == 0
    assert "returned False" in str(error)
    assert fake_ray.wait_calls == [
        ((consumer, producer), 1, TEST_TIMEOUT_S),
        ((producer,), 1, 0.0),
    ]
    assert fake_ray.get_calls == [[consumer]]


def test_producer_exception_preserves_cause_without_waiting_for_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-failed")
    consumer = _Ref("consumer-never-ready")
    root_cause = ValueError("collective sender exploded")
    fake_ray = _FakeRay(
        completion_order=[producer],
        results={producer: root_cause},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="sync-weights",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    error = exc_info.value
    assert error.operation == "sync-weights"
    assert error.participant.role == "producer"
    assert error.participant.index == 0
    assert error.__cause__ is root_cause
    assert "collective sender exploded" in str(error)
    assert fake_ray.wait_calls == [
        ((consumer, producer), 1, TEST_TIMEOUT_S),
        ((consumer,), 1, 0.0),
    ]
    assert fake_ray.get_calls == [[producer], producer]


def test_producer_false_is_observed_without_waiting_for_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-failed")
    consumer = _Ref("consumer-never-ready")
    fake_ray = _FakeRay(
        completion_order=[producer],
        results={producer: False},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="sync-weights",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "producer"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "producer returned False; expected exactly None or True" in str(
        exc_info.value
    )
    assert fake_ray.get_calls == [[producer]]


def test_consumer_exception_preserves_cause_without_waiting_for_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-never-ready")
    consumer = _Ref("consumer-failed")
    root_cause = RuntimeError("destination load exploded")
    fake_ray = _FakeRay(
        completion_order=[consumer],
        results={consumer: root_cause},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="sync-weights",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "consumer"
    assert exc_info.value.__cause__ is root_cause
    assert fake_ray.get_calls == [[consumer], consumer]


def test_batch_get_fallback_preserves_consumer_first_result_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-remote-error")
    consumer = _Ref("consumer-invalid-result")
    producer_failure = RuntimeError("producer exploded")
    fake_ray = _FakeRay(
        completion_order=[consumer, producer],
        results={consumer: False, producer: producer_failure},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="consumer-first-fallback",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "consumer"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert fake_ray.get_calls == [[consumer, producer], consumer]


def test_all_participants_succeed_in_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer_zero = _Ref("producer-0")
    producer_one = _Ref("producer-1")
    consumer_zero = _Ref("consumer-0")
    consumer_one = _Ref("consumer-1")
    completion_order = [consumer_one, producer_zero, consumer_zero, producer_one]
    fake_ray = _FakeRay(
        completion_order=completion_order.copy(),
        results={
            producer_zero: True,
            producer_one: True,
            consumer_zero: True,
            consumer_one: True,
        },
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    summary = refit_supervisor.supervise_refit_futures(
        operation="sync-weights",
        producer_futures=[producer_zero, producer_one],
        consumer_futures=[consumer_zero, consumer_one],
        result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
        timeout_s=TEST_TIMEOUT_S,
    )

    assert summary.operation == "sync-weights"
    assert summary.producer_count == 2
    assert summary.consumer_count == 2
    assert tuple((item.role, item.index) for item in summary.completion_order) == (
        ("consumer", 0),
        ("consumer", 1),
        ("producer", 0),
        ("producer", 1),
    )
    assert [call[1:] for call in fake_ray.wait_calls] == [
        (1, TEST_TIMEOUT_S),
        (3, 0.0),
    ]
    assert fake_ray.wait_fetch_local_calls == [True, True]
    assert fake_ray.get_calls == [
        [consumer_zero, consumer_one, producer_zero, producer_one]
    ]
    assert all(
        timeout is not None and 0.0 < timeout <= TEST_TIMEOUT_S
        for timeout in fake_ray.get_timeouts
    )
    with pytest.raises(FrozenInstanceError):
        summary.operation = "mutated"


@pytest.mark.parametrize(
    "invalid_result",
    [None, 0, 1, "true", object()],
    ids=["none", "zero", "integer-one", "string", "object"],
)
def test_terminal_result_must_be_exact_true(
    monkeypatch: pytest.MonkeyPatch,
    invalid_result: object,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer")
    consumer = _Ref("consumer")
    fake_ray = _FakeRay(
        completion_order=[producer],
        results={producer: invalid_result},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(
        refit_supervisor.RefitParticipantFailure,
        match="expected exactly True",
    ) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="strict-ack",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.operation == "strict-ack"
    assert exc_info.value.participant.role == "producer"
    assert exc_info.value.participant.index == 0


@pytest.mark.parametrize(
    ("producer_futures", "consumer_futures", "missing_role"),
    [
        ([], [_Ref("consumer")], "producer"),
        ([_Ref("producer")], [], "consumer"),
        ([], [], "producer"),
    ],
)
def test_each_participant_group_must_be_nonempty(
    monkeypatch: pytest.MonkeyPatch,
    producer_futures: list[_Ref],
    consumer_futures: list[_Ref],
    missing_role: str,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("input validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match=f"{missing_role}.*must not be empty"):
        refit_supervisor.supervise_refit_futures(
            operation="empty-group",
            producer_futures=producer_futures,
            consumer_futures=consumer_futures,
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


@pytest.mark.parametrize(
    ("producer_futures", "consumer_futures", "invalid_role"),
    [
        ([None], [_Ref("consumer")], "producer"),
        ([_Ref("producer")], [None], "consumer"),
    ],
)
def test_none_participant_reference_is_rejected_before_ray_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    producer_futures: list[object],
    consumer_futures: list[object],
    invalid_role: str,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("input validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match=f"{invalid_role}.*reference.*None"):
        refit_supervisor.supervise_refit_futures(
            operation="none-ref",
            producer_futures=producer_futures,
            consumer_futures=consumer_futures,
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


def test_timeout_uses_one_deadline_across_all_participants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-ready")
    consumer = _Ref("consumer-never-ready")
    clock = _FakeClock(now=100.0)
    fake_ray = _FakeRay(
        completion_order=[producer],
        results={producer: True},
        on_wait=lambda timeout: clock.advance(0.4 if timeout > 0 else 0.0),
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    monkeypatch.setattr(refit_supervisor, "monotonic", clock)

    with pytest.raises(refit_supervisor.RefitSupervisionTimeout) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="deadline",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=1.0,
        )

    error = exc_info.value
    assert error.operation == "deadline"
    assert error.timeout_s == 1.0
    assert tuple((item.role, item.index) for item in error.pending) == (
        ("consumer", 0),
    )
    assert fake_ray.wait_calls[0][1:] == (1, 1.0)
    assert fake_ray.wait_calls[1][1:] == (1, 0.0)
    assert fake_ray.wait_calls[2][1] == 1
    assert fake_ray.wait_calls[2][2] == pytest.approx(0.6)
    assert fake_ray.get_calls == [[producer]]
    assert fake_ray.get_timeouts == [pytest.approx(0.6)]


@pytest.mark.parametrize(
    "timeout_s",
    [
        None,
        0.0,
        -1.0,
        math.nan,
        math.inf,
        -math.inf,
        1e308,
        True,
        "1.0",
        10**10_000,
        _FloatSubclass(1.0),
    ],
    ids=[
        "none",
        "zero",
        "negative",
        "nan",
        "infinity",
        "negative-infinity",
        "ray-millisecond-overflow",
        "bool",
        "string",
        "overflowing-int",
        "float-subclass",
    ],
)
def test_timeout_must_be_a_positive_finite_number(
    monkeypatch: pytest.MonkeyPatch,
    timeout_s: object,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("input validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match="timeout_s must be a positive finite"):
        refit_supervisor.supervise_refit_futures(
            operation="invalid-timeout",
            producer_futures=[_Ref("producer")],
            consumer_futures=[_Ref("consumer")],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=timeout_s,
        )


def test_timeout_normalizer_is_a_public_shared_contract() -> None:
    refit_supervisor = _load_refit_supervisor()

    assert "normalize_refit_timeout_s" in refit_supervisor.__all__
    assert refit_supervisor.normalize_refit_timeout_s(37) == 37.0


def test_recoverable_failure_classifier_walks_wrapped_watchdog_abort() -> None:
    from nemo_rl.distributed.refit_watchdog import RefitAborted

    refit_supervisor = _load_refit_supervisor()
    participant = refit_supervisor.RefitParticipant("consumer", 0)
    abort = RefitAborted("peer stopped participating")
    try:
        raise abort
    except RefitAborted as cause:
        wrapped = refit_supervisor.RefitParticipantFailure(
            operation="collective",
            participant=participant,
            detail=str(cause),
        )
        wrapped.__cause__ = cause

    assert refit_supervisor.is_recoverable_refit_failure(wrapped) is True


def test_lost_refit_context_is_never_classified_as_recoverable() -> None:
    from nemo_rl.distributed.refit_watchdog import (
        REFIT_CONTEXT_LOST_TOKEN,
        RefitAborted,
    )

    refit_supervisor = _load_refit_supervisor()
    failure = RefitAborted(f"{REFIT_CONTEXT_LOST_TOKEN} trainer stream is orphaned")

    assert refit_supervisor.is_recoverable_refit_failure(failure) is False


def test_protocol_failure_is_not_classified_as_recoverable() -> None:
    refit_supervisor = _load_refit_supervisor()
    failure = refit_supervisor.RefitParticipantFailure(
        operation="collective",
        participant=refit_supervisor.RefitParticipant("consumer", 0),
        detail="result normalization rejected False",
    )

    assert refit_supervisor.is_recoverable_refit_failure(failure) is False


def test_supervision_timeout_is_classified_as_recoverable() -> None:
    refit_supervisor = _load_refit_supervisor()
    failure = refit_supervisor.RefitSupervisionTimeout(
        operation="collective",
        timeout_s=10.0,
        pending=(refit_supervisor.RefitParticipant("consumer", 0),),
    )

    assert refit_supervisor.is_recoverable_refit_failure(failure) is True


def test_recovery_settle_is_bounded_and_does_not_fetch_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer")
    fake_ray = _FakeRay(completion_order=[producer], results={producer: True})
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    refit_supervisor.settle_refit_futures_for_recovery([producer], 12.5, "train")

    assert fake_ray.wait_calls == [((producer,), 1, 12.5)]
    assert fake_ray.wait_fetch_local_calls == [False]
    assert fake_ray.get_calls == []


def test_timeout_is_a_required_fail_fast_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("signature validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(TypeError, match="timeout_s"):
        refit_supervisor.supervise_refit_futures(
            operation="missing-timeout",
            producer_futures=[_Ref("producer")],
            consumer_futures=[_Ref("consumer")],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
        )


@pytest.mark.parametrize(
    "operation",
    ["", "   ", " leading", "trailing ", _StrSubclass("subclass"), object()],
    ids=["empty", "whitespace", "leading", "trailing", "str-subclass", "object"],
)
def test_operation_must_be_an_exact_nonempty_stripped_string(
    monkeypatch: pytest.MonkeyPatch,
    operation: object,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("input validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(
        ValueError,
        match="operation must be an exact non-empty stripped string",
    ):
        refit_supervisor.supervise_refit_futures(
            operation=operation,
            producer_futures=[_Ref("producer")],
            consumer_futures=[_Ref("consumer")],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


@pytest.mark.parametrize(
    ("producer_futures", "consumer_futures"),
    [
        (
            lambda shared: [shared, shared],
            lambda shared: [_Ref(f"consumer-for-{shared.name}")],
        ),
        (lambda shared: [shared], lambda shared: [shared]),
    ],
    ids=["within-role", "across-roles"],
)
def test_duplicate_reference_is_rejected_before_ray_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    producer_futures: Callable[[_Ref], list[_Ref]],
    consumer_futures: Callable[[_Ref], list[_Ref]],
) -> None:
    refit_supervisor = _load_refit_supervisor()
    shared = _Ref("shared")

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("input validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match="duplicate participant reference"):
        refit_supervisor.supervise_refit_futures(
            operation="duplicate-ref",
            producer_futures=producer_futures(shared),
            consumer_futures=consumer_futures(shared),
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


def test_distinct_equal_references_are_rejected_as_one_logical_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("equal")
    consumer = _Ref("equal")
    assert producer == consumer and producer is not consumer

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("duplicate validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match="duplicate participant reference"):
        refit_supervisor.supervise_refit_futures(
            operation="logical-duplicate",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


def test_equal_ref_returned_by_ray_keeps_participant_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer")
    consumer = _Ref("consumer")
    fake_ray = _FakeRay(
        completion_order=[consumer, producer],
        results={producer: True, consumer: True},
        copy_ready_ref=True,
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    summary = refit_supervisor.supervise_refit_futures(
        operation="equivalent-returned-ref",
        producer_futures=[producer],
        consumer_futures=[consumer],
        result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
        timeout_s=TEST_TIMEOUT_S,
    )

    assert tuple((item.role, item.index) for item in summary.completion_order) == (
        ("consumer", 0),
        ("producer", 0),
    )
    assert fake_ray.get_calls[0] == [consumer, producer]
    assert fake_ray.get_calls[0][0] is not consumer


def test_unhashable_reference_is_rejected_before_ray_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def fail_if_ray_is_loaded() -> object:
        raise AssertionError("reference validation must run before Ray is loaded")

    monkeypatch.setattr(refit_supervisor, "_load_ray", fail_if_ray_is_loaded)

    with pytest.raises(ValueError, match=r"producer\[0\] reference must be hashable"):
        refit_supervisor.supervise_refit_futures(
            operation="unhashable-ref",
            producer_futures=[[]],
            consumer_futures=[_Ref("consumer")],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )


@pytest.mark.parametrize(
    "partition_kind",
    ["too-many-ready", "duplicate", "unknown", "missing"],
)
def test_blocking_wait_must_return_an_exact_pending_partition(
    monkeypatch: pytest.MonkeyPatch,
    partition_kind: str,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer")
    consumer = _Ref("consumer")
    unknown = _Ref("unknown")
    response: tuple[list[object], list[object]]
    if partition_kind == "too-many-ready":
        response = ([producer, consumer], [])
    elif partition_kind == "duplicate":
        response = ([producer], [producer, consumer])
    elif partition_kind == "unknown":
        response = ([unknown], [producer, consumer])
    else:
        response = ([producer], [])
    fake_ray = _ScriptedRay(
        wait_responses=[response],
        results={producer: True, consumer: True, unknown: True},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitSupervisionProtocolError) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="malformed-blocking-partition",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.operation == "malformed-blocking-partition"
    assert fake_ray.get_calls == []


def test_nonblocking_drain_must_return_an_exact_pending_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    first_producer = _Ref("producer-0")
    pending_producer = _Ref("producer-1")
    consumer = _Ref("consumer")
    fake_ray = _ScriptedRay(
        wait_responses=[
            ([first_producer], [pending_producer, consumer]),
            ([consumer], []),
        ],
        results={first_producer: True, pending_producer: True, consumer: True},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitSupervisionProtocolError) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="malformed-drain-partition",
            producer_futures=[first_producer, pending_producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.operation == "malformed-drain-partition"
    assert fake_ray.get_calls == []


def test_nonblocking_ready_batch_processes_consumers_before_producers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    first_producer = _Ref("producer-first")
    failed_producer = _Ref("producer-failed")
    failed_consumer = _Ref("consumer-failed")
    fake_ray = _FakeRay(
        completion_order=[first_producer, failed_producer, failed_consumer],
        results={
            first_producer: True,
            failed_producer: False,
            failed_consumer: False,
        },
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="consumer-first-drain",
            producer_futures=[first_producer, failed_producer],
            consumer_futures=[failed_consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "consumer"
    assert fake_ray.get_calls == [[failed_consumer, first_producer, failed_producer]]
    assert [call[1:] for call in fake_ray.wait_calls] == [
        (1, TEST_TIMEOUT_S),
        (2, 0.0),
    ]


def test_entire_first_ready_wave_processes_consumer_before_blocking_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-selected-by-blocking-wait")
    consumer = _Ref("consumer-ready-in-same-wave")
    fake_ray = _ScriptedRay(
        wait_responses=[
            ([producer], [consumer]),
            ([consumer], []),
        ],
        results={producer: False, consumer: False},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="whole-wave-consumer-priority",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "consumer"
    assert fake_ray.get_calls == [[consumer, producer]]
    assert [call[1:] for call in fake_ray.wait_calls] == [
        (1, TEST_TIMEOUT_S),
        (1, 0.0),
    ]


def test_empty_ready_at_deadline_raises_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-never-ready")
    consumer = _Ref("consumer-never-ready")
    fake_ray = _FakeRay(completion_order=[], results={})
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitSupervisionTimeout) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="deadline",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.operation == "deadline"
    assert exc_info.value.timeout_s == TEST_TIMEOUT_S
    assert fake_ray.get_calls == []


def test_exhausted_deadline_does_not_start_another_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-ready")
    consumer = _Ref("consumer-pending")
    clock = _FakeClock(now=10.0)
    fake_ray = _FakeRay(
        completion_order=[producer, consumer],
        results={producer: True, consumer: True},
        on_wait=lambda timeout: clock.advance(1.1 if timeout > 0 else 0.0),
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    monkeypatch.setattr(refit_supervisor, "monotonic", clock)

    with pytest.raises(refit_supervisor.RefitSupervisionTimeout) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="expired-before-next-wait",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=1.0,
        )

    assert tuple((item.role, item.index) for item in exc_info.value.pending) == (
        ("consumer", 0),
        ("producer", 0),
    )
    assert len(fake_ray.wait_calls) == 1
    assert fake_ray.get_calls == []


def test_deadline_covers_every_drained_result_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-first")
    consumer = _Ref("consumer-drained")
    clock = _FakeClock(now=20.0)
    fake_ray = _FakeRay(
        completion_order=[producer, consumer],
        results={producer: True, consumer: True},
    )

    def slow_normalizer(participant: _Participant, result: object) -> bool:
        assert result is True
        if participant.role == "consumer":
            clock.advance(1.1)
        return True

    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    monkeypatch.setattr(refit_supervisor, "monotonic", clock)

    with pytest.raises(refit_supervisor.RefitSupervisionTimeout) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="normalizer-deadline",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=slow_normalizer,
            timeout_s=1.0,
        )

    assert tuple((item.role, item.index) for item in exc_info.value.pending) == (
        ("consumer", 0),
        ("producer", 0),
    )
    assert fake_ray.get_calls == [[consumer, producer]]
    assert [call[1:] for call in fake_ray.wait_calls] == [(1, 1.0), (1, 0.0)]


def test_deadline_is_checked_after_nonblocking_drain_before_result_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-first")
    consumer = _Ref("consumer-drained")
    clock = _FakeClock(now=30.0)
    fake_ray = _FakeRay(
        completion_order=[producer, consumer],
        results={producer: True, consumer: True},
        on_wait=lambda timeout: clock.advance(1.1 if timeout == 0.0 else 0.0),
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    monkeypatch.setattr(refit_supervisor, "monotonic", clock)

    with pytest.raises(refit_supervisor.RefitSupervisionTimeout) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="nonblocking-drain-deadline",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_exact_true_refit_result,
            timeout_s=1.0,
        )

    assert tuple((item.role, item.index) for item in exc_info.value.pending) == (
        ("consumer", 0),
        ("producer", 0),
    )
    assert fake_ray.get_calls == []


def test_current_result_contract_accepts_existing_actor_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer_none = _Ref("producer-none")
    producer_true = _Ref("producer-true")
    consumer = _Ref("consumer")
    fake_ray = _FakeRay(
        completion_order=[producer_none, producer_true, consumer],
        results={producer_none: None, producer_true: True, consumer: True},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    summary = refit_supervisor.supervise_refit_futures(
        operation="current-contract",
        producer_futures=[producer_none, producer_true],
        consumer_futures=[consumer],
        result_normalizer=refit_supervisor.normalize_current_refit_result,
        timeout_s=TEST_TIMEOUT_S,
    )

    assert tuple(completion.result for completion in summary.completions) == (
        True,
        True,
        True,
    )
    assert tuple((item.role, item.index) for item in summary.completion_order) == (
        ("consumer", 0),
        ("producer", 0),
        ("producer", 1),
    )


def test_current_result_contract_rejects_consumer_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-never-ready")
    consumer = _Ref("consumer-none")
    fake_ray = _FakeRay(
        completion_order=[consumer],
        results={consumer: None},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="current-contract",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=refit_supervisor.normalize_current_refit_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    assert exc_info.value.participant.role == "consumer"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "consumer returned NoneType; expected exactly True" in str(exc_info.value)


def test_result_normalizer_exception_preserves_cause_and_participant_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer-never-ready")
    consumer = _Ref("consumer-ready")
    fake_ray = _FakeRay(
        completion_order=[consumer],
        results={consumer: "typed-ack"},
    )
    root_cause = LookupError("transaction id mismatch")

    def normalize_result(participant: object, result: object) -> object:
        del participant, result
        raise root_cause

    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as exc_info:
        refit_supervisor.supervise_refit_futures(
            operation="typed-transaction",
            producer_futures=[producer],
            consumer_futures=[consumer],
            result_normalizer=normalize_result,
            timeout_s=TEST_TIMEOUT_S,
        )

    error = exc_info.value
    assert error.operation == "typed-transaction"
    assert error.participant.role == "consumer"
    assert error.participant.index == 0
    assert error.__cause__ is root_cause


def test_normalized_typed_results_are_returned_without_side_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = _Ref("producer")
    consumer = _Ref("consumer")
    fake_ray = _FakeRay(
        completion_order=[consumer, producer],
        results={producer: "tx-producer", consumer: "tx-consumer"},
    )

    @dataclass(frozen=True)
    class Ack:
        role: str
        value: str

    def normalize_result(participant: _Participant, result: object) -> Ack:
        if type(result) is not str:
            raise ValueError("expected a string transaction acknowledgement")
        return Ack(role=participant.role, value=result)

    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    summary = refit_supervisor.supervise_refit_futures(
        operation="typed-transaction",
        producer_futures=[producer],
        consumer_futures=[consumer],
        result_normalizer=normalize_result,
        timeout_s=TEST_TIMEOUT_S,
    )

    assert summary.completions[0].result == Ack(role="consumer", value="tx-consumer")
    assert summary.completions[1].result == Ack(role="producer", value="tx-producer")
    with pytest.raises(FrozenInstanceError):
        summary.completions[0].result = Ack(role="consumer", value="mutated")


@pytest.mark.parametrize(
    ("role", "index"),
    [
        ("sender", 0),
        (_StrSubclass("producer"), 0),
        ("producer", -1),
        ("consumer", True),
        ("consumer", 1.0),
        ("consumer", _IntSubclass(1)),
    ],
    ids=[
        "unknown-role",
        "role-subclass",
        "negative-index",
        "bool-index",
        "float-index",
        "index-subclass",
    ],
)
def test_refit_participant_rejects_forged_identity(
    role: object,
    index: object,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    with pytest.raises(ValueError):
        refit_supervisor.RefitParticipant(role=role, index=index)


def test_refit_completion_requires_an_exact_participant_root() -> None:
    refit_supervisor = _load_refit_supervisor()

    with pytest.raises(ValueError, match="participant"):
        refit_supervisor.RefitCompletion(participant=object(), result=True)


def test_refit_completion_revalidates_an_adversarially_mutated_participant() -> None:
    refit_supervisor = _load_refit_supervisor()
    participant = refit_supervisor.RefitParticipant("producer", 0)
    object.__setattr__(participant, "index", True)

    with pytest.raises(ValueError, match="index"):
        refit_supervisor.RefitCompletion(participant=participant, result=True)


def test_refit_completion_rejects_participant_missing_a_forged_field() -> None:
    refit_supervisor = _load_refit_supervisor()
    participant = object.__new__(refit_supervisor.RefitParticipant)
    object.__setattr__(participant, "role", "producer")

    with pytest.raises(ValueError, match="fully initialized"):
        refit_supervisor.RefitCompletion(participant=participant, result=True)


@pytest.mark.parametrize(
    ("operation", "producer_count", "consumer_count"),
    [
        ("", 1, 1),
        (" operation", 1, 1),
        (_StrSubclass("operation"), 1, 1),
        ("operation", 0, 1),
        ("operation", 1, 0),
        ("operation", True, 1),
        ("operation", 1, 1.0),
        ("operation", _IntSubclass(1), 1),
    ],
    ids=[
        "empty-operation",
        "unstripped-operation",
        "operation-subclass",
        "zero-producers",
        "zero-consumers",
        "bool-producer-count",
        "float-consumer-count",
        "producer-count-subclass",
    ],
)
def test_refit_summary_rejects_invalid_metadata(
    operation: object,
    producer_count: object,
    consumer_count: object,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("producer", 0), result=True
    )
    consumer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("consumer", 0), result=True
    )

    with pytest.raises(ValueError):
        refit_supervisor.RefitSupervisionSummary(
            operation=operation,
            producer_count=producer_count,
            consumer_count=consumer_count,
            completions=(producer, consumer),
        )


def test_refit_summary_requires_exact_nonempty_completion_tuple() -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("producer", 0), result=True
    )
    consumer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("consumer", 0), result=True
    )

    with pytest.raises(ValueError, match="exact non-empty tuple"):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=1,
            consumer_count=1,
            completions=[producer, consumer],
        )
    with pytest.raises(ValueError, match="exact non-empty tuple"):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=1,
            consumer_count=1,
            completions=(),
        )
    with pytest.raises(ValueError, match="RefitCompletion"):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=1,
            consumer_count=1,
            completions=(producer, object()),
        )


@pytest.mark.parametrize("invalid_layout", ["duplicate", "partial", "gap"])
def test_refit_summary_rejects_forged_participant_layout(
    invalid_layout: str,
) -> None:
    refit_supervisor = _load_refit_supervisor()

    def completion(role: str, index: int) -> object:
        return refit_supervisor.RefitCompletion(
            participant=refit_supervisor.RefitParticipant(role, index), result=True
        )

    if invalid_layout == "duplicate":
        completions = (completion("producer", 0), completion("producer", 0))
        producer_count, consumer_count = 1, 1
    elif invalid_layout == "partial":
        completions = (completion("producer", 0), completion("consumer", 0))
        producer_count, consumer_count = 2, 1
    else:
        completions = (
            completion("producer", 0),
            completion("producer", 2),
            completion("consumer", 0),
        )
        producer_count, consumer_count = 2, 1

    with pytest.raises(ValueError):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=producer_count,
            consumer_count=consumer_count,
            completions=completions,
        )


@pytest.mark.parametrize("mutation", ["completion", "participant"])
def test_refit_summary_revalidates_adversarially_mutated_nested_values(
    mutation: str,
) -> None:
    refit_supervisor = _load_refit_supervisor()
    producer_participant = refit_supervisor.RefitParticipant("producer", 0)
    producer = refit_supervisor.RefitCompletion(
        participant=producer_participant, result=True
    )
    consumer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("consumer", 0), result=True
    )
    if mutation == "completion":
        object.__setattr__(producer, "participant", object())
    else:
        object.__setattr__(producer_participant, "role", "sender")

    with pytest.raises(ValueError):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=1,
            consumer_count=1,
            completions=(producer, consumer),
        )


def test_refit_summary_rejects_completion_missing_a_forged_field() -> None:
    refit_supervisor = _load_refit_supervisor()
    producer = object.__new__(refit_supervisor.RefitCompletion)
    object.__setattr__(
        producer,
        "participant",
        refit_supervisor.RefitParticipant("producer", 0),
    )
    consumer = refit_supervisor.RefitCompletion(
        participant=refit_supervisor.RefitParticipant("consumer", 0), result=True
    )

    with pytest.raises(ValueError, match="fully initialized"):
        refit_supervisor.RefitSupervisionSummary(
            operation="operation",
            producer_count=1,
            consumer_count=1,
            completions=(producer, consumer),
        )


def test_refit_summary_post_init_rejects_missing_forged_metadata() -> None:
    refit_supervisor = _load_refit_supervisor()
    summary = object.__new__(refit_supervisor.RefitSupervisionSummary)
    object.__setattr__(summary, "operation", "operation")
    object.__setattr__(summary, "producer_count", 1)
    object.__setattr__(summary, "completions", ())

    with pytest.raises(ValueError, match="fully initialized"):
        summary.__post_init__()
