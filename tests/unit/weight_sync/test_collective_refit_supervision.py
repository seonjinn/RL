# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Runtime-free supervision contracts for the two non-colocated NCCL paths."""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).parents[3]


def _load_refit_supervisor() -> ModuleType:
    module_path = _REPO_ROOT / "nemo_rl/weight_sync/refit_supervisor.py"
    spec = importlib.util.spec_from_file_location(
        "_collective_refit_supervisor_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


refit_supervisor = _load_refit_supervisor()


class _RecoverableRefitError(RuntimeError):
    pass


class _LostRefitContextError(_RecoverableRefitError):
    pass


class _RayActorError(RuntimeError):
    pass


def _should_settle_failure(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    recoverable = False
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, _LostRefitContextError):
            return False
        recoverable = recoverable or isinstance(
            current,
            (
                _RecoverableRefitError,
                _RayActorError,
                refit_supervisor.RefitSupervisionTimeout,
            ),
        )
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return recoverable


def _load_synchronizer_module(*, filename: str, module_name: str) -> ModuleType:
    """Execute synchronizer definitions without importing GPU/runtime packages."""
    source_path = _REPO_ROOT / f"nemo_rl/weight_sync/{filename}"
    parsed = ast.parse(source_path.read_text())
    definitions_only = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *[
                node
                for node in parsed.body
                if not isinstance(node, (ast.Import, ast.ImportFrom))
            ],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(definitions_only)
    ray_stub = SimpleNamespace(
        get=lambda _refs: [True],
        wait=lambda refs, **_kwargs: (list(refs), []),
    )
    module = ModuleType(module_name)
    module.__dict__.update(
        {
            "Any": Any,
            "Optional": type(None) | float,
            "Sequence": Sequence,
            "nullcontext": nullcontext,
            "ray": ray_stub,
            "Timer": object,
            "WeightSynchronizer": object,
            "RefitMembership": object,
            "plan_refit_membership": lambda **_kwargs: None,
            "make_nccl_reshard_refit_info_wire_safe": lambda value: value,
            "normalize_current_refit_result": (
                refit_supervisor.normalize_current_refit_result
            ),
            "normalize_refit_timeout_s": refit_supervisor.normalize_refit_timeout_s,
            "RefitSupervisionTimeout": refit_supervisor.RefitSupervisionTimeout,
            "RayActorError": _RayActorError,
            "is_refit_abort": lambda error: isinstance(error, _RecoverableRefitError),
            "is_refit_context_lost": lambda error: isinstance(
                error, _LostRefitContextError
            ),
            "monotonic": monotonic,
            "_settle_before_propagating": (
                refit_supervisor.settle_refit_futures_for_recovery
            ),
            "is_recoverable_refit_failure": _should_settle_failure,
            "supervise_refit_futures": refit_supervisor.supervise_refit_futures,
        }
    )
    exec(compile(definitions_only, source_path, "exec"), module.__dict__)
    return module


collective_module = _load_synchronizer_module(
    filename="collective_weight_synchronizer.py",
    module_name="_collective_weight_synchronizer_under_test",
)
reshard_module = _load_synchronizer_module(
    filename="nccl_reshard_weight_synchronizer.py",
    module_name="_nccl_reshard_weight_synchronizer_under_test",
)
CollectiveWeightSynchronizer = collective_module.CollectiveWeightSynchronizer
NcclReshardWeightSynchronizer = reshard_module.NcclReshardWeightSynchronizer


class _FakeRefitRay:
    """Small deterministic stand-in for the Ray calls made by the supervisor."""

    def __init__(
        self,
        *,
        ready_order: list[object],
        results: dict[object, object],
    ) -> None:
        self._ready_order = ready_order.copy()
        self._results = results
        self.get_calls: list[object] = []

    def wait(
        self,
        refs: list[object],
        *,
        num_returns: int,
        timeout: float,
        fetch_local: bool,
    ) -> tuple[list[object], list[object]]:
        del timeout, fetch_local
        ref_ids = {id(ref) for ref in refs}
        ready = [ref for ref in self._ready_order if id(ref) in ref_ids][:num_returns]
        ready_ids = {id(ref) for ref in ready}
        self._ready_order = [
            ref for ref in self._ready_order if id(ref) not in ready_ids
        ]
        return ready, [ref for ref in refs if id(ref) not in ready_ids]

    def get(self, ref: object, *, timeout: float) -> object:
        del timeout
        self.get_calls.append(ref)
        refs = ref if type(ref) is list else [ref]
        results: list[object] = []
        for item in refs:
            result = self._results[item]
            if isinstance(result, BaseException):
                raise result
            results.append(result)
        return results if type(ref) is list else results[0]


class _Policy:
    def __init__(self, producer_refs: list[object], events: list[str]) -> None:
        self._producer_refs = producer_refs
        self._events = events
        self.broadcast_kwargs: dict[str, object] | None = None
        self.reshard_kwargs: dict[str, object] | None = None

    def broadcast_weights_for_collective(self, **kwargs: object) -> list[object]:
        self._events.append("producer")
        self.broadcast_kwargs = kwargs
        return self._producer_refs

    def nccl_reshard_refit(self, **kwargs: object) -> list[object]:
        self._events.append("producer")
        self.reshard_kwargs = kwargs
        return self._producer_refs


class _Generation:
    def __init__(
        self,
        consumer_refs: list[object],
        events: list[str],
        *,
        supports_worker_timeout: object,
    ) -> None:
        self._consumer_refs = consumer_refs
        self._events = events
        self._supports_worker_timeout = supports_worker_timeout
        self.collective_kwargs: dict[str, object] | None = None
        self.reshard_kwargs: dict[str, object] | None = None

    @property
    def supports_refit_worker_timeout(self) -> object:
        self._events.append("capability")
        return self._supports_worker_timeout

    def get_collective_sender_spec(self) -> SimpleNamespace:
        return SimpleNamespace(buffer_size_bytes=1024, num_buffers=2)

    def update_weights_from_collective(self, **kwargs: object) -> list[object]:
        self._events.append("consumer")
        self.collective_kwargs = kwargs
        return self._consumer_refs

    def nccl_reshard_refit(self, **kwargs: object) -> list[object]:
        self._events.append("consumer")
        self.reshard_kwargs = kwargs
        return self._consumer_refs


@dataclass(frozen=True)
class _TransportCase:
    name: str
    synchronizer_type: type[Any]
    module: ModuleType
    operation: str


_TRANSPORTS = (
    _TransportCase(
        name="collective",
        synchronizer_type=CollectiveWeightSynchronizer,
        module=collective_module,
        operation="collective-weight-sync",
    ),
    _TransportCase(
        name="nccl_reshard",
        synchronizer_type=NcclReshardWeightSynchronizer,
        module=reshard_module,
        operation="nccl-reshard-weight-sync",
    ),
)


def _make_synchronizer(
    case: _TransportCase,
    *,
    timeout_s: object = None,
    supports_worker_timeout: object = True,
    recover_refit_failures: object = False,
    producer_refs: list[object] | None = None,
    consumer_refs: list[object] | None = None,
) -> tuple[Any, _Policy, _Generation, list[str]]:
    events: list[str] = []
    policy = _Policy([object()] if producer_refs is None else producer_refs, events)
    generation = _Generation(
        [object()] if consumer_refs is None else consumer_refs,
        events,
        supports_worker_timeout=supports_worker_timeout,
    )
    synchronizer = case.synchronizer_type(
        policy=policy,
        generation=generation,
        train_cluster=None,
        inference_cluster=None,
        refit_timeout_s=timeout_s,
        recover_refit_failures=recover_refit_failures,
    )
    return synchronizer, policy, generation, events


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("supports_worker_timeout", "expected_consumer_timeout_s"),
    [(False, None), (True, 300.0)],
)
def test_default_deadline_routes_only_to_capable_consumers(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
    supports_worker_timeout: bool,
    expected_consumer_timeout_s: float | None,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, policy, generation, events = _make_synchronizer(
        case,
        supports_worker_timeout=supports_worker_timeout,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    fake_ray = _FakeRefitRay(
        ready_order=[consumer_ref, producer_ref],
        results={producer_ref: None, consumer_ref: True},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)

    synchronizer.sync_weights()

    producer_kwargs = policy.broadcast_kwargs or policy.reshard_kwargs
    consumer_kwargs = generation.collective_kwargs or generation.reshard_kwargs
    assert producer_kwargs is not None
    assert consumer_kwargs is not None
    assert producer_kwargs["refit_timeout_s"] == 300.0
    assert consumer_kwargs["refit_timeout_s"] == expected_consumer_timeout_s
    assert events[:2] == ["capability", "producer"]
    assert fake_ray.get_calls == [[consumer_ref, producer_ref]]
    assert synchronizer.is_stale is False


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
def test_fatal_consumer_failure_is_observed_without_waiting_for_pending_peers(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        timeout_s=17.5,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    root_cause = LookupError("consumer failed before producer unwound")
    fake_ray = _FakeRefitRay(
        ready_order=[consumer_ref],
        results={producer_ref: None, consumer_ref: root_cause},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    settle_calls: list[tuple[list[object], float, str]] = []
    monkeypatch.setattr(
        case.module,
        "_settle_before_propagating",
        lambda futures, budget_s, what: settle_calls.append(
            (list(futures), budget_s, what)
        ),
    )

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as failure:
        synchronizer.sync_weights()

    assert failure.value.participant == refit_supervisor.RefitParticipant("consumer", 0)
    assert failure.value.operation == case.operation
    assert failure.value.__cause__ is root_cause
    assert fake_ray.get_calls == [[consumer_ref], consumer_ref]
    assert settle_calls == []
    assert synchronizer.is_stale is True


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
def test_recoverable_failure_settles_both_sides_only_when_recovery_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        timeout_s=17.5,
        recover_refit_failures=True,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    root_cause = _RecoverableRefitError("collective watchdog aborted")
    fake_ray = _FakeRefitRay(
        ready_order=[consumer_ref],
        results={producer_ref: None, consumer_ref: root_cause},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    settle_calls: list[tuple[list[object], float, str]] = []
    monkeypatch.setattr(
        case.module,
        "_settle_before_propagating",
        lambda futures, budget_s, what: settle_calls.append(
            (list(futures), budget_s, what)
        ),
    )

    with pytest.raises(refit_supervisor.RefitParticipantFailure) as failure:
        synchronizer.sync_weights()

    assert failure.value.__cause__ is root_cause
    assert [call[0::2] for call in settle_calls] == [
        ([producer_ref], "train"),
        ([consumer_ref], "generation"),
    ]
    assert all(30.0 <= call[1] <= 47.5 for call in settle_calls)


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
def test_unrecoverable_context_loss_never_waits_before_propagation(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        recover_refit_failures=True,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    fake_ray = _FakeRefitRay(
        ready_order=[consumer_ref],
        results={
            producer_ref: None,
            consumer_ref: _LostRefitContextError("trainer context lost"),
        },
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    settle_calls: list[object] = []
    monkeypatch.setattr(
        case.module,
        "_settle_before_propagating",
        lambda *_args: settle_calls.append(object()),
    )

    with pytest.raises(refit_supervisor.RefitParticipantFailure):
        synchronizer.sync_weights()

    assert settle_calls == []


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
def test_consumer_submission_failure_still_settles_the_started_producer(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
) -> None:
    producer_ref = object()
    synchronizer, _, generation, _ = _make_synchronizer(
        case,
        recover_refit_failures=True,
        producer_refs=[producer_ref],
    )
    root_cause = _RayActorError("consumer actor disappeared during submission")

    def fail_consumer_submission(**_kwargs: object) -> list[object]:
        raise root_cause

    if case.name == "collective":
        generation.update_weights_from_collective = fail_consumer_submission
    else:
        generation.nccl_reshard_refit = fail_consumer_submission
    settle_calls: list[tuple[list[object], str]] = []
    monkeypatch.setattr(
        case.module,
        "_settle_before_propagating",
        lambda futures, _budget_s, what: settle_calls.append((list(futures), what)),
    )

    with pytest.raises(RuntimeError) as failure:
        synchronizer.sync_weights()

    assert failure.value is root_cause
    assert settle_calls == [([producer_ref], "train"), ([], "generation")]
    assert synchronizer.is_stale is True


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize("invalid_recovery", [None, 0, 1, "yes", object()])
def test_recovery_capability_requires_an_exact_bool(
    case: _TransportCase,
    invalid_recovery: object,
) -> None:
    with pytest.raises(
        ValueError, match="recover_refit_failures must be an exact bool"
    ):
        _make_synchronizer(case, recover_refit_failures=invalid_recovery)


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "invalid_timeout_s",
    [0.0, -1.0, float("inf"), float("nan"), True, "17.5"],
)
def test_invalid_controller_deadline_fails_at_construction(
    case: _TransportCase,
    invalid_timeout_s: object,
) -> None:
    events: list[str] = []
    policy = _Policy([object()], events)
    generation = _Generation([object()], events, supports_worker_timeout=True)

    with pytest.raises(ValueError, match="timeout_s must be"):
        case.synchronizer_type(
            policy=policy,
            generation=generation,
            train_cluster=None,
            inference_cluster=None,
            refit_timeout_s=invalid_timeout_s,
        )

    assert events == []


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize("invalid_capability", [None, 0, 1, "yes", object()])
def test_invalid_worker_timeout_capability_fails_before_producer_submission(
    case: _TransportCase,
    invalid_capability: object,
) -> None:
    synchronizer, policy, generation, events = _make_synchronizer(
        case,
        timeout_s=17.5,
        supports_worker_timeout=invalid_capability,
    )

    with pytest.raises(RuntimeError, match="supports_refit_worker_timeout.*bool"):
        synchronizer.sync_weights()

    assert events == ["capability"]
    assert policy.broadcast_kwargs is None
    assert policy.reshard_kwargs is None
    assert generation.collective_kwargs is None
    assert generation.reshard_kwargs is None
    assert synchronizer.is_stale is True


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("failed_role", "invalid_result"),
    [
        ("producer", False),
        ("producer", 0),
        ("producer", "done"),
        ("consumer", False),
        ("consumer", None),
        ("consumer", 1),
        ("consumer", "done"),
    ],
)
def test_terminal_results_use_the_exact_current_protocol(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
    failed_role: str,
    invalid_result: object,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    results: dict[object, object] = {producer_ref: None, consumer_ref: True}
    failed_ref = producer_ref if failed_role == "producer" else consumer_ref
    results[failed_ref] = invalid_result
    fake_ray = _FakeRefitRay(ready_order=[failed_ref], results=results)
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: fake_ray)
    monkeypatch.setattr(case.module, "_settle_before_propagating", lambda *_args: None)

    with pytest.raises(
        refit_supervisor.RefitParticipantFailure,
        match=rf"{failed_role}\[0\].*expected",
    ):
        synchronizer.sync_weights()

    assert synchronizer.is_stale is True


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("producer_refs", "consumer_refs", "expected_message"),
    [
        ([], [object()], "producer participant group"),
        ([object()], [], "consumer participant group"),
    ],
)
def test_empty_participant_groups_cannot_report_success(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
    producer_refs: list[object],
    consumer_refs: list[object],
    expected_message: str,
) -> None:
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        producer_refs=producer_refs,
        consumer_refs=consumer_refs,
    )
    monkeypatch.setattr(case.module, "_settle_before_propagating", lambda *_args: None)

    with pytest.raises(ValueError, match=expected_message):
        synchronizer.sync_weights()

    assert synchronizer.is_stale is True


@pytest.mark.parametrize("case", _TRANSPORTS, ids=lambda case: case.name)
def test_new_attempt_marks_a_previously_fresh_synchronizer_stale_until_full_success(
    monkeypatch: pytest.MonkeyPatch,
    case: _TransportCase,
) -> None:
    producer_ref = object()
    consumer_ref = object()
    synchronizer, _, _, _ = _make_synchronizer(
        case,
        producer_refs=[producer_ref],
        consumer_refs=[consumer_ref],
    )
    active_ray = _FakeRefitRay(
        ready_order=[consumer_ref, producer_ref],
        results={producer_ref: True, consumer_ref: True},
    )
    monkeypatch.setattr(refit_supervisor, "_load_ray", lambda: active_ray)
    monkeypatch.setattr(case.module, "_settle_before_propagating", lambda *_args: None)

    synchronizer.sync_weights()
    assert synchronizer.is_stale is False

    active_ray = _FakeRefitRay(
        ready_order=[consumer_ref],
        results={producer_ref: True, consumer_ref: False},
    )
    with pytest.raises(refit_supervisor.RefitParticipantFailure):
        synchronizer.sync_weights()

    assert synchronizer.is_stale is True
