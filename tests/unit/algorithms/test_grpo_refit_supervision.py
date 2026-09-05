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

"""Framework-free contract tests for the inline GRPO refit coordinator."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


def _load_refit_supervisor() -> ModuleType:
    module_path = Path(__file__).parents[3] / "nemo_rl/weight_sync/refit_supervisor.py"
    spec = importlib.util.spec_from_file_location(
        "_grpo_refit_supervisor_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_REFIT_SUPERVISOR = _load_refit_supervisor()
RefitParticipantFailure = _REFIT_SUPERVISOR.RefitParticipantFailure
normalize_current_refit_result = _REFIT_SUPERVISOR.normalize_current_refit_result
normalize_refit_timeout_s = _REFIT_SUPERVISOR.normalize_refit_timeout_s
supervise_refit_futures = _REFIT_SUPERVISOR.supervise_refit_futures


class _SGLangGeneration:
    pass


def _load_refit_policy_generation(
    *, supervisor: object = supervise_refit_futures
) -> FunctionType:
    """Load only the target function without importing GRPO's GPU dependencies."""
    repo_root = Path(__file__).parents[3]
    source_path = repo_root / "nemo_rl/algorithms/grpo.py"
    parsed = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "refit_policy_generation"
    )
    isolated_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "SGLangGeneration": _SGLangGeneration,
        "normalize_current_refit_result": normalize_current_refit_result,
        "normalize_refit_timeout_s": normalize_refit_timeout_s,
        "nullcontext": nullcontext,
        "os": os,
        "supervise_refit_futures": supervisor,
    }
    exec(compile(isolated_module, source_path, "exec"), namespace)
    loaded = namespace["refit_policy_generation"]
    assert isinstance(loaded, FunctionType)
    return loaded


def _load_async_cleanup_factory(
    *, ray_runtime: object, shutdown_environments: object
) -> FunctionType:
    """Load the cleanup factory without importing GRPO's GPU dependencies."""
    repo_root = Path(__file__).parents[3]
    source_path = repo_root / "nemo_rl/algorithms/grpo.py"
    parsed = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_async_grpo_cleanup"
    )
    isolated_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "ray": ray_runtime,
        "shutdown_environments": shutdown_environments,
        "threading": threading,
        "time": time,
    }
    exec(compile(isolated_module, source_path, "exec"), namespace)
    loaded = namespace["_make_async_grpo_cleanup"]
    assert isinstance(loaded, FunctionType)
    return loaded


def _load_async_startup_guard() -> FunctionType:
    repo_root = Path(__file__).parents[3]
    source_path = repo_root / "nemo_rl/algorithms/grpo.py"
    parsed = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_async_grpo_startup_with_cleanup"
    )
    isolated_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated_module)
    namespace: dict[str, object] = {}
    exec(compile(isolated_module, source_path, "exec"), namespace)
    loaded = namespace["_run_async_grpo_startup_with_cleanup"]
    assert isinstance(loaded, FunctionType)
    return loaded


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def _inline_refit_participants() -> tuple[MagicMock, MagicMock, object, object]:
    policy = MagicMock()
    policy_generation = MagicMock()
    policy_generation.weight_synchronizer = None
    policy_generation.supports_refit_worker_timeout = False
    policy_generation.prepare_for_generation.return_value = True
    producer_future = object()
    consumer_future = object()
    return policy, policy_generation, producer_future, consumer_future


def test_colocated_refit_submits_both_participant_groups_to_one_supervisor() -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    policy.stream_weights_via_ipc_zmq.return_value = [producer_future]
    generation.update_weights_via_ipc_zmq.return_value = [consumer_future]
    kv_scales = {"layer.0": 0.5}

    refit_policy_generation(
        policy,
        generation,
        colocated_inference=True,
        _refit_buffer_size_gb=1.0,
        kv_scales=kv_scales,
    )

    policy.stream_weights_via_ipc_zmq.assert_called_once_with(
        buffer_size_bytes=1024**3,
        kv_scales=kv_scales,
    )
    generation.update_weights_via_ipc_zmq.assert_called_once_with()
    supervisor.assert_called_once_with(
        operation="colocated_ipc_policy_generation_refit",
        producer_futures=[producer_future],
        consumer_futures=[consumer_future],
        result_normalizer=normalize_current_refit_result,
        timeout_s=300.0,
    )


@pytest.mark.parametrize(
    ("supports_worker_timeout", "expected_worker_timeout_s"),
    [(False, None), (True, 17.5)],
)
def test_collective_refit_routes_only_supported_worker_deadline_and_supervises_together(
    supports_worker_timeout: bool,
    expected_worker_timeout_s: float | None,
) -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    generation.supports_refit_worker_timeout = supports_worker_timeout
    policy.broadcast_weights_for_collective.return_value = [producer_future]
    generation.update_weights_from_collective.return_value = [consumer_future]

    refit_policy_generation(
        policy,
        generation,
        colocated_inference=False,
        refit_timeout_s=17.5,
    )

    policy.broadcast_weights_for_collective.assert_called_once_with(
        kv_scales=None,
        refit_timeout_s=17.5,
    )
    generation.update_weights_from_collective.assert_called_once_with(
        refit_timeout_s=expected_worker_timeout_s
    )
    supervisor.assert_called_once_with(
        operation="non_colocated_collective_policy_generation_refit",
        producer_futures=[producer_future],
        consumer_futures=[consumer_future],
        result_normalizer=normalize_current_refit_result,
        timeout_s=17.5,
    )


@pytest.mark.parametrize(
    "invalid_timeout_s",
    [0.0, -1.0, float("inf"), float("nan"), True, "17.5"],
)
def test_invalid_refit_timeout_fails_before_either_rpc_is_submitted(
    invalid_timeout_s: object,
) -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, _, _ = _inline_refit_participants()

    with pytest.raises(ValueError, match="timeout_s must be"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=False,
            refit_timeout_s=invalid_timeout_s,
        )

    policy.broadcast_weights_for_collective.assert_not_called()
    generation.update_weights_from_collective.assert_not_called()
    supervisor.assert_not_called()


@pytest.mark.parametrize(
    "invalid_capability",
    [None, 0, 1, "False", object()],
)
def test_invalid_worker_timeout_capability_fails_before_collective_rpc_submission(
    invalid_capability: object,
) -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, _, _ = _inline_refit_participants()
    generation.supports_refit_worker_timeout = invalid_capability

    with pytest.raises(RuntimeError, match="supports_refit_worker_timeout.*bool"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=False,
            refit_timeout_s=17.5,
        )

    policy.broadcast_weights_for_collective.assert_not_called()
    generation.update_weights_from_collective.assert_not_called()
    supervisor.assert_not_called()


@pytest.mark.parametrize("invalid_ack", [False, None, 1, "True"])
def test_colocated_refit_requires_exact_weights_prepare_ack_before_transfer(
    invalid_ack: object,
) -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, _, _ = _inline_refit_participants()
    generation.prepare_for_generation.return_value = invalid_ack

    with pytest.raises(RuntimeError, match="weights.*exactly True"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=True,
            _refit_buffer_size_gb=1.0,
        )

    generation.prepare_for_generation.assert_called_once_with(
        tags=["weights"], refit_timeout_s=300.0
    )
    policy.stream_weights_via_ipc_zmq.assert_not_called()
    generation.update_weights_via_ipc_zmq.assert_not_called()
    policy.offload_after_refit.assert_not_called()
    supervisor.assert_not_called()


@pytest.mark.parametrize("invalid_ack", [False, None, 1, "True"])
def test_colocated_refit_requires_exact_kv_cache_prepare_ack_after_transfer(
    invalid_ack: object,
) -> None:
    supervisor = MagicMock()
    refit_policy_generation = _load_refit_policy_generation(supervisor=supervisor)
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    generation.prepare_for_generation.side_effect = [True, invalid_ack]
    policy.stream_weights_via_ipc_zmq.return_value = [producer_future]
    generation.update_weights_via_ipc_zmq.return_value = [consumer_future]

    with pytest.raises(RuntimeError, match="kv_cache.*exactly True"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=True,
            _refit_buffer_size_gb=1.0,
        )

    assert generation.prepare_for_generation.call_args_list == [
        call(tags=["weights"], refit_timeout_s=300.0),
        call(tags=["kv_cache"], refit_timeout_s=300.0),
    ]
    policy.offload_after_refit.assert_called_once_with()
    supervisor.assert_called_once()


def test_refit_supervision_remains_inside_existing_timer_scope() -> None:
    timer_scope = MagicMock()
    timer_scope.__enter__.return_value = None
    timer = MagicMock()
    timer.time.return_value = timer_scope

    def supervise_inside_timer(**_: object) -> None:
        timer_scope.__enter__.assert_called_once_with()
        timer_scope.__exit__.assert_not_called()

    refit_policy_generation = _load_refit_policy_generation(
        supervisor=supervise_inside_timer
    )
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    policy.broadcast_weights_for_collective.return_value = [producer_future]
    generation.update_weights_from_collective.return_value = [consumer_future]

    assert (
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=False,
            timer=timer,
        )
        == {}
    )

    timer.time.assert_called_once_with(
        "prepare_for_generation/transfer_and_update_weights"
    )
    timer_scope.__exit__.assert_called_once_with(None, None, None)


class _SequentialFakeRay:
    def __init__(
        self,
        *,
        results: dict[object, object],
        failures: dict[object, Exception] | None = None,
    ) -> None:
        self.results = results
        self.failures = failures or {}
        self.get_calls: list[object] = []

    def wait(
        self,
        object_refs: list[object],
        *,
        num_returns: int,
        timeout: float,
        fetch_local: bool,
    ) -> tuple[list[object], list[object]]:
        del num_returns, fetch_local
        if timeout == 0.0:
            return [], object_refs
        return object_refs[:1], object_refs[1:]

    def get(self, object_ref: object, /, *, timeout: float) -> object:
        del timeout
        self.get_calls.append(object_ref)
        object_refs = object_ref if type(object_ref) is list else [object_ref]
        results: list[object] = []
        for item in object_refs:
            failure = self.failures.get(item)
            if failure is not None:
                raise failure
            results.append(self.results[item])
        return results if type(object_ref) is list else results[0]


@pytest.mark.parametrize(
    ("producer_result", "consumer_result", "failed_role"),
    [
        (None, False, "consumer"),
        (None, None, "consumer"),
        (None, "True", "consumer"),
        (None, 1, "consumer"),
        (False, True, "producer"),
        ("True", True, "producer"),
        (1, True, "producer"),
    ],
)
def test_refit_rejects_malformed_ack_before_post_refit_transitions(
    monkeypatch: pytest.MonkeyPatch,
    producer_result: object,
    consumer_result: object,
    failed_role: str,
) -> None:
    refit_policy_generation = _load_refit_policy_generation()
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    policy.stream_weights_via_ipc_zmq.return_value = [producer_future]
    generation.update_weights_via_ipc_zmq.return_value = [consumer_future]
    fake_ray = _SequentialFakeRay(
        results={
            producer_future: producer_result,
            consumer_future: consumer_result,
        }
    )
    monkeypatch.setattr(_REFIT_SUPERVISOR, "_load_ray", lambda: fake_ray)

    with pytest.raises(RefitParticipantFailure, match=rf"{failed_role}\[0\]"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=True,
            _refit_buffer_size_gb=1.0,
        )

    policy.offload_after_refit.assert_not_called()
    generation.prepare_for_generation.assert_called_once_with(
        tags=["weights"], refit_timeout_s=300.0
    )


def test_ready_consumer_failure_surfaces_while_producer_remains_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refit_policy_generation = _load_refit_policy_generation()
    policy, generation, producer_future, consumer_future = _inline_refit_participants()
    policy.stream_weights_via_ipc_zmq.return_value = [producer_future]
    generation.update_weights_via_ipc_zmq.return_value = [consumer_future]
    fake_ray = _SequentialFakeRay(
        results={producer_future: None},
        failures={consumer_future: RuntimeError("consumer failed")},
    )
    monkeypatch.setattr(_REFIT_SUPERVISOR, "_load_ray", lambda: fake_ray)

    with pytest.raises(RefitParticipantFailure, match="consumer failed"):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=True,
            _refit_buffer_size_gb=1.0,
        )

    assert fake_ray.get_calls == [[consumer_future], consumer_future]
    policy.offload_after_refit.assert_not_called()
    generation.prepare_for_generation.assert_called_once_with(
        tags=["weights"], refit_timeout_s=300.0
    )


@pytest.mark.parametrize(
    ("producer_futures", "consumer_futures", "expected_error"),
    [
        ([], [object()], "producer participant group must not be empty"),
        ([object()], [], "consumer participant group must not be empty"),
    ],
)
def test_empty_participant_group_fails_before_post_refit_transitions(
    producer_futures: list[object],
    consumer_futures: list[object],
    expected_error: str,
) -> None:
    refit_policy_generation = _load_refit_policy_generation()
    policy, generation, _, _ = _inline_refit_participants()
    policy.stream_weights_via_ipc_zmq.return_value = producer_futures
    generation.update_weights_via_ipc_zmq.return_value = consumer_futures

    with pytest.raises(ValueError, match=expected_error):
        refit_policy_generation(
            policy,
            generation,
            colocated_inference=True,
            _refit_buffer_size_gb=1.0,
        )

    policy.offload_after_refit.assert_not_called()
    generation.prepare_for_generation.assert_called_once_with(
        tags=["weights"], refit_timeout_s=300.0
    )


def test_async_terminal_cleanup_has_one_bounded_budget_and_stops_participants_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_started = threading.Event()
    checkpoint_block = threading.Event()
    cleanup_events: list[str] = []
    cleanup_events_lock = threading.Lock()

    def record_cleanup_event(name: str) -> None:
        with cleanup_events_lock:
            cleanup_events.append(name)

    def block_checkpoint_shutdown() -> None:
        record_cleanup_event("checkpoint")
        checkpoint_started.set()
        checkpoint_block.wait()

    collector = object()
    replay_buffer = object()
    shared_environment = object()
    other_environment = object()
    participant_names = {
        id(collector): "collector",
        id(replay_buffer): "replay_buffer",
        id(shared_environment): "shared_environment",
        id(other_environment): "other_environment",
    }
    ray_runtime = SimpleNamespace(
        kill=MagicMock(
            side_effect=lambda participant: record_cleanup_event(
                participant_names[id(participant)]
            )
        )
    )
    shutdown_environments = MagicMock()
    make_cleanup = _load_async_cleanup_factory(
        ray_runtime=ray_runtime,
        shutdown_environments=shutdown_environments,
    )
    checkpointer = SimpleNamespace(
        shutdown=MagicMock(side_effect=block_checkpoint_shutdown)
    )
    generation = SimpleNamespace(
        shutdown=MagicMock(side_effect=lambda: record_cleanup_event("generation"))
    )
    policy = SimpleNamespace(
        shutdown=MagicMock(side_effect=lambda: record_cleanup_event("policy"))
    )
    flush_telemetry = MagicMock()
    task_to_env = {"train": shared_environment}
    val_task_to_env = {
        "shared": shared_environment,
        "validation": other_environment,
    }

    terminal_cleanup, graceful_cleanup = make_cleanup(
        checkpointer=checkpointer,
        trajectory_collector=collector,
        replay_buffer=replay_buffer,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
        policy_generation=generation,
        policy=policy,
        flush_collector_telemetry=flush_telemetry,
        terminal_cleanup_budget_s=0.05,
    )

    started_at = time.monotonic()
    terminal_cleanup()
    elapsed_s = time.monotonic() - started_at
    terminal_cleanup()
    graceful_cleanup()
    capsys.readouterr()

    assert elapsed_s < 0.2
    assert checkpoint_started.wait(timeout=0.1)
    assert not checkpoint_block.is_set()
    assert cleanup_events[-1] == "checkpoint"
    assert set(cleanup_events[:-1]) == {
        "collector",
        "replay_buffer",
        "shared_environment",
        "other_environment",
        "generation",
        "policy",
    }
    checkpoint_block.set()
    checkpointer.shutdown.assert_called_once_with()
    flush_telemetry.assert_not_called()
    ray_runtime.kill.assert_has_calls(
        [
            call(collector),
            call(replay_buffer),
            call(shared_environment),
            call(other_environment),
        ],
        any_order=True,
    )
    assert ray_runtime.kill.call_count == 4
    shutdown_environments.assert_not_called()
    generation.shutdown.assert_called_once_with()
    policy.shutdown.assert_called_once_with()


def test_async_graceful_cleanup_retains_checkpoint_durability_and_is_idempotent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cleanup_error = KeyboardInterrupt("cleanup interrupted")
    ray_runtime = SimpleNamespace(kill=MagicMock(side_effect=cleanup_error))
    shutdown_environments = MagicMock(side_effect=cleanup_error)
    make_cleanup = _load_async_cleanup_factory(
        ray_runtime=ray_runtime,
        shutdown_environments=shutdown_environments,
    )
    checkpointer = SimpleNamespace(shutdown=MagicMock(side_effect=cleanup_error))
    collector = object()
    replay_buffer = object()
    generation = SimpleNamespace(shutdown=MagicMock(side_effect=cleanup_error))
    policy = SimpleNamespace(shutdown=MagicMock(side_effect=cleanup_error))
    flush_telemetry = MagicMock(side_effect=cleanup_error)
    task_to_env = {"train": object()}
    val_task_to_env = {"validation": object()}

    terminal_cleanup, graceful_cleanup = make_cleanup(
        checkpointer=checkpointer,
        trajectory_collector=collector,
        replay_buffer=replay_buffer,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
        policy_generation=generation,
        policy=policy,
        flush_collector_telemetry=flush_telemetry,
    )

    graceful_cleanup()
    graceful_cleanup()
    terminal_cleanup()
    capsys.readouterr()

    checkpointer.shutdown.assert_called_once_with()
    flush_telemetry.assert_called_once_with()
    assert ray_runtime.kill.call_args_list == [call(collector), call(replay_buffer)]
    shutdown_environments.assert_called_once_with(task_to_env, val_task_to_env)
    generation.shutdown.assert_called_once_with()
    policy.shutdown.assert_called_once_with()


def test_async_startup_guard_preserves_base_exception_identity() -> None:
    run_startup = _load_async_startup_guard()
    root_cause = KeyboardInterrupt("startup interrupted")
    terminal_cleanup = MagicMock(
        side_effect=SystemExit("cleanup must not replace startup failure")
    )

    def fail_startup() -> None:
        raise root_cause

    with pytest.raises(KeyboardInterrupt) as raised:
        run_startup(fail_startup, terminal_cleanup)

    assert raised.value is root_cause
    terminal_cleanup.assert_called_once_with()


def test_async_startup_failure_covers_every_post_actor_stage() -> None:
    source_path = Path(__file__).parents[3] / "nemo_rl/algorithms/grpo.py"
    parsed = ast.parse(source_path.read_text())
    train_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "async_grpo_train"
    )
    startup_function = next(
        node
        for node in ast.walk(train_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_startup"
    )
    startup_calls = set(_call_names(startup_function))
    startup_attributes = {
        node.attr
        for node in ast.walk(startup_function)
        if isinstance(node, ast.Attribute)
    }
    assert {
        "refit_policy_generation",
        "prepare_for_generation",
        "clear_logger_metrics",
    } <= startup_calls
    assert {
        "set_weight_version",
        "start_collection",
        "pause",
        "resume",
        "size",
        "check_health",
        "has_complete_batch",
        "get_status",
    } <= startup_attributes

    guarded_call = next(
        node
        for node in ast.walk(train_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_async_grpo_startup_with_cleanup"
    )
    assert isinstance(guarded_call.args[0], ast.Name)
    assert guarded_call.args[0].id == "_run_startup"
    assert isinstance(guarded_call.args[1], ast.Name)
    assert guarded_call.args[1].id == "_terminal_cleanup"

    validation_try = next(
        node
        for node in ast.walk(startup_function)
        if isinstance(node, ast.Try)
        and bool(node.body)
        and _call_names(node.body[0]) == ["validate"]
    )
    assert _call_names(validation_try.body[0]) == ["validate"]
    assert "finish_generation" in {
        name for statement in validation_try.orelse for name in _call_names(statement)
    }
    validation_lifecycle_try = next(
        node
        for node in ast.walk(startup_function)
        if isinstance(node, ast.Try)
        and validation_try in node.body
        and any(
            isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
            for handler in node.handlers
        )
    )
    validation_failure_handler = next(
        handler
        for handler in validation_lifecycle_try.handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    )
    assert "resume" in {
        node.attr
        for node in ast.walk(validation_failure_handler)
        if isinstance(node, ast.Attribute)
    }
    assert isinstance(validation_failure_handler.body[-1], ast.Raise)
    assert validation_failure_handler.body[-1].exc is None
    assert "resume" in {
        node.attr
        for statement in validation_lifecycle_try.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
    }

    main_try = next(
        node
        for node in ast.walk(train_node)
        if isinstance(node, ast.Try) and "maybe_gpu_profile_step" in _call_names(node)
    )
    terminal_handler = next(
        handler
        for handler in main_try.handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    )
    assert _call_names(terminal_handler) == ["_terminal_cleanup"]
    cleanup_try = next(
        node
        for node in terminal_handler.body
        if isinstance(node, ast.Try) and "_terminal_cleanup" in _call_names(node)
    )
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in cleanup_try.handlers
    )
    assert isinstance(terminal_handler.body[-1], ast.Raise)
    assert terminal_handler.body[-1].exc is None
    assert "_graceful_cleanup" in {
        name for statement in main_try.finalbody for name in _call_names(statement)
    }


@pytest.mark.parametrize(
    ("source_relative_path", "class_name", "expected"),
    [
        ("nemo_rl/models/generation/interfaces.py", "GenerationInterface", False),
        (
            "nemo_rl/models/generation/vllm/vllm_generation.py",
            "VllmGeneration",
            True,
        ),
    ],
)
def test_refit_worker_timeout_capability_is_explicit_on_interface_and_vllm(
    source_relative_path: str,
    class_name: str,
    expected: bool,
) -> None:
    source_path = Path(__file__).parents[3] / source_relative_path
    parsed = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    property_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "supports_refit_worker_timeout"
    )

    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in property_node.decorator_list
    )
    return_node = next(
        node for node in property_node.body if isinstance(node, ast.Return)
    )
    assert isinstance(return_node.value, ast.Constant)
    assert return_node.value.value is expected
