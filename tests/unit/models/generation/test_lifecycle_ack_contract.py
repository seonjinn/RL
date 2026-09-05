# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Framework-free tests for generation lifecycle acknowledgement handling."""

from __future__ import annotations

import ast
from math import isfinite
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).parents[4]


def _normalize_timeout(timeout_s: object) -> float:
    if type(timeout_s) not in (int, float):
        raise ValueError("timeout_s must be a positive finite int or float")
    normalized = float(cast(int | float, timeout_s))
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_s must be a positive finite int or float")
    return normalized


class _FakeRay:
    def __init__(self, result: object) -> None:
        self._result = result
        self.get_calls: list[object] = []
        self.get_timeouts: list[float] = []

    def get(self, refs: object, *, timeout: float) -> object:
        self.get_calls.append(refs)
        self.get_timeouts.append(timeout)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _find_function(path: Path, name: str) -> ast.FunctionDef:
    parsed = ast.parse(path.read_text())
    return next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _find_assignment(path: Path, name: str) -> ast.Assign:
    parsed = ast.parse(path.read_text())
    return next(
        node
        for node in parsed.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )


def _find_method(
    path: Path, class_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    parsed = ast.parse(path.read_text())
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _load_lifecycle_harness(
    *, backend: str, ray_result: object
) -> tuple[object, _FakeRay]:
    interfaces_path = _REPO_ROOT / "nemo_rl/models/generation/interfaces.py"
    default_timeout = _find_assignment(
        interfaces_path, "DEFAULT_GENERATION_LIFECYCLE_TIMEOUT_S"
    )
    helpers = [
        _find_function(interfaces_path, "normalize_generation_lifecycle_timeout_s"),
        _find_function(interfaces_path, "require_generation_leader_count"),
        _find_function(interfaces_path, "await_exact_worker_phase_results"),
        _find_function(interfaces_path, "require_exact_generation_phase_acks"),
        _find_function(interfaces_path, "await_generation_phase_acks"),
    ]
    if backend == "vllm":
        source_path = _REPO_ROOT / "nemo_rl/models/generation/vllm/vllm_generation.py"
        class_name = "VllmGeneration"
    elif backend == "trtllm":
        source_path = (
            _REPO_ROOT / "nemo_rl/models/generation/trtllm/trtllm_generation.py"
        )
        class_name = "TrtllmGeneration"
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported test backend: {backend}")

    methods = [
        _find_method(source_path, class_name, "prepare_for_generation"),
        _find_method(source_path, class_name, "finish_generation"),
    ]
    harness = ast.ClassDef(
        name="Harness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                default_timeout,
                *helpers,
                harness,
            ],
            type_ignores=[],
        )
    )
    fake_ray = _FakeRay(ray_result)
    namespace: dict[str, Any] = {
        "cast": cast,
        "ray": fake_ray,
        "isfinite": isfinite,
    }
    exec(compile(module, source_path, "exec"), namespace)
    namespace["normalize_generation_lifecycle_timeout_s"] = _normalize_timeout
    instance = namespace["Harness"]()
    instance.worker_group = MagicMock()
    instance.worker_group.dp_size = 2
    instance.worker_group.workers = [object(), object()]
    instance.worker_group.run_all_workers_single_data.return_value = [
        object(),
        object(),
    ]
    instance.dp_size = 2
    if backend == "vllm":
        instance.cfg = {
            "colocated": {"enabled": True},
            "vllm_cfg": {"async_engine": False},
        }
    else:
        instance.colocated_enabled = True
    return instance, fake_ray


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("method_name", ["prepare_for_generation", "finish_generation"])
def test_lifecycle_requires_exact_true_from_every_submitted_worker(
    backend: str, method_name: str
) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[True, True],
    )

    assert getattr(generation, method_name)() is True
    assert fake_ray.get_timeouts == [300.0]


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("method_name", ["prepare_for_generation", "finish_generation"])
def test_lifecycle_uses_explicit_refit_deadline(backend: str, method_name: str) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[True, True],
    )

    assert getattr(generation, method_name)(refit_timeout_s=17.5) is True
    assert fake_ray.get_timeouts == [17.5]


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("result", [[], [None, None], [True], [True, False], [True, 1]])
def test_lifecycle_rejects_empty_partial_or_malformed_acknowledgements(
    backend: str, result: object
) -> None:
    generation, _ = _load_lifecycle_harness(backend=backend, ray_result=result)

    with pytest.raises(RuntimeError, match="lifecycle|acknowledgement|worker"):
        generation.prepare_for_generation()


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_lifecycle_preserves_worker_exception_identity(backend: str) -> None:
    root_cause = RuntimeError("worker wake failed")
    generation, _ = _load_lifecycle_harness(
        backend=backend,
        ray_result=root_cause,
    )

    with pytest.raises(RuntimeError, match="worker wake failed") as caught:
        generation.prepare_for_generation()

    assert caught.value is root_cause


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_empty_future_group_is_rejected_before_ray_get(backend: str) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[],
    )
    generation.worker_group.run_all_workers_single_data.return_value = []

    with pytest.raises(RuntimeError, match="no lifecycle workers"):
        generation.finish_generation()

    assert fake_ray.get_calls == []


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_lifecycle_timeout_error_propagates_without_ack_fallback(backend: str) -> None:
    timeout = TimeoutError("worker lifecycle exceeded deadline")
    generation, _ = _load_lifecycle_harness(backend=backend, ray_result=timeout)

    with pytest.raises(TimeoutError) as caught:
        generation.prepare_for_generation(refit_timeout_s=0.01)

    assert caught.value is timeout


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("invalid_timeout", [None, 0, -1, True, float("inf"), "1"])
def test_invalid_lifecycle_timeout_fails_before_ray_get(
    backend: str, invalid_timeout: object
) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[True, True],
    )

    with pytest.raises(ValueError, match="positive finite"):
        generation.prepare_for_generation(refit_timeout_s=invalid_timeout)

    assert fake_ray.get_calls == []


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("future_count", [1, 3], ids=["missing", "extra"])
def test_lifecycle_rejects_future_count_that_disagrees_with_configured_dp(
    backend: str, future_count: int
) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[True] * future_count,
    )
    generation.worker_group.run_all_workers_single_data.return_value = [
        object() for _ in range(future_count)
    ]

    with pytest.raises(RuntimeError, match=r"expected exactly 2.*got"):
        generation.prepare_for_generation()

    assert fake_ray.get_calls == []


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_lifecycle_rejects_configured_runtime_dp_mismatch_before_ray_get(
    backend: str,
) -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend=backend,
        ray_result=[True, True],
    )
    generation.worker_group.dp_size = 3

    with pytest.raises(RuntimeError, match=r"configured.*runtime.*DP"):
        generation.finish_generation()

    assert fake_ray.get_calls == []


def test_generation_timeout_normalization_delegates_to_shared_contract() -> None:
    function = _find_function(
        _REPO_ROOT / "nemo_rl/models/generation/interfaces.py",
        "normalize_generation_lifecycle_timeout_s",
    )

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "nemo_rl.weight_sync.refit_supervisor"
        and any(alias.name == "normalize_refit_timeout_s" for alias in node.names)
        for node in ast.walk(function)
    )


def test_non_colocated_prepare_remains_a_noop() -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend="vllm",
        ray_result=RuntimeError("must not be read"),
    )
    generation.cfg["colocated"]["enabled"] = False

    assert generation.prepare_for_generation() is True
    generation.worker_group.run_all_workers_single_data.assert_not_called()
    assert fake_ray.get_calls == []


def test_trtllm_non_colocated_prepare_remains_a_noop() -> None:
    generation, fake_ray = _load_lifecycle_harness(
        backend="trtllm",
        ray_result=RuntimeError("must not be read"),
    )
    generation.colocated_enabled = False

    assert generation.prepare_for_generation() is True
    generation.worker_group.run_all_workers_single_data.assert_not_called()
    assert fake_ray.get_calls == []


def test_lifecycle_helper_rejects_non_list_ray_result() -> None:
    helper = _find_function(
        _REPO_ROOT / "nemo_rl/models/generation/interfaces.py",
        "require_exact_generation_phase_acks",
    )
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, "interfaces.py", "exec"), namespace)

    with pytest.raises(RuntimeError, match="exact list"):
        namespace["require_exact_generation_phase_acks"](
            phase="weights",
            expected_count=1,
            results=(True,),
        )


def test_trtllm_wake_rejects_missing_engine() -> None:
    source_path = _REPO_ROOT / "nemo_rl/models/generation/trtllm/trtllm_worker_async.py"
    method = _find_method(
        source_path, "TrtllmAsyncGenerationWorkerImpl", "wake_up_async"
    )
    harness = ast.ClassDef(
        name="Harness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                harness,
            ],
            type_ignores=[],
        )
    )
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, source_path, "exec"), namespace)
    worker = namespace["Harness"]()
    worker.llm = None

    import asyncio

    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(worker.wake_up_async())


@pytest.mark.parametrize("method_name", ["sleep_async", "reset_prefix_cache_async"])
def test_trtllm_sleep_and_cache_reset_reject_missing_engine(
    method_name: str,
) -> None:
    source_path = _REPO_ROOT / "nemo_rl/models/generation/trtllm/trtllm_worker_async.py"
    method = _find_method(source_path, "TrtllmAsyncGenerationWorkerImpl", method_name)
    harness = ast.ClassDef(
        name="Harness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                harness,
            ],
            type_ignores=[],
        )
    )
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, source_path, "exec"), namespace)
    worker = namespace["Harness"]()
    worker.llm = None

    import asyncio

    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(getattr(worker, method_name)())
