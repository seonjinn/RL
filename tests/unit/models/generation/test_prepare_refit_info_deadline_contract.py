# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Framework-free contracts for bounded refit metadata preparation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional, cast

import pytest

_REPO_ROOT = Path(__file__).parents[4]
_DEFAULT_TIMEOUT_S = 300.0


class _FakeRay:
    def __init__(self, results: object) -> None:
        self.results = results
        self.calls: list[tuple[object, float | None]] = []

    def get(self, refs: object, *, timeout: float | None = None) -> object:
        self.calls.append((refs, timeout))
        return self.results


class _WorkerGroup:
    def __init__(self, *, worker_count: int, futures: list[object]) -> None:
        self.workers = [object() for _ in range(worker_count)]
        self.dp_size = worker_count
        self.futures = futures

    def run_all_workers_single_data(
        self, *args: object, **kwargs: object
    ) -> list[object]:
        return self.futures


def _find_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    parsed = ast.parse(path.read_text())
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _normalize_timeout(timeout_s: object) -> float:
    if type(timeout_s) not in (int, float):
        raise ValueError("timeout_s must be a positive finite int or float")
    normalized = float(timeout_s)
    if normalized <= 0:
        raise ValueError("timeout_s must be a positive finite int or float")
    return normalized


def _await_exact_results(
    *, phase: str, expected_count: int, futures: object, timeout_s: object
) -> list[object]:
    timeout = _normalize_timeout(timeout_s)
    if type(futures) is not list or len(futures) != expected_count:
        actual = len(futures) if isinstance(futures, list) else "non-list"
        raise RuntimeError(
            f"worker phase {phase!r} expected exactly {expected_count} participant "
            f"futures, got {actual}"
        )
    results = _ACTIVE_RAY.get(futures, timeout=timeout)
    if type(results) is not list or len(results) != expected_count:
        actual = len(results) if isinstance(results, list) else "non-list"
        raise RuntimeError(
            f"worker phase {phase!r} expected exactly {expected_count} participant "
            f"results, got {actual}"
        )
    return results


def _require_leader_count(
    *, phase: str, configured_dp_size: object, runtime_dp_size: object
) -> int:
    del phase
    if configured_dp_size != runtime_dp_size:
        raise RuntimeError("configured DP size disagrees with runtime DP size")
    return cast(int, configured_dp_size)


_ACTIVE_RAY: _FakeRay


def _load_prepare_harness(
    backend: str, *, worker_count: int = 2, future_count: int = 2
) -> tuple[object, _FakeRay]:
    if backend == "policy":
        source_path = _REPO_ROOT / "nemo_rl/models/policy/lm_policy.py"
        class_name = "Policy"
        results: object = [{"rank": 0}, {"rank": 0}]
    elif backend == "vllm":
        source_path = _REPO_ROOT / "nemo_rl/models/generation/vllm/vllm_generation.py"
        class_name = "VllmGeneration"
        results = [None, None]
    elif backend == "trtllm":
        source_path = (
            _REPO_ROOT / "nemo_rl/models/generation/trtllm/trtllm_generation.py"
        )
        class_name = "TrtllmGeneration"
        results = [None, None]
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported backend: {backend}")

    method = _find_method(source_path, class_name, "prepare_refit_info")
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
                ast.Assign(
                    targets=[
                        ast.Name(
                            id="DEFAULT_GENERATION_LIFECYCLE_TIMEOUT_S",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=_DEFAULT_TIMEOUT_S),
                ),
                harness,
            ],
            type_ignores=[],
        )
    )
    fake_ray = _FakeRay(results)
    global _ACTIVE_RAY
    _ACTIVE_RAY = fake_ray
    namespace: dict[str, Any] = {
        "Any": Any,
        "Optional": Optional,
        "cast": cast,
        "ray": fake_ray,
        "normalize_generation_lifecycle_timeout_s": _normalize_timeout,
        "await_exact_worker_phase_results": _await_exact_results,
        "require_generation_leader_count": _require_leader_count,
    }
    exec(compile(module, source_path, "exec"), namespace)
    instance = namespace["Harness"]()
    instance.worker_group = _WorkerGroup(
        worker_count=worker_count,
        futures=[object() for _ in range(future_count)],
    )
    instance.dp_size = worker_count
    if backend == "vllm":
        instance.cfg = {"vllm_cfg": {"async_engine": False}}
    return instance, fake_ray


def _load_local_prepare_harness(backend: str) -> tuple[object, list[object]]:
    if backend == "megatron":
        source_path = (
            _REPO_ROOT / "nemo_rl/models/generation/megatron/megatron_generation.py"
        )
        class_name = "MegatronGeneration"
    elif backend == "sglang":
        source_path = (
            _REPO_ROOT / "nemo_rl/models/generation/sglang/sglang_generation.py"
        )
        class_name = "SGLangGeneration"
    elif backend == "dynamo":
        source_path = (
            _REPO_ROOT / "nemo_rl/models/generation/dynamo/dynamo_generation.py"
        )
        class_name = "DynamoGeneration"
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported backend: {backend}")

    method = _find_method(source_path, class_name, "prepare_refit_info")
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
                ast.Assign(
                    targets=[
                        ast.Name(
                            id="DEFAULT_GENERATION_LIFECYCLE_TIMEOUT_S",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=_DEFAULT_TIMEOUT_S),
                ),
                harness,
            ],
            type_ignores=[],
        )
    )
    timeout_calls: list[object] = []

    def normalize(timeout_s: object) -> float:
        timeout_calls.append(timeout_s)
        return _normalize_timeout(timeout_s)

    namespace: dict[str, Any] = {
        "Any": Any,
        "Optional": Optional,
        "normalize_generation_lifecycle_timeout_s": normalize,
    }
    exec(compile(module, source_path, "exec"), namespace)
    instance = namespace["Harness"]()
    if backend == "dynamo":

        class Channel:
            def prepare(self, state_dict_info: object) -> None:
                self.state_dict_info = state_dict_info

        instance._refit_channel = Channel()
    return instance, timeout_calls


@pytest.mark.parametrize("backend", ["policy", "vllm", "trtllm"])
def test_prepare_refit_info_has_a_finite_default_deadline(backend: str) -> None:
    instance, fake_ray = _load_prepare_harness(backend)

    if backend == "policy":
        assert instance.prepare_refit_info() == {"rank": 0}
    else:
        assert instance.prepare_refit_info({"weight": object()}) is None

    assert fake_ray.calls[0][1] == _DEFAULT_TIMEOUT_S


@pytest.mark.parametrize("backend", ["policy", "vllm", "trtllm"])
def test_prepare_refit_info_honors_an_explicit_deadline(backend: str) -> None:
    instance, fake_ray = _load_prepare_harness(backend)

    if backend == "policy":
        instance.prepare_refit_info(refit_timeout_s=17.5)
    else:
        instance.prepare_refit_info({"weight": object()}, refit_timeout_s=17.5)

    assert fake_ray.calls[0][1] == 17.5


@pytest.mark.parametrize("backend", ["policy", "vllm", "trtllm"])
def test_prepare_refit_info_rejects_missing_participant_before_ray_get(
    backend: str,
) -> None:
    instance, fake_ray = _load_prepare_harness(backend, future_count=1)

    with pytest.raises(RuntimeError, match=r"expected exactly 2.*got 1"):
        if backend == "policy":
            instance.prepare_refit_info()
        else:
            instance.prepare_refit_info({"weight": object()})

    assert fake_ray.calls == []


@pytest.mark.parametrize("backend", ["policy", "vllm", "trtllm"])
def test_prepare_refit_info_rejects_missing_participant_result(backend: str) -> None:
    instance, fake_ray = _load_prepare_harness(backend)
    fake_ray.results = [None]

    with pytest.raises(RuntimeError, match=r"expected exactly 2.*results.*got 1"):
        if backend == "policy":
            instance.prepare_refit_info()
        else:
            instance.prepare_refit_info({"weight": object()})


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_generation_prepare_refit_info_requires_exact_none(backend: str) -> None:
    instance, fake_ray = _load_prepare_harness(backend)
    fake_ray.results = [None, True]

    with pytest.raises(RuntimeError, match="must return exact None"):
        instance.prepare_refit_info({"weight": object()})


@pytest.mark.parametrize(
    ("path", "class_name"),
    [
        ("nemo_rl/models/policy/interfaces.py", "ColocatablePolicyInterface"),
        ("nemo_rl/models/generation/interfaces.py", "GenerationInterface"),
    ],
)
def test_prepare_refit_info_interface_exposes_finite_default(
    path: str, class_name: str
) -> None:
    method = _find_method(_REPO_ROOT / path, class_name, "prepare_refit_info")

    assert method.args.defaults
    default = method.args.defaults[-1]
    assert isinstance(default, ast.Name)
    assert default.id == "DEFAULT_GENERATION_LIFECYCLE_TIMEOUT_S"


@pytest.mark.parametrize("backend", ["megatron", "sglang", "dynamo"])
def test_local_prepare_refit_info_validates_its_finite_default(backend: str) -> None:
    instance, timeout_calls = _load_local_prepare_harness(backend)

    instance.prepare_refit_info({"weight": object()})

    assert timeout_calls == [_DEFAULT_TIMEOUT_S]


@pytest.mark.parametrize("backend", ["megatron", "sglang", "dynamo"])
def test_local_prepare_refit_info_honors_explicit_deadline(backend: str) -> None:
    instance, timeout_calls = _load_local_prepare_harness(backend)

    instance.prepare_refit_info({"weight": object()}, refit_timeout_s=17.5)

    assert timeout_calls == [17.5]
