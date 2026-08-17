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

from __future__ import annotations

import ast
import copy
import gc
import sys
import weakref
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.megatron.cuda_graph_lifecycle import (
    TECudaGraphLifecycle,
    TECudaGraphScheduleKey,
)
from nemo_rl.models.megatron.cuda_graph_storage import (
    GraphStorageFingerprint,
    StorageChange,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_PATH = _REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"


class _Phase(Enum):
    IDLE = auto()
    SPLIT_OPEN_BEFORE_FIRST = auto()
    GRAPH_SCHEDULE_LIVE = auto()
    SPLIT_OPEN_AFTER_FIRST = auto()


class _ReplayableTestIterator:
    def __init__(self, items: list[Any]) -> None:
        self._items = tuple(items)
        self._iterator = iter(self._items)

    def __iter__(self) -> "_ReplayableTestIterator":
        return self

    def __next__(self) -> Any:
        return next(self._iterator)

    def replay(self) -> Any:
        return iter(self._items)


def _extract_worker_methods(
    method_names: set[str], namespace: dict[str, Any] | None = None
) -> type:
    """Compile selected production methods without importing GPU dependencies."""
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    }
    missing = method_names - methods.keys()
    assert not missing, f"worker is missing required methods: {sorted(missing)}"
    for method in methods.values():
        method.decorator_list = []

    class_kwargs: dict[str, Any] = {
        "name": "_Worker",
        "bases": [],
        "keywords": [],
        "body": list(methods.values()),
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_kwargs["type_params"] = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.ClassDef(**class_kwargs),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    globals_dict = {
        "Any": Any,
        "Iterator": Any,
        "ProcessedMicrobatch": Any,
        "TECudaGraphScheduleKey": TECudaGraphScheduleKey,
        "_TECudaGraphPreflightSample": SimpleNamespace,
        "_TECudaGraphWorkerPhase": _Phase,
        "clear_router_replay": lambda _model: None,
        "record_router_replay_graph_error": lambda _model, _error: None,
    }
    if namespace is not None:
        globals_dict.update(namespace)
    exec(compile(module, str(_WORKER_PATH), "exec"), globals_dict)
    worker_type = globals_dict["_Worker"]
    if (
        "_ensure_te_cuda_graph_schedule" in method_names
        and "_assert_hybridep_preprocess_capture_padding_disabled"
        not in method_names
    ):
        worker_type._assert_hybridep_preprocess_capture_padding_disabled = (
            lambda _self: None
        )
    return worker_type


def test_runtime_schedule_provider_and_explicit_drained_phases() -> None:
    worker_type = _extract_worker_methods(
        {
            "_te_cuda_graph_runtime_num_microbatches",
            "_assert_te_cuda_graph_model_drained",
        }
    )
    worker = worker_type()
    worker._te_cuda_graph_runtime_schedule_count = 7

    worker._te_cuda_graph_phase = _Phase.IDLE
    assert worker._te_cuda_graph_runtime_num_microbatches() == 7
    assert worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_BEFORE_FIRST
    worker._train_step_state = {
        "te_cuda_graph_key": None,
        "total_num_microbatches": 0,
    }
    assert worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.GRAPH_SCHEDULE_LIVE
    assert not worker._assert_te_cuda_graph_model_drained()

    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_AFTER_FIRST
    assert not worker._assert_te_cuda_graph_model_drained()


def test_router_replay_generation_provider_is_typed_and_strictly_advancing() -> None:
    worker_type = _extract_worker_methods(
        {"_next_router_replay_microbatch_generation"}
    )
    worker = worker_type()
    worker._next_router_route_generation = 4

    generations = [
        worker._next_router_replay_microbatch_generation(),
        worker._next_router_replay_microbatch_generation(),
        worker._next_router_replay_microbatch_generation(),
    ]

    assert generations == [4, 5, 6]
    assert all(type(generation) is int for generation in generations)


def test_initialization_uses_configured_cache_capacity_and_topology_only_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    lifecycle_calls: list[dict[str, Any]] = []

    class FakeManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            manager_calls.append((args, kwargs))

        @classmethod
        def from_helper(cls, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("from_helper must not use the global calculator")

    class FakeLifecycle:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle_calls.append(kwargs)

    fake_module = SimpleNamespace(TECudaGraphBankManager=FakeManager)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.te_cuda_graph_bank",
        fake_module,
    )
    worker_type = _extract_worker_methods(
        {
            "_initialize_te_cuda_graph_lifecycle",
            "_te_cuda_graph_runtime_num_microbatches",
        },
        {
            "TECudaGraphLifecycle": FakeLifecycle,
            "_TE_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY": 2,
            "_TE_CUDA_GRAPH_WARMUP_STEPS": 3,
            "log": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        },
    )
    worker = worker_type()
    topology_layers = (object(), object())
    topology_helper = SimpleNamespace(flattened_callables=topology_layers)
    sample_args: list[Any] = []
    worker._build_te_cuda_graph_helper = lambda sample: (
        sample_args.append(sample) or topology_helper
    )
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(cuda_graph_modules=("attention",))
    )
    worker.cfg = {"megatron_cfg": {"cuda_graph_max_cached_schedules": 3}}
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._assert_te_cuda_graph_model_drained = lambda: True

    worker._initialize_te_cuda_graph_lifecycle()

    assert sample_args == [None]
    assert lifecycle_calls == [{"capacity": 3, "warmup_steps": 3}]
    assert len(manager_calls) == 1
    args, kwargs = manager_calls[0]
    assert args == (topology_layers,)
    assert kwargs["cuda_graph_modules"] == ("attention",)
    assert kwargs["assert_model_drained"]() is True
    assert kwargs["runtime_num_microbatches"]() == 1
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None


def test_initialization_preserves_two_bank_default_when_config_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_calls: list[dict[str, Any]] = []

    class FakeManager:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class FakeLifecycle:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle_calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.te_cuda_graph_bank",
        SimpleNamespace(TECudaGraphBankManager=FakeManager),
    )
    worker_type = _extract_worker_methods(
        {
            "_initialize_te_cuda_graph_lifecycle",
            "_te_cuda_graph_runtime_num_microbatches",
        },
        {
            "TECudaGraphLifecycle": FakeLifecycle,
            "_TE_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY": 2,
            "_TE_CUDA_GRAPH_WARMUP_STEPS": 3,
            "log": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        },
    )
    worker = worker_type()
    worker._build_te_cuda_graph_helper = lambda _sample: SimpleNamespace(
        flattened_callables=()
    )
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(cuda_graph_modules=("attention",))
    )
    worker.cfg = {"megatron_cfg": {}}
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._assert_te_cuda_graph_model_drained = lambda: True

    worker._initialize_te_cuda_graph_lifecycle()

    assert lifecycle_calls == [{"capacity": 2, "warmup_steps": 3}]


def test_effective_graph_config_rpc_uses_validated_model_config() -> None:
    worker_type = _extract_worker_methods(
        {"get_effective_te_cuda_graph_config"},
        {"_TE_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY": 2},
    )
    worker = worker_type()
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(
            cuda_graph_impl="transformer_engine",
            thd_max_packed_sequences=65,
        )
    )
    worker._te_cuda_graph_lifecycle = object()
    worker.cfg = {"megatron_cfg": {"cuda_graph_max_cached_schedules": 3}}

    assert worker.get_effective_te_cuda_graph_config() == {
        "cuda_graph_impl": "transformer_engine",
        "thd_max_packed_sequences": 65,
        "cuda_graph_max_cached_schedules": 3,
        "training_enabled": True,
    }


def test_disabled_graph_config_omits_irrelevant_cache_capacity() -> None:
    worker_type = _extract_worker_methods(
        {"get_effective_te_cuda_graph_config"},
        {"_TE_CUDA_GRAPH_DEFAULT_CACHE_CAPACITY": 2},
    )
    worker = worker_type()
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(
            cuda_graph_impl="none",
            thd_max_packed_sequences=None,
        )
    )
    worker._te_cuda_graph_lifecycle = None
    worker.cfg = {"megatron_cfg": {}}

    assert worker.get_effective_te_cuda_graph_config() == {
        "cuda_graph_impl": "none",
        "thd_max_packed_sequences": None,
        "cuda_graph_max_cached_schedules": None,
        "training_enabled": False,
    }


def test_peek_preserves_actual_microbatches_and_counts_geometry_on_yield() -> None:
    worker_type = _extract_worker_methods(
        {
            "_peek_te_cuda_graph_training_iterator",
            "_record_te_cuda_graph_geometry",
        }
    )
    worker = worker_type()
    packed_params = object()
    first = SimpleNamespace(
        packed_seq_params=packed_params,
        packed_geometry=SimpleNamespace(
            logical_tokens=11,
            padded_tokens=16,
            capacity_tokens=32,
            real_sequence_count=3,
            cu_seqlens_capacity_entries=9,
        ),
    )
    second = SimpleNamespace(
        packed_seq_params=object(),
        packed_geometry=SimpleNamespace(
            logical_tokens=7,
            padded_tokens=8,
            capacity_tokens=32,
            real_sequence_count=2,
            cu_seqlens_capacity_entries=9,
        ),
    )
    call_state = SimpleNamespace(
        logical_tokens=0,
        padded_tokens=0,
        capacity_tokens=0,
    )

    actual_first, preserved = worker._peek_te_cuda_graph_training_iterator(
        iter((first, second)), call_state
    )

    assert actual_first is first
    assert actual_first.packed_seq_params is packed_params
    assert call_state.logical_tokens == 0
    assert list(preserved) == [first, second]
    assert (
        call_state.logical_tokens,
        call_state.padded_tokens,
        call_state.capacity_tokens,
    ) == (18, 24, 64)


def test_first_microbatch_preflight_failure_is_raised_collectively() -> None:
    worker_type = _extract_worker_methods(
        {"_collectively_peek_te_cuda_graph_training_iterator"}
    )
    worker = worker_type()
    local_error = ValueError("bad local THD geometry")
    worker._peek_te_cuda_graph_training_iterator = lambda *_args: (_ for _ in ()).throw(
        local_error
    )
    collective_calls: list[tuple[Any, str]] = []

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        raise RuntimeError("collective preflight failure") from error

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively

    with pytest.raises(RuntimeError, match="collective preflight failure"):
        worker._collectively_peek_te_cuda_graph_training_iterator(iter(()), object())

    assert collective_calls == [(local_error, "microbatch preflight")]


def test_router_route_preflight_runs_before_bank_schedule_activation() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    for method_name in ("train", "_train_microbatch_body_impl"):
        calls = [node for node in ast.walk(methods[method_name]) if isinstance(node, ast.Call)]
        preflight_line = min(
            call.lineno
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "_collectively_preflight_router_replay_microbatches"
        )
        ensure_line = min(
            call.lineno
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "_ensure_te_cuda_graph_schedule"
        )
        assert preflight_line < ensure_line


def test_router_route_preflight_rejects_later_invalid_microbatch_collectively() -> None:
    worker_type = _extract_worker_methods(
        {"_collectively_preflight_router_replay_microbatches"}
    )
    worker = worker_type()
    worker.model = object()
    worker._router_replay_enabled = True
    worker._active_router_route_generation = None
    call_state = SimpleNamespace(
        router_route_generation=None,
        router_route_signature=None,
    )
    microbatches = [
        SimpleNamespace(
            packed_seq_params=object(),
            packed_geometry=object(),
            routed_experts_cp_sharded="routes-1",
            structural_padding_mask_cp_sharded="mask-1",
            microbatch_generation=7,
        ),
        SimpleNamespace(
            packed_seq_params=object(),
            packed_geometry=object(),
            routed_experts_cp_sharded="routes-2",
            structural_padding_mask_cp_sharded="mask-2",
            microbatch_generation=8,
        ),
    ]
    collective_calls: list[tuple[Any, str]] = []
    validated: list[int] = []

    def validate_route(
        _model: object,
        routes: str,
        _mask: str,
        *,
        microbatch_generation: int,
    ) -> tuple[str, ...]:
        validated.append(microbatch_generation)
        if routes == "routes-2":
            raise ValueError("duplicate experts in later microbatch")
        return ("physical-signature",)

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        if error is not None:
            raise RuntimeError("collective later route failure") from error

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively
    worker._record_te_cuda_graph_geometry = lambda *_args: None

    method = worker._collectively_preflight_router_replay_microbatches
    method.__globals__["validate_router_replay_graph_microbatch"] = validate_route

    with pytest.raises(RuntimeError, match="collective later route failure"):
        method(
            _ReplayableTestIterator(microbatches),
            call_state,
        )

    assert validated == [7, 8]
    assert len(collective_calls) == 1
    assert collective_calls[0][1] == "router route preflight"
    assert isinstance(collective_calls[0][0], ValueError)


def test_router_route_preflight_rejects_later_generation_mismatch_collectively() -> (
    None
):
    worker_type = _extract_worker_methods(
        {"_collectively_preflight_router_replay_microbatches"}
    )
    worker = worker_type()
    worker.model = object()
    worker._router_replay_enabled = True
    worker._active_router_route_generation = None
    call_state = SimpleNamespace(
        router_route_generation=None,
        router_route_signature=None,
    )
    microbatches = [
        SimpleNamespace(
            packed_seq_params=object(),
            packed_geometry=object(),
            routed_experts_cp_sharded=object(),
            structural_padding_mask_cp_sharded=object(),
            microbatch_generation=generation,
        )
        for generation in (10, 11)
    ]
    collective_calls: list[tuple[Any, str]] = []
    agreements: list[tuple[int, str]] = []

    def agree(value: int, *, name: str, group: Any = None) -> int:
        del group
        agreements.append((value, name))
        if name.endswith("generation[1]"):
            raise RuntimeError("10 != 11")
        return value

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        if error is not None:
            raise RuntimeError("collective later generation mismatch") from error

    worker._collectively_validate_te_cuda_graph_integer = agree
    worker._collectively_raise_te_cuda_graph_failure = raise_collectively
    worker._record_te_cuda_graph_geometry = lambda *_args: None
    method = worker._collectively_preflight_router_replay_microbatches
    method.__globals__["validate_router_replay_graph_microbatch"] = (
        lambda *_args, **_kwargs: ("physical-signature",)
    )

    with pytest.raises(RuntimeError, match="collective later generation mismatch"):
        method(_ReplayableTestIterator(microbatches), call_state)

    assert agreements == [
        (2, "router route microbatch count"),
        (10, "router route microbatch generation[0]"),
        (11, "router route microbatch generation[1]"),
    ]
    assert collective_calls[0] == (None, "router route preflight")
    assert collective_calls[-1][1] == "router route generation agreement"


def test_all_microbatch_preflight_retains_only_compact_metadata_for_replay() -> None:
    class Payload:
        pass

    class ReplayableMicrobatches:
        def __init__(self) -> None:
            self.preflight_payload_refs: list[weakref.ReferenceType[Payload]] = []
            self._iterator = self._make(preflight=True)

        def _make(self, *, preflight: bool) -> Any:
            for generation in (20, 21, 22):
                payload = Payload()
                if preflight:
                    self.preflight_payload_refs.append(weakref.ref(payload))
                yield SimpleNamespace(
                    identity=generation,
                    full_cuda_payload=payload,
                    packed_seq_params=SimpleNamespace(generation=generation),
                    packed_geometry=SimpleNamespace(
                        logical_tokens=1,
                        padded_tokens=2,
                        capacity_tokens=4,
                    ),
                    routed_experts_cp_sharded=f"routes-{generation}",
                    structural_padding_mask_cp_sharded=f"mask-{generation}",
                    microbatch_generation=generation,
                )

        def __iter__(self) -> "ReplayableMicrobatches":
            return self

        def __next__(self) -> Any:
            return next(self._iterator)

        def replay(self) -> Any:
            return self._make(preflight=False)

    worker_type = _extract_worker_methods(
        {
            "_collectively_preflight_router_replay_microbatches",
            "_record_te_cuda_graph_geometry",
        }
    )
    worker = worker_type()
    worker.model = object()
    worker._router_replay_enabled = True
    worker._active_router_route_generation = None
    worker._collectively_raise_te_cuda_graph_failure = (
        lambda error, *, operation: (_ for _ in ()).throw(error)
        if error is not None
        else None
    )
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    call_state = SimpleNamespace(
        router_route_generation=None,
        router_route_signature=None,
        logical_tokens=0,
        padded_tokens=0,
        capacity_tokens=0,
    )
    source = ReplayableMicrobatches()
    method = worker._collectively_preflight_router_replay_microbatches
    method.__globals__["validate_router_replay_graph_microbatch"] = (
        lambda *_args, **_kwargs: ("physical-signature",)
    )

    first_sample, schedule_iterator = method(
        source,
        call_state,
        expected_num_microbatches=3,
    )
    gc.collect()

    assert all(reference() is None for reference in source.preflight_payload_refs)
    assert not hasattr(first_sample, "full_cuda_payload")
    assert [microbatch.identity for microbatch in schedule_iterator] == [20, 21, 22]


@pytest.mark.parametrize("method_name", ["train", "_train_microbatch_body_impl"])
def test_non_r3_active_bank_never_requires_router_launch_evidence(
    method_name: str,
) -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    resolver_calls = [
        call
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr
        == "_collectively_resolve_router_replay_graph_launch_expected"
    ]

    assert len(resolver_calls) == 1
    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in resolver_calls[0].keywords
    }
    assert keywords["enabled"] == "use_router_replay"

    worker_type = _extract_worker_methods(
        {
            "_collectively_resolve_router_replay_graph_launch_expected",
            "_te_cuda_graph_launch_expected",
        }
    )
    worker = worker_type()
    key = TECudaGraphScheduleKey(2)
    worker._te_cuda_graph_bank_manager = SimpleNamespace(active_bank=object())
    worker._te_cuda_graph_installed_key = key
    worker._collectively_raise_te_cuda_graph_failure = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-R3 training must not resolve route launch evidence")
        )
    )

    assert worker._te_cuda_graph_launch_expected(key) is True
    assert (
        worker._collectively_resolve_router_replay_graph_launch_expected(
            key,
            enabled=False,
        )
        is False
    )


def test_rank_local_launch_expectation_failure_is_collective_before_schedule() -> (
    None
):
    worker_type = _extract_worker_methods(
        {
            "_collectively_raise_te_cuda_graph_failure",
            "_collectively_resolve_router_replay_graph_launch_expected",
        }
    )
    worker = worker_type()
    bank_resets: list[str] = []
    schedule_entries: list[str] = []
    failed = SimpleNamespace(item=lambda: 1)
    fake_torch = SimpleNamespace(
        int32=object(),
        tensor=lambda *_args, **_kwargs: failed,
        distributed=SimpleNamespace(
            ReduceOp=SimpleNamespace(MAX=object()),
            all_reduce=lambda *_args, **_kwargs: None,
        ),
    )
    resolver = worker._collectively_resolve_router_replay_graph_launch_expected
    resolver.__globals__["torch"] = fake_torch
    resolver.__globals__["clear_router_replay"] = lambda _model: None
    worker.model = object()
    worker._active_router_route_generation = 17
    worker._te_cuda_graph_device = lambda: "cpu"
    worker._reset_te_cuda_graph_banks_after_failure = lambda: bank_resets.append(
        "reset"
    )
    worker._te_cuda_graph_launch_expected = lambda _key: (_ for _ in ()).throw(
        RuntimeError("rank-local key/bank mismatch")
    )
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("agreement must wait for local failure resolution")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="router replay launch expectation failed collectively",
    ):
        worker._collectively_resolve_router_replay_graph_launch_expected(
            TECudaGraphScheduleKey(2),
            enabled=True,
        )
        schedule_entries.append("entered")

    assert schedule_entries == []
    assert bank_resets == ["reset"]
    assert worker._active_router_route_generation is None


def test_post_preflight_regions_are_enclosed_by_route_cleanup() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    required_names = {
        "LossPostProcessor",
        "get_rerun_state_machine",
        "maybe_r3_trace_stage",
    }
    required_attributes = {
        "zero_grad_buffer",
        "_copy_main_params_to_param_buffer",
        "_set_moe_grad_scale_func",
        "_set_mtp_grad_scale_func",
    }

    for method_name in ("train",):
        method = methods[method_name]
        cleanup_with = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr
                == "_router_replay_lifecycle_cleanup"
                for item in node.items
            )
        )
        covered_calls = [
            node for statement in cleanup_with.body for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        all_calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        method_names_present = {
            call.func.id for call in all_calls if isinstance(call.func, ast.Name)
        }
        method_attrs_present = {
            call.func.attr
            for call in all_calls
            if isinstance(call.func, ast.Attribute)
        }
        covered_names = {
            call.func.id for call in covered_calls if isinstance(call.func, ast.Name)
        }
        covered_attrs = {
            call.func.attr
            for call in covered_calls
            if isinstance(call.func, ast.Attribute)
        }
        assert required_names.intersection(method_names_present) <= covered_names
        assert required_attributes.intersection(method_attrs_present) <= covered_attrs

    split_wrapper = methods["_train_microbatch_body"]
    split_cleanup = next(
        node
        for node in ast.walk(split_wrapper)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "_router_replay_lifecycle_cleanup"
            for item in node.items
        )
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_train_microbatch_body_impl"
        for statement in split_cleanup.body
        for node in ast.walk(statement)
    )
    split_impl_calls = [
        node
        for node in ast.walk(methods["_train_microbatch_body_impl"])
        if isinstance(node, ast.Call)
    ]
    split_names = {
        call.func.id for call in split_impl_calls if isinstance(call.func, ast.Name)
    }
    assert required_names <= split_names


@pytest.mark.parametrize(
    "region",
    [
        "loss processor",
        "rerun setup",
        "zero grad",
        "parameter copy",
        "gradient scale setup",
        "trace context setup",
    ],
)
def test_router_cleanup_runs_for_every_preschedule_exception_region(
    region: str,
) -> None:
    cleared: list[object] = []
    worker_type = _extract_worker_methods(
        {"_router_replay_lifecycle_cleanup"},
        {"clear_router_replay": lambda model: cleared.append(model)},
    )
    worker_type._router_replay_lifecycle_cleanup = contextmanager(
        worker_type._router_replay_lifecycle_cleanup
    )
    worker = worker_type()
    worker.model = object()
    worker._active_router_route_generation = 17

    with pytest.raises(RuntimeError, match=region):
        with worker._router_replay_lifecycle_cleanup(enabled=True):
            raise RuntimeError(region)

    assert cleared == [worker.model]
    assert worker._active_router_route_generation is None


def test_graph_launch_expectation_tracks_the_installed_active_bank() -> None:
    worker_type = _extract_worker_methods({"_te_cuda_graph_launch_expected"})
    worker = worker_type()
    key = TECudaGraphScheduleKey(2)
    worker._te_cuda_graph_bank_manager = SimpleNamespace(active_bank=None)
    worker._te_cuda_graph_installed_key = None

    assert worker._te_cuda_graph_launch_expected(key) is False

    worker._te_cuda_graph_bank_manager.active_bank = object()
    worker._te_cuda_graph_installed_key = key
    assert worker._te_cuda_graph_launch_expected(key) is True

    worker._te_cuda_graph_installed_key = None
    with pytest.raises(RuntimeError, match="installed key and active bank disagree"):
        worker._te_cuda_graph_launch_expected(key)


def test_training_handoffs_pass_explicit_graph_launch_expectation() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    for method_name in ("train", "_train_microbatch_body_impl"):
        handoffs = [
            call
            for call in ast.walk(methods[method_name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "megatron_forward_backward"
        ]
        assert handoffs
        for handoff in handoffs:
            keywords = {keyword.arg for keyword in handoff.keywords}
            assert "router_replay_graph_schedule_key" in keywords
            assert "router_replay_graph_launch_expected" in keywords


def test_router_route_counters_reduce_globally_and_block_unsafe_state() -> None:
    class FakeTensor:
        def __init__(self, values: int | list[int]) -> None:
            self.values = [values] if isinstance(values, int) else values

        def __getitem__(self, index: int) -> "FakeTensor":
            return FakeTensor([self.values[index]])

        def item(self) -> int:
            assert len(self.values) == 1
            return self.values[0]

    calls: list[tuple[Any, Any]] = []
    reduce_op = SimpleNamespace(MAX="max", SUM="sum")

    def all_reduce(tensor: FakeTensor, *, op: Any) -> None:
        calls.append((op, None))
        if op == reduce_op.SUM:
            tensor.values = [value * 8 for value in tensor.values]

    fake_torch = SimpleNamespace(
        int32="int32",
        int64="int64",
        tensor=lambda values, *, dtype, device: FakeTensor(values),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    counter_fields = (
        "route_payloads_produced",
        "route_payloads_copied",
        "route_graph_launches",
        "route_eager_warmup_payloads",
        "fallback_count",
        "missing_route_count",
        "stale_generation_count",
        "malformed_route_count",
        "out_of_range_count",
        "duplicate_route_count",
        "cp_mismatch_count",
    )
    worker_type = _extract_worker_methods(
        {"_finalize_router_replay_graph_counters"},
        {
            "torch": fake_torch,
            "ROUTER_REPLAY_GRAPH_COUNTER_FIELDS": counter_fields,
            "ROUTER_REPLAY_GRAPH_UNSAFE_COUNTER_FIELDS": counter_fields[4:],
            "trace_router_replay_graph_counters": lambda _counters: None,
        },
    )
    worker = worker_type()
    worker.model = object()
    worker._te_cuda_graph_device = lambda: "cuda"
    current = {name: 0 for name in counter_fields}
    current.update(
        route_payloads_produced=3,
        route_payloads_copied=2,
        route_graph_launches=2,
        stale_generation_count=1,
    )
    worker._snapshot_router_replay_graph_counters = lambda: current
    call_state = SimpleNamespace(
        router_replay_counter_snapshot={name: 0 for name in counter_fields}
    )

    with pytest.raises(RuntimeError, match="stale_generation_count=8"):
        worker._finalize_router_replay_graph_counters(call_state)

    assert calls == [(reduce_op.MAX, None), (reduce_op.SUM, None)]


def test_router_route_counters_require_produced_copied_launch_consistency() -> None:
    class FakeTensor:
        def __init__(self, values: int | list[int]) -> None:
            self.values = [values] if isinstance(values, int) else values

        def __getitem__(self, index: int) -> "FakeTensor":
            return FakeTensor([self.values[index]])

        def item(self) -> int:
            assert len(self.values) == 1
            return self.values[0]

    reduce_op = SimpleNamespace(MAX="max", SUM="sum")
    fake_torch = SimpleNamespace(
        int32="int32",
        int64="int64",
        tensor=lambda values, *, dtype, device: FakeTensor(values),
        distributed=SimpleNamespace(
            ReduceOp=reduce_op,
            all_reduce=lambda _tensor, *, op: None,
        ),
    )
    counter_fields = (
        "route_payloads_produced",
        "route_payloads_copied",
        "route_graph_launches",
        "route_eager_warmup_payloads",
        "fallback_count",
        "missing_route_count",
        "stale_generation_count",
        "malformed_route_count",
        "out_of_range_count",
        "duplicate_route_count",
        "cp_mismatch_count",
    )
    worker_type = _extract_worker_methods(
        {"_finalize_router_replay_graph_counters"},
        {
            "torch": fake_torch,
            "ROUTER_REPLAY_GRAPH_COUNTER_FIELDS": counter_fields,
            "ROUTER_REPLAY_GRAPH_UNSAFE_COUNTER_FIELDS": counter_fields[4:],
            "trace_router_replay_graph_counters": lambda _counters: None,
        },
    )
    worker = worker_type()
    worker.model = object()
    worker._te_cuda_graph_device = lambda: "cuda"
    current = {name: 0 for name in counter_fields}
    current.update(
        route_payloads_produced=3,
        route_payloads_copied=2,
        route_graph_launches=2,
    )
    worker._snapshot_router_replay_graph_counters = lambda: current
    call_state = SimpleNamespace(
        router_replay_counter_snapshot={name: 0 for name in counter_fields}
    )

    with pytest.raises(RuntimeError, match="inconsistent route evidence"):
        worker._finalize_router_replay_graph_counters(call_state)


def test_schedule_uses_exact_sample_and_two_entry_lru() -> None:
    worker_type = _extract_worker_methods({"_ensure_te_cuda_graph_schedule"})
    worker = worker_type()
    worker._te_cuda_graph_lifecycle = TECudaGraphLifecycle(
        capacity=2,
        warmup_steps=0,
    )
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._te_cuda_graph_installed_key = None
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 2
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    worker._collectively_validate_te_cuda_graph_storage_before_replay = (
        lambda *, operation: None
    )
    worker._bind_te_cuda_graph_storage_after_capture = lambda: None
    manager = SimpleNamespace(active_bank=None, uninstall=lambda: None)
    worker._te_cuda_graph_bank_manager = manager
    captures: list[tuple[TECudaGraphScheduleKey, Any]] = []

    class Bank:
        def __init__(self, key: TECudaGraphScheduleKey) -> None:
            self.key = key
            self.reset_count = 0

        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            self.reset_count += 1
            if manager.active_bank is self:
                manager.active_bank = None

    def capture(key: TECudaGraphScheduleKey, sample: Any) -> Bank:
        captures.append((key, sample))
        return Bank(key)

    worker._capture_te_cuda_graph_bank = capture
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    samples = [SimpleNamespace(packed_seq_params=object()) for _ in range(4)]
    call_states = []

    for count, first in zip((5, 3, 5, 7), samples):
        call_state = SimpleNamespace(
            capture_count=0,
            replay_count=0,
            cache_hit_count=0,
            cache_miss_count=0,
            eviction_count=0,
            normalized_schedule_key=None,
        )
        call_states.append(call_state)
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=count,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=True,
        )

    assert [(key.num_microbatches, sample) for key, sample in captures] == [
        (5, samples[0].packed_seq_params),
        (3, samples[1].packed_seq_params),
        (7, samples[3].packed_seq_params),
    ]
    assert sum(state.capture_count for state in call_states) == 3
    assert sum(state.cache_hit_count for state in call_states) == 1
    assert sum(state.cache_miss_count for state in call_states) == 3
    assert sum(state.eviction_count for state in call_states) == 1
    assert [state.normalized_schedule_key for state in call_states] == [5, 3, 5, 7]
    assert worker._te_cuda_graph_runtime_schedule_count == 7
    assert worker._te_cuda_graph_installed_key == TECudaGraphScheduleKey(7)


def test_remote_storage_drift_stops_every_rank_before_graph_activation() -> None:
    worker_type = _extract_worker_methods({"_ensure_te_cuda_graph_schedule"})
    worker = worker_type()
    events: list[str] = []
    worker._te_cuda_graph_lifecycle = SimpleNamespace(
        ensure_active=lambda *_args: events.append("activate")
    )
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._te_cuda_graph_installed_key = None
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 1
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )

    def reject_before_activation(*, operation: str) -> None:
        events.append(operation)
        raise RuntimeError("storage drift on another rank")

    worker._collectively_validate_te_cuda_graph_storage_before_replay = (
        reject_before_activation
    )
    call_state = SimpleNamespace(normalized_schedule_key=None)
    first = SimpleNamespace(packed_seq_params=object())

    with pytest.raises(RuntimeError, match="another rank"):
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=1,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=True,
        )

    assert events == ["pre-activation storage validation"]


def test_exactly_three_successful_updates_precede_first_capture() -> None:
    worker_type = _extract_worker_methods(
        {
            "_ensure_te_cuda_graph_schedule",
            "_record_te_cuda_graph_optimizer_step",
        }
    )
    worker = worker_type()
    worker._te_cuda_graph_lifecycle = TECudaGraphLifecycle(
        capacity=2,
        warmup_steps=3,
    )
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 1
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    worker._collectively_validate_te_cuda_graph_storage_before_replay = (
        lambda *, operation: None
    )
    worker._bind_te_cuda_graph_storage_after_capture = lambda: None
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._global_te_cuda_graph_optimizer_success = lambda successful: successful
    manager = SimpleNamespace(active_bank=None, uninstall=lambda: None)
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = None
    worker._te_cuda_graph_runtime_schedule_count = 1
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    captures: list[Any] = []

    class Bank:
        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            manager.active_bank = None

    worker._capture_te_cuda_graph_bank = lambda key, sample: (
        captures.append(sample) or Bank()
    )
    first = SimpleNamespace(packed_seq_params=object())

    def ensure_once() -> SimpleNamespace:
        state = SimpleNamespace(
            capture_count=0,
            replay_count=0,
            cache_hit_count=0,
            cache_miss_count=0,
            eviction_count=0,
            normalized_schedule_key=None,
        )
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=9,
            first_microbatch=first,
            call_state=state,
            ensure_active=True,
        )
        return state

    state = ensure_once()
    assert state.capture_count == 0
    assert state.cache_miss_count == 1
    for _ in range(2):
        worker._record_te_cuda_graph_optimizer_step(True)
        state = ensure_once()
        assert state.capture_count == 0
        assert state.cache_miss_count == 1
    worker._record_te_cuda_graph_optimizer_step(False)
    state = ensure_once()
    assert state.capture_count == 0
    assert state.cache_miss_count == 1
    worker._record_te_cuda_graph_optimizer_step(True)
    state = ensure_once()
    assert state.capture_count == 1
    assert state.cache_miss_count == 1
    assert captures == [first.packed_seq_params]


def test_split_schedule_pins_first_key_without_second_transition() -> None:
    worker_type = _extract_worker_methods({"_ensure_te_cuda_graph_schedule"})
    worker = worker_type()
    transitions: list[int] = []

    class Lifecycle:
        def ensure_active(self, key: TECudaGraphScheduleKey, capture: Any) -> Any:
            transitions.append(key.num_microbatches)
            bank = capture()
            bank.activate()
            return SimpleNamespace(status="captured", evicted_key=None)

    manager = SimpleNamespace(active_bank=None)

    class Bank:
        def activate(self) -> None:
            manager.active_bank = self

        def reset(self) -> None:
            manager.active_bank = None

    worker._te_cuda_graph_lifecycle = Lifecycle()
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = None
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(overlap_moe_expert_parallel_comm=False)
    )
    worker._te_cuda_graph_pipeline_parallel_size = lambda: 2
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name, group=None: value
    )
    worker._collectively_raise_te_cuda_graph_failure = lambda error, *, operation: None
    worker._collectively_validate_te_cuda_graph_storage_before_replay = (
        lambda *, operation: None
    )
    worker._bind_te_cuda_graph_storage_after_capture = lambda: None
    worker._assert_te_cuda_graph_model_drained = lambda: True
    worker._capture_te_cuda_graph_bank = lambda key, sample: Bank()
    worker._install_te_cuda_graph_manual_hooks = lambda: None
    first = SimpleNamespace(packed_seq_params=object())
    call_state = SimpleNamespace(
        capture_count=0,
        replay_count=0,
        cache_hit_count=0,
        cache_miss_count=0,
        eviction_count=0,
        normalized_schedule_key=None,
    )

    key = worker._ensure_te_cuda_graph_schedule(
        num_microbatches=5,
        first_microbatch=first,
        call_state=call_state,
        ensure_active=True,
    )
    assert key == TECudaGraphScheduleKey(5)
    assert (
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=5,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=False,
        )
        == key
    )
    assert transitions == [5]
    with pytest.raises(RuntimeError, match="pinned"):
        worker._ensure_te_cuda_graph_schedule(
            num_microbatches=3,
            first_microbatch=first,
            call_state=call_state,
            ensure_active=False,
        )


def test_capture_helper_receives_exact_first_metadata_identity() -> None:
    allocator_values = iter((100, 200, 140, 275))
    log_calls: list[tuple[Any, ...]] = []
    worker_type = _extract_worker_methods(
        {"_capture_te_cuda_graph_bank"},
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    memory_allocated=lambda: next(allocator_values),
                    memory_reserved=lambda: next(allocator_values),
                )
            ),
            "log": SimpleNamespace(
                info=lambda *args: log_calls.append(args),
                exception=lambda *_args, **_kwargs: None,
            ),
        },
    )
    worker = worker_type()
    sample = object()
    manager = object()
    bank = SimpleNamespace(reset=lambda: None)
    helper = SimpleNamespace(
        create_cuda_graph_bank=lambda actual_manager, *, num_microbatches: (
            bank
            if actual_manager is manager and num_microbatches == 4
            else (_ for _ in ()).throw(AssertionError("capture inputs changed"))
        )
    )
    samples: list[Any] = []
    worker._build_te_cuda_graph_helper = lambda actual_sample: (
        samples.append(actual_sample) or helper
    )
    worker._te_cuda_graph_bank_manager = manager
    worker.should_disable_forward_pre_hook = False
    worker._model_parallel_te_cuda_graphs_created = lambda actual_helper: (
        actual_helper is helper
    )

    assert worker._capture_te_cuda_graph_bank(TECudaGraphScheduleKey(4), sample) is bank
    assert samples == [sample]
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is sample
    assert log_calls == [
        (
            "TE CUDA Graph bank captured: schedule_key=%d "
            "allocated_before_bytes=%d allocated_after_bytes=%d "
            "allocated_delta_bytes=%d reserved_before_bytes=%d "
            "reserved_after_bytes=%d reserved_delta_bytes=%d.",
            4,
            100,
            140,
            40,
            200,
            275,
            75,
        )
    ]


def test_capture_releases_registered_bank_when_hook_restore_fails() -> None:
    worker_type = _extract_worker_methods(
        {"_capture_te_cuda_graph_bank"},
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    memory_allocated=lambda: 100,
                    memory_reserved=lambda: 200,
                )
            ),
            "log": SimpleNamespace(
                info=lambda *_args, **_kwargs: None,
                exception=lambda *_args, **_kwargs: None,
            ),
        },
    )
    worker = worker_type()
    events: list[str] = []
    bank = SimpleNamespace(reset=lambda: events.append("bank.reset"))
    helper = SimpleNamespace(
        create_cuda_graph_bank=lambda manager, *, num_microbatches: (
            events.append("capture") or bank
        )
    )
    worker._build_te_cuda_graph_helper = lambda sample: helper
    worker._te_cuda_graph_bank_manager = object()
    worker.should_disable_forward_pre_hook = True
    worker._forward_pre_hook_enabled = lambda: True
    worker.disable_forward_pre_hook = lambda *, param_sync: events.append("disable")

    def fail_enable() -> None:
        events.append("enable")
        raise RuntimeError("hook restore failed")

    worker.enable_forward_pre_hook = fail_enable

    with pytest.raises(RuntimeError, match="hook restore failed"):
        worker._capture_te_cuda_graph_bank(TECudaGraphScheduleKey(2), object())

    assert events == ["disable", "capture", "enable", "bank.reset"]


def test_iterator_modes_eager_isolation_and_relocation_order_are_explicit() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    def iterator_flags(method_name: str) -> list[str]:
        return [
            ast.unparse(keyword.value)
            for call in ast.walk(methods[method_name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "get_microbatch_iterator"
            for keyword in call.keywords
            if keyword.arg == "for_cuda_graph_training"
        ]

    assert iterator_flags("train") == ["te_cuda_graph_call_state is not None"]
    assert iterator_flags("_train_microbatch_body_impl") == [
        "state['te_cuda_graph_call_state'] is not None"
    ]
    assert iterator_flags("get_logprobs") == ["False"]
    assert iterator_flags("get_topk_logits") == ["False"]

    for method_name in ("get_logprobs", "use_reference_model", "get_topk_logits"):
        assert any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_deactivate_te_cuda_graphs_for_eager_path"
            for call in ast.walk(methods[method_name])
        )


def test_metrics_validate_and_reduce_across_tp_cp_then_pp_for_cp_replicas() -> None:
    class FakeTensor:
        def __init__(self, value: Any) -> None:
            self.values = list(value) if isinstance(value, list) else [value]

        def clone(self) -> "FakeTensor":
            return FakeTensor(list(self.values))

        def item(self) -> Any:
            assert len(self.values) == 1
            return self.values[0]

        def zero_(self) -> "FakeTensor":
            self.values = [0 for _ in self.values]
            return self

        def __getitem__(self, index: int) -> "FakeTensor":
            return FakeTensor(self.values[index])

    reduce_op = SimpleNamespace(MIN="min", MAX="max", SUM="sum")

    collective_calls: list[tuple[Any, Any]] = []

    def all_reduce(tensor: FakeTensor, *, op: Any, group: Any = None) -> None:
        collective_calls.append((op, group))
        assert group in {"tp_cp", "pp"}
        if op != reduce_op.SUM:
            return
        if len(tensor.values) == 2:
            tensor.values = [5, 8] if group == "tp_cp" else [13, 21]
        elif len(tensor.values) == 3 and group == "pp":
            tensor.values = [23, 27, 32]

    fake_torch = SimpleNamespace(
        int64="int64",
        tensor=lambda value, *, dtype, device: FakeTensor(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    fake_parallel_state = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 1,
    )
    worker_type = _extract_worker_methods(
        {
            "_collectively_validate_te_cuda_graph_integer",
            "_finalize_te_cuda_graph_call",
            "_te_cuda_graph_token_capacity_per_microbatch",
        },
        {
            "torch": fake_torch,
            "parallel_state": fake_parallel_state,
            "get_pg_collection": lambda model: SimpleNamespace(
                tp_cp="tp_cp", pp="pp", mp="mp"
            ),
            "CudaGraphStepMetrics": __import__(
                "nemo_rl.models.megatron.cuda_graph_lifecycle",
                fromlist=["CudaGraphStepMetrics"],
            ).CudaGraphStepMetrics,
            "asdict": __import__("dataclasses").asdict,
            "cast": lambda _type, value: value,
        },
    )
    worker = worker_type()
    worker.model = object()
    worker.cfg = {"sequence_packing": {"train_mb_tokens": 32}}
    worker.megatron_cfg = SimpleNamespace(
        model=SimpleNamespace(thd_max_packed_sequences=9)
    )
    worker._te_cuda_graph_device = lambda: "cpu"
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        execution_counter_delta=lambda start: SimpleNamespace(
            graph_calls=5,
            eligible_calls=8,
        )
    )
    call_state = SimpleNamespace(
        execution_snapshot=object(),
        capture_count=1,
        replay_count=4,
        cache_hit_count=2,
        cache_miss_count=3,
        eviction_count=1,
        logical_tokens=23,
        padded_tokens=27,
        capacity_tokens=32,
        normalized_schedule_key=5,
    )

    metrics, contract = worker._finalize_te_cuda_graph_call(call_state)

    assert metrics == {
        "capture_count": 1,
        "replay_count": 4,
        "cache_hit_count": 2,
        "cache_miss_count": 3,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 13,
        "eligible_calls": 21,
        "logical_tokens": 23,
        "padded_tokens": 27,
        "capacity_tokens": 32,
    }
    assert contract == {
        "normalized_schedule_key": 5,
        "token_capacity_per_microbatch": 32,
        "thd_max_packed_sequences": 9,
    }
    validation_groups = [
        group for op, group in collective_calls if op in {reduce_op.MIN, reduce_op.MAX}
    ]
    sum_groups = [group for op, group in collective_calls if op == reduce_op.SUM]
    assert validation_groups == ["tp_cp", "tp_cp", "pp", "pp"] * 12
    assert sum_groups == ["tp_cp", "pp", "tp_cp", "pp"]
    assert all(group != "mp" for _, group in collective_calls)


def test_metric_mismatch_reaches_pp_collectives_before_raise() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def clone(self) -> "Scalar":
            return Scalar(self.value)

        def item(self) -> int:
            return self.value

    reduce_op = SimpleNamespace(MIN="min", MAX="max", SUM="sum")
    collective_calls: list[tuple[Any, Any]] = []

    def all_reduce(tensor: Scalar, *, op: Any, group: Any = None) -> None:
        collective_calls.append((op, group))
        if group == "tp_cp" and op == reduce_op.MIN:
            tensor.value = 1
        elif group == "tp_cp" and op == reduce_op.MAX:
            tensor.value = 2

    fake_torch = SimpleNamespace(
        int64="int64",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {
            "_collectively_validate_te_cuda_graph_integer",
            "_finalize_te_cuda_graph_call",
        },
        {
            "torch": fake_torch,
            "get_pg_collection": lambda model: SimpleNamespace(tp_cp="tp_cp", pp="pp"),
        },
    )
    worker = worker_type()
    worker.model = object()
    worker._te_cuda_graph_device = lambda: "cpu"
    worker._te_cuda_graph_bank_manager = object()
    call_state = SimpleNamespace(
        capture_count=1,
        replay_count=0,
        cache_hit_count=0,
        cache_miss_count=1,
        eviction_count=0,
        normalized_schedule_key=1,
    )

    with pytest.raises(RuntimeError, match="capture_count differs across ranks"):
        worker._finalize_te_cuda_graph_call(call_state)

    assert collective_calls == [
        (reduce_op.MIN, "tp_cp"),
        (reduce_op.MAX, "tp_cp"),
        (reduce_op.MIN, "pp"),
        (reduce_op.MAX, "pp"),
    ]


def test_global_optimizer_consensus_uses_world_min() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    calls: list[tuple[Any, Any]] = []
    reduce_op = SimpleNamespace(MIN=object())

    def all_reduce(tensor: Scalar, *, op: Any) -> None:
        calls.append((op, None))
        tensor.value = 0

    fake_torch = SimpleNamespace(
        int32="int32",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {"_global_te_cuda_graph_optimizer_success"},
        {"torch": fake_torch},
    )
    worker = worker_type()
    worker._te_cuda_graph_device = lambda: "cuda"

    assert not worker._global_te_cuda_graph_optimizer_success(True)
    assert calls == [(reduce_op.MIN, None)]


def test_remote_graph_failure_raises_and_never_falls_back() -> None:
    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    reduce_op = SimpleNamespace(MAX=object())

    def all_reduce(tensor: Scalar, *, op: Any) -> None:
        assert op is reduce_op.MAX
        tensor.value = 1

    fake_torch = SimpleNamespace(
        int32="int32",
        tensor=lambda value, *, dtype, device: Scalar(value),
        distributed=SimpleNamespace(ReduceOp=reduce_op, all_reduce=all_reduce),
    )
    worker_type = _extract_worker_methods(
        {"_collectively_raise_te_cuda_graph_failure"},
        {
            "torch": fake_torch,
            "clear_router_replay": lambda model: cleanup.append("route.clear"),
        },
    )
    worker = worker_type()
    worker._te_cuda_graph_device = lambda: "cuda"
    cleanup: list[str] = []
    worker.model = object()
    worker._active_router_route_generation = 9
    worker._reset_te_cuda_graph_banks_after_failure = lambda: cleanup.append("reset")

    with pytest.raises(RuntimeError, match="another rank"):
        worker._collectively_raise_te_cuda_graph_failure(
            None,
            operation="replay",
        )
    assert cleanup == ["route.clear", "reset"]
    assert worker._active_router_route_generation is None


def test_split_success_and_abort_clear_route_lifecycle_state() -> None:
    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    methods = {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }

    for method_name in ("_finish_train_step_body", "abort_train_step"):
        calls = [node for node in ast.walk(methods[method_name]) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Name)
            and call.func.id == "clear_router_replay"
            for call in calls
        )
        assignments = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Assign)
        ]
        assert any(
            any(
                isinstance(target, ast.Attribute)
                and target.attr == "_active_router_route_generation"
                for target in assignment.targets
            )
            and isinstance(assignment.value, ast.Constant)
            and assignment.value.value is None
            for assignment in assignments
        )


def test_split_abort_restores_idle_and_runs_deferred_failure_cleanup() -> None:
    worker_type = _extract_worker_methods({"abort_train_step"})
    worker = worker_type()
    events: list[str] = []
    state = {
        "te_cuda_graph_key": object(),
        "te_cuda_graph_call_state": object(),
    }
    worker._train_step_state = state
    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_AFTER_FIRST
    worker._te_cuda_graph_reset_required = True
    worker._restore_saved_grad_sync_func = lambda actual: events.append("restore")
    worker.model = SimpleNamespace(zero_grad_buffer=lambda: events.append("model.zero"))
    worker.optimizer = SimpleNamespace(zero_grad=lambda: events.append("opt.zero"))
    worker._reset_te_cuda_graph_banks_after_failure = lambda: events.append("reset")

    worker.abort_train_step()

    assert events == ["restore", "model.zero", "opt.zero", "reset"]
    assert worker._train_step_state is None
    assert worker._te_cuda_graph_phase is _Phase.IDLE
    assert state["te_cuda_graph_key"] is None
    assert state["te_cuda_graph_call_state"] is None


def test_empty_graph_split_finish_fails_before_optimizer_or_scheduler_mutation() -> (
    None
):
    worker_type = _extract_worker_methods({"_validate_te_cuda_graph_split_finish"})
    worker = worker_type()
    call_state = SimpleNamespace(
        normalized_schedule_key=None,
        capacity_tokens=0,
    )
    state = {
        "te_cuda_graph_key": None,
        "te_cuda_graph_call_state": call_state,
        "total_num_microbatches": 0,
    }
    collective_calls: list[tuple[Any, str]] = []

    def raise_collectively(error: Any, *, operation: str) -> None:
        collective_calls.append((error, operation))
        if error is not None:
            raise error

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively

    with pytest.raises(RuntimeError, match="no processed microbatch"):
        worker._validate_te_cuda_graph_split_finish(state)
    assert len(collective_calls) == 1
    assert collective_calls[0][1] == "split finish preflight"

    tree = ast.parse(_WORKER_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    finish = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_finish_train_step_body"
    )
    calls = [node for node in ast.walk(finish) if isinstance(node, ast.Call)]
    guard_line = min(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_validate_te_cuda_graph_split_finish"
    )
    mutation_lines = [
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr in {"scale_gradients", "step"}
    ]
    assert mutation_lines
    assert guard_line < min(mutation_lines)


def test_empty_graph_split_can_abort_without_finish_validation() -> None:
    worker_type = _extract_worker_methods({"abort_train_step"})
    worker = worker_type()
    events: list[str] = []
    state = {
        "te_cuda_graph_key": None,
        "te_cuda_graph_call_state": object(),
    }
    worker._train_step_state = state
    worker._te_cuda_graph_phase = _Phase.SPLIT_OPEN_BEFORE_FIRST
    worker._te_cuda_graph_reset_required = False
    worker._restore_saved_grad_sync_func = lambda actual: events.append("restore")
    worker.model = SimpleNamespace(zero_grad_buffer=lambda: events.append("model.zero"))
    worker.optimizer = SimpleNamespace(zero_grad=lambda: events.append("opt.zero"))

    worker.abort_train_step()

    assert events == ["restore", "model.zero", "opt.zero"]
    assert worker._train_step_state is None
    assert worker._te_cuda_graph_phase is _Phase.IDLE


@pytest.mark.parametrize(
    ("successful", "global_success", "expected"),
    [(True, True, [True]), (True, False, [False]), (False, False, [False])],
)
def test_warmup_records_only_global_optimizer_consensus(
    successful: bool, global_success: bool, expected: list[bool]
) -> None:
    worker_type = _extract_worker_methods({"_record_te_cuda_graph_optimizer_step"})
    worker = worker_type()
    recorded: list[bool] = []
    worker._te_cuda_graph_lifecycle = SimpleNamespace(
        record_optimizer_step=lambda *, successful: recorded.append(successful)
    )
    worker._global_te_cuda_graph_optimizer_success = lambda _: global_success

    worker._record_te_cuda_graph_optimizer_step(successful)

    assert recorded == expected


def test_storage_reset_preserves_lifecycle_and_manager_but_discards_capture_refs() -> (
    None
):
    worker_type = _extract_worker_methods(
        {"_reset_te_cuda_graph_banks_for_storage_relocation"}
    )
    worker = worker_type()
    events: list[str] = []
    lifecycle = SimpleNamespace(reset_banks=lambda: events.append("reset"))
    manager = object()
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_lifecycle = lifecycle
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = object()
    worker._te_cuda_graph_capture_helper = object()
    worker._te_cuda_graph_capture_sample_packed_seq_params = object()

    worker._reset_te_cuda_graph_banks_for_storage_relocation()

    assert events == ["reset"]
    assert worker._te_cuda_graph_lifecycle is lifecycle
    assert worker._te_cuda_graph_bank_manager is manager
    assert worker._te_cuda_graph_installed_key is None
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None


def test_eval_extra_state_restore_resets_graph_banks_before_storage_load() -> None:
    worker_type = _extract_worker_methods({"_restore_model_extra_state_dict"})
    worker = worker_type()
    events: list[str] = []
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: events.append(
        "reset"
    )
    worker.model = SimpleNamespace(
        load_state_dict=lambda state, *, strict: events.append(("load", state, strict))
    )
    extra_state = {"layer._extra_state": object()}

    worker._restore_model_extra_state_dict({})
    assert events == []

    worker._restore_model_extra_state_dict(extra_state)
    assert events == ["reset", ("load", extra_state, False)]


def test_prepare_for_lp_inference_preserves_graph_owned_storage() -> None:
    class FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

    class FakeWakeupTensor:
        def cuda(self) -> None:
            events.append("allocator_wakeup")

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        randn=lambda _size: FakeWakeupTensor(),
    )
    fake_gc = SimpleNamespace(collect=lambda: events.append("gc"))
    worker_type = _extract_worker_methods(
        {"prepare_for_lp_inference"},
        {"torch": fake_torch, "gc": fake_gc},
    )
    worker = worker_type()
    events: list[Any] = []
    worker._te_cuda_graph_lifecycle = object()
    worker.model = SimpleNamespace(eval=lambda: events.append("eval"))
    worker.move_model = lambda model, device, **kwargs: (
        events.append(("move_model", device, kwargs)) or model
    )
    worker.optimizer = None
    worker.optimizer_cpu_offload = False
    worker.offload_optimizer_for_logprob = False

    worker.prepare_for_lp_inference()

    assert events == ["eval"]


def test_finish_inference_preserves_graph_owned_storage() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
    )
    fake_gc = SimpleNamespace(collect=lambda: events.append("gc"))
    worker_type = _extract_worker_methods(
        {"finish_inference"},
        {"torch": fake_torch, "gc": fake_gc},
    )
    worker = worker_type()
    events: list[Any] = []
    worker._te_cuda_graph_lifecycle = object()
    worker.model = SimpleNamespace(eval=lambda: events.append("eval"))
    worker.move_model = lambda model, device, **kwargs: (
        events.append(("move_model", device, kwargs)) or model
    )

    worker.finish_inference()

    assert events == ["eval"]


@pytest.mark.parametrize("method_name", ("offload_before_refit", "offload_after_refit"))
def test_refit_offload_is_rejected_before_persistent_graph_storage_moves(
    method_name: str,
) -> None:
    worker_type = _extract_worker_methods({method_name})
    worker = worker_type()
    worker._te_cuda_graph_lifecycle = object()
    events: list[str] = []
    worker.finalize_async_save = lambda: events.append("finalize")
    worker.move_model = lambda *_args, **_kwargs: events.append("move_model")

    with pytest.raises(RuntimeError, match="non-offloading refit"):
        getattr(worker, method_name)()

    assert events == []


def test_optimizer_state_move_preserves_training_graph_bank() -> None:
    class FakeTensor:
        is_cuda = True

    fake_tensor = FakeTensor()
    fake_torch = SimpleNamespace(is_tensor=lambda value: value is fake_tensor)
    fake_chained_optimizer = type("FakeChainedOptimizer", (), {})
    worker_type = _extract_worker_methods(
        {"move_optimizer"},
        {
            "torch": fake_torch,
            "ChainedOptimizer": fake_chained_optimizer,
        },
    )
    worker = worker_type()
    events: list[str] = []
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: events.append(
        "reset"
    )
    worker.optimizer = SimpleNamespace(_get_state=lambda: {0: {"exp_avg": fake_tensor}})

    worker.move_optimizer("cuda")

    assert events == []


def test_unimplemented_checkpoint_load_preserves_training_graph_bank() -> None:
    worker_type = _extract_worker_methods({"load_checkpoint"})
    worker = worker_type()
    resets: list[str] = []
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: resets.append(
        "reset"
    )

    with pytest.raises(NotImplementedError, match="outside of the init function"):
        worker.load_checkpoint("/tmp/checkpoint")

    assert resets == []


def test_prepare_for_training_reuses_resident_graph_storage() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
    )
    worker_type = _extract_worker_methods(
        {"prepare_for_training"},
        {"torch": fake_torch},
    )
    worker = worker_type()
    events: list[Any] = []
    worker._te_cuda_graph_lifecycle = object()
    worker.model = SimpleNamespace(train=lambda: events.append("train"))
    worker.move_model = lambda model, device, **kwargs: (
        events.append(("move_model", device, kwargs)) or model
    )
    worker.move_optimizer = lambda device: events.append(("move_optimizer", device))
    worker.optimizer = object()
    worker.optimizer_cpu_offload = False
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 2}}

    worker.prepare_for_training()

    assert events == ["train"]


def test_unexpected_storage_drift_invalidates_before_replay() -> None:
    worker_type = _extract_worker_methods(
        {"_validate_te_cuda_graph_storage_before_replay"},
        {
            "StorageChange": StorageChange,
            "classify_storage_change": lambda before, after: (
                StorageChange.NONE if before == after else StorageChange.MODEL
            ),
        },
    )
    worker = worker_type()
    expected = GraphStorageFingerprint(model=(), grads=())
    changed = GraphStorageFingerprint(
        model=(),
        grads=(),
    )
    worker._te_cuda_graph_storage_fingerprint = expected
    worker._capture_te_cuda_graph_storage = lambda: expected
    resets: list[str] = []
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: resets.append(
        "reset"
    )

    worker._validate_te_cuda_graph_storage_before_replay()
    assert resets == []

    worker._capture_te_cuda_graph_storage = lambda: changed
    worker._te_cuda_graph_storage_fingerprint = GraphStorageFingerprint(
        model=(),
        grads=(object(),),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="storage changed"):
        worker._validate_te_cuda_graph_storage_before_replay()
    assert resets == []


def test_storage_validation_converts_remote_drift_to_collective_failure() -> None:
    worker_type = _extract_worker_methods(
        {"_collectively_validate_te_cuda_graph_storage_before_replay"}
    )
    worker = worker_type()
    events: list[tuple[Any, str]] = []
    worker._validate_te_cuda_graph_storage_before_replay = lambda: None

    def raise_collectively(error: Any, *, operation: str) -> None:
        events.append((error, operation))
        raise RuntimeError("failed on another rank")

    worker._collectively_raise_te_cuda_graph_failure = raise_collectively

    with pytest.raises(RuntimeError, match="another rank"):
        worker._collectively_validate_te_cuda_graph_storage_before_replay(
            operation="pre-training storage validation"
        )

    assert events == [(None, "pre-training storage validation")]


def test_storage_validation_routes_local_drift_through_collective_cleanup() -> None:
    worker_type = _extract_worker_methods(
        {"_collectively_validate_te_cuda_graph_storage_before_replay"}
    )
    worker = worker_type()
    local_error = RuntimeError("local storage drift")
    events: list[tuple[Any, str]] = []

    def validate() -> None:
        raise local_error

    worker._validate_te_cuda_graph_storage_before_replay = validate
    worker._collectively_raise_te_cuda_graph_failure = (
        lambda error, *, operation: events.append((error, operation))
    )

    worker._collectively_validate_te_cuda_graph_storage_before_replay(
        operation="reference restore storage validation"
    )

    assert events == [(local_error, "reference restore storage validation")]


def test_same_shape_extra_state_uses_module_setter_not_tensor_copy() -> None:
    class FakeTensor:
        shape = (4,)

        def numel(self) -> int:
            return 4

        def copy_(self, _source: Any) -> None:
            events.append("copy")

    target_module = SimpleNamespace(
        set_extra_state=lambda value: events.append(("set_extra_state", value))
    )
    model = SimpleNamespace(
        state_dict=lambda: {"layer._extra_state": FakeTensor()},
        get_submodule=lambda path: target_module,
    )
    worker_type = _extract_worker_methods(
        {"_apply_state_dict_to_model"},
        {"torch": SimpleNamespace(Tensor=FakeTensor)},
    )
    worker = worker_type()
    worker.model = model
    events: list[Any] = []
    source = FakeTensor()

    worker._apply_state_dict_to_model(
        {"layer._extra_state": source},
        raise_if_key_missing=True,
    )

    assert events == [("set_extra_state", source)]


def test_empty_te_extra_state_is_a_storage_preserving_noop() -> None:
    class EmptyTensor:
        shape = (0,)

        def numel(self) -> int:
            return 0

    destination = EmptyTensor()
    source = EmptyTensor()
    model = SimpleNamespace(
        state_dict=lambda: {"layer._extra_state": destination},
        get_submodule=lambda _path: SimpleNamespace(
            set_extra_state=lambda _value: events.append("set_extra_state")
        ),
    )
    fake_torch = SimpleNamespace(Tensor=EmptyTensor)
    worker_type = _extract_worker_methods(
        {
            "_apply_state_dict_to_model",
            "_reference_state_dict_preserves_storage",
        },
        {"torch": fake_torch},
    )
    worker = worker_type()
    worker.model = model
    worker.fp8_cfg = {"enabled": False}
    worker.megatron_cfg = SimpleNamespace(model=SimpleNamespace(fp8=None))
    events: list[str] = []

    assert worker._reference_state_dict_preserves_storage(
        {"layer._extra_state": source}
    )
    worker._apply_state_dict_to_model(
        {"layer._extra_state": source},
        raise_if_key_missing=True,
    )

    assert events == []


def test_none_mcore_extra_state_is_a_storage_preserving_noop() -> None:
    class FakeTensor:
        pass

    model = SimpleNamespace(
        state_dict=lambda: {"layer._extra_state": None},
        get_submodule=lambda _path: SimpleNamespace(
            set_extra_state=lambda _value: events.append("set_extra_state")
        ),
    )
    fake_torch = SimpleNamespace(Tensor=FakeTensor)
    worker_type = _extract_worker_methods(
        {
            "_apply_state_dict_to_model",
            "_reference_state_dict_preserves_storage",
        },
        {"torch": fake_torch},
    )
    worker = worker_type()
    worker.model = model
    worker.fp8_cfg = {"enabled": False}
    worker.megatron_cfg = SimpleNamespace(model=SimpleNamespace(fp8=None))
    events: list[str] = []

    assert worker._reference_state_dict_preserves_storage(
        {"layer._extra_state": None}
    )
    worker._apply_state_dict_to_model(
        {"layer._extra_state": None},
        raise_if_key_missing=True,
    )

    assert events == []


@pytest.mark.parametrize(
    ("raw_fp8", "effective_fp8", "state_key", "expected"),
    (
        (False, None, "weight", True),
        (True, None, "weight", False),
        (False, "e4m3", "weight", False),
        (False, None, "layer._extra_state", False),
    ),
)
def test_reference_storage_preservation_is_bf16_plain_tensor_only(
    raw_fp8: bool,
    effective_fp8: Any,
    state_key: str,
    expected: bool,
) -> None:
    class FakeTensor:
        shape = (2, 4)

        def numel(self) -> int:
            return 8

    tensor = FakeTensor()
    worker_type = _extract_worker_methods(
        {"_reference_state_dict_preserves_storage"},
        {"torch": SimpleNamespace(Tensor=FakeTensor)},
    )
    worker = worker_type()
    worker.fp8_cfg = {"enabled": raw_fp8}
    worker.megatron_cfg = SimpleNamespace(model=SimpleNamespace(fp8=effective_fp8))
    worker.model = SimpleNamespace(state_dict=lambda: {state_key: tensor})

    assert worker._reference_state_dict_preserves_storage({state_key: tensor}) is expected


def test_bf16_reference_swap_preserves_training_bank() -> None:
    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def to(self, **_kwargs: Any) -> FakeTensor:
            return self

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        no_grad=lambda: FakeNoGrad(),
        cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache")),
    )
    worker_type = _extract_worker_methods(
        {"use_reference_model"},
        {
            "torch": fake_torch,
            "gc": SimpleNamespace(collect=lambda: events.append("gc")),
            "TrainingSamplingParams": object,
        },
    )
    worker = worker_type()
    events: list[str] = []
    policy_tensor = FakeTensor()
    reference_state = {"weight": FakeTensor()}
    worker.model = SimpleNamespace(state_dict=lambda: {"weight": policy_tensor})
    worker.reference_state_dict = reference_state
    worker._te_cuda_graph_lifecycle = object()
    worker._deactivate_te_cuda_graphs_for_eager_path = lambda: events.append(
        "deactivate"
    )
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: events.append(
        "reset"
    )
    worker._reference_state_dict_preserves_storage = lambda _state: True
    worker._collectively_raise_te_cuda_graph_failure = (
        lambda error, *, operation: None
    )
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name: value
    )
    worker._apply_state_dict_to_model = (
        lambda state, *, raise_if_key_missing: events.append(
            "apply_reference" if state is reference_state else "restore_policy"
        )
    )
    worker.should_disable_forward_pre_hook = False
    worker.sampling_params = None
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 1}}

    generator = worker.use_reference_model()
    next(generator)
    events.append("body")
    with pytest.raises(StopIteration):
        next(generator)

    assert events == [
        "deactivate",
        "apply_reference",
        "body",
        "restore_policy",
    ]


def test_reference_body_exception_restores_policy_and_bank() -> None:
    class BodyError(RuntimeError):
        pass

    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def to(self, **_kwargs: Any) -> FakeTensor:
            return self

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    fake_torch = SimpleNamespace(Tensor=FakeTensor, no_grad=lambda: FakeNoGrad())
    worker_type = _extract_worker_methods(
        {"use_reference_model"},
        {
            "torch": fake_torch,
            "TrainingSamplingParams": object,
        },
    )
    worker = worker_type()
    events: list[str] = []
    policy_tensor = FakeTensor()
    reference_state = {"weight": FakeTensor()}
    worker.model = SimpleNamespace(state_dict=lambda: {"weight": policy_tensor})
    worker.reference_state_dict = reference_state
    worker._te_cuda_graph_lifecycle = object()
    worker._deactivate_te_cuda_graphs_for_eager_path = lambda: None
    worker._reset_te_cuda_graph_banks_for_storage_relocation = lambda: events.append(
        "reset"
    )
    worker._reference_state_dict_preserves_storage = lambda _state: True
    worker._collectively_raise_te_cuda_graph_failure = (
        lambda error, *, operation: None
    )
    worker._collectively_validate_te_cuda_graph_integer = (
        lambda value, *, name: value
    )
    worker._apply_state_dict_to_model = (
        lambda state, *, raise_if_key_missing: events.append(
            "apply_reference" if state is reference_state else "restore_policy"
        )
    )
    worker.should_disable_forward_pre_hook = False
    worker.sampling_params = None
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 0}}

    generator = worker.use_reference_model()
    next(generator)
    with pytest.raises(BodyError):
        generator.throw(BodyError("reference logprob failed"))

    assert events == ["apply_reference", "restore_policy"]


def test_reference_snapshot_failure_restores_forward_pre_hook() -> None:
    class SnapshotError(RuntimeError):
        pass

    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def to(self, **_kwargs: Any) -> FakeTensor:
            raise SnapshotError("D2H snapshot failed")

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    worker_type = _extract_worker_methods(
        {"use_reference_model"},
        {"torch": SimpleNamespace(Tensor=FakeTensor, no_grad=lambda: FakeNoGrad())},
    )
    worker = worker_type()
    worker.model = SimpleNamespace(state_dict=lambda: {"weight": FakeTensor()})
    worker.reference_state_dict = {"weight": FakeTensor()}
    worker._te_cuda_graph_lifecycle = None
    worker._deactivate_te_cuda_graphs_for_eager_path = lambda: None
    worker.should_disable_forward_pre_hook = True
    events: list[str] = []
    worker.disable_forward_pre_hook = lambda: events.append("disable")
    worker.enable_forward_pre_hook = lambda: events.append("enable")

    generator = worker.use_reference_model()
    with pytest.raises(SnapshotError, match="D2H snapshot failed"):
        next(generator)

    assert events == ["disable", "enable"]


def test_eager_uninstall_requires_idle_and_clears_only_installed_bank() -> None:
    worker_type = _extract_worker_methods({"_deactivate_te_cuda_graphs_for_eager_path"})
    worker = worker_type()
    events: list[str] = []
    worker._te_cuda_graph_phase = _Phase.IDLE
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        active_bank=object(), uninstall=lambda: events.append("uninstall")
    )
    worker._te_cuda_graph_installed_key = object()

    worker._deactivate_te_cuda_graphs_for_eager_path()

    assert events == ["uninstall"]
    assert worker._te_cuda_graph_installed_key is None

    worker._te_cuda_graph_phase = _Phase.GRAPH_SCHEDULE_LIVE
    with pytest.raises(RuntimeError, match="drained"):
        worker._deactivate_te_cuda_graphs_for_eager_path()


def test_shutdown_aborts_closes_lifecycle_and_detaches_manager() -> None:
    events: list[str] = []

    class FakeBase:
        @staticmethod
        def shutdown(_worker: Any) -> bool:
            events.append("base")
            return True

    worker_type = _extract_worker_methods(
        {"shutdown"},
        {"AbstractPolicyWorker": FakeBase},
    )
    worker = worker_type()
    worker._train_step_state = {"open": True}
    worker.abort_train_step = lambda: (
        events.append("abort") or setattr(worker, "_train_step_state", None)
    )
    worker._te_cuda_graph_lifecycle = SimpleNamespace(
        close=lambda: events.append("lifecycle.close")
    )
    worker._te_cuda_graph_bank_manager = SimpleNamespace(
        close=lambda: events.append("manager.close")
    )
    worker._te_cuda_graph_installed_key = object()
    worker._te_cuda_graph_capture_helper = object()
    worker._te_cuda_graph_capture_sample_packed_seq_params = object()

    assert worker.shutdown()
    assert events == ["abort", "lifecycle.close", "manager.close", "base"]
    assert worker._te_cuda_graph_lifecycle is None
    assert worker._te_cuda_graph_bank_manager is None
    assert worker._te_cuda_graph_installed_key is None
    assert worker._te_cuda_graph_capture_helper is None
    assert worker._te_cuda_graph_capture_sample_packed_seq_params is None
