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
import os
import warnings
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LM_POLICY_PATH = _REPO_ROOT / "nemo_rl/models/policy/lm_policy.py"
_TQ_POLICY_PATH = _REPO_ROOT / "nemo_rl/models/policy/tq_policy.py"


def _extract_class_methods(
    source_path: Path,
    class_name: str,
    method_names: set[str],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Compile selected production methods without importing GPU dependencies."""
    tree = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name: copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    }
    missing = method_names - methods.keys()
    assert not missing, f"{class_name} is missing required methods: {sorted(missing)}"
    for method in methods.values():
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *methods.values(),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in method_names}


def _extract_module_functions(
    source_path: Path,
    function_names: set[str],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Compile selected top-level production functions in a light namespace."""
    tree = ast.parse(source_path.read_text())
    functions = {
        node.name: copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    }
    missing = function_names - functions.keys()
    assert not missing, f"Missing required functions: {sorted(missing)}"
    for function in functions.values():
        function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions.values(),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


class _WorkerGroup:
    def __init__(
        self,
        *,
        worker_results: list[dict[str, Any]] | None = None,
        single_data_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.dispatches: list[tuple[str, dict[str, Any]]] = []
        self.worker_results = worker_results or [_worker_result()]
        self.single_data_results = single_data_results or list(self.worker_results)

    def run_all_workers_sharded_data(
        self,
        method_name: str,
        **kwargs: Any,
    ) -> object:
        self.dispatches.append((method_name, kwargs))
        return object()

    def get_all_worker_results(self, _futures: object) -> list[dict[str, Any]]:
        return self.worker_results

    def run_all_workers_single_data(
        self,
        method_name: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.dispatches.append((method_name, kwargs))
        return self.single_data_results


class _ShardingAnnotations:
    def __init__(self, data_parallel_size: int) -> None:
        self.data_parallel_size = data_parallel_size

    def get_axis_size(self, axis: str) -> int:
        assert axis == "data_parallel"
        return self.data_parallel_size


class _RecordingBatch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def shard_by_batch_size(
        self,
        shards: int,
        *,
        batch_size: int | None,
        sequence_packing_args: dict[str, Any] | None = None,
        dynamic_batching_args: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "shards": shards,
                "batch_size": batch_size,
                "sequence_packing_args": sequence_packing_args,
                "dynamic_batching_args": dynamic_batching_args,
            }
        )
        shards_out = [object() for _ in range(shards)]
        if sequence_packing_args is not None or dynamic_batching_args is not None:
            return shards_out, list(range(shards))
        return shards_out


@dataclass
class _Meta:
    partition_id: str = "train"
    task_name: str = "train"
    sample_ids: list[str] | None = None
    fields: list[str] | tuple[str, ...] | None = None
    sequence_lengths: list[int] | None = None
    extra_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sample_ids is None:
            self.sample_ids = ["s0", "s1", "s2", "s3", "s4", "s5"]
        if self.sequence_lengths is None:
            self.sequence_lengths = [7, 6, 5, 4, 3, 2]
        if self.extra_info is None:
            self.extra_info = {}


_META_SHARD_CALLS: list[dict[str, Any]] = []


def _record_shard_meta_for_dp(meta: _Meta, **kwargs: Any) -> tuple[list[_Meta], None]:
    _META_SHARD_CALLS.append({"meta": meta, **kwargs})
    return [meta for _ in range(kwargs["dp_world"])], None


def _fields_with_optional_routed_experts(
    fields: tuple[str, ...],
    *,
    enabled: bool,
) -> list[str]:
    del enabled
    return list(fields)


@dataclass(frozen=True)
class _EffectiveTECudaGraphConfig:
    cuda_graph_impl: str
    thd_max_packed_sequences: int | None
    cuda_graph_max_cached_schedules: int | None
    training_enabled: bool


_CUDA_GRAPH_METRICS = {
    "capture_count": 1,
    "replay_count": 2,
    "cache_hit_count": 3,
    "cache_miss_count": 2,
    "eviction_count": 0,
    "fallback_count": 0,
    "graph_calls": 8,
    "eligible_calls": 10,
    "logical_tokens": 80,
    "padded_tokens": 100,
    "capacity_tokens": 120,
    "coverage": 0.8,
    "capacity_utilization": 2 / 3,
    "padding_utilization": 0.8,
}


def _worker_result(
    *,
    cuda_graph_metrics: dict[str, int | float] | None = None,
    replica_leader: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "global_loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {"loss": [1.0]},
        "is_replica_leader": replica_leader,
    }
    if cuda_graph_metrics is not None:
        result["cuda_graph_metrics"] = cuda_graph_metrics
    return result


def _aggregate_cuda_graph_metrics(
    results: list[dict[str, Any]],
) -> dict[str, int | float] | None:
    present = [
        result["cuda_graph_metrics"]
        for result in results
        if "cuda_graph_metrics" in result
    ]
    if not present:
        return None
    if len(present) != len(results) or any(
        metrics != present[0] for metrics in present
    ):
        raise ValueError("mixed CUDA Graph metrics")
    return dict(present[0])


class _Ray:
    @staticmethod
    def get(value: Any) -> Any:
        return value


class _FakeArray:
    def reshape(self, *_shape: int) -> object:
        return object()


class _FakeNumpy:
    @staticmethod
    def arange(_size: int) -> _FakeArray:
        return _FakeArray()


class _NamedSharding:
    def __init__(self, *, layout: object, names: list[str]) -> None:
        del layout, names

    def get_axis_size(self, axis: str) -> int:
        assert axis == "data_parallel"
        return 1


class _ConstructionCluster:
    _sorted_bundle_indices = None

    def __init__(self, world_size: int = 1) -> None:
        self._world_size = world_size

    def world_size(self) -> int:
        return self._world_size


class _ConstructionWorkerGroup(_WorkerGroup):
    effective_results: list[dict[str, Any]] = []
    instances: list[_ConstructionWorkerGroup] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(single_data_results=list(self.effective_results))
        self.instances.append(self)


class _UnsupportedFlopTracker:
    @classmethod
    def from_config(cls, *_args: Any) -> None:
        raise ValueError("unsupported in isolated policy test")


_METHOD_GLOBALS: dict[str, Any] = {
    "Any": Any,
    "DP_TRAIN_FIELDS": ("input_ids", "input_lengths"),
    "DynamicBatchingConfig": dict,
    "_EffectiveTECudaGraphConfig": _EffectiveTECudaGraphConfig,
    "LP_SEED_FIELDS": ("input_ids", "input_lengths"),
    "SequencePackingConfig": dict,
    "_aggregate_megatron_flops_metrics": lambda *_args: {},
    "aggregate_cuda_graph_metrics": _aggregate_cuda_graph_metrics,
    "defaultdict": defaultdict,
    "cast": cast,
    "fields_with_optional_routed_experts": _fields_with_optional_routed_experts,
    "get_theoretical_tflops": lambda *_args: 0.0,
    "nullcontext": nullcontext,
    "replace": replace,
    "ray": _Ray,
    "shard_meta_for_dp": _record_shard_meta_for_dp,
    "warnings": warnings,
}


@lru_cache(maxsize=1)
def _effective_config_api() -> tuple[Any, type[Any]]:
    method_globals = dict(_METHOD_GLOBALS)
    method_globals.update(
        {
            "FLOPTracker": _UnsupportedFlopTracker,
            "NamedSharding": _NamedSharding,
            "RayQueue": object,
            "RayWorkerBuilder": lambda *_args, **_kwargs: object(),
            "RayWorkerGroup": _ConstructionWorkerGroup,
            "get_default_hf_config": lambda _model_name: object(),
            "np": _FakeNumpy,
            "os": os,
            "resolve_policy_worker_cls": lambda default, _config: default,
        }
    )
    resolver = _extract_module_functions(
        _LM_POLICY_PATH,
        {"_resolve_effective_te_cuda_graph_config"},
        method_globals,
    )["_resolve_effective_te_cuda_graph_config"]
    policy_methods = _extract_class_methods(
        _LM_POLICY_PATH,
        "Policy",
        {"__init__", "_cache_effective_te_cuda_graph_config"},
        method_globals,
    )
    return resolver, type("ConstructedPolicyHarness", (), policy_methods)


@lru_cache(maxsize=1)
def _harness_types() -> tuple[type[Any], type[Any]]:
    method_globals = dict(_METHOD_GLOBALS)
    policy_methods = _extract_class_methods(
        _LM_POLICY_PATH,
        "Policy",
        {
            "_sequence_packing_args_for_call",
            "_shard_for_logprob",
            "_shard_for_train",
            "train",
        },
        method_globals,
    )
    policy_type = type("PolicyHarness", (), policy_methods)
    _extract_module_functions(
        _TQ_POLICY_PATH,
        {"_aggregate_train_results"},
        method_globals,
    )
    tq_methods = _extract_class_methods(
        _TQ_POLICY_PATH,
        "TQPolicy",
        {
            "_logprob_dispatch",
            "_packing_args",
            "finish_train_step",
            "train_from_meta",
            "train_microbatches_from_meta",
        },
        method_globals,
    )
    tq_policy_type = type("TQPolicyHarness", (policy_type,), tq_methods)
    return policy_type, tq_policy_type


def _policy_config(
    *,
    cuda_graph_impl: str | None = "transformer_engine",
    capacity: object = 5,
) -> dict[str, Any]:
    megatron_cfg: dict[str, Any] = {
        "enabled": True,
        "thd_max_packed_sequences": capacity,
    }
    if cuda_graph_impl is not None:
        megatron_cfg["cuda_graph_impl"] = cuda_graph_impl
    return {
        "train_global_batch_size": 6,
        "train_micro_batch_size": 2,
        "megatron_cfg": megatron_cfg,
        "sequence_packing": {
            "enabled": True,
            "algorithm": "modified_first_fit_decreasing",
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 64,
        },
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 64,
            "sequence_length_round": 8,
        },
    }


def _effective_config(
    *,
    cuda_graph_impl: str = "transformer_engine",
    capacity: int | None = 5,
    cache_capacity: int | None = 3,
    training_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "cuda_graph_impl": cuda_graph_impl,
        "thd_max_packed_sequences": capacity,
        "cuda_graph_max_cached_schedules": cache_capacity,
        "training_enabled": training_enabled,
    }


def _resolved_effective_config(
    *,
    cuda_graph_impl: str = "transformer_engine",
    capacity: int | None = 5,
    cache_capacity: int | None = 3,
    training_enabled: bool = True,
) -> _EffectiveTECudaGraphConfig:
    return _EffectiveTECudaGraphConfig(
        cuda_graph_impl=cuda_graph_impl,
        thd_max_packed_sequences=capacity,
        cuda_graph_max_cached_schedules=cache_capacity,
        training_enabled=training_enabled,
    )


def test_effective_config_cache_has_exact_internal_type_and_init_annotation() -> None:
    tree = ast.parse(_LM_POLICY_PATH.read_text())
    config_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_EffectiveTECudaGraphConfig"
    )
    assert [
        (node.target.id, ast.unparse(node.annotation))
        for node in config_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ] == [
        ("cuda_graph_impl", "str"),
        ("thd_max_packed_sequences", "int | None"),
        ("cuda_graph_max_cached_schedules", "int | None"),
        ("training_enabled", "bool"),
    ]
    dataclass_decorator = next(
        decorator
        for decorator in config_class.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "dataclass"
    )
    assert any(
        keyword.arg == "frozen"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in dataclass_decorator.keywords
    )
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_effective_te_cuda_graph_config"
    )
    assert resolver.returns is not None
    assert ast.unparse(resolver.returns) == "_EffectiveTECudaGraphConfig"

    policy_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Policy"
    )
    init_method = next(
        node
        for node in policy_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    cache_annotation = next(
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
        and node.target.attr == "_effective_te_cuda_graph_config"
    )
    assert ast.unparse(cache_annotation.annotation) == "_EffectiveTECudaGraphConfig"
    cache_call = next(
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_cache_effective_te_cuda_graph_config"
    )
    assert cache_annotation.lineno < cache_call.lineno


def _construction_policy_config(*, megatron_enabled: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name": "test/model",
        "train_global_batch_size": 6,
        "train_micro_batch_size": 2,
        "tokenizer": {},
        "draft": {"enabled": False},
        "sequence_packing": {
            "enabled": True,
            "algorithm": "modified_first_fit_decreasing",
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 64,
        },
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 64,
            "sequence_length_round": 8,
        },
        "make_sequence_length_divisible_by": 8,
    }
    if megatron_enabled:
        config.update(
            {
                "megatron_cfg": {
                    "enabled": True,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "env_vars": {},
                },
                "dtensor_cfg": {"enabled": False},
            }
        )
    else:
        config.update(
            {
                "megatron_cfg": {"enabled": False},
                "dtensor_cfg": {
                    "enabled": True,
                    "_v2": False,
                    "lora_cfg": {"enabled": False},
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                    "env_vars": {},
                },
            }
        )
    return config


def _base_sequence_packing_args() -> dict[str, Any]:
    return {
        "algorithm": "modified_first_fit_decreasing",
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": 8,
    }


def _make_policy(
    *,
    cuda_graph_impl: str | None = "transformer_engine",
    capacity: object = 5,
    use_sequence_packing: bool = True,
    effective_config: dict[str, Any] | None = None,
) -> Any:
    policy_type, _ = _harness_types()
    policy = policy_type()
    policy.cfg = _policy_config(
        cuda_graph_impl=cuda_graph_impl,
        capacity=capacity,
    )
    policy.use_sequence_packing = use_sequence_packing
    policy.use_dynamic_batches = False
    policy.sequence_packing_args = _base_sequence_packing_args()
    policy.data_parallel_size = 3
    policy.flops_tracker = None
    policy.worker_group = _WorkerGroup()
    if effective_config is None:
        training_enabled = cuda_graph_impl == "transformer_engine"
        policy._effective_te_cuda_graph_config = _resolved_effective_config(
            cuda_graph_impl=cuda_graph_impl or "none",
            capacity=cast(int | None, capacity),
            training_enabled=training_enabled,
        )
    else:
        policy._effective_te_cuda_graph_config = _resolved_effective_config(
            cuda_graph_impl=effective_config["cuda_graph_impl"],
            capacity=effective_config["thd_max_packed_sequences"],
            cache_capacity=effective_config["cuda_graph_max_cached_schedules"],
            training_enabled=effective_config["training_enabled"],
        )
    return policy


def _make_tq_policy(
    *,
    cuda_graph_impl: str | None = "transformer_engine",
    capacity: object = 5,
    effective_config: dict[str, Any] | None = None,
) -> Any:
    _, tq_policy_type = _harness_types()
    policy = tq_policy_type()
    policy.cfg = _policy_config(
        cuda_graph_impl=cuda_graph_impl,
        capacity=capacity,
    )
    policy.use_sequence_packing = True
    policy.use_dynamic_batches = False
    policy.sequence_packing_args = _base_sequence_packing_args()
    policy.dynamic_batching_args = {}
    policy.sharding_annotations = _ShardingAnnotations(3)
    policy.flops_tracker = None
    policy.worker_group = _WorkerGroup()
    policy._router_replay_enabled = False
    policy._stamp_pad_seqlen = lambda _meta: None
    if effective_config is None:
        training_enabled = cuda_graph_impl == "transformer_engine"
        policy._effective_te_cuda_graph_config = _resolved_effective_config(
            cuda_graph_impl=cuda_graph_impl or "none",
            capacity=cast(int | None, capacity),
            training_enabled=training_enabled,
        )
    else:
        policy._effective_te_cuda_graph_config = _resolved_effective_config(
            cuda_graph_impl=effective_config["cuda_graph_impl"],
            capacity=effective_config["thd_max_packed_sequences"],
            cache_capacity=effective_config["cuda_graph_max_cached_schedules"],
            training_enabled=effective_config["training_enabled"],
        )
    return policy


def test_effective_config_resolver_accepts_identical_training_ranks() -> None:
    resolver, _ = _effective_config_api()
    expected = _effective_config()

    resolved = resolver([dict(expected), dict(expected)])

    assert resolved == _resolved_effective_config()
    assert resolved is not expected


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (_effective_config(capacity=5), _effective_config(capacity=6)),
        (_effective_config(cache_capacity=2), _effective_config(cache_capacity=3)),
    ),
)
def test_effective_config_resolver_rejects_rank_disagreement(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    resolver, _ = _effective_config_api()

    with pytest.raises(ValueError, match="consistent across all Megatron workers"):
        resolver([first, second])


@pytest.mark.parametrize(
    "worker_results",
    [
        [],
        [None],
        [{"cuda_graph_impl": "transformer_engine"}],
        [{**_effective_config(), "unexpected": 1}],
        [_effective_config(cuda_graph_impl=cast(Any, 1))],
        [_effective_config(capacity=cast(Any, True))],
        [_effective_config(capacity=cast(Any, "5"))],
        [_effective_config(cache_capacity=cast(Any, True))],
        [_effective_config(cache_capacity=cast(Any, "3"))],
        [_effective_config(cache_capacity=0)],
        [_effective_config(training_enabled=cast(Any, 1))],
        [_effective_config(cuda_graph_impl="local")],
        [_effective_config(capacity=None)],
    ],
)
def test_effective_config_resolver_fails_closed_on_malformed_results(
    worker_results: list[Any],
) -> None:
    resolver, _ = _effective_config_api()

    with pytest.raises((TypeError, ValueError)):
        resolver(worker_results)


def test_megatron_policy_construction_caches_worker_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, policy_type = _effective_config_api()
    expected = _effective_config(capacity=7)
    _ConstructionWorkerGroup.instances.clear()
    _ConstructionWorkerGroup.effective_results = [dict(expected), dict(expected)]
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.0")

    policy = policy_type(
        cluster=_ConstructionCluster(world_size=1),
        config=_construction_policy_config(megatron_enabled=True),
        tokenizer=object(),
    )

    assert policy._effective_te_cuda_graph_config == _resolved_effective_config(
        capacity=7
    )
    assert _ConstructionWorkerGroup.instances[-1].dispatches == [
        ("get_effective_te_cuda_graph_config", {})
    ]


def test_dtensor_policy_construction_skips_worker_config_rpc() -> None:
    _, policy_type = _effective_config_api()
    _ConstructionWorkerGroup.instances.clear()
    _ConstructionWorkerGroup.effective_results = [_effective_config()]

    policy = policy_type(
        cluster=_ConstructionCluster(world_size=1),
        config=_construction_policy_config(megatron_enabled=False),
        tokenizer=object(),
    )

    assert policy._effective_te_cuda_graph_config == _resolved_effective_config(
        cuda_graph_impl="none",
        capacity=None,
        cache_capacity=None,
        training_enabled=False,
    )
    assert _ConstructionWorkerGroup.instances[-1].dispatches == []


def test_lm_inherited_worker_effective_graph_config_adds_sequence_cap() -> None:
    policy = _make_policy(
        cuda_graph_impl=None,
        effective_config=_effective_config(capacity=5),
    )
    data = _RecordingBatch()

    policy._shard_for_train(data, 6, eval_mode=False)

    assert data.calls[0]["sequence_packing_args"]["max_sequences_per_microbatch"] == 4


def test_lm_disabled_worker_effective_graph_config_ignores_stale_raw_config() -> None:
    policy = _make_policy(
        cuda_graph_impl="transformer_engine",
        capacity=5,
        effective_config=_effective_config(
            cuda_graph_impl="transformer_engine",
            capacity=5,
            training_enabled=False,
        ),
    )
    data = _RecordingBatch()

    policy._shard_for_train(data, 6, eval_mode=False)

    assert "max_sequences_per_microbatch" not in data.calls[0]["sequence_packing_args"]


def test_tq_inherited_worker_effective_graph_config_adds_sequence_cap() -> None:
    policy = _make_tq_policy(
        cuda_graph_impl=None,
        effective_config=_effective_config(capacity=5),
    )

    sequence_args, dynamic_args = policy._packing_args(
        "train_mb_tokens",
        for_cuda_graph_training=True,
    )

    assert dynamic_args is None
    assert sequence_args is not None
    assert sequence_args["max_sequences_per_microbatch"] == 4


def test_lm_real_train_uses_canonical_sequence_cap_and_dp_batch_semantics() -> None:
    policy = _make_policy()
    data = _RecordingBatch()
    base_args = dict(policy.sequence_packing_args)

    policy.train(data, loss_fn=object(), eval_mode=False)

    assert len(data.calls) == 1
    call = data.calls[0]
    assert call["shards"] == 3
    assert call["batch_size"] == 6
    assert call["sequence_packing_args"] == {
        **base_args,
        "max_tokens_per_microbatch": 128,
        "max_sequences_per_microbatch": 4,
    }
    assert call["sequence_packing_args"] is not policy.sequence_packing_args
    assert policy.sequence_packing_args == base_args


def test_lm_train_exposes_cuda_graph_metrics_outside_microbatch_metrics() -> None:
    policy = _make_policy()
    policy.worker_group = _WorkerGroup(
        worker_results=[
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
        ]
    )

    result = policy.train(_RecordingBatch(), loss_fn=object())

    assert result["cuda_graph_metrics"] == _CUDA_GRAPH_METRICS
    assert "cuda_graph_metrics" not in result["all_mb_metrics"]


def test_lm_train_omits_absent_cuda_graph_metrics() -> None:
    policy = _make_policy()
    policy.worker_group = _WorkerGroup(worker_results=[_worker_result()])

    result = policy.train(_RecordingBatch(), loss_fn=object())

    assert "cuda_graph_metrics" not in result


def test_lm_train_propagates_mixed_cuda_graph_metrics_error() -> None:
    policy = _make_policy()
    policy.worker_group = _WorkerGroup(
        worker_results=[
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
            _worker_result(),
        ]
    )

    with pytest.raises(ValueError, match="mixed CUDA Graph metrics"):
        policy.train(_RecordingBatch(), loss_fn=object())


def test_lm_eval_train_preserves_tokens_without_sequence_cap() -> None:
    policy = _make_policy()
    data = _RecordingBatch()
    base_args = dict(policy.sequence_packing_args)

    policy.train(data, loss_fn=object(), eval_mode=True)

    assert data.calls[0]["sequence_packing_args"] == {
        **base_args,
        "max_tokens_per_microbatch": 128,
    }
    assert policy.sequence_packing_args == base_args


def test_lm_logprob_does_not_inherit_training_sequence_cap() -> None:
    policy = _make_policy()
    train_data = _RecordingBatch()
    logprob_data = _RecordingBatch()
    base_args = dict(policy.sequence_packing_args)

    policy._shard_for_train(train_data, 6, eval_mode=False)
    policy._shard_for_logprob(logprob_data)

    assert (
        train_data.calls[0]["sequence_packing_args"]["max_sequences_per_microbatch"]
        == 4
    )
    assert logprob_data.calls[0]["sequence_packing_args"] == {
        **base_args,
        "max_tokens_per_microbatch": 64,
    }
    assert policy.sequence_packing_args == base_args


@pytest.mark.parametrize("cuda_graph_impl", [None, "none", "local"])
def test_lm_ordinary_training_never_adds_graph_sequence_cap(
    cuda_graph_impl: str | None,
) -> None:
    policy = _make_policy(cuda_graph_impl=cuda_graph_impl)
    data = _RecordingBatch()

    policy._shard_for_train(data, 6, eval_mode=False)

    assert data.calls[0]["sequence_packing_args"] == {
        **_base_sequence_packing_args(),
        "max_tokens_per_microbatch": 128,
    }


@pytest.mark.parametrize(
    ("capacity", "error_type"),
    [
        (None, TypeError),
        (True, TypeError),
        (1.0, TypeError),
        (1, ValueError),
        (0, ValueError),
    ],
)
def test_lm_real_graph_train_rejects_invalid_canonical_capacity(
    capacity: object,
    error_type: type[Exception],
) -> None:
    policy = _make_policy(capacity=capacity)
    data = _RecordingBatch()

    with pytest.raises(error_type, match="thd_max_packed_sequences"):
        policy._shard_for_train(data, 6, eval_mode=False)

    assert data.calls == []


def test_lm_real_graph_train_rejects_disabled_sequence_packing() -> None:
    policy = _make_policy(use_sequence_packing=False)
    data = _RecordingBatch()

    with pytest.raises(ValueError, match="sequence_packing.enabled=true"):
        policy._shard_for_train(data, 6, eval_mode=False)

    assert data.calls == []


def test_lm_eval_does_not_consume_invalid_training_graph_capacity() -> None:
    policy = _make_policy(capacity=True)
    data = _RecordingBatch()

    policy._shard_for_train(data, 6, eval_mode=True)

    assert "max_sequences_per_microbatch" not in data.calls[0]["sequence_packing_args"]


def test_tq_real_train_uses_canonical_sequence_cap_and_dp_batch_semantics() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()
    base_args = dict(policy.sequence_packing_args)

    policy.train_from_meta(_Meta(), loss_fn=object(), eval_mode=False)

    call = _META_SHARD_CALLS[-1]
    assert call["dp_world"] == 3
    assert call["batch_size"] == 6
    assert call["sequence_packing_args"] == {
        **base_args,
        "max_tokens_per_microbatch": 128,
        "max_sequences_per_microbatch": 4,
    }
    assert call["sequence_packing_args"] is not policy.sequence_packing_args
    assert policy.sequence_packing_args == base_args


def test_tq_sync_train_exposes_cuda_graph_metrics() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()
    policy.worker_group = _WorkerGroup(
        worker_results=[
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
        ]
    )

    result = policy.train_from_meta(_Meta(), loss_fn=object())

    assert result["cuda_graph_metrics"] == _CUDA_GRAPH_METRICS
    assert "cuda_graph_metrics" not in result["all_mb_metrics"]


def test_tq_split_train_exposes_cuda_graph_metrics() -> None:
    policy = _make_tq_policy()
    policy.worker_group = _WorkerGroup(
        single_data_results=[
            _worker_result(
                cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS),
                replica_leader=True,
            ),
            _worker_result(
                cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS),
                replica_leader=False,
            ),
        ]
    )

    result = policy.finish_train_step()

    assert result["cuda_graph_metrics"] == _CUDA_GRAPH_METRICS
    assert "cuda_graph_metrics" not in result["all_mb_metrics"]


def test_tq_sync_train_omits_absent_cuda_graph_metrics() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()
    policy.worker_group = _WorkerGroup(worker_results=[_worker_result()])

    result = policy.train_from_meta(_Meta(), loss_fn=object())

    assert "cuda_graph_metrics" not in result


def test_tq_sync_train_propagates_mixed_cuda_graph_metrics_error() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()
    policy.worker_group = _WorkerGroup(
        worker_results=[
            _worker_result(cuda_graph_metrics=dict(_CUDA_GRAPH_METRICS)),
            _worker_result(),
        ]
    )

    with pytest.raises(ValueError, match="mixed CUDA Graph metrics"):
        policy.train_from_meta(_Meta(), loss_fn=object())


def test_tq_eval_train_preserves_tokens_without_sequence_cap() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()

    policy.train_from_meta(_Meta(), loss_fn=object(), eval_mode=True)

    assert _META_SHARD_CALLS[-1]["sequence_packing_args"] == {
        **_base_sequence_packing_args(),
        "max_tokens_per_microbatch": 128,
    }


def test_tq_split_train_uses_canonical_sequence_cap_with_unbounded_batch() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()

    policy.train_microbatches_from_meta(_Meta())

    call = _META_SHARD_CALLS[-1]
    assert call["dp_world"] == 3
    assert call["batch_size"] is None
    assert call["sequence_packing_args"] == {
        **_base_sequence_packing_args(),
        "max_tokens_per_microbatch": 128,
        "max_sequences_per_microbatch": 4,
    }


def test_tq_logprob_does_not_receive_graph_training_sequence_cap() -> None:
    _META_SHARD_CALLS.clear()
    policy = _make_tq_policy()

    policy._logprob_dispatch(
        _Meta(),
        task_name="prev_lp",
        worker_method="get_logprobs_presharded",
        timer_prefix="get_logprobs",
        timer=None,
        common_kwargs={},
    )

    assert _META_SHARD_CALLS[-1]["batch_size"] is None
    assert _META_SHARD_CALLS[-1]["sequence_packing_args"] == {
        **_base_sequence_packing_args(),
        "max_tokens_per_microbatch": 64,
    }


@pytest.mark.parametrize("cuda_graph_impl", [None, "none", "local"])
def test_tq_ordinary_training_never_adds_graph_sequence_cap(
    cuda_graph_impl: str | None,
) -> None:
    policy = _make_tq_policy(cuda_graph_impl=cuda_graph_impl)

    sequence_args, dynamic_args = policy._packing_args(
        "train_mb_tokens",
        for_cuda_graph_training=True,
    )

    assert dynamic_args is None
    assert sequence_args == {
        **_base_sequence_packing_args(),
        "max_tokens_per_microbatch": 128,
    }
