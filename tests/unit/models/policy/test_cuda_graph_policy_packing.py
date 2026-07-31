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


class _WorkerGroup:
    def __init__(self) -> None:
        self.dispatches: list[tuple[str, dict[str, Any]]] = []

    def run_all_workers_sharded_data(
        self,
        method_name: str,
        **kwargs: Any,
    ) -> object:
        self.dispatches.append((method_name, kwargs))
        return object()

    def get_all_worker_results(self, _futures: object) -> list[dict[str, Any]]:
        return [
            {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {},
            }
        ]


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


def _aggregate_train_results(_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"loss": 1.0, "grad_norm": 0.5, "all_mb_metrics": {}}


_METHOD_GLOBALS: dict[str, Any] = {
    "Any": Any,
    "DP_TRAIN_FIELDS": ("input_ids", "input_lengths"),
    "DynamicBatchingConfig": dict,
    "LP_SEED_FIELDS": ("input_ids", "input_lengths"),
    "SequencePackingConfig": dict,
    "_aggregate_megatron_flops_metrics": lambda *_args: {},
    "_aggregate_train_results": _aggregate_train_results,
    "defaultdict": defaultdict,
    "cast": cast,
    "fields_with_optional_routed_experts": _fields_with_optional_routed_experts,
    "get_theoretical_tflops": lambda *_args: 0.0,
    "nullcontext": nullcontext,
    "replace": replace,
    "shard_meta_for_dp": _record_shard_meta_for_dp,
    "warnings": warnings,
}


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
    tq_methods = _extract_class_methods(
        _TQ_POLICY_PATH,
        "TQPolicy",
        {
            "_logprob_dispatch",
            "_packing_args",
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
    return policy


def _make_tq_policy(
    *,
    cuda_graph_impl: str | None = "transformer_engine",
    capacity: object = 5,
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
    return policy


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
