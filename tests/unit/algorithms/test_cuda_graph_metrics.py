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
import math
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTILS_PATH = _REPO_ROOT / "nemo_rl/algorithms/utils.py"
_CONSTANT_NAMES = {
    "_CUDA_GRAPH_RAW_METRIC_KEYS",
    "_CUDA_GRAPH_REPLICATED_METRIC_KEYS",
    "_CUDA_GRAPH_SUM_METRIC_KEYS",
    "_CUDA_GRAPH_CONTRACT_MINIMUMS",
    "_CUDA_GRAPH_RATIO_KEYS",
    "_CUDA_GRAPH_POLICY_METRIC_KEYS",
}
_FUNCTION_NAMES = {
    "_require_exact_mapping",
    "_require_plain_nonnegative_integers",
    "_validate_cuda_graph_counter_order",
    "_cuda_graph_ratios",
    "aggregate_cuda_graph_metrics",
    "merge_cuda_graph_metrics",
}


def _load_cuda_graph_helpers() -> dict[str, Any]:
    """Compile the actual helper definitions without importing GPU dependencies."""
    tree = ast.parse(_UTILS_PATH.read_text())
    selected: list[ast.stmt] = []
    found_constants: set[str] = set()
    found_functions: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in _CONSTANT_NAMES
        ):
            selected.append(node)
            found_constants.add(node.target.id)
        elif isinstance(node, ast.FunctionDef) and node.name in _FUNCTION_NAMES:
            selected.append(node)
            found_functions.add(node.name)

    assert found_constants == _CONSTANT_NAMES
    assert found_functions == _FUNCTION_NAMES
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Iterable": Iterable,
        "Mapping": Mapping,
        "math": math,
    }
    exec(compile(module, str(_UTILS_PATH), "exec"), namespace)
    return namespace


_HELPERS = _load_cuda_graph_helpers()
aggregate_cuda_graph_metrics = _HELPERS["aggregate_cuda_graph_metrics"]
merge_cuda_graph_metrics = _HELPERS["merge_cuda_graph_metrics"]


def _raw_metrics(**overrides: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "capture_count": 2,
        "replay_count": 5,
        "cache_hit_count": 7,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 1,
        "eligible_calls": 1,
        "logical_tokens": 4,
        "padded_tokens": 5,
        "capacity_tokens": 8,
    }
    metrics.update(overrides)
    return metrics


def _contract(**overrides: Any) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "normalized_schedule_key": 9,
        "token_capacity_per_microbatch": 10,
        "thd_max_packed_sequences": 4,
    }
    contract.update(overrides)
    return contract


def _worker_result(
    *,
    metrics: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "loss": 0.5,
        "cuda_graph_metrics": _raw_metrics() if metrics is None else metrics,
        "cuda_graph_contract": _contract() if contract is None else contract,
    }


def _policy_metrics(**overrides: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "capture_count": 2,
        "replay_count": 5,
        "cache_hit_count": 7,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 2,
        "eligible_calls": 4,
        "logical_tokens": 6,
        "padded_tokens": 12,
        "capacity_tokens": 20,
        "coverage": 0.5,
        "capacity_utilization": 0.3,
        "padding_utilization": 0.5,
    }
    metrics.update(overrides)
    return metrics


def test_aggregate_cuda_graph_metrics_selects_sums_and_recomputes_ratios() -> None:
    result = aggregate_cuda_graph_metrics(
        [
            _worker_result(),
            _worker_result(
                metrics=_raw_metrics(
                    graph_calls=1,
                    eligible_calls=3,
                    logical_tokens=2,
                    padded_tokens=7,
                    capacity_tokens=12,
                )
            ),
        ]
    )

    assert result == {
        "capture_count": 2,
        "replay_count": 5,
        "cache_hit_count": 7,
        "eviction_count": 1,
        "fallback_count": 0,
        "graph_calls": 2,
        "eligible_calls": 4,
        "logical_tokens": 6,
        "padded_tokens": 12,
        "capacity_tokens": 20,
        "coverage": 0.5,
        "capacity_utilization": 0.3,
        "padding_utilization": 0.5,
    }
    assert result is not None
    assert all(type(result[key]) is int for key in tuple(result)[:10])
    assert all(type(result[key]) is float for key in tuple(result)[10:])


def test_aggregate_cuda_graph_metrics_never_averages_worker_ratios() -> None:
    result = aggregate_cuda_graph_metrics(
        [
            _worker_result(),
            _worker_result(
                metrics=_raw_metrics(
                    graph_calls=1,
                    eligible_calls=3,
                    logical_tokens=2,
                    padded_tokens=7,
                    capacity_tokens=12,
                )
            ),
        ]
    )

    assert result is not None
    assert result["coverage"] == pytest.approx(2 / 4)
    assert result["capacity_utilization"] == pytest.approx(6 / 20)
    assert result["padding_utilization"] == pytest.approx(6 / 12)
    assert result["coverage"] != pytest.approx((1.0 + 1 / 3) / 2)
    assert result["capacity_utilization"] != pytest.approx((4 / 8 + 2 / 12) / 2)
    assert result["padding_utilization"] != pytest.approx((4 / 5 + 2 / 7) / 2)


def test_aggregate_cuda_graph_metrics_uses_float_zero_for_zero_denominators() -> None:
    result = aggregate_cuda_graph_metrics(
        [
            _worker_result(
                metrics=_raw_metrics(
                    graph_calls=0,
                    eligible_calls=0,
                    logical_tokens=0,
                    padded_tokens=0,
                    capacity_tokens=0,
                )
            )
        ]
    )

    assert result is not None
    assert result["coverage"] == 0.0
    assert result["capacity_utilization"] == 0.0
    assert result["padding_utilization"] == 0.0
    assert type(result["coverage"]) is float
    assert type(result["capacity_utilization"]) is float
    assert type(result["padding_utilization"]) is float


@pytest.mark.parametrize(
    ("mapping_name", "field"),
    [
        ("cuda_graph_metrics", "capture_count"),
        ("cuda_graph_metrics", "replay_count"),
        ("cuda_graph_metrics", "cache_hit_count"),
        ("cuda_graph_metrics", "eviction_count"),
        ("cuda_graph_metrics", "fallback_count"),
        ("cuda_graph_contract", "normalized_schedule_key"),
        ("cuda_graph_contract", "token_capacity_per_microbatch"),
        ("cuda_graph_contract", "thd_max_packed_sequences"),
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_replicated_field_mismatch(
    mapping_name: str, field: str
) -> None:
    second = _worker_result()
    second[mapping_name][field] += 1

    with pytest.raises(ValueError, match=field):
        aggregate_cuda_graph_metrics([_worker_result(), second])


def test_aggregate_cuda_graph_metrics_rejects_nonzero_fallback() -> None:
    with pytest.raises(ValueError, match="fallback_count"):
        aggregate_cuda_graph_metrics(
            [_worker_result(metrics=_raw_metrics(fallback_count=1))]
        )


def test_aggregate_cuda_graph_metrics_returns_none_when_all_mappings_absent() -> None:
    assert aggregate_cuda_graph_metrics([]) is None
    assert aggregate_cuda_graph_metrics([{"loss": 0.5}, {"grad_norm": 1.0}]) is None


@pytest.mark.parametrize(
    "worker_results",
    [
        [{"loss": 0.5}, _worker_result()],
        [{"cuda_graph_metrics": _raw_metrics()}],
        [{"cuda_graph_contract": _contract()}],
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_mixed_or_one_sided_presence(
    worker_results: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="cuda_graph"):
        aggregate_cuda_graph_metrics(worker_results)


@pytest.mark.parametrize("mapping_name", ["cuda_graph_metrics", "cuda_graph_contract"])
@pytest.mark.parametrize("mutation", ["empty", "partial", "unknown"])
def test_aggregate_cuda_graph_metrics_rejects_nonexact_nested_keys(
    mapping_name: str, mutation: str
) -> None:
    worker = _worker_result()
    nested = worker[mapping_name]
    if mutation == "empty":
        nested.clear()
    elif mutation == "partial":
        nested.pop(next(iter(nested)))
    else:
        nested["unknown"] = 0

    with pytest.raises(ValueError, match=mapping_name):
        aggregate_cuda_graph_metrics([worker])


@pytest.mark.parametrize(
    ("mapping_name", "field", "value"),
    [
        ("cuda_graph_metrics", "capture_count", True),
        ("cuda_graph_metrics", "graph_calls", 1.0),
        ("cuda_graph_metrics", "logical_tokens", "4"),
        ("cuda_graph_metrics", "capacity_tokens", -1),
        ("cuda_graph_contract", "normalized_schedule_key", False),
        ("cuda_graph_contract", "token_capacity_per_microbatch", 10.0),
        ("cuda_graph_contract", "thd_max_packed_sequences", -1),
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_non_plain_or_negative_integers(
    mapping_name: str, field: str, value: Any
) -> None:
    worker = _worker_result()
    worker[mapping_name][field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        aggregate_cuda_graph_metrics([worker])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_schedule_key", 0),
        ("token_capacity_per_microbatch", 0),
        ("thd_max_packed_sequences", 0),
        ("thd_max_packed_sequences", 1),
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_contract_values_below_minimum(
    field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field):
        aggregate_cuda_graph_metrics(
            [_worker_result(contract=_contract(**{field: value}))]
        )


@pytest.mark.parametrize(
    ("mapping_name", "value"),
    [
        ("cuda_graph_metrics", None),
        ("cuda_graph_metrics", []),
        ("cuda_graph_contract", None),
        ("cuda_graph_contract", []),
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_non_mapping_payloads(
    mapping_name: str, value: Any
) -> None:
    worker = _worker_result()
    worker[mapping_name] = value

    with pytest.raises((TypeError, ValueError), match=mapping_name):
        aggregate_cuda_graph_metrics([worker])


@pytest.mark.parametrize(
    "metrics",
    [
        _raw_metrics(graph_calls=2, eligible_calls=1),
        _raw_metrics(logical_tokens=6, padded_tokens=5),
        _raw_metrics(padded_tokens=9, capacity_tokens=8),
    ],
)
def test_aggregate_cuda_graph_metrics_rejects_invalid_counter_order(
    metrics: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        aggregate_cuda_graph_metrics([_worker_result(metrics=metrics)])


def test_merge_cuda_graph_metrics_prefixes_exact_policy_mapping() -> None:
    destination: dict[str, Any] = {"loss": 0.25}

    result = merge_cuda_graph_metrics(
        destination, {"cuda_graph_metrics": _policy_metrics()}
    )

    assert result is None
    assert destination == {
        "loss": 0.25,
        "cuda_graph/capture_count": 2,
        "cuda_graph/replay_count": 5,
        "cuda_graph/cache_hit_count": 7,
        "cuda_graph/eviction_count": 1,
        "cuda_graph/fallback_count": 0,
        "cuda_graph/graph_calls": 2,
        "cuda_graph/eligible_calls": 4,
        "cuda_graph/logical_tokens": 6,
        "cuda_graph/padded_tokens": 12,
        "cuda_graph/capacity_tokens": 20,
        "cuda_graph/coverage": 0.5,
        "cuda_graph/capacity_utilization": 0.3,
        "cuda_graph/padding_utilization": 0.5,
    }
    assert "cuda_graph_metrics" not in destination


def test_merge_cuda_graph_metrics_uses_each_validated_value_once() -> None:
    class ChangingMapping(Mapping[str, Any]):
        def __init__(self, values: dict[str, Any]) -> None:
            self._values = values
            self._reads = {key: 0 for key in values}

        def __getitem__(self, key: str) -> Any:
            self._reads[key] += 1
            if self._reads[key] > 1:
                return "changed-after-validation"
            return self._values[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    metrics = _policy_metrics()
    destination: dict[str, Any] = {}

    merge_cuda_graph_metrics(
        destination, {"cuda_graph_metrics": ChangingMapping(metrics)}
    )

    assert destination == {f"cuda_graph/{key}": value for key, value in metrics.items()}


def test_merge_cuda_graph_metrics_noops_only_when_mapping_is_absent() -> None:
    destination = {"loss": 0.25}

    merge_cuda_graph_metrics(destination, {"grad_norm": 1.0})

    assert destination == {"loss": 0.25}
    with pytest.raises((TypeError, ValueError), match="cuda_graph_metrics"):
        merge_cuda_graph_metrics(destination, {"cuda_graph_metrics": None})


def test_merge_cuda_graph_metrics_rejects_non_mapping_payload() -> None:
    with pytest.raises((TypeError, ValueError), match="cuda_graph_metrics"):
        merge_cuda_graph_metrics({}, {"cuda_graph_metrics": []})


@pytest.mark.parametrize("mutation", ["empty", "partial", "unknown"])
def test_merge_cuda_graph_metrics_rejects_nonexact_keys(mutation: str) -> None:
    policy_metrics = _policy_metrics()
    if mutation == "empty":
        policy_metrics.clear()
    elif mutation == "partial":
        policy_metrics.pop("padding_utilization")
    else:
        policy_metrics["unknown"] = 0

    with pytest.raises(ValueError, match="cuda_graph_metrics"):
        merge_cuda_graph_metrics({}, {"cuda_graph_metrics": policy_metrics})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_count", True),
        ("graph_calls", 2.0),
        ("eligible_calls", -1),
        ("coverage", 1),
        ("capacity_utilization", float("nan")),
        ("padding_utilization", -0.5),
    ],
)
def test_merge_cuda_graph_metrics_rejects_malformed_values(
    field: str, value: Any
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        merge_cuda_graph_metrics(
            {}, {"cuda_graph_metrics": _policy_metrics(**{field: value})}
        )


def test_merge_cuda_graph_metrics_rejects_inconsistent_ratio() -> None:
    with pytest.raises(ValueError, match="coverage"):
        merge_cuda_graph_metrics(
            {}, {"cuda_graph_metrics": _policy_metrics(coverage=0.25)}
        )


def test_merge_cuda_graph_metrics_rejects_collision_without_partial_update() -> None:
    destination = {"loss": 0.25, "cuda_graph/capture_count": 99}
    before = destination.copy()

    with pytest.raises(ValueError, match="cuda_graph/capture_count"):
        merge_cuda_graph_metrics(destination, {"cuda_graph_metrics": _policy_metrics()})

    assert destination == before
