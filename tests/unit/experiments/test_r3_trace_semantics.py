from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_trace_module() -> ModuleType:
    path = REPO_ROOT / "nemo_rl" / "utils" / "r3_trace.py"
    spec = importlib.util.spec_from_file_location("test_r3_trace_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_semantics_distinguishes_moe_routes_from_structural_zeroes() -> None:
    trace = _load_trace_module()
    rows = [
        [[0, 0, 0], [1, 4, 7], [-1, -1, -1]],
        [[0, 0, 0], [2, 5, 8], [-1, -1, -1]],
    ]

    semantics = trace._routed_experts_semantics(rows)

    assert semantics == {
        "layer_count": 3,
        "populated_layer_indices": [1],
        "valid_route_rows_by_layer": [0, 2, 0],
        "default_route_rows_by_layer": [0, 0, 0],
        "missing_route_rows_by_layer": [0, 0, 2],
        "zero_route_rows_by_layer": [2, 0, 0],
        "valid_route_rows": 2,
        "missing_route_rows": 2,
        "structural_zero_rows": 2,
        "duplicate_valid_rows": 0,
        "negative_valid_rows": 0,
        "zero_rows_in_populated_layers": 0,
    }


def test_route_semantics_reports_invalid_rows_in_populated_layer() -> None:
    trace = _load_trace_module()
    rows = [
        [[1, 2, 3]],
        [[0, 0, 0]],
        [[4, 4, 5]],
        [[-1, 6, 7]],
    ]

    semantics = trace._routed_experts_semantics(rows)

    assert semantics["populated_layer_indices"] == [0]
    assert semantics["valid_route_rows_by_layer"] == [3]
    assert semantics["missing_route_rows_by_layer"] == [0]
    assert semantics["zero_route_rows_by_layer"] == [1]
    assert semantics["zero_rows_in_populated_layers"] == 1
    assert semantics["duplicate_valid_rows"] == 1
    assert semantics["negative_valid_rows"] == 1


def test_route_semantics_distinguishes_default_routes_from_model_routes() -> None:
    trace = _load_trace_module()
    rows = [
        [[0, 0, 0], [1, 4, 7]],
        [[0, 1, 2], [0, 1, 2]],
    ]

    semantics = trace._routed_experts_semantics(rows)

    assert semantics["populated_layer_indices"] == [1]
    assert semantics["valid_route_rows_by_layer"] == [1, 2]
    assert semantics["default_route_rows_by_layer"] == [1, 1]
    assert semantics["zero_route_rows_by_layer"] == [1, 0]


def test_route_semantics_tensor_and_list_paths_match_for_topk_one() -> None:
    torch = pytest.importorskip("torch")
    trace = _load_trace_module()
    rows = [[[0]], [[-1]], [[7]]]

    list_semantics = trace._routed_experts_semantics(rows)
    tensor_semantics = trace._routed_experts_semantics(torch.tensor(rows))

    assert tensor_semantics == list_semantics
    assert list_semantics["valid_route_rows_by_layer"] == [1]
    assert list_semantics["default_route_rows_by_layer"] == [0]
    assert list_semantics["missing_route_rows_by_layer"] == [1]
    assert list_semantics["zero_route_rows_by_layer"] == [1]
