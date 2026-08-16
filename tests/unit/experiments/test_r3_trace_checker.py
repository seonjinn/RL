from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_checker() -> ModuleType:
    path = REPO_ROOT / "tools" / "check_r3_trace.py"
    spec = importlib.util.spec_from_file_location("test_r3_trace_checker_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_legacy_trace(
    trace_dir: Path,
    *,
    populated_layers: list[int],
    zero_route_rows_by_layer: list[int] | None = None,
    missing_route_rows_by_layer: list[int] | None = None,
    duplicate_producer: bool = False,
    omit_consumer_hash: bool = False,
) -> None:
    zero_route_rows_by_layer = zero_route_rows_by_layer or [
        0 if layer_idx in populated_layers else 4 for layer_idx in range(4)
    ]
    missing_route_rows_by_layer = missing_route_rows_by_layer or [0, 0, 0, 0]
    hashes = {
        "input_ids": {
            "valid_sha256": "a" * 64,
            "valid_shape": [4],
            "dtype": "torch.int64",
        },
        "routed_experts": {
            "valid_sha256": "b" * 64,
            "valid_shape": [4, 4, 2],
            "dtype": "torch.int32",
            "semantics": {
                "layer_count": 4,
                "populated_layer_indices": populated_layers,
                "valid_route_rows_by_layer": [
                    4 if layer_idx in populated_layers else 0 for layer_idx in range(4)
                ],
                "default_route_rows_by_layer": [0, 0, 0, 0],
                "missing_route_rows_by_layer": missing_route_rows_by_layer,
                "zero_route_rows_by_layer": zero_route_rows_by_layer,
                "duplicate_valid_rows": 0,
                "negative_valid_rows": 0,
                "zero_rows_in_populated_layers": 0,
            },
        },
    }
    records = [
        {
            "event": "rollout_payload_sample",
            "key": "legacy-step-1-sample-0",
            "valid_length": 4,
            **hashes,
        },
        {
            "event": "policy_payload_sample",
            "stage": "prev_lp",
            "key": "legacy-step-1-sample-0",
            "valid_length": 4,
            **hashes,
        },
        {
            "event": "policy_payload_sample",
            "stage": "train",
            "key": "legacy-step-1-sample-0",
            "valid_length": 4,
            **hashes,
        },
        {
            "event": "router_replay_assignment",
            "stage": "prev-logprob",
            "payload_idx": 1,
        },
        {
            "event": "router_replay_assignment",
            "stage": "train",
            "payload_idx": 1,
        },
        {
            "event": "router_replay_assignment",
            "stage": "prev-logprob",
            "payload_idx": 3,
        },
        {
            "event": "router_replay_assignment",
            "stage": "train",
            "payload_idx": 3,
        },
        {
            "event": "router_replay_action",
            "stage": "prev-logprob",
            "action": "replay_forward",
        },
        {
            "event": "router_replay_action",
            "stage": "train",
            "action": "replay_forward",
        },
        {
            "event": "router_replay_action",
            "stage": "train",
            "action": "replay_backward",
        },
        {
            "event": "router_replay_forward_verify",
            "stage": "prev-logprob",
            "action": "replay_forward",
            "matches_expected": True,
        },
        {
            "event": "router_replay_forward_verify",
            "stage": "train",
            "action": "replay_forward",
            "matches_expected": True,
        },
        {
            "event": "router_replay_forward_verify",
            "stage": "train",
            "action": "replay_backward",
            "matches_expected": True,
        },
        {
            "event": "cp_routed_experts",
            "stage": "prev-logprob",
            "cp_token_identity_verified_count": 1,
        },
        {
            "event": "cp_routed_experts",
            "stage": "train",
            "cp_token_identity_verified_count": 1,
        },
    ]
    if duplicate_producer:
        records.insert(1, dict(records[0]))
    if omit_consumer_hash:
        records[1 if not duplicate_producer else 2]["routed_experts"] = {
            "valid_shape": [4, 4, 2],
            "dtype": "torch.int32",
        }
    trace_dir.mkdir()
    (trace_dir / "r3_trace_test.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def test_checker_accepts_hash_matched_legacy_policy_payloads(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    assert (
        checker.check_trace(
            trace_dir,
            require_forward_verify=True,
            require_cp_identity=True,
        )
        == 0
    )


def test_checker_accepts_training_without_backward_recomputation(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [
        record
        for record in records
        if not (
            record.get("event") == "router_replay_forward_verify"
            and record.get("stage") == "train"
            and record.get("action") == "replay_backward"
        )
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir, require_forward_verify=True) == 0


def test_checker_accepts_default_padding_routes_on_structural_layers(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    producer = records[0]
    semantics = producer["routed_experts"]["semantics"]
    semantics["populated_layer_indices"] = [1, 3]
    semantics["valid_route_rows_by_layer"] = [1, 4, 1, 4]
    semantics["default_route_rows_by_layer"] = [1, 0, 1, 0]
    semantics["zero_route_rows_by_layer"] = [3, 0, 3, 0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 0


def test_checker_rejects_all_zero_rollout_routes(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[])

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_zero_routes_on_one_expected_moe_layer(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(
        trace_dir,
        populated_layers=[3],
        zero_route_rows_by_layer=[4, 4, 4, 0],
    )

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_default_only_routes_on_expected_moe_layer(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    semantics = records[0]["routed_experts"]["semantics"]
    semantics["populated_layer_indices"] = [3]
    semantics["default_route_rows_by_layer"] = [0, 4, 0, 0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_missing_routes_on_expected_moe_layer(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    semantics = records[0]["routed_experts"]["semantics"]
    semantics["populated_layer_indices"] = [3]
    semantics["valid_route_rows_by_layer"] = [0, 0, 0, 4]
    semantics["missing_route_rows_by_layer"] = [0, 4, 0, 0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_duplicate_producer_key(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(
        trace_dir,
        populated_layers=[1, 3],
        duplicate_producer=True,
    )

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_missing_consumer_hash(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(
        trace_dir,
        populated_layers=[1, 3],
        omit_consumer_hash=True,
    )

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize("invalid_shape", ([4], [4, 4, "2"], [4, 4, 0]))
def test_checker_rejects_invalid_route_shape_without_crashing(
    tmp_path: Path,
    invalid_shape: list[object],
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        routed_experts = record.get("routed_experts")
        if routed_experts is not None:
            routed_experts["valid_shape"] = invalid_shape
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_missing_routes_on_non_moe_layer(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(
        trace_dir,
        populated_layers=[1, 3],
        zero_route_rows_by_layer=[4, 0, 0, 0],
        missing_route_rows_by_layer=[0, 0, 4, 0],
    )

    assert checker.check_trace(trace_dir) == 1


def test_checker_requires_forward_and_cp_evidence_for_each_stage(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [
        record
        for record in records
        if not (
            record.get("stage") == "train"
            and record.get("event")
            in {"router_replay_forward_verify", "cp_routed_experts"}
        )
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert (
        checker.check_trace(
            trace_dir,
            require_forward_verify=True,
            require_cp_identity=True,
        )
        == 1
    )
