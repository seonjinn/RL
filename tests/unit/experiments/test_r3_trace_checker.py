from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_CAPABILITY = "r3_router_cuda_graph_input_v1"
SAMPLE_IDENTITY = "f" * 64
SOURCE_IDENTITIES = [{"key": "legacy-step-1-sample-0", "identity": SAMPLE_IDENTITY}]
GRAPH_COUNTERS = {
    "route_payloads_produced": 2,
    "route_payloads_copied": 2,
    "route_graph_launches": 2,
    "route_eager_warmup_payloads": 0,
    "fallback_count": 0,
    "missing_route_count": 0,
    "stale_generation_count": 0,
    "malformed_route_count": 0,
    "out_of_range_count": 0,
    "duplicate_route_count": 0,
    "cp_mismatch_count": 0,
}


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
            "rank": 0,
            "key": "legacy-step-1-sample-0",
            "sample_identity": SAMPLE_IDENTITY,
            "valid_length": 4,
            **hashes,
        },
        {
            "event": "policy_payload_sample",
            "rank": 0,
            "stage": "prev_lp",
            "key": "legacy-step-1-sample-0",
            "sample_identity": SAMPLE_IDENTITY,
            "valid_length": 4,
            **hashes,
        },
        {
            "event": "policy_payload_sample",
            "rank": 0,
            "stage": "train",
            "key": "legacy-step-1-sample-0",
            "sample_identity": SAMPLE_IDENTITY,
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
    route_digests = {1: "c" * 64, 3: "d" * 64}
    layer_numbers = {1: 2, 3: 4}
    for record in records:
        if record.get("event") == "router_replay_assignment":
            payload_idx = record["payload_idx"]
            record["layer_number"] = layer_numbers[payload_idx]
            record["tensor"] = {
                "shape": [4, 2],
                "dtype": "torch.int64",
                "sha256": route_digests[payload_idx],
                "preview": [],
            }
            record["rank"] = 0
            record["trace_step"] = 1
            record["microbatch_generation"] = (
                11 if record["stage"] == "prev-logprob" else 17
            )
            record["route_digest"] = route_digests[payload_idx]
            record["source_sample_identities"] = SOURCE_IDENTITIES
    records = [
        record
        for record in records
        if not (
            record.get("event") == "router_replay_action"
            and record.get("action") == "replay_forward"
        )
    ]
    for stage, generation in (("prev-logprob", 11), ("train", 17)):
        for payload_idx, layer_number in layer_numbers.items():
            records.append(
                {
                    "event": "router_replay_action",
                    "stage": stage,
                    "action": "replay_forward",
                    "layer_number": layer_number,
                    "payload_idx": payload_idx,
                    "rank": 0,
                    "trace_step": 1,
                    "microbatch_generation": generation,
                    "route_digest": route_digests[payload_idx],
                    "source_sample_identities": SOURCE_IDENTITIES,
                }
            )
    for graph_index, (payload_idx, layer_number) in enumerate(
        layer_numbers.items()
    ):
        records.append(
            {
                "event": "router_replay_graph_consumer",
                "rank": 0,
                "trace_step": 1,
                "stage": "train",
                "action": "replay_forward",
                "layer_number": layer_number,
                "payload_idx": payload_idx,
                "microbatch_generation": 17,
                "route_digest": route_digests[payload_idx],
                "physical_signature": {
                    "shape": [4, 2],
                    "dtype": "torch.int64",
                    "device_type": "cuda",
                    "topk": 2,
                    "num_experts": 8,
                },
                "bank_id": 5,
                "graph_index": graph_index,
                "schedule_key": 5,
                "copy_generation": graph_index + 1,
                "successful_graph_launch": True,
                "capability_version": GRAPH_CAPABILITY,
                "source_sample_identities": SOURCE_IDENTITIES,
            }
        )
    records.append(
        {
            "event": "router_replay_graph_counters",
            "stage": "train",
            "counters": GRAPH_COUNTERS,
        }
    )
    if duplicate_producer:
        records.insert(1, dict(records[0]))
    if omit_consumer_hash:
        records[1 if not duplicate_producer else 2]["routed_experts"] = {
            "valid_shape": [4, 4, 2],
            "dtype": "torch.int32",
        }
    records = _production_warmup_trace(records)
    trace_dir.mkdir()
    (trace_dir / "r3_trace_test.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def _rewrite_trace(
    trace_dir: Path,
    mutate: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    updated = mutate(records)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in updated),
        encoding="utf-8",
    )


def _production_warmup_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_templates = [
        record
        for record in records
        if record.get("stage") == "train"
        and record.get("event")
        in {"router_replay_assignment", "router_replay_action", "router_replay_graph_consumer"}
        and (
            record.get("event") != "router_replay_action"
            or record.get("action") == "replay_forward"
        )
    ]
    records = [record for record in records if record not in train_templates]
    records = [
        record
        for record in records
        if record.get("event") != "router_replay_graph_counters"
    ]
    for step in range(1, 5):
        successful = step == 4
        for microbatch_idx in range(5):
            generation = 100 + (step - 1) * 5 + microbatch_idx
            for source in train_templates:
                clone = json.loads(json.dumps(source))
                clone["trace_step"] = step
                clone["enclosing_call_id"] = 1
                clone["world_size"] = 1
                clone["microbatch_generation"] = generation
                if clone["event"] == "router_replay_graph_consumer":
                    if successful:
                        clone["copy_generation"] = microbatch_idx + 1
                    else:
                        clone["successful_graph_launch"] = False
                        clone["bank_id"] = None
                        clone["graph_index"] = None
                        clone["copy_generation"] = None
                records.append(clone)
        counters = dict(GRAPH_COUNTERS)
        counters.update(
            route_payloads_produced=10,
            route_payloads_copied=10 if successful else 0,
            route_graph_launches=10 if successful else 0,
            route_eager_warmup_payloads=0 if successful else 10,
        )
        records.append(
            {
                "event": "router_replay_graph_counters",
                "stage": "train",
                "rank": 0,
                "world_size": 1,
                "trace_step": step,
                "enclosing_call_id": 1,
                "schedule_key": 5,
                "num_microbatches": 5,
                "counters": counters,
            }
        )
    summary_counters = dict(GRAPH_COUNTERS)
    summary_counters.update(
        route_payloads_produced=40,
        route_payloads_copied=10,
        route_graph_launches=10,
        route_eager_warmup_payloads=30,
    )
    records.append(
        {
            "event": "router_replay_graph_counter_summary",
            "scope": "rank_local_completed",
            "stage": "train",
            "rank": 0,
            "world_size": 1,
            "enclosing_call_id": 1,
            "schedule_key": 5,
            "num_microbatches": 5,
            "local_calls": [
                {
                    "trace_step": step,
                    "schedule_key": 5,
                    "num_microbatches": 5,
                }
                for step in range(1, 5)
            ],
            "local_trace_step_range": [1, 4],
            "local_call_count": 4,
            "schedule_keys": [5],
            "counters": summary_counters,
        }
    )
    return records


def _minimal_enclosing_counter_records(
    *,
    world_size: int,
    calls_by_rank: dict[int, dict[int, list[int]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, calls in calls_by_rank.items():
        for enclosing_call_id, trace_steps in calls.items():
            local_counters = dict(GRAPH_COUNTERS)
            local_counters.update(
                route_payloads_produced=1,
                route_payloads_copied=0,
                route_graph_launches=0,
                route_eager_warmup_payloads=1,
            )
            for trace_step in trace_steps:
                records.append(
                    {
                        "event": "router_replay_graph_counters",
                        "stage": "train",
                        "rank": rank,
                        "world_size": world_size,
                        "enclosing_call_id": enclosing_call_id,
                        "trace_step": trace_step,
                        "schedule_key": 1,
                        "num_microbatches": 1,
                        "counters": dict(local_counters),
                    }
                )
            summary_counters = dict(GRAPH_COUNTERS)
            summary_counters.update(
                route_payloads_produced=len(trace_steps),
                route_payloads_copied=0,
                route_graph_launches=0,
                route_eager_warmup_payloads=len(trace_steps),
            )
            records.append(
                {
                    "event": "router_replay_graph_counter_summary",
                    "scope": "rank_local_completed",
                    "stage": "train",
                    "rank": rank,
                    "world_size": world_size,
                    "enclosing_call_id": enclosing_call_id,
                    "local_calls": [
                        {
                            "trace_step": trace_step,
                            "schedule_key": 1,
                            "num_microbatches": 1,
                        }
                        for trace_step in sorted(trace_steps)
                    ],
                    "local_trace_step_range": [
                        min(trace_steps),
                        max(trace_steps),
                    ],
                    "local_call_count": len(trace_steps),
                    "schedule_keys": [1],
                    "counters": summary_counters,
                }
            )
    return records


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


def test_checker_rejects_non_boolean_forward_verifier_result(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record.get("event") == "router_replay_forward_verify":
            record["matches_expected"] = "false"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir, require_forward_verify=True) == 1


@pytest.mark.parametrize("invalid_payload_idx", (-1, "bad", True))
def test_checker_rejects_malformed_router_assignment_payload_index(
    tmp_path: Path,
    invalid_payload_idx: object,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records.append(
        {
            "event": "router_replay_assignment",
            "stage": "train",
            "payload_idx": invalid_payload_idx,
        }
    )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize("invalid_count", ("1", True, 1.5))
def test_checker_rejects_malformed_cp_identity_count(
    tmp_path: Path,
    invalid_count: object,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record.get("event") == "cp_routed_experts":
            record["cp_token_identity_verified_count"] = invalid_count
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir, require_cp_identity=True) == 1


def test_checker_rejects_boolean_lengths_and_tensor_dimensions(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if "valid_length" in record:
            record["valid_length"] = True
        input_ids = record.get("input_ids")
        if input_ids is not None:
            input_ids["valid_shape"] = [1]
        routed_experts = record.get("routed_experts")
        if routed_experts is not None:
            routed_experts["valid_shape"] = [1, 4, 2]
    semantics = records[0]["routed_experts"]["semantics"]
    semantics["valid_route_rows_by_layer"] = [0, 1, 0, 1]
    semantics["zero_route_rows_by_layer"] = [1, 0, 1, 0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_boolean_tensor_dimension(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    path = next(trace_dir.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if "valid_length" in record:
            record["valid_length"] = 1
        input_ids = record.get("input_ids")
        if input_ids is not None:
            input_ids["valid_shape"] = [True]
        routed_experts = record.get("routed_experts")
        if routed_experts is not None:
            routed_experts["valid_shape"] = [1, 4, 2]
    semantics = records[0]["routed_experts"]["semantics"]
    semantics["valid_route_rows_by_layer"] = [0, 1, 0, 1]
    semantics["zero_route_rows_by_layer"] = [1, 0, 1, 0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_missing_graph_consumers(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    _rewrite_trace(
        trace_dir,
        lambda records: [
            record
            for record in records
            if record.get("event") != "router_replay_graph_consumer"
        ],
    )

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_stale_graph_microbatch_generation(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def make_stale(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_consumer"
        )
        graph["microbatch_generation"] = 16
        return records

    _rewrite_trace(trace_dir, make_stale)

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (("layer_number", 3), ("payload_idx", 0)),
)
def test_checker_rejects_wrong_graph_layer_payload_mapping(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def corrupt_mapping(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_consumer"
        )
        graph[field] = value
        return records

    _rewrite_trace(trace_dir, corrupt_mapping)

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_graph_route_digest_mismatch(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def corrupt_digest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_consumer"
        )
        graph["route_digest"] = "e" * 64
        return records

    _rewrite_trace(trace_dir, corrupt_digest)

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("shape", [4, True]),
        ("dtype", "torch.int32"),
        ("device_type", 1),
        ("topk", True),
        ("num_experts", 0),
    ),
)
def test_checker_rejects_malformed_graph_physical_signature(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def corrupt_signature(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_consumer"
        )
        graph["physical_signature"][field] = value
        return records

    _rewrite_trace(trace_dir, corrupt_signature)

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_graph_evidence_for_subset_of_required_stage_mappings(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])
    _rewrite_trace(
        trace_dir,
        lambda records: [
            record
            for record in records
            if not (
                record.get("event") == "router_replay_graph_consumer"
                and record.get("payload_idx") == 3
            )
        ],
    )

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize("unsafe_counter", ["fallback_count", "cp_mismatch_count"])
def test_checker_rejects_unsafe_graph_counters(
    tmp_path: Path,
    unsafe_counter: str,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def increment_unsafe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter_record = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_counters"
        )
        counter_record["counters"][unsafe_counter] = 1
        return records

    _rewrite_trace(trace_dir, increment_unsafe)

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_disconnected_rollout_producer_replacement(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def replace_producer(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        producer = next(r for r in records if r.get("event") == "rollout_payload_sample")
        producer["sample_identity"] = "9" * 64
        return records

    _rewrite_trace(trace_dir, replace_producer)
    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_added_generation_without_graph_consumer(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def add_generation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in ("router_replay_assignment", "router_replay_action"):
            source = next(
                r
                for r in records
                if r.get("event") == event
                and r.get("stage") == "train"
                and r.get("layer_number") == 2
                and (event != "router_replay_action" or r.get("action") == "replay_forward")
            )
            clone = json.loads(json.dumps(source))
            clone["trace_step"] = 2
            clone["microbatch_generation"] = 18
            records.append(clone)
        return records

    _rewrite_trace(trace_dir, add_generation)
    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_duplicate_conflicting_graph_consumer(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def duplicate_graph(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        source = next(r for r in records if r.get("event") == "router_replay_graph_consumer")
        clone = json.loads(json.dumps(source))
        clone["graph_index"] = 99
        records.append(clone)
        return records

    _rewrite_trace(trace_dir, duplicate_graph)
    assert checker.check_trace(trace_dir) == 1


def test_checker_accepts_three_warmups_then_success(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    assert checker.check_trace(trace_dir) == 0


def test_checker_rejects_completed_local_calls_without_final_summary(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    _rewrite_trace(
        trace_dir,
        lambda records: [
            record
            for record in records
            if record.get("event") != "router_replay_graph_counter_summary"
        ],
    )

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_abort_after_successful_local_evidence(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def abort_after_success(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert any(
            record.get("event") == "router_replay_graph_consumer"
            and record.get("successful_graph_launch") is True
            for record in records
        )
        return [
            record
            for record in records
            if record.get("event") != "router_replay_graph_counter_summary"
        ]

    _rewrite_trace(trace_dir, abort_after_success)

    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_summary_from_earlier_enclosing_call(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def move_evidence_to_later_call(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for record in records:
            if record.get("event") in {
                "router_replay_assignment",
                "router_replay_action",
                "router_replay_graph_consumer",
                "router_replay_graph_counters",
            } and record.get("stage") == "train" and "trace_step" in record:
                record["enclosing_call_id"] = 2
        return records

    _rewrite_trace(trace_dir, move_evidence_to_later_call)

    assert checker.check_trace(trace_dir) == 1


def test_checker_requires_cross_rank_summary_agreement(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def add_second_rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rank_one: list[dict[str, Any]] = []
        for record in records:
            if record.get("rank") != 0 or record.get("event") not in {
                "router_replay_assignment",
                "router_replay_action",
                "router_replay_graph_consumer",
                "router_replay_graph_counters",
                "router_replay_graph_counter_summary",
            }:
                continue
            clone = json.loads(json.dumps(record))
            clone["rank"] = 1
            clone["world_size"] = 2
            rank_one.append(clone)
            for record in [*records, *rank_one]:
                if record.get("event") in {
                    "router_replay_assignment",
                    "router_replay_action",
                    "router_replay_graph_consumer",
                    "router_replay_graph_counters",
                    "router_replay_graph_counter_summary",
                }:
                    record["world_size"] = 2
        return [*records, *rank_one]

    _rewrite_trace(trace_dir, add_second_rank)

    assert checker.check_trace(trace_dir) == 0

    def disagree(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_counter_summary"
            and record.get("rank") == 1
        )
        summary["schedule_keys"] = []
        return records

    _rewrite_trace(trace_dir, disagree)

    assert checker.check_trace(trace_dir) == 1


def test_summary_checker_rejects_interleaved_enclosing_calls() -> None:
    checker = _load_checker()
    records = _minimal_enclosing_counter_records(
        world_size=1,
        calls_by_rank={0: {1: [1, 3], 2: [2, 4]}},
    )

    failures = checker._failures_for_graph_counter_summaries(records)

    assert any("interleaved" in failure for failure in failures)


def test_summary_checker_rejects_missing_world_rank() -> None:
    checker = _load_checker()
    records = _minimal_enclosing_counter_records(
        world_size=2,
        calls_by_rank={0: {1: [1, 2]}},
    )

    failures = checker._failures_for_graph_counter_summaries(records)

    assert any("rank set" in failure for failure in failures)


def test_summary_checker_accepts_two_sequential_calls_on_all_ranks() -> None:
    checker = _load_checker()
    records = _minimal_enclosing_counter_records(
        world_size=2,
        calls_by_rank={
            0: {1: [1, 2], 2: [3, 4]},
            1: {1: [1, 2], 2: [3, 4]},
        },
    )

    assert checker._failures_for_graph_counter_summaries(records) == []


def test_checker_rejects_duplicate_enclosing_call_summary(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def duplicate_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_counter_summary"
        )
        return [*records, json.loads(json.dumps(summary))]

    _rewrite_trace(trace_dir, duplicate_summary)

    assert checker.check_trace(trace_dir) == 1


def test_checker_requires_summary_for_each_completed_enclosing_call(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def append_second_completed_call(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        second: list[dict[str, Any]] = []
        for record in records:
            if record.get("event") not in {
                "router_replay_assignment",
                "router_replay_action",
                "router_replay_graph_consumer",
                "router_replay_graph_counters",
            } or record.get("stage") != "train" or "trace_step" not in record:
                continue
            clone = json.loads(json.dumps(record))
            clone["enclosing_call_id"] = 2
            clone["trace_step"] += 4
            if "microbatch_generation" in clone:
                clone["microbatch_generation"] += 100
            if clone.get("event") == "router_replay_graph_consumer":
                clone["successful_graph_launch"] = True
                clone["bank_id"] = 5
                clone["graph_index"] = {1: 0, 3: 1}[clone["payload_idx"]]
                clone["copy_generation"] = (
                    clone["microbatch_generation"] * 10 + clone["graph_index"]
                )
            if clone.get("event") == "router_replay_graph_counters":
                clone["counters"] = dict(clone["counters"])
                clone["counters"].update(
                    route_payloads_copied=10,
                    route_graph_launches=10,
                    route_eager_warmup_payloads=0,
                )
            second.append(clone)
        summary = json.loads(
            json.dumps(
                next(
                    record
                    for record in records
                    if record.get("event")
                    == "router_replay_graph_counter_summary"
                )
            )
        )
        summary["enclosing_call_id"] = 2
        summary["local_calls"] = [
            {
                "trace_step": step,
                "schedule_key": 5,
                "num_microbatches": 5,
            }
            for step in range(5, 9)
        ]
        summary["local_trace_step_range"] = [5, 8]
        summary["counters"] = dict(summary["counters"])
        summary["counters"].update(
            route_payloads_copied=40,
            route_graph_launches=40,
            route_eager_warmup_payloads=0,
        )
        return [*records, *second, summary]

    _rewrite_trace(trace_dir, append_second_completed_call)

    assert checker.check_trace(trace_dir) == 0

    _rewrite_trace(
        trace_dir,
        lambda records: [
            record
            for record in records
            if not (
                record.get("event") == "router_replay_graph_counter_summary"
                and record.get("enclosing_call_id") == 2
            )
        ],
    )

    assert checker.check_trace(trace_dir) == 1


@pytest.mark.parametrize("missing_step", (1, 4))
def test_checker_rejects_missing_call_local_counter(
    tmp_path: Path,
    missing_step: int,
) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def remove_counter(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if not (
                record.get("event") == "router_replay_graph_counters"
                and record.get("trace_step") == missing_step
            )
        ]

    _rewrite_trace(trace_dir, remove_counter)
    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_duplicate_graph_call_counter(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def duplicate_counter(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_counters"
        )
        records.append(json.loads(json.dumps(counter)))
        return records

    _rewrite_trace(trace_dir, duplicate_counter)
    assert checker.check_trace(trace_dir) == 1


def test_checker_rejects_mixed_success_within_warmup_call(tmp_path: Path) -> None:
    checker = _load_checker()
    trace_dir = tmp_path / "trace"
    _write_legacy_trace(trace_dir, populated_layers=[1, 3])

    def mix_call(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = next(
            record
            for record in records
            if record.get("event") == "router_replay_graph_consumer"
            and record.get("trace_step") == 1
        )
        graph.update(
            successful_graph_launch=True,
            bank_id=5,
            graph_index=0,
            copy_generation=1,
        )
        return records

    _rewrite_trace(trace_dir, mix_call)
    assert checker.check_trace(trace_dir) == 1
