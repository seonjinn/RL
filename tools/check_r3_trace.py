#!/usr/bin/env python3
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

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FETCH_STAGE_TO_REPLAY_STAGE = {
    "prev_lp": "prev-logprob",
    "train": "train",
}
REQUIRED_FETCH_STAGES = ("prev_lp", "train")
REQUIRED_REPLAY_STAGES = ("prev-logprob", "train")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _iter_records(trace_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                record["_trace_file"] = str(path)
                record["_trace_line"] = line_no
                records.append(record)
    return records


def _tensor_signature(
    record: dict[str, Any],
    field: str,
    *,
    label: str,
    failures: list[str],
) -> tuple[str, tuple[int, ...], str] | None:
    tensor = record.get(field)
    if not isinstance(tensor, dict):
        failures.append(f"missing {field} tensor record for {label}")
        return None
    digest = tensor.get("valid_sha256")
    shape = tensor.get("valid_shape")
    dtype = tensor.get("dtype")
    if not isinstance(digest, str) or not _valid_sha256(digest):
        failures.append(f"invalid {field} valid_sha256 for {label}")
        return None
    if not (
        isinstance(shape, list)
        and shape
        and all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
    ):
        failures.append(f"invalid {field} valid_shape for {label}")
        return None
    if not isinstance(dtype, str) or not dtype:
        failures.append(f"invalid {field} dtype for {label}")
        return None
    return digest, tuple(int(dimension) for dimension in shape), dtype


def _failures_for_fetch_matches(
    producer_by_key: dict[str, dict[str, Any]],
    fetch_by_stage_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[str]:
    failures = []
    for key, producer in producer_by_key.items():
        producer_input = _tensor_signature(
            producer, "input_ids", label=f"producer key={key}", failures=failures
        )
        producer_routed = _tensor_signature(
            producer,
            "routed_experts",
            label=f"producer key={key}",
            failures=failures,
        )
        for stage in REQUIRED_FETCH_STAGES:
            fetch_records = fetch_by_stage_key.get((stage, key), [])
            if not fetch_records:
                failures.append(
                    f"missing policy payload record for stage={stage} key={key}"
                )
                continue
            for fetch_record in fetch_records:
                label = f"stage={stage} key={key} rank={fetch_record.get('rank')}"
                if fetch_record.get("valid_length") != producer.get("valid_length"):
                    failures.append(
                        "valid_length mismatch "
                        f"{label}: producer={producer.get('valid_length')} "
                        f"fetch={fetch_record.get('valid_length')}"
                    )
                fetch_input = _tensor_signature(
                    fetch_record, "input_ids", label=label, failures=failures
                )
                fetch_routed = _tensor_signature(
                    fetch_record, "routed_experts", label=label, failures=failures
                )
                if producer_input is not None and fetch_input is not None:
                    if producer_input != fetch_input:
                        failures.append(
                            "input_ids signature mismatch "
                            f"{label}: producer={producer_input} fetch={fetch_input}"
                        )
                if producer_routed is not None and fetch_routed is not None:
                    if producer_routed != fetch_routed:
                        failures.append(
                            "routed_experts signature mismatch "
                            f"{label}: producer={producer_routed} fetch={fetch_routed}"
                        )
    return failures


def _failures_for_route_semantics(
    producer_by_key: dict[str, dict[str, Any]],
    expected_payload_indices: set[int],
) -> list[str]:
    failures = []
    for key, producer in producer_by_key.items():
        valid_length = producer.get("valid_length")
        input_shape = producer.get("input_ids", {}).get("valid_shape")
        routed_shape = producer.get("routed_experts", {}).get("valid_shape")
        valid_routed_shape = (
            isinstance(routed_shape, list)
            and len(routed_shape) == 3
            and all(type(dim) is int for dim in routed_shape)
            and routed_shape[0] == valid_length
            and routed_shape[1] > 0
            and routed_shape[2] > 0
        )
        if not isinstance(valid_length, int) or valid_length <= 0:
            failures.append(f"invalid producer valid_length for key={key}")
        if not (
            isinstance(input_shape, list)
            and len(input_shape) == 1
            and input_shape[0] == valid_length
        ):
            failures.append(
                f"input_ids valid_shape disagrees with length for key={key}"
            )
        if not valid_routed_shape:
            failures.append(
                f"routed_experts valid_shape disagrees with length for key={key}"
            )
        semantics = producer.get("routed_experts", {}).get("semantics")
        if not isinstance(semantics, dict):
            failures.append(f"missing routed_experts semantics for key={key}")
            continue
        layer_count = semantics.get("layer_count")
        populated = semantics.get("populated_layer_indices")
        valid_by_layer = semantics.get("valid_route_rows_by_layer")
        default_by_layer = semantics.get("default_route_rows_by_layer")
        missing_by_layer = semantics.get("missing_route_rows_by_layer")
        zero_by_layer = semantics.get("zero_route_rows_by_layer")
        if not isinstance(layer_count, int) or layer_count <= 0:
            failures.append(f"invalid routed_experts layer_count for key={key}")
            continue
        raw_layer_fields = {
            "valid_route_rows_by_layer": valid_by_layer,
            "default_route_rows_by_layer": default_by_layer,
            "missing_route_rows_by_layer": missing_by_layer,
            "zero_route_rows_by_layer": zero_by_layer,
        }
        invalid_layer_fields = [
            field
            for field, counts in raw_layer_fields.items()
            if not (
                isinstance(counts, list)
                and len(counts) == layer_count
                and all(isinstance(count, int) and count >= 0 for count in counts)
            )
        ]
        for field in invalid_layer_fields:
            failures.append(f"invalid routed_experts {field} for key={key}")
        if invalid_layer_fields:
            continue
        assert isinstance(valid_by_layer, list)
        assert isinstance(default_by_layer, list)
        assert isinstance(missing_by_layer, list)
        assert isinstance(zero_by_layer, list)
        per_layer_fields: tuple[list[int], ...] = (
            [int(count) for count in valid_by_layer],
            [int(count) for count in missing_by_layer],
            [int(count) for count in zero_by_layer],
        )
        default_counts = [int(count) for count in default_by_layer]
        for layer_idx, (default_count, valid_count) in enumerate(
            zip(default_counts, per_layer_fields[0])
        ):
            if default_count > valid_count:
                failures.append(
                    "routed_experts default row count exceeds valid row count "
                    f"for key={key} layer={layer_idx}: "
                    f"default={default_count} valid={valid_count}"
                )
        if isinstance(valid_length, int) and valid_length > 0:
            for layer_idx in range(layer_count):
                row_count = sum(counts[layer_idx] for counts in per_layer_fields)
                if row_count != valid_length:
                    failures.append(
                        "routed_experts semantic row count mismatch "
                        f"for key={key} layer={layer_idx}: "
                        f"expected={valid_length} actual={row_count}"
                    )
        if valid_routed_shape and routed_shape[1] != layer_count:
            failures.append(
                f"routed_experts semantic layer_count disagrees with shape for key={key}"
            )
        if not (
            isinstance(populated, list)
            and all(
                isinstance(index, int) and 0 <= index < layer_count
                for index in populated
            )
        ):
            failures.append(f"invalid populated routed-expert layers for key={key}")
            continue
        if not populated:
            failures.append(f"no populated routed-expert layers for key={key}")
        populated_indices = [int(index) for index in populated]
        valid_counts, missing_counts, zero_counts = per_layer_fields
        observed_populated = {
            layer_idx
            for layer_idx, count in enumerate(valid_counts)
            if count > default_counts[layer_idx]
        }
        if set(populated_indices) != observed_populated:
            failures.append(
                f"populated routed-expert layers disagree with row counts for key={key}"
            )
        unexpected_populated = set(populated_indices) - expected_payload_indices
        if unexpected_populated:
            failures.append(
                f"unexpected populated routed-expert layers for key={key}: "
                f"{sorted(unexpected_populated)}"
            )
        if isinstance(valid_length, int) and valid_length > 0:
            invalid_structural_layers = {
                layer_idx
                for layer_idx in range(layer_count)
                if layer_idx not in expected_payload_indices
                and (
                    valid_counts[layer_idx] != default_counts[layer_idx]
                    or missing_counts[layer_idx] != 0
                    or zero_counts[layer_idx] + valid_counts[layer_idx] != valid_length
                )
            }
            if invalid_structural_layers:
                failures.append(
                    f"invalid structural routed-expert layers for key={key}: "
                    f"{sorted(invalid_structural_layers)}"
                )
        out_of_range = {
            index for index in expected_payload_indices if index >= layer_count
        }
        if out_of_range:
            failures.append(
                f"expected routed-expert layers out of range for key={key}: "
                f"{sorted(out_of_range)}"
            )
        elif isinstance(valid_length, int) and valid_length > 0 and valid_routed_shape:
            topk = routed_shape[2]
            missing_expected = {
                index for index in expected_payload_indices if missing_counts[index] > 0
            }
            if missing_expected:
                failures.append(
                    f"missing routes on expected MoE layers for key={key}: "
                    f"{sorted(missing_expected)}"
                )
            zero_expected = {
                index
                for index in expected_payload_indices
                if topk > 1 and zero_counts[index] > 0
            }
            if zero_expected:
                failures.append(
                    f"zero routes on expected MoE layers for key={key}: "
                    f"{sorted(zero_expected)}"
                )
            incomplete_expected = {
                index
                for index in expected_payload_indices
                if topk > 1 and valid_counts[index] != valid_length
            }
            if incomplete_expected:
                failures.append(
                    f"incomplete valid routes on expected MoE layers for key={key}: "
                    f"{sorted(incomplete_expected)}"
                )
            default_only_expected = {
                index
                for index in expected_payload_indices
                if topk > 1
                and valid_counts[index] > 0
                and valid_counts[index] == default_counts[index]
            }
            if default_only_expected:
                failures.append(
                    f"only default routes on expected MoE layers for key={key}: "
                    f"{sorted(default_only_expected)}"
                )
        for field in (
            "duplicate_valid_rows",
            "negative_valid_rows",
        ):
            count = semantics.get(field)
            if not isinstance(count, int) or count != 0:
                failures.append(
                    f"invalid routed_experts semantics key={key} {field}={count}"
                )
    return failures


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = Counter(record.get("event", "<missing>") for record in records)
    producer_by_key: dict[str, dict[str, Any]] = {}
    fetch_by_stage_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    replay_assignments_by_stage = Counter()
    replay_actions_by_stage_action = Counter()
    replay_forward_verify_by_stage_action = Counter()
    cp_identity_verified_counts = []
    cp_identity_verified_by_stage = Counter()
    payload_indices_by_stage: dict[str, set[int]] = defaultdict(set)
    duplicate_producer_keys: set[str] = set()
    ranks_by_event: dict[str, set[int]] = defaultdict(set)

    for record in records:
        event = str(record.get("event", "<missing>"))
        rank = record.get("rank")
        if isinstance(rank, int):
            ranks_by_event[event].add(rank)
        if event == "rollout_payload_sample":
            key = record["key"]
            if key in producer_by_key:
                duplicate_producer_keys.add(key)
            else:
                producer_by_key[key] = record
        elif event in {"tq_fetch_sample", "policy_payload_sample"}:
            fetch_by_stage_key[(record["stage"], record["key"])].append(record)
        elif event == "router_replay_assignment":
            stage = str(record.get("stage", "<missing>"))
            replay_assignments_by_stage[stage] += 1
            payload_idx = record.get("payload_idx")
            if isinstance(payload_idx, int) and payload_idx >= 0:
                payload_indices_by_stage[stage].add(payload_idx)
        elif event == "router_replay_action":
            replay_actions_by_stage_action[(record["stage"], record["action"])] += 1
        elif event == "router_replay_forward_verify":
            replay_forward_verify_by_stage_action[
                (record["stage"], record["action"])
            ] += 1
        elif event == "cp_routed_experts":
            verified_count = record.get("cp_token_identity_verified_count")
            if verified_count is not None:
                cp_identity_verified_counts.append(int(verified_count))
                if int(verified_count) > 0:
                    stage = str(record.get("stage", "<missing>"))
                    cp_identity_verified_by_stage[stage] += int(verified_count)

    return {
        "events": events,
        "producer_by_key": producer_by_key,
        "fetch_by_stage_key": fetch_by_stage_key,
        "replay_assignments_by_stage": replay_assignments_by_stage,
        "replay_actions_by_stage_action": replay_actions_by_stage_action,
        "replay_forward_verify_by_stage_action": replay_forward_verify_by_stage_action,
        "cp_identity_verified_counts": cp_identity_verified_counts,
        "cp_identity_verified_by_stage": cp_identity_verified_by_stage,
        "payload_indices_by_stage": payload_indices_by_stage,
        "duplicate_producer_keys": duplicate_producer_keys,
        "ranks_by_event": ranks_by_event,
    }


def check_trace(
    trace_dir: Path,
    *,
    require_forward_verify: bool = False,
    require_cp_identity: bool = False,
) -> int:
    records = _iter_records(trace_dir)
    summary = _summarize(records)
    failures: list[str] = []

    producer_by_key = summary["producer_by_key"]
    if not producer_by_key:
        failures.append("no rollout_payload_sample records found")

    for key in sorted(summary["duplicate_producer_keys"]):
        failures.append(f"duplicate rollout producer key={key}")

    payload_indices_by_stage = summary["payload_indices_by_stage"]
    for stage in REQUIRED_REPLAY_STAGES:
        if not payload_indices_by_stage[stage]:
            failures.append(f"no valid router payload indices for stage={stage}")
    if payload_indices_by_stage["prev-logprob"] != payload_indices_by_stage["train"]:
        failures.append(
            "router payload indices differ between prev-logprob and train: "
            f"prev={sorted(payload_indices_by_stage['prev-logprob'])} "
            f"train={sorted(payload_indices_by_stage['train'])}"
        )
    expected_payload_indices = set().union(
        *(payload_indices_by_stage[stage] for stage in REQUIRED_REPLAY_STAGES)
    )

    failures.extend(
        _failures_for_route_semantics(producer_by_key, expected_payload_indices)
    )
    failures.extend(
        _failures_for_fetch_matches(
            producer_by_key,
            summary["fetch_by_stage_key"],
        )
    )

    replay_assignments_by_stage = summary["replay_assignments_by_stage"]
    for stage in REQUIRED_REPLAY_STAGES:
        if replay_assignments_by_stage[stage] == 0:
            failures.append(f"no router_replay_assignment records for stage={stage}")

    replay_actions_by_stage_action = summary["replay_actions_by_stage_action"]
    for stage in REQUIRED_REPLAY_STAGES:
        if replay_actions_by_stage_action[(stage, "replay_forward")] == 0:
            failures.append(f"no replay_forward action records for stage={stage}")
    if replay_actions_by_stage_action[("train", "replay_backward")] == 0:
        failures.append("no replay_backward action records for stage=train")

    replay_forward_verify_records = [
        record
        for record in records
        if record.get("event") == "router_replay_forward_verify"
    ]
    if require_forward_verify:
        required_verifier_actions = (
            ("prev-logprob", "replay_forward"),
            ("train", "replay_forward"),
        )
        for stage, action in required_verifier_actions:
            if summary["replay_forward_verify_by_stage_action"][(stage, action)] == 0:
                failures.append(
                    "no router_replay_forward_verify records for "
                    f"stage={stage} action={action}"
                )
    for record in replay_forward_verify_records:
        if not record.get("matches_expected"):
            failures.append(
                "router replay forward verifier mismatch "
                f"stage={record.get('stage')} action={record.get('action')} "
                f"layer={record.get('layer_number')} rank={record.get('rank')}"
            )

    cp_identity_verified_counts = summary["cp_identity_verified_counts"]
    if require_cp_identity:
        for stage in REQUIRED_REPLAY_STAGES:
            if summary["cp_identity_verified_by_stage"][stage] <= 0:
                failures.append(
                    f"no positive CP token-identity verification for stage={stage}"
                )

    print(f"Trace dir: {trace_dir}")
    print(f"Records: {len(records)}")
    print("Events:")
    for event, count in sorted(summary["events"].items()):
        ranks = sorted(summary["ranks_by_event"].get(event, set()))
        rank_text = (
            f" ranks={ranks[:8]}{'...' if len(ranks) > 8 else ''}" if ranks else ""
        )
        print(f"  {event}: {count}{rank_text}")
    print("Producer keys:")
    for key, record in sorted(producer_by_key.items()):
        print(
            "  "
            f"{key}: len={record.get('valid_length')} "
            f"routed={record['routed_experts']['valid_sha256'][:12]}"
        )
    print("Replay assignments:")
    for stage, count in sorted(replay_assignments_by_stage.items()):
        print(f"  {stage}: {count}")
    print("Replay actions:")
    for (stage, action), count in sorted(replay_actions_by_stage_action.items()):
        print(f"  {stage}/{action}: {count}")
    replay_forward_verify_by_stage_action = summary[
        "replay_forward_verify_by_stage_action"
    ]
    if replay_forward_verify_by_stage_action:
        print("RouterReplay forward verifier:")
        for (stage, action), count in sorted(
            replay_forward_verify_by_stage_action.items()
        ):
            print(f"  {stage}/{action}: {count}")
    if cp_identity_verified_counts:
        print(
            "CP token identity verifier: "
            f"{len(cp_identity_verified_counts)} records, "
            f"{sum(cp_identity_verified_counts)} checked token rows"
        )

    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "\nPASS: producer routed_experts matched policy payloads, and replay was set."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an env-gated NeMo-RL R3 route trace."
    )
    parser.add_argument(
        "trace_dir", type=Path, help="Directory containing r3_trace_*.jsonl"
    )
    parser.add_argument(
        "--require-forward-verify",
        action="store_true",
        help="Require RouterReplay.get_replay_topk verifier records.",
    )
    parser.add_argument(
        "--require-cp-identity",
        action="store_true",
        help="Require CP token identity verifier records.",
    )
    args = parser.parse_args()
    if not args.trace_dir.is_dir():
        print(f"trace_dir is not a directory: {args.trace_dir}", file=sys.stderr)
        return 2
    return check_trace(
        args.trace_dir,
        require_forward_verify=args.require_forward_verify,
        require_cp_identity=args.require_cp_identity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
