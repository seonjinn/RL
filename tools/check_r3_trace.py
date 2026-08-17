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
GRAPH_CONSUMER_STAGE = "train"
GRAPH_INPUT_CAPABILITY = "r3_router_cuda_graph_input_v1"
GRAPH_COUNTER_FIELDS = (
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
UNSAFE_GRAPH_COUNTER_FIELDS = (
    "fallback_count",
    "missing_route_count",
    "stale_generation_count",
    "malformed_route_count",
    "out_of_range_count",
    "duplicate_route_count",
    "cp_mismatch_count",
)


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
        and all(type(dimension) is int and dimension >= 0 for dimension in shape)
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
                fetch_valid_length = fetch_record.get("valid_length")
                if type(fetch_valid_length) is not int or fetch_valid_length <= 0:
                    failures.append(f"invalid fetch valid_length {label}")
                elif fetch_valid_length != producer.get("valid_length"):
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
        if type(valid_length) is not int or valid_length <= 0:
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
        if type(layer_count) is not int or layer_count <= 0:
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
                and all(type(count) is int and count >= 0 for count in counts)
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
        if type(valid_length) is int and valid_length > 0:
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
                type(index) is int and 0 <= index < layer_count for index in populated
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
        if type(valid_length) is int and valid_length > 0:
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
        elif type(valid_length) is int and valid_length > 0 and valid_routed_shape:
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
            if type(count) is not int or count != 0:
                failures.append(
                    f"invalid routed_experts semantics key={key} {field}={count}"
                )
    return failures


def _trace_tensor_signature(
    record: dict[str, Any],
    *,
    label: str,
    failures: list[str],
) -> tuple[str, tuple[int, int], str] | None:
    tensor = record.get("tensor")
    if not isinstance(tensor, dict):
        failures.append(f"missing router assignment tensor for {label}")
        return None
    digest = tensor.get("sha256")
    shape = tensor.get("shape")
    dtype = tensor.get("dtype")
    if not _valid_sha256(digest):
        failures.append(f"invalid router assignment digest for {label}")
        return None
    if not (
        isinstance(shape, list)
        and len(shape) == 2
        and all(type(dimension) is int and dimension > 0 for dimension in shape)
    ):
        failures.append(f"invalid router assignment shape for {label}")
        return None
    if not isinstance(dtype, str) or not dtype:
        failures.append(f"invalid router assignment dtype for {label}")
        return None
    return str(digest), (int(shape[0]), int(shape[1])), dtype


def _graph_physical_signature(
    record: dict[str, Any],
    *,
    label: str,
    failures: list[str],
) -> tuple[tuple[int, int], str, str, int, int] | None:
    signature = record.get("physical_signature")
    if not isinstance(signature, dict):
        failures.append(f"missing graph physical_signature for {label}")
        return None
    shape = signature.get("shape")
    dtype = signature.get("dtype")
    device_type = signature.get("device_type")
    topk = signature.get("topk")
    num_experts = signature.get("num_experts")
    if not (
        isinstance(shape, list)
        and len(shape) == 2
        and all(type(dimension) is int and dimension > 0 for dimension in shape)
    ):
        failures.append(f"invalid graph physical_signature shape for {label}")
        return None
    if dtype != "torch.int64":
        failures.append(f"invalid graph physical_signature dtype for {label}")
        return None
    if not isinstance(device_type, str) or not device_type:
        failures.append(f"invalid graph physical_signature device_type for {label}")
        return None
    if type(topk) is not int or topk <= 0 or topk != shape[1]:
        failures.append(f"invalid graph physical_signature topk for {label}")
        return None
    if type(num_experts) is not int or num_experts < topk:
        failures.append(f"invalid graph physical_signature num_experts for {label}")
        return None
    return (
        (int(shape[0]), int(shape[1])),
        dtype,
        device_type,
        int(topk),
        int(num_experts),
    )


def _failures_for_graph_consumers(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    producer_identities = {
        (record.get("key"), record.get("sample_identity"))
        for record in records
        if record.get("event") == "rollout_payload_sample"
    }
    fetch_identities = {
        stage: {
            (record.get("key"), record.get("sample_identity"))
            for record in records
            if record.get("event") in {"policy_payload_sample", "tq_fetch_sample"}
            and record.get("stage") == stage
        }
        for stage in REQUIRED_FETCH_STAGES
    }

    def identity_key(record: dict[str, Any], label: str) -> tuple[tuple[str, str], ...] | None:
        raw = record.get("source_sample_identities")
        if not isinstance(raw, list) or not raw:
            failures.append(f"missing source sample identities for {label}")
            return None
        result: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                failures.append(f"malformed source sample identities for {label}")
                return None
            key, identity = item.get("key"), item.get("identity")
            if not isinstance(key, str) or not key or not _valid_sha256(identity):
                failures.append(f"malformed source sample identities for {label}")
                return None
            result.append((key, str(identity)))
        if len(set(result)) != len(result):
            failures.append(f"duplicate source sample identities for {label}")
            return None
        identities = tuple(result)
        for pair in identities:
            if pair not in producer_identities:
                failures.append(f"source identity is not linked to rollout for {label}")
            for fetch_stage in REQUIRED_FETCH_STAGES:
                if pair not in fetch_identities[fetch_stage]:
                    failures.append(
                        f"source identity is not linked to {fetch_stage} fetch for {label}"
                    )
        return identities

    def consumer_key(record: dict[str, Any], label: str) -> tuple[Any, ...] | None:
        rank = record.get("rank")
        step = record.get("trace_step")
        generation = record.get("microbatch_generation")
        layer = record.get("layer_number")
        payload = record.get("payload_idx")
        digest = record.get("route_digest")
        if type(rank) is not int or rank < 0 or type(step) is not int or step < 1:
            failures.append(f"invalid rank/trace_step for {label}")
            return None
        if type(generation) is not int or generation < 0:
            failures.append(f"invalid microbatch_generation for {label}")
            return None
        if type(layer) is not int or layer <= 0 or type(payload) is not int or payload < 0:
            failures.append(f"invalid layer/payload mapping for {label}")
            return None
        if not _valid_sha256(digest):
            failures.append(f"invalid route_digest for {label}")
            return None
        identities = identity_key(record, label)
        if identities is None:
            return None
        return rank, step, generation, layer, payload, str(digest), identities

    keyed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
        "prev-logprob": {},
        "prev_action": {},
        "train": {},
        "train_action": {},
        "graph": {},
    }
    for record in records:
        event = record.get("event")
        stage = record.get("stage")
        bucket: str | None = None
        if event == "router_replay_assignment" and stage in REQUIRED_REPLAY_STAGES:
            bucket = str(stage)
        elif event == "router_replay_action" and record.get("action") == "replay_forward":
            if stage == "train":
                bucket = "train_action"
            elif stage == "prev-logprob":
                bucket = "prev_action"
        elif event == "router_replay_graph_consumer":
            bucket = "graph"
        if bucket is None:
            continue
        label = f"event={event} stage={stage} rank={record.get('rank')}"
        key = consumer_key(record, label)
        if key is None:
            continue
        if key in keyed[bucket]:
            failures.append(f"duplicate/conflicting {bucket} consumer key {key[:-1]}")
        else:
            keyed[bucket][key] = record
        if event == "router_replay_assignment":
            signature = _trace_tensor_signature(record, label=label, failures=failures)
            if signature is not None and signature[0] != key[5]:
                failures.append(f"assignment tensor digest differs from route_digest for {label}")

    train_keys = set(keyed["train"])
    action_keys = set(keyed["train_action"])
    graph_keys = set(keyed["graph"])
    if train_keys != action_keys:
        failures.append("train assignment/replay_forward keyed completeness differs")
    if set(keyed["prev-logprob"]) != set(keyed["prev_action"]):
        failures.append("prev-logprob assignment/replay_forward keyed completeness differs")
    if train_keys != graph_keys:
        failures.append("train assignment/graph keyed completeness differs")
    if not graph_keys:
        failures.append("no router_replay_graph_consumer records for stage=train")

    prev_mapping_identity = {(key[3], key[4], key[5], key[6]) for key in keyed["prev-logprob"]}
    train_mapping_identity = {(key[3], key[4], key[5], key[6]) for key in train_keys}
    if prev_mapping_identity != train_mapping_identity:
        failures.append("router mapping/content identity differs between prev-logprob and train")

    surfaces: dict[tuple[Any, ...], list[tuple[int, int, bool, Any]]] = defaultdict(list)
    surface_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for key, record in keyed["graph"].items():
        label = f"rank={key[0]} step={key[1]} generation={key[2]} layer={key[3]} payload={key[4]}"
        if record.get("stage") != GRAPH_CONSUMER_STAGE or record.get("action") != "replay_forward":
            failures.append(f"invalid graph consumer stage/action for {label}")
        signature = _graph_physical_signature(record, label=label, failures=failures)
        if signature is None or signature[2] != "cuda":
            failures.append(f"graph physical signature is not CUDA for {label}")
            continue
        assignment = keyed["train"].get(key)
        assignment_signature = _trace_tensor_signature(assignment, label=label, failures=failures) if assignment else None
        if assignment_signature and (signature[0], signature[1]) != (assignment_signature[1], assignment_signature[2]):
            failures.append(f"graph physical signature differs from assignment for {label}")
        if record.get("capability_version") != GRAPH_INPUT_CAPABILITY:
            failures.append(f"invalid graph capability_version for {label}")
        schedule_key = record.get("schedule_key")
        if type(schedule_key) is not int or schedule_key < 1:
            failures.append(f"invalid graph schedule_key for {label}")
            continue
        successful = record.get("successful_graph_launch")
        if type(successful) is not bool:
            failures.append(f"invalid successful_graph_launch for {label}")
            continue
        if successful:
            for field, minimum in (("bank_id", 1), ("graph_index", 0), ("copy_generation", 1)):
                value = record.get(field)
                if type(value) is not int or value < minimum:
                    failures.append(f"invalid graph {field} for {label}")
            copy_generation = record.get("copy_generation")
            if record.get("bank_id") != schedule_key:
                failures.append(f"normalized bank_id differs from schedule_key for {label}")
        else:
            if any(record.get(field) is not None for field in ("bank_id", "graph_index", "copy_generation")):
                failures.append(f"unsuccessful graph consumer has non-null launch fields for {label}")
            copy_generation = 0
        surface = (key[0], key[3], key[4], schedule_key)
        previous_signature = surface_signatures.setdefault(surface, signature)
        if previous_signature != signature:
            failures.append(f"physical signature changed for surface={surface}")
        surfaces[surface].append(
            (key[1], int(copy_generation or 0), successful, record.get("graph_index"))
        )

    for surface, launches in surfaces.items():
        launches.sort()
        warmups = 0
        last_copy = 0
        promoted = False
        graph_index: int | None = None
        for _, copy_generation, successful, current_graph_index in launches:
            if not successful:
                warmups += 1
                if promoted or warmups > 3:
                    failures.append(f"invalid graph warmup window for surface={surface}")
            else:
                promoted = True
                if graph_index is None:
                    graph_index = current_graph_index
                elif graph_index != current_graph_index:
                    failures.append(f"graph index changed for surface={surface}")
                if copy_generation <= last_copy:
                    failures.append(f"copy generation did not strictly advance for surface={surface}")
                last_copy = copy_generation
        if not promoted:
            failures.append(f"no successful graph launch after warmup for surface={surface}")
    return failures


def _failures_for_graph_counters(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    counter_records = [
        record
        for record in records
        if record.get("event") == "router_replay_graph_counters"
    ]
    if not counter_records:
        return ["no router_replay_graph_counters records for stage=train"]
    saw_successful_window = False
    for record in counter_records:
        label = f"stage={record.get('stage')} rank={record.get('rank')}"
        if record.get("stage") != GRAPH_CONSUMER_STAGE:
            failures.append(f"invalid graph counter stage for {label}")
        counters = record.get("counters")
        if not isinstance(counters, dict) or set(counters) != set(GRAPH_COUNTER_FIELDS):
            failures.append(f"invalid graph counter schema for {label}")
            continue
        if any(
            type(counters[field]) is not int or counters[field] < 0
            for field in GRAPH_COUNTER_FIELDS
        ):
            failures.append(f"invalid graph counter value for {label}")
            continue
        unsafe = [
            f"{field}={counters[field]}"
            for field in UNSAFE_GRAPH_COUNTER_FIELDS
            if counters[field] != 0
        ]
        if unsafe:
            failures.append(f"unsafe graph counters for {label}: {', '.join(unsafe)}")
        produced = counters["route_payloads_produced"]
        copied = counters["route_payloads_copied"]
        launched = counters["route_graph_launches"]
        warmup = counters["route_eager_warmup_payloads"]
        if produced != copied + warmup or copied != launched:
            failures.append(
                "inconsistent graph route counters for "
                f"{label}: produced={produced} copied={copied} "
                f"launched={launched} warmup={warmup}"
            )
        if launched > 0:
            saw_successful_window = True
        elif warmup <= 0:
            failures.append(f"zero-launch graph counters lack warmup evidence for {label}")
    if not saw_successful_window:
        failures.append("no post-warmup graph counter record has a successful launch")
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
    invalid_cp_identity_verified_counts_by_stage = Counter()
    payload_indices_by_stage: dict[str, set[int]] = defaultdict(set)
    invalid_assignment_payload_indices_by_stage = Counter()
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
            if type(payload_idx) is int and payload_idx >= 0:
                payload_indices_by_stage[stage].add(payload_idx)
            else:
                invalid_assignment_payload_indices_by_stage[stage] += 1
        elif event == "router_replay_action":
            replay_actions_by_stage_action[(record["stage"], record["action"])] += 1
        elif event == "router_replay_forward_verify":
            replay_forward_verify_by_stage_action[
                (record["stage"], record["action"])
            ] += 1
        elif event == "cp_routed_experts":
            verified_count = record.get("cp_token_identity_verified_count")
            if verified_count is not None:
                stage = str(record.get("stage", "<missing>"))
                if type(verified_count) is not int or verified_count < 0:
                    invalid_cp_identity_verified_counts_by_stage[stage] += 1
                    continue
                cp_identity_verified_counts.append(verified_count)
                if verified_count > 0:
                    cp_identity_verified_by_stage[stage] += verified_count

    return {
        "events": events,
        "producer_by_key": producer_by_key,
        "fetch_by_stage_key": fetch_by_stage_key,
        "replay_assignments_by_stage": replay_assignments_by_stage,
        "replay_actions_by_stage_action": replay_actions_by_stage_action,
        "replay_forward_verify_by_stage_action": replay_forward_verify_by_stage_action,
        "cp_identity_verified_counts": cp_identity_verified_counts,
        "cp_identity_verified_by_stage": cp_identity_verified_by_stage,
        "invalid_cp_identity_verified_counts_by_stage": (
            invalid_cp_identity_verified_counts_by_stage
        ),
        "payload_indices_by_stage": payload_indices_by_stage,
        "invalid_assignment_payload_indices_by_stage": (
            invalid_assignment_payload_indices_by_stage
        ),
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
        invalid_count = summary["invalid_assignment_payload_indices_by_stage"][stage]
        if invalid_count > 0:
            failures.append(
                "invalid router_replay_assignment payload_idx records for "
                f"stage={stage}: {invalid_count}"
            )
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
    failures.extend(_failures_for_graph_consumers(records))
    failures.extend(_failures_for_graph_counters(records))

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
        if record.get("matches_expected") is not True:
            failures.append(
                "router replay forward verifier mismatch "
                f"stage={record.get('stage')} action={record.get('action')} "
                f"layer={record.get('layer_number')} rank={record.get('rank')}"
            )

    cp_identity_verified_counts = summary["cp_identity_verified_counts"]
    if require_cp_identity:
        for stage in REQUIRED_REPLAY_STAGES:
            invalid_count = summary["invalid_cp_identity_verified_counts_by_stage"][
                stage
            ]
            if invalid_count > 0:
                failures.append(
                    "invalid CP token-identity verification records for "
                    f"stage={stage}: {invalid_count}"
                )
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
