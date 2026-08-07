import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from experiments.mxfp8_moe_tactic_audit.schema import (
    ReplayProfile,
    RoutingSignature,
    TacticMeasurement,
    TacticPair,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_3_ROUTING_SIGNATURE_FIXTURE = (
    ROOT / "tests/fixtures/mxfp8_moe_tactic_audit/task3-routing-signature.jsonl"
)


def _valid_row() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_revision": "qwen3-30ba3b-test",
        "layer_family": "routed_experts",
        "num_tokens": 2,
        "global_num_experts": 4,
        "local_num_experts": 4,
        "top_k": 2,
        "hidden_size": 2048,
        "intermediate_size": 768,
        "expert_counts": [1, 2, 1, 0],
        "sampled_gpu_time_us": 17.5,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 16,
        "cuda_graph_state": "trace-eager",
        "weight_layout": "MajorK",
        "quantization": "MXFP8",
        "runtime_fingerprint": "runtime-sha256",
    }


def _structural_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key != "sampled_gpu_time_us"
    }


def test_routing_signature_accepts_producer_generated_task_3_jsonl() -> None:
    lines = TASK_3_ROUTING_SIGNATURE_FIXTURE.read_text(encoding="ascii").splitlines()
    assert len(lines) == 1
    row = cast(dict[str, object], json.loads(lines[0]))
    signature = RoutingSignature.from_json(row)

    assert signature.to_json() == row


def test_routing_signature_normalizes_direct_list_counts_to_immutable_tuple() -> None:
    input_counts = [1, 2, 1, 0]
    signature = RoutingSignature(
        schema_version=1,
        model_revision="qwen3-30ba3b-test",
        layer_family="routed_experts",
        num_tokens=2,
        global_num_experts=4,
        local_num_experts=4,
        top_k=2,
        hidden_size=2048,
        intermediate_size=768,
        expert_counts=cast(tuple[int, ...], input_counts),
        sampled_gpu_time_us=17.5,
        tp_size=1,
        ep_size=1,
        dp_size=16,
        cuda_graph_state="trace-eager",
        weight_layout="MajorK",
        quantization="MXFP8",
        runtime_fingerprint="runtime-sha256",
    )

    assert signature.expert_counts == (1, 2, 1, 0)
    assert isinstance(signature.expert_counts, tuple)
    input_counts[0] = 4
    assert signature.expert_counts == (1, 2, 1, 0)
    with pytest.raises(AttributeError):
        cast(list[int], signature.expert_counts).append(5)
    assert RoutingSignature.from_json(signature.to_json()) == signature


def test_routing_signature_rejects_histogram_sum_mismatch() -> None:
    row = _valid_row()
    row["expert_counts"] = [0] * 4

    with pytest.raises(ValueError, match="num_tokens \\* top_k"):
        RoutingSignature.from_json(row)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("expert_counts", [1, 3], "global_num_experts"),
        ("expert_counts", [1, -1, 3, 1], "nonnegative"),
        ("num_tokens", 0, "positive"),
        ("global_num_experts", 0, "positive"),
        ("local_num_experts", 0, "positive"),
        ("top_k", 0, "positive"),
        ("hidden_size", 0, "positive"),
        ("intermediate_size", 0, "positive"),
        ("tp_size", 0, "positive"),
        ("ep_size", 0, "positive"),
        ("dp_size", 0, "positive"),
        ("sampled_gpu_time_us", 0.0, "finite and positive"),
        ("sampled_gpu_time_us", float("inf"), "finite and positive"),
        ("quantization", "FP8", "MXFP8"),
    ],
)
def test_routing_signature_rejects_invalid_trace_fields(
    field: str, value: object, match: str
) -> None:
    row = _valid_row()
    row[field] = value

    with pytest.raises(ValueError, match=match):
        RoutingSignature.from_json(row)


def test_routing_signature_round_trip_and_structural_key_are_canonical() -> None:
    row = _valid_row()
    signature = RoutingSignature.from_json(row)
    expected_key = hashlib.sha256(
        json.dumps(
            _structural_row(row),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

    assert RoutingSignature.from_json(signature.to_json()) == signature
    assert signature.signature_key() == expected_key


def test_routing_signature_key_excludes_sampled_gpu_time() -> None:
    first = RoutingSignature.from_json(_valid_row())
    second_row = _valid_row()
    second_row["sampled_gpu_time_us"] = 32.0
    second = RoutingSignature.from_json(second_row)

    assert second.signature_key() == first.signature_key()


def test_tactic_pair_round_trip() -> None:
    pair = TacticPair(gemm1=64, gemm2=11)

    assert TacticPair.from_json(pair.to_json()) == pair


def test_tactic_measurement_round_trip() -> None:
    measurement = TacticMeasurement(
        signature_key="signature-sha256",
        tactic=TacticPair(gemm1=64, gemm2=11),
        median_us=4.0,
        p95_us=4.5,
        cv=0.02,
        warmups=3,
        repetitions=10,
        finite=True,
        deterministic=True,
        max_abs_error=0.0,
        cosine_similarity=1.0,
        failure=None,
    )

    assert TacticMeasurement.from_json(measurement.to_json()) == measurement


@pytest.mark.parametrize(
    "weight", [0.25, 0.5, 0.75]
)
def test_replay_profile_from_signature_round_trip(
    weight: float,
) -> None:
    signature = RoutingSignature.from_json(_valid_row())
    profile = ReplayProfile.from_signature(signature, weight=weight)

    assert profile.signature_key == signature.signature_key()
    assert profile.aggregate_gpu_time_us == signature.sampled_gpu_time_us
    assert profile.call_count == 1
    assert profile.normalized_weight == weight
    assert profile.skew_class == "median-skew"
    assert ReplayProfile.from_json(profile.to_json()) == profile
