import json
import math
from pathlib import Path

import pytest

from experiments.mxfp8_moe_tactic_audit.schema import RoutingSignature
from experiments.mxfp8_moe_tactic_audit.select_profiles import (
    ObservedSignature,
    aggregate_signatures,
    main,
    select_profiles,
)


def _signature(
    *,
    expert_counts: tuple[int, ...] = (2, 2, 2, 2),
    sampled_gpu_time_us: float = 1.0,
    runtime_fingerprint: str = "runtime-a",
) -> RoutingSignature:
    return RoutingSignature(
        schema_version=1,
        model_revision="qwen3-30ba3b-test",
        layer_family="routed_experts",
        num_tokens=4,
        global_num_experts=4,
        local_num_experts=4,
        top_k=2,
        hidden_size=2048,
        intermediate_size=768,
        expert_counts=expert_counts,
        sampled_gpu_time_us=sampled_gpu_time_us,
        tp_size=1,
        ep_size=1,
        dp_size=16,
        cuda_graph_state="trace-eager",
        weight_layout="MajorK",
        quantization="MXFP8",
        runtime_fingerprint=runtime_fingerprint,
    )


def _observed(
    signature: RoutingSignature, *, weight: float, call_count: int = 1
) -> ObservedSignature:
    return ObservedSignature(
        signature=signature,
        signature_key=signature.signature_key(),
        call_count=call_count,
        aggregate_gpu_time_us=weight,
    )


def _write_trace(path: Path, signatures: list[RoutingSignature]) -> None:
    path.write_text(
        "\n".join(json.dumps(signature.to_json(), sort_keys=True) for signature in signatures)
        + "\n",
        encoding="ascii",
    )


def test_aggregate_signatures_merges_exact_structural_keys_by_gpu_time(tmp_path: Path) -> None:
    first = _signature(sampled_gpu_time_us=20.0)
    same_structure = _signature(sampled_gpu_time_us=30.0)
    distinct = _signature(expert_counts=(5, 1, 1, 1), sampled_gpu_time_us=10.0)
    trace = tmp_path / "rank-0.jsonl"
    _write_trace(trace, [first, same_structure, distinct])

    observed = aggregate_signatures([trace])

    assert [(item.signature_key, item.call_count, item.aggregate_gpu_time_us) for item in observed] == [
        (first.signature_key(), 2, 50.0),
        (distinct.signature_key(), 1, 10.0),
    ]


def test_select_profiles_covers_exactly_95_percent_in_weight_then_key_order() -> None:
    signatures = [
        _signature(expert_counts=(2, 2, 2, 2)),
        _signature(expert_counts=(4, 2, 1, 1)),
        _signature(expert_counts=(7, 1, 0, 0)),
        _signature(expert_counts=(3, 3, 1, 1)),
    ]
    observed = [
        _observed(signature, weight=weight)
        for signature, weight in zip(signatures, (50.0, 30.0, 15.0, 5.0), strict=True)
    ]

    selection = select_profiles(observed, coverage=0.95)

    assert selection.covered_weight == pytest.approx(0.95)
    assert [item.signature_key for item in selection.selected] == [
        signatures[0].signature_key(),
        signatures[1].signature_key(),
        signatures[2].signature_key(),
    ]
    assert [item.normalized_weight for item in selection.selected] == [0.5, 0.3, 0.15]


def test_select_profiles_preserves_entropy_class_representatives_in_high_weight_bucket() -> None:
    balanced = _signature(expert_counts=(2, 2, 2, 2))
    median_skew = _signature(expert_counts=(4, 2, 1, 1))
    high_skew = _signature(expert_counts=(7, 1, 0, 0))
    lower_weight = _signature(expert_counts=(3, 3, 1, 1))
    observed = [
        _observed(signature, weight=weight)
        for signature, weight in (
            (balanced, 30.0),
            (median_skew, 30.0),
            (high_skew, 30.0),
            (lower_weight, 10.0),
        )
    ]

    selection = select_profiles(observed, coverage=0.95)

    assert {item.skew_class for item in selection.selected} >= {
        "balanced",
        "median-skew",
        "high-skew",
    }
    assert {item.signature_key for item in selection.selected} >= {
        balanced.signature_key(),
        median_skew.signature_key(),
        high_skew.signature_key(),
    }


@pytest.mark.parametrize(
    "paths, match",
    [
        ([], "at least one trace"),
        (["empty.jsonl"], "no trace rows"),
    ],
)
def test_aggregate_signatures_rejects_empty_traces(
    tmp_path: Path, paths: list[str], match: str
) -> None:
    for path in paths:
        (tmp_path / path).touch()

    with pytest.raises(ValueError, match=match):
        aggregate_signatures([tmp_path / path for path in paths])


def test_aggregate_signatures_rejects_mixed_runtime_fingerprints(tmp_path: Path) -> None:
    trace = tmp_path / "rank-0.jsonl"
    _write_trace(
        trace,
        [
            _signature(runtime_fingerprint="runtime-a"),
            _signature(runtime_fingerprint="runtime-b"),
        ],
    )

    with pytest.raises(ValueError, match="runtime fingerprints"):
        aggregate_signatures([trace])


@pytest.mark.parametrize("invalid_time", [0.0, -1.0, math.inf, math.nan])
def test_aggregate_signatures_rejects_invalid_sampled_gpu_time(
    tmp_path: Path, invalid_time: float
) -> None:
    trace = tmp_path / "rank-0.jsonl"
    row = _signature().to_json()
    row["sampled_gpu_time_us"] = invalid_time
    trace.write_text(json.dumps(row) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="sampled_gpu_time_us"):
        aggregate_signatures([trace])


def test_select_profiles_rejects_achieved_coverage_below_requested_threshold() -> None:
    signature = _signature()
    observed = [_observed(signature, weight=1.0)]

    with pytest.raises(ValueError, match="coverage"):
        select_profiles(observed, coverage=1.01)


def test_main_writes_deterministic_selected_profiles_json(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    first = _signature(sampled_gpu_time_us=20.0)
    duplicate = _signature(sampled_gpu_time_us=30.0)
    distinct = _signature(expert_counts=(5, 1, 1, 1), sampled_gpu_time_us=10.0)
    _write_trace(trace_dir / "rank-1.jsonl", [distinct])
    _write_trace(trace_dir / "rank-0.jsonl", [first, duplicate])
    output = tmp_path / "selected_profiles.json"

    assert main(["--trace-dir", str(trace_dir), "--coverage", "0.95", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["covered_weight"] == pytest.approx(1.0)
    assert payload["total_gpu_time_us"] == 60.0
    assert [item["signature_key"] for item in payload["selected_profiles"]] == [
        first.signature_key(),
        distinct.signature_key(),
    ]
    assert [(item["call_count"], item["normalized_weight"]) for item in payload["all_observed"]] == [
        (2, pytest.approx(50.0 / 60.0)),
        (1, pytest.approx(10.0 / 60.0)),
    ]
