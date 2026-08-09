import json
from dataclasses import replace
from itertools import product
import math
from pathlib import Path

import pytest

import experiments.mxfp8_moe_tactic_audit.select_profiles as profile_selection
from experiments.mxfp8_moe_tactic_audit.schema import RoutingSignature
from experiments.mxfp8_moe_tactic_audit.select_profiles import (
    ObservedSignature,
    aggregate_signatures,
    main,
    select_representative_profiles,
    select_profiles,
)


def _signature(
    *,
    expert_counts: tuple[int, ...] = (2, 2, 2, 2),
    model_revision: str = "qwen3-30ba3b-test",
    sampled_gpu_time_us: float = 1.0,
    runtime_fingerprint: str = "runtime-a",
) -> RoutingSignature:
    return RoutingSignature(
        schema_version=1,
        model_revision=model_revision,
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


def _expected_skew_class(expert_counts: tuple[int, ...]) -> str:
    total = sum(expert_counts)
    entropy = -sum(
        probability * math.log(probability)
        for count in expert_counts
        if count > 0
        for probability in (count / total,)
    )
    normalized_entropy = entropy / math.log(len(expert_counts))
    if normalized_entropy >= 0.90:
        return "balanced"
    if normalized_entropy < 0.65:
        return "high-skew"
    return "median-skew"


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


def test_select_profiles_preserves_sole_skew_representative_after_coverage() -> None:
    all_signatures = [
        _signature(expert_counts=counts)
        for counts in product(range(9), repeat=4)
        if sum(counts) == 8
    ]
    high_skew = max(
        (
            signature
            for signature in all_signatures
            if _expected_skew_class(signature.expert_counts) == "high-skew"
        ),
        key=RoutingSignature.signature_key,
    )
    non_high_skew = sorted(
        (
            signature
            for signature in all_signatures
            if _expected_skew_class(signature.expert_counts) != "high-skew"
            and signature.signature_key() < high_skew.signature_key()
        ),
        key=RoutingSignature.signature_key,
    )
    observed = [_observed(signature, weight=1.0) for signature in non_high_skew[:19]]
    observed.append(_observed(high_skew, weight=1.0))

    selection = select_profiles(observed, coverage=0.95)

    assert high_skew.signature_key() in [item.signature_key for item in selection.selected]
    assert {item.skew_class for item in selection.selected} >= {"high-skew"}
    assert [item.signature_key for item in selection.selected] == sorted(
        item.signature_key for item in selection.selected
    )


def test_select_profiles_uses_precision_safe_full_coverage() -> None:
    observed = [
        _observed(_signature(expert_counts=counts), weight=weight)
        for counts, weight in (
            ((2, 2, 2, 2), 0.7),
            ((4, 2, 1, 1), 0.2),
            ((7, 1, 0, 0), 0.1),
        )
    ]

    selection = select_profiles(observed, coverage=1.0)

    assert selection.covered_weight == 1.0
    assert len(selection.selected) == 3


def test_select_profiles_requires_strict_requested_coverage() -> None:
    coverage = 0.95
    one_ulp_below = math.nextafter(coverage, 0.0)
    first = _signature(expert_counts=(2, 2, 2, 2))
    second = _signature(expert_counts=(2, 2, 1, 3))
    observed = [
        _observed(first, weight=one_ulp_below),
        _observed(second, weight=1.0 - one_ulp_below),
    ]

    selection = select_profiles(observed, coverage=coverage)

    assert selection.covered_weight >= coverage
    assert [item.signature_key for item in selection.selected] == [
        first.signature_key(),
        second.signature_key(),
    ]


def test_select_profiles_uses_single_pass_weight_accumulation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = [
        _observed(
            _signature(model_revision=f"qwen3-30ba3b-test-{index}"),
            weight=1.0,
        )
        for index in range(500)
    ]

    def fail_repeated_prefix_scan(_: object) -> float:
        raise AssertionError("selection must not repeatedly rescan prefix weights")

    monkeypatch.setattr(profile_selection.math, "fsum", fail_repeated_prefix_scan)

    selection = select_profiles(observed, coverage=0.95)

    assert selection.covered_weight >= 0.95


def test_representative_selection_bounds_unique_histograms_per_workload_bucket() -> None:
    signatures = [
        _signature(
            expert_counts=(first, second, third, 8 - first - second - third)
        )
        for first in range(9)
        for second in range(9 - first)
        for third in range(9 - first - second)
    ]
    observed = [
        _observed(signature, weight=float(index + 1))
        for index, signature in enumerate(signatures)
    ]

    selection = select_representative_profiles(observed, coverage=0.95)

    assert len(selection.selected) <= 3
    assert selection.covered_weight == pytest.approx(1.0)
    assert sum(profile.call_count for profile in selection.selected) == len(observed)
    assert sum(
        profile.aggregate_gpu_time_us for profile in selection.selected
    ) == pytest.approx(sum(item.aggregate_gpu_time_us for item in observed))


def test_representative_selection_keeps_only_high_weight_workload_buckets() -> None:
    heavy = _observed(_signature(sampled_gpu_time_us=95.0), weight=95.0)
    light_signature = replace(
        _signature(sampled_gpu_time_us=5.0),
        num_tokens=8,
        expert_counts=(4, 4, 4, 4),
    )
    light = _observed(light_signature, weight=5.0)

    selection = select_representative_profiles([heavy, light], coverage=0.95)

    assert [profile.signature.num_tokens for profile in selection.selected] == [4]
    assert selection.covered_weight == pytest.approx(0.95)


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


def test_aggregate_signatures_reads_utf8_jsonl(tmp_path: Path) -> None:
    trace = tmp_path / "rank-0.jsonl"
    signature = _signature(
        model_revision="qwen3-30b-caf\u00e9",
        runtime_fingerprint="runtime-\u00e9",
    )
    trace.write_text(
        json.dumps(signature.to_json(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    observed = aggregate_signatures([trace])

    assert observed[0].signature.model_revision == "qwen3-30b-caf\u00e9"
    assert observed[0].signature.runtime_fingerprint == "runtime-\u00e9"


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
