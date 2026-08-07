"""Aggregate MXFP8 MoE routing traces into deterministic replay profiles."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Literal, cast

if __package__:
    from .schema import ReplayProfile, RoutingSignature
else:
    from schema import ReplayProfile, RoutingSignature


SkewClass = Literal["balanced", "median-skew", "high-skew"]
WorkloadBucket = tuple[int | str, ...]
Weight = Fraction


@dataclass(frozen=True)
class ObservedSignature:
    """One structural routing signature aggregated across trace rows."""

    signature: RoutingSignature
    signature_key: str
    call_count: int
    aggregate_gpu_time_us: float

    def __post_init__(self) -> None:
        """Validate that aggregation remains tied to one structural signature."""
        if not isinstance(self.signature, RoutingSignature):
            raise ValueError("signature must be a RoutingSignature")
        if self.signature_key != self.signature.signature_key():
            raise ValueError("signature_key must match signature")
        if isinstance(self.call_count, bool) or not isinstance(self.call_count, int) or self.call_count <= 0:
            raise ValueError("call_count must be positive")
        if not math.isfinite(self.aggregate_gpu_time_us) or self.aggregate_gpu_time_us <= 0:
            raise ValueError("aggregate_gpu_time_us must be finite and positive")


@dataclass(frozen=True)
class ProfileSelection:
    """Selected replay profiles and the complete observed routing population."""

    selected: tuple[ReplayProfile, ...]
    all_observed: tuple[ObservedSignature, ...]
    covered_weight: float
    total_gpu_time_us: float

    def __post_init__(self) -> None:
        """Validate the selection coverage summary."""
        if not self.all_observed:
            raise ValueError("all_observed must not be empty")
        if not math.isfinite(self.total_gpu_time_us) or self.total_gpu_time_us <= 0:
            raise ValueError("total_gpu_time_us must be finite and positive")
        if not math.isfinite(self.covered_weight) or not 0 < self.covered_weight <= 1:
            raise ValueError("covered_weight must be in (0, 1]")


def _read_signature_rows(paths: Sequence[Path]) -> list[RoutingSignature]:
    """Parse every trace row and reject missing or malformed trace input."""
    if not paths:
        raise ValueError("at least one trace path is required")

    signatures: list[RoutingSignature] = []
    for path in paths:
        with path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                if not line.strip():
                    raise ValueError(f"blank trace row in {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed JSON trace row in {path}:{line_number}") from error
                if not isinstance(row, Mapping):
                    raise ValueError(f"trace row in {path}:{line_number} must be an object")
                signatures.append(RoutingSignature.from_json(cast(Mapping[str, object], row)))

    if not signatures:
        raise ValueError("no trace rows were found")
    return signatures


def aggregate_signatures(paths: Sequence[Path]) -> list[ObservedSignature]:
    """Aggregate exact structural signatures by their sampled GPU execution time.

    Args:
        paths: Rank-local JSONL trace paths.

    Returns:
        Observed signatures, sorted by descending aggregate GPU time then key.

    Raises:
        ValueError: The traces are empty, malformed, have invalid timings, or
            were collected from more than one runtime fingerprint.
    """
    signatures = _read_signature_rows(paths)
    runtime_fingerprints = {signature.runtime_fingerprint for signature in signatures}
    if len(runtime_fingerprints) != 1:
        raise ValueError("trace rows have mixed runtime fingerprints")

    aggregated: dict[str, ObservedSignature] = {}
    for signature in signatures:
        signature_key = signature.signature_key()
        previous = aggregated.get(signature_key)
        if previous is None:
            aggregated[signature_key] = ObservedSignature(
                signature=signature,
                signature_key=signature_key,
                call_count=1,
                aggregate_gpu_time_us=signature.sampled_gpu_time_us,
            )
        else:
            aggregated[signature_key] = ObservedSignature(
                signature=previous.signature,
                signature_key=signature_key,
                call_count=previous.call_count + 1,
                aggregate_gpu_time_us=(
                    previous.aggregate_gpu_time_us + signature.sampled_gpu_time_us
                ),
            )

    return sorted(
        aggregated.values(),
        key=lambda item: (-item.aggregate_gpu_time_us, item.signature_key),
    )


def _classify_skew(signature: RoutingSignature) -> SkewClass:
    """Classify routing balance from normalized expert-count entropy."""
    counts = signature.expert_counts
    if len(counts) == 1:
        return "balanced"

    total_count = sum(counts)
    entropy = -sum(
        probability * math.log(probability)
        for count in counts
        if count > 0
        for probability in (count / total_count,)
    )
    normalized_entropy = entropy / math.log(len(counts))
    if normalized_entropy >= 0.90:
        return "balanced"
    if normalized_entropy < 0.65:
        return "high-skew"
    return "median-skew"


def _workload_bucket_key(signature: RoutingSignature) -> WorkloadBucket:
    """Return non-routing execution dimensions that define one workload bucket."""
    return (
        signature.schema_version,
        signature.model_revision,
        signature.layer_family,
        signature.num_tokens,
        signature.global_num_experts,
        signature.local_num_experts,
        signature.top_k,
        signature.hidden_size,
        signature.intermediate_size,
        signature.tp_size,
        signature.ep_size,
        signature.dp_size,
        signature.cuda_graph_state,
        signature.weight_layout,
        signature.quantization,
        signature.runtime_fingerprint,
    )


def _weight(value: float) -> Weight:
    """Return the exact rational value of a finite binary float."""
    return Fraction.from_float(float(value))


def _bucket_weights(observed: Sequence[ObservedSignature]) -> dict[WorkloadBucket, Weight]:
    """Aggregate observed GPU time by non-routing workload bucket."""
    bucket_weights: dict[WorkloadBucket, Weight] = {}
    for item in observed:
        bucket = _workload_bucket_key(item.signature)
        bucket_weights[bucket] = bucket_weights.get(bucket, Fraction()) + _weight(
            item.aggregate_gpu_time_us
        )
    return bucket_weights


def _meets_coverage(
    covered_gpu_time: Weight, total_gpu_time: Weight, coverage: float
) -> bool:
    """Return whether the reported float coverage meets the strict threshold."""
    return float(covered_gpu_time / total_gpu_time) >= coverage


def _high_weight_buckets(
    bucket_weights: Mapping[WorkloadBucket, Weight], coverage: float
) -> set[WorkloadBucket]:
    """Return buckets whose aggregate GPU time covers the requested fraction."""
    ordered_buckets = sorted(
        bucket_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )
    total_gpu_time = sum((weight for _, weight in ordered_buckets), Fraction())
    selected_buckets: set[WorkloadBucket] = set()
    selected_gpu_time = Fraction()
    for bucket, weight in ordered_buckets:
        selected_buckets.add(bucket)
        selected_gpu_time += weight
        if _meets_coverage(selected_gpu_time, total_gpu_time, coverage):
            break
    return selected_buckets


def _replay_profile(
    observed: ObservedSignature, *, total_gpu_time_us: float
) -> ReplayProfile:
    """Convert one aggregate observation to a weighted replay profile."""
    return ReplayProfile(
        signature=observed.signature,
        signature_key=observed.signature_key,
        aggregate_gpu_time_us=observed.aggregate_gpu_time_us,
        call_count=observed.call_count,
        normalized_weight=observed.aggregate_gpu_time_us / total_gpu_time_us,
        skew_class=_classify_skew(observed.signature),
    )


def select_profiles(
    observed: Sequence[ObservedSignature], coverage: float = 0.95
) -> ProfileSelection:
    """Select highest-weight replay profiles until requested GPU-time coverage.

    Args:
        observed: Exact structural-signature aggregates.
        coverage: Required fraction of aggregate sampled GPU time to cover.

    Returns:
        A deterministic coverage selection and all raw observed signatures.

    Raises:
        ValueError: The observations are empty or the requested coverage cannot
            be achieved from valid GPU-time weights.
    """
    if not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage must be finite and in (0, 1]")
    if not observed:
        raise ValueError("observed signatures must not be empty")

    ordered_observed = tuple(
        sorted(
            observed,
            key=lambda item: (-item.aggregate_gpu_time_us, item.signature_key),
        )
    )
    observed_weights = tuple(_weight(item.aggregate_gpu_time_us) for item in ordered_observed)
    total_gpu_time = sum(observed_weights, Fraction())
    total_gpu_time_us = float(total_gpu_time)
    if not math.isfinite(total_gpu_time_us) or total_gpu_time_us <= 0:
        raise ValueError("total GPU time must be finite and positive")

    coverage_observed: list[ObservedSignature] = []
    coverage_gpu_time = Fraction()
    for item, item_weight in zip(ordered_observed, observed_weights, strict=True):
        coverage_observed.append(item)
        coverage_gpu_time += item_weight
        if _meets_coverage(coverage_gpu_time, total_gpu_time, coverage):
            break

    qualifying_buckets = _high_weight_buckets(_bucket_weights(ordered_observed), coverage)
    selected_by_key = {item.signature_key: item for item in coverage_observed}
    representatives: dict[tuple[WorkloadBucket, SkewClass], ObservedSignature] = {}
    for item in ordered_observed:
        bucket = _workload_bucket_key(item.signature)
        if bucket in qualifying_buckets:
            representatives.setdefault((bucket, _classify_skew(item.signature)), item)
    selected_by_key.update(
        {item.signature_key: item for item in representatives.values()}
    )
    selected_observed = sorted(
        selected_by_key.values(),
        key=lambda item: (-item.aggregate_gpu_time_us, item.signature_key),
    )
    selected_gpu_time = sum(
        (_weight(item.aggregate_gpu_time_us) for item in selected_observed),
        Fraction(),
    )
    covered_weight = float(selected_gpu_time / total_gpu_time)
    if not _meets_coverage(selected_gpu_time, total_gpu_time, coverage):
        raise ValueError("achieved coverage is below the requested threshold")
    return ProfileSelection(
        selected=tuple(
            _replay_profile(item, total_gpu_time_us=total_gpu_time_us)
            for item in selected_observed
        ),
        all_observed=ordered_observed,
        covered_weight=covered_weight,
        total_gpu_time_us=total_gpu_time_us,
    )


def _all_observed_payload(selection: ProfileSelection) -> list[dict[str, object]]:
    """Serialize every raw signature with its aggregate GPU-time weight."""
    return [
        _replay_profile(
            observed,
            total_gpu_time_us=selection.total_gpu_time_us,
        ).to_json()
        for observed in selection.all_observed
    ]


def _write_selection(path: Path, selection: ProfileSelection) -> None:
    """Write the deterministic selection artifact as ASCII JSON."""
    payload = {
        "all_observed": _all_observed_payload(selection),
        "covered_weight": selection.covered_weight,
        "selected_profiles": [profile.to_json() for profile in selection.selected],
        "total_gpu_time_us": selection.total_gpu_time_us,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for profile selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate trace files and write selected profiles."""
    args = _parse_args(argv)
    try:
        trace_paths = sorted(args.trace_dir.glob("*.jsonl"))
        selection = select_profiles(aggregate_signatures(trace_paths), coverage=args.coverage)
        _write_selection(args.output, selection)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
