"""Aggregate MXFP8 MoE routing traces into deterministic replay profiles."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, cast

if __package__:
    from .schema import ReplayProfile, RoutingSignature
else:
    from schema import ReplayProfile, RoutingSignature


SkewClass = Literal["balanced", "median-skew", "high-skew"]


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
        with path.open(encoding="ascii") as trace_file:
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
    total_gpu_time_us = math.fsum(item.aggregate_gpu_time_us for item in ordered_observed)
    if not math.isfinite(total_gpu_time_us) or total_gpu_time_us <= 0:
        raise ValueError("total GPU time must be finite and positive")

    selected: list[ReplayProfile] = []
    selected_gpu_time_us = 0.0
    for item in ordered_observed:
        selected.append(_replay_profile(item, total_gpu_time_us=total_gpu_time_us))
        selected_gpu_time_us += item.aggregate_gpu_time_us
        if selected_gpu_time_us / total_gpu_time_us >= coverage:
            break

    covered_weight = selected_gpu_time_us / total_gpu_time_us
    if covered_weight < coverage:
        raise ValueError("achieved coverage is below the requested threshold")
    return ProfileSelection(
        selected=tuple(selected),
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
