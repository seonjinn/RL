#!/usr/bin/env python3
"""Gate MXFP8 MoE micro results and deterministic generation artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import string
import sys
from typing import cast

try:
    from .schema import TACTIC_MEASUREMENT_FIELDS, TacticMeasurement, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from schema import TACTIC_MEASUREMENT_FIELDS, TacticMeasurement, TacticPair


MIN_COSINE_SIMILARITY = 0.999
MAX_MXFP8_ABS_ERROR = 0.1
REQUIRED_STOCK_COMPARISONS = frozenset(
    {"fc1_activated_intermediate", "fc2_reduced_output"}
)
REQUIRED_REFERENCE_SKEW_CLASSES = frozenset({"balanced", "high-skew"})


@dataclass(frozen=True)
class CorrectnessSummary:
    """Promotion verdict for tactic-level numerical measurements."""

    passed: bool
    checked_tactics: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class GenerationComparison:
    """Exact-token comparison for two deterministic generation runs."""

    passed: bool
    compared_examples: int
    mismatched_ids: tuple[str, ...]


@dataclass(frozen=True)
class _GenerationRun:
    ordered_ids: tuple[str, ...]
    prompt_sha256: Mapping[str, str]
    token_ids: Mapping[str, tuple[int, ...]]
    provenance: Mapping[str, object]


def _measurement_label(measurement: TacticMeasurement) -> str:
    return (
        f"{measurement.signature_key}/"
        f"({measurement.tactic.gemm1},{measurement.tactic.gemm2})"
    )


def validate_micro(
    measurements: Sequence[TacticMeasurement],
) -> CorrectnessSummary:
    """Validate Task 6 measurements against promotion-blocking micro gates."""
    failures: list[str] = []
    seen: set[tuple[str, TacticPair]] = set()
    for measurement in measurements:
        label = _measurement_label(measurement)
        key = (measurement.signature_key, measurement.tactic)
        if key in seen:
            failures.append(f"{label}: duplicate tactic measurement")
        seen.add(key)

        numeric_values = (
            measurement.median_us,
            measurement.p95_us,
            measurement.cv,
            measurement.max_abs_error,
            measurement.cosine_similarity,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            failures.append(f"{label}: nonfinite metric or output")
            continue
        if measurement.failure is not None:
            if "intermediate_api_unavailable" in measurement.failure:
                failures.append(
                    f"{label}: missing FC1/FC2 stock comparison ({measurement.failure})"
                )
            else:
                failures.append(f"{label}: {measurement.failure}")
        if not measurement.finite:
            failures.append(f"{label}: nonfinite output")
        if not measurement.deterministic:
            failures.append(f"{label}: nondeterministic CUDA Graph replay")
        if measurement.warmups != 3 or measurement.repetitions < 10:
            failures.append(f"{label}: incomplete CUDA Graph replay evidence")
        if measurement.median_us <= 0 or measurement.p95_us <= 0:
            failures.append(f"{label}: nonpositive timing evidence")
        if measurement.cosine_similarity < MIN_COSINE_SIMILARITY:
            failures.append(
                f"{label}: cosine similarity {measurement.cosine_similarity} "
                f"is below {MIN_COSINE_SIMILARITY}"
            )
        if not 0 <= measurement.max_abs_error <= MAX_MXFP8_ABS_ERROR:
            failures.append(
                f"{label}: max_abs_error {measurement.max_abs_error} exceeds "
                f"the stock-relative MXFP8 bound {MAX_MXFP8_ABS_ERROR}"
            )

    if not measurements:
        failures.append("no tactic measurements were provided")
    return CorrectnessSummary(
        passed=not failures,
        checked_tactics=len(measurements),
        failures=tuple(failures),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read JSONL {path}: {error}") from error
    if not lines:
        raise ValueError(f"JSONL {path} must not be empty")
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"JSONL {path} has an empty line at {line_number}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSONL {path} line {line_number} is not valid JSON"
            ) from error
        rows.append(_require_mapping(raw, f"JSONL {path} line {line_number}"))
    return rows


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in string.hexdigits for character in value
    )


def _validate_generation_provenance(
    provenance: Mapping[str, object],
) -> None:
    for field_name in ("model_revision", "tokenizer_revision"):
        value = provenance.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"generation provenance missing {field_name}")
    runtime_fingerprint = provenance.get("runtime_fingerprint")
    if not (
        isinstance(runtime_fingerprint, str)
        and runtime_fingerprint
        or isinstance(runtime_fingerprint, Mapping)
        and runtime_fingerprint
    ):
        raise ValueError("generation provenance missing runtime_fingerprint")
    decoding = _require_mapping(provenance.get("decoding"), "provenance.decoding")
    if (
        decoding.get("mode") != "greedy"
        or decoding.get("temperature") != 0
        or decoding.get("top_p") != 1
    ):
        raise ValueError(
            "generation decoding must be greedy with temperature=0 and top_p=1"
        )
    seed = decoding.get("seed")
    max_tokens = decoding.get("max_tokens")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("generation decoding seed must be an integer")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("generation decoding max_tokens must be positive")


def _load_generation(path: Path) -> _GenerationRun:
    ordered_ids: list[str] = []
    prompt_sha256: dict[str, str] = {}
    token_ids: dict[str, tuple[int, ...]] = {}
    run_provenance: Mapping[str, object] | None = None
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        identifier = row.get("id")
        prompt_hash = row.get("prompt_sha256")
        raw_token_ids = row.get("token_ids")
        provenance = _require_mapping(
            row.get("provenance"), f"{path} line {line_number} provenance"
        )
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path} line {line_number} has no nonempty id")
        if identifier in token_ids:
            raise ValueError(f"{path} has duplicate id {identifier!r}")
        if not isinstance(prompt_hash, str) or not _is_sha256(prompt_hash):
            raise ValueError(f"{path} line {line_number} has invalid prompt_sha256")
        if not isinstance(raw_token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in raw_token_ids
        ):
            raise ValueError(f"{path} line {line_number} has invalid token_ids")
        if run_provenance is None:
            _validate_generation_provenance(provenance)
            run_provenance = provenance
        elif provenance != run_provenance:
            raise ValueError(f"{path} contains inconsistent generation provenance")
        ordered_ids.append(identifier)
        prompt_sha256[identifier] = prompt_hash
        token_ids[identifier] = tuple(cast(list[int], raw_token_ids))
    if run_provenance is None:
        raise ValueError(f"generation JSONL {path} must not be empty")
    return _GenerationRun(
        ordered_ids=tuple(ordered_ids),
        prompt_sha256=prompt_sha256,
        token_ids=token_ids,
        provenance=run_provenance,
    )


def compare_generations(stock: Path, candidate: Path) -> GenerationComparison:
    """Require identical provenance, prompts, IDs, and output token IDs."""
    stock_run = _load_generation(stock)
    candidate_run = _load_generation(candidate)
    if stock_run.provenance != candidate_run.provenance:
        raise ValueError("stock/candidate generation provenance mismatch")
    if set(stock_run.ordered_ids) != set(candidate_run.ordered_ids):
        raise ValueError("stock/candidate generation example IDs mismatch")
    mismatched_ids: list[str] = []
    for identifier in stock_run.ordered_ids:
        if (
            stock_run.prompt_sha256[identifier]
            != candidate_run.prompt_sha256[identifier]
        ):
            raise ValueError(f"stock/candidate prompt mismatch for id {identifier!r}")
        if stock_run.token_ids[identifier] != candidate_run.token_ids[identifier]:
            mismatched_ids.append(identifier)
    return GenerationComparison(
        passed=not mismatched_ids,
        compared_examples=len(stock_run.ordered_ids),
        mismatched_ids=tuple(mismatched_ids),
    )


def _load_measurements(path: Path) -> tuple[TacticMeasurement, ...]:
    measurements: list[TacticMeasurement] = []
    for row in _read_jsonl(path):
        measurement_row = {
            field_name: row[field_name]
            for field_name in TACTIC_MEASUREMENT_FIELDS
            if field_name in row
        }
        measurements.append(TacticMeasurement.from_json(measurement_row))
    return tuple(measurements)


def _load_json_object(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    return _require_mapping(raw, str(path))


def _evidence_failures(
    measurements: Sequence[TacticMeasurement], evidence_path: Path
) -> tuple[str, ...]:
    evidence = _load_json_object(evidence_path)
    raw_rows = evidence.get("measurement_evidence")
    if not isinstance(raw_rows, list):
        raise ValueError("evidence.measurement_evidence must be an array")
    evidence_by_key: dict[tuple[str, TacticPair], Mapping[str, object]] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"measurement_evidence[{index}]")
        signature_key = row.get("signature_key")
        if not isinstance(signature_key, str) or not signature_key:
            raise ValueError(f"measurement_evidence[{index}] has no signature_key")
        tactic = TacticPair.from_json(
            _require_mapping(row.get("tactic"), f"measurement_evidence[{index}].tactic")
        )
        key = (signature_key, tactic)
        if key in evidence_by_key:
            raise ValueError(
                f"duplicate measurement evidence for {signature_key}/{tactic}"
            )
        evidence_by_key[key] = row

    failures: list[str] = []
    for measurement in measurements:
        label = _measurement_label(measurement)
        row = evidence_by_key.get((measurement.signature_key, measurement.tactic))
        if row is None:
            failures.append(f"{label}: missing micro comparison evidence")
            continue
        if row.get("routing_counts_match") is not True:
            failures.append(f"{label}: routing count mismatch")
        raw_comparisons = row.get("stock_comparisons")
        if not isinstance(raw_comparisons, list) or not all(
            isinstance(comparison, str) for comparison in raw_comparisons
        ):
            failures.append(f"{label}: missing FC1/FC2 stock comparisons")
        elif not REQUIRED_STOCK_COMPARISONS.issubset(raw_comparisons):
            failures.append(f"{label}: missing FC1/FC2 stock comparisons")
        if row.get("within_upstream_mxfp8_bounds") is not True:
            failures.append(f"{label}: outside upstream MXFP8 MoE numerical bounds")

    raw_references = evidence.get("bf16_python_references")
    if not isinstance(raw_references, list):
        raise ValueError("evidence.bf16_python_references must be an array")
    valid_skew_classes: set[str] = set()
    for index, raw_reference in enumerate(raw_references):
        reference = _require_mapping(raw_reference, f"bf16_python_references[{index}]")
        skew_class = reference.get("skew_class")
        cosine = reference.get("cosine_similarity")
        max_error = reference.get("max_abs_error")
        if skew_class not in REQUIRED_REFERENCE_SKEW_CLASSES:
            continue
        if (
            reference.get("finite") is True
            and isinstance(cosine, (int, float))
            and not isinstance(cosine, bool)
            and math.isfinite(cosine)
            and cosine >= MIN_COSINE_SIMILARITY
            and isinstance(max_error, (int, float))
            and not isinstance(max_error, bool)
            and math.isfinite(max_error)
            and 0 <= max_error <= MAX_MXFP8_ABS_ERROR
            and reference.get("within_upstream_mxfp8_bounds") is True
        ):
            valid_skew_classes.add(cast(str, skew_class))
    for missing in sorted(REQUIRED_REFERENCE_SKEW_CLASSES - valid_skew_classes):
        failures.append(f"missing valid {missing} BF16/Python MoE reference evidence")
    return tuple(failures)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    micro_parser = subparsers.add_parser("micro", help="validate micro measurements")
    micro_parser.add_argument("--measurements", type=Path, required=True)
    micro_parser.add_argument("--evidence", type=Path, required=True)
    generation_parser = subparsers.add_parser(
        "generation", help="compare deterministic generation JSONL"
    )
    generation_parser.add_argument("--stock", type=Path, required=True)
    generation_parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "micro":
            measurements = _load_measurements(args.measurements)
            summary = validate_micro(measurements)
            evidence_failures = _evidence_failures(measurements, args.evidence)
            if evidence_failures:
                summary = CorrectnessSummary(
                    passed=False,
                    checked_tactics=summary.checked_tactics,
                    failures=summary.failures + evidence_failures,
                )
            result: CorrectnessSummary | GenerationComparison = summary
        else:
            result = compare_generations(args.stock, args.candidate)
    except ValueError as error:
        print(f"correctness gate error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), sort_keys=True, ensure_ascii=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
