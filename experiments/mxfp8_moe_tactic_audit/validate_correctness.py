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
from typing import Literal, cast

try:
    from .collect_results import comparison_run_bindings
    from .schema import TACTIC_MEASUREMENT_FIELDS, TacticMeasurement, TacticPair
except ImportError:  # pragma: no cover - direct script execution
    from collect_results import comparison_run_bindings
    from schema import TACTIC_MEASUREMENT_FIELDS, TacticMeasurement, TacticPair


MIN_COSINE_SIMILARITY = 0.999
MAX_MXFP8_ABS_ERROR = 0.1
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
class MicroMeasurementEvidence:
    """Evidence that one measured tactic preserved the Task 6 contracts."""

    signature_key: str
    tactic: TacticPair
    skew_class: Literal["balanced", "median-skew", "high-skew"]
    routing_counts_match: bool
    fc1_stock_compared: bool
    fc2_stock_compared: bool
    within_upstream_mxfp8_bounds: bool

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> MicroMeasurementEvidence:
        """Parse one measurement-evidence JSON object."""
        signature_key = row.get("signature_key")
        skew_class = row.get("skew_class")
        if not isinstance(signature_key, str) or not signature_key:
            raise ValueError("measurement evidence has no signature_key")
        if skew_class not in {"balanced", "median-skew", "high-skew"}:
            raise ValueError("measurement evidence has invalid skew_class")
        boolean_fields = (
            "routing_counts_match",
            "fc1_stock_compared",
            "fc2_stock_compared",
            "within_upstream_mxfp8_bounds",
        )
        if any(type(row.get(field_name)) is not bool for field_name in boolean_fields):
            raise ValueError("measurement evidence flags must be booleans")
        return cls(
            signature_key=signature_key,
            tactic=TacticPair.from_json(
                _require_mapping(row.get("tactic"), "measurement evidence tactic")
            ),
            skew_class=cast(
                Literal["balanced", "median-skew", "high-skew"], skew_class
            ),
            routing_counts_match=cast(bool, row["routing_counts_match"]),
            fc1_stock_compared=cast(bool, row["fc1_stock_compared"]),
            fc2_stock_compared=cast(bool, row["fc2_stock_compared"]),
            within_upstream_mxfp8_bounds=cast(
                bool, row["within_upstream_mxfp8_bounds"]
            ),
        )


@dataclass(frozen=True)
class Bf16PythonReferenceEvidence:
    """Final-output comparison against the upstream BF16/Python MoE reference."""

    signature_key: str
    tactic: TacticPair
    skew_class: Literal["balanced", "high-skew"]
    comparison_target: Literal["fc2_final"]
    finite: bool
    max_abs_error: float
    cosine_similarity: float
    within_upstream_mxfp8_bounds: bool

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> Bf16PythonReferenceEvidence:
        """Parse one BF16/Python reference-evidence JSON object."""
        signature_key = row.get("signature_key")
        skew_class = row.get("skew_class")
        comparison_target = row.get("comparison_target")
        if not isinstance(signature_key, str) or not signature_key:
            raise ValueError("BF16/Python reference has no signature_key")
        if skew_class not in REQUIRED_REFERENCE_SKEW_CLASSES:
            raise ValueError("BF16/Python reference has invalid skew_class")
        if comparison_target != "fc2_final":
            raise ValueError("BF16/Python comparison_target must be fc2_final")
        finite = row.get("finite")
        within_bounds = row.get("within_upstream_mxfp8_bounds")
        if type(finite) is not bool or type(within_bounds) is not bool:
            raise ValueError("BF16/Python reference flags must be booleans")
        max_abs_error = row.get("max_abs_error")
        cosine_similarity = row.get("cosine_similarity")
        for field_name, value in (
            ("max_abs_error", max_abs_error),
            ("cosine_similarity", cosine_similarity),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"BF16/Python reference {field_name} must be finite")
        return cls(
            signature_key=signature_key,
            tactic=TacticPair.from_json(
                _require_mapping(row.get("tactic"), "BF16/Python reference tactic")
            ),
            skew_class=cast(Literal["balanced", "high-skew"], skew_class),
            comparison_target="fc2_final",
            finite=cast(bool, finite),
            max_abs_error=float(cast(float | int, max_abs_error)),
            cosine_similarity=float(cast(float | int, cosine_similarity)),
            within_upstream_mxfp8_bounds=cast(bool, within_bounds),
        )


@dataclass(frozen=True)
class MicroCorrectnessEvidence:
    """Complete evidence required by the authoritative micro gate."""

    measurement_evidence: tuple[MicroMeasurementEvidence, ...]
    bf16_python_references: tuple[Bf16PythonReferenceEvidence, ...]

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> MicroCorrectnessEvidence:
        """Parse complete micro-correctness evidence."""
        raw_measurements = row.get("measurement_evidence")
        raw_references = row.get("bf16_python_references")
        if not isinstance(raw_measurements, list):
            raise ValueError("evidence.measurement_evidence must be an array")
        if not isinstance(raw_references, list):
            raise ValueError("evidence.bf16_python_references must be an array")
        return cls(
            measurement_evidence=tuple(
                MicroMeasurementEvidence.from_json(
                    _require_mapping(item, f"measurement_evidence[{index}]")
                )
                for index, item in enumerate(raw_measurements)
            ),
            bf16_python_references=tuple(
                Bf16PythonReferenceEvidence.from_json(
                    _require_mapping(item, f"bf16_python_references[{index}]")
                )
                for index, item in enumerate(raw_references)
            ),
        )


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
    evidence: MicroCorrectnessEvidence | None = None,
) -> CorrectnessSummary:
    """Validate Task 6 measurements against promotion-blocking micro gates."""
    failures: list[str] = []
    seen: set[tuple[str, TacticPair]] = set()
    successful_measurements: set[tuple[str, TacticPair]] = set()
    for measurement in measurements:
        label = _measurement_label(measurement)
        key = (measurement.signature_key, measurement.tactic)
        failure_count = len(failures)
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
        if len(failures) == failure_count:
            successful_measurements.add(key)

    if not measurements:
        failures.append("no tactic measurements were provided")
    if evidence is None:
        failures.append("micro correctness evidence is required")
        return CorrectnessSummary(
            passed=False,
            checked_tactics=len(measurements),
            failures=tuple(failures),
        )

    measured_keys = {(row.signature_key, row.tactic) for row in measurements}
    evidence_by_key: dict[tuple[str, TacticPair], MicroMeasurementEvidence] = {}
    valid_evidence_keys: set[tuple[str, TacticPair]] = set()
    for row in evidence.measurement_evidence:
        key = (row.signature_key, row.tactic)
        label = f"{row.signature_key}/({row.tactic.gemm1},{row.tactic.gemm2})"
        if key in evidence_by_key:
            failures.append(f"{label}: duplicate micro comparison evidence")
            continue
        evidence_by_key[key] = row
        if key not in measured_keys:
            failures.append(f"{label}: evidence has no matching measured tactic")

    for measurement in measurements:
        key = (measurement.signature_key, measurement.tactic)
        label = _measurement_label(measurement)
        row = evidence_by_key.get(key)
        if row is None:
            failures.append(f"{label}: missing micro comparison evidence")
            continue
        row_is_valid = True
        if row.routing_counts_match is not True:
            failures.append(f"{label}: routing count mismatch")
            row_is_valid = False
        if row.fc1_stock_compared is not True:
            failures.append(f"{label}: missing FC1 stock comparison")
            row_is_valid = False
        if row.fc2_stock_compared is not True:
            failures.append(f"{label}: missing FC2 stock comparison")
            row_is_valid = False
        if row.within_upstream_mxfp8_bounds is not True:
            failures.append(f"{label}: outside upstream MXFP8 MoE numerical bounds")
            row_is_valid = False
        if row_is_valid and key in successful_measurements:
            valid_evidence_keys.add(key)

    valid_reference_skew_classes: set[str] = set()
    for reference in evidence.bf16_python_references:
        key = (reference.signature_key, reference.tactic)
        row = evidence_by_key.get(key)
        label = (
            f"{reference.signature_key}/"
            f"({reference.tactic.gemm1},{reference.tactic.gemm2})"
        )
        reference_is_valid = True
        if key not in valid_evidence_keys or row is None:
            failures.append(
                f"{label}: BF16/Python reference is not bound to a successful "
                "measured tactic"
            )
            reference_is_valid = False
        elif row.skew_class != reference.skew_class:
            failures.append(
                f"{label}: BF16/Python reference skew_class does not match "
                "measurement evidence"
            )
            reference_is_valid = False
        if reference.comparison_target != "fc2_final":
            failures.append(
                f"{label}: BF16/Python reference comparison_target must be fc2_final"
            )
            reference_is_valid = False
        if (
            reference.finite is not True
            or not math.isfinite(reference.max_abs_error)
            or not 0 <= reference.max_abs_error <= MAX_MXFP8_ABS_ERROR
            or not math.isfinite(reference.cosine_similarity)
            or reference.cosine_similarity < MIN_COSINE_SIMILARITY
            or reference.within_upstream_mxfp8_bounds is not True
        ):
            failures.append(f"{label}: BF16/Python reference failed numerical gates")
            reference_is_valid = False
        if reference_is_valid:
            valid_reference_skew_classes.add(reference.skew_class)
    for missing in sorted(
        REQUIRED_REFERENCE_SKEW_CLASSES - valid_reference_skew_classes
    ):
        failures.append(f"missing valid {missing} BF16/Python MoE reference evidence")
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


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
    if decoding.get("mode") != "greedy":
        raise ValueError("generation decoding mode must be greedy")
    for field_name, expected in (("temperature", 0), ("top_p", 1)):
        value = decoding.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value != expected
        ):
            raise ValueError(
                f"generation decoding {field_name} must be numeric {expected}"
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
        elif _canonical_json(provenance) != _canonical_json(run_provenance):
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
    if _canonical_json(stock_run.provenance) != _canonical_json(
        candidate_run.provenance
    ):
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
    generation_parser.add_argument("--stock-run-root", type=Path, action="append")
    generation_parser.add_argument("--candidate-run-root", type=Path, action="append")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "micro":
            measurements = _load_measurements(args.measurements)
            evidence = MicroCorrectnessEvidence.from_json(
                _load_json_object(args.evidence)
            )
            result: CorrectnessSummary | GenerationComparison = validate_micro(
                measurements, evidence
            )
        else:
            result = compare_generations(args.stock, args.candidate)
    except ValueError as error:
        print(f"correctness gate error: {error}", file=sys.stderr)
        return 2
    payload = asdict(result)
    if args.command == "generation":
        try:
            stock_bindings = tuple(args.stock_run_root or (args.stock,))
            candidate_bindings = tuple(args.candidate_run_root or (args.candidate,))
            payload.update(comparison_run_bindings(stock_bindings, candidate_bindings))
        except ValueError as error:
            print(f"correctness gate error: {error}", file=sys.stderr)
            return 2
        payload["deterministic_generation"] = result.passed
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
