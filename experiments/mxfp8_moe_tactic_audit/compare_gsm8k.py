#!/usr/bin/env python3
"""Compare two provenance-matched full GSM8K evaluation result directories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import sys
from typing import cast

try:
    from .collect_results import comparison_artifact_bindings
except ImportError:  # pragma: no cover - direct script execution
    from collect_results import comparison_artifact_bindings


DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
EXPECTED_TOTAL = 1_319
BOOTSTRAP_SEED = 20260807
BOOTSTRAP_SAMPLES = 10_000
RESULTS_FILENAME = "results.json"
PER_EXAMPLE_FILENAME = "per_example.jsonl"


@dataclass(frozen=True)
class PairedGsm8kComparison:
    """Paired correctness and statistical-regression verdict."""

    stock_accuracy: float
    candidate_accuracy: float
    candidate_only_wins: int
    stock_only_wins: int
    mcnemar_p_value: float
    delta_ci95: tuple[float, float]
    passed: bool
    both_correct: int = 0
    both_wrong: int = 0
    accuracy_delta: float = 0.0
    provenance_matched: bool = False
    matched_examples: int = 0
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    paired_outcomes: tuple[int, ...] = ()
    paired_outcomes_sha256: str = ""


@dataclass(frozen=True)
class _Gsm8kRun:
    correct_by_id: Mapping[str, bool]
    provenance: Mapping[str, object]
    model_revision: str
    tokenizer_revision: str
    generation_args: Mapping[str, object]
    runtime_fingerprint: object
    evaluator_contract: Mapping[str, object]


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _load_json_object(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    return _require_mapping(raw, str(path))


def _canonical_sha256(value: object) -> str:
    encoded = _canonical_json(value).encode("ascii")
    return sha256(encoded).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _contract_field(
    provenance: Mapping[str, object],
    contract: Mapping[str, object],
    field_name: str,
) -> object:
    if field_name in contract:
        return contract[field_name]
    if field_name in provenance:
        return provenance[field_name]
    raise ValueError(f"GSM8K provenance missing {field_name}")


def _validate_generation_args(
    generation_args: Mapping[str, object], evaluator: Mapping[str, object]
) -> None:
    if generation_args.get("mode") != "greedy":
        raise ValueError("GSM8K generation_args.mode must equal 'greedy'")
    for field_name, expected in (("temperature", 0), ("top_p", 1)):
        value = generation_args.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value != expected
        ):
            raise ValueError(
                f"GSM8K generation_args.{field_name} must be numeric {expected}"
            )
    seed = generation_args.get("seed")
    max_tokens = generation_args.get("max_tokens")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("GSM8K generation_args.seed must be an integer")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("GSM8K generation_args.max_tokens must be positive")
    for field_name in ("temperature", "top_p", "seed", "max_tokens"):
        evaluator_value = evaluator.get(field_name)
        if isinstance(evaluator_value, bool) or not isinstance(
            evaluator_value, (int, float)
        ):
            raise ValueError(f"GSM8K evaluator {field_name} must be numeric")
        if _canonical_json(evaluator_value) != _canonical_json(
            generation_args.get(field_name)
        ):
            raise ValueError(
                f"GSM8K evaluator {field_name} disagrees with generation arguments"
            )


def _load_per_example(path: Path, provenance_sha256: str) -> dict[str, bool]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read GSM8K rows {path}: {error}") from error
    if len(lines) != EXPECTED_TOTAL:
        raise ValueError(f"GSM8K result must contain exactly {EXPECTED_TOTAL} rows")
    correct_by_id: dict[str, bool] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"GSM8K rows contain an empty line at {line_number}")
        try:
            row = _require_mapping(json.loads(line), f"{path} line {line_number}")
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {line_number} is not valid JSON") from error
        identifier = row.get("id")
        correct = row.get("correct")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{path} line {line_number} has no id")
        if identifier in correct_by_id:
            raise ValueError(f"{path} has duplicate id {identifier!r}")
        if not isinstance(correct, bool):
            raise ValueError(f"{path} line {line_number} correct must be boolean")
        if row.get("provenance_sha256") != provenance_sha256:
            raise ValueError(f"{path} line {line_number} provenance SHA mismatch")
        correct_by_id[identifier] = correct
    expected_ids = {f"gsm8k-{index:04d}" for index in range(EXPECTED_TOTAL)}
    if set(correct_by_id) != expected_ids:
        raise ValueError(
            "GSM8K result does not contain exactly 1319 matched dataset IDs"
        )
    return correct_by_id


def _load_run(root: Path) -> _Gsm8kRun:
    results = _load_json_object(root / RESULTS_FILENAME)
    total = results.get("total")
    if isinstance(total, bool) or total != EXPECTED_TOTAL:
        raise ValueError(f"GSM8K results.json total must equal {EXPECTED_TOTAL}")
    provenance = _require_mapping(results.get("provenance"), "results.provenance")
    provenance_sha256 = _require_nonempty_string(
        results.get("provenance_sha256"), "results.provenance_sha256"
    )
    if provenance_sha256 != _canonical_sha256(provenance):
        raise ValueError("GSM8K results provenance SHA mismatch")

    contract = _require_mapping(
        provenance.get("evaluation_contract"), "provenance.evaluation_contract"
    )
    dataset = _require_mapping(contract.get("dataset"), "evaluation_contract.dataset")
    evaluator = _require_mapping(provenance.get("evaluator"), "provenance.evaluator")
    if (
        dataset.get("sha256") != DATASET_SHA256
        or evaluator.get("dataset_sha256") != DATASET_SHA256
    ):
        raise ValueError("GSM8K result does not use the immutable dataset SHA256")
    dataset_total = dataset.get("total")
    evaluator_limit = evaluator.get("limit")
    if (
        isinstance(dataset_total, bool)
        or dataset_total != EXPECTED_TOTAL
        or isinstance(evaluator_limit, bool)
        or evaluator_limit != EXPECTED_TOTAL
    ):
        raise ValueError("GSM8K dataset and evaluator totals must equal 1319")
    if not isinstance(dataset.get("revision"), str) or not dataset.get("revision"):
        raise ValueError("GSM8K dataset revision must be recorded")

    model_revision = _require_nonempty_string(
        _contract_field(provenance, contract, "model_revision"), "model_revision"
    )
    tokenizer_revision = _require_nonempty_string(
        _contract_field(provenance, contract, "tokenizer_revision"),
        "tokenizer_revision",
    )
    generation_args = _require_mapping(
        _contract_field(provenance, contract, "generation_args"), "generation_args"
    )
    _validate_generation_args(generation_args, evaluator)
    runtime_fingerprint = _contract_field(provenance, contract, "runtime_fingerprint")
    if not (
        isinstance(runtime_fingerprint, str)
        and runtime_fingerprint
        or isinstance(runtime_fingerprint, Mapping)
        and runtime_fingerprint
    ):
        raise ValueError("GSM8K runtime_fingerprint must be nonempty")

    correct_by_id = _load_per_example(root / PER_EXAMPLE_FILENAME, provenance_sha256)
    correct_count = sum(correct_by_id.values())
    aggregate_correct = results.get("correct")
    if isinstance(aggregate_correct, bool) or aggregate_correct != correct_count:
        raise ValueError(
            "GSM8K aggregate correct count disagrees with per-example rows"
        )
    exact_match = results.get("exact_match")
    if (
        isinstance(exact_match, bool)
        or not isinstance(exact_match, (int, float))
        or not math.isfinite(exact_match)
        or exact_match != correct_count / EXPECTED_TOTAL
    ):
        raise ValueError("GSM8K aggregate exact_match disagrees with per-example rows")
    return _Gsm8kRun(
        correct_by_id=correct_by_id,
        provenance=provenance,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        generation_args=generation_args,
        runtime_fingerprint=runtime_fingerprint,
        evaluator_contract=evaluator,
    )


def exact_mcnemar_p_value(candidate_only: int, stock_only: int) -> float:
    """Return the exact two-sided binomial McNemar p-value."""
    for field_name, value in (
        ("candidate_only", candidate_only),
        ("stock_only", stock_only),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a nonnegative integer")
    disagreements = candidate_only + stock_only
    if disagreements == 0:
        return 1.0
    lower_tail = sum(
        math.comb(disagreements, index)
        for index in range(min(candidate_only, stock_only) + 1)
    )
    exact_probability = min(Fraction(1, 1), Fraction(2 * lower_tail, 2**disagreements))
    return float(exact_probability)


def paired_bootstrap_ci(
    stock_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
) -> tuple[float, float]:
    """Return a deterministic paired percentile CI for candidate-stock delta."""
    if not stock_correct or len(stock_correct) != len(candidate_correct):
        raise ValueError("paired bootstrap inputs must be nonempty and equal length")
    if any(type(value) is not bool for value in (*stock_correct, *candidate_correct)):
        raise ValueError("paired bootstrap inputs must contain booleans")
    deltas = tuple(
        int(candidate) - int(stock)
        for stock, candidate in zip(stock_correct, candidate_correct, strict=True)
    )
    if not any(deltas):
        return (0.0, 0.0)
    return paired_outcome_bootstrap_ci(deltas, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES)


@lru_cache(maxsize=16)
def paired_outcome_bootstrap_ci(
    deltas: tuple[int, ...], seed: int, samples: int
) -> tuple[float, float]:
    """Recompute the producer's deterministic paired percentile interval."""
    if not deltas or any(delta not in {-1, 0, 1} for delta in deltas):
        raise ValueError("paired outcomes must contain only -1, 0, or 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if not any(deltas):
        return (0.0, 0.0)
    rng = random.Random(seed)
    sample_size = len(deltas)
    estimates = [
        sum(rng.choice(deltas) for _ in range(sample_size)) / sample_size
        for _ in range(samples)
    ]
    estimates.sort()
    lower_index = math.floor(0.025 * (samples - 1))
    upper_index = math.ceil(0.975 * (samples - 1))
    return (estimates[lower_index], estimates[upper_index])


def paired_outcomes_sha256(deltas: Sequence[int]) -> str:
    """Hash the exact ordered outcomes used for the published interval."""
    return sha256(_canonical_json(list(deltas)).encode("ascii")).hexdigest()


def _matching_evaluator_contract(evaluator: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "model",
        "seed",
        "temperature",
        "top_p",
        "max_tokens",
        "concurrency",
        "system_prompt",
        "scoring",
    )
    return {field_name: evaluator.get(field_name) for field_name in fields}


def compare_gsm8k(stock: Path, candidate: Path) -> PairedGsm8kComparison:
    """Compare full GSM8K runs after enforcing their matched provenance."""
    stock_run = _load_run(stock)
    candidate_run = _load_run(candidate)
    if stock_run.model_revision != candidate_run.model_revision:
        raise ValueError("stock/candidate model revision mismatch")
    if stock_run.tokenizer_revision != candidate_run.tokenizer_revision:
        raise ValueError("stock/candidate tokenizer revision mismatch")
    if _canonical_json(stock_run.generation_args) != _canonical_json(
        candidate_run.generation_args
    ):
        raise ValueError("stock/candidate generation arguments mismatch")
    if _canonical_json(stock_run.runtime_fingerprint) != _canonical_json(
        candidate_run.runtime_fingerprint
    ):
        raise ValueError("stock/candidate runtime fingerprint mismatch")
    if _canonical_json(
        _matching_evaluator_contract(stock_run.evaluator_contract)
    ) != _canonical_json(
        _matching_evaluator_contract(candidate_run.evaluator_contract)
    ):
        raise ValueError("stock/candidate evaluator generation arguments mismatch")
    stock_ids = set(stock_run.correct_by_id)
    candidate_ids = set(candidate_run.correct_by_id)
    if stock_ids != candidate_ids or len(stock_ids) != EXPECTED_TOTAL:
        raise ValueError("stock/candidate GSM8K example IDs mismatch")

    ordered_ids = tuple(sorted(stock_ids))
    stock_correct = tuple(
        stock_run.correct_by_id[identifier] for identifier in ordered_ids
    )
    candidate_correct = tuple(
        candidate_run.correct_by_id[identifier] for identifier in ordered_ids
    )
    both_correct = sum(
        stock and candidate
        for stock, candidate in zip(stock_correct, candidate_correct, strict=True)
    )
    candidate_only = sum(
        not stock and candidate
        for stock, candidate in zip(stock_correct, candidate_correct, strict=True)
    )
    stock_only = sum(
        stock and not candidate
        for stock, candidate in zip(stock_correct, candidate_correct, strict=True)
    )
    both_wrong = EXPECTED_TOTAL - both_correct - candidate_only - stock_only
    stock_accuracy = sum(stock_correct) / EXPECTED_TOTAL
    candidate_accuracy = sum(candidate_correct) / EXPECTED_TOTAL
    delta = candidate_accuracy - stock_accuracy
    paired_outcomes = tuple(
        int(candidate) - int(stock)
        for stock, candidate in zip(stock_correct, candidate_correct, strict=True)
    )
    p_value = exact_mcnemar_p_value(candidate_only, stock_only)
    delta_ci95 = paired_outcome_bootstrap_ci(
        paired_outcomes, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES
    )
    passed = p_value >= 0.05 and delta_ci95[0] <= 0 <= delta_ci95[1]
    return PairedGsm8kComparison(
        stock_accuracy=stock_accuracy,
        candidate_accuracy=candidate_accuracy,
        candidate_only_wins=candidate_only,
        stock_only_wins=stock_only,
        mcnemar_p_value=p_value,
        delta_ci95=delta_ci95,
        passed=passed,
        both_correct=both_correct,
        both_wrong=both_wrong,
        accuracy_delta=delta,
        provenance_matched=True,
        matched_examples=EXPECTED_TOTAL,
        paired_outcomes=paired_outcomes,
        paired_outcomes_sha256=paired_outcomes_sha256(paired_outcomes),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        comparison = compare_gsm8k(args.stock, args.candidate)
    except ValueError as error:
        print(f"GSM8K comparison error: {error}", file=sys.stderr)
        return 2
    payload = asdict(comparison)
    try:
        payload.update(comparison_artifact_bindings(args.stock, args.candidate))
    except ValueError as error:
        print(f"GSM8K comparison error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
