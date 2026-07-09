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

"""Compare baseline and speculative generation parity artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STOP_FEATURE = "<STOP>"
SPECDEC_SETTING_KEYS = {
    "draft_model",
    "draft_sample_method",
    "draft_tp",
    "method",
    "num_speculative_tokens",
}
MATCHED_METADATA_KEYS = {
    "batch_size",
    "git_commit",
    "prompt_count",
    "prompt_data",
    "requested_samples",
    "samples_per_prompt",
}


def validate_metadata_contract(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_mode: str,
) -> None:
    """Require clean, matched runs before comparing generated samples."""
    for label, metadata in (("baseline", baseline), ("candidate", candidate)):
        if metadata.get("status") != "passed":
            raise ValueError(
                f"{label} status is not passed: {metadata.get('status')!r}"
            )
        if metadata.get("cleanup_errors"):
            raise ValueError(f"{label} contains cleanup_errors")
        if metadata.get("mode") != expected_mode:
            raise ValueError(
                f"{label} mode mismatch: expected {expected_mode!r}, "
                f"got {metadata.get('mode')!r}"
            )
        if not isinstance(metadata.get("settings"), Mapping):
            raise ValueError(f"{label} contains no settings mapping")

    for key in sorted(MATCHED_METADATA_KEYS):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(
                f"metadata mismatch for {key}: "
                f"baseline={baseline.get(key)!r}, candidate={candidate.get(key)!r}"
            )

    baseline_settings = dict(baseline["settings"])
    candidate_settings = dict(candidate["settings"])
    setting_keys = baseline_settings.keys() | candidate_settings.keys()
    for key in sorted(setting_keys - SPECDEC_SETTING_KEYS):
        if baseline_settings.get(key) != candidate_settings.get(key):
            raise ValueError(
                f"settings mismatch for {key}: "
                f"baseline={baseline_settings.get(key)!r}, "
                f"candidate={candidate_settings.get(key)!r}"
            )

    if baseline_settings.get("draft_model") is not None:
        raise ValueError("baseline draft_model must be unset")
    if candidate_settings.get("draft_model") is None:
        raise ValueError("candidate draft_model must be set")


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    if not rows:
        raise ValueError(f"{label} contains no samples")

    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        prompt_id = str(row.get("prompt_id", ""))
        sample_id = str(row.get("sample_id", ""))
        if not prompt_id or not sample_id:
            raise ValueError(f"{label} row {index} has an empty prompt_id or sample_id")
        key = (prompt_id, sample_id)
        if key in seen:
            raise ValueError(f"{label} duplicates sample {key!r}")
        seen.add(key)

        token_ids = row.get("token_ids")
        token_logprobs = row.get("token_logprobs")
        if not isinstance(token_ids, list) or not isinstance(token_logprobs, list):
            raise ValueError(f"{label} row {index} must contain token/logprob lists")
        if len(token_ids) != len(token_logprobs):
            raise ValueError(
                f"{label} row {index} token/logprob length mismatch: "
                f"tokens={len(token_ids)}, logprobs={len(token_logprobs)}"
            )
        if not token_ids:
            raise ValueError(f"{label} row {index} contains an empty generation")
        if any(not isinstance(token_id, int) for token_id in token_ids):
            raise ValueError(f"{label} row {index} contains a non-integer token id")
        for value in token_logprobs:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{label} row {index} contains a non-finite logprob")
            if float(value) > 1e-5:
                raise ValueError(f"{label} row {index} contains a positive logprob")
        if not isinstance(row.get("truncated", False), bool):
            raise ValueError(f"{label} row {index} has a non-boolean truncated value")


def _group_by_prompt(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    return dict(grouped)


def _sequence_features(
    row: Mapping[str, Any], *, max_positions: int
) -> dict[tuple[Any, ...], float]:
    token_ids = row["token_ids"]
    features: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    previous: int | str | None = None
    for position in range(max_positions):
        token: int | str = (
            int(token_ids[position]) if position < len(token_ids) else STOP_FEATURE
        )
        features[("position", position, token)] += 1
        if previous is not None:
            features[("pair", position - 1, previous, token)] += 1
        previous = token
        if token == STOP_FEATURE:
            break
    tail = token_ids[max_positions:]
    if tail:
        tail_scale = 1.0 / len(tail)
        tail_bucket_count = min(16, len(tail))
        for tail_position, token_id in enumerate(tail):
            bucket = min(
                tail_position * tail_bucket_count // len(tail),
                tail_bucket_count - 1,
            )
            features[("tail_token", bucket, int(token_id))] += tail_scale
            if tail_position > 0:
                features[
                    (
                        "tail_pair",
                        bucket,
                        int(tail[tail_position - 1]),
                        int(token_id),
                    )
                ] += tail_scale
        features[("tail_last", int(tail[-1]))] += 1
    return dict(features)


def _mean_features(
    rows: Sequence[Mapping[str, Any]], *, max_positions: int
) -> dict[tuple[Any, ...], float]:
    totals: defaultdict[tuple[Any, ...], float] = defaultdict(float)
    for row in rows:
        for feature, value in _sequence_features(
            row, max_positions=max_positions
        ).items():
            totals[feature] += value
    scale = 1.0 / len(rows)
    return {feature: count * scale for feature, count in totals.items()}


def _prompt_stratified_mmd(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_positions: int,
) -> float:
    prompt_scores: list[float] = []
    for prompt_id in sorted(baseline):
        baseline_mean = _mean_features(baseline[prompt_id], max_positions=max_positions)
        candidate_mean = _mean_features(
            candidate[prompt_id], max_positions=max_positions
        )
        features = baseline_mean.keys() | candidate_mean.keys()
        prompt_scores.append(
            sum(
                (baseline_mean.get(feature, 0.0) - candidate_mean.get(feature, 0.0))
                ** 2
                for feature in features
            )
        )
    return sum(prompt_scores) / len(prompt_scores)


def _permutation_p_value(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    permutations: int,
    max_positions: int,
    seed: int,
) -> tuple[float, float]:
    if permutations <= 0:
        raise ValueError(f"permutations must be positive, got {permutations}")

    observed = _prompt_stratified_mmd(baseline, candidate, max_positions=max_positions)
    rng = random.Random(seed)
    exceedances = 0
    prompt_ids = sorted(baseline)
    for _ in range(permutations):
        permuted_baseline: dict[str, list[Mapping[str, Any]]] = {}
        permuted_candidate: dict[str, list[Mapping[str, Any]]] = {}
        for prompt_id in prompt_ids:
            baseline_rows = list(baseline[prompt_id])
            combined = baseline_rows + list(candidate[prompt_id])
            rng.shuffle(combined)
            split = len(baseline_rows)
            permuted_baseline[prompt_id] = combined[:split]
            permuted_candidate[prompt_id] = combined[split:]
        permuted = _prompt_stratified_mmd(
            permuted_baseline,
            permuted_candidate,
            max_positions=max_positions,
        )
        if permuted >= observed - 1e-15:
            exceedances += 1
    return observed, (exceedances + 1) / (permutations + 1)


def _bootstrap_mmd_upper_bound(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    confidence: float,
    max_positions: int,
    seed: int,
) -> float:
    if replicates <= 0:
        raise ValueError(f"bootstrap replicates must be positive, got {replicates}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"bootstrap confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(replicates):
        sampled_baseline: dict[str, list[Mapping[str, Any]]] = {}
        sampled_candidate: dict[str, list[Mapping[str, Any]]] = {}
        for prompt_id in sorted(baseline):
            baseline_rows = baseline[prompt_id]
            candidate_rows = candidate[prompt_id]
            sampled_baseline[prompt_id] = [
                rng.choice(baseline_rows) for _ in baseline_rows
            ]
            sampled_candidate[prompt_id] = [
                rng.choice(candidate_rows) for _ in candidate_rows
            ]
        scores.append(
            _prompt_stratified_mmd(
                sampled_baseline,
                sampled_candidate,
                max_positions=max_positions,
            )
        )
    scores.sort()
    index = min(math.ceil(confidence * len(scores)) - 1, len(scores) - 1)
    return scores[max(index, 0)]


def _max_prompt_delta(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    value,
) -> float:
    deltas: list[float] = []
    for prompt_id in sorted(baseline):
        baseline_values = [float(value(row)) for row in baseline[prompt_id]]
        candidate_values = [float(value(row)) for row in candidate[prompt_id]]
        deltas.append(
            abs(
                sum(baseline_values) / len(baseline_values)
                - sum(candidate_values) / len(candidate_values)
            )
        )
    return max(deltas, default=0.0)


def _max_prompt_relative_delta(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    value,
) -> tuple[float, float]:
    absolute_deltas: list[float] = []
    relative_deltas: list[float] = []
    for prompt_id in sorted(baseline):
        baseline_values = [float(value(row)) for row in baseline[prompt_id]]
        candidate_values = [float(value(row)) for row in candidate[prompt_id]]
        baseline_mean = sum(baseline_values) / len(baseline_values)
        candidate_mean = sum(candidate_values) / len(candidate_values)
        absolute_delta = abs(baseline_mean - candidate_mean)
        absolute_deltas.append(absolute_delta)
        relative_deltas.append(absolute_delta / max(abs(baseline_mean), 1.0))
    return max(absolute_deltas, default=0.0), max(relative_deltas, default=0.0)


def _mean_selected_token_logprob(row: Mapping[str, Any]) -> float:
    values = [float(value) for value in row["token_logprobs"]]
    return sum(values) / len(values)


def _max_prompt_logprob_mean_delta(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    return _max_prompt_delta(baseline, candidate, _mean_selected_token_logprob)


def _logprob_permutation_p_value(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    if permutations <= 0:
        raise ValueError(f"permutations must be positive, got {permutations}")

    observed = _max_prompt_logprob_mean_delta(baseline, candidate)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted_baseline: dict[str, list[Mapping[str, Any]]] = {}
        permuted_candidate: dict[str, list[Mapping[str, Any]]] = {}
        for prompt_id in sorted(baseline):
            baseline_rows = list(baseline[prompt_id])
            combined = baseline_rows + list(candidate[prompt_id])
            rng.shuffle(combined)
            split = len(baseline_rows)
            permuted_baseline[prompt_id] = combined[:split]
            permuted_candidate[prompt_id] = combined[split:]
        permuted = _max_prompt_logprob_mean_delta(permuted_baseline, permuted_candidate)
        if permuted >= observed - 1e-15:
            exceedances += 1
    return observed, (exceedances + 1) / (permutations + 1)


def _bootstrap_logprob_delta_upper_bound(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> float:
    if replicates <= 0:
        raise ValueError(f"bootstrap replicates must be positive, got {replicates}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"bootstrap confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled_baseline: dict[str, list[Mapping[str, Any]]] = {}
        sampled_candidate: dict[str, list[Mapping[str, Any]]] = {}
        for prompt_id in sorted(baseline):
            baseline_rows = baseline[prompt_id]
            candidate_rows = candidate[prompt_id]
            sampled_baseline[prompt_id] = [
                rng.choice(baseline_rows) for _ in baseline_rows
            ]
            sampled_candidate[prompt_id] = [
                rng.choice(candidate_rows) for _ in candidate_rows
            ]
        deltas.append(
            _max_prompt_logprob_mean_delta(sampled_baseline, sampled_candidate)
        )
    deltas.sort()
    index = min(math.ceil(confidence * len(deltas)) - 1, len(deltas) - 1)
    return deltas[max(index, 0)]


def _analyze_greedy(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    logprob_atol: float,
) -> dict[str, Any]:
    baseline = {
        (str(row["prompt_id"]), str(row["sample_id"])): row for row in baseline_rows
    }
    candidate = {
        (str(row["prompt_id"]), str(row["sample_id"])): row for row in candidate_rows
    }
    if baseline.keys() != candidate.keys():
        raise ValueError("greedy baseline and candidate sample identities do not match")

    sequence_mismatches = 0
    truncation_mismatches = 0
    max_logprob_delta = 0.0
    for key, baseline_row in baseline.items():
        candidate_row = candidate[key]
        if baseline_row["token_ids"] != candidate_row["token_ids"]:
            sequence_mismatches += 1
            continue
        if bool(baseline_row.get("truncated", False)) != bool(
            candidate_row.get("truncated", False)
        ):
            truncation_mismatches += 1
        max_logprob_delta = max(
            max_logprob_delta,
            max(
                (
                    abs(float(left) - float(right))
                    for left, right in zip(
                        baseline_row["token_logprobs"],
                        candidate_row["token_logprobs"],
                        strict=True,
                    )
                ),
                default=0.0,
            ),
        )

    return {
        "exact_sequence_match": {
            "passed": sequence_mismatches == 0,
            "mismatched_samples": sequence_mismatches,
        },
        "termination_match": {
            "passed": truncation_mismatches == 0,
            "mismatched_samples": truncation_mismatches,
        },
        "chosen_logprob_match": {
            "passed": sequence_mismatches == 0 and max_logprob_delta <= logprob_atol,
            "max_abs_delta": max_logprob_delta,
            "atol": logprob_atol,
        },
    }


def analyze_parity_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    permutations: int = 9999,
    min_samples_per_prompt: int = 64,
    alpha: float = 0.01,
    sequence_mmd_margin: float = 0.5,
    equivalence_confidence: float = 0.99,
    bootstrap_replicates: int = 499,
    length_relative_margin: float = 0.05,
    truncation_rate_margin: float = 0.05,
    logprob_atol: float = 0.05,
    sampled_logprob_mean_margin: float = 0.1,
    max_positions: int = 64,
    seed: int = 20260709,
) -> dict[str, Any]:
    _validate_rows(baseline_rows, label="baseline")
    _validate_rows(candidate_rows, label="candidate")
    if mode not in {"greedy", "sampled"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if sampled_logprob_mean_margin < 0.0:
        raise ValueError(
            "sampled logprob mean margin must be non-negative, got "
            f"{sampled_logprob_mean_margin}"
        )

    if mode == "greedy":
        checks = _analyze_greedy(
            baseline_rows, candidate_rows, logprob_atol=logprob_atol
        )
        passed = all(bool(check["passed"]) for check in checks.values())
        return {
            "mode": mode,
            "status": "passed" if passed else "failed",
            "checks": checks,
        }

    baseline = _group_by_prompt(baseline_rows)
    candidate = _group_by_prompt(candidate_rows)
    if baseline.keys() != candidate.keys():
        raise ValueError("sampled baseline and candidate prompt sets do not match")
    unequal_counts = {
        prompt_id: (len(baseline[prompt_id]), len(candidate[prompt_id]))
        for prompt_id in baseline
        if len(baseline[prompt_id]) != len(candidate[prompt_id])
    }
    if unequal_counts:
        raise ValueError(f"sampled prompt cohort sizes do not match: {unequal_counts}")

    minimum_count = min(len(rows) for rows in baseline.values())
    if minimum_count < min_samples_per_prompt:
        return {
            "mode": mode,
            "status": "inconclusive",
            "reason": (
                f"minimum samples per prompt is {minimum_count}, below required "
                f"{min_samples_per_prompt}"
            ),
            "checks": {},
        }

    observed_mmd, p_value = _permutation_p_value(
        baseline,
        candidate,
        permutations=permutations,
        max_positions=max_positions,
        seed=seed,
    )
    mmd_upper_bound = _bootstrap_mmd_upper_bound(
        baseline,
        candidate,
        replicates=bootstrap_replicates,
        confidence=equivalence_confidence,
        max_positions=max_positions,
        seed=seed + 1,
    )
    max_absolute_length_delta, max_relative_length_delta = _max_prompt_relative_delta(
        baseline, candidate, lambda row: len(row["token_ids"])
    )
    max_truncation_rate_delta = _max_prompt_delta(
        baseline, candidate, lambda row: bool(row.get("truncated", False))
    )
    max_logprob_mean_delta, logprob_p_value = _logprob_permutation_p_value(
        baseline,
        candidate,
        permutations=permutations,
        seed=seed + 2,
    )
    logprob_delta_upper_bound = _bootstrap_logprob_delta_upper_bound(
        baseline,
        candidate,
        replicates=bootstrap_replicates,
        confidence=equivalence_confidence,
        seed=seed + 3,
    )

    checks = {
        "sequence_distribution": {
            "passed": p_value >= alpha and mmd_upper_bound <= sequence_mmd_margin,
            "detected_shift": p_value < alpha,
            "equivalent": mmd_upper_bound <= sequence_mmd_margin,
            "mmd": observed_mmd,
            "bootstrap_upper_mmd": mmd_upper_bound,
            "mmd_margin": sequence_mmd_margin,
            "equivalence_confidence": equivalence_confidence,
            "bootstrap_replicates": bootstrap_replicates,
            "p_value": p_value,
            "alpha": alpha,
            "permutations": permutations,
            "max_positions": max_positions,
        },
        "length_equivalence": {
            "passed": max_relative_length_delta <= length_relative_margin,
            "max_absolute_mean_delta": max_absolute_length_delta,
            "max_relative_mean_delta": max_relative_length_delta,
            "relative_margin": length_relative_margin,
        },
        "truncation_rate_equivalence": {
            "passed": max_truncation_rate_delta <= truncation_rate_margin,
            "max_absolute_rate_delta": max_truncation_rate_delta,
            "margin": truncation_rate_margin,
        },
        "selected_token_logprob_equivalence": {
            "passed": logprob_p_value >= alpha
            and logprob_delta_upper_bound <= sampled_logprob_mean_margin,
            "detected_shift": logprob_p_value < alpha,
            "equivalent": logprob_delta_upper_bound <= sampled_logprob_mean_margin,
            "statistic": "max_prompt_delta_of_sequence_mean_selected_logprob",
            "max_absolute_mean_delta": max_logprob_mean_delta,
            "bootstrap_upper_mean_delta": logprob_delta_upper_bound,
            "mean_delta_margin": sampled_logprob_mean_margin,
            "equivalence_confidence": equivalence_confidence,
            "bootstrap_replicates": bootstrap_replicates,
            "p_value": logprob_p_value,
            "alpha": alpha,
            "permutations": permutations,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    detected_failure = (
        bool(checks["sequence_distribution"]["detected_shift"])
        or not bool(checks["length_equivalence"]["passed"])
        or not bool(checks["truncation_rate_equivalence"]["passed"])
        or bool(checks["selected_token_logprob_equivalence"]["detected_shift"])
    )
    return {
        "mode": mode,
        "status": "passed"
        if passed
        else "failed"
        if detected_failure
        else "inconclusive",
        "prompt_count": len(baseline),
        "samples_per_prompt": minimum_count,
        "checks": checks,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-metadata-json", type=Path)
    parser.add_argument("--candidate-metadata-json", type=Path)
    parser.add_argument("--mode", choices=("greedy", "sampled"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--min-samples-per-prompt", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--sampled-logprob-mean-margin", type=float, default=0.1)
    parser.add_argument("--equivalence-confidence", type=float, default=0.99)
    parser.add_argument("--bootstrap-replicates", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_metadata_path = (
        args.baseline_metadata_json
        or args.baseline_jsonl.with_suffix(
            args.baseline_jsonl.suffix + ".metadata.json"
        )
    )
    candidate_metadata_path = (
        args.candidate_metadata_json
        or args.candidate_jsonl.with_suffix(
            args.candidate_jsonl.suffix + ".metadata.json"
        )
    )
    baseline_metadata = _load_json_object(baseline_metadata_path)
    candidate_metadata = _load_json_object(candidate_metadata_path)
    validate_metadata_contract(
        baseline_metadata,
        candidate_metadata,
        expected_mode=args.mode,
    )
    report = analyze_parity_rows(
        _load_jsonl(args.baseline_jsonl),
        _load_jsonl(args.candidate_jsonl),
        mode=args.mode,
        permutations=args.permutations,
        min_samples_per_prompt=args.min_samples_per_prompt,
        alpha=args.alpha,
        sampled_logprob_mean_margin=args.sampled_logprob_mean_margin,
        equivalence_confidence=args.equivalence_confidence,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    report["metadata_contract"] = {
        "status": "passed",
        "baseline": str(baseline_metadata_path),
        "candidate": str(candidate_metadata_path),
        "git_commit": baseline_metadata.get("git_commit"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"passed": 0, "failed": 1, "inconclusive": 2}[str(report["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
