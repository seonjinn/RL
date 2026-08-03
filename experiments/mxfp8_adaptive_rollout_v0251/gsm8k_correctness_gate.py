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
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _validate_dataset(
    dataset_path: Path, manifest_path: Path, expected_rows: int
) -> str:
    manifest = _load_object(manifest_path)
    encoded = dataset_path.read_bytes()
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if manifest.get("jsonl_sha256") != actual_sha256:
        raise ValueError("GSM8K JSONL SHA256 does not match its provenance manifest")
    if manifest.get("row_count") != expected_rows:
        raise ValueError("GSM8K manifest row count does not match the correctness gate")
    rows = dataset_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} GSM8K rows, found {len(rows)}")
    expected_ids = [f"gsm8k-test-{index:04d}" for index in range(expected_rows)]
    actual_ids: list[str] = []
    for line_number, line in enumerate(rows, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid GSM8K JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"GSM8K row {line_number} is not an object")
        if not all(
            isinstance(row.get(key), str) and row[key]
            for key in ("input", "output", "sample_id")
        ):
            raise ValueError(f"GSM8K row {line_number} is malformed")
        actual_ids.append(row["sample_id"])
    if actual_ids != expected_ids:
        raise ValueError("GSM8K sample IDs are missing, duplicated, or out of order")
    return actual_sha256


def _validate_config(directory: Path) -> dict[str, Any]:
    config = _load_object(directory / "config.json")
    expected = {
        "model_name": "Qwen/Qwen3-235B-A22B",
        "metric": "pass@k",
        "k_value": 1,
        "num_tests_per_prompt": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{directory.name} config has invalid {key}: {config.get(key)!r}"
            )
    dataset_name = config.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.endswith(
        "gsm8k_test.jsonl"
    ):
        raise ValueError(f"{directory.name} config does not use the pinned GSM8K JSONL")
    return config


def _load_samples(directory: Path, expected_rows: int) -> list[dict[str, Any]]:
    payload = _load_object(directory / "evaluation_data.json")
    samples = payload.get("evaluation_data")
    if not isinstance(samples, list) or len(samples) != expected_rows:
        count = len(samples) if isinstance(samples, list) else "non-list"
        raise ValueError(
            f"{directory.name} expected {expected_rows} samples, found {count}"
        )
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{directory.name} sample {index} is not an object")
        if sample.get("sample_index") != index:
            raise ValueError(f"{directory.name} sample index mismatch at row {index}")
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"]:
            raise ValueError(f"{directory.name} sample {index} has no prompt")
        if not isinstance(sample.get("response"), str) or not sample["response"]:
            raise ValueError(f"{directory.name} sample {index} has no response")
        reward = sample.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(
                f"{directory.name} sample {index} has a non-numeric reward"
            )
        if not math.isfinite(float(reward)) or float(reward) not in (0.0, 1.0):
            raise ValueError(f"{directory.name} sample {index} has a non-binary reward")
    return samples


def _one_sided_binomial_p_value(losses: int, gains: int) -> float:
    discordant = losses + gains
    if discordant == 0 or losses <= gains:
        return 1.0
    numerator = sum(
        math.comb(discordant, count) for count in range(losses, discordant + 1)
    )
    return numerator / (2**discordant)


def evaluate_gate(
    *,
    baseline_dir: Path,
    adaptive_dir: Path,
    dataset_path: Path,
    manifest_path: Path,
    expected_rows: int,
    alpha: float,
    min_baseline_accuracy: float,
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not 0.0 <= min_baseline_accuracy <= 1.0:
        raise ValueError("minimum baseline accuracy must be between zero and one")
    dataset_sha256 = _validate_dataset(dataset_path, manifest_path, expected_rows)
    baseline_config = _validate_config(baseline_dir)
    adaptive_config = _validate_config(adaptive_dir)
    if baseline_config != adaptive_config:
        raise ValueError("baseline and adaptive evaluation configs are not identical")

    baseline_samples = _load_samples(baseline_dir, expected_rows)
    adaptive_samples = _load_samples(adaptive_dir, expected_rows)
    baseline_rewards: list[int] = []
    adaptive_rewards: list[int] = []
    for index, (baseline, adaptive) in enumerate(
        zip(baseline_samples, adaptive_samples, strict=True)
    ):
        if baseline["prompt"] != adaptive["prompt"]:
            raise ValueError(f"baseline/adaptive prompt mismatch at sample {index}")
        baseline_rewards.append(int(baseline["reward"]))
        adaptive_rewards.append(int(adaptive["reward"]))

    gains = sum(
        base == 0 and adaptive == 1
        for base, adaptive in zip(baseline_rewards, adaptive_rewards, strict=True)
    )
    losses = sum(
        base == 1 and adaptive == 0
        for base, adaptive in zip(baseline_rewards, adaptive_rewards, strict=True)
    )
    baseline_accuracy = sum(baseline_rewards) / expected_rows
    adaptive_accuracy = sum(adaptive_rewards) / expected_rows
    if baseline_accuracy < min_baseline_accuracy:
        raise ValueError(
            "baseline accuracy is below the validity floor: "
            f"{baseline_accuracy:.6f} < {min_baseline_accuracy:.6f}"
        )
    p_value = _one_sided_binomial_p_value(losses, gains)
    regression = adaptive_accuracy < baseline_accuracy and p_value <= alpha
    return {
        "status": "fail" if regression else "pass",
        "test": "one-sided exact paired binomial test on discordant rewards",
        "alpha": alpha,
        "min_baseline_accuracy": min_baseline_accuracy,
        "row_count": expected_rows,
        "dataset_sha256": dataset_sha256,
        "baseline_accuracy": baseline_accuracy,
        "adaptive_accuracy": adaptive_accuracy,
        "absolute_accuracy_delta": adaptive_accuracy - baseline_accuracy,
        "paired": {
            "adaptive_gains": gains,
            "adaptive_losses": losses,
            "ties": expected_rows - gains - losses,
            "one_sided_p_value": p_value,
        },
    }


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate Qwen3-235B adaptive GSM8K accuracy"
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1319)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-baseline-accuracy", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = evaluate_gate(
            baseline_dir=args.baseline_dir,
            adaptive_dir=args.adaptive_dir,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            expected_rows=args.expected_rows,
            alpha=args.alpha,
            min_baseline_accuracy=args.min_baseline_accuracy,
        )
    except ValueError as error:
        raise SystemExit(f"Qwen235 GSM8K correctness gate failed: {error}") from error

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _atomic_write(args.output, encoded)
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(
            "Qwen235 GSM8K correctness gate failed: "
            "statistically significant accuracy regression"
        )


if __name__ == "__main__":
    main()
