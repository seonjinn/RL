from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_response_validity(
    evaluation_path: Path,
    *,
    expected_rows: int,
    max_repetitive_fraction: float,
) -> dict[str, Any]:
    if expected_rows <= 0:
        raise ValueError("expected rows must be positive")
    if not 0.0 <= max_repetitive_fraction <= 1.0:
        raise ValueError("maximum repetitive fraction must be between zero and one")

    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    samples = payload.get("evaluation_data")
    if not isinstance(samples, list) or len(samples) != expected_rows:
        count = len(samples) if isinstance(samples, list) else "non-list"
        raise ValueError(f"expected {expected_rows} responses, found {count}")

    responses: list[str] = []
    repetitive_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("sample_index") != index:
            raise ValueError(f"invalid sample at index {index}")
        response = sample.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"empty response at index {index}")
        responses.append(response)
        normalized = response.strip()
        if len(normalized) >= 32 and len(set(normalized)) <= 2:
            repetitive_count += 1

    repetitive_fraction = repetitive_count / expected_rows
    if repetitive_fraction > max_repetitive_fraction:
        raise ValueError(
            "repetitive-response fraction exceeds the validity limit: "
            f"{repetitive_fraction:.6f} > {max_repetitive_fraction:.6f}"
        )
    return {
        "status": "pass",
        "row_count": expected_rows,
        "unique_response_count": len(set(responses)),
        "repetitive_response_count": repetitive_count,
        "repetitive_response_fraction": repetitive_fraction,
        "max_repetitive_fraction": max_repetitive_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated token diversity")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--max-repetitive-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = evaluate_response_validity(
            args.evaluation,
            expected_rows=args.expected_rows,
            max_repetitive_fraction=args.max_repetitive_fraction,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"response validity gate failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
