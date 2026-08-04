from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _assistant_response(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def evaluate_refit_validation(
    validation_jsonl: Path,
    *,
    expected_rows: int,
    max_repetitive_fraction: float,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in validation_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")

    responses = [_assistant_response(row["content"]) for row in rows]
    empty_count = sum(not response.strip() for response in responses)
    repetitive_count = sum(
        len(set(response.strip())) <= 1 for response in responses if response.strip()
    )
    repetitive_fraction = repetitive_count / len(responses)
    if empty_count:
        raise ValueError(f"found {empty_count} empty responses")
    if repetitive_fraction > max_repetitive_fraction:
        raise ValueError(
            "repetitive-response fraction "
            f"{repetitive_fraction:.4f} exceeds {max_repetitive_fraction:.4f}"
        )

    return {
        "status": "pass",
        "row_count": len(rows),
        "unique_response_count": len(set(responses)),
        "repetitive_response_count": repetitive_count,
        "repetitive_response_fraction": repetitive_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--max-repetitive-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted(args.log_root.glob("exp_*/val_data_step0.jsonl"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one validation artifact under {args.log_root}, found {candidates}"
        )
    report = evaluate_refit_validation(
        candidates[0],
        expected_rows=args.expected_rows,
        max_repetitive_fraction=args.max_repetitive_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
