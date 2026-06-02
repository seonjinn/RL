#!/usr/bin/env python3
"""Compare baseline and SpecDec standalone vLLM batch sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, dict[str, Any]] = {}
    for row in data.get("results", []):
        rows[int(row["bs"])] = row
    return rows


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--specdec", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    specdec = load_results(args.specdec)
    batch_sizes = sorted(set(baseline) | set(specdec))

    rows: list[dict[str, Any]] = []
    for bs in batch_sizes:
        base_row = baseline.get(bs)
        spec_row = specdec.get(bs)
        base_tput = (
            float(base_row["output_tok_s_per_gpu"]) if base_row is not None else None
        )
        spec_tput = (
            float(spec_row["output_tok_s_per_gpu"]) if spec_row is not None else None
        )
        speedup = (
            spec_tput / base_tput
            if base_tput is not None and spec_tput is not None and base_tput > 0
            else None
        )
        rows.append(
            {
                "batch_size": bs,
                "baseline_output_tok_s_per_gpu": base_tput,
                "specdec_output_tok_s_per_gpu": spec_tput,
                "generation_speedup": speedup,
                "baseline_latency_s": (
                    float(base_row["latency_s"]) if base_row is not None else None
                ),
                "specdec_latency_s": (
                    float(spec_row["latency_s"]) if spec_row is not None else None
                ),
            }
        )

    print(
        "| batch size | baseline tok/s/GPU | SpecDec tok/s/GPU | "
        "generation speedup | baseline latency | SpecDec latency |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        speedup = row["generation_speedup"]
        print(
            f"| {row['batch_size']} | "
            f"{fmt(row['baseline_output_tok_s_per_gpu'])} | "
            f"{fmt(row['specdec_output_tok_s_per_gpu'])} | "
            f"{fmt(speedup, 3)}x | "
            f"{fmt(row['baseline_latency_s'])}s | "
            f"{fmt(row['specdec_latency_s'])}s |"
        )

    complete = [
        row
        for row in rows
        if row["baseline_output_tok_s_per_gpu"] is not None
        and row["specdec_output_tok_s_per_gpu"] is not None
    ]
    positive = [
        row
        for row in complete
        if row["generation_speedup"] is not None and row["generation_speedup"] > 1.0
    ]
    best = max(
        complete,
        key=lambda row: row["generation_speedup"] or float("-inf"),
        default=None,
    )
    summary = {
        "num_complete_batch_sizes": len(complete),
        "positive_batch_sizes": [row["batch_size"] for row in positive],
        "best_batch_size": best["batch_size"] if best else None,
        "best_generation_speedup": best["generation_speedup"] if best else None,
        "rows": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
