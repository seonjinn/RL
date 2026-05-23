#!/usr/bin/env python3
"""Compare GRPO reward and logprob-error metrics from result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = (
    "_step",
    "train/reward",
    "train/token_mult_prob_error",
    "train/max_seq_mult_prob_error",
    "train/mean_seq_mult_prob_error",
    "train/min_seq_mult_prob_error",
    "train/max_seq_mult_prob_error_after_mask",
    "train/mean_seq_mult_prob_error_after_mask",
    "train/num_masked_seqs_by_logprob_error",
)


def latest_summary(result_dir: Path) -> Path | None:
    summaries = sorted(
        result_dir.rglob("wandb-summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def load_metrics(summary_path: Path) -> dict[str, object]:
    data = json.loads(summary_path.read_text())
    return {key: data.get(key) for key in METRIC_KEYS if key in data}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print reward and token/sequence logprob-error metrics."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="Result directories containing nemorl_logs/wandb-summary.json files.",
    )
    args = parser.parse_args()

    for result_dir in args.result_dirs:
        summary_path = latest_summary(result_dir)
        print(f"\n{result_dir}")
        if summary_path is None:
            print("  no wandb-summary.json found yet")
            continue
        print(f"  summary: {summary_path}")
        for key, value in load_metrics(summary_path).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
