#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from urllib.parse import urlparse

import wandb


METRICS = (
    "train/gen_kl_error",
    "train/policy_kl_error",
    "train/token_mult_prob_error",
    "train/reward",
    "train/loss",
    "timing/train/total_step_time",
    "performance/tokens_per_sec_per_gpu",
    "timing/train/generation",
    "performance/generation_tokens_per_sec_per_gpu",
    "timing/train/prepare_for_generation/total",
    "timing/train/prepare_for_generation/transfer_and_update_weights",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
)


def run_path(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 4 or parts[-2] != "runs":
        raise ValueError(f"not a W&B run URL: {url}")
    return "/".join((parts[-4], parts[-3], parts[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_urls", nargs="+")
    parser.add_argument("--start-step", type=int, default=3)
    parser.add_argument("--end-step", type=int, default=20)
    args = parser.parse_args()

    reports = []
    requested = set(range(args.start_step, args.end_step + 1))
    for url in args.run_urls:
        run = wandb.Api().run(run_path(url))
        values: dict[str, list[float]] = defaultdict(list)
        included: set[int] = set()
        for row in run.scan_history(
            min_step=args.start_step,
            max_step=args.end_step + 1,
            page_size=1000,
        ):
            step = row.get("_step")
            if not isinstance(step, int) or step not in requested:
                continue
            included.add(step)
            for metric in METRICS:
                value = row.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric = float(value)
                    if not math.isnan(numeric):
                        values[metric].append(numeric)

        ranges = {
            metric: {
                "min": min(metric_values),
                "max": max(metric_values),
                "valid_count": len(metric_values),
            }
            for metric, metric_values in values.items()
            if metric_values
        }
        reports.append(
            {
                "url": url,
                "run_name": run.name,
                "requested_steps": sorted(requested),
                "included_steps": sorted(included),
                "missing_steps": sorted(requested - included),
                "ranges": ranges,
            }
        )

    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
