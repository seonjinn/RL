"""Pull per-step W&B history for SpecDec perf runs and summarize a shared step window.

    python extract_metrics.py --out report/data run_name_a run_name_b ...

Writes <out>/<run_name>.csv (every scalar key, one row per step) and
<out>/summary.csv (arithmetic mean of KEYS over steps [--first, --last]).
A run name matching several W&B runs resolves to the most recent one.
"""

import argparse
import csv
import math
from pathlib import Path

import wandb

KEYS = [
    "timing/train/generation",
    "timing/train/prepare_for_generation/total",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "timing/train/total_step_time",
    "performance/generation_tokens_per_sec_per_gpu",
    "performance/policy_training_tokens_per_sec_per_gpu",
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    "performance/tokens_per_sec_per_gpu",
    "train/reward",
    "train/mean_gen_tokens_per_sample",
    "train/token_mult_prob_error",
    "train/gen_kl_error",
    "train/approx_entropy",
]


def resolve_run(api: wandb.Api, path: str, name: str) -> wandb.apis.public.Run:
    runs = sorted(
        api.runs(path, filters={"display_name": name}),
        key=lambda r: r.created_at,
    )
    if not runs:
        raise SystemExit(f"no W&B run named {name!r} in {path}")
    return runs[-1]


def finite_mean(values: list[int | float | None]) -> float:
    kept = [v for v in values if v is not None and math.isfinite(v)]
    return sum(kept) / len(kept) if kept else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--project", default="sna-async-grpo-gb200")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--first", type=int, default=3)
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    path = f"{args.entity or api.default_entity}/{args.project}"
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name in args.runs:
        run = resolve_run(api, path, name)
        rows = [
            {k: v for k, v in row.items() if isinstance(v, (int, float))}
            for row in run.scan_history()
        ]
        rows = [r for r in rows if "_step" in r]
        columns = sorted({k for r in rows for k in r})
        with open(args.out / f"{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        window = [r for r in rows if args.first <= r["_step"] <= args.last]
        steps = sorted({r["_step"] for r in window})
        summary = {
            "run": name,
            "run_id": run.id,
            "state": run.state,
            "steps_in_window": len(steps),
            "max_step": max((r["_step"] for r in rows), default=-1),
        }
        # get_step_metrics() lands in the train/ dict, so acceptance is train/vllm/*
        spec_keys = sorted(
            {
                k
                for r in window
                for k in r
                if k.startswith("train/vllm/") or k.startswith("train/draft")
            }
        )
        for key in KEYS + spec_keys:
            summary[key] = finite_mean([r.get(key) for r in window if key in r])
        summary_rows.append(summary)
        print(f"{name}: id={run.id} state={run.state} window_steps={len(steps)}")

    columns = list(dict.fromkeys(k for r in summary_rows for k in r))
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
