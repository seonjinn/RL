from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def _latest_positive(metrics: Mapping[str, Any], name: str) -> float:
    series = metrics.get(name)
    if isinstance(series, (int, float)):
        value = float(series)
    elif isinstance(series, dict) and series:
        value = float(series[max(series, key=lambda step: int(step))])
    else:
        raise RuntimeError(f"missing gate metric: {name}")
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be finite and positive, got {value}")
    return value


def validate_validation_history(rows: Iterable[Mapping[str, Any]]) -> None:
    steps = {
        int(row["_step"])
        for row in rows
        if all(
            key in row
            for key in ("_step", "validation/accuracy", "validation/avg_length")
        )
        and math.isfinite(float(row["validation/accuracy"]))
        and math.isfinite(float(row["validation/avg_length"]))
    }
    if len(steps) < 2:
        raise RuntimeError("gate requires initial and final validation metrics")


def validate_gate(metrics: Mapping[str, Any], log_text: str) -> None:
    _latest_positive(metrics, "train/loss")
    _latest_positive(metrics, "train/grad_norm")
    acceptance = _latest_positive(metrics, "train/vllm/spec_acceptance_rate")
    if acceptance > 1:
        raise RuntimeError("spec acceptance rate must not exceed one")
    target_refits = re.findall(
        r"MegatronPolicyWorker\[rank=0\].*GPU Memory after refit complete",
        log_text,
    )
    if len(target_refits) < 2:
        raise RuntimeError("gate requires two target refits")
    if re.search(
        r"draft_(?:update_probe=complete|refit_manifest=.*draft_count=[1-9]|refit_(?:load|finalize)=complete)",
        log_text,
    ):
        raise RuntimeError("fixed public drafter must remain immutable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wandb-run-path", required=True)
    args = parser.parse_args()
    import wandb

    error: Exception | None = None
    for _ in range(12):
        try:
            run = wandb.Api(timeout=60).run(args.wandb_run_path)
            validate_gate(dict(run.summary), args.log.read_text(errors="replace"))
            validate_validation_history(
                run.scan_history(
                    keys=["_step", "validation/accuracy", "validation/avg_length"],
                    page_size=1000,
                )
            )
            print("fixed_dspark_gate=complete")
            return
        except Exception as caught:
            error = caught
            time.sleep(10)
    raise RuntimeError(
        f"W&B gate evidence unavailable for {args.wandb_run_path}: {error}"
    )


if __name__ == "__main__":
    main()
