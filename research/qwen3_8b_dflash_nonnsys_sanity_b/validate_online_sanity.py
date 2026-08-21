import argparse
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


def _latest_positive(metrics: dict[str, Any], name: str) -> float:
    series = metrics.get(name)
    if isinstance(series, (int, float)):
        value = float(series)
    elif isinstance(series, dict) and series:
        value = float(series[max(series, key=lambda step: int(step))])
    else:
        raise RuntimeError(f"missing sanity metric: {name}")
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be finite and positive, got {value}")
    return value


def validate_validation_history(rows: Iterable[Mapping[str, Any]]) -> None:
    validation_steps: set[int] = set()
    for row in rows:
        if not all(
            key in row
            for key in ("_step", "validation/accuracy", "validation/avg_length")
        ):
            continue
        accuracy = float(row["validation/accuracy"])
        avg_length = float(row["validation/avg_length"])
        if not math.isfinite(accuracy) or not math.isfinite(avg_length):
            raise RuntimeError("validation metrics must be finite")
        validation_steps.add(int(row["_step"]))
    if len(validation_steps) < 2:
        raise RuntimeError("sanity requires initial and final validation metrics")


def validate_sanity(metrics: dict[str, Any], log_text: str) -> None:
    _latest_positive(metrics, "train/draft_loss")
    _latest_positive(metrics, "train/draft_grad_norm")
    acceptance = _latest_positive(metrics, "train/vllm/spec_acceptance_rate")
    if acceptance > 1:
        raise RuntimeError("train/vllm/spec_acceptance_rate must not exceed 1")
    for marker in (
        "draft_refit_manifest=",
        "draft_refit_load=complete",
        "draft_refit_finalize=complete",
    ):
        if log_text.count(marker) < 2:
            raise RuntimeError(f"missing sanity marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wandb-run-path", required=True)
    args = parser.parse_args()
    import wandb  # Optional cluster dependency.

    error: Exception | None = None
    for _ in range(12):
        try:
            run = wandb.Api(timeout=60).run(args.wandb_run_path)
            validate_sanity(dict(run.summary), args.log.read_text(errors="replace"))
            validate_validation_history(
                run.scan_history(
                    keys=["_step", "validation/accuracy", "validation/avg_length"],
                    page_size=1000,
                )
            )
            break
        except Exception as caught:
            error = caught
            time.sleep(10)
    else:
        raise RuntimeError(
            f"W&B sanity evidence unavailable for {args.wandb_run_path}: {error}"
        )
    print("online_dflash_nonnsys_sanity=complete")


if __name__ == "__main__":
    main()
