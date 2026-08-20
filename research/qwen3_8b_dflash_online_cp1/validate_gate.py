import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def _latest_positive(metrics: dict[str, Any], name: str) -> float:
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


def _validate_draft_update(log_text: str) -> None:
    matches = re.findall(
        r"draft_update_probe=complete grad_l2=([0-9.eE+-]+) "
        r"checksum_before=([0-9.eE+-]+) checksum_after=([0-9.eE+-]+) "
        r"delta=([0-9.eE+-]+)",
        log_text,
    )
    if not matches:
        raise RuntimeError("missing draft update proof")
    if len(matches) < 2:
        raise RuntimeError(
            "gate requires at least two draft updates to prove post-refit generation"
        )
    grad_l2, before, after, delta = map(float, matches[-1])
    if not all(math.isfinite(value) for value in (grad_l2, before, after, delta)):
        raise RuntimeError("draft update proof must be finite")
    if grad_l2 <= 0 or delta <= 0 or before == after:
        raise RuntimeError("draft update proof requires gradient and parameter change")


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
        raise RuntimeError("gate requires initial and final validation metrics")


def validate_gate(metrics: dict[str, Any], log_text: str) -> None:
    _latest_positive(metrics, "train/draft_loss")
    _latest_positive(metrics, "train/draft_grad_norm")
    acceptance = _latest_positive(metrics, "train/vllm/spec_acceptance_rate")
    if acceptance > 1:
        raise RuntimeError("train/vllm/spec_acceptance_rate must not exceed 1")
    _validate_draft_update(log_text)
    for marker in (
        "draft_refit_manifest=",
        "draft_refit_load=complete",
        "draft_refit_finalize=complete",
    ):
        if log_text.count(marker) < 2:
            raise RuntimeError(f"missing gate marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wandb-run-path")
    args = parser.parse_args()
    if (args.metrics is None) == (args.wandb_run_path is None):
        raise RuntimeError("provide exactly one of metrics or --wandb-run-path")
    if args.metrics is not None:
        metrics = json.loads(args.metrics.read_text())
    else:
        import wandb

        error: Exception | None = None
        for _ in range(12):
            try:
                run = wandb.Api(timeout=60).run(args.wandb_run_path)
                metrics = dict(run.summary)
                validate_gate(metrics, args.log.read_text(errors="replace"))
                validate_validation_history(
                    run.scan_history(
                        keys=[
                            "_step",
                            "validation/accuracy",
                            "validation/avg_length",
                        ],
                        page_size=1000,
                    )
                )
                break
            except Exception as caught:
                error = caught
                time.sleep(10)
        else:
            raise RuntimeError(
                f"W&B gate evidence unavailable for {args.wandb_run_path}: {error}"
            )
        print("online_dflash_gate=complete")
        return
    validate_gate(metrics, args.log.read_text(errors="replace"))
    print("online_dflash_gate=complete")


if __name__ == "__main__":
    main()
