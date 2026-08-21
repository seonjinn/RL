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


def _validate_draft_updates(log_text: str) -> None:
    pattern = (
        r"draft_update_probe=complete grad_l2=([0-9.eE+-]+) "
        r"checksum_sum_before=([0-9.eE+-]+) "
        r"checksum_sum_after=([0-9.eE+-]+) "
        r"checksum_l2_before=([0-9.eE+-]+) "
        r"checksum_l2_after=([0-9.eE+-]+) "
        r"delta=([0-9.eE+-]+)"
    )
    matches = re.findall(pattern, log_text)
    if len(matches) < 2:
        raise RuntimeError("gate requires at least two proven draft updates")
    for match in matches[-2:]:
        values = tuple(map(float, match))
        grad_l2, sum_before, sum_after, l2_before, l2_after, delta = values
        if not all(map(math.isfinite, values)):
            raise RuntimeError("draft update proof must be finite")
        if grad_l2 <= 0 or delta <= 0:
            raise RuntimeError("draft update requires gradient and parameter change")
        if sum_before == sum_after and l2_before == l2_after:
            raise RuntimeError("draft parameters did not change")


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
    _latest_positive(metrics, "train/draft_loss")
    _latest_positive(metrics, "train/draft_grad_norm")
    acceptance = _latest_positive(metrics, "train/vllm/spec_acceptance_rate")
    if acceptance > 1:
        raise RuntimeError("spec acceptance rate must not exceed one")
    _validate_draft_updates(log_text)
    for marker in (
        "draft_refit_manifest=",
        "draft_refit_load=complete",
        "draft_refit_finalize=complete",
    ):
        if log_text.count(marker) < 2:
            raise RuntimeError(f"gate requires two markers: {marker}")


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
            print("online_dspark_gate=complete")
            return
        except Exception as caught:
            error = caught
            time.sleep(10)
    raise RuntimeError(
        f"W&B gate evidence unavailable for {args.wandb_run_path}: {error}"
    )


if __name__ == "__main__":
    main()
