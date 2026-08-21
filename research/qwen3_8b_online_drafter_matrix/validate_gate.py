#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import time


def _positive(summary: dict[str, object], key: str) -> float:
    value = float(summary[key])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{key} must be finite and positive")
    return value


def validate(arm: str, summary: dict[str, object], log: str) -> None:
    _positive(summary, "train/grad_norm")
    if arm == "baseline":
        if "draft_update_probe=complete" in log:
            raise RuntimeError("baseline unexpectedly updated a drafter")
        return
    if arm.startswith("dflash-fixed-"):
        acceptance = _positive(summary, "train/vllm/spec_acceptance_rate")
        if acceptance > 1:
            raise RuntimeError("acceptance rate exceeds one")
        if "draft_update_probe=complete" in log:
            raise RuntimeError("fixed arm unexpectedly updated a drafter")
        return
    _positive(summary, "train/draft_loss")
    _positive(summary, "train/draft_grad_norm")
    acceptance = _positive(summary, "train/vllm/spec_acceptance_rate")
    if acceptance > 1:
        raise RuntimeError("acceptance rate exceeds one")
    if len(re.findall(r"draft_update_probe=complete", log)) < 2:
        raise RuntimeError("online arm requires two draft updates")
    for marker in (
        "draft_refit_manifest=",
        "draft_refit_load=complete",
        "draft_refit_finalize=complete",
    ):
        if log.count(marker) < 2:
            raise RuntimeError(f"missing two occurrences of {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--wandb-run-path", required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    import wandb

    error: Exception | None = None
    for _ in range(12):
        try:
            run = wandb.Api(timeout=60).run(args.wandb_run_path)
            validate(args.arm, dict(run.summary), args.log.read_text(errors="replace"))
            print(f"matrix_gate={args.arm}:complete")
            return
        except Exception as caught:
            error = caught
            time.sleep(10)
    raise RuntimeError(f"W&B gate evidence unavailable: {error}")


if __name__ == "__main__":
    main()
