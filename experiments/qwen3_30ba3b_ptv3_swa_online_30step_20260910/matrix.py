#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


PTV3_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "specdec_ptv23/ptv3_swa"
)


@dataclass(frozen=True)
class Row:
    arm: str
    method: str
    k: int
    fixed_interval: int
    drafter_checkpoint: str
    max_steps: int = 30
    training_mode: str = "online-fixed"
    runtime_cohort: str = "stable-vllm-0.25.1-fap"


def checkpoint(method: str) -> str:
    return (
        f"{PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-{method}-b8-16n/"
        "exported-checkpoint-44000"
    )


def rows() -> list[Row]:
    return [
        Row(
            arm=f"{method}_k{k}_fixed{interval}",
            method=method,
            k=k,
            fixed_interval=interval,
            drafter_checkpoint=checkpoint(method),
        )
        for method, k in (("dflash", 5), ("dspark", 7))
        for interval in (5, 10)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = {
        "name": "qwen3-30ba3b-ptv3-swa-online-30step",
        "contract": "30 optimizer steps, official 4n4g performance recipe",
        "rows": [asdict(row) for row in rows()],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for row in rows():
        print(
            f"{row.arm:22} {row.training_mode:12} "
            f"{row.drafter_checkpoint}"
        )


if __name__ == "__main__":
    main()
