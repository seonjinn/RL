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
    drafter_checkpoint: str | None
    method: str | None
    k: int
    status: str = "ready"
    training_mode: str = "frozen"
    policy_draft_enabled: bool = False
    draft_refit_enabled: bool = False
    runtime_cohort: str = "stable-vllm-0.25.1-fap"


def checkpoint(method: str) -> str:
    return (
        f"{PTV3_ROOT}/sd2p3swa-q30-base-ptv3swe-{method}-b8-16n/"
        "exported-checkpoint-44000"
    )


def rows() -> list[Row]:
    result = [Row("baseline", None, None, 0)]
    for method in ("dflash", "dspark"):
        result.extend(
            Row(f"{method}_k{k}", checkpoint(method), method, k)
            for k in (3, 5, 7)
        )
    result.append(
        Row(
            "dflash2_k7",
            checkpoint("dflash2"),
            "dflash",
            7,
            status="blocked-vllm-0.28",
            runtime_cohort="dflash2-vllm-0.28-pending",
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = {
        "name": "qwen3-30ba3b-ptv3-swa-frozen-20step",
        "contract": "20 optimizer steps, official 4n4g performance recipe",
        "rows": [asdict(row) for row in rows()],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for row in rows():
        print(f"{row.arm:12} {row.status:20} {row.drafter_checkpoint or '-'}")


if __name__ == "__main__":
    main()
