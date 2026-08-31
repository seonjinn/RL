#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


PTV2_ROOT = "/lustre/fsw/portfolios/coreai/users/sna/specdec_ptv23/ptv2_final"


@dataclass(frozen=True)
class Row:
    domain: str
    arm: str
    drafter_checkpoint: str | None
    method: str | None
    k: int
    training_mode: str = "frozen"
    policy_draft_enabled: bool = False
    draft_refit_enabled: bool = False
    runtime_cohort: str = "stable-vllm-0.25.1"
    status: str = "ready"


def checkpoint(domain: str, drafter: str) -> str:
    target = "base" if domain == "math" else "thinking"
    return (
        f"{PTV2_ROOT}/sd2en-q30-{target}-ptv2en-{drafter}-b8-16n/"
        "exported-checkpoint-25391"
    )


def rows() -> list[Row]:
    return [
        Row("math", "baseline", None, None, 0),
        Row("math", "dflash_k7", checkpoint("math", "dflash"), "dflash", 7),
        Row("math", "dspark_k5", checkpoint("math", "dspark"), "dspark", 5),
        Row(
            "math",
            "dflash2_k7",
            checkpoint("math", "dflash2"),
            "dflash",
            7,
            runtime_cohort="dflash2-vllm-pr52816",
            status="runtime-preflight",
        ),
        Row("swe", "baseline", None, None, 0),
        Row("swe", "dflash_k3", checkpoint("swe", "dflash"), "dflash", 3),
        Row("swe", "dflash_k5", checkpoint("swe", "dflash"), "dflash", 5),
        Row("swe", "dflash_k7", checkpoint("swe", "dflash"), "dflash", 7),
        Row("swe", "dspark_k3", checkpoint("swe", "dspark"), "dspark", 3),
        Row("swe", "dspark_k5", checkpoint("swe", "dspark"), "dspark", 5),
        Row(
            "swe",
            "dflash2_k7",
            checkpoint("swe", "dflash2"),
            "dflash",
            7,
            runtime_cohort="dflash2-vllm-pr52816",
            status="runtime-preflight",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = {
        "name": "qwen3-30ba3b-ptv2-frozen-first-20step",
        "math_contract": "20 optimizer steps, official 4n4g performance recipe",
        "swe_contract": "20 fixed validation instances, rollout-only",
        "rows": [asdict(row) for row in rows()],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for row in rows():
        print(f"{row.domain:4} {row.arm:12} {row.status:18} {row.drafter_checkpoint or '-'}")


if __name__ == "__main__":
    main()
