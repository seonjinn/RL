"""Deterministic condition and dependency matrix for segmented Q8 runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex


@dataclass(frozen=True, slots=True)
class SegmentArm:
    label: str
    method: str
    arm: str
    seqs: int | None


@dataclass(frozen=True, slots=True)
class Segment:
    condition: SegmentArm
    previous_step: int
    stop_step: int


@dataclass(frozen=True, slots=True)
class SubmissionInputs:
    expected_head: str
    bundle: str
    bundle_sha: str
    result_parent: str
    account: str
    script: str
    log_dir: str


def build_segment_arms() -> tuple[SegmentArm, ...]:
    conditions = [
        SegmentArm("baseline-default", "none", "baseline", None),
        SegmentArm("baseline-s64", "none", "baseline", 64),
        SegmentArm("baseline-s128", "none", "baseline", 128),
    ]
    for method in ("dflash", "dspark"):
        for cadence in ("frozen", "fixed-10", "always"):
            for seqs in (64, 128):
                conditions.append(
                    SegmentArm(
                        f"{method}-{cadence}-s{seqs}",
                        method,
                        f"{method}-{cadence}",
                        seqs,
                    )
                )
    return tuple(conditions)


def stage_task_count(stage: int) -> int:
    if not 1 <= stage <= 20:
        raise ValueError("stage must be in [1, 20]")
    return 15 if stage <= 15 else 3


def segment_for(*, task_index: int, stage: int) -> Segment:
    count = stage_task_count(stage)
    if not 0 <= task_index < count:
        raise ValueError(f"task index {task_index} is invalid for stage {stage}")
    condition = build_segment_arms()[task_index]
    interval = 15 if condition.method == "none" else 20
    return Segment(
        condition=condition,
        previous_step=(stage - 1) * interval,
        stop_step=stage * interval,
    )


def build_stage_command(
    inputs: SubmissionInputs,
    *,
    stage: int,
    dependency: str | None,
) -> list[str]:
    task_count = stage_task_count(stage)
    log_dir = Path(inputs.log_dir)
    command = [
        "sbatch",
        "--parsable",
        f"--array=0-{task_count - 1}",
        f"--account={inputs.account}",
        "--partition=batch",
        "--qos=normal",
        "--nodes=1",
        "--ntasks=1",
        "--exclusive",
        "--gres=gpu:4",
        "--segment=1",
        "--time=04:00:00",
        f"--job-name=q8-300-s{stage:02d}",
        f"--output={log_dir}/stage-{stage:02d}-%A_%a.out",
        f"--error={log_dir}/stage-{stage:02d}-%A_%a.err",
    ]
    if dependency is not None:
        command.extend(
            [
                f"--dependency=aftercorr:{dependency}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    command.extend(
        [
            inputs.script,
            inputs.expected_head,
            inputs.bundle,
            inputs.bundle_sha,
            inputs.result_parent,
            str(stage),
        ]
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    segment = segment_for(task_index=args.task_index, stage=args.stage)
    if not args.shell:
        raise ValueError("only --shell output is supported")
    values = {
        "condition_label": segment.condition.label,
        "method": segment.condition.method,
        "arm": segment.condition.arm,
        "seqs": "default" if segment.condition.seqs is None else segment.condition.seqs,
        "previous_step": segment.previous_step,
        "stop_step": segment.stop_step,
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
