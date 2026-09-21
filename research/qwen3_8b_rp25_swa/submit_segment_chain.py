"""Submit fail-closed correlated arrays for the 300-step Q8 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from research.qwen3_8b_rp25_swa.segment_matrix import (
    SubmissionInputs,
    build_stage_command,
)


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _job_id(receipt: str) -> str:
    value = receipt.split(";", 1)[0].strip()
    if not value.isdigit():
        raise RuntimeError(f"unexpected sbatch receipt: {receipt!r}")
    return value


def submit(
    inputs: SubmissionInputs,
    *,
    specdec_only: bool = False,
) -> dict[str, object]:
    result_parent = Path(inputs.result_parent)
    if result_parent.exists():
        raise FileExistsError(f"result parent must be fresh: {result_parent}")
    Path(inputs.log_dir).mkdir(parents=True, exist_ok=False)

    preflight = []
    task_range = (3, 14) if specdec_only else None
    preflight_stages = (1,) if specdec_only else (1, 16)
    final_stage = 15 if specdec_only else 20
    for stage in preflight_stages:
        command = build_stage_command(
            inputs,
            stage=stage,
            dependency=None,
            task_range=task_range,
        )
        test_command = [*command[:1], "--test-only", *command[2:]]
        preflight.append({"stage": stage, "receipt": _run(test_command)})

    stages = []
    dependency: str | None = None
    for stage in range(1, final_stage + 1):
        command = build_stage_command(
            inputs,
            stage=stage,
            dependency=dependency,
            task_range=task_range,
        )
        receipt = _run(command)
        job_id = _job_id(receipt)
        stages.append(
            {
                "stage": stage,
                "job_id": job_id,
                "dependency": dependency,
                "command": command,
            }
        )
        dependency = job_id
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "specdec-only" if specdec_only else "full-matrix",
        "preflight": preflight,
        "stages": stages,
    }
    receipt_path = result_parent / "submission-receipt.json"
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha", required=True)
    parser.add_argument("--result-parent", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--specdec-only", action="store_true")
    parser.add_argument(
        "--script",
        default="research/qwen3_8b_rp25_swa/run_segment_array.sbatch",
    )
    args = parser.parse_args()
    inputs = SubmissionInputs(
        expected_head=args.expected_head,
        bundle=args.bundle,
        bundle_sha=args.bundle_sha,
        result_parent=args.result_parent,
        account=args.account,
        script=args.script,
        log_dir=f"{args.result_parent}/scheduler-logs",
    )
    payload = submit(inputs, specdec_only=args.specdec_only)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
