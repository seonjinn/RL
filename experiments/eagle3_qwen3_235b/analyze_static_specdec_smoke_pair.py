#!/usr/bin/env python3
"""Analyze a submitted baseline/static-Eagle3 NeMo-RL smoke pair.

The submitter writes ``latest_static_specdec_smoke_jobs.txt`` with:

    baseline_job=<jobid>
    specdec_job=<jobid>

This wrapper resolves those job ids to ``<repo>/<jobid>-logs/ray-driver.log``
and then delegates to ``analyze_specdec_smoke.py`` with the standard gate.
Explicit log paths can be supplied for already-copied logs or synthetic tests.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_JOB_FILE = ROOT / "latest_static_specdec_smoke_jobs.txt"
DEFAULT_CLUSTER_REPO = Path(
    os.environ.get(
        "SMOKE_REPO_ROOT",
        os.environ.get(
            "NEMO_RL_REPO_ROOT",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
        ),
    )
)


def parse_job_file(path: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    if not path.exists():
        return jobs
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        jobs[key.strip()] = value.strip()
    return jobs


def job_log_path(repo_root: Path, job_id: str) -> Path:
    if not job_id or not job_id.isdigit():
        raise ValueError(f"invalid Slurm job id: {job_id!r}")
    ray_driver = repo_root / f"{job_id}-logs" / "ray-driver.log"
    if ray_driver.exists():
        return ray_driver
    return repo_root / f"{job_id}-logs"


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} log path is not visible: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, default=DEFAULT_JOB_FILE)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_CLUSTER_REPO)
    parser.add_argument("--baseline-job")
    parser.add_argument("--specdec-job")
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--specdec-log", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--drop-first-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop the lowest step id. Leave disabled for the default 1-step smoke.",
    )
    parser.add_argument("--gen-outlier-threshold-s", type=float, default=800.0)
    parser.add_argument("--min-generation-speedup-pct", type=float, default=10.0)
    parser.add_argument("--min-acceptance-rate", type=float, default=0.45)
    parser.add_argument(
        "--fail-on-missing-spec-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print-command-only",
        action="store_true",
        help="Only print the delegated analyze_specdec_smoke.py command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = parse_job_file(args.job_file)

    baseline_job = args.baseline_job or jobs.get("baseline_job")
    specdec_job = args.specdec_job or jobs.get("specdec_job")

    try:
        baseline_log = (
            existing_path(args.baseline_log, "baseline")
            if args.baseline_log
            else existing_path(job_log_path(args.repo_root, baseline_job or ""), "baseline")
        )
        specdec_log = (
            existing_path(args.specdec_log, "specdec")
            if args.specdec_log
            else existing_path(job_log_path(args.repo_root, specdec_job or ""), "specdec")
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        print(
            "Hint: pass --baseline-log/--specdec-log explicitly, or run after both jobs "
            "have created <jobid>-logs/ray-driver.log.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        str(EXP / "analyze_specdec_smoke.py"),
        str(specdec_log),
        "--baseline",
        str(baseline_log),
        "--gen-outlier-threshold-s",
        str(args.gen_outlier_threshold_s),
        "--min-generation-speedup-pct",
        str(args.min_generation_speedup_pct),
        "--min-acceptance-rate",
        str(args.min_acceptance_rate),
    ]
    if args.drop_first_step:
        cmd.append("--drop-first-step")
    if args.fail_on_missing_spec_metrics:
        cmd.append("--fail-on-missing-spec-metrics")
    else:
        cmd.append("--no-fail-on-missing-spec-metrics")
    if args.markdown_out:
        cmd += ["--markdown-out", str(args.markdown_out)]
    if args.json_out:
        cmd += ["--json-out", str(args.json_out)]

    print(" ".join(cmd))
    if args.print_command_only:
        return 0

    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
