"""Render, validate, and submit the matched Qwen3-235B deep-refit GPU gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from experiments.q235_rp25_perf_20260917.launch import (
    ARMS,
    RECIPE,
    SITES,
    _drafter_path,
    configuration,
    render,
    required_inputs,
    sbatch_arguments,
)


COHORT = {
    "baseline": False,
    "dflash_k7": True,
}


def _manifest(
    *,
    source: Path,
    site: str,
    account: str,
    arm: str,
    run_name: str,
) -> dict[str, object]:
    spec = SITES[site]
    deep_refit = COHORT[arm]
    source_sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    overrides = configuration(
        steps=3,
        site=site,
        arm=arm,
        deep_refit=deep_refit,
    )
    overrides["logger.wandb.name"] = run_name
    manifest: dict[str, object] = {
        "account": account,
        "arm": arm,
        "container": str(spec.container),
        "deep_refit": deep_refit,
        "drafter": None,
        "overrides": overrides,
        "partition": spec.partition,
        "recipe": str(RECIPE),
        "run_name": run_name,
        "site": site,
        "source": str(source),
        "source_sha": source_sha,
        "steps": 3,
        "target": str(spec.target),
    }
    if ARMS[arm].method is not None:
        manifest["drafter"] = str(_drafter_path(site, ARMS[arm]))
    return manifest


def _validate_submission_inputs(source: Path, site: str, arm: str) -> None:
    missing = [str(path) for path in required_inputs(site, arm) if not path.is_file()]
    if missing:
        raise RuntimeError("missing required input: " + ", ".join(missing))
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not exported")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise RuntimeError(f"source checkout is dirty:\n{status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--arm", choices=("baseline", "dflash_k7", "both"), default="both"
    )
    parser.add_argument("--site", choices=tuple(SITES), default="ptyche")
    parser.add_argument("--account")
    args = parser.parse_args()

    source = Path(
        os.environ.get("Q235_SOURCE", "/home/sna/nemorl-q235-deep-refit-20260917")
    )
    spec = SITES[args.site]
    account = args.account or spec.default_account
    selected_arms = tuple(COHORT) if args.arm == "both" else (args.arm,)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = spec.base / "experiments/q235-specdec-deep-refit-20260917"

    for arm in selected_arms:
        lifecycle_label = "DeepRefit" if COHORT[arm] else "Legacy"
        run_name = f"Qwen3-235B-{ARMS[arm].label}-{lifecycle_label}-3step-{stamp}"
        run_dir = artifact_root / run_name
        script = render(
            account=account,
            run_name=run_name,
            steps=3,
            site=args.site,
            arm=arm,
            directory=run_dir,
            deep_refit=COHORT[arm],
        )
        if args.render:
            print(script)
            continue

        _validate_submission_inputs(source, args.site, arm)
        run_dir.mkdir(parents=True, exist_ok=False)
        (spec.artifacts / "shared-megatron-initial-checkpoint").mkdir(
            parents=True, exist_ok=True
        )
        manifest = _manifest(
            source=source,
            site=args.site,
            account=account,
            arm=arm,
            run_name=run_name,
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        job = run_dir / "job.sbatch"
        job.write_text(script)
        job.chmod(0o700)

        test_only = subprocess.run(
            sbatch_arguments(job, test_only=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "test-only.txt").write_text(test_only.stdout)
        print(test_only.stdout, end="", flush=True)
        test_only.check_returncode()
        if args.submit:
            submission = subprocess.run(
                sbatch_arguments(job, test_only=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            (run_dir / "submission.txt").write_text(submission.stdout)
            print(submission.stdout, end="", flush=True)
            submission.check_returncode()


if __name__ == "__main__":
    main()
