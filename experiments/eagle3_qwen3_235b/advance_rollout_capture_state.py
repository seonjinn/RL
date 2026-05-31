#!/usr/bin/env python3
"""Advance the Qwen3 rollout-capture state after a capture job finishes.

Default behavior is no-submit and no-write except reports. With --materialize,
the script converts visible train_data_step*.jsonl files into the rollout
conversation JSONL, then refreshes rollout/corpus reports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_SPECDEC_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)


def parse_args() -> argparse.Namespace:
    artifact_default = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_default)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("SWE_REPO_ROOT") or os.environ.get("REPO_ROOT") or DEFAULT_SPECDEC_RL_DIR),
    )
    parser.add_argument(
        "--rollout-log-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ROLLOUT_LOG_DIR",
                artifact_default / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke",
            )
        ),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path(
            os.environ.get(
                "ROLLOUT_CONVERSATIONS",
                artifact_default / "data/qwen3_235b_swe_rollout_conversations.jsonl",
            )
        ),
    )
    parser.add_argument("--job-id", help="Explicit Slurm job id for the rollout capture job.")
    parser.add_argument("--job-file", type=Path, help="Optional job id file passed to analyze_rollout_capture_job.py.")
    parser.add_argument(
        "--report-prefix",
        help=(
            "Optional prefix for subordinate rollout reports. Use this for "
            "fallback/alternate capture state refreshes that should not "
            "overwrite canonical rollout_capture_* reports."
        ),
    )
    parser.add_argument("--target-context", choices=("swe_rl", "math", "general"), default=os.environ.get("EAGLE3_TARGET_CONTEXT", "swe_rl"))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "coreai_dlalgo_nemorl"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--materialize", action="store_true", help="Run materialize_rollout_capture_corpus.sh when train_data exists.")
    parser.add_argument("--run-bootstrap-dry-run", action="store_true", help="Run bootstrap_eagle3_path.sh with SUBMIT=false after corpus PASS.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def shell_join(env: dict[str, str], command: list[str]) -> str:
    return " ".join([*(f"{k}={shlex.quote(v)}" for k, v in env.items()), *(shlex.quote(x) for x in command)])


def run(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": shell_join(env or {}, cmd),
        "returncode": result.returncode,
        "output_tail": result.stdout[-6000:],
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def reports(args: argparse.Namespace) -> dict[str, Path]:
    root = args.artifact_root / "reports"
    stem = args.report_prefix or "rollout_capture"
    return {
        "job_json": root / f"{stem}_job_analysis.json",
        "job_md": root / f"{stem}_job_analysis.md",
        "artifact_json": root / f"{stem}_analysis.json",
        "artifact_md": root / f"{stem}_analysis.md",
        "corpus_json": root / ("corpus_strategy.json" if args.report_prefix is None else f"{stem}_corpus_strategy.json"),
        "corpus_md": root / ("corpus_strategy.md" if args.report_prefix is None else f"{stem}_corpus_strategy.md"),
        "pipeline_json": root / "eagle3_pipeline_analysis.json",
        "pipeline_md": root / "eagle3_pipeline_analysis.md",
    }


def analyze_job(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py",
        "--artifact-root",
        str(args.artifact_root),
        "--repo-root",
        str(args.repo_root),
        "--rollout-log-dir",
        str(args.rollout_log_dir),
        "--output-data",
        str(args.output_data),
        "--markdown-out",
        str(paths["job_md"]),
        "--json-out",
        str(paths["job_json"]),
    ]
    if args.job_id:
        cmd.extend(["--job-id", args.job_id])
    if args.job_file:
        cmd.extend(["--job-file", str(args.job_file)])
    result = run(cmd)
    payload = load_json(paths["job_json"]) if paths["job_json"].exists() else {}
    return payload, result


def analyze_artifacts(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/analyze_rollout_capture.py",
            "--artifact-root",
            str(args.artifact_root),
            "--rollout-log-dir",
            str(args.rollout_log_dir),
            "--output-data",
            str(args.output_data),
            "--markdown-out",
            str(paths["artifact_md"]),
            "--json-out",
            str(paths["artifact_json"]),
        ]
    )
    payload = load_json(paths["artifact_json"]) if paths["artifact_json"].exists() else {}
    return payload, result


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "ROLLOUT_LOG_DIR": str(args.rollout_log_dir),
        "OUTPUT_DATA": str(args.output_data),
    }
    return run(["bash", "experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh"], env=env)


def analyze_corpus(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py",
            "--artifact-root",
            str(args.artifact_root),
            "--target-context",
            args.target_context,
            "--input-data",
            str(args.output_data),
            "--rollout-capture-analysis-json",
            str(paths["artifact_json"]),
            "--markdown-out",
            str(paths["corpus_md"]),
            "--json-out",
            str(paths["corpus_json"]),
        ]
    )
    payload = load_json(paths["corpus_json"]) if paths["corpus_json"].exists() else {}
    return payload, result


def bootstrap_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "INPUT_DATA": str(args.output_data),
        "SUBMIT": "false",
        "RUN_PILOT": "true",
        "PREP_DRY_RUN": "true",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "SWE_REPO_ROOT": str(args.repo_root),
    }
    return run(["bash", "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh"], env=env)


def commands(args: argparse.Namespace) -> dict[str, str]:
    capture_script = "run_math_rollout_capture_smoke.sh" if args.target_context == "math" else "run_rollout_capture_smoke.sh"
    base_env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "SWE_REPO_ROOT": str(args.repo_root),
        "ROLLOUT_LOG_DIR": str(args.rollout_log_dir),
        "ROLLOUT_CONVERSATIONS": str(args.output_data),
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "EAGLE3_TARGET_CONTEXT": args.target_context,
    }
    rollout_submit_preflight = load_json_if_exists(args.artifact_root / "reports/rollout_capture_submit_preflight.json")
    preflight_submit = (
        rollout_submit_preflight.get("commands", {}).get("submit")
        if isinstance(rollout_submit_preflight.get("commands"), dict)
        else None
    )
    return {
        "submit_capture": str(preflight_submit)
        if preflight_submit
        else shell_join(
            {**base_env, "DRY_RUN": "false", "MAX_NUM_STEPS": "1"},
            ["bash", f"experiments/eagle3_qwen3_235b/{capture_script}"],
        ),
        "poll": shell_join(
            base_env,
            [
                "python3",
                "experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py",
                "--target-context",
                args.target_context,
                *(["--job-id", args.job_id] if args.job_id else []),
                *(["--report-prefix", args.report_prefix] if args.report_prefix else []),
            ],
        ),
        "materialize": shell_join(
            {"ARTIFACT_ROOT": str(args.artifact_root), "ROLLOUT_LOG_DIR": str(args.rollout_log_dir), "OUTPUT_DATA": str(args.output_data)},
            ["bash", "experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh"],
        ),
        "materialize_and_refresh": shell_join(
            base_env,
            [
                "python3",
                "experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py",
                "--materialize",
                "--target-context",
                args.target_context,
                *(["--job-id", args.job_id] if args.job_id else []),
                *(["--report-prefix", args.report_prefix] if args.report_prefix else []),
            ],
        ),
        "pipeline_dry_run": shell_join(
            {
                "ARTIFACT_ROOT": str(args.artifact_root),
                "INPUT_DATA": str(args.output_data),
                "SUBMIT": "false",
                "RUN_PILOT": "true",
                "EAGLE3_TARGET_CONTEXT": args.target_context,
                "SBATCH_ACCOUNT": args.sbatch_account,
                "SBATCH_PARTITION": args.sbatch_partition,
            },
            ["bash", "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh"],
        ),
    }


def decide(job_status: str, artifact_status: str, corpus_status: str | None, did_materialize: bool) -> dict[str, str]:
    if job_status == "fail" or artifact_status == "fail" or corpus_status == "fail":
        return {"overall_status": "fail", "next_step": "inspect_logs", "detail": "a rollout/corpus check failed"}
    if job_status == "running":
        return {"overall_status": "running", "next_step": "poll", "detail": "rollout capture job is still running or queued"}
    if job_status in {"not_submitted", "missing_capture"}:
        return {"overall_status": job_status, "next_step": "submit_capture", "detail": "no usable rollout capture train_data is visible yet"}
    if job_status == "needs_materialize" or artifact_status == "needs_materialize":
        if did_materialize:
            return {"overall_status": "needs_review", "next_step": "rerun", "detail": "materialization ran but corpus did not reach pass"}
        return {"overall_status": "needs_materialize", "next_step": "materialize", "detail": "train_data exists and should be converted to conversations"}
    if job_status == "pass" and artifact_status == "pass" and corpus_status == "pass":
        return {"overall_status": "pass", "next_step": "pipeline_dry_run", "detail": "rollout corpus validates and is ready for hidden-state pipeline"}
    return {"overall_status": "needs_review", "next_step": "inspect_reports", "detail": "state combination needs manual review"}


def render_markdown(data: dict[str, Any]) -> str:
    decision = data["decision"]
    command_key = decision["next_step"]
    command = data["commands"].get(command_key, "# inspect generated reports")
    lines = [
        "# Rollout Capture State Advance",
        "",
        f"Overall: **{decision['overall_status'].upper()}**",
        f"Next step: `{decision['next_step']}`",
        "",
        decision["detail"],
        "",
        "| report | status |",
        "| --- | --- |",
        f"| rollout job | {data['job'].get('overall_status')} |",
        f"| rollout artifacts | {data['artifacts'].get('overall_status')} |",
        f"| corpus strategy | {(data.get('corpus_strategy') or {}).get('overall_status', 'not_run')} |",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paths = reports(args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifact_root / "reports/advance_rollout_capture_state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w", encoding="utf-8")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)

    job, job_run = analyze_job(args, paths)
    artifacts, artifact_run = analyze_artifacts(args, paths)
    materialize_run = None
    if args.materialize and artifacts.get("overall_status") == "needs_materialize":
        materialize_run = materialize(args)
        job, job_run = analyze_job(args, paths)
        artifacts, artifact_run = analyze_artifacts(args, paths)
    corpus = None
    corpus_run = None
    if artifacts.get("overall_status") == "pass":
        corpus, corpus_run = analyze_corpus(args, paths)
    bootstrap_run = None
    if args.run_bootstrap_dry_run and artifacts.get("overall_status") == "pass" and (corpus or {}).get("overall_status") == "pass":
        bootstrap_run = bootstrap_dry_run(args)

    decision = decide(
        str(job.get("overall_status")),
        str(artifacts.get("overall_status")),
        (corpus or {}).get("overall_status"),
        materialize_run is not None,
    )
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "repo_root": str(args.repo_root),
        "rollout_log_dir": str(args.rollout_log_dir),
        "output_data": str(args.output_data),
        "decision": decision,
        "job": job,
        "artifacts": artifacts,
        "corpus_strategy": corpus,
        "commands": commands(args),
        "runs": {
            "job_analysis": job_run,
            "artifact_analysis": artifact_run,
            "materialize": materialize_run,
            "corpus_strategy": corpus_run,
            "bootstrap_dry_run": bootstrap_run,
        },
    }

    text = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    print(text)
    return 1 if decision["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
