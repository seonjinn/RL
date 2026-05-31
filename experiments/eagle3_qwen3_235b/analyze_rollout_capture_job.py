#!/usr/bin/env python3
"""Analyze a submitted Qwen3 SWE rollout-capture job and corpus artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from analyze_rollout_capture import (
    default_output_data,
    default_rollout_log_dir,
    inspect_train_data,
    overall_status as artifact_status,
    recommendation,
    train_data_files,
    validate_output,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_REPO_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)

FAIL_RE = re.compile(
    r"\b("
    r"traceback|runtimeerror|valueerror|assertionerror|exception|"
    r"failed|fail\b|error\b|outofmemory|oom|killed|cancelled|timeout|"
    r"slurmstepd: error|segmentation fault|raytaskerror"
    r")",
    re.IGNORECASE,
)

BENIGN_RE = re.compile(
    r"(failures?:\s*0|failure_count['\"]?:\s*0|filtered_rewards|FAIL_TO_PASS|PASS_TO_PASS)",
    re.IGNORECASE,
)

SUCCESS_MARKERS = [
    "log_batched_dict_as_jsonl",
    "train_data_step",
    "Training finished",
    "Training complete",
    "max_num_steps",
]

ROOT_CAUSE_PATTERNS = [
    (re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"), "missing python module: {match}"),
    (re.compile(r"ImportError: (.+)"), "import error: {match}"),
    (re.compile(r"CUDA out of memory", re.IGNORECASE), "CUDA out of memory"),
    (re.compile(r"torch\._C.*AcceleratorError"), "torch/native-library skew"),
    (re.compile(r"No such file or directory: ['\"]([^'\"]+)['\"]"), "missing file: {match}"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--repo-root", type=Path, default=Path(os.environ.get("SWE_REPO_ROOT", DEFAULT_REPO_ROOT)))
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--rollout-log-dir", type=Path)
    parser.add_argument("--output-data", type=Path)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--sample-lines", type=int, default=200)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--infer-flat-content-roles", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-failure", action="store_true")
    return parser.parse_args()


def read_text(path: Path | None, limit_bytes: int = 1_000_000) -> str:
    if path is None or not path.exists():
        return ""
    with path.open("rb") as fh:
        data = fh.read(limit_bytes)
    return data.decode("utf-8", errors="replace")


def read_job_id(args: argparse.Namespace) -> tuple[str | None, Path]:
    job_file = args.job_file or args.repo_root / "latest_235b_swe_job_id.txt"
    if args.job_id:
        return args.job_id, job_file
    if not job_file.exists():
        return None, job_file
    text = job_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\b(\d+)\b", text)
    return (match.group(1) if match else text.strip() or None), job_file


def run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return {"available": False, "command": cmd[0], "output": ""}
    return {
        "available": True,
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def slurm_state(job_id: str | None) -> dict[str, Any]:
    if not job_id or not job_id.isdigit():
        return {"job_id": job_id, "status": "not_submitted"}
    squeue = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"])
    state = [
        line.strip()
        for line in (squeue.get("output") or "").splitlines()
        if line.strip() and not line.lower().startswith(("slurm_", "squeue:"))
    ]
    if squeue.get("returncode") == 0 and state:
        return {"job_id": job_id, "status": "in_queue", "state": state[0], "squeue": squeue}
    sacct = run_command(["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-P", "-n"])
    rows = [line for line in (sacct.get("output") or "").splitlines() if line.strip()]
    primary = rows[0].split("|") if rows else []
    status = "completed_or_unknown"
    if len(primary) >= 2 and primary[1]:
        status = primary[1].lower()
    return {"job_id": job_id, "status": status, "squeue": squeue, "sacct": sacct, "sacct_rows": rows[:12]}


def collect_logs(repo_root: Path, job_id: str | None) -> list[Path]:
    if not job_id:
        return []
    paths: list[Path] = []
    candidates = [
        repo_root / f"monitor_235b_swe_{job_id}.log",
        repo_root / f"{job_id}-logs/ray-driver.log",
    ]
    log_dir = repo_root / f"{job_id}-logs"
    if log_dir.exists():
        candidates.extend(sorted(log_dir.glob("*.log")))
        candidates.extend(sorted(log_dir.glob("**/*.log"))[:40])
    candidates.extend(sorted(repo_root.glob(f"*{job_id}*.out")))
    candidates.extend(sorted(repo_root.glob(f"*{job_id}*.err")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def useful_tail(text: str, limit: int = 18) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def has_failure(text: str) -> bool:
    for line in text.splitlines():
        if BENIGN_RE.search(line):
            continue
        if FAIL_RE.search(line):
            return True
    return False


def root_causes(text: str, max_items: int = 6) -> list[str]:
    causes: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if BENIGN_RE.search(line):
            continue
        for pattern, label in ROOT_CAUSE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1).strip() if match.groups() else ""
            cause = label.format(match=value)
            if cause not in seen:
                seen.add(cause)
                causes.append(cause)
            break
        if len(causes) >= max_items:
            break
    return causes


def log_summary(logs: list[Path]) -> dict[str, Any]:
    texts = {str(path): read_text(path) for path in logs[:24]}
    combined = "\n".join(texts.values())
    return {
        "files": list(texts),
        "file_count": len(logs),
        "has_failure_text": has_failure(combined),
        "root_causes": root_causes(combined),
        "success_markers": [marker for marker in SUCCESS_MARKERS if marker in combined],
        "tail": useful_tail(combined),
    }


def analyze_artifacts(args: argparse.Namespace, rollout_log_dir: Path, output_data: Path) -> dict[str, Any]:
    files = train_data_files(rollout_log_dir)
    train = inspect_train_data(files, args)
    validation_json = args.validation_json or output_data.with_suffix(".validation.json")
    output = validate_output(output_data, validation_json, args.max_seq_len)
    return {
        "overall_status": artifact_status(train, output),
        "rollout_log_dir": str(rollout_log_dir),
        "train_data": train,
        "output_data": output,
        "recommendation": recommendation(args, rollout_log_dir, output_data),
    }


def decide(slurm: dict[str, Any], logs: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
    artifact_overall = artifacts["overall_status"]
    if slurm.get("status") == "in_queue":
        return "running", f"Slurm job is still {slurm.get('state')}"
    if logs.get("root_causes"):
        return "fail", "; ".join(logs["root_causes"])
    if logs["has_failure_text"]:
        return "fail", "failure-like text found in rollout capture logs"
    if artifact_overall == "pass":
        return "pass", "materialized rollout corpus validates"
    if artifact_overall == "needs_materialize":
        return "needs_materialize", "train_data_step JSONL exists and should be materialized"
    if slurm.get("status") in {"failed", "cancelled", "timeout", "out_of_memory"}:
        return "fail", f"Slurm job ended as {slurm.get('status')}"
    if artifact_overall == "missing_capture" and slurm.get("job_id"):
        return "missing_capture", "job id exists but rollout train_data_step JSONL is not present"
    return "not_submitted", "no rollout capture job id or train_data_step JSONL is visible"


def render_markdown(data: dict[str, Any]) -> str:
    artifacts = data["artifacts"]
    rec = artifacts["recommendation"]
    status = data["overall_status"]
    lines = [
        "# Rollout Capture Job Analysis",
        "",
        f"Overall: **{status.upper()}**",
        "",
        f"Detail: {data['detail']}",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| job id | `{data['job'].get('job_id') or '-'}` |",
        f"| slurm status | `{data['slurm'].get('status')}` |",
        f"| rollout log dir | `{artifacts['rollout_log_dir']}` |",
        f"| train files | {artifacts['train_data']['file_count']} |",
        f"| extractable conversations | {artifacts['train_data']['extractable_conversations']} |",
        f"| output status | {artifacts['output_data']['status']} |",
        f"| log files | {data['logs']['file_count']} |",
        f"| log failure text | {data['logs']['has_failure_text']} |",
        f"| root cause | {', '.join(data['logs'].get('root_causes') or ['-'])} |",
        "",
        "## Next Command",
        "",
        "```bash",
    ]
    if status == "needs_materialize":
        lines.append(rec["materialize_command"])
    elif status == "pass":
        lines.append(rec["pipeline_dry_run_command"])
    elif status == "running":
        lines.append(f"python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py --job-id {data['job'].get('job_id')}")
    else:
        lines.append(rec["capture_plan_command"])
    lines += ["```", ""]
    tail = data["logs"].get("tail") or []
    if tail:
        lines += ["## Log Tail", "", "```text"]
        lines.extend(tail)
        lines += ["```", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rollout_log_dir = default_rollout_log_dir(args)
    output_data = default_output_data(args)
    job_id, job_file = read_job_id(args)
    slurm = slurm_state(job_id)
    logs = log_summary(collect_logs(args.repo_root, job_id))
    artifacts = analyze_artifacts(args, rollout_log_dir, output_data)
    overall, detail = decide(slurm, logs, artifacts)
    data = {
        "overall_status": overall,
        "detail": detail,
        "artifact_root": str(args.artifact_root),
        "repo_root": str(args.repo_root),
        "job": {"job_id": job_id, "job_file": str(job_file), "job_file_exists": job_file.exists()},
        "slurm": slurm,
        "logs": logs,
        "artifacts": artifacts,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    text = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    print(text)
    return 1 if args.fail_on_failure and overall == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
