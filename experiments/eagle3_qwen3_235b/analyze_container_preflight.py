#!/usr/bin/env python3
"""Analyze the preflight-only Slurm job for a selected Qwen3 Eagle3 container."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_FILE = ROOT / "latest_eagle3_container_preflight_job.txt"
DEFAULT_LOGS_DIR = ROOT / "logs"
JOB_NAME = "q235b-eagle3-preflight"

FAIL_RE = re.compile(
    r"\b("
    r"traceback|runtimeerror|valueerror|assertionerror|exception|"
    r"failed|fail\b|error\b|outofmemory|oom|killed|cancelled|timeout|"
    r"slurmstepd: error|segmentation fault"
    r")",
    re.IGNORECASE,
)

BENIGN_RE = re.compile(
    r"(Preflight passed|Recipe override validation passed|dry-run passed|"
    r"0 config checks failed|failures?:\s*0)",
    re.IGNORECASE,
)

SUCCESS_MARKERS = [
    "Preflight passed.",
    "Recipe override validation passed",
    "chat template assistant mask validation passed",
    "validated assistant loss mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, default=DEFAULT_JOB_FILE)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--cluster-probe-json", type=Path, default=None)
    parser.add_argument("--pipeline-preflight-json", type=Path, default=None)
    parser.add_argument("--pipeline-preflight-markdown", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=os.environ.get("ARTIFACT_ROOT"))
    parser.add_argument("--modelopt-dir", type=Path, default=os.environ.get("MODELOPT_DIR"))
    parser.add_argument("--verifier-config-dir", type=Path, default=os.environ.get("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--input-data", type=Path, default=os.environ.get("INPUT_DATA"))
    parser.add_argument("--chat-template", type=Path, default=os.environ.get("CHAT_TEMPLATE"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", ""))
    parser.add_argument("--mounts", default=os.environ.get("MOUNTS", ""))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "<account>"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-failure", action="store_true")
    return parser.parse_args()


def parse_kv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_logs(logs_dir: Path, job_id: str | None) -> tuple[Path | None, Path | None]:
    if not job_id or not job_id.isdigit():
        return None, None
    exact_out = logs_dir / f"{JOB_NAME}_{job_id}.out"
    exact_err = logs_dir / f"{JOB_NAME}_{job_id}.err"
    if exact_out.exists() or exact_err.exists():
        return exact_out if exact_out.exists() else None, exact_err if exact_err.exists() else None
    out_matches = sorted(logs_dir.glob(f"*_{job_id}.out")) if logs_dir.exists() else []
    err_matches = sorted(logs_dir.glob(f"*_{job_id}.err")) if logs_dir.exists() else []
    return (out_matches[0] if out_matches else None, err_matches[0] if err_matches else None)


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


def success_markers(text: str) -> list[str]:
    return [marker for marker in SUCCESS_MARKERS if marker in text]


def load_probe(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided"}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    checks = payload.get("checks") or []
    failures = [item for item in checks if item.get("required") and item.get("status") == "fail"]
    return {
        "status": payload.get("overall_status", "unknown"),
        "path": str(path),
        "required_failures": failures,
        "checks": len(checks),
    }


def load_pipeline_preflight(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided"}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    return {
        "status": payload.get("overall_status", "unknown"),
        "path": str(path),
        "slurm_job_id": payload.get("slurm_job_id"),
        "counts": payload.get("counts"),
        "failures": payload.get("failures") or [],
    }


def shell_env_command(env: dict[str, str], script: str) -> str:
    return " ".join([f"{key}={shlex.quote(value)}" for key, value in env.items()] + ["bash", shlex.quote(script)])


def read_export_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        for token in tokens:
            if "=" not in token or token.startswith("-"):
                continue
            key, value = token.split("=", 1)
            if key.replace("_", "").isalnum() and key[:1].isalpha():
                values[key] = value
    return values


def build_submit_command(args: argparse.Namespace, job_values: dict[str, str]) -> str:
    artifact_root = args.artifact_root or ROOT / "outputs/qwen3_235b_eagle3"
    resource_profile_env = Path(artifact_root) / "reports/eagle3_resource_profile.env"
    resource_env = read_export_env(resource_profile_env)
    preflight_gpus = (
        job_values.get("preflight_gpus_per_node")
        or os.environ.get("PREFLIGHT_GPUS_PER_NODE")
        or resource_env.get("DUMP_GPUS_PER_NODE")
        or "1"
    )
    env = {
        "SUBMIT": "true",
        "ARTIFACT_ROOT": str(artifact_root),
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "RESOURCE_PROFILE_ENV": str(resource_profile_env),
        "PREFLIGHT_GPUS_PER_NODE": preflight_gpus,
    }
    optional = {
        "MODELOPT_DIR": args.modelopt_dir,
        "VERIFIER_CONFIG_DIR": args.verifier_config_dir,
        "INPUT_DATA": args.input_data,
        "CHAT_TEMPLATE": args.chat_template,
        "CONTAINER": args.container or job_values.get("container"),
        "MOUNTS": args.mounts,
        "PREFLIGHT_JSON": args.pipeline_preflight_json or job_values.get("preflight_json"),
        "PREFLIGHT_MARKDOWN": args.pipeline_preflight_markdown or job_values.get("preflight_markdown"),
    }
    env.update({key: str(value) for key, value in optional.items() if value})
    return shell_env_command(env, "experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    job_values = parse_kv_file(args.job_file)
    job_id = job_values.get("preflight_job")
    container = args.container or job_values.get("container", "")
    report = job_values.get("report")
    cluster_probe_json = args.cluster_probe_json
    if cluster_probe_json is None and report:
        report_path = Path(report)
        if report_path.suffix == ".md":
            cluster_probe_json = report_path.with_suffix(".json")
    pipeline_preflight_json = args.pipeline_preflight_json
    if pipeline_preflight_json is None and job_values.get("preflight_json"):
        pipeline_preflight_json = Path(job_values["preflight_json"])

    out_log, err_log = resolve_logs(args.logs_dir, job_id)
    out_text = read_text(out_log)
    err_text = read_text(err_log)
    combined = "\n".join(part for part in (out_text, err_text) if part)
    markers = success_markers(combined)
    failure = has_failure(combined)
    probe = load_probe(cluster_probe_json)
    pipeline_preflight = load_pipeline_preflight(pipeline_preflight_json)
    pipeline_status = pipeline_preflight.get("status")
    pipeline_job_id = str(pipeline_preflight.get("slurm_job_id") or "")
    stale_pipeline_report = bool(
        job_id
        and job_id.isdigit()
        and pipeline_job_id
        and pipeline_job_id != job_id
    )

    if not args.job_file.exists():
        status = "missing"
        detail = "container preflight job file is missing"
    elif not job_id:
        status = "missing"
        detail = "preflight_job is absent from job file"
    elif not job_id.isdigit():
        status = "planned"
        detail = "job file contains a dry-run placeholder; no Slurm job has been submitted"
    elif stale_pipeline_report:
        status = "fail"
        detail = f"structured preflight report is stale: report job {pipeline_job_id}, job file {job_id}"
    elif pipeline_status == "fail":
        status = "fail"
        detail = "structured preflight report contains failed checks"
    elif pipeline_status == "pass":
        status = "pass"
        detail = "structured preflight report is PASS"
    elif not out_log and not err_log:
        status = "missing"
        detail = f"no Slurm logs found for {JOB_NAME}_{job_id}"
    elif failure:
        status = "fail"
        detail = "failure-like text found in preflight logs"
    elif "Preflight passed." in markers:
        status = "pass"
        detail = "preflight success marker found in container Slurm logs"
    elif markers:
        status = "warn"
        detail = "partial success markers found, but final preflight marker is missing"
    else:
        status = "running_or_unknown"
        detail = "logs exist without failure text, but no preflight success marker was found"

    overall = "pass" if status == "pass" else ("fail" if status == "fail" else "incomplete")
    if probe.get("required_failures"):
        overall = "fail"

    if overall == "pass":
        next_action = {
            "summary": "Container preflight passed. Proceed to RUN_PILOT=true hidden-state dump/train/export dry-run or submit.",
            "submit_command": None,
            "notes": [
                "Reuse the same CONTAINER and MOUNTS for the full Eagle3 pipeline.",
                "Keep the first heavy run in RUN_PILOT=true mode before full hidden-state dump.",
            ],
        }
    elif status == "planned":
        next_action = {
            "summary": "Submit the preflight-only job after reviewing the dry-run command.",
            "submit_command": build_submit_command(args, job_values),
            "notes": [
                "This submits only slurm_preflight.sbatch, not hidden-state dump or training.",
                "The job proves the selected container can import ModelOpt and validate the chat template.",
            ],
        }
    else:
        next_action = {
            "summary": "Inspect or rerun the container preflight before any heavy Eagle3 stage.",
            "submit_command": build_submit_command(args, job_values),
            "notes": [
                "Do not start hidden-state dump until this analyzer reports PASS.",
                "If logs are missing, check squeue/sacct for the preflight job state first.",
            ],
        }

    return {
        "overall_status": overall,
        "job_file": str(args.job_file),
        "logs_dir": str(args.logs_dir),
        "job_name": JOB_NAME,
        "job_id": job_id,
        "status": status,
        "detail": detail,
        "container": container,
        "out_log": str(out_log) if out_log else None,
        "err_log": str(err_log) if err_log else None,
        "cluster_probe": probe,
        "pipeline_preflight": pipeline_preflight,
        "evidence": {
            "out_bytes": len(out_text),
            "err_bytes": len(err_text),
            "has_failure_text": failure,
            "success_markers": markers,
            "tail": useful_tail(combined),
        },
        "next_action": next_action,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Eagle3 Container Preflight Analysis",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| job id | `{payload.get('job_id') or '-'}` |",
        f"| status | `{payload.get('status')}` |",
        f"| detail | {payload.get('detail', '').replace('|', '/')} |",
        f"| container | `{payload.get('container') or '-'}` |",
        f"| out log | `{payload.get('out_log') or '-'}` |",
        f"| err log | `{payload.get('err_log') or '-'}` |",
        f"| cluster probe | `{payload.get('cluster_probe', {}).get('status')}` |",
        f"| structured preflight | `{payload.get('pipeline_preflight', {}).get('status')}` |",
        "",
        "## Next Action",
        "",
        payload.get("next_action", {}).get("summary", ""),
    ]
    notes = payload.get("next_action", {}).get("notes") or []
    if notes:
        lines += ["", "Notes:"]
        lines.extend(f"- {note}" for note in notes)
    command = payload.get("next_action", {}).get("submit_command")
    if command:
        lines += ["", "Submit command:", "", "```bash", command, "```"]
    tail = payload.get("evidence", {}).get("tail") or []
    if tail:
        lines += ["", "## Log Tail", "", "```text"]
        lines.extend(tail)
        lines.append("```")
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    payload = analyze(args)
    write_outputs(payload, args)
    if args.fail_on_failure and payload["overall_status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
