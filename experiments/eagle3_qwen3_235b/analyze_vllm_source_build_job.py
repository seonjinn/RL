#!/usr/bin/env python3
"""Summarize the vLLM source-build Slurm job and target-site evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_JOB_FILE = ROOT / "latest_vllm_native_source_build_job.txt"
DEFAULT_LOGS_DIR = ROOT / "logs"
JOB_NAME = "q235b-vllm-build"

FAIL_RE = re.compile(
    r"\b(traceback|runtimeerror|valueerror|assertionerror|exception|failed|fail\b|"
    r"error\b|outofmemory|oom|killed|cancelled|timeout|slurmstepd: error|segmentation fault)",
    re.IGNORECASE,
)
BENIGN_RE = re.compile(r"(still running|source build|failures?:\s*0|failure_count['\"]?:\s*0)", re.IGNORECASE)
ROOT_CAUSE_PATTERNS = [
    (re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"), "missing python module: {match}"),
    (re.compile(r"ImportError: (.+)"), "import error: {match}"),
    (re.compile(r"undefined symbol: ([^\s`]+)"), "undefined symbol: {match}"),
    (re.compile(r"CUDA out of memory", re.IGNORECASE), "CUDA out of memory"),
    (re.compile(r"FAILED: (.+)"), "ninja/cmake failed target: {match}"),
    (re.compile(r"subprocess-exited-with-error"), "pip build subprocess exited with error"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--job-file", type=Path, default=DEFAULT_JOB_FILE)
    parser.add_argument("--job-id")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--source-build-json", type=Path)
    parser.add_argument("--source-build-markdown", type=Path)
    parser.add_argument("--output-site", type=Path)
    parser.add_argument("--tmp-site", type=Path)
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


def read_text(path: Path | None, limit_bytes: int = 2_000_000) -> str:
    if path is None or not path.exists():
        return ""
    with path.open("rb") as fh:
        data = fh.read(limit_bytes)
    return data.decode("utf-8", errors="replace")


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


def parse_scontrol_kv(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in output.replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and key not in parsed:
            parsed[key] = value
    return parsed


def resolve_job(args: argparse.Namespace, job_values: dict[str, str]) -> str | None:
    return args.job_id or job_values.get("vllm_native_source_build_job")


def resolve_paths(args: argparse.Namespace, job_values: dict[str, str], job_id: str | None) -> dict[str, Path]:
    source_json = args.source_build_json or Path(
        job_values.get("json") or args.artifact_root / "reports/vllm_native_source_build.json"
    )
    source_md = args.source_build_markdown or Path(
        job_values.get("markdown") or args.artifact_root / "reports/vllm_native_source_build.md"
    )
    output_site = args.output_site or Path(
        job_values.get("output_site") or args.artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"
    )
    tmp_site = args.tmp_site or Path(f"{output_site}.tmp.{job_id or 'unknown'}")
    return {
        "source_json": source_json,
        "source_md": source_md,
        "output_site": output_site,
        "tmp_site": tmp_site,
    }


def resolve_logs(logs_dir: Path, job_id: str | None) -> tuple[Path | None, Path | None]:
    if not job_id:
        return None, None
    exact_out = logs_dir / f"{JOB_NAME}_{job_id}.out"
    exact_err = logs_dir / f"{JOB_NAME}_{job_id}.err"
    if exact_out.exists() or exact_err.exists():
        return exact_out if exact_out.exists() else None, exact_err if exact_err.exists() else None
    out_matches = sorted(logs_dir.glob(f"*_{job_id}.out")) if logs_dir.exists() else []
    err_matches = sorted(logs_dir.glob(f"*_{job_id}.err")) if logs_dir.exists() else []
    return (out_matches[0] if out_matches else None, err_matches[0] if err_matches else None)


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"top-level JSON is not an object: {path}"
    return payload, None


def slurm_state(job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"status": "missing_job_id"}
    squeue = run_command(["squeue", "-j", job_id, "-h", "-o", "%i|%T|%M|%D|%R|%S|%l"])
    rows = [line for line in (squeue.get("output") or "").splitlines() if line.strip()]
    if squeue.get("returncode") == 0 and rows:
        parts = rows[0].split("|")
        return {
            "status": "in_queue",
            "job_id": parts[0] if len(parts) > 0 else job_id,
            "state": parts[1] if len(parts) > 1 else "unknown",
            "elapsed": parts[2] if len(parts) > 2 else "",
            "nodes": parts[3] if len(parts) > 3 else "",
            "reason": parts[4] if len(parts) > 4 else "",
            "start": parts[5] if len(parts) > 5 else "",
            "time_limit": parts[6] if len(parts) > 6 else "",
            "squeue": squeue,
        }
    sacct = run_command(["sacct", "-j", job_id, "--format=JobID,JobName,State,Elapsed,Start,End,ExitCode", "-P", "-n"])
    rows = [line for line in (sacct.get("output") or "").splitlines() if line.strip()]
    primary = rows[0].split("|") if rows else []
    return {
        "status": "terminal_or_unknown",
        "job_id": job_id,
        "state": primary[2] if len(primary) > 2 else "unknown",
        "elapsed": primary[3] if len(primary) > 3 else "",
        "start": primary[4] if len(primary) > 4 else "",
        "end": primary[5] if len(primary) > 5 else "",
        "exit_code": primary[6] if len(primary) > 6 else "",
        "sacct": sacct,
        "sacct_rows": rows[:20],
    }


def scontrol_summary(job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"available": False, "reason": "missing job id"}
    result = run_command(["scontrol", "show", "job", job_id])
    output = result.get("output") or ""
    if not result.get("available") or result.get("returncode") != 0 or not output:
        return {"available": False, "command": result.get("command"), "output": output}
    fields = parse_scontrol_kv(output)
    keep = [
        "JobId",
        "JobName",
        "JobState",
        "Reason",
        "RunTime",
        "TimeLimit",
        "StartTime",
        "EndTime",
        "NodeList",
        "BatchHost",
        "NumNodes",
        "NumCPUs",
        "ExitCode",
    ]
    return {
        "available": True,
        "command": result.get("command"),
        "fields": {key: fields.get(key, "") for key in keep if key in fields},
    }


def sstat_summary(job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"available": False, "reason": "missing job id"}
    step_id = f"{job_id}.0"
    result = run_command(["sstat", "-j", step_id, "--format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize", "-P"])
    output = result.get("output") or ""
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) < 2:
        return {"available": False, "command": result.get("command"), "output": output}
    header = rows[0].split("|")
    values = rows[1].split("|")
    parsed = {key: values[idx] if idx < len(values) else "" for idx, key in enumerate(header)}
    parsed["available"] = True
    parsed["command"] = result.get("command")
    return parsed


def path_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        result["is_dir"] = path.is_dir()
        result["du"] = run_command(["du", "-sh", str(path)]).get("output", "").split("\t")[0]
        if path.is_dir():
            top_entries = sorted(item.name for item in path.iterdir())[:80]
            result["top_level_count_sampled"] = len(top_entries)
            result["sample_top_level_entries"] = top_entries
            so_files = sorted(str(item) for item in path.glob("**/*.so"))[:20]
            abi3_files = sorted(str(item) for item in path.glob("**/*.abi3.so"))[:20]
            result["so_file_count_sampled"] = len(so_files)
            result["abi3_file_count_sampled"] = len(abi3_files)
            result["sample_so_files"] = so_files[:10]
            result["sample_abi3_files"] = abi3_files[:10]
    return result


def log_file_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "exists": False}
    text = read_text(path)
    return {
        "path": str(path),
        "exists": True,
        "line_count": len(text.splitlines()),
        "size_bytes": path.stat().st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime)),
    }


def useful_tail(text: str, limit: int = 28) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def has_failure(text: str) -> bool:
    for line in text.splitlines():
        if BENIGN_RE.search(line):
            continue
        if FAIL_RE.search(line):
            return True
    return False


def root_causes(text: str, max_items: int = 8) -> list[str]:
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
            if len(causes) >= max_items:
                return causes
    return causes


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    job_values = parse_kv_file(args.job_file)
    job_id = resolve_job(args, job_values)
    paths = resolve_paths(args, job_values, job_id)
    out_log, err_log = resolve_logs(args.logs_dir, job_id)
    out_text = read_text(out_log)
    err_text = read_text(err_log)
    log_text = "\n".join(part for part in [out_text, err_text] if part)
    source_payload, source_error = load_json(paths["source_json"])
    source_status = (source_payload or {}).get("overall_status") or (source_payload or {}).get("status")
    slurm = slurm_state(job_id)
    sstat = sstat_summary(job_id)
    scontrol = scontrol_summary(job_id)
    slurm_state_name = str(slurm.get("state") or "").upper()
    log_failure = has_failure(log_text)
    causes = root_causes(log_text)

    if source_status == "pass":
        overall = "pass"
        detail = "source-build report is PASS"
    elif source_status == "fail":
        overall = "fail"
        detail = "source-build report is FAIL"
    elif slurm_state_name in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
        overall = "running"
        detail = f"source-build job is {slurm_state_name}"
    elif slurm_state_name.startswith("TIMEOUT"):
        overall = "timeout"
        detail = "source-build job timed out before writing PASS"
    elif slurm_state_name in {"FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"} or log_failure:
        overall = "fail"
        detail = f"source-build job ended as {slurm_state_name or 'unknown'}"
    else:
        overall = "incomplete"
        detail = "source-build job has no PASS report yet"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "detail": detail,
        "job_id": job_id,
        "job_file": str(args.job_file),
        "slurm": slurm,
        "scontrol": scontrol,
        "sstat": sstat,
        "source_report": {
            "json": str(paths["source_json"]),
            "markdown": str(paths["source_md"]),
            "status": source_status,
            "error": source_error,
            "payload": source_payload,
        },
        "paths": {
            "output_site": path_summary(paths["output_site"]),
            "tmp_site": path_summary(paths["tmp_site"]),
            "out_log": str(out_log) if out_log else None,
            "err_log": str(err_log) if err_log else None,
        },
        "logs": {
            "out": log_file_summary(out_log),
            "err": log_file_summary(err_log),
        },
        "log_failure_detected": log_failure,
        "root_causes": causes,
        "out_tail": useful_tail(out_text),
        "err_tail": useful_tail(err_text),
        "next_step": next_step(overall),
    }


def next_step(overall: str) -> str:
    if overall == "pass":
        return "Run source-site ABI probe, then rollout smoke."
    if overall == "running":
        return "Keep polling; do not submit another source build while this job is running."
    if overall == "timeout":
        return "Submit a longer source-build retry or confirm the timeout watchdog submitted one."
    if overall == "fail":
        return "Inspect root causes and patch the build wrapper before retrying."
    return "Wait for terminal Slurm state or source-build report."


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# vLLM Source Build Job Analysis",
        "",
        f"Overall: **{str(payload['overall_status']).upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Job: `{payload.get('job_id')}`",
        "",
        f"Detail: {payload.get('detail')}",
        f"Next step: {payload.get('next_step')}",
        "",
        "## Slurm",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    slurm = payload.get("slurm") or {}
    for key in ["state", "elapsed", "time_limit", "nodes", "reason", "start", "end", "exit_code"]:
        if key in slurm:
            lines.append(f"| {key} | `{slurm.get(key)}` |")
    scontrol = payload.get("scontrol") if isinstance(payload.get("scontrol"), dict) else {}
    if scontrol.get("available"):
        lines += [
            "",
            "## Slurm Job Detail",
            "",
            "| field | value |",
            "| --- | --- |",
        ]
        for key, value in (scontrol.get("fields") or {}).items():
            lines.append(f"| {key} | `{value}` |")
    sstat = payload.get("sstat") if isinstance(payload.get("sstat"), dict) else {}
    if sstat.get("available"):
        lines += [
            "",
            "## Resource Activity",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| AveCPU | `{sstat.get('AveCPU')}` |",
            f"| AveRSS | `{sstat.get('AveRSS')}` |",
            f"| MaxRSS | `{sstat.get('MaxRSS')}` |",
            f"| MaxVMSize | `{sstat.get('MaxVMSize')}` |",
        ]
    source = payload.get("source_report") or {}
    lines += [
        "",
        "## Reports",
        "",
        f"- source JSON: `{source.get('json')}`",
        f"- source Markdown: `{source.get('markdown')}`",
        f"- source status: `{source.get('status')}`",
        f"- source error: `{source.get('error')}`",
        "",
        "## Paths",
        "",
    ]
    for label in ["output_site", "tmp_site"]:
        item = ((payload.get("paths") or {}).get(label) or {})
        lines.append(f"- {label}: `{item.get('path')}` exists={item.get('exists')} du=`{item.get('du', '-')}`")
        entries = item.get("sample_top_level_entries") or []
        if entries:
            lines.append(f"  - sampled top-level entries: `{', '.join(entries[:16])}`")
    logs = payload.get("logs") if isinstance(payload.get("logs"), dict) else {}
    if logs:
        lines += ["", "## Log Files", "", "| file | lines | bytes | mtime |", "| --- | ---: | ---: | --- |"]
        for label in ["out", "err"]:
            item = logs.get(label) or {}
            lines.append(
                f"| `{item.get('path')}` | {item.get('line_count', '-')} | "
                f"{item.get('size_bytes', '-')} | `{item.get('mtime', '-')}` |"
            )
    if payload.get("root_causes"):
        lines += ["", "## Root Causes", ""]
        lines.extend(f"- {cause}" for cause in payload["root_causes"])
    lines += ["", "## Log Tail", ""]
    for label in ["out_tail", "err_tail"]:
        lines += [f"### {label}", "", "```text"]
        lines.extend(payload.get(label) or [""])
        lines += ["```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.fail_on_failure and payload["overall_status"] in {"fail", "timeout"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
