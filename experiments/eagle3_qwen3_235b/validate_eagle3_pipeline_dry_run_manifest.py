#!/usr/bin/env python3
"""Validate that Eagle3 pipeline dry-runs leave an analyzable job manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
JOB_FILE = ROOT / "latest_eagle3_pipeline_jobs.txt"
EXPECTED_PLACEHOLDERS = {
    "preflight_job": "PREFLIGHT_JOB_ID",
    "dump_job": "DUMP_JOB_ID",
    "validate_hiddens_job": "VALIDATE_HIDDENS_JOB_ID",
    "train_job": "TRAIN_JOB_ID",
    "export_job": "EXPORT_JOB_ID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def parse_kv(path: Path) -> dict[str, str]:
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


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    lines = [
        "# Eagle3 Pipeline Dry-run Manifest Validation",
        "",
        f"Overall: **{str(payload['overall_status']).upper()}**",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {str(check['status']).upper()} | {check['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "artifact"
    env = {
        "SUBMIT": "false",
        "RUN_PILOT": "true",
        "ARTIFACT_ROOT": str(artifact),
        "SBATCH_ACCOUNT": "dummy",
        "SBATCH_PARTITION": "batch",
        "INPUT_DATA": str(artifact / "data/conversations.jsonl"),
        "HIDDEN_STATES_DIR": str(artifact / "hidden_states"),
        "OUTPUT_DIR": str(artifact / "modelopt_ckpt"),
        "TRAINED_CKPT": str(artifact / "modelopt_ckpt"),
        "EXPORT_DIR": str(artifact / "exported_hf"),
        "VLLM_DRAFT_DIR": str(artifact / "vllm_draft"),
        "VERIFIER_CONFIG_DIR": str(artifact / "verifier_config"),
        "CHAT_TEMPLATE": str(artifact / "templates/qwen3_generation_template.jinja2"),
        "ARCH_ENV_FILE": "",
        "REFERENCE_ARCH": str(ROOT / "experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json"),
        "MODELOPT_DIR": str(ROOT / "Model-Optimizer"),
        "START_PIPELINE_WATCHER": "false",
    }
    completed = run(["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"], env=env)
    manifest = parse_kv(JOB_FILE)
    problems: list[str] = []
    if completed.returncode != 0:
        problems.append(f"dry-run returned {completed.returncode}")
    for key, value in EXPECTED_PLACEHOLDERS.items():
        if manifest.get(key) != value:
            problems.append(f"{key}={manifest.get(key)!r}, expected {value!r}")
    add(
        checks,
        "dry-run writes stage placeholders",
        "fail" if problems else "pass",
        "pipeline dry-run manifest is incomplete" if problems else "all stage placeholders are written",
        problems=problems,
        manifest=manifest,
        stdout_tail=completed.stdout[-3000:],
    )

    analysis_json = root / "pipeline_analysis.json"
    analysis_md = root / "pipeline_analysis.md"
    analyzed = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py",
            "--job-file",
            str(JOB_FILE),
            "--input-data",
            env["INPUT_DATA"],
            "--hidden-states-dir",
            env["HIDDEN_STATES_DIR"],
            "--hidden-validation-json",
            str(artifact / "hidden_states/validation_summary.json"),
            "--training-checkpoint-json",
            str(artifact / "reports/eagle3_training_checkpoint.json"),
            "--output-dir",
            env["OUTPUT_DIR"],
            "--export-dir",
            env["EXPORT_DIR"],
            "--vllm-draft-dir",
            env["VLLM_DRAFT_DIR"],
            "--export-artifacts-json",
            str(artifact / "reports/eagle3_export_artifacts.json"),
            "--verifier-config-dir",
            env["VERIFIER_CONFIG_DIR"],
            "--reference-arch",
            env["REFERENCE_ARCH"],
            "--chat-template",
            env["CHAT_TEMPLATE"],
            "--sbatch-account",
            "dummy",
            "--run-pilot",
            "true",
            "--json-out",
            str(analysis_json),
            "--markdown-out",
            str(analysis_md),
        ]
    )
    analysis_payload: dict[str, Any] = {}
    if analysis_json.exists():
        analysis_payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    planned = [
        item.get("stage")
        for item in analysis_payload.get("stages", [])
        if item.get("status") == "planned"
    ]
    problems = []
    if analyzed.returncode != 0:
        problems.append(f"analysis returned {analyzed.returncode}")
    if analysis_payload.get("overall_status") != "incomplete":
        problems.append(f"unexpected analysis status: {analysis_payload.get('overall_status')!r}")
    if planned != ["preflight", "dump", "validate_hiddens", "train", "export"]:
        problems.append(f"planned stages mismatch: {planned}")
    add(
        checks,
        "dry-run manifest is analyzable",
        "fail" if problems else "pass",
        "analyzer did not understand dry-run manifest" if problems else "analyzer reports all stages as planned",
        problems=problems,
        planned_stages=planned,
        stdout_tail=analyzed.stdout[-3000:],
    )


def main() -> int:
    args = parse_args()
    root = Path(tempfile.mkdtemp(prefix="eagle3_pipeline_dry_run_"))
    backup = root / "latest_eagle3_pipeline_jobs.backup"
    had_original = JOB_FILE.exists()
    if had_original:
        shutil.copyfile(JOB_FILE, backup)
    checks: list[dict[str, Any]] = []
    try:
        validate(root, checks)
    finally:
        if had_original:
            shutil.copyfile(backup, JOB_FILE)
        else:
            JOB_FILE.unlink(missing_ok=True)
        if args.keep_temp:
            add(checks, "temporary directory retained", "warn", str(root))
        else:
            shutil.rmtree(root, ignore_errors=True)
    counts = status_counts(checks)
    overall_status = "fail" if counts.get("fail", 0) else "pass"
    payload = {"overall_status": overall_status, "counts": counts, "checks": checks}
    write_json(args.json_out, payload)
    write_markdown(args.markdown_out, payload)
    print(args.markdown_out.read_text(encoding="utf-8") if args.markdown_out else json.dumps(payload, indent=2))
    return 1 if overall_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
