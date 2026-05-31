#!/usr/bin/env python3
"""Summarize a submitted Qwen3 Eagle3 Slurm pipeline.

The submitter writes latest_eagle3_pipeline_jobs.txt with stage job ids:

    preflight_job=<jobid>
    dump_job=<jobid>
    validate_hiddens_job=<jobid>
    train_job=<jobid>
    export_job=<jobid>

Slurm logs use logs/%x_%j.{out,err}. This parser combines those logs with the
expected artifact paths so a pilot failure can be triaged without opening five
files by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_FILE = ROOT / "latest_eagle3_pipeline_jobs.txt"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"

STAGES = [
    ("preflight", "preflight_job", "q235b-eagle3-preflight"),
    ("dump", "dump_job", "q235b-eagle3-dump"),
    ("validate_hiddens", "validate_hiddens_job", "q235b-eagle3-validate-hiddens"),
    ("train", "train_job", "q235b-eagle3-train"),
    ("export", "export_job", "q235b-eagle3-export"),
]

STAGE_RUN_FLAGS = {
    "preflight": "RUN_PREFLIGHT",
    "dump": "RUN_DUMP",
    "validate_hiddens": "RUN_VALIDATE_HIDDENS",
    "train": "RUN_TRAIN",
    "export": "RUN_EXPORT",
}

FAIL_RE = re.compile(
    r"\b("
    r"traceback|runtimeerror|valueerror|assertionerror|exception|"
    r"failed|fail\b|error\b|outofmemory|oom|killed|cancelled|timeout|"
    r"slurmstepd: error|segmentation fault"
    r")",
    re.IGNORECASE,
)

BENIGN_RE = re.compile(
    r"(failure_count['\"]?:\s*0|0 config checks failed|failures?:\s*0|"
    r"Preflight passed|Recipe override validation passed|dry-run passed)",
    re.IGNORECASE,
)

SUCCESS_MARKERS = {
    "preflight": ["Preflight passed."],
    "dump": ["Successfully processed all", "Successfully processed "],
    "validate_hiddens": ["validated ", "modelopt_loader_validation"],
    "train": ["TrainOutput", "Training completed", "Saving model checkpoint", "Step "],
    "export": ["Exported checkpoint to", "Config checks failed", "OK       ", "status': 'passed'"],
}


@dataclass
class StageResult:
    stage: str
    job_key: str
    job_name: str
    job_id: str | None
    status: str
    detail: str
    out_log: str | None = None
    err_log: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    tail: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, default=DEFAULT_JOB_FILE)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--modelopt-dir", type=Path, default=os.environ.get("MODELOPT_DIR"))
    parser.add_argument("--verifier-config-dir", type=Path, default=os.environ.get("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--reference-arch", type=Path, default=os.environ.get("REFERENCE_ARCH"))
    parser.add_argument("--arch-env-file", type=Path, default=os.environ.get("ARCH_ENV_FILE"))
    parser.add_argument("--chat-template", type=Path, default=os.environ.get("CHAT_TEMPLATE"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER"))
    parser.add_argument("--mounts", default=os.environ.get("MOUNTS"))
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--hidden-states-dir", type=Path)
    parser.add_argument("--hidden-validation-json", type=Path)
    parser.add_argument("--training-checkpoint-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--vllm-draft-dir", type=Path)
    parser.add_argument("--export-artifacts-json", type=Path)
    parser.add_argument(
        "--sbatch-account",
        default=os.environ.get("SBATCH_ACCOUNT", "<account>"),
        help="Account value to place in the generated resume command.",
    )
    parser.add_argument(
        "--run-pilot",
        default=os.environ.get("RUN_PILOT", "false"),
        help="RUN_PILOT value to place in the generated resume command.",
    )
    parser.add_argument(
        "--sbatch-partition",
        default=os.environ.get("SBATCH_PARTITION"),
        help="Optional partition value to place in the generated resume command.",
    )
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Return nonzero if any stage is FAIL. Missing/running stages do not fail.",
    )
    return parser.parse_args()


def parse_job_file(path: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    if not path.exists():
        return jobs
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        jobs[key.strip()] = value.strip()
    return jobs


def resolve_logs(logs_dir: Path, job_name: str, job_id: str | None) -> tuple[Path | None, Path | None]:
    if not job_id:
        return None, None
    candidates = [
        (logs_dir / f"{job_name}_{job_id}.out", logs_dir / f"{job_name}_{job_id}.err"),
    ]
    out_matches = sorted(logs_dir.glob(f"*_{job_id}.out")) if logs_dir.exists() else []
    err_matches = sorted(logs_dir.glob(f"*_{job_id}.err")) if logs_dir.exists() else []
    if out_matches or err_matches:
        candidates.append((out_matches[0] if out_matches else None, err_matches[0] if err_matches else None))
    for out_path, err_path in candidates:
        if (out_path and out_path.exists()) or (err_path and err_path.exists()):
            return out_path if out_path and out_path.exists() else None, err_path if err_path and err_path.exists() else None
    return None, None


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def useful_tail(text: str, limit: int = 12) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def has_failure(text: str) -> bool:
    for line in text.splitlines():
        if BENIGN_RE.search(line):
            continue
        if FAIL_RE.search(line):
            return True
    return False


def has_success_marker(stage: str, text: str) -> bool:
    return any(marker in text for marker in SUCCESS_MARKERS.get(stage, []))


def artifact_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if args.input_data:
        evidence["input_data_exists"] = args.input_data.exists()
    if args.hidden_states_dir:
        evidence["hidden_states_pt_files"] = len(list(args.hidden_states_dir.glob("*.pt"))) if args.hidden_states_dir.exists() else 0
    if args.hidden_validation_json:
        evidence["hidden_validation_json_exists"] = args.hidden_validation_json.exists()
    if args.training_checkpoint_json:
        evidence["training_checkpoint_json_exists"] = args.training_checkpoint_json.exists()
        if args.training_checkpoint_json.exists():
            try:
                payload = json.loads(args.training_checkpoint_json.read_text(encoding="utf-8"))
                evidence["training_checkpoint_status"] = payload.get("overall_status")
            except Exception as exc:
                evidence["training_checkpoint_status"] = f"invalid: {exc}"
    if args.output_dir:
        evidence["output_dir_nonempty"] = args.output_dir.exists() and any(args.output_dir.iterdir())
    if args.export_dir:
        evidence["export_config_exists"] = (args.export_dir / "config.json").exists()
    if args.vllm_draft_dir:
        evidence["vllm_config_exists"] = (args.vllm_draft_dir / "config.json").exists()
        evidence["vllm_safetensors_count"] = (
            len(list(args.vllm_draft_dir.glob("*.safetensors"))) if args.vllm_draft_dir.exists() else 0
        )
        evidence["vllm_weights_exists"] = evidence["vllm_safetensors_count"] > 0
    if args.export_artifacts_json:
        evidence["export_artifacts_json_exists"] = args.export_artifacts_json.exists()
        if args.export_artifacts_json.exists():
            try:
                payload = json.loads(args.export_artifacts_json.read_text(encoding="utf-8"))
                evidence["export_artifacts_status"] = payload.get("overall_status")
            except Exception as exc:
                evidence["export_artifacts_status"] = f"invalid: {exc}"
    return evidence


def stage_artifact_status(stage: str, args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    evidence = artifact_evidence(args)
    if stage == "dump":
        return evidence.get("hidden_states_pt_files", 0) > 0, evidence
    if stage == "validate_hiddens":
        return evidence.get("hidden_validation_json_exists", False), evidence
    if stage == "train":
        if args.training_checkpoint_json and evidence.get("training_checkpoint_json_exists"):
            return evidence.get("training_checkpoint_status") == "pass", evidence
        return evidence.get("output_dir_nonempty", False), evidence
    if stage == "export":
        training_ok = True
        if args.training_checkpoint_json:
            training_ok = evidence.get("training_checkpoint_status") == "pass"
        if args.export_artifacts_json:
            return evidence.get("export_artifacts_status") == "pass" and training_ok, evidence
        return bool(evidence.get("export_config_exists") or evidence.get("vllm_config_exists")) and training_ok, evidence
    return True, evidence


def analyze_stage(
    stage: str,
    job_key: str,
    job_name: str,
    job_id: str | None,
    args: argparse.Namespace,
) -> StageResult:
    if not job_id:
        return StageResult(stage, job_key, job_name, None, "missing", f"{job_key} is absent from job file")
    if not job_id.isdigit():
        return StageResult(stage, job_key, job_name, job_id, "planned", f"{job_key} is a dry-run placeholder")

    out_log, err_log = resolve_logs(args.logs_dir, job_name, job_id)
    out_text = read_text(out_log)
    err_text = read_text(err_log)
    combined = "\n".join(part for part in (out_text, err_text) if part)
    artifact_ok, artifacts = stage_artifact_status(stage, args)

    evidence = {
        "out_bytes": len(out_text),
        "err_bytes": len(err_text),
        "has_failure_text": has_failure(combined),
        "has_success_marker": has_success_marker(stage, combined),
        **artifacts,
    }

    if not out_log and not err_log:
        return StageResult(
            stage,
            job_key,
            job_name,
            job_id,
            "missing",
            f"no Slurm logs found for {job_name}_{job_id}",
            evidence=evidence,
        )

    if has_failure(combined):
        return StageResult(
            stage,
            job_key,
            job_name,
            job_id,
            "fail",
            "failure-like text found in Slurm logs",
            str(out_log) if out_log else None,
            str(err_log) if err_log else None,
            evidence,
            useful_tail(combined),
        )

    marker_ok = has_success_marker(stage, combined)
    if marker_ok and artifact_ok:
        status = "pass"
        detail = "success marker and expected artifacts found"
    elif marker_ok:
        status = "warn"
        detail = "success marker found, but expected artifact evidence is incomplete"
    elif artifact_ok:
        status = "warn"
        detail = "artifact evidence found, but log success marker is missing"
    else:
        status = "running_or_unknown"
        detail = "logs exist without failure text, but success/artifact evidence is incomplete"

    return StageResult(
        stage,
        job_key,
        job_name,
        job_id,
        status,
        detail,
        str(out_log) if out_log else None,
        str(err_log) if err_log else None,
        evidence,
        useful_tail(combined),
    )


def summarize(results: list[StageResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def path_or_placeholder(path: Path | None, placeholder: str) -> str:
    return str(path) if path else placeholder


def shell_env_command(env: dict[str, str], script: str) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    parts += ["bash", shlex.quote(script)]
    return " ".join(parts)


def bool_env_text(value: str | bool | None) -> str:
    return "true" if str(value).lower() in {"true", "1", "yes"} else "false"


def build_next_action(results: list[StageResult], args: argparse.Namespace, overall: str) -> dict[str, Any]:
    if overall == "pass":
        return {
            "summary": (
                "All core pipeline stages have pass-level evidence. "
                "Next run the trained-draft smoke/sweep jobs and completion audit."
            ),
            "first_open_stage": None,
            "first_open_status": None,
            "completed_stages": [result.stage for result in results],
            "resume_env": {},
            "resume_command": None,
            "notes": [
                "Use RUN_TRAINED_DRAFT_SMOKE=true and/or RUN_TRAINED_DRAFT_SWEEP=true when launching post-export checks.",
                "Treat completion audit PASS as the handoff gate for a usable Qwen3-235B Thinking Eagle3 draft model.",
            ],
        }

    first_open_index = next(
        (index for index, result in enumerate(results) if result.status != "pass"),
        None,
    )
    if first_open_index is None:
        first_open_index = 0
    first_open = results[first_open_index]

    run_flags = {
        STAGE_RUN_FLAGS[result.stage]: ("false" if index < first_open_index else "true")
        for index, result in enumerate(results)
    }
    resume_env = {
        "SUBMIT": "false",
        "RUN_PILOT": bool_env_text(args.run_pilot),
        **run_flags,
        "RUN_TRAINED_DRAFT_SMOKE": "false",
        "RUN_TRAINED_DRAFT_SWEEP": "false",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "BASE_MODEL": args.base_model,
        "INPUT_DATA": path_or_placeholder(args.input_data, "<conversations.jsonl>"),
        "HIDDEN_STATES_DIR": path_or_placeholder(args.hidden_states_dir, "<hidden_states_dir>"),
        "HIDDEN_STATES_VALIDATION_JSON": path_or_placeholder(
            args.hidden_validation_json,
            "<hidden_states_dir>/validation_summary.json",
        ),
        "OUTPUT_DIR": path_or_placeholder(args.output_dir, "<modelopt_checkpoint_dir>"),
        "TRAINING_CKPT_VALIDATION_JSON": path_or_placeholder(
            args.training_checkpoint_json,
            "<reports>/eagle3_training_checkpoint.json",
        ),
        "EXPORT_DIR": path_or_placeholder(args.export_dir, "<export_dir>"),
        "VLLM_DRAFT_DIR": path_or_placeholder(args.vllm_draft_dir, "<vllm_draft_dir>"),
        "VERIFIER_CONFIG_DIR": path_or_placeholder(args.verifier_config_dir, "<verifier_config_dir>"),
    }
    optional_env = {
        "SBATCH_PARTITION": args.sbatch_partition,
        "MODELOPT_DIR": args.modelopt_dir,
        "REFERENCE_ARCH": args.reference_arch,
        "ARCH_ENV_FILE": args.arch_env_file,
        "CHAT_TEMPLATE": args.chat_template,
        "CONTAINER": args.container,
        "MOUNTS": args.mounts,
    }
    resume_env.update({key: str(value) for key, value in optional_env.items() if value})
    command = shell_env_command(
        resume_env,
        "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh",
    )

    notes = [
        "Generated command is a dry-run by default; change SUBMIT=true only after reviewing the printed plan.",
        "Stages before first_open_stage are disabled so already-produced artifacts are reused.",
    ]
    if first_open.status == "running_or_unknown":
        notes.insert(
            0,
            "Confirm the Slurm job is no longer running before submitting a duplicate resume.",
        )
    if first_open.status == "warn":
        notes.insert(
            0,
            "Inspect the warning before resubmitting; artifact evidence and log markers disagree.",
        )
    if first_open.status == "planned":
        notes.insert(
            0,
            "The job file contains dry-run placeholders, so no Slurm job has been proven submitted yet.",
        )

    return {
        "summary": (
            f"Resume from {first_open.stage} ({first_open.status}). "
            "The command below disables earlier pass-level stages and reruns the open tail of the pipeline."
        ),
        "first_open_stage": first_open.stage,
        "first_open_status": first_open.status,
        "completed_stages": [result.stage for result in results[:first_open_index]],
        "resume_env": resume_env,
        "resume_command": command,
        "notes": notes,
    }


def result_payload(results: list[StageResult], args: argparse.Namespace) -> dict[str, Any]:
    counts = summarize(results)
    overall = "fail" if counts.get("fail") else "incomplete"
    if results and all(result.status == "pass" for result in results):
        overall = "pass"
    return {
        "overall_status": overall,
        "job_file": str(args.job_file),
        "logs_dir": str(args.logs_dir),
        "counts": counts,
        "next_action": build_next_action(results, args, overall),
        "stages": [
            {
                "stage": result.stage,
                "job_key": result.job_key,
                "job_name": result.job_name,
                "job_id": result.job_id,
                "status": result.status,
                "detail": result.detail,
                "out_log": result.out_log,
                "err_log": result.err_log,
                "evidence": result.evidence,
                "tail": result.tail,
            }
            for result in results
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Eagle3 Pipeline Analysis",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        "",
        "| stage | job id | status | detail | out | err |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for stage in payload["stages"]:
        out_log = stage["out_log"] or "-"
        err_log = stage["err_log"] or "-"
        lines.append(
            f"| {stage['stage']} | {stage['job_id'] or '-'} | {stage['status'].upper()} | "
            f"{stage['detail'].replace('|', '/')} | {out_log} | {err_log} |"
        )
    next_action = payload.get("next_action") or {}
    if next_action:
        lines += ["", "## Next Action", "", next_action.get("summary", "")]
        notes = next_action.get("notes") or []
        if notes:
            lines += ["", "Notes:"]
            lines.extend(f"- {note}" for note in notes)
        command = next_action.get("resume_command")
        if command:
            lines += ["", "Resume dry-run command:", "", "```bash", command, "```"]
    failing = [stage for stage in payload["stages"] if stage["status"] == "fail"]
    if failing:
        lines += ["", "## Failure Tails"]
        for stage in failing:
            lines += ["", f"### {stage['stage']}", "", "```text"]
            lines.extend(stage["tail"] or ["<empty>"])
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
    jobs = parse_job_file(args.job_file)
    results = [
        analyze_stage(stage, job_key, job_name, jobs.get(job_key), args)
        for stage, job_key, job_name in STAGES
    ]
    payload = result_payload(results, args)
    write_outputs(payload, args)
    if args.fail_on_failure and payload["overall_status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
