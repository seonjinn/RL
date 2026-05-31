#!/usr/bin/env python3
"""Submit the Qwen3-235B Eagle3 pilot pipeline only after preflight is ready.

This helper is intentionally stricter than running the preflight report's
`commands.pilot_submit` by hand. It refuses to submit unless the preflight JSON
proves `submit_ready=true`, the input corpus exists, and the command is the
expected local `submit_eagle3_pipeline.sh` path.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
REQUIRED_PIPELINE_JOBS = ("dump_job", "train_job", "export_job")
REQUIRED_PIPELINE_ENV = (
    "ARTIFACT_ROOT",
    "INPUT_DATA",
    "HIDDEN_STATES_DIR",
    "OUTPUT_DIR",
    "EXPORT_DIR",
    "VLLM_DRAFT_DIR",
    "VERIFIER_CONFIG_DIR",
    "MODELOPT_DIR",
    "CHAT_TEMPLATE",
    "ARCH_ENV_FILE",
    "REFERENCE_ARCH",
    "BASE_MODEL",
    "ANSWER_ONLY_LOSS",
    "TRAINING_SEQ_LEN",
    "MAX_SEQ_LEN",
    "DUMP_GPUS_PER_NODE",
    "TRAIN_GPUS_PER_NODE",
    "EXPORT_GPUS_PER_NODE",
    "TP",
)
POSITIVE_INT_ENV = ("TRAINING_SEQ_LEN", "MAX_SEQ_LEN", "DUMP_GPUS_PER_NODE", "TRAIN_GPUS_PER_NODE", "EXPORT_GPUS_PER_NODE", "TP")
REQUIRED_PREFLIGHT_PASS_CHECKS = (
    ("paths", "input conversation JSONL"),
    ("paths", "chat template"),
    ("paths", "verifier config.json"),
    ("paths", "Eagle3 architecture JSON"),
    ("paths", "Eagle3 architecture env"),
    ("modelopt", "TRT-LLM loss-mask patch"),
    ("data", "answer-only chat template tags"),
    ("data", "corpus strategy"),
    ("data", "rollout state advance"),
    ("execution", "container preflight"),
    ("data", "training conversation validation"),
    ("slurm", "GPU capacity vs pipeline requests"),
    ("validation", "local ModelOpt pipeline preflight"),
    ("dry_run", "ModelOpt wrapper dry-runs"),
    ("dry_run", "Slurm pipeline dry-run"),
)
PILOT_PREFLIGHT_PASS_CHECKS = (("data", "pilot minimum rows"),)
PREFLIGHT_ENV_FIELDS = {
    "ARTIFACT_ROOT": ("artifact_root",),
    "INPUT_DATA": ("input_data",),
    "HIDDEN_STATES_DIR": ("hidden_states_dir",),
    "OUTPUT_DIR": ("output_dir",),
    "EXPORT_DIR": ("export_dir",),
    "VLLM_DRAFT_DIR": ("vllm_draft_dir",),
    "VERIFIER_CONFIG_DIR": ("verifier_config_dir",),
    "MODELOPT_DIR": ("modelopt_dir",),
    "CHAT_TEMPLATE": ("chat_template",),
    "ARCH_ENV_FILE": ("arch_env_file",),
    "REFERENCE_ARCH": ("reference_arch",),
    "BASE_MODEL": ("base_model",),
    "ANSWER_ONLY_LOSS": ("answer_only_loss",),
    "TRAINING_SEQ_LEN": ("training_seq_len",),
    "MAX_SEQ_LEN": ("max_seq_len",),
    "DUMP_GPUS_PER_NODE": ("resource_request", "dump_gpus_per_node"),
    "TRAIN_GPUS_PER_NODE": ("resource_request", "train_gpus_per_node"),
    "EXPORT_GPUS_PER_NODE": ("resource_request", "export_gpus_per_node"),
    "TP": ("resource_request", "tp"),
}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path(os.environ.get("PIPELINE_SUBMIT_PREFLIGHT_JSON", artifact_root / "reports/eagle3_pipeline_submit_preflight.json")),
    )
    parser.add_argument("--execute", action="store_true", help="Actually run the preflight report's pilot_submit command.")
    parser.add_argument("--allow-heavy-gpu", action="store_true", help="Required with --execute because the command submits GPU Slurm jobs.")
    parser.add_argument(
        "--exit-zero-if-not-ready",
        action="store_true",
        help=(
            "Return 0 for a no-submit readiness check when the only failures are "
            "expected not-ready gates such as missing corpus/preflight readiness."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid json: {exc}"


def parse_env_command(command: str) -> tuple[dict[str, str], list[str]]:
    tokens = shlex.split(command)
    env: dict[str, str] = {}
    index = 0
    for index, token in enumerate(tokens):
        if "=" not in token or token.startswith("-"):
            break
        key, value = token.split("=", 1)
        if not key.replace("_", "").isalnum() or not key[:1].isalpha():
            break
        env[key] = value
    else:
        index = len(tokens)
    return env, tokens[index:]


def command_string(env: dict[str, str], argv: list[str]) -> str:
    return " ".join([*(f"{key}={shlex.quote(value)}" for key, value in env.items()), *(shlex.quote(part) for part in argv)])


def job_file_jobs(path: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    if not path.exists():
        return jobs
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value:
            jobs[key.strip()] = value
    return jobs


def copy_job_file(source: Path, artifact_root: Path) -> Path | None:
    if not source.exists():
        return None
    target = artifact_root / "reports" / "eagle3_pipeline_jobs.env"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return target


def nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def normalized_expected(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def check_lookup(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return {}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        area = str(check.get("area") or "")
        name = str(check.get("name") or "")
        if area and name:
            indexed[(area, name)] = check
    return indexed


def check_evidence(check: dict[str, Any] | None) -> dict[str, Any]:
    evidence = (check or {}).get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def validate_pass_preflight_contract(payload: dict[str, Any], *, run_pilot: bool) -> tuple[str, dict[str, Any]]:
    """Require a pass preflight to prove the critical gates, not just claim pass."""
    if not (payload.get("overall_status") == "pass" and payload.get("submit_ready") is True):
        return "skip", {"reason": "preflight is not ready"}
    if not isinstance(payload.get("checks"), list):
        return "fail", {"missing_checks_array": True}

    indexed = check_lookup(payload)
    required = list(REQUIRED_PREFLIGHT_PASS_CHECKS)
    if run_pilot:
        required.extend(PILOT_PREFLIGHT_PASS_CHECKS)
    missing: list[str] = []
    nonpass: list[dict[str, Any]] = []
    for area, name in required:
        check = indexed.get((area, name))
        if check is None:
            missing.append(f"{area}/{name}")
        elif check.get("status") != "pass":
            nonpass.append({"area": area, "name": name, "status": check.get("status"), "detail": check.get("detail")})

    corpus_evidence = check_evidence(indexed.get(("data", "corpus strategy")))
    provenance = corpus_evidence.get("rollout_provenance") if isinstance(corpus_evidence.get("rollout_provenance"), dict) else {}
    corpus_proven = (
        provenance.get("proves_actual_rollout_corpus") is True
        and provenance.get("output_matches_input") is True
        and provenance.get("input_valid") is True
    )

    conversation_evidence = check_evidence(indexed.get(("data", "training conversation validation")))
    conversation_summary = conversation_evidence.get("summary") if isinstance(conversation_evidence.get("summary"), dict) else {}
    conversation_valid = conversation_evidence.get("returncode") == 0 and int(conversation_summary.get("valid_rows") or 0) > 0

    pilot_evidence = check_evidence(indexed.get(("data", "pilot minimum rows")))
    try:
        pilot_valid_rows = int(pilot_evidence.get("valid_rows") or 0)
        pilot_min_rows = int(pilot_evidence.get("min_pilot_rows") or payload.get("min_pilot_rows") or 0)
    except (TypeError, ValueError):
        pilot_valid_rows = 0
        pilot_min_rows = 0
    pilot_rows_ok = (not run_pilot) or (pilot_min_rows > 0 and pilot_valid_rows >= pilot_min_rows)

    problems: dict[str, Any] = {
        "missing": missing,
        "nonpass": nonpass,
        "corpus_proven": corpus_proven,
        "conversation_valid": conversation_valid,
        "pilot_rows_ok": pilot_rows_ok,
        "valid_rows": pilot_valid_rows,
        "min_pilot_rows": pilot_min_rows,
    }
    if missing or nonpass or not corpus_proven or not conversation_valid or not pilot_rows_ok:
        return "fail", problems
    return "pass", problems


def validate(payload: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], str | None]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, **evidence: Any) -> None:
        checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})

    if payload.get("overall_status") == "pass" and payload.get("submit_ready") is True:
        add("preflight readiness", "pass", "preflight reports submit_ready=true")
    else:
        add(
            "preflight readiness",
            "fail",
            "preflight is not ready",
            overall_status=payload.get("overall_status"),
            submit_ready=payload.get("submit_ready"),
        )

    input_data = Path(str(payload.get("input_data") or ""))
    if input_data.exists() and input_data.stat().st_size > 0:
        add("input corpus", "pass", "input corpus exists and is nonempty", path=str(input_data), size_bytes=input_data.stat().st_size)
    else:
        add("input corpus", "fail", "input corpus is missing or empty", path=str(input_data), exists=input_data.exists())

    command = ((payload.get("commands") or {}).get("pilot_submit") or "").strip()
    if not command:
        add("pilot submit command", "fail", "commands.pilot_submit is missing")
        return checks, None

    env, argv = parse_env_command(command)
    expected_script = ["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"]
    if argv == expected_script and env.get("SUBMIT") == "true":
        add("pilot submit command", "pass", "pilot command targets submit_eagle3_pipeline.sh with SUBMIT=true")
    else:
        add("pilot submit command", "fail", "pilot command is not the expected submit wrapper", argv=argv, submit=env.get("SUBMIT"))

    for key in REQUIRED_PIPELINE_ENV:
        value = env.get(key)
        if value:
            add(f"env {key}", "pass", "required env is set", value=value)
        else:
            add(f"env {key}", "fail", "required env is missing")
    answer_only = str(env.get("ANSWER_ONLY_LOSS") or "").lower()
    if answer_only in {"true", "1", "yes"}:
        add("answer-only loss mode", "pass", "pilot pipeline will preserve assistant-only loss masking", value=env.get("ANSWER_ONLY_LOSS"))
    else:
        add(
            "answer-only loss mode",
            "fail",
            "Qwen3 SWE/RL Eagle3 submit must keep ANSWER_ONLY_LOSS=true",
            value=env.get("ANSWER_ONLY_LOSS"),
        )
    for key in POSITIVE_INT_ENV:
        raw = env.get(key)
        try:
            value = int(str(raw))
            ok = value > 0
        except (TypeError, ValueError):
            value = None
            ok = False
        if ok:
            add(f"numeric env {key}", "pass", "numeric submit env is positive", value=value)
        else:
            add(f"numeric env {key}", "fail", "numeric submit env must be a positive integer", value=raw)
    for env_key, payload_keys in PREFLIGHT_ENV_FIELDS.items():
        expected = nested_get(payload, payload_keys)
        if expected is None:
            continue
        actual = env.get(env_key)
        expected_text = normalized_expected(expected)
        if actual == expected_text:
            add(f"env {env_key} matches preflight", "pass", "submit env matches the preflight report", value=actual)
        else:
            add(
                f"env {env_key} matches preflight",
                "fail",
                "submit env differs from the preflight report",
                expected=expected_text,
                actual=actual,
                payload_path=".".join(payload_keys),
            )

    contract_status, contract_evidence = validate_pass_preflight_contract(payload, run_pilot=boolish(env.get("RUN_PILOT", "true")))
    if contract_status == "pass":
        add(
            "preflight critical checks",
            "pass",
            "preflight report includes pass evidence for rollout, runtime, ModelOpt, and dry-run gates",
            **contract_evidence,
        )
    elif contract_status == "fail":
        add(
            "preflight critical checks",
            "fail",
            "preflight report does not prove all critical submit gates",
            **contract_evidence,
        )

    return checks, command


def is_expected_not_ready(checks: list[dict[str, Any]], command: str | None, *, executing: bool) -> bool:
    if executing or not command:
        return False
    failed = {check["name"] for check in checks if check.get("status") == "fail"}
    expected_failures = {"preflight readiness", "input corpus"}
    return bool(failed) and failed.issubset(expected_failures)


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Pipeline Gated Submit",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Executed: **{str(data['executed']).lower()}**",
        f"Expected not-ready: **{str(data.get('expected_not_ready', False)).lower()}**",
        "",
        "Command:",
        "",
        "```bash",
        data.get("command") or "# no command",
        "```",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in data["checks"]:
        lines.append(f"| {check['name']} | {check['status'].upper()} | {check['detail'].replace('|', '/')} |")
    jobs = data.get("jobs") or {}
    if jobs:
        lines += ["", "Submitted jobs:", ""]
        for key, value in jobs.items():
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload, load_error = load_json(args.preflight_json)
    if load_error:
        checks = [{"name": "preflight report", "status": "fail", "detail": load_error, "evidence": {"path": str(args.preflight_json)}}]
        command = None
    else:
        assert payload is not None
        checks, command = validate(payload, args)
    ready = bool(command) and all(check["status"] == "pass" for check in checks)

    run_result: dict[str, Any] | None = None
    jobs: dict[str, str] = {}
    job_file = ROOT / "latest_eagle3_pipeline_jobs.txt"
    job_file_copy: Path | None = None
    if args.execute:
        if not args.allow_heavy_gpu:
            checks.append({"name": "heavy GPU allow flag", "status": "fail", "detail": "--allow-heavy-gpu is required with --execute", "evidence": {}})
            ready = False
        if ready and command:
            env, argv = parse_env_command(command)
            merged = os.environ.copy()
            merged.update(env)
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env=merged,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            run_result = {
                "command": command_string(env, argv),
                "returncode": result.returncode,
                "output_tail": result.stdout[-8000:],
            }
            jobs = job_file_jobs(job_file)
            job_file_copy = copy_job_file(job_file, args.artifact_root)
            if result.returncode != 0:
                checks.append({"name": "pipeline submit execution", "status": "fail", "detail": "submit command returned nonzero", "evidence": run_result})
                ready = False
            else:
                checks.append({"name": "pipeline submit execution", "status": "pass", "detail": "submit command returned zero", "evidence": run_result})
                missing_jobs = [key for key in REQUIRED_PIPELINE_JOBS if not jobs.get(key)]
                if missing_jobs:
                    checks.append(
                        {
                            "name": "pipeline job file",
                            "status": "fail",
                            "detail": "pipeline submit did not record all required stage job ids",
                            "evidence": {
                                "job_file": str(job_file),
                                "job_file_copy": str(job_file_copy) if job_file_copy else None,
                                "jobs": jobs,
                                "missing_jobs": missing_jobs,
                            },
                        }
                    )
                    ready = False
                else:
                    checks.append(
                        {
                            "name": "pipeline job file",
                            "status": "pass",
                            "detail": "pipeline job file contains required stage job ids",
                            "evidence": {
                                "job_file": str(job_file),
                                "job_file_copy": str(job_file_copy) if job_file_copy else None,
                                "jobs": jobs,
                            },
                        }
                    )

    expected_not_ready = is_expected_not_ready(checks, command, executing=args.execute)
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "preflight_json": str(args.preflight_json),
        "overall_status": "pass" if ready else "fail",
        "expected_not_ready": expected_not_ready,
        "executed": bool(args.execute and run_result and run_result.get("returncode") == 0),
        "command": command,
        "checks": checks,
        "run": run_result,
        "jobs": jobs,
        "job_file": str(job_file),
        "job_file_copy": str(job_file_copy) if job_file_copy else None,
        "required_job_keys": list(REQUIRED_PIPELINE_JOBS),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(data), encoding="utf-8")
    print(render_markdown(data))
    if ready:
        return 0
    if args.exit_zero_if_not_ready and expected_not_ready:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
