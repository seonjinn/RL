#!/usr/bin/env python3
"""Validate no-submit gated pipeline submit semantics.

This is a synthetic, no-submit contract test for
submit_eagle3_pipeline_if_ready.py. It proves that an expected missing-corpus
state can be recorded without failing operator refresh, while malformed or
unsafe submit commands still return nonzero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pilot_command(
    input_data: Path,
    *,
    submit: str = "true",
    script: str = "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh",
    omit_env: set[str] | None = None,
) -> str:
    omit_env = omit_env or set()
    artifact_root = input_data.parent.parent
    env = {
        "SUBMIT": submit,
        "ARTIFACT_ROOT": str(artifact_root),
        "RUN_PILOT": "true",
        "SBATCH_ACCOUNT": "coreai_dlalgo_nemorl",
        "SBATCH_PARTITION": "batch",
        "DUMP_GPUS_PER_NODE": "4",
        "TRAIN_GPUS_PER_NODE": "4",
        "EXPORT_GPUS_PER_NODE": "1",
        "TP": "4",
        "INPUT_DATA": str(input_data),
        "HIDDEN_STATES_DIR": "/tmp/hiddens",
        "OUTPUT_DIR": "/tmp/modelopt_ckpt",
        "EXPORT_DIR": "/tmp/exported_hf",
        "VLLM_DRAFT_DIR": "/tmp/vllm_draft",
        "VERIFIER_CONFIG_DIR": "/tmp/verifier",
        "MODELOPT_DIR": "/tmp/modelopt",
        "CHAT_TEMPLATE": "/tmp/qwen3_generation_template.jinja2",
        "ARCH_ENV_FILE": "/tmp/eagle3_architecture.env",
        "REFERENCE_ARCH": "/tmp/eagle3_architecture.json",
        "BASE_MODEL": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "ANSWER_ONLY_LOSS": "true",
        "TRAINING_SEQ_LEN": "16384",
        "MAX_SEQ_LEN": "16384",
    }
    for key in omit_env:
        env.pop(key, None)
    prefix = " ".join(f"{key}={value}" for key, value in env.items())
    return f"{prefix} bash {script}"


def write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(count):
        rows.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"task {idx}"},
                        {"role": "assistant", "content": f"answer {idx}"},
                    ]
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def preflight_checks(*, valid_rows: int = 8, proves_rollout: bool = True) -> list[dict[str, Any]]:
    return [
        {"area": "paths", "name": "input conversation JSONL", "status": "pass", "detail": "visible"},
        {"area": "paths", "name": "chat template", "status": "pass", "detail": "visible"},
        {"area": "paths", "name": "verifier config.json", "status": "pass", "detail": "visible"},
        {"area": "paths", "name": "Eagle3 architecture JSON", "status": "pass", "detail": "visible"},
        {"area": "paths", "name": "Eagle3 architecture env", "status": "pass", "detail": "visible"},
        {"area": "modelopt", "name": "TRT-LLM loss-mask patch", "status": "pass", "detail": "validated"},
        {"area": "data", "name": "answer-only chat template tags", "status": "pass", "detail": "generation tags present"},
        {
            "area": "data",
            "name": "corpus strategy",
            "status": "pass",
            "detail": "corpus strategy proves target-aligned rollout corpus",
            "evidence": {
                "rollout_provenance": {
                    "proves_actual_rollout_corpus": proves_rollout,
                    "output_matches_input": True,
                    "input_valid": True,
                }
            },
        },
        {"area": "data", "name": "rollout state advance", "status": "pass", "detail": "ready for pipeline"},
        {"area": "execution", "name": "container preflight", "status": "pass", "detail": "container preflight passed"},
        {
            "area": "data",
            "name": "training conversation validation",
            "status": "pass",
            "detail": "conversation JSONL validates",
            "evidence": {"returncode": 0, "summary": {"valid_rows": valid_rows}},
        },
        {
            "area": "data",
            "name": "pilot minimum rows",
            "status": "pass",
            "detail": "pilot rows satisfy threshold",
            "evidence": {"valid_rows": valid_rows, "min_pilot_rows": 8},
        },
        {"area": "slurm", "name": "GPU capacity vs pipeline requests", "status": "pass", "detail": "capacity proven"},
        {"area": "validation", "name": "local ModelOpt pipeline preflight", "status": "pass", "detail": "local preflight passed"},
        {"area": "dry_run", "name": "ModelOpt wrapper dry-runs", "status": "pass", "detail": "wrappers passed"},
        {"area": "dry_run", "name": "Slurm pipeline dry-run", "status": "pass", "detail": "dry-run passed"},
    ]


def write_preflight(
    path: Path,
    *,
    status: str,
    submit_ready: bool,
    input_data: Path,
    command: str | None = None,
    include_checks: bool = True,
    proves_rollout: bool = True,
) -> None:
    artifact_root = input_data.parent.parent
    payload = {
        "generated_at": "synthetic",
        "overall_status": status,
        "submit_ready": submit_ready,
        "artifact_root": str(artifact_root),
        "input_data": str(input_data),
        "hidden_states_dir": "/tmp/hiddens",
        "output_dir": "/tmp/modelopt_ckpt",
        "export_dir": "/tmp/exported_hf",
        "vllm_draft_dir": "/tmp/vllm_draft",
        "verifier_config_dir": "/tmp/verifier",
        "modelopt_dir": "/tmp/modelopt",
        "chat_template": "/tmp/qwen3_generation_template.jinja2",
        "arch_env_file": "/tmp/eagle3_architecture.env",
        "reference_arch": "/tmp/eagle3_architecture.json",
        "base_model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "answer_only_loss": "true",
        "training_seq_len": 16384,
        "max_seq_len": 16384,
        "min_pilot_rows": 8,
        "resource_request": {
            "dump_gpus_per_node": 4,
            "train_gpus_per_node": 4,
            "export_gpus_per_node": 1,
            "tp": 4,
        },
        "commands": {"pilot_submit": command if command is not None else pilot_command(input_data)},
    }
    if include_checks:
        payload["checks"] = preflight_checks(proves_rollout=proves_rollout)
    write_json(path, payload)


def run_helper(root: Path, preflight: Path, *, exit_zero_if_not_ready: bool = False) -> dict[str, Any]:
    out = root / "reports/eagle3_pipeline_gated_submit.json"
    md = out.with_suffix(".md")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--artifact-root",
        str(root),
        "--preflight-json",
        str(preflight),
        "--json-out",
        str(out),
        "--markdown-out",
        str(md),
    ]
    if exit_zero_if_not_ready:
        cmd.append("--exit-zero-if-not-ready")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload = read_json(out)
    return {
        "returncode": result.returncode,
        "payload": payload,
        "output_tail": result.stdout[-4000:],
    }


def scenario_not_ready_without_flag(root: Path) -> dict[str, Any]:
    input_data = root / "data/missing.jsonl"
    preflight = root / "reports/not_ready_preflight.json"
    write_preflight(preflight, status="incomplete", submit_ready=False, input_data=input_data)
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def scenario_not_ready_with_flag(root: Path) -> dict[str, Any]:
    input_data = root / "data/missing.jsonl"
    preflight = root / "reports/not_ready_flag_preflight.json"
    write_preflight(preflight, status="incomplete", submit_ready=False, input_data=input_data)
    return run_helper(root, preflight, exit_zero_if_not_ready=True)


def scenario_ready_no_execute(root: Path) -> dict[str, Any]:
    input_data = root / "data/ready.jsonl"
    write_rows(input_data, 8)
    preflight = root / "reports/ready_preflight.json"
    write_preflight(preflight, status="pass", submit_ready=True, input_data=input_data)
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def scenario_ready_missing_preflight_checks(root: Path) -> dict[str, Any]:
    input_data = root / "data/ready_missing_preflight_checks.jsonl"
    write_rows(input_data, 8)
    preflight = root / "reports/ready_missing_preflight_checks.json"
    write_preflight(preflight, status="pass", submit_ready=True, input_data=input_data, include_checks=False)
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def scenario_ready_weak_rollout_provenance(root: Path) -> dict[str, Any]:
    input_data = root / "data/ready_weak_rollout_provenance.jsonl"
    write_rows(input_data, 8)
    preflight = root / "reports/ready_weak_rollout_provenance.json"
    write_preflight(preflight, status="pass", submit_ready=True, input_data=input_data, proves_rollout=False)
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def scenario_bad_command_with_flag(root: Path) -> dict[str, Any]:
    input_data = root / "data/missing_bad_command.jsonl"
    preflight = root / "reports/bad_command_preflight.json"
    write_preflight(
        preflight,
        status="incomplete",
        submit_ready=False,
        input_data=input_data,
        command=pilot_command(input_data, script="experiments/eagle3_qwen3_235b/wrong_submit.sh"),
    )
    return run_helper(root, preflight, exit_zero_if_not_ready=True)


def scenario_ready_missing_critical_env(root: Path) -> dict[str, Any]:
    input_data = root / "data/ready_missing_critical_env.jsonl"
    write_rows(input_data, 8)
    preflight = root / "reports/ready_missing_critical_env_preflight.json"
    write_preflight(
        preflight,
        status="pass",
        submit_ready=True,
        input_data=input_data,
        command=pilot_command(input_data, omit_env={"CHAT_TEMPLATE"}),
    )
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def scenario_ready_mismatched_input_data(root: Path) -> dict[str, Any]:
    input_data = root / "data/ready_payload_input.jsonl"
    command_input = root / "data/ready_command_input.jsonl"
    write_rows(input_data, 8)
    write_rows(command_input, 8)
    preflight = root / "reports/ready_mismatched_input_data_preflight.json"
    write_preflight(
        preflight,
        status="pass",
        submit_ready=True,
        input_data=input_data,
        command=pilot_command(command_input),
    )
    return run_helper(root, preflight, exit_zero_if_not_ready=False)


def check_not_ready_without_flag(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    if payload.get("expected_not_ready") is not True:
        problems.append("expected_not_ready is not true")
    if payload.get("executed") is not False:
        problems.append("executed is not false")
    return problems


def check_not_ready_with_flag(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 0:
        problems.append(f"returncode {result['returncode']!r} != 0")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    if payload.get("expected_not_ready") is not True:
        problems.append("expected_not_ready is not true")
    if payload.get("executed") is not False:
        problems.append("executed is not false")
    return problems


def check_ready_no_execute(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 0:
        problems.append(f"returncode {result['returncode']!r} != 0")
    if payload.get("overall_status") != "pass":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'pass'")
    if payload.get("expected_not_ready") is not False:
        problems.append("expected_not_ready is not false")
    if payload.get("executed") is not False:
        problems.append("executed is not false")
    return problems


def check_ready_missing_preflight_checks(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    failed = {check.get("name") for check in payload.get("checks", []) if check.get("status") == "fail"}
    if "preflight critical checks" not in failed:
        problems.append("missing checks[] did not fail the critical preflight coverage gate")
    return problems


def check_ready_weak_rollout_provenance(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    failed = {check.get("name") for check in payload.get("checks", []) if check.get("status") == "fail"}
    if "preflight critical checks" not in failed:
        problems.append("weak rollout provenance did not fail the critical preflight coverage gate")
    return problems


def check_bad_command_with_flag(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("expected_not_ready") is not False:
        problems.append("bad command was incorrectly treated as expected_not_ready")
    failed = {check.get("name") for check in payload.get("checks", []) if check.get("status") == "fail"}
    if "pilot submit command" not in failed:
        problems.append("bad command did not fail pilot submit command check")
    return problems


def check_ready_missing_critical_env(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    if payload.get("expected_not_ready") is not False:
        problems.append("missing critical env was incorrectly treated as expected_not_ready")
    failed = {check.get("name") for check in payload.get("checks", []) if check.get("status") == "fail"}
    if "env CHAT_TEMPLATE" not in failed:
        problems.append("missing CHAT_TEMPLATE did not fail the critical env gate")
    return problems


def check_ready_mismatched_input_data(result: dict[str, Any]) -> list[str]:
    payload = result["payload"]
    problems: list[str] = []
    if result["returncode"] != 1:
        problems.append(f"returncode {result['returncode']!r} != 1")
    if payload.get("overall_status") != "fail":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'fail'")
    if payload.get("expected_not_ready") is not False:
        problems.append("mismatched input data was incorrectly treated as expected_not_ready")
    failed = {check.get("name") for check in payload.get("checks", []) if check.get("status") == "fail"}
    if "env INPUT_DATA matches preflight" not in failed:
        problems.append("mismatched INPUT_DATA did not fail the preflight consistency gate")
    return problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Pipeline Gated Submit Contract",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["scenarios"]:
        detail = "; ".join(item["problems"]) if item["problems"] else "-"
        lines.append(f"| {item['name']} | {item['status']} | {detail.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="pipeline_gated_submit_contract_"))
    try:
        raw = [
            ("not_ready_without_flag", scenario_not_ready_without_flag(temp_root), check_not_ready_without_flag),
            ("not_ready_with_flag", scenario_not_ready_with_flag(temp_root), check_not_ready_with_flag),
            ("ready_no_execute", scenario_ready_no_execute(temp_root), check_ready_no_execute),
            (
                "ready_missing_preflight_checks",
                scenario_ready_missing_preflight_checks(temp_root),
                check_ready_missing_preflight_checks,
            ),
            (
                "ready_weak_rollout_provenance",
                scenario_ready_weak_rollout_provenance(temp_root),
                check_ready_weak_rollout_provenance,
            ),
            ("bad_command_with_flag", scenario_bad_command_with_flag(temp_root), check_bad_command_with_flag),
            ("ready_missing_critical_env", scenario_ready_missing_critical_env(temp_root), check_ready_missing_critical_env),
            ("ready_mismatched_input_data", scenario_ready_mismatched_input_data(temp_root), check_ready_mismatched_input_data),
        ]
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    scenarios: list[dict[str, Any]] = []
    problems: list[str] = []
    for name, result, checker in raw:
        item_problems = checker(result)
        scenarios.append(
            {
                "name": name,
                "status": "pass" if not item_problems else "fail",
                "returncode": result["returncode"],
                "problems": item_problems,
            }
        )
        problems.extend(f"{name}: {problem}" for problem in item_problems)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "pass" if not problems else "fail",
        "scenarios": scenarios,
        "problems": problems,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if payload["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
