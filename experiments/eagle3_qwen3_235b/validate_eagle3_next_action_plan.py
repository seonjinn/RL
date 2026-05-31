#!/usr/bin/env python3
"""Validate the semantic shape of eagle3_next_actions.json.

This validator does not submit jobs. It checks that the next-action report is
safe to hand to an operator: action ids are unique, Slurm/GPU flags are
accurate for known actions, ready actions have commands, and follow-up analyzer
commands are present for submitted gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")

KNOWN_ACTION_RULES: dict[str, dict[str, Any]] = {
    "probe_remote_hosts": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": [
            "probe_eagle3_remote_host.py",
            "--include-ssh-config-hosts",
            "--hosts",
            "--remote-workdir",
            "eagle3_remote_host_probe.json",
        ],
        "after_contains": ["plan_eagle3_next_actions.py", "refresh_eagle3_operator_state.py"],
    },
    "submit_container_preflight": {
        "submits_slurm": True,
        "heavy_gpu": False,
        "command_contains": ["submit_eagle3_container_preflight.sh", "SUBMIT=true"],
        "after_contains": [
            "analyze_container_preflight.py",
            "--modelopt-dir",
            "--verifier-config-dir",
            "--input-data",
            "--chat-template",
            "--mounts",
            "plan_eagle3_next_actions.py",
        ],
    },
    "submit_rollout_capture": {
        "submits_slurm": True,
        "heavy_gpu": True,
        "command_contains": ["run_rollout_capture_smoke.sh", "DRY_RUN=false", "MAX_NUM_STEPS=1"],
        "after_contains": [
            "analyze_rollout_capture_job.py",
            "advance_rollout_capture_state.py",
            "plan_eagle3_next_actions.py",
        ],
    },
    "poll_vllm_source_build": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": ["squeue", "sacct", "vllm_native_source_build.md"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "submit_vllm_source_build": {
        "submits_slurm": True,
        "heavy_gpu": False,
        "command_contains": ["submit_vllm_native_source_build.sh", "SUBMIT=true", "SBATCH_ACCOUNT=coreai_dlalgo_nemorl"],
        "after_contains": ["analyze_vllm_source_build_job.py", "vllm_source_build_job_analysis.json", "plan_eagle3_next_actions.py"],
    },
    "submit_source_vllm_abi_probe": {
        "submits_slurm": True,
        "heavy_gpu": False,
        "command_contains": ["submit_vllm_native_abi_probe.sh", "SUBMIT=true", "VLLM_SITE_CANDIDATES"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "submit_megatron_compat_probe": {
        "submits_slurm": True,
        "heavy_gpu": False,
        "command_contains": ["submit_megatron_compat_probe.sh", "SUBMIT=true", "SBATCH_ACCOUNT=coreai_dlalgo_nemorl"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "poll_megatron_compat_probe": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": [
            "followup_megatron_probe_to_rollout.sh",
            "PROBE_JOB_ID=",
            "SUBMIT_ROLLOUT=false",
        ],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "submit_eagle3_pilot_pipeline": {
        "submits_slurm": True,
        "heavy_gpu": True,
        "command_contains": [
            "submit_eagle3_pipeline_if_ready.py",
            "--preflight-json",
            "eagle3_pipeline_submit_preflight.json",
            "--json-out",
            "eagle3_pipeline_gated_submit.json",
            "--execute",
            "--allow-heavy-gpu",
        ],
        "after_contains": [
            "analyze_eagle3_pipeline.py",
            "--job-file",
            "--logs-dir",
            "--hidden-validation-json",
            "--training-checkpoint-json",
            "--export-artifacts-json",
            "--markdown-out",
            "--json-out",
            "plan_eagle3_next_actions.py",
        ],
    },
    "submit_full_swegym_rollout": {
        "submits_slurm": True,
        "heavy_gpu": True,
        "command_contains": [
            "submit_full_rollout_after_smoke_if_ready.py",
            "--execute",
            "--allow-heavy-gpu",
            "--start-watcher",
            "--allow-background",
        ],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "submit_rollout_fallback": {
        "submits_slurm": True,
        "heavy_gpu": True,
        "command_contains": [
            "submit_source_vllm_rollout_smoke.sh",
            "DRY_RUN=false",
            "OUTPUT_CONVERSATIONS",
            "NUM_NODES=",
            "NUM_GEN_NODES=",
        ],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "run_post_export_artifact_validations": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": [
            "validate_eagle3_training_checkpoint.py",
            "compare_eagle3_configs.py",
            "validate_eagle3_export_artifacts.py",
            "--require-modelopt-state-load",
            "--fail-on-error",
        ],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "submit_trained_draft_spec_tokens_sweep": {
        "submits_slurm": True,
        "heavy_gpu": True,
        "command_contains": [
            "submit_trained_draft_spec_tokens_sweep.sh",
            "SUBMIT=true",
            "ARTIFACT_ROOT",
            "REPO_ROOT",
            "SWE_REPO_ROOT",
            "CONFIG_FILE",
            "ENV_FILE",
            "CHAT_TEMPLATE",
            "VLLM_DRAFT_DIR",
        ],
        "after_contains": [
            "analyze_spec_tokens_sweep.py",
            "--fail-on-missing-spec-metrics",
            "audit_eagle3_completion.py",
            "plan_eagle3_next_actions.py",
        ],
    },
    "run_pipeline_submit_preflight": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": [
            "preflight_eagle3_pipeline_submit.py",
            "--container-preflight-json",
            "eagle3_pipeline_submit_preflight.json",
        ],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "rollout_poll": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": ["advance_rollout_capture_state.py"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "rollout_materialize": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": ["materialize"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "rollout_materialize_and_refresh": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": ["advance_rollout_capture_state.py", "--materialize"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
    "rollout_pipeline_dry_run": {
        "submits_slurm": False,
        "heavy_gpu": False,
        "command_contains": ["bootstrap_eagle3_path.sh"],
        "after_contains": ["plan_eagle3_next_actions.py"],
    },
}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path(os.environ.get("NEXT_ACTION_PLAN_JSON", artifact_root / "reports/eagle3_next_actions.json")),
    )
    parser.add_argument(
        "--expect-ready-action",
        action="append",
        default=[],
        help="Require this action id to be present with status=ready_for_operator. Repeatable.",
    )
    parser.add_argument(
        "--forbid-ready-action",
        action="append",
        default=[],
        help="Require this action id not to be ready_for_operator. Repeatable.",
    )
    parser.add_argument("--require-after-commands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"plan JSON is not visible: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse plan JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"plan JSON top-level value is not an object: {path}")
    return payload


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def action_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = plan.get("next_actions") or []
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def check_top_level(plan: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    required = ["overall_status", "artifact_root", "build_path", "reports", "training", "next_actions", "blockers"]
    missing = [key for key in required if key not in plan]
    if missing:
        add(checks, "schema", "top-level keys", "fail", "missing required keys", missing=missing)
    else:
        add(checks, "schema", "top-level keys", "pass", "required top-level keys are present")
    if not isinstance(plan.get("next_actions"), list):
        add(checks, "schema", "next_actions list", "fail", "next_actions is not a list")
    else:
        add(checks, "schema", "next_actions list", "pass", "next_actions is a list", count=len(plan["next_actions"]))


def check_unique_actions(plan: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    ids = [str(item.get("id")) for item in plan.get("next_actions") or [] if isinstance(item, dict) and item.get("id")]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        add(checks, "schema", "unique action ids", "fail", "duplicate action ids found", duplicates=duplicates)
    else:
        add(checks, "schema", "unique action ids", "pass", "action ids are unique", ids=ids)


def contains_all(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle not in text]


def check_action_rules(plan: dict[str, Any], checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    actions = action_map(plan)
    for action_id, item in actions.items():
        required_fields = ["id", "title", "status", "stage", "submits_slurm", "heavy_gpu"]
        missing = [key for key in required_fields if key not in item]
        if missing:
            add(checks, "action", action_id, "fail", "action is missing required fields", missing=missing)
            continue
        if item.get("status") == "ready_for_operator" and not item.get("command"):
            add(checks, "action", action_id, "fail", "ready action has no command")
            continue
        rule = KNOWN_ACTION_RULES.get(action_id)
        if not rule:
            add(checks, "action", action_id, "warn", "no built-in rule for this action id", action_status=item.get("status"))
            continue
        problems: list[str] = []
        if item.get("submits_slurm") is not rule["submits_slurm"]:
            problems.append(f"submits_slurm={item.get('submits_slurm')}, expected {rule['submits_slurm']}")
        if item.get("heavy_gpu") is not rule["heavy_gpu"]:
            problems.append(f"heavy_gpu={item.get('heavy_gpu')}, expected {rule['heavy_gpu']}")
        command = str(item.get("command") or "")
        missing_command = contains_all(command, rule["command_contains"]) if command else []
        if item.get("status") == "ready_for_operator" and missing_command:
            problems.append(f"command missing snippets: {missing_command}")
        after_commands = item.get("after_commands") or []
        after_text = "\n".join(str(command) for command in after_commands)
        missing_after = contains_all(after_text, rule["after_contains"])
        if args.require_after_commands and item.get("status") == "ready_for_operator" and missing_after:
            problems.append(f"after_commands missing snippets: {missing_after}")
        if problems:
            add(checks, "action", action_id, "fail", "; ".join(problems), action=item)
        else:
            add(
                checks,
                "action",
                action_id,
                "pass",
                "action flags, command, and follow-up commands match the expected rule",
                action_status=item.get("status"),
                submits_slurm=item.get("submits_slurm"),
                heavy_gpu=item.get("heavy_gpu"),
                after_command_count=len(after_commands),
            )


def check_rollout_poll_outputs(plan: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    item = action_map(plan).get("rollout_poll")
    if not item or item.get("status") != "ready_for_operator":
        return
    report = str(item.get("report") or "")
    command = str(item.get("command") or "")
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        add(checks, "action", "rollout_poll outputs", "fail", f"cannot shell-split command: {exc}")
        return
    if not report:
        add(checks, "action", "rollout_poll outputs", "fail", "rollout_poll action is missing report path")
        return
    if "--json-out" not in tokens:
        add(checks, "action", "rollout_poll outputs", "fail", "rollout_poll command is missing --json-out")
        return
    idx = tokens.index("--json-out")
    observed = tokens[idx + 1] if idx + 1 < len(tokens) else ""
    if observed == report:
        add(checks, "action", "rollout_poll outputs", "pass", "rollout_poll command updates the selected state report", report=report)
    else:
        add(
            checks,
            "action",
            "rollout_poll outputs",
            "fail",
            "rollout_poll --json-out does not match the selected state report",
            report=report,
            observed=observed,
        )


def check_expectations(plan: dict[str, Any], checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    actions = action_map(plan)
    for action_id in args.expect_ready_action:
        item = actions.get(action_id)
        if item and item.get("status") == "ready_for_operator":
            add(checks, "expectation", action_id, "pass", "expected ready action is present")
        else:
            add(checks, "expectation", action_id, "fail", "expected ready action is missing or not ready", action=item)
    for action_id in args.forbid_ready_action:
        item = actions.get(action_id)
        if item and item.get("status") == "ready_for_operator":
            add(checks, "expectation", action_id, "fail", "forbidden action is ready", action=item)
        else:
            add(checks, "expectation", action_id, "pass", "forbidden ready action is not present")


def check_training(plan: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    stages = (plan.get("training") or {}).get("stages") or []
    by_name = {item.get("name"): item for item in stages if isinstance(item, dict)}
    missing = [name for name in ("pilot", "swegym_first_calibration", "production_candidate") if name not in by_name]
    if missing:
        add(checks, "training", "stage plan", "fail", "missing expected training stages", missing=missing)
        return
    optional_expected = ["target_domain_calibration", "generic_optional"]
    optional_missing = [name for name in optional_expected if name not in by_name]
    pilot = by_name["pilot"]
    first_calibration = by_name["swegym_first_calibration"]
    target_calibration = by_name.get("target_domain_calibration")
    production = by_name["production_candidate"]
    problems = []
    if int(pilot.get("examples") or 0) > 64 or int(pilot.get("max_steps") or 0) > 50:
        problems.append("pilot stage is not small enough for a wiring smoke")
    if int(first_calibration.get("examples") or 0) < 1000:
        problems.append("first SWE-Gym calibration stage has too few examples")
    if target_calibration and int(target_calibration.get("examples") or 0) < 10000:
        problems.append("target-domain calibration stage has too few examples")
    if int(production.get("examples") or 0) < 50000:
        problems.append("production candidate has too few examples")
    if optional_missing:
        problems.append(f"optional training stages missing: {', '.join(optional_missing)}")
    if problems:
        add(checks, "training", "stage plan", "fail", "; ".join(problems), stages=stages)
    else:
        add(
            checks,
            "training",
            "stage plan",
            "pass",
            "pilot, SWE-Gym first calibration, target-domain calibration, production, and generic optional stages are scaled as expected",
            pilot=pilot,
            swegym_first_calibration=first_calibration,
            target_domain_calibration=target_calibration,
            production_candidate=production,
        )


def overall_status(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "fail" for item in checks):
        return "fail"
    if any(item["status"] == "warn" for item in checks):
        return "warn"
    return "pass"


def build_payload(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    check_top_level(plan, checks)
    check_unique_actions(plan, checks)
    check_action_rules(plan, checks, args)
    check_rollout_poll_outputs(plan, checks)
    check_expectations(plan, checks, args)
    check_training(plan, checks)
    counts: dict[str, int] = {}
    for item in checks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks),
        "plan_json": str(args.plan_json),
        "plan_overall_status": plan.get("overall_status"),
        "counts": counts,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Next-Action Plan Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Plan status: `{payload.get('plan_overall_status')}`",
        f"Plan JSON: `{payload['plan_json']}`",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for item in payload["checks"]:
        lines.append(
            f"| {item['area']} | {item['name']} | {item['status'].upper()} | "
            f"{item['detail'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan_json)
    payload = build_payload(plan, args)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if payload["overall_status"] == "fail":
        return 1
    if args.fail_on_warn and payload["overall_status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
