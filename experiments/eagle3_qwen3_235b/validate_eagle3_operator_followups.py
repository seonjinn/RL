#!/usr/bin/env python3
"""Validate guarded Slurm follow-up reports for Eagle3 operator actions.

This validator does not submit jobs or run analyzers. It checks that
run_eagle3_slurm_followups.py reports are structurally valid, linked to the
current operator sheet, and preserve the safety invariant: no follow-up
analyzer may run until Slurm job ids are terminal and the submit execution
record succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")

SAFE_WAIT_STATES = {"not_submitted", "waiting", "unknown", "blocked"}
READY_STATES = {"ready_for_followup"}
PASS_STATES = {"pass"}
FAIL_STATES = {"fail", "no_followups"}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--followup-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--expect-action",
        action="append",
        default=[],
        help="Require a follow-up report for this Slurm action id. Repeatable.",
    )
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    defaults = {
        "plan_json": Path(os.environ.get("NEXT_ACTION_PLAN_JSON", root / "reports/eagle3_next_actions.json")),
        "operator_sheet_json": Path(os.environ.get("OPERATOR_SHEET_JSON", root / "reports/eagle3_operator_sheet.json")),
        "followup_dir": Path(os.environ.get("OPERATOR_FOLLOWUP_DIR", root / "reports/operator_followups")),
        "json_out": Path(
            os.environ.get("OPERATOR_FOLLOWUP_VALIDATION_JSON", root / "reports/eagle3_operator_followups_validation.json")
        ),
        "markdown_out": Path(
            os.environ.get(
                "OPERATOR_FOLLOWUP_VALIDATION_MARKDOWN",
                root / "reports/eagle3_operator_followups_validation.md",
            )
        ),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def load_json(path: Path | None) -> tuple[Any | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid JSON: {exc}"


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def plan_action_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in as_list(plan.get("next_actions")):
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def sheet_slurm_actions(sheet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in as_list(sheet.get("ready_actions")):
        if isinstance(item, dict) and item.get("id") and item.get("submits_slurm"):
            result[str(item["id"])] = item
    return result


def validate_report(
    checks: list[dict[str, Any]],
    action_id: str,
    sheet_item: dict[str, Any],
    plan_actions: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    report_path = Path(str(sheet_item.get("followup_record") or ""))
    if not report_path:
        add(checks, "report", action_id, "fail", "operator sheet is missing followup_record")
        return None
    execution_path = Path(str(sheet_item.get("execution_record") or ""))
    raw, error = load_json(report_path)
    if error:
        if not execution_path.exists():
            add(
                checks,
                "report",
                action_id,
                "pass",
                "follow-up report is not required before the submit execution record exists",
                path=str(report_path),
                execution_record=str(execution_path),
            )
            if action_id not in plan_actions:
                add(checks, "action", action_id, "warn", "action is not present in the current next-action plan")
            elif not plan_actions[action_id].get("submits_slurm"):
                add(checks, "action", action_id, "fail", "plan action is not marked submits_slurm")
            else:
                add(checks, "action", action_id, "pass", "unsubmitted Slurm action is linked to a Slurm next action")
            add(checks, "mode", action_id, "pass", "no follow-up mode exists before submit execution")
            add(checks, "after", action_id, "pass", "no follow-up commands ran before submit execution")
            add(checks, "slurm", action_id, "pass", "no concrete Slurm jobs exist before submit execution")
            return {
                "overall_status": "not_submitted",
                "mode": "not_started",
                "jobs": [],
                "after_commands": [],
                "after_returncodes": [],
            }
        add(checks, "report", action_id, "fail", error, path=str(report_path))
        return None
    report = as_dict(raw)
    status = str(report.get("overall_status") or "unknown")
    mode = str(report.get("mode") or "")
    jobs = [item for item in as_list(report.get("jobs")) if isinstance(item, dict)]
    after_commands = [str(item) for item in as_list(report.get("after_commands"))]
    after_rows = [item for item in as_list(report.get("after_returncodes")) if isinstance(item, dict)]

    required = [
        "generated_at",
        "artifact_root",
        "plan_json",
        "operator_sheet_json",
        "action_id",
        "overall_status",
        "mode",
        "execution_record",
        "job_files",
        "jobs",
        "after_commands",
        "after_returncodes",
    ]
    missing = [key for key in required if key not in report]
    if missing:
        add(checks, "schema", action_id, "fail", "follow-up report is missing required keys", missing=missing)
    else:
        add(checks, "schema", action_id, "pass", "follow-up report has required keys")

    if report.get("action_id") != action_id:
        add(
            checks,
            "action",
            action_id,
            "fail",
            "follow-up report action_id does not match operator sheet",
            report_action_id=report.get("action_id"),
        )
    elif action_id not in plan_actions:
        add(checks, "action", action_id, "warn", "action is not present in the current next-action plan")
    elif not plan_actions[action_id].get("submits_slurm"):
        add(checks, "action", action_id, "fail", "plan action is not marked submits_slurm")
    else:
        add(checks, "action", action_id, "pass", "follow-up report is linked to a Slurm next action")

    if mode not in {"inspect_only", "execute_after"}:
        add(checks, "mode", action_id, "fail", "follow-up report has an unknown mode", mode=mode)
    elif mode == "inspect_only" and after_rows:
        add(checks, "mode", action_id, "fail", "inspect-only report must not contain after_returncodes", count=len(after_rows))
    elif mode == "execute_after" and status not in PASS_STATES and after_rows:
        add(checks, "mode", action_id, "fail", "non-pass execute_after report contains after_returncodes", report_status=status)
    else:
        add(checks, "mode", action_id, "pass", "follow-up mode preserves no-run safety")

    if not after_commands:
        add(checks, "after", action_id, "fail", "Slurm action follow-up report has no after_commands")
    elif status in SAFE_WAIT_STATES:
        if after_rows:
            add(checks, "after", action_id, "fail", "waiting/not-submitted follow-up report must not run after_commands")
        else:
            add(checks, "after", action_id, "pass", "waiting/not-submitted follow-up report did not run after_commands")
    elif status in READY_STATES and after_rows:
        add(checks, "after", action_id, "fail", "ready_for_followup report should be inspect-only until --execute-after")
    elif status in PASS_STATES:
        expected = len(after_commands)
        ok = mode == "execute_after" and len(after_rows) == expected and all(row.get("returncode") == 0 for row in after_rows)
        add(
            checks,
            "after",
            action_id,
            "pass" if ok else "fail",
            "pass report ran all follow-up commands successfully" if ok else "pass report does not prove all follow-up commands succeeded",
            expected=expected,
            observed=len(after_rows),
            after_returncodes=after_rows,
        )
    elif status in FAIL_STATES:
        add(checks, "after", action_id, "fail", "follow-up guard reported failure", report_status=status)
    else:
        add(checks, "after", action_id, "warn", "follow-up guard reported an unclassified state", report_status=status)

    if status == "not_submitted":
        if jobs:
            add(checks, "slurm", action_id, "fail", "not_submitted report contains concrete jobs", jobs=jobs)
        else:
            add(checks, "slurm", action_id, "pass", "not_submitted report has no concrete Slurm jobs")
    elif status == "waiting":
        if any(job.get("status") == "active" for job in jobs):
            add(checks, "slurm", action_id, "pass", "waiting report has at least one active Slurm job")
        else:
            add(checks, "slurm", action_id, "fail", "waiting report lacks active Slurm job evidence", jobs=jobs)
    elif status == "ready_for_followup":
        execution = as_dict(report.get("execution_record"))
        if jobs and all(job.get("terminal") for job in jobs) and execution.get("status") == "pass":
            add(checks, "slurm", action_id, "pass", "ready_for_followup proves terminal jobs and successful execution record")
        else:
            add(
                checks,
                "slurm",
                action_id,
                "fail",
                "ready_for_followup lacks terminal job or execution-record proof",
                jobs=jobs,
                execution_record=execution,
            )
    elif status == "pass":
        if jobs and all(job.get("terminal") for job in jobs):
            add(checks, "slurm", action_id, "pass", "pass report has terminal Slurm jobs")
        else:
            add(checks, "slurm", action_id, "fail", "pass report lacks terminal Slurm job evidence", jobs=jobs)
    else:
        add(checks, "slurm", action_id, "pass", "follow-up report state is safe for non-execution", report_status=status)

    return report


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    plan_raw, plan_error = load_json(args.plan_json)
    sheet_raw, sheet_error = load_json(args.operator_sheet_json)
    plan = as_dict(plan_raw)
    sheet = as_dict(sheet_raw)
    plan_actions = plan_action_map(plan)
    slurm_actions = sheet_slurm_actions(sheet)

    checks: list[dict[str, Any]] = []
    if plan_error:
        add(checks, "input", "next-action plan", "fail", plan_error, path=str(args.plan_json))
    else:
        add(checks, "input", "next-action plan", "pass", "next-action plan is readable", actions=len(plan_actions))
    if sheet_error:
        add(checks, "input", "operator sheet", "fail", sheet_error, path=str(args.operator_sheet_json))
    else:
        add(checks, "input", "operator sheet", "pass", "operator sheet is readable", slurm_actions=len(slurm_actions))

    expected = set(slurm_actions)
    expected.update(str(item) for item in args.expect_action)
    if not expected:
        add(checks, "schema", "slurm actions", "pass", "operator sheet contains no Slurm ready actions requiring follow-up")

    reports: dict[str, dict[str, Any]] = {}
    for action_id in sorted(expected):
        sheet_item = slurm_actions.get(action_id)
        if not sheet_item:
            add(checks, "expectation", action_id, "fail", "expected action is missing from operator sheet Slurm actions")
            continue
        report = validate_report(checks, action_id, sheet_item, plan_actions)
        if report:
            reports[action_id] = report

    state_counts: dict[str, int] = {}
    for report in reports.values():
        status = str(report.get("overall_status") or "unknown")
        state_counts[status] = state_counts.get(status, 0) + 1

    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1

    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"
    else:
        status = "pass"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "artifact_root": str(args.artifact_root),
        "plan_json": str(args.plan_json),
        "operator_sheet_json": str(args.operator_sheet_json),
        "followup_dir": str(args.followup_dir),
        "expected_actions": sorted(expected),
        "counts": counts,
        "followup_state_counts": state_counts,
        "reports": {
            action_id: {
                "path": slurm_actions[action_id].get("followup_record"),
                "overall_status": report.get("overall_status"),
                "mode": report.get("mode"),
                "job_count": len(as_list(report.get("jobs"))),
                "after_returncode_count": len(as_list(report.get("after_returncodes"))),
            }
            for action_id, report in sorted(reports.items())
            if action_id in slurm_actions
        },
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator Follow-Up Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Plan JSON: `{payload['plan_json']}`",
        f"Operator sheet JSON: `{payload['operator_sheet_json']}`",
        f"Follow-up dir: `{payload['followup_dir']}`",
        "",
        "This report is no-submit. It validates the safety shape of guarded Slurm follow-up reports.",
        "",
        "## Reports",
        "",
        "| action | status | mode | jobs | after returncodes | report |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for action_id, report in payload["reports"].items():
        lines.append(
            f"| `{action_id}` | `{report.get('overall_status')}` | `{report.get('mode')}` | "
            f"{report.get('job_count')} | {report.get('after_returncode_count')} | `{report.get('path')}` |"
        )
    if not payload["reports"]:
        lines.append("| - | - | - | - | - | - |")

    lines += [
        "",
        "## Checks",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['area']} | {check['name']} | {check['status'].upper()} | {detail} |")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = with_defaults(parse_args())
    payload = build_payload(args)
    markdown = render_markdown(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if payload["overall_status"] == "fail":
        return 1
    if args.fail_on_warn and payload["overall_status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
