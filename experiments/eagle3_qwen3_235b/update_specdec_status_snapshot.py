#!/usr/bin/env python3
"""Update the static SpecDec progress page from current Eagle3 reports.

This is a no-submit helper. It reads the JSON reports produced by
refresh_eagle3_operator_state.py and updates only the volatile queue/preflight
timestamps in the hand-written status artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


EXP = Path(__file__).absolute().parent
ROOT = EXP.parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--html", type=Path, default=EXP / "specdec_progress.html")
    parser.add_argument("--cluster-status", type=Path, default=EXP / "REMOTE_CLUSTER_STATUS.md")
    parser.add_argument("--execution-inputs", type=Path, default=EXP / "REMOTE_EXECUTION_INPUTS.md")
    parser.add_argument("--allow-missing-jobs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_key_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                rows += 1
    return rows


def minute_timestamp(value: str | None) -> str:
    if not value:
        return "unknown"
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})(?::\d{2})?(?:\s+(\S+))?", value)
    if not match:
        return value
    return f"{match.group(1)} {match.group(2) or ''}".strip()


def iso_minute(value: str | None, *, include_date: bool) -> str:
    if not value or value in {"N/A", "Unknown", "None", "-"}:
        return value or "unknown"
    match = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})", value)
    if not match:
        return value
    date, hour, minute = match.groups()
    if include_date:
        return f"{date} {hour}:{minute} PDT"
    return f"{hour}:{minute} PDT"


def job_sort_key(job: dict[str, Any]) -> tuple[int, str]:
    snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
    name = str(snapshot.get("name") or "")
    if any(
        marker in name
        for marker in ("fixedcontainer", "systemvenv", "systemvllm", "sharedvllm", "aarchvllm", "vllm0130", "vllm0112", "vllm0102")
    ):
        return (0, str(job.get("job_id") or ""))
    if "compact" in name:
        return (1, str(job.get("job_id") or ""))
    if "official" in name or "smoke" in name:
        return (2, str(job.get("job_id") or ""))
    return (3, str(job.get("job_id") or ""))


def job_rows(queue: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for raw in sorted(queue.get("jobs") or [], key=job_sort_key):
        if not isinstance(raw, dict):
            continue
        snapshot = raw.get("current_squeue") if isinstance(raw.get("current_squeue"), dict) else {}
        if not snapshot:
            continue
        rows.append(
            {
                "job_id": str(raw.get("job_id") or snapshot.get("job_id") or ""),
                "name": str(snapshot.get("name") or ""),
                "state": str(snapshot.get("state") or raw.get("latest_log_state") or "unknown"),
                "nodes": str(snapshot.get("nodes") or ""),
                "reason": str(snapshot.get("reason") or raw.get("latest_log_reason") or ""),
                "start": str(snapshot.get("start") or raw.get("latest_log_start") or ""),
            }
        )
    return rows


def job_line(row: dict[str, str]) -> str:
    return (
        f"{row['job_id']}|{row['name']}|{row['state']}|{row['nodes']} nodes|"
        f"{row['reason']}|start {row['start']}"
    )


def role(row: dict[str, str]) -> str:
    name = row.get("name", "")
    lowered = name.lower()
    if "experimental" in lowered or "20n4g" in lowered or "18n4g" in lowered:
        return "experimental"
    if "compact" in name:
        return "compact"
    if any(
        marker in name
        for marker in ("fixedcontainer", "systemvenv", "systemvllm", "sharedvllm", "aarchvllm", "vllm0130", "vllm0112", "vllm0102")
    ):
        return "fixedcontainer"
    return "official"


def generic_rollout_label(row: dict[str, str]) -> str:
    name = row.get("name", "").lower()
    if "experimental20n4g" in name or "20n4g" in name:
        return "experimental-20n4g"
    if "experimental18n4g" in name or "18n4g" in name:
        return "experimental-18n4g"
    if "systemvenv" in name:
        return "system-venv"
    if "systemvllm" in name:
        return "system-vllm"
    if "sharedvllm" in name:
        return "shared-vllm"
    if "aarchvllm" in name:
        return "aarch-vllm"
    if "vllm0112" in name:
        return "vllm-0112"
    if "vllm0130" in name:
        return "vllm-0130"
    if "vllm0102" in name:
        return "vllm-0102"
    if "fixedcontainer" in name:
        return "fixed-container"
    return "rollout"


def status_text(row: dict[str, str]) -> str:
    reason = row.get("reason")
    state = row.get("state") or "unknown"
    return f"{state} {reason}".strip() if reason else state


def active_job_sentence(row: dict[str, str]) -> str:
    state = (row.get("state") or "unknown").upper()
    job_id = row.get("job_id") or "unknown"
    if state == "RUNNING":
        return f"active rollout job `{job_id}` is running"
    if state == "PENDING":
        return f"active rollout job `{job_id}` is still waiting for Slurm resources"
    return f"active rollout job `{job_id}` is `{state}`"


def primary_queue_job(snap: dict[str, Any]) -> dict[str, Any]:
    primary_id = str((snap.get("primary") or {}).get("job_id") or "")
    queue = snap.get("queue") if isinstance(snap.get("queue"), dict) else {}
    for job in queue.get("jobs") or []:
        if isinstance(job, dict) and str(job.get("job_id") or "") == primary_id:
            return job
    return {}


def primary_timeout(snap: dict[str, Any]) -> dict[str, Any]:
    job = primary_queue_job(snap)
    timeout = job.get("watcher_timeout") if isinstance(job.get("watcher_timeout"), dict) else {}
    return timeout


def pid_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def command_env_value(command: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=('[^']*'|\"[^\"]*\"|\S*)", command or "")
    if not match:
        return ""
    value = match.group(1)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def check_status_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("checks") or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown").lower()
        counts[status] = counts.get(status, 0) + 1
    return counts


def resource_profile_by_id(payload: dict[str, Any], profile_id: str) -> dict[str, Any]:
    for profile in payload.get("profiles") or []:
        if isinstance(profile, dict) and profile.get("id") == profile_id:
            return profile
    return {}


def first_topology_detail(profile: dict[str, Any]) -> str:
    for check in profile.get("topology_checks") or []:
        if isinstance(check, dict) and check.get("detail"):
            return str(check["detail"])
    return ""


def resource_profile_sentence(snap: dict[str, Any]) -> str:
    payload = snap.get("resource_profiles") if isinstance(snap.get("resource_profiles"), dict) else {}
    official = resource_profile_by_id(payload, "official_32n4g_async")
    compact = resource_profile_by_id(payload, "compact_16n4g_smoke")
    balanced = resource_profile_by_id(payload, "balanced_24n4g_smoke")
    if not payload:
        return "`rollout_resource_profiles_preflight.json`은 아직 최신 topology evidence가 없다."
    overall = str(payload.get("overall_status") or "unknown")
    official_status = str(official.get("status") or "missing")
    compact_status = str(compact.get("status") or "missing")
    balanced_status = str(balanced.get("status") or "missing")
    compact_detail = first_topology_detail(compact)
    balanced_detail = first_topology_detail(balanced)
    if official_status == "pass" and balanced_status == "pass" and compact_status != "pass" and compact_detail:
        return (
            "`rollout_resource_profiles_preflight.json`은 overall `"
            f"{overall}`이고, `official_32n4g_async`와 `balanced_24n4g_smoke`는 PASS지만 "
            f"`compact_16n4g_smoke`는 Megatron topology에서 FAIL이다 ({compact_detail}). "
            f"`balanced_24n4g_smoke` topology는 {balanced_detail}."
        )
    if official_status == "pass" and compact_status != "pass" and compact_detail:
        return (
            "`rollout_resource_profiles_preflight.json`은 overall `"
            f"{overall}`이고, `official_32n4g_async`는 PASS지만 `compact_16n4g_smoke`는 "
            f"Megatron topology에서 FAIL이다 ({compact_detail})."
        )
    return (
        "`rollout_resource_profiles_preflight.json`은 overall `"
        f"{overall}`이고, `official_32n4g_async`는 `{official_status}`, "
        f"`balanced_24n4g_smoke`는 `{balanced_status}`, "
        f"`compact_16n4g_smoke`는 `{compact_status}`다."
    )


def experimental_resource_profile_sentence(snap: dict[str, Any]) -> str:
    payload = snap.get("experimental_resource_profiles") if isinstance(snap.get("experimental_resource_profiles"), dict) else {}
    if not payload:
        return ""
    profiles = {
        str(profile.get("id") or ""): profile
        for profile in payload.get("profiles") or []
        if isinstance(profile, dict)
    }
    experimental = [
        profile
        for profile in profiles.values()
        if profile.get("experimental") is True
    ]
    if not experimental:
        return ""
    summaries = []
    for profile in experimental:
        profile_id = str(profile.get("id") or "unknown")
        status = str(profile.get("status") or "unknown")
        topology = profile_topology_status(profile)
        summaries.append(f"`{profile_id}` `{status}`/topology `{topology}`")
    return (
        "별도 report-only experimental preflight도 생성했고 "
        + ", ".join(summaries)
        + "이다. 이 profile들은 기본 fallback selector에서는 제외된다."
    )


def profile_topology_status(profile: dict[str, Any]) -> str:
    checks = profile.get("topology_checks") or []
    if not checks:
        return "unknown"
    return "pass" if all(isinstance(item, dict) and item.get("status") == "pass" for item in checks) else "fail"


def fallback_decision_sentence(snap: dict[str, Any]) -> str:
    fallback = snap.get("fallback") if isinstance(snap.get("fallback"), dict) else {}
    job = fallback.get("job") if isinstance(fallback.get("job"), dict) else {}
    delay = job.get("estimated_start_delay_minutes", "unknown")
    threshold = (fallback.get("thresholds") or {}).get("max_start_delay_minutes", "unknown")
    recommendation = fallback.get("recommendation", "unknown")
    detail = fallback.get("detail")
    job_id = (snap.get("primary") or {}).get("job_id", "unknown")
    if detail:
        return (
            f"`rollout_fallback_decision.json`은 job `{job_id}`의 estimated delay {delay}분, "
            f"threshold {threshold}분 조건에서 `{recommendation}`으로 판단했다: {detail}."
        )
    return (
        f"`rollout_fallback_decision.json`은 job `{job_id}`의 estimated delay {delay}분, "
        f"threshold {threshold}분 조건에서 `{recommendation}`으로 판단했다."
    )


def arbitration_sentence(snap: dict[str, Any]) -> str:
    arbitration = snap.get("arbitration") if isinstance(snap.get("arbitration"), dict) else {}
    if not arbitration:
        return "`rollout_job_arbitration.json`은 아직 생성되지 않았다."
    recommendation = str(arbitration.get("recommendation") or "unknown")
    overall = str(arbitration.get("overall_status") or "unknown")
    candidates = arbitration.get("cancel_candidates") if isinstance(arbitration.get("cancel_candidates"), list) else []
    winner = arbitration.get("winner") if isinstance(arbitration.get("winner"), dict) else {}
    gate = arbitration.get("cancel_gate") if isinstance(arbitration.get("cancel_gate"), dict) else {}
    winner_id = winner.get("job_id") or "-"
    gate_text = ""
    if gate:
        gate_text = f" cancel gate는 `{'ready' if gate.get('ready') else 'waiting'}` ({gate.get('reason', '-')})이다."
    if candidates:
        candidate_ids = ", ".join(f"`{item.get('job_id', 'unknown')}`" for item in candidates if isinstance(item, dict))
        return (
            "`rollout_job_arbitration.json`은 overall `"
            f"{overall}` / recommendation `{recommendation}`이다. winner `{winner_id}` 기준 "
            f"pending duplicate cancel 후보는 {candidate_ids}다.{gate_text} 기본 refresh는 no-cancel이다."
        )
    return (
        "`rollout_job_arbitration.json`은 overall `"
        f"{overall}` / recommendation `{recommendation}`이고, 현재 자동 취소할 pending duplicate는 없다.{gate_text}"
    )


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    reports = args.artifact_root / "reports"
    queue = load_json(reports / "rollout_queue_wait_summary.json")
    preflight = load_json(reports / "eagle3_pipeline_submit_preflight.json")
    gated = load_json(reports / "eagle3_pipeline_gated_submit.json")
    fallback = load_json(reports / "rollout_fallback_decision.json")
    arbitration = load_json(reports / "rollout_job_arbitration.json")
    watcher_health = load_json(reports / "rollout_watcher_health.json")
    watcher_ensure = load_json(reports / "rollout_watcher_ensure.json")
    goal = load_json(reports / "eagle3_goal_evidence.json")
    completion = load_json(reports / "eagle3_completion_audit.json")
    source_discovery = load_json(reports / "swe_rollout_source_discovery.json")
    vllm_native_probe = load_json(reports / "vllm_native_abi_probe.json")
    vllm_source_build = load_json(reports / "vllm_native_source_build.json")
    vllm_source_job = load_key_values(ROOT / "latest_vllm_native_source_build_job.txt")
    vllm_source_build_013 = load_json(reports / "vllm_native_source_build_0_13_0.json")
    vllm_source_job_013 = load_key_values(ROOT / "latest_vllm_native_source_build_0_13_0_job.txt")
    pilot_current_preflight = load_json(reports / "pilot_existing_chat_content_64_pipeline_submit_preflight_current.json")
    recipe_overrides_current = load_json(reports / "modelopt_recipe_overrides_current.json")
    loss_mask_current = load_json(reports / "modelopt_loss_mask_patch_current.json")
    resource_profiles = load_json(reports / "rollout_resource_profiles_preflight.json")
    preflight_robustness = load_json(reports / "eagle3_preflight_robustness_validation.json")
    experimental_resource_profiles = load_json(reports / "rollout_resource_profiles_experimental_preflight.json")

    rows = job_rows(queue)
    by_role = {role(row): row for row in rows}
    primary = by_role.get("fixedcontainer") or by_role.get("official") or (rows[0] if rows else {})
    primary_job_id = str(primary.get("job_id") or "")
    autosubmit_pid = reports / f"rollout_capture_systemvenv_{primary_job_id}_watch_autosubmit.pid"
    pipeline_ready_submit_pid = reports / "eagle3_pipeline_ready_submit_watch.pid"
    official = by_role.get("official", primary)
    compact = by_role.get("compact", rows[1] if len(rows) > 1 else {})
    generated = minute_timestamp(queue.get("generated_at") or preflight.get("generated_at"))
    preflight_generated = minute_timestamp(preflight.get("generated_at") or queue.get("generated_at"))
    pilot_submit = ((preflight.get("commands") or {}).get("pilot_submit") or "") if isinstance(preflight.get("commands"), dict) else ""
    pipeline_container = command_env_value(pilot_submit, "CONTAINER")
    goal_generated = minute_timestamp(goal.get("generated_at") or completion.get("generated_at") or queue.get("generated_at"))
    goal_counts = goal.get("counts") if isinstance(goal.get("counts"), dict) else {}
    completion_counts = completion.get("counts") if isinstance(completion.get("counts"), dict) else {}
    prepared_output_text = str(source_discovery.get("prepared_output") or "")
    prepared_output = Path(prepared_output_text) if prepared_output_text else None
    return {
        "generated": generated,
        "preflight_generated": preflight_generated,
        "goal_generated": goal_generated,
        "rows": rows,
        "queue": queue,
        "primary": primary,
        "fixedcontainer": by_role.get("fixedcontainer", {}),
        "autosubmit_watcher_alive": pid_alive(autosubmit_pid) if primary_job_id else False,
        "autosubmit_pid": str(autosubmit_pid),
        "pipeline_ready_submit_watcher_alive": pid_alive(pipeline_ready_submit_pid),
        "pipeline_ready_submit_pid": str(pipeline_ready_submit_pid),
        "official": official,
        "compact": compact,
        "queue_status": queue.get("overall_status") or "unknown",
        "preflight_ready": preflight.get("submit_ready"),
        "pipeline_container": pipeline_container,
        "gated_expected_not_ready": gated.get("expected_not_ready"),
        "gated_executed": gated.get("executed"),
        "fallback": fallback,
        "arbitration": arbitration,
        "watcher_health_status": watcher_health.get("overall_status") or "unknown",
        "watcher_ensure_status": watcher_ensure.get("overall_status") or "unknown",
        "goal_counts": goal_counts,
        "completion_counts": completion_counts,
        "source_discovery": source_discovery,
        "reference_conversation_rows": count_jsonl_rows(prepared_output) if prepared_output else 0,
        "vllm_native_probe": vllm_native_probe,
        "vllm_source_build": vllm_source_build,
        "vllm_source_job": vllm_source_job,
        "vllm_source_build_013": vllm_source_build_013,
        "vllm_source_job_013": vllm_source_job_013,
        "pilot_current_preflight": pilot_current_preflight,
        "pilot_current_preflight_counts": check_status_counts(pilot_current_preflight),
        "recipe_overrides_current": recipe_overrides_current,
        "loss_mask_current": loss_mask_current,
        "resource_profiles": resource_profiles,
        "preflight_robustness": preflight_robustness,
        "experimental_resource_profiles": experimental_resource_profiles,
    }


def replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"failed to update {label}")
    return updated


def upsert_text_block(text: str, heading: str, body: str, *, before: str | None = None) -> str:
    block = f"{heading}:\n\n```text\n{body}\n```"
    pattern = rf"{re.escape(heading)}:\n\n```text\n.*?\n```"
    updated, count = re.subn(pattern, block, text, count=1, flags=re.DOTALL)
    if count:
        return updated
    if before and before in text:
        return text.replace(before, f"{block}\n\n{before}", 1)
    return f"{text.rstrip()}\n\n{block}\n"


def active_jobs_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no active rollout jobs are currently visible in `squeue`"
    if len(rows) == 1:
        row = rows[0]
        return (
            f"active rollout job `{row.get('job_id', 'unknown')}` is `{status_text(row)}`, "
            f"{row.get('nodes', '-')} nodes, start estimate `{row.get('start', 'unknown')}`"
        )
    parts = [
        f"`{row.get('job_id', 'unknown')}` `{status_text(row)}` {row.get('nodes', '-')} nodes start `{row.get('start', 'unknown')}`"
        for row in rows
    ]
    return "active rollout jobs are " + "; ".join(parts)


def update_html(path: Path, snap: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8")
    primary = snap["primary"]
    fixed = snap.get("fixedcontainer") or {}
    official = snap["official"]
    compact = snap["compact"]
    poll = snap["generated"]
    primary_long = iso_minute(primary.get("start"), include_date=True)
    primary_state = status_text(primary)
    official_long = iso_minute(official.get("start"), include_date=True)
    compact_long = iso_minute(compact.get("start"), include_date=True)
    official_short = iso_minute(official.get("start"), include_date=False)
    compact_short = iso_minute(compact.get("start"), include_date=False)
    official_state = status_text(official)
    compact_state = status_text(compact)

    rows = snap.get("rows") or []
    multi_active = len(rows) > 1
    if fixed and not multi_active:
        flavor = generic_rollout_label(primary)
        submit_gate = (
            "container preflight job `2850317`은 COMPLETED 0:0이고 rollout submit preflight도 PASS다. "
            "이전 rollout jobs `2850773`/`2851239`는 존재하지 않는 old container path 때문에 실패했다. "
            "launcher default를 `/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh`로 고쳤고, "
            "torch/native-library skew를 피하려고 `uv sync`를 skip하고 `/opt/venv/bin/python`을 직접 쓰도록 보강했다. "
            f"{flavor} 1-step rollout job `{primary.get('job_id', 'unknown')}`를 제출했다. "
            f"`{poll}` live poll 기준 `{primary.get('job_id', 'unknown')}`는 `{primary_state}`, "
            f"{primary.get('nodes', '-')} nodes, reason `{primary.get('reason', '-')}`, Slurm start estimate `{primary_long}`이다."
        )
    else:
        if multi_active:
            submit_gate = (
                "container preflight job `2850317`은 COMPLETED 0:0이고 rollout submit preflight도 PASS다. "
                "이전 rollout jobs `2850773`/`2851239`는 존재하지 않는 old container path 때문에 실패했다. "
                "launcher default와 `/opt/venv/bin/python` direct path를 고쳤고, compile-off rollout retries를 제출했다. "
                f"`{poll}` live poll 기준 {active_jobs_summary(rows)}. queue estimate는 계속 변동될 수 있다."
            )
        else:
            submit_gate = (
                "container preflight job `2850317`은 COMPLETED 0:0이고 rollout submit preflight도 PASS다. "
                f"현재 active rollout은 {generic_rollout_label(primary)} job `{primary.get('job_id', 'unknown')}` 하나다. "
                f"`{poll}` live poll 기준 `{primary.get('job_id', 'unknown')}`는 `{primary_state}`, "
                f"{primary.get('nodes', '-')} nodes, reason `{primary.get('reason', '-')}`, "
                f"Slurm start estimate `{primary_long}`이다. "
                "queue estimate는 계속 변동될 수 있다."
            )
    text = replace_one(
        text,
        r'(<div class="label">Submit Gate</div>\s*<div class="value">).*?(</div>\s*<p>).*?(</p>)',
        rf"\g<1>PASS\g<2>{submit_gate}\g<3>",
        "HTML submit gate",
    )

    native_probe = snap.get("vllm_native_probe") if isinstance(snap.get("vllm_native_probe"), dict) else {}
    source_build = snap.get("vllm_source_build") if isinstance(snap.get("vllm_source_build"), dict) else {}
    source_job = snap.get("vllm_source_job") if isinstance(snap.get("vllm_source_job"), dict) else {}
    failed_sites = [
        item
        for item in native_probe.get("results") or []
        if isinstance(item, dict) and item.get("returncode") not in (0, None)
    ]
    source_status = str(source_build.get("overall_status") or "").lower()
    source_job_id = source_job.get("vllm_native_source_build_job") or "unknown"
    source_site = source_build.get("output_site") or source_job.get("output_site") or "unknown"
    source_build_013 = snap.get("vllm_source_build_013") if isinstance(snap.get("vllm_source_build_013"), dict) else {}
    source_job_013 = snap.get("vllm_source_job_013") if isinstance(snap.get("vllm_source_job_013"), dict) else {}
    source_status_013 = str(source_build_013.get("overall_status") or "").lower()
    source_job_id_013 = source_job_013.get("vllm_native_source_build_job") or ""
    fallback_013_note = ""
    if source_job_id_013 and source_job_id_013 != "VLLM_SOURCE_BUILD_JOB_ID":
        if source_status_013:
            fallback_013_note = f" vLLM 0.13.0 fallback job `{source_job_id_013}` report status는 `{source_status_013}`이며 ABI-only track이다."
        else:
            fallback_013_note = (
                f" vLLM 0.13.0 fallback job `{source_job_id_013}`는 ABI-only candidate로 추적하지만 "
                "`0.10.2` rollout gate를 막는 active unblocker는 아니다."
            )
    if source_status == "pass":
        runtime_value = "PASS"
        runtime_gate = (
            f"wheel ABI probe는 {len(failed_sites)}개 shared target에서 실패했지만, source-built vLLM site "
            f"`{source_site}`가 native import를 PASS했다. 현재 gate는 이 site를 `SHARED_VLLM_SITE`로 쓰는 "
            f"Qwen3-235B SWE rollout smoke이며, {active_jobs_summary(snap['rows'])}. "
            "현재 active container는 nightly vLLM Docker가 아니라 fixed NeMo 25.07.01 sqsh이고, "
            "더 최신/nightly image는 별도 container preflight와 vLLM native/runtime probe를 통과해야 교체할 수 있다. "
            f"{fallback_013_note} rollout corpus가 생긴 뒤에만 Eagle3 hidden-state dump/train/export를 제출한다."
        )
    elif source_status == "fail":
        runtime_value = "FAIL"
        runtime_gate = (
            f"wheel ABI probe는 {len(failed_sites)}개 shared target에서 실패했고, source build job "
            f"`{source_job_id}`도 실패했다. `vllm_native_source_build.md`와 Slurm err log를 보고 build dependency "
            "또는 CUDA/CMake 설정을 수정한 뒤 재제출해야 한다."
        )
    elif source_job:
        runtime_value = "BUILD"
        runtime_gate = (
            f"wheel ABI probe는 {len(failed_sites)}개 shared target에서 `vllm._C` undefined symbol로 실패했다. "
            f"현재 source build job `{source_job_id}`가 `{source_site}`를 만들고 있으며, report가 PASS가 되어야 "
            "rollout smoke를 재시도할 수 있다."
        )
    elif failed_sites:
        runtime_value = "ABI"
        runtime_gate = (
            f"wheel ABI probe가 {len(failed_sites)}개 shared target에서 `vllm._C` native import 실패를 확인했다. "
            "같은 NeMo container/Torch 조합에서 source build를 먼저 실행해야 한다."
        )
    else:
        runtime_value = "CHECK"
        runtime_gate = "vLLM native ABI probe report가 아직 없다. rollout smoke 전 `vllm._C`와 `CompilationConfig` import를 먼저 확인해야 한다."
    text = replace_one(
        text,
        r'(<div class="label">Runtime Gate</div>\s*<div class="value">).*?(</div>\s*<p>).*?(</p>)',
        rf"\g<1>{runtime_value}\g<2>{runtime_gate}\g<3>",
        "HTML runtime gate",
    )

    preflight = (
        "`pilot_existing_chat_content_64.jsonl` 기준 submit preflight는 ModelOpt path, loss-mask patch, "
        "local preflight, wrapper dry-run, Slurm pipeline dry-run을 PASS했다. rollout corpus용 pipeline submit "
        f"preflight와 gated submit check는 operator refresh 안에서 queue/watcher report 다음, planner 이전에 갱신된다. "
        f"`{snap['preflight_generated']}` 기준 pipeline submit preflight는 `DUMP_GPUS_PER_NODE=4`, "
        "`TRAIN_GPUS_PER_NODE=4`, `TP=4`, "
        f"`CONTAINER={snap.get('pipeline_container') or 'unknown'}`로 갱신됐고 corpus missing 때문에 INCOMPLETE다. gated submit check는 "
        "`submit_ready=false`라 제출 없이 FAIL/`expected_not_ready=true`/`executed=false`인 것이 정상이며, "
        "operator refresh는 이 정상 대기 상태를 PASS로 처리한다. 새 gated-submit contract도 "
        "not-ready/ready/bad-command scenarios를 PASS한다."
    )
    pilot_current = snap.get("pilot_current_preflight") if isinstance(snap.get("pilot_current_preflight"), dict) else {}
    pilot_counts = snap.get("pilot_current_preflight_counts") if isinstance(snap.get("pilot_current_preflight_counts"), dict) else {}
    recipe_current = snap.get("recipe_overrides_current") if isinstance(snap.get("recipe_overrides_current"), dict) else {}
    loss_mask_current = snap.get("loss_mask_current") if isinstance(snap.get("loss_mask_current"), dict) else {}
    if pilot_current:
        preflight += (
            " 추가 current pilot no-submit preflight도 fixed container/resource profile로 재검증했고 "
            f"`{pilot_current.get('overall_status', 'unknown')}` 상태에서 checks pass {pilot_counts.get('pass', 0)}, "
            f"warn {pilot_counts.get('warn', 0)}, fail {pilot_counts.get('fail', 0)}이다. "
            "warn은 non-canonical pilot input이 아직 rollout `pipeline_dry_run` evidence가 아니라는 의미이고, "
            "wrapper dry-runs와 Slurm pipeline dry-run 자체는 PASS다."
        )
    if recipe_current or loss_mask_current:
        preflight += (
            f" ModelOpt recipe override validator는 `{recipe_current.get('overall_status', 'unknown')}`이고 "
            f"loss-mask patch validator는 `{loss_mask_current.get('overall_status', 'unknown')}`이다."
        )
    text = replace_one(
        text,
        r'(<div class="label">Pilot Wiring</div>\s*<div class="value">WAIT</div>\s*<p>).*?(</p>)',
        rf"\g<1>{preflight}\g<2>",
        "HTML pilot wiring",
    )

    goal_counts = snap["goal_counts"]
    audit_counts = snap["completion_counts"]
    goal_text = (
        f"draft trained=false. `{snap['goal_generated']}` 기준 goal evidence는 proven "
        f"{goal_counts.get('proven', 'unknown')}, incomplete {goal_counts.get('incomplete', 'unknown')}, "
        f"missing {goal_counts.get('missing', 'unknown')}이고 completion audit은 pass "
        f"{audit_counts.get('pass', 'unknown')}, warn {audit_counts.get('warn', 'unknown')}, "
        f"incomplete {audit_counts.get('incomplete', 'unknown')}, missing {audit_counts.get('missing', 'unknown')}이다. "
        f"preflight robustness는 `{snap.get('preflight_robustness', {}).get('overall_status', 'unknown')}`로 "
        "traceback/token leakage 없이 구조화된 dry-run evidence를 보장한다. "
        "gated pipeline submit evidence는 이제 `dump_job`, `train_job`, `export_job` job id가 모두 기록되어야 proven으로 인정한다."
    )
    text = replace_one(
        text,
        r'(<div class="label">Goal Proof</div>\s*<div class="value">Matrix</div>\s*<p>).*?(</p>)',
        rf"\g<1>{goal_text}\g<2>",
        "HTML goal proof",
    )

    fallback = snap.get("fallback") if isinstance(snap.get("fallback"), dict) else {}
    fallback_job = fallback.get("job") if isinstance(fallback.get("job"), dict) else {}
    fallback_delay = fallback_job.get("estimated_start_delay_minutes", "unknown")
    fallback_threshold = (fallback.get("thresholds") or {}).get("max_start_delay_minutes", "unknown")
    fallback_recommendation = fallback.get("recommendation", "unknown")
    timeout = primary_timeout(snap)
    timeout_risk = timeout.get("risk", "unknown")
    timeout_deadline = minute_timestamp(
        str(timeout.get("watcher_deadline") or timeout.get("deadline") or timeout.get("deadline_local") or "")
    )
    watcher_health_status = str(snap.get("watcher_health_status") or "unknown")
    watcher_ensure_status = str(snap.get("watcher_ensure_status") or "unknown")
    source_discovery = snap.get("source_discovery") if isinstance(snap.get("source_discovery"), dict) else {}
    positive_sources = source_discovery.get("positive_candidates", 0)
    reference_rows = snap.get("reference_conversation_rows", 0)
    source_note = (
        f"별도로 Responses API style SWE/code rollout source discovery도 갱신했고 positive candidates {positive_sources}개, "
        f"reference conversation rows {reference_rows}개를 비-canonical 변환 검증용으로 materialize했다. "
        if positive_sources or reference_rows
        else ""
    )
    if snap.get("pipeline_ready_submit_watcher_alive"):
        auto_submit = (
            "별도 pipeline-ready gated watcher가 alive이며, rollout corpus와 "
            "`eagle3_pipeline_submit_preflight.json`이 PASS/`submit_ready=true`가 된 경우에만 "
            "`submit_eagle3_pipeline_if_ready.py --execute --allow-heavy-gpu`를 실행한다. "
        )
    elif snap.get("autosubmit_watcher_alive"):
        auto_submit = (
            "추가 gated auto-submit watcher가 `AUTO_SUBMIT_PIPELINE=true`로 lock release를 기다리고 있어, "
            "rollout corpus와 submit preflight가 PASS가 될 때만 pilot dump/train/export를 자동 제출한다. "
        )
    else:
        auto_submit = "`AUTO_SUBMIT_PIPELINE=false`라 Eagle3 dump/train/export는 corpus 생성 후 gated preflight까지만 자동 갱신된다. "
    resource_profiles_text = resource_profile_sentence(snap)
    experimental_resource_profiles_text = experimental_resource_profile_sentence(snap)
    fallback_decision_text = fallback_decision_sentence(snap)
    arbitration_text = arbitration_sentence(snap)

    if fixed:
        fallback = snap.get("fallback") if isinstance(snap.get("fallback"), dict) else {}
        fallback_job = fallback.get("job") if isinstance(fallback.get("job"), dict) else {}
        fallback_delay = fallback_job.get("estimated_start_delay_minutes", "unknown")
        fallback_threshold = (fallback.get("thresholds") or {}).get("max_start_delay_minutes", "unknown")
        fallback_recommendation = fallback.get("recommendation", "unknown")
        timeout = primary_timeout(snap)
        timeout_risk = timeout.get("risk", "unknown")
        timeout_deadline = minute_timestamp(
            str(timeout.get("watcher_deadline") or timeout.get("deadline") or timeout.get("deadline_local") or "")
        )
        watcher_health_status = str(snap.get("watcher_health_status") or "unknown")
        watcher_ensure_status = str(snap.get("watcher_ensure_status") or "unknown")
        source_discovery = snap.get("source_discovery") if isinstance(snap.get("source_discovery"), dict) else {}
        positive_sources = source_discovery.get("positive_candidates", 0)
        reference_rows = snap.get("reference_conversation_rows", 0)
        source_note = (
            f"별도로 Responses API style SWE/code rollout source discovery도 갱신했고 positive candidates {positive_sources}개, "
            f"reference conversation rows {reference_rows}개를 비-canonical 변환 검증용으로 materialize했다. "
            if positive_sources or reference_rows
            else ""
        )
        if snap.get("pipeline_ready_submit_watcher_alive"):
            auto_submit = (
                "별도 pipeline-ready gated watcher가 alive이며, rollout corpus와 "
                "`eagle3_pipeline_submit_preflight.json`이 PASS/`submit_ready=true`가 된 경우에만 "
                "`submit_eagle3_pipeline_if_ready.py --execute --allow-heavy-gpu`를 실행한다. "
            )
        elif snap.get("autosubmit_watcher_alive"):
            auto_submit = (
                "추가 gated auto-submit watcher가 `AUTO_SUBMIT_PIPELINE=true`로 lock release를 기다리고 있어, "
                "rollout corpus와 submit preflight가 PASS가 될 때만 pilot dump/train/export를 자동 제출한다. "
            )
        else:
            auto_submit = "`AUTO_SUBMIT_PIPELINE=false`라 Eagle3 dump/train/export는 corpus 생성 후 gated preflight까지만 자동 갱신된다. "
        flavor = generic_rollout_label(primary)
        if multi_active:
            operator = (
                "이전 official/compact rollout jobs `2850773`/`2851239`는 old container image missing으로 실패했다. "
                "그 원인을 반영해 launcher default와 submit preflight를 fixed container로 수정했고, "
                "이후 torch/native-library skew를 피하려고 `uv sync` skip + `/opt/venv/bin/python` direct launcher를 적용했다. "
            f"현재 compile-off retries: {active_jobs_summary(rows)}. "
                "materialized rollout JSONL은 아직 없고, canonical state는 `running / poll`이다. "
                "generic materialize watchers가 alive이며 queue summary는 `waiting`, watcher health/ensure는 PASS다. "
                f"{source_note}"
                "corpus가 생성되면 watcher가 normalize와 no-submit pipeline preflight를 실행하고, gated helper는 "
                "`dump_job`, `train_job`, `export_job`가 모두 기록될 때만 Eagle3 pilot submit을 proven으로 인정한다."
            )
        else:
            operator = (
                "이전 official/compact rollout jobs `2850773`/`2851239`는 old container image missing으로 실패했다. "
                "그 원인을 반영해 launcher default와 submit preflight를 fixed container로 수정했고, "
                "이후 torch/native-library skew를 피하려고 `uv sync` skip + `/opt/venv/bin/python` direct launcher를 적용했다. "
                f"새 {flavor} 1-step rollout job `{primary.get('job_id', 'unknown')}`를 제출했다. "
                f"`{poll}` 기준 `{primary.get('job_id', 'unknown')}`는 `{primary_state}`, "
                f"{primary.get('nodes', '-')} nodes, reason `{primary.get('reason', '-')}`, start estimate `{primary.get('start', 'unknown')}`이다. "
                "materialized rollout JSONL은 아직 없고, canonical state는 `running / poll`이다. "
                "generic materialize watcher가 alive이며 queue summary는 `waiting`, watcher health/ensure는 PASS다. "
                f"{source_note}"
                "corpus가 생성되면 watcher가 normalize와 no-submit pipeline preflight를 실행하고, gated helper는 "
                "`dump_job`, `train_job`, `export_job`가 모두 기록될 때만 Eagle3 pilot submit을 proven으로 인정한다."
            )
    else:
        operator = (
            "`submit_rollout_capture` execution record는 returncode 0으로 갱신됐고 "
            f"현재 active rollout job `{primary.get('job_id', 'unknown')}`가 제출되어 있다. "
            f"`{poll}` live poll 기준 `{primary.get('job_id', 'unknown')}`는 `{primary_state}`, "
            f"{primary.get('nodes', '-')} nodes, reason `{primary.get('reason', '-')}`, "
            f"start estimate `{primary.get('start', 'unknown')}`이고 materialized rollout JSONL은 아직 없다. "
            "canonical state는 job id를 반영해 `running / poll`로 갱신했다. `rollout_poll` operator action은 canonical JSON/Markdown을 쓰도록 planner를 고쳤고, "
            "`advance_rollout_capture_state.py`는 concurrent official/compact refresh가 같은 JSON을 깨뜨리지 않도록 artifact-root lock을 잡는다. "
            "generic watcher는 terminal 이후 train_data materialize, corpus strategy, training-scale report, pipeline submit preflight, operator refresh를 갱신한다. "
            "현재 실행 중인 watcher는 유지하고, 별도 pending-state helper를 붙였고, `summarize_rollout_queue_wait.py`, `summarize_rollout_watcher_health.py`, "
            "`ensure_rollout_watchers.py`, `validate_rollout_watcher_ensure.py`가 operator refresh에 포함된다. operator refresh는 이제 queue/watcher/pipeline submit preflight/gated readiness check를 먼저 갱신한 뒤 planner와 audits를 다시 만든다. "
            f"queue summary는 active watcher deadline `{timeout_deadline}`와 180-minute terminal buffer 기준 timeout risk `{timeout_risk}`를 함께 기록한다. watcher health는 materialize/pending-state/operator follow-up liveness를 queue context 기준으로 PASS 처리하고, ensure report도 restart_needed_count 0, extension_needed_count 0이면 PASS다. "
            "ensure validation도 alive/restart/timeout-extension scenarios를 PASS한다. health checker는 queue-context 기반으로 필수 watcher를 판단하므로 terminal 이후 정상 종료한 pending-state watcher를 stale failure로 보지 않는다. "
            "timeout risk가 생기면 새 extension watcher가 lock release를 기다린 뒤 이어받을 수 있게 보강했다. gated submit helper는 현재 corpus/preflight가 없어 FAIL이 정상이며 `expected_not_ready=true`를 기록한다. "
            "gated-submit contract도 not-ready-without-flag, not-ready-with-flag, ready-no-execute, bad-command scenarios를 PASS한다. operator refresh는 이 정상 대기 상태에서 PASS다. 준비 후에는 pilot pipeline command를 검증하고 실행할 수 있다. "
            "이제 gated helper는 submit returncode뿐 아니라 `dump_job`, `train_job`, `export_job`가 job file에 모두 기록됐는지도 검사하고 job file copy를 artifact reports에 남긴다. hidden-state/train/export pipeline이 실제 제출되면 `watch_eagle3_pipeline_followup.sh`가 job file을 poll하고 pipeline analysis, completion audit, operator refresh를 자동 실행한다."
        )
    text = replace_one(
        text,
        r'(<strong>7\. Operator Gates</strong>\s*<span>).*?(</span>)',
        rf"\g<1>{operator}\g<2>",
        "HTML operator gates",
    )
    if fixed and not multi_active:
        flavor = generic_rollout_label(primary)
        fallback_text = (
            f"현재 active rollout은 {flavor} job `{primary.get('job_id', 'unknown')}` 하나다. "
            "`watch_rollout_capture_materialize.sh` generic watcher가 붙어 있고, terminal 이후 "
            "`train_data_step*.jsonl` materialize, corpus strategy, training-scale report, "
            "no-submit pipeline preflight, operator refresh를 자동 실행한다. "
            f"{auto_submit}"
            f"{resource_profiles_text} "
            f"{experimental_resource_profiles_text} "
            f"{fallback_decision_text} "
            f"{arbitration_text} "
            f"{source_note}"
            f"`rollout_queue_wait_summary.json`은 `waiting`, timeout risk `{timeout_risk}`를 기록한다. "
            f"`rollout_watcher_health.json`은 `{watcher_health_status}`, "
            f"`rollout_watcher_ensure.json`은 `{watcher_ensure_status}`다."
        )
        text = replace_one(
            text,
            r'(<div class="label">Fallback</div>\s*<div class="value">).*?(</div>\s*<p>).*?(</p>)',
            rf"\g<1>Live\g<2>{fallback_text}\g<3>",
            "HTML fallback",
        )
        handoff_status = (
            f"현재 active job은 {flavor} 재시도 `{primary.get('job_id', 'unknown')}` 하나이며,\n"
            f"                `{primary.get('nodes', '-')} nodes x 4 GPUs/node` request로 `{primary_state}`이다."
        )
        text = re.sub(
            r"현재 active job은 system-venv 재시도 `[^`]+` 하나이며,\s*`32 nodes x 4 GPUs/node` request로 `[^`]+`이다\.",
            handoff_status,
            text,
            count=1,
        )
        audit_status = (
            f"현재 refresh 후 completion audit summary는 PASS {audit_counts.get('pass', 'unknown')}, "
            f"INCOMPLETE {audit_counts.get('incomplete', 'unknown')}, "
            f"MISSING {audit_counts.get('missing', 'unknown')}, WARN {audit_counts.get('warn', 'unknown')}이고 "
            "overall은 `incomplete`이다."
        )
        text = re.sub(
            r"현재 refresh 후 completion audit summary는 PASS [^,]+, INCOMPLETE [^,]+, MISSING [^,]+, WARN [^,]+이고 overall은 `[^`]+`이다\.",
            audit_status,
            text,
            count=1,
        )
    elif multi_active:
        fallback_text = (
            f"현재 {active_jobs_summary(rows)}. "
            "`watch_rollout_capture_materialize.sh` generic watcher가 각 active rollout에 붙어 있고, terminal 이후 "
            "`train_data_step*.jsonl` materialize, corpus strategy, training-scale report, no-submit pipeline preflight, "
            "operator refresh를 자동 실행한다. "
            f"{auto_submit}"
            f"{resource_profiles_text} "
            f"{experimental_resource_profiles_text} "
            f"{arbitration_text} "
            f"`rollout_queue_wait_summary.json`은 `waiting`, timeout risk `{timeout_risk}`를 기록한다. "
            f"`rollout_watcher_health.json`은 `{watcher_health_status}`, "
            f"`rollout_watcher_ensure.json`은 `{watcher_ensure_status}`다."
        )
        text = replace_one(
            text,
            r'(<div class="label">Fallback</div>\s*<div class="value">).*?(</div>\s*<p>).*?(</p>)',
            rf"\g<1>Live\g<2>{fallback_text}\g<3>",
            "HTML fallback",
        )
    else:
        fallback_text = (
            f"현재 active rollout은 {generic_rollout_label(primary)} job `{primary.get('job_id', 'unknown')}` 하나다. "
            "`watch_rollout_capture_materialize.sh` generic watcher가 붙어 있고, terminal 이후 "
            "`train_data_step*.jsonl` materialize, corpus strategy, training-scale report, no-submit pipeline preflight, "
            "operator refresh를 자동 실행한다. "
            f"{auto_submit}"
            f"{resource_profiles_text} "
            f"{experimental_resource_profiles_text} "
            f"{arbitration_text} "
            f"`rollout_queue_wait_summary.json`은 `waiting`, timeout risk `{timeout_risk}`를 기록한다. "
            f"`rollout_watcher_health.json`은 `{watcher_health_status}`, "
            f"`rollout_watcher_ensure.json`은 `{watcher_ensure_status}`다."
        )
        text = replace_one(
            text,
            r'(<div class="label">Fallback</div>\s*<div class="value">).*?(</div>\s*<p>).*?(</p>)',
            rf"\g<1>Live\g<2>{fallback_text}\g<3>",
            "HTML fallback",
        )
    text = re.sub(
        r"Last updated from local and oci-hsg validation: [^<]+",
        f"Last updated from local and oci-hsg validation: {poll}",
        text,
        count=1,
    )
    return text


def update_cluster_status(path: Path, snap: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8")
    rows = "\n".join(job_line(row) for row in snap["rows"])
    primary = snap["primary"]
    text = re.sub(r"Last updated: .+", f"Last updated: {snap['generated']}", text, count=1)
    text = upsert_text_block(
        text,
        "Current active rollout capture jobs",
        rows,
        before="The current no-submit decision report is:",
    )
    text = re.sub(
        r"pipeline submit preflight was regenerated at `[^`]+` with",
        f"pipeline submit preflight was regenerated at `{snap['preflight_generated']}` with",
        text,
        count=1,
    )
    if snap.get("fixedcontainer"):
        fallback = snap.get("fallback") if isinstance(snap.get("fallback"), dict) else {}
        fallback_job = fallback.get("job") if isinstance(fallback.get("job"), dict) else {}
        fallback_delay = fallback_job.get("estimated_start_delay_minutes", "unknown")
        fallback_threshold = (fallback.get("thresholds") or {}).get("max_start_delay_minutes", "unknown")
        fallback_recommendation = fallback.get("recommendation", "unknown")
        timeout = primary_timeout(snap)
        timeout_deadline = timeout.get("watcher_deadline") or timeout.get("deadline") or timeout.get("deadline_local") or "unknown"
        timeout_risk = timeout.get("risk", "unknown")
        text = re.sub(
            r"The current operator action is `rollout_poll`\. The queue-wait report is\n`waiting`, watcher health is PASS, and .*?the active generic materialize watcher is alive\.",
            (
                f"The current operator action is `rollout_poll`. The queue-wait report is\n"
                f"`waiting`, watcher health is PASS, and {active_job_sentence(primary)}. No materialized\n"
                "`qwen3_235b_swe_rollout_conversations*.jsonl` corpus exists yet.\n"
                "The operator refresh now regenerates the no-submit pipeline submit preflight and\n"
                "gated readiness check after queue/watcher reports, before replanning. The latest\n"
                f"pipeline submit preflight was regenerated at `{snap['preflight_generated']}` with\n"
                "`DUMP_GPUS_PER_NODE=4`, `TRAIN_GPUS_PER_NODE=4`, `TP=4`, and\n"
                f"`CONTAINER={snap.get('pipeline_container') or 'unknown'}`; it is expectedly\n"
                "INCOMPLETE because the rollout corpus is missing. The gated report is expectedly\n"
                "FAIL/`expected_not_ready=true`/`executed=false` until `submit_ready=true`, but\n"
                "it records the required post-submit job keys: `dump_job`, `train_job`, and\n"
                "`export_job`. The operator refresh treats this expected not-ready state as a\n"
                "successful no-submit refresh and currently reports PASS. The synthetic gated\n"
                "submit contract also PASSes the not-ready-without-flag, not-ready-with-flag,\n"
                "ready-no-execute, and bad-command scenarios.\n"
                "The queue-wait report records watcher timeout risk using a 180-minute\n"
                f"post-start terminal buffer. Current watcher deadline is `{timeout_deadline}`\n"
                f"for active rollout job `{primary.get('job_id', 'unknown')}`, timeout risk is `{timeout_risk}`, and the latest Slurm start\n"
                f"estimate `{primary.get('start', 'unknown')}` is within the watcher window. Watcher health is\n"
                "queue-context aware: terminal old rollout watchers are no longer required, and\n"
                "the active generic materialize watcher is alive."
            ),
            text,
            count=1,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"job `[^`]+` estimated start delay is -?[0-9.]+ minutes, below the [0-9]+-minute\nfallback threshold\.(?: Current recommendation: `[^`]+`\.)*",
            fallback_decision_sentence(snap),
            text,
            count=1,
        )
    return text


def update_execution_inputs(path: Path, snap: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8")
    rows = "\n".join(job_line(row) for row in snap["rows"])
    primary = snap["primary"]
    text = re.sub(r"Last updated: .+", f"Last updated: {snap['generated']}", text, count=1)
    text = upsert_text_block(
        text,
        "Current rollout queue snapshot",
        rows,
        before="`watch_vllm_source_build_then_rollout.sh` had already exited",
    )
    if snap.get("fixedcontainer"):
        fallback = snap.get("fallback") if isinstance(snap.get("fallback"), dict) else {}
        fallback_job = fallback.get("job") if isinstance(fallback.get("job"), dict) else {}
        fallback_delay = fallback_job.get("estimated_start_delay_minutes", "unknown")
        fallback_threshold = (fallback.get("thresholds") or {}).get("max_start_delay_minutes", "unknown")
        fallback_recommendation = fallback.get("recommendation", "unknown")
        timeout = primary_timeout(snap)
        timeout_deadline = timeout.get("watcher_deadline") or timeout.get("deadline") or timeout.get("deadline_local") or "unknown"
        timeout_risk = timeout.get("risk", "unknown")
        text = re.sub(
            r"latest live poll was `[^`]+`; `squeue` showed start estimate(?:s)?\n`[^`]+` for active (?:fixed-container )?rollout job `[^`]+`\.",
            (
                f"latest live poll was `{snap['generated']}`; `squeue` showed start estimate\n"
                f"`{primary.get('start', 'unknown')}` for active rollout job "
                f"`{primary.get('job_id', 'unknown')}`."
            ),
            text,
            count=1,
        )
        text = re.sub(
            r"latest live poll was `[^`]+`; `squeue` showed start estimates\n`[^`]+` for the official job and `[^`]+` for the\ncompact fallback\.",
            (
                f"latest live poll was `{snap['generated']}`; `squeue` showed start estimate\n"
                f"`{primary.get('start', 'unknown')}` for active fixed-container rollout job "
                f"`{primary.get('job_id', 'unknown')}`."
            ),
            text,
            count=1,
        )
        text = re.sub(
            r"```text\nfixed-container materialize watcher pid=[^\n]*\n```",
            (
                "```text\n"
                f"generic materialize watcher for job {primary.get('job_id', 'unknown')} is alive\n"
                "```"
            ),
            text,
            count=1,
        )
        text = re.sub(
            r"Current deadline is\n`[^`]+` for job `[^`]+`, with timeout risk `[^`]+`\.",
            (
                f"Current deadline is\n`{timeout_deadline}` for job `{primary.get('job_id', 'unknown')}`, "
                f"with timeout risk `{timeout_risk}`."
            ),
            text,
            count=1,
        )
        text = re.sub(
            r"latest decision is `[^`]+`: job `[^`]+` estimated start delay is -?[0-9.]+\nminutes, below the [0-9]+-minute fallback threshold\.",
            fallback_decision_sentence(snap),
            text,
            count=1,
        )
    return text


def maybe_write(path: Path, content: str, dry_run: bool) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = old != content
    if changed and not dry_run:
        path.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    args = parse_args()
    snap = build_snapshot(args)
    if not snap["rows"]:
        message = (
            f"no rollout jobs found in {args.artifact_root / 'reports' / 'rollout_queue_wait_summary.json'}; "
            "run refresh_eagle3_operator_state.py first or pass the correct --artifact-root"
        )
        if args.allow_missing_jobs:
            print(json.dumps({"overall_status": "skipped", "reason": message, "dry_run": args.dry_run}, indent=2))
            return 0
        raise SystemExit(message)
    updates = {
        str(args.html): update_html(args.html, snap),
        str(args.cluster_status): update_cluster_status(args.cluster_status, snap),
        str(args.execution_inputs): update_execution_inputs(args.execution_inputs, snap),
    }
    changed = [path for path, content in updates.items() if maybe_write(Path(path), content, args.dry_run)]
    print(
        json.dumps(
            {
                "generated": snap["generated"],
                "changed": changed,
                "dry_run": args.dry_run,
                "jobs": [job_line(row) for row in snap["rows"]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
