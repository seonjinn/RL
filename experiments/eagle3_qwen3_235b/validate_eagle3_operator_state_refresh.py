#!/usr/bin/env python3
"""Validate the no-submit operator-state refresh contract.

This is intentionally separate from validate_eagle3_preflight_robustness.py:
refresh_eagle3_operator_state.py itself runs the preflight-robustness validator,
so putting this check there would recurse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SECRET_SENTINELS = {
    "WANDB_API_KEY": "EAGLE3_SENTINEL_WANDB_SHOULD_NOT_LEAK",
    "HUGGINGFACE_TOKEN": "EAGLE3_SENTINEL_HF_SHOULD_NOT_LEAK",
    "GITHUB_TOKEN": "EAGLE3_SENTINEL_GITHUB_SHOULD_NOT_LEAK",
    "GITLAB_TOKEN": "EAGLE3_SENTINEL_GITLAB_SHOULD_NOT_LEAK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(SECRET_SENTINELS)
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


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid json: {exc}"
    return payload if isinstance(payload, dict) else None, None


def file_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def contains_any(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle and needle in text]


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def report_status(refresh: dict[str, Any] | None, name: str) -> str:
    reports = (refresh or {}).get("reports") if isinstance((refresh or {}).get("reports"), dict) else {}
    report = reports.get(name) if isinstance(reports.get(name), dict) else {}
    return str(report.get("status") or "missing")


def step_names(refresh: dict[str, Any] | None) -> list[str]:
    return [
        str(item.get("name"))
        for item in (refresh or {}).get("steps") or []
        if isinstance(item, dict) and item.get("name")
    ]


def goal_requirement(goal: dict[str, Any] | None, requirement: str) -> dict[str, Any]:
    for item in (goal or {}).get("requirements") or (goal or {}).get("rows") or []:
        if isinstance(item, dict) and item.get("requirement") == requirement:
            return item
    return {}


def validate_refresh_contract(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "operator_state_refresh"
    report_dir = artifact / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    refresh_json = report_dir / "eagle3_operator_state_refresh.json"
    refresh_md = report_dir / "eagle3_operator_state_refresh.md"
    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py",
            "--artifact-root",
            str(artifact),
            "--skip-remote-host-probe",
            "--json-out",
            str(refresh_json),
            "--markdown-out",
            str(refresh_md),
        ]
    )
    refresh, refresh_error = read_json(refresh_json)
    goal, goal_error = read_json(report_dir / "eagle3_goal_evidence.json")
    input_discovery, input_discovery_error = read_json(artifact / "eagle3_input_discovery.json")
    provenance, provenance_error = read_json(report_dir / "eagle3_provenance.json")
    cluster_probe, cluster_probe_error = read_json(report_dir / "cluster_environment_probe.json")
    remote_diag, remote_diag_error = read_json(report_dir / "eagle3_remote_access_diagnostics.json")
    readiness, readiness_error = read_json(report_dir / "eagle3_readiness.json")
    rollout_submit, rollout_submit_error = read_json(report_dir / "rollout_capture_submit_preflight.json")
    loss_mask, loss_mask_error = read_json(report_dir / "modelopt_loss_mask_patch.json")
    recipe, recipe_error = read_json(report_dir / "modelopt_recipe_overrides_current.json")
    training_path, training_path_error = read_json(report_dir / "eagle3_training_path_manifest.json")
    training_path_validation, training_path_validation_error = read_json(
        report_dir / "eagle3_training_path_manifest_validation.json"
    )
    nemo_drift, nemo_drift_error = read_json(report_dir / "nemo_rl_eagle3_drift.json")
    nemo_integration, nemo_integration_error = read_json(report_dir / "nemo_rl_specdec_integration.json")
    gated_contract, gated_contract_error = read_json(report_dir / "eagle3_pipeline_gated_submit_contract.json")
    safe_preflight, safe_preflight_error = read_json(report_dir / "eagle3_operator_safe_actions_preflight.json")
    combined = "\n".join(
        [
            result.stdout,
            file_text(refresh_json),
            file_text(refresh_md),
            file_text(artifact / "eagle3_input_discovery.json"),
            file_text(report_dir / "eagle3_provenance.json"),
            file_text(report_dir / "cluster_environment_probe.json"),
            file_text(report_dir / "eagle3_remote_access_diagnostics.json"),
            file_text(report_dir / "eagle3_readiness.json"),
            file_text(report_dir / "rollout_capture_submit_preflight.json"),
            file_text(report_dir / "hayate_inventory.txt"),
            file_text(report_dir / "eagle3_goal_evidence.json"),
            file_text(report_dir / "modelopt_loss_mask_patch.json"),
            file_text(report_dir / "modelopt_recipe_overrides_current.json"),
            file_text(report_dir / "eagle3_training_path_manifest.json"),
            file_text(report_dir / "eagle3_training_path_manifest_validation.json"),
            file_text(report_dir / "nemo_rl_eagle3_drift.json"),
            file_text(report_dir / "nemo_rl_specdec_integration.json"),
            file_text(report_dir / "eagle3_pipeline_gated_submit_contract.json"),
            file_text(report_dir / "eagle3_operator_safe_actions_preflight.json"),
        ]
    )
    leaked = contains_any(combined, list(SECRET_SENTINELS.values()))
    traces = contains_any(combined, ["Traceback (most recent call last)"])
    expected_steps = {
        "discover_run_inputs",
        "collect_provenance",
        "probe_cluster_environment",
        "diagnose_remote_access",
        "validate_modelopt_loss_mask_patch",
        "check_modelopt_upstream_drift",
        "validate_modelopt_recipe_overrides_current",
        "build_training_path_manifest",
        "validate_training_path_manifest",
        "preflight_rollout_capture_submit",
        "check_nemo_rl_eagle3_drift",
        "validate_nemo_rl_specdec_integration",
        "validate_pipeline_gated_submit_contract",
        "validate_preflight_robustness",
        "preflight_operator_safe_actions",
        "inventory_hayate_artifacts",
        "audit_readiness",
        "audit_goal_evidence",
    }
    observed_steps = set(step_names(refresh))
    missing_steps = sorted(expected_steps - observed_steps)
    loss_mask_req = goal_requirement(goal, "ModelOpt path and Qwen3 loss-mask patch are known")
    recipe_req = goal_requirement(goal, "ModelOpt Eagle3 recipe overrides match the Qwen3-235B Thinking architecture")
    training_path_req = goal_requirement(goal, "Qwen3 Eagle3 training path manifest is defined")
    rl_route_req = goal_requirement(goal, "RL integration route is fixed exported draft first")
    allowed_failed_steps = {
        "audit_readiness",
        "discover_run_inputs",
        "plan_next_actions",
        "validate_resource_profile_application",
        "summarize_rollout_queue_wait",
        "preflight_rollout_resource_profiles",
        "preflight_rollout_capture_submit",
        "preflight_operator_ready_submit",
    }
    failed_steps = {
        str(item.get("name"))
        for item in (refresh or {}).get("steps") or []
        if isinstance(item, dict) and item.get("returncode")
    }
    unexpected_failed_steps = sorted(failed_steps - allowed_failed_steps)
    problems = []
    if result.returncode != 0 or refresh_error:
        problems.append("refresh command did not return zero and write JSON")
    if missing_steps:
        problems.append(f"refresh is missing expected steps: {missing_steps}")
    if unexpected_failed_steps:
        problems.append(f"refresh has unexpected failed steps: {unexpected_failed_steps}")
    if input_discovery_error or not isinstance((input_discovery or {}).get("verifier_candidates"), list):
        problems.append("refresh did not write structured eagle3_input_discovery.json")
    if provenance_error or not (provenance or {}).get("critical_files"):
        problems.append("refresh did not write structured eagle3_provenance.json with critical file hashes")
    if cluster_probe_error or (cluster_probe or {}).get("overall_status") not in {"pass", "fail", "warn"}:
        problems.append("refresh did not write structured cluster_environment_probe.json")
    if remote_diag_error or (remote_diag or {}).get("overall_status") not in {
        "pass",
        "blocked_local_dns",
        "unreachable",
        "missing_probe",
        "missing_hosts",
    }:
        problems.append("refresh did not write structured eagle3_remote_access_diagnostics.json")
    interpretation = (
        (remote_diag or {}).get("gate_interpretation")
        if isinstance((remote_diag or {}).get("gate_interpretation"), dict)
        else {}
    )
    if interpretation.get("remote_path_absence_proven") is not False:
        problems.append("remote access diagnostics did not preserve remote_path_absence_proven=false")
    if readiness_error or (readiness or {}).get("overall_status") not in {"pass", "fail", "warn"}:
        problems.append("refresh did not write structured eagle3_readiness.json")
    if rollout_submit_error or (rollout_submit or {}).get("overall_status") not in {"pass", "warn", "fail"}:
        problems.append("refresh did not write structured rollout_capture_submit_preflight.json")
    if not file_text(report_dir / "hayate_inventory.txt").strip():
        problems.append("refresh did not write Hayate inventory text")
    if loss_mask_error or (loss_mask or {}).get("overall_status") != "pass":
        problems.append("refresh did not write PASS modelopt_loss_mask_patch.json")
    if recipe_error or (recipe or {}).get("overall_status") != "pass":
        problems.append("refresh did not write PASS modelopt_recipe_overrides_current.json")
    if training_path_error or (training_path or {}).get("overall_status") not in {"defined", "pass"}:
        problems.append("refresh did not write defined eagle3_training_path_manifest.json")
    if training_path_validation_error or (training_path_validation or {}).get("overall_status") != "pass":
        problems.append("refresh did not write PASS eagle3_training_path_manifest_validation.json")
    if not (training_path or {}).get("path_defined"):
        problems.append("training path manifest did not set path_defined=true")
    training_path_gates = {
        str(item.get("id"))
        for item in (training_path or {}).get("gates") or []
        if isinstance(item, dict) and item.get("id")
    }
    required_training_path_gates = {
        "reference_and_architecture",
        "remote_hayate_reference_probe",
        "modelopt_loss_and_recipe",
        "target_rollout_corpus",
        "runtime_container",
        "hidden_train_export_submit",
        "trained_artifact_contracts",
    }
    missing_training_path_gates = sorted(required_training_path_gates - training_path_gates)
    if missing_training_path_gates:
        problems.append(f"training path manifest is missing expected gates: {missing_training_path_gates}")
    closure_contracts = {
        str(item.get("id")): item
        for item in (training_path or {}).get("gate_closure_contracts") or []
        if isinstance(item, dict) and item.get("id")
    }
    if set(closure_contracts) != required_training_path_gates:
        problems.append("training path manifest did not write gate_closure_contracts for every expected gate")
    target_contract = closure_contracts.get("target_rollout_corpus") or {}
    target_report_labels = {
        str(item.get("label"))
        for item in target_contract.get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    if not {"rollout_state", "corpus_strategy"}.issubset(target_report_labels):
        problems.append("target_rollout_corpus closure contract does not require rollout_state and corpus_strategy")
    runtime_contract = closure_contracts.get("runtime_container") or {}
    runtime_report_labels = {
        str(item.get("label"))
        for item in runtime_contract.get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    if not {"container_preflight", "vllm_source_build", "vllm_abi_probe", "megatron_compat"}.issubset(
        runtime_report_labels
    ):
        problems.append("runtime_container closure contract does not require all runtime evidence reports")
    reference_evidence = (
        (training_path or {}).get("reference_evidence")
        if isinstance((training_path or {}).get("reference_evidence"), dict)
        else {}
    )
    if not isinstance(reference_evidence.get("remote_reference_proven"), bool):
        problems.append("training path manifest did not record boolean remote_reference_proven")
    if (
        reference_evidence.get("remote_reference_proven") is False
        and "remote_hayate_reference_probe" not in ((training_path or {}).get("open_gates") or [])
    ):
        problems.append("training path manifest did not keep remote_hayate_reference_probe open while remote reference is unproven")
    for key in ["local_modelopt", "remote_probe", "hayate_modelopt", "hayate_specforge"]:
        if not isinstance(reference_evidence.get(key), dict):
            problems.append(f"training path manifest reference_evidence missing object: {key}")
    reference_decisions = (
        (training_path or {}).get("reference_decisions")
        if isinstance((training_path or {}).get("reference_decisions"), dict)
        else {}
    )
    route = reference_decisions.get("training_route") if isinstance(reference_decisions.get("training_route"), dict) else {}
    modelopt_source = (
        reference_decisions.get("modelopt_source") if isinstance(reference_decisions.get("modelopt_source"), dict) else {}
    )
    specforge = (
        reference_decisions.get("specforge_qwen3_235b")
        if isinstance(reference_decisions.get("specforge_qwen3_235b"), dict)
        else {}
    )
    if route.get("primary_route") != "fixed_exported_eagle3_draft_first":
        problems.append("training path manifest did not record fixed exported draft as the primary route decision")
    if modelopt_source.get("source_of_truth") != "local_modelopt":
        problems.append("training path manifest did not record local_modelopt as ModelOpt source of truth")
    if not specforge.get("matched_fields") or not specforge.get("rejected_fields"):
        problems.append("training path manifest did not record SpecForge matched/rejected reference fields")
    if nemo_drift_error or (nemo_drift or {}).get("overall_status") not in {"pass", "warn", "incomplete"}:
        problems.append("refresh did not write structured nemo_rl_eagle3_drift.json")
    if "fixed exported" not in str((nemo_drift or {}).get("recommendation") or "").lower():
        problems.append("NeMo-RL drift report does not record the fixed exported draft first route")
    if nemo_integration_error or (nemo_integration or {}).get("overall_status") not in {"pass", "warn"}:
        problems.append("refresh did not write structured nemo_rl_specdec_integration.json")
    scenario_statuses = {
        str(item.get("name")): str(item.get("status"))
        for item in (gated_contract or {}).get("scenarios") or []
        if isinstance(item, dict) and item.get("name")
    }
    if gated_contract_error or (gated_contract or {}).get("overall_status") != "pass":
        problems.append("refresh did not write PASS eagle3_pipeline_gated_submit_contract.json")
    if safe_preflight_error or (safe_preflight or {}).get("overall_status") not in {"pass", "warn"}:
        problems.append("refresh did not write structured eagle3_operator_safe_actions_preflight.json")
    safe_action_filter = set((safe_preflight or {}).get("action_filter") or [])
    if not safe_action_filter.intersection({"probe_remote_hosts", "poll_megatron_compat_probe"}):
        problems.append("safe-actions preflight did not record any expected safe action filter")
    for scenario in ["ready_missing_critical_env", "ready_mismatched_input_data"]:
        if scenario_statuses.get(scenario) != "pass":
            problems.append(f"gated submit contract did not pass scenario: {scenario}")
    if goal_error or (goal or {}).get("overall_status") not in {"incomplete", "pass"}:
        problems.append("refresh did not write structured goal evidence")
    if loss_mask_req.get("status") != "proven":
        problems.append("goal evidence did not mark ModelOpt loss-mask requirement proven")
    if recipe_req.get("status") != "proven":
        problems.append("goal evidence did not mark ModelOpt recipe override requirement proven")
    if training_path_req.get("status") != "proven":
        problems.append("goal evidence did not mark training path manifest requirement proven")
    training_path_req_evidence = (
        training_path_req.get("evidence") if isinstance(training_path_req.get("evidence"), dict) else {}
    )
    if training_path_req_evidence.get("reference_evidence_contract_ok") is not True:
        problems.append("goal evidence did not record a valid training-path reference evidence contract")
    if training_path_req_evidence.get("reference_decisions_contract_ok") is not True:
        problems.append("goal evidence did not record a valid training-path reference decisions contract")
    if training_path_req_evidence.get("gate_closure_contracts_ok") is not True:
        problems.append("goal evidence did not record a valid training-path gate closure contract")
    if not isinstance(training_path_req_evidence.get("remote_reference_proven"), bool):
        problems.append("goal evidence did not record boolean training-path remote_reference_proven")
    if (
        training_path_req_evidence.get("remote_reference_proven") is False
        and "remote_hayate_reference_probe" not in (training_path_req_evidence.get("open_gates") or [])
    ):
        problems.append("goal evidence did not keep remote_hayate_reference_probe open while remote reference is unproven")
    if rl_route_req.get("status") != "proven":
        problems.append("goal evidence did not mark fixed-draft-first RL route requirement proven")
    if report_status(refresh, "modelopt_loss_mask_patch") != "pass":
        problems.append("refresh summary did not record modelopt_loss_mask_patch=pass")
    if report_status(refresh, "input_discovery") not in {"pass", "warn"}:
        problems.append("refresh summary did not record structured input_discovery status")
    if report_status(refresh, "provenance") not in {"pass", "warn"}:
        problems.append("refresh summary did not record structured provenance status")
    if report_status(refresh, "cluster_environment") not in {"pass", "fail", "warn"}:
        problems.append("refresh summary did not record structured cluster_environment status")
    if report_status(refresh, "remote_access_diagnostics") not in {
        "pass",
        "blocked_local_dns",
        "unreachable",
        "missing_probe",
        "missing_hosts",
    }:
        problems.append("refresh summary did not record structured remote_access_diagnostics status")
    if report_status(refresh, "readiness") not in {"pass", "fail", "warn"}:
        problems.append("refresh summary did not record structured readiness status")
    if report_status(refresh, "rollout_submit_preflight") not in {"pass", "warn", "fail"}:
        problems.append("refresh summary did not record structured rollout_submit_preflight status")
    if report_status(refresh, "hayate_inventory") not in {"present", "missing", "empty"}:
        problems.append("refresh summary did not record Hayate inventory text status")
    if report_status(refresh, "modelopt_recipe_overrides_current") != "pass":
        problems.append("refresh summary did not record modelopt_recipe_overrides_current=pass")
    if report_status(refresh, "training_path_manifest") not in {"defined", "pass"}:
        problems.append("refresh summary did not record training_path_manifest=defined/pass")
    if report_status(refresh, "training_path_manifest_validation") != "pass":
        problems.append("refresh summary did not record training_path_manifest_validation=pass")
    if report_status(refresh, "nemo_rl_eagle3_drift") not in {"pass", "warn", "incomplete"}:
        problems.append("refresh summary did not record structured nemo_rl_eagle3_drift status")
    if report_status(refresh, "nemo_rl_specdec_integration") not in {"pass", "warn"}:
        problems.append("refresh summary did not record structured nemo_rl_specdec_integration status")
    if report_status(refresh, "pipeline_gated_submit_contract") != "pass":
        problems.append("refresh summary did not record pipeline_gated_submit_contract=pass")
    if report_status(refresh, "operator_safe_actions_preflight") not in {"pass", "warn"}:
        problems.append("refresh summary did not record operator_safe_actions_preflight=pass/warn")
    if leaked:
        problems.append(f"secret sentinel leaked: {leaked}")
    if traces:
        problems.append("traceback leaked into refresh output")

    if problems:
        add(
            checks,
            "operator state refresh preserves ModelOpt evidence",
            "fail",
            "refresh_eagle3_operator_state.py did not preserve the no-submit ModelOpt evidence contract",
            problems=problems,
            returncode=result.returncode,
            refresh_error=refresh_error,
            goal_error=goal_error,
            input_discovery_error=input_discovery_error,
            provenance_error=provenance_error,
            cluster_probe_error=cluster_probe_error,
            readiness_error=readiness_error,
            rollout_submit_error=rollout_submit_error,
            loss_mask_error=loss_mask_error,
            recipe_error=recipe_error,
            training_path_error=training_path_error,
            training_path_validation_error=training_path_validation_error,
            nemo_drift_error=nemo_drift_error,
            nemo_integration_error=nemo_integration_error,
            gated_contract_error=gated_contract_error,
            safe_preflight_error=safe_preflight_error,
            overall_status=(refresh or {}).get("overall_status"),
            failed_steps=sorted(failed_steps),
            missing_steps=missing_steps,
            loss_mask_status=(loss_mask or {}).get("overall_status"),
            recipe_status=(recipe or {}).get("overall_status"),
            training_path_status=(training_path or {}).get("overall_status"),
            training_path_validation_status=(training_path_validation or {}).get("overall_status"),
            nemo_drift_status=(nemo_drift or {}).get("overall_status"),
            nemo_integration_status=(nemo_integration or {}).get("overall_status"),
            gated_contract_status=(gated_contract or {}).get("overall_status"),
            gated_contract_scenarios=scenario_statuses,
            safe_preflight_status=(safe_preflight or {}).get("overall_status"),
            safe_preflight_action_filter=(safe_preflight or {}).get("action_filter"),
            loss_mask_requirement=loss_mask_req,
            recipe_requirement=recipe_req,
            training_path_requirement=training_path_req,
            rl_route_requirement=rl_route_req,
            output_tail=combined[-4000:],
        )
        return
    add(
        checks,
        "operator state refresh preserves ModelOpt evidence",
        "pass",
        "refresh_eagle3_operator_state.py writes loss-mask, recipe, and goal-evidence reports even on lightweight hosts",
        overall_status=(refresh or {}).get("overall_status"),
        expected_local_failures=sorted(failed_steps),
        input_discovery_status=report_status(refresh, "input_discovery"),
        provenance_status=report_status(refresh, "provenance"),
        cluster_environment_status=report_status(refresh, "cluster_environment"),
        readiness_status=report_status(refresh, "readiness"),
        rollout_submit_preflight_status=report_status(refresh, "rollout_submit_preflight"),
        hayate_inventory_status=report_status(refresh, "hayate_inventory"),
        loss_mask_requirement=loss_mask_req.get("status"),
        recipe_requirement=recipe_req.get("status"),
        training_path_status=(training_path or {}).get("overall_status"),
        training_path_validation_status=(training_path_validation or {}).get("overall_status"),
        training_path_requirement=training_path_req.get("status"),
        rl_route_requirement=rl_route_req.get("status"),
        nemo_drift_status=(nemo_drift or {}).get("overall_status"),
        nemo_integration_status=(nemo_integration or {}).get("overall_status"),
        gated_submit_contract_scenarios=scenario_statuses,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator State Refresh Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {check['status'].upper()} | {check['detail']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_operator_refresh_validation_"))
    checks: list[dict[str, Any]] = []
    try:
        validate_refresh_contract(temp_root, checks)
    finally:
        if args.keep_temp:
            checks.append(
                {
                    "name": "temporary artifacts",
                    "status": "info",
                    "detail": str(temp_root),
                    "evidence": {},
                }
            )
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
    overall = "pass" if checks and all(check["status"] in {"pass", "info"} for check in checks) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "counts": status_counts(checks),
        "checks": checks,
    }
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
