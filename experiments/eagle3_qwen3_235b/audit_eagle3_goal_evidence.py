#!/usr/bin/env python3
"""Build a requirement-by-requirement evidence matrix for the Qwen3 Eagle3 goal.

This report is no-submit. It does not replace the final completion audit; it
shows what evidence exists today for the user-facing objective and what proof is
still missing before we can claim a trained Qwen3-235B Eagle3 draft is usable in
the RL context.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
EXPECTED_ARTIFACT_FLOW = [
    "rollout_conversation_corpus",
    "verifier_hidden_states",
    "modelopt_checkpoint",
    "hf_eagle3_export",
    "vllm_eagle3_draft",
    "rl_vllm_draft_validation",
]


@dataclass
class Requirement:
    order: int
    area: str
    requirement: str
    status: str
    proof_required: str
    current_evidence: str
    next_step: str
    evidence: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--reference-arch", type=Path, default=EXP / "qwen3_235b_thinking_eagle3_architecture.json")
    parser.add_argument("--qwen3-static-inputs-json", type=Path)
    parser.add_argument("--qwen3-static-inputs-validation-json", type=Path)
    parser.add_argument("--remote-host-probe-json", type=Path)
    parser.add_argument("--hayate-workflow-json", type=Path)
    parser.add_argument("--hayate-specforge-reference-json", type=Path)
    parser.add_argument("--draft-inventory-json", type=Path)
    parser.add_argument("--modelopt-loss-mask-json", type=Path)
    parser.add_argument("--modelopt-recipe-overrides-json", type=Path)
    parser.add_argument("--upstream-drift-json", type=Path)
    parser.add_argument("--nemo-rl-drift-json", type=Path)
    parser.add_argument("--training-path-manifest-json", type=Path)
    parser.add_argument("--training-path-manifest-validation-json", type=Path)
    parser.add_argument("--corpus-strategy-json", type=Path)
    parser.add_argument("--rollout-state-json", type=Path)
    parser.add_argument("--rollout-queue-wait-json", type=Path)
    parser.add_argument("--rollout-watcher-health-json", type=Path)
    parser.add_argument("--container-preflight-json", type=Path)
    parser.add_argument("--vllm-source-build-json", type=Path)
    parser.add_argument("--vllm-abi-probe-json", type=Path)
    parser.add_argument("--vllm-source-job-file", type=Path)
    parser.add_argument("--pipeline-submit-preflight-json", type=Path)
    parser.add_argument("--pipeline-gated-submit-json", type=Path)
    parser.add_argument("--pipeline-dry-run-validation-json", type=Path)
    parser.add_argument("--pipeline-analysis-json", type=Path)
    parser.add_argument("--hidden-validation-json", type=Path)
    parser.add_argument("--training-checkpoint-json", type=Path)
    parser.add_argument("--export-artifacts-json", type=Path)
    parser.add_argument("--sweep-json", type=Path)
    parser.add_argument("--next-action-plan-json", type=Path)
    parser.add_argument("--next-action-validation-json", type=Path)
    parser.add_argument("--operator-sheet-json", type=Path)
    parser.add_argument("--operator-execution-json", type=Path)
    parser.add_argument("--operator-followup-validation-json", type=Path)
    parser.add_argument("--megatron-probe-followup-validation-json", type=Path)
    parser.add_argument("--preflight-robustness-validation-json", type=Path)
    parser.add_argument("--completion-audit-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-if-complete-missing", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = args.artifact_root
    reports = root / "reports"
    rollout_state = select_rollout_state_report(root)
    defaults = {
        "qwen3_static_inputs_json": reports / "qwen3_static_inputs.json",
        "qwen3_static_inputs_validation_json": reports / "qwen3_static_inputs_materialization_validation.json",
        "remote_host_probe_json": reports / "eagle3_remote_host_probe.json",
        "hayate_workflow_json": reports / "hayate_modelopt_workflow.json",
        "hayate_specforge_reference_json": reports / "hayate_specforge_reference.json",
        "draft_inventory_json": reports / "eagle3_draft_config_inventory.json",
        "modelopt_loss_mask_json": reports / "modelopt_loss_mask_patch.json",
        "modelopt_recipe_overrides_json": reports / "modelopt_recipe_overrides_current.json",
        "upstream_drift_json": reports / "modelopt_upstream_drift.json",
        "nemo_rl_drift_json": reports / "nemo_rl_eagle3_drift.json",
        "training_path_manifest_json": reports / "eagle3_training_path_manifest.json",
        "training_path_manifest_validation_json": reports / "eagle3_training_path_manifest_validation.json",
        "corpus_strategy_json": reports / "corpus_strategy.json",
        "rollout_state_json": rollout_state,
        "rollout_queue_wait_json": reports / "rollout_queue_wait_summary.json",
        "rollout_watcher_health_json": reports / "rollout_watcher_health.json",
        "container_preflight_json": reports / "container_preflight_analysis.json",
        "vllm_source_build_json": reports / "vllm_native_source_build.json",
        "vllm_abi_probe_json": reports / "vllm_native_abi_probe.json",
        "vllm_source_job_file": ROOT / "latest_vllm_native_source_build_job.txt",
        "pipeline_submit_preflight_json": reports / "eagle3_pipeline_submit_preflight.json",
        "pipeline_gated_submit_json": reports / "eagle3_pipeline_gated_submit.json",
        "pipeline_dry_run_validation_json": reports / "eagle3_pipeline_dry_run_manifest_validation.json",
        "pipeline_analysis_json": reports / "eagle3_pipeline_analysis.json",
        "hidden_validation_json": root / "hidden_states" / "validation_summary.json",
        "training_checkpoint_json": reports / "eagle3_training_checkpoint.json",
        "export_artifacts_json": reports / "eagle3_export_artifacts.json",
        "sweep_json": reports / "trained_draft_spec_tokens_sweep.json",
        "next_action_plan_json": reports / "eagle3_next_actions.json",
        "next_action_validation_json": reports / "eagle3_next_actions_validation.json",
        "operator_sheet_json": reports / "eagle3_operator_sheet.json",
        "operator_execution_json": reports / "eagle3_operator_execution.json",
        "operator_followup_validation_json": reports / "eagle3_operator_followups_validation.json",
        "megatron_probe_followup_validation_json": reports / "megatron_probe_followup_validation.json",
        "preflight_robustness_validation_json": reports / "eagle3_preflight_robustness_validation.json",
        "completion_audit_json": reports / "eagle3_completion_audit.json",
        "json_out": reports / "eagle3_goal_evidence.json",
        "markdown_out": reports / "eagle3_goal_evidence.md",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def json_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(payload.get("overall_status") or payload.get("status") or decision.get("overall_status") or "unknown")


def active_rollout_job_ids(root: Path) -> set[str]:
    queue_path = root / "reports" / "rollout_queue_wait_summary.json"
    if not queue_path.exists():
        return set()
    try:
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
    ids: set[str] = set()
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
        state = str(snapshot.get("state") or "").upper()
        if state in active_states:
            job_id = str(job.get("job_id") or snapshot.get("job_id") or "")
            if job_id:
                ids.add(job_id)
    return ids


def rollout_state_job_id(payload: dict[str, Any]) -> str:
    for key in ("job_id", "rollout_job_id"):
        value = payload.get(key)
        if value:
            return str(value)
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    if job.get("job_id"):
        return str(job["job_id"])
    return ""


def select_rollout_state_report(root: Path) -> Path:
    reports = root / "reports"
    default = reports / "rollout_capture_state_advance.json"
    active_ids = active_rollout_job_ids(root)
    candidates: list[tuple[int, float, Path]] = []
    for path in reports.glob("rollout_capture*_state_advance.json"):
        if path.name == "rollout_capture_compact16n4g_state_advance.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id = rollout_state_job_id(payload)
        status = json_status(payload)
        priority = 0
        if active_ids and job_id in active_ids:
            priority = 3
        elif status in {"running", "pass"}:
            priority = 2
        elif path == default:
            priority = 1
        candidates.append((priority, path.stat().st_mtime, path))
    if not candidates:
        return default
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "not provided"
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"top-level JSON is not an object: {path}"
    return payload, None


def status_of(payload: dict[str, Any] | None, error: str | None = None) -> str:
    if error:
        return "missing"
    if not payload:
        return "missing"
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(
        payload.get("overall_status")
        or payload.get("status")
        or decision.get("overall_status")
        or "unknown"
    )


def nested(payload: dict[str, Any] | None, keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def compact_actions(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    actions = (plan or {}).get("next_actions") or []
    result = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "stage": item.get("stage"),
                "submits_slurm": item.get("submits_slurm"),
                "heavy_gpu": item.get("heavy_gpu"),
            }
        )
    return result


def ready_actions(plan: dict[str, Any] | None) -> list[str]:
    return [
        str(item.get("id"))
        for item in (plan or {}).get("next_actions") or []
        if isinstance(item, dict) and item.get("status") == "ready_for_operator" and item.get("command")
    ]


def artifact_flow_rows(training_path: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        item
        for item in (training_path or {}).get("artifact_flow") or []
        if isinstance(item, dict)
    ]


def artifact_flow_ids(training_path: dict[str, Any] | None) -> list[str]:
    return [str(item.get("id")) for item in artifact_flow_rows(training_path) if item.get("id")]


def artifact_flow_complete(training_path: dict[str, Any] | None) -> bool:
    rows = artifact_flow_rows(training_path)
    return (
        bool(rows)
        and artifact_flow_ids(training_path) == EXPECTED_ARTIFACT_FLOW
        and (training_path or {}).get("artifact_flow_complete") is True
        and all(item.get("proof_status") == "pass" for item in rows)
    )


def report_label(path: Path | None, payload: dict[str, Any] | None, error: str | None) -> str:
    if error:
        return error
    return f"{path}: status={status_of(payload)}"


def read_key_values(path: Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, "not provided"
    if not path.exists():
        return {}, f"not visible: {path}"
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values, None


def source_vllm_site(args: argparse.Namespace, source_build: dict[str, Any] | None, source_job: dict[str, str]) -> str:
    return str(
        (source_build or {}).get("output_site")
        or source_job.get("output_site")
        or args.artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"
    )


def abi_probe_site_result(abi_probe: dict[str, Any] | None, source_site: str) -> dict[str, Any] | None:
    for item in (abi_probe or {}).get("results") or []:
        if isinstance(item, dict) and str(item.get("site") or "") == source_site:
            return item
    return None


def abi_probe_site_passed(abi_probe: dict[str, Any] | None, source_site: str) -> bool:
    item = abi_probe_site_result(abi_probe, source_site)
    parsed = item.get("parsed") if isinstance((item or {}).get("parsed"), dict) else {}
    return bool(
        item
        and item.get("returncode") == 0
        and parsed.get("vllm_c_ok") is True
        and parsed.get("compilation_config_ok") is not False
    )


def abi_probe_site_failed(abi_probe: dict[str, Any] | None, source_site: str) -> bool:
    item = abi_probe_site_result(abi_probe, source_site)
    return bool(item and not abi_probe_site_passed(abi_probe, source_site))


def add(
    rows: list[Requirement],
    area: str,
    requirement: str,
    status: str,
    proof_required: str,
    current_evidence: str,
    next_step: str,
    **evidence: Any,
) -> None:
    rows.append(
        Requirement(
            order=len(rows) + 1,
            area=area,
            requirement=requirement,
            status=status,
            proof_required=proof_required,
            current_evidence=current_evidence,
            next_step=next_step,
            evidence=evidence,
        )
    )


def check_architecture(args: argparse.Namespace, rows: list[Requirement]) -> None:
    payload, error = load_json(args.reference_arch)
    if error:
        add(
            rows,
            "model_contract",
            "Qwen3-235B Thinking Eagle3 architecture is derived",
            "missing",
            "Reference JSON with Qwen3-235B Thinking Eagle3 fields.",
            error,
            "Regenerate architecture metadata from Qwen3-235B Thinking-2507 config.",
            path=str(args.reference_arch),
        )
        return
    cfg = payload.get("eagle_architecture_config", payload)
    expected = {
        "num_hidden_layers": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
        "rope_theta": 5000000,
    }
    mismatches = {key: {"actual": cfg.get(key), "expected": value} for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        add(
            rows,
            "model_contract",
            "Qwen3-235B Thinking Eagle3 architecture is derived",
            "fail",
            "Reference JSON must match expected Eagle3/Qwen3 fields.",
            "Architecture JSON exists but fields do not match the expected Thinking-2507 contract.",
            "Fix reference architecture before training.",
            mismatches=mismatches,
            path=str(args.reference_arch),
        )
        return
    add(
        rows,
        "model_contract",
        "Qwen3-235B Thinking Eagle3 architecture is derived",
        "proven",
        "Reference JSON with Qwen3-235B Thinking Eagle3 fields.",
        f"{args.reference_arch} matches expected Eagle3/Qwen3 fields.",
        "No action required unless verifier model/version changes.",
        path=str(args.reference_arch),
        expected=expected,
    )


def check_statuses(payload: dict[str, Any] | None) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in (payload or {}).get("checks") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name:
            statuses[str(name)] = str(item.get("status") or "unknown")
    return statuses


def build_requirements(args: argparse.Namespace) -> list[Requirement]:
    rows: list[Requirement] = []

    static_inputs, static_inputs_error = load_json(args.qwen3_static_inputs_json)
    static_inputs_validation, static_inputs_validation_error = load_json(args.qwen3_static_inputs_validation_json)
    remote_host_probe, remote_host_probe_error = load_json(args.remote_host_probe_json)
    hayate, hayate_error = load_json(args.hayate_workflow_json)
    hayate_specforge, hayate_specforge_error = load_json(args.hayate_specforge_reference_json)
    draft_inventory, draft_inventory_error = load_json(args.draft_inventory_json)
    loss_mask, loss_mask_error = load_json(args.modelopt_loss_mask_json)
    recipe_overrides, recipe_overrides_error = load_json(args.modelopt_recipe_overrides_json)
    upstream, upstream_error = load_json(args.upstream_drift_json)
    nemo_drift, nemo_error = load_json(args.nemo_rl_drift_json)
    training_path, training_path_error = load_json(args.training_path_manifest_json)
    training_path_validation, training_path_validation_error = load_json(args.training_path_manifest_validation_json)
    corpus, corpus_error = load_json(args.corpus_strategy_json)
    rollout, rollout_error = load_json(args.rollout_state_json)
    rollout_queue_wait, rollout_queue_wait_error = load_json(args.rollout_queue_wait_json)
    rollout_watcher_health, rollout_watcher_health_error = load_json(args.rollout_watcher_health_json)
    container, container_error = load_json(args.container_preflight_json)
    vllm_source_build, vllm_source_build_error = load_json(args.vllm_source_build_json)
    vllm_abi_probe, vllm_abi_probe_error = load_json(args.vllm_abi_probe_json)
    vllm_source_job, vllm_source_job_error = read_key_values(args.vllm_source_job_file)
    pipeline_submit, pipeline_submit_error = load_json(args.pipeline_submit_preflight_json)
    pipeline_gated_submit, pipeline_gated_submit_error = load_json(args.pipeline_gated_submit_json)
    pipeline_dry_run_validation, pipeline_dry_run_validation_error = load_json(args.pipeline_dry_run_validation_json)
    pipeline, pipeline_error = load_json(args.pipeline_analysis_json)
    hidden, hidden_error = load_json(args.hidden_validation_json)
    training_ckpt, training_ckpt_error = load_json(args.training_checkpoint_json)
    export_artifacts, export_error = load_json(args.export_artifacts_json)
    sweep, sweep_error = load_json(args.sweep_json)
    plan, plan_error = load_json(args.next_action_plan_json)
    plan_validation, plan_validation_error = load_json(args.next_action_validation_json)
    operator, operator_error = load_json(args.operator_sheet_json)
    operator_sheet_validation_path = args.artifact_root / "reports/eagle3_operator_sheet_validation.json"
    operator_sheet_validation, operator_sheet_validation_error = load_json(operator_sheet_validation_path)
    operator_execution, operator_execution_error = load_json(args.operator_execution_json)
    operator_followup_validation, operator_followup_validation_error = load_json(args.operator_followup_validation_json)
    megatron_probe_followup, megatron_probe_followup_error = load_json(args.megatron_probe_followup_validation_json)
    preflight_robustness, preflight_robustness_error = load_json(args.preflight_robustness_validation_json)
    completion, completion_error = load_json(args.completion_audit_json)

    check_architecture(args, rows)

    static_input_status = status_of(static_inputs, static_inputs_error)
    static_validation_status = status_of(static_inputs_validation, static_inputs_validation_error)
    static_check_statuses = check_statuses(static_inputs)
    required_static_checks = [
        "config.json materialized",
        "tokenizer_config.json materialized",
        "generation_config.json materialized",
        "Eagle3 architecture derived",
        "Qwen3 generation template prepared",
    ]
    required_static_pass = all(static_check_statuses.get(name) == "pass" for name in required_static_checks)
    static_inputs_ready = (
        static_inputs is not None
        and static_input_status in {"pass", "warn"}
        and required_static_pass
        and static_validation_status == "pass"
    )
    add(
        rows,
        "model_contract",
        "Qwen3 verifier static inputs are materialized and pipeline-ready",
        "proven" if static_inputs_ready else ("incomplete" if static_inputs or static_inputs_validation else "missing"),
        "qwen3_static_inputs.json PASS/WARN with required verifier config, architecture, and template outputs, plus materializer validation PASS.",
        f"static_inputs={report_label(args.qwen3_static_inputs_json, static_inputs, static_inputs_error)}; validation={report_label(args.qwen3_static_inputs_validation_json, static_inputs_validation, static_inputs_validation_error)}",
        "Run the container preflight to prove the template mask check inside the target runtime.",
        static_input_status=static_input_status,
        static_validation_status=static_validation_status,
        required_static_checks={name: static_check_statuses.get(name) for name in required_static_checks},
        counts=(static_inputs or {}).get("counts"),
        outputs=(static_inputs or {}).get("outputs"),
        validation_counts=(static_inputs_validation or {}).get("counts"),
    )

    remote_status = status_of(remote_host_probe, remote_host_probe_error)
    reachable_hosts = (remote_host_probe or {}).get("reachable_hosts") or []
    reachable_records = [
        host
        for host in (remote_host_probe or {}).get("hosts") or []
        if isinstance(host, dict) and host.get("reachable")
    ]
    visible_paths = [
        str(item.get("path") or "")
        for host in reachable_records
        for item in host.get("paths") or []
        if isinstance(item, dict) and item.get("exists") and item.get("readable")
    ]
    hayate_modelopt_paths = [
        path
        for path in visible_paths
        if "TensorRT-Model-Optimizer" in path or path.endswith("/hiso/code/Model-Optimizer")
    ]
    hayate_draft_paths = [
        path
        for path in visible_paths
        if "SpecForge" in path or path.endswith("/feat-eagle3-online-specdec/models")
    ]
    remote_workdir = str((remote_host_probe or {}).get("remote_workdir") or "")
    artifact_root = str((remote_host_probe or {}).get("artifact_root") or "")
    remote_path_proven = (
        bool(reachable_hosts)
        and bool(hayate_modelopt_paths)
        and bool(hayate_draft_paths)
        and (not remote_workdir or remote_workdir in visible_paths)
        and (not artifact_root or artifact_root in visible_paths)
    )
    add(
        rows,
        "reference",
        "Remote execution host and Hayate path probe is recorded",
        "proven" if remote_status == "pass" and remote_path_proven else ("incomplete" if remote_host_probe else "missing"),
        "eagle3_remote_host_probe.json PASS with at least one reachable host and checked remote ModelOpt/Hayate paths.",
        report_label(args.remote_host_probe_json, remote_host_probe, remote_host_probe_error),
        "Run probe_eagle3_remote_host.py after DNS/VPN recovers, then use the reachable alias for remote follow-up.",
        remote_status=remote_status,
        reachable_hosts=reachable_hosts,
        counts=(remote_host_probe or {}).get("counts"),
        hayate_modelopt_paths=hayate_modelopt_paths,
        hayate_draft_paths=hayate_draft_paths,
        remote_workdir=remote_workdir,
        artifact_root=artifact_root,
        remote_path_proven=remote_path_proven,
    )

    hayate_status = status_of(hayate, hayate_error)
    hayate_class = nested(hayate, ["classification", "classification"]) or nested(hayate, ["classification"])
    add(
        rows,
        "reference",
        "Hayate/Hiso Eagle3 workflow has been analyzed and classified",
        "proven"
        if hayate and hayate_status in {"reference_only", "needs_review", "pass", "warn"}
        else ("incomplete" if hayate else "missing"),
        "Accessible Hayate ModelOpt workflow report showing reusable vs non-drop-in parts.",
        report_label(args.hayate_workflow_json, hayate, hayate_error),
        "Re-run analyze_hayate_modelopt_workflow.py if Hayate changes the checkout.",
        classification=hayate_class,
        hayate_status=hayate_status,
        hayate_source=(hayate or {}).get("source"),
        live_hayate_visible=(hayate or {}).get("live_hayate_visible"),
        snapshot=(hayate or {}).get("snapshot"),
    )

    hayate_specforge_status = status_of(hayate_specforge, hayate_specforge_error)
    qwen3_235b_comparison = (hayate_specforge or {}).get("qwen3_235b_comparison") or {}
    comparison_rows = [
        row for row in qwen3_235b_comparison.get("rows") or [] if isinstance(row, dict)
    ]
    matched_fields = [row.get("field") for row in comparison_rows if row.get("match") is True]
    mismatched_fields = [row.get("field") for row in comparison_rows if row.get("match") is False]
    add(
        rows,
        "reference",
        "Hayate/Hiso SpecForge Qwen3 reference has been compared against current Qwen3-235B Thinking config",
        "proven"
        if hayate_specforge_status in {"reference_only", "matches_current"}
        else ("incomplete" if hayate_specforge else "missing"),
        "Hayate SpecForge report with Qwen3-235B field comparison and conclusion about non-drop-in config differences.",
        report_label(args.hayate_specforge_reference_json, hayate_specforge, hayate_specforge_error),
        "Use the SpecForge example flags as reference only; derive final Eagle3 config from the current verifier config.",
        specforge_status=hayate_specforge_status,
        specforge_source=(hayate_specforge or {}).get("source"),
        live_specforge_visible=(hayate_specforge or {}).get("live_specforge_visible"),
        bundled_reference=(hayate_specforge or {}).get("bundled_reference"),
        matched_fields=matched_fields,
        mismatched_fields=mismatched_fields,
        conclusion=qwen3_235b_comparison.get("conclusion"),
    )

    draft_inventory_status = status_of(draft_inventory, draft_inventory_error)
    draft_warnings = (draft_inventory or {}).get("warnings") or []
    draft_configs_scanned = int((draft_inventory or {}).get("configs_scanned") or 0)
    draft_inventory_proven = draft_inventory is not None and (
        draft_inventory_status in {"pass", "warn"} and (draft_configs_scanned > 0 or bool(draft_warnings))
    )
    add(
        rows,
        "reference",
        "Existing Hayate draft artifacts have been inventoried or access limitations are recorded",
        "proven" if draft_inventory_proven else ("incomplete" if draft_inventory else "missing"),
        "Draft config inventory report over Hayate and local export roots, including permission/access warnings.",
        report_label(args.draft_inventory_json, draft_inventory, draft_inventory_error),
        "Re-run inventory_eagle3_draft_configs.py if Hayate exposes new draft model paths or permissions change.",
        inventory_status=draft_inventory_status,
        configs_scanned=draft_configs_scanned,
        roots=(draft_inventory or {}).get("roots"),
        root_statuses=(draft_inventory or {}).get("root_statuses"),
        warning_count=len(draft_warnings),
        warnings=draft_warnings[:4],
        recommendation=(draft_inventory or {}).get("recommendation"),
    )

    modelopt_ok = status_of(loss_mask, loss_mask_error) == "pass"
    upstream_ok = upstream is not None and status_of(upstream, upstream_error) not in {"missing", "fail"}
    upstream_decision = (upstream or {}).get("decision") if isinstance((upstream or {}).get("decision"), dict) else {}
    add(
        rows,
        "modelopt",
        "ModelOpt path and Qwen3 loss-mask patch are known",
        "proven" if modelopt_ok else ("incomplete" if loss_mask else "missing"),
        "ModelOpt loss-mask patch validator PASS plus drift/provenance context.",
        f"loss_mask={report_label(args.modelopt_loss_mask_json, loss_mask, loss_mask_error)}; upstream={report_label(args.upstream_drift_json, upstream, upstream_error)}",
        "Keep the TRT-LLM hidden-state loss_mask patch applied before hidden-state dump.",
        loss_mask_status=status_of(loss_mask, loss_mask_error),
        upstream_status=status_of(upstream, upstream_error),
        training_source_decision=upstream_decision.get("overall_status"),
        upstream_head_matches=upstream_decision.get("upstream_head_matches"),
        allowed_focus_diffs=upstream_decision.get("allowed_focus_diffs") or [],
        disallowed_focus_diffs=upstream_decision.get("disallowed_focus_diffs") or [],
        upstream_context_available=upstream_ok,
    )

    recipe_status = status_of(recipe_overrides, recipe_overrides_error)
    add(
        rows,
        "modelopt",
        "ModelOpt Eagle3 recipe overrides match the Qwen3-235B Thinking architecture",
        "proven" if recipe_status == "pass" else ("incomplete" if recipe_overrides else "missing"),
        "modelopt_recipe_overrides_current.json PASS proving the offline training wrapper emits valid current ModelOpt recipe overrides for the derived Qwen3-235B Eagle3 architecture.",
        report_label(args.modelopt_recipe_overrides_json, recipe_overrides, recipe_overrides_error),
        "Run validate_modelopt_recipe_overrides.py for the offline wrapper before submitting hidden-state/train/export jobs.",
        recipe_status=recipe_status,
        wrapper=(recipe_overrides or {}).get("wrapper"),
        training_mode=(recipe_overrides or {}).get("training_mode"),
        recipe_config=(recipe_overrides or {}).get("recipe_config"),
        override_count=(recipe_overrides or {}).get("override_count"),
        check_status_counts=(recipe_overrides or {}).get("counts"),
        warnings=(recipe_overrides or {}).get("warnings") or [],
    )

    nemo_status = status_of(nemo_drift, nemo_error)
    generation_first = "fixed exported" in str((nemo_drift or {}).get("recommendation", "")).lower()
    add(
        rows,
        "rl_context",
        "RL integration route is fixed exported draft first",
        "proven" if nemo_drift and nemo_status in {"pass", "warn", "incomplete"} else "missing",
        "NeMo-RL drift/integration evidence that generation-only fixed draft is the primary route.",
        report_label(args.nemo_rl_drift_json, nemo_drift, nemo_error),
        "Use online draft training only after fixed-draft speed/reward smoke is proven.",
        nemo_status=nemo_status,
        generation_first=generation_first,
        recommendation=(nemo_drift or {}).get("recommendation"),
    )

    path_status = status_of(training_path, training_path_error)
    path_validation_status = status_of(training_path_validation, training_path_validation_error)
    training_path_gates = [
        str(item.get("id"))
        for item in (training_path or {}).get("gates") or []
        if isinstance(item, dict) and item.get("id")
    ]
    required_training_path_gates = {
        "reference_and_architecture",
        "remote_hayate_reference_probe",
        "modelopt_loss_and_recipe",
        "target_rollout_corpus",
        "runtime_container",
        "hidden_train_export_submit",
        "trained_artifact_contracts",
    }
    missing_training_path_gates = sorted(required_training_path_gates - set(training_path_gates))
    training_path_closure_contracts = {
        str(item.get("id")): item
        for item in (training_path or {}).get("gate_closure_contracts") or []
        if isinstance(item, dict) and item.get("id")
    }
    target_contract_labels = {
        str(item.get("label"))
        for item in (training_path_closure_contracts.get("target_rollout_corpus") or {}).get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    runtime_contract_labels = {
        str(item.get("label"))
        for item in (training_path_closure_contracts.get("runtime_container") or {}).get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    gate_closure_contracts_ok = (
        set(training_path_closure_contracts) == required_training_path_gates
        and {"rollout_state", "corpus_strategy"}.issubset(target_contract_labels)
        and {"container_preflight", "vllm_source_build", "vllm_abi_probe", "megatron_compat"}.issubset(
            runtime_contract_labels
        )
    )
    training_path_artifact_flow = artifact_flow_rows(training_path)
    training_path_artifact_flow_ids = artifact_flow_ids(training_path)
    training_path_artifact_flow_complete = artifact_flow_complete(training_path)
    training_path_reference_evidence = (
        (training_path or {}).get("reference_evidence")
        if isinstance((training_path or {}).get("reference_evidence"), dict)
        else {}
    )
    training_path_open_gates = (training_path or {}).get("open_gates") or []
    remote_reference_proven = training_path_reference_evidence.get("remote_reference_proven")
    remote_reference_gate_state_ok = (
        remote_reference_proven is True
        or (
            remote_reference_proven is False
            and "remote_hayate_reference_probe" in training_path_open_gates
        )
    )
    reference_evidence_contract_ok = (
        isinstance(remote_reference_proven, bool)
        and remote_reference_gate_state_ok
        and isinstance(training_path_reference_evidence.get("local_modelopt"), dict)
        and isinstance(training_path_reference_evidence.get("remote_probe"), dict)
        and isinstance(training_path_reference_evidence.get("hayate_modelopt"), dict)
        and isinstance(training_path_reference_evidence.get("hayate_specforge"), dict)
    )
    training_path_reference_decisions = (
        (training_path or {}).get("reference_decisions")
        if isinstance((training_path or {}).get("reference_decisions"), dict)
        else {}
    )
    training_path_route = (
        training_path_reference_decisions.get("training_route")
        if isinstance(training_path_reference_decisions.get("training_route"), dict)
        else {}
    )
    training_path_modelopt_source = (
        training_path_reference_decisions.get("modelopt_source")
        if isinstance(training_path_reference_decisions.get("modelopt_source"), dict)
        else {}
    )
    training_path_specforge = (
        training_path_reference_decisions.get("specforge_qwen3_235b")
        if isinstance(training_path_reference_decisions.get("specforge_qwen3_235b"), dict)
        else {}
    )
    training_path_hayate_workflow = (
        training_path_reference_decisions.get("hayate_workflow")
        if isinstance(training_path_reference_decisions.get("hayate_workflow"), dict)
        else {}
    )
    training_path_matched_fields = set(training_path_specforge.get("matched_fields") or [])
    training_path_rejected_fields = {
        str(item.get("field"))
        for item in training_path_specforge.get("rejected_fields") or []
        if isinstance(item, dict) and item.get("field")
    }
    reference_decisions_contract_ok = (
        training_path_route.get("primary_route") == "fixed_exported_eagle3_draft_first"
        and training_path_modelopt_source.get("source_of_truth") == "local_modelopt"
        and training_path_modelopt_source.get("upstream_drift_status") in {"pass", "warn"}
        and {"aux_layers", "hidden_size"}.issubset(training_path_matched_fields)
        and bool(training_path_rejected_fields)
        and training_path_hayate_workflow.get("role") == "reference_only"
    )
    path_defined = (
        bool((training_path or {}).get("path_defined"))
        and path_status in {"defined", "pass"}
        and path_validation_status == "pass"
        and not missing_training_path_gates
        and reference_evidence_contract_ok
        and reference_decisions_contract_ok
        and gate_closure_contracts_ok
    )
    add(
        rows,
        "training_path",
        "Qwen3 Eagle3 training path manifest is defined",
        "proven" if path_defined else ("incomplete" if training_path else "missing"),
        "eagle3_training_path_manifest.json records the fixed exported Eagle3 draft route, ModelOpt/Hayate reference roles, remote reference proof state, ordered gates, and current operator actions, with synthetic manifest contract validation PASS.",
        f"manifest={report_label(args.training_path_manifest_json, training_path, training_path_error)}; validation={report_label(args.training_path_manifest_validation_json, training_path_validation, training_path_validation_error)}",
        "Regenerate build_eagle3_training_path_manifest.py after changing any corpus, runtime, ModelOpt, Hayate, or pipeline gate.",
        manifest_status=path_status,
        validation_status=path_validation_status,
        path_defined=bool((training_path or {}).get("path_defined")),
        primary_route=(training_path or {}).get("primary_route"),
        open_gates=training_path_open_gates,
        gate_ids=training_path_gates,
        missing_gate_ids=missing_training_path_gates,
        ready_actions=(training_path or {}).get("ready_actions"),
        final_artifacts_complete=(training_path or {}).get("final_artifacts_complete"),
        artifact_flow_complete=(training_path or {}).get("artifact_flow_complete"),
        artifact_flow_ids=training_path_artifact_flow_ids,
        reference_evidence_contract_ok=reference_evidence_contract_ok,
        reference_decisions_contract_ok=reference_decisions_contract_ok,
        gate_closure_contracts_ok=gate_closure_contracts_ok,
        remote_reference_proven=remote_reference_proven,
        reference_evidence=training_path_reference_evidence,
        reference_decisions=training_path_reference_decisions,
        gate_closure_contracts=training_path_closure_contracts,
        artifact_flow=training_path_artifact_flow,
    )
    add(
        rows,
        "training_path",
        "End-to-end Eagle3 artifact flow contract is complete",
        "proven" if training_path_artifact_flow_complete else ("incomplete" if training_path else "missing"),
        "training path manifest artifact_flow records and proves rollout corpus, hidden states, ModelOpt checkpoint, HF export, vLLM draft, and RL/vLLM sweep artifacts.",
        f"manifest={report_label(args.training_path_manifest_json, training_path, training_path_error)}; artifact_flow_complete={(training_path or {}).get('artifact_flow_complete')}",
        "Close the open artifact_flow rows by producing the required rollout, hidden-state, checkpoint, export, and sweep reports.",
        artifact_flow_complete=(training_path or {}).get("artifact_flow_complete"),
        expected_artifact_flow=EXPECTED_ARTIFACT_FLOW,
        artifact_flow_ids=training_path_artifact_flow_ids,
        open_artifact_flow=[
            {
                "id": item.get("id"),
                "proof_status": item.get("proof_status"),
                "required_reports": item.get("required_reports"),
                "report_statuses": item.get("report_statuses"),
            }
            for item in training_path_artifact_flow
            if item.get("proof_status") != "pass"
        ],
    )

    corpus_decision = corpus.get("decision") if isinstance((corpus or {}).get("decision"), dict) else {}
    corpus_primary = corpus_decision.get("primary_source")
    corpus_target = (corpus or {}).get("target_context")
    corpus_provenance = (
        corpus_decision.get("provenance")
        if isinstance(corpus_decision.get("provenance"), dict)
        else (corpus or {}).get("rollout_alignment")
        if isinstance((corpus or {}).get("rollout_alignment"), dict)
        else {}
    )
    corpus_ready = (
        status_of(corpus, corpus_error) == "pass"
        and corpus_target == "swe_rl"
        and corpus_primary == "actual_rl_rollout"
        and corpus_provenance.get("proves_actual_rollout_corpus") is True
    )
    add(
        rows,
        "data",
        "Training corpus source is actual Qwen3 SWE/RL rollout data",
        "proven" if corpus_ready else ("incomplete" if corpus else "missing"),
        "Corpus strategy PASS with target_context=swe_rl and primary_source=actual_rl_rollout.",
        report_label(args.corpus_strategy_json, corpus, corpus_error),
        "Capture/materialize real Qwen3 SWE/RL rollout conversations; keep math data supplemental for this target.",
        target_context=corpus_target,
        primary_source=corpus_primary,
        provenance=corpus_provenance,
        decision=corpus_decision,
    )

    rollout_decision = rollout.get("decision") if isinstance((rollout or {}).get("decision"), dict) else {}
    rollout_ready = rollout_decision.get("overall_status") == "pass" and rollout_decision.get("next_step") == "pipeline_dry_run"
    add(
        rows,
        "data",
        "Rollout capture has produced pipeline-ready conversations",
        "proven" if rollout_ready else ("incomplete" if rollout else "missing"),
        "Rollout state PASS with next_step=pipeline_dry_run and output conversation JSONL visible.",
        report_label(args.rollout_state_json, rollout, rollout_error),
        "Run the 1-step rollout-capture smoke, analyze logs, then materialize conversations.",
        decision=rollout_decision,
        output_data=(rollout or {}).get("output_data"),
    )

    queue_status = status_of(rollout_queue_wait, rollout_queue_wait_error)
    health_status = status_of(rollout_watcher_health, rollout_watcher_health_error)
    queue_jobs = (rollout_queue_wait or {}).get("jobs") or []
    required_watchers = [
        item
        for item in (rollout_watcher_health or {}).get("watchers") or []
        if isinstance(item, dict) and item.get("required_now")
    ]
    monitor_ready = queue_status in {"idle", "waiting", "terminal_or_unknown", "pass"} and health_status == "pass"
    add(
        rows,
        "operator",
        "Rollout queue and watcher monitoring is live",
        "proven" if monitor_ready else ("incomplete" if rollout_queue_wait or rollout_watcher_health else "missing"),
        "rollout_queue_wait_summary.json records queue state and rollout_watcher_health.json PASS records required watcher liveness.",
        f"queue={report_label(args.rollout_queue_wait_json, rollout_queue_wait, rollout_queue_wait_error)}; health={report_label(args.rollout_watcher_health_json, rollout_watcher_health, rollout_watcher_health_error)}",
        "Keep queue/watcher reports refreshed while waiting for rollout corpus; this does not replace the rollout corpus gate.",
        queue_status=queue_status,
        queue_counts=(rollout_queue_wait or {}).get("counts"),
        queue_jobs=[
            {
                "job_id": item.get("job_id"),
                "state": (item.get("current_squeue") or {}).get("state"),
                "start": (item.get("current_squeue") or {}).get("start"),
                "sample_count": item.get("sample_count"),
                "start_estimate_changes": item.get("start_estimate_changes"),
            }
            for item in queue_jobs
            if isinstance(item, dict)
        ],
        health_status=health_status,
        required_watcher_count=len(required_watchers),
        dead_or_missing_required_watchers=(rollout_watcher_health or {}).get("dead_or_missing_required_watchers"),
        stale_reports=(rollout_watcher_health or {}).get("stale_reports"),
    )

    container_ready = status_of(container, container_error) == "pass"
    add(
        rows,
        "runtime",
        "Selected Slurm container has passed ModelOpt preflight",
        "proven" if container_ready else ("incomplete" if container else "missing"),
        "container_preflight_analysis.json PASS for the exact container/mount/account/partition.",
        report_label(args.container_preflight_json, container, container_error),
        "Submit only the container preflight first; do not run hidden-state dump until it passes.",
        container=(container or {}).get("container"),
        job_id=(container or {}).get("job_id"),
    )

    source_site = source_vllm_site(args, vllm_source_build, vllm_source_job)
    source_status = status_of(vllm_source_build, vllm_source_build_error)
    abi_status = status_of(vllm_abi_probe, vllm_abi_probe_error)
    abi_covers_source = abi_probe_site_passed(vllm_abi_probe, source_site)
    abi_source_failed = abi_probe_site_failed(vllm_abi_probe, source_site)
    source_job_id = vllm_source_job.get("vllm_native_source_build_job")
    vllm_runtime_pass = source_status == "pass" and abi_status == "pass" and abi_covers_source
    vllm_runtime_fail = source_status == "fail" or abi_source_failed
    add(
        rows,
        "runtime",
        "Source-built vLLM runtime passes native ABI probe",
        "proven" if vllm_runtime_pass else ("fail" if vllm_runtime_fail else "incomplete"),
        "vllm_native_source_build.json PASS plus vllm_native_abi_probe.json PASS for the source-built site.",
        f"source_build={report_label(args.vllm_source_build_json, vllm_source_build, vllm_source_build_error)}; abi_probe={report_label(args.vllm_abi_probe_json, vllm_abi_probe, vllm_abi_probe_error)}",
        "Wait for source build PASS, then run the source-site ABI probe before rollout capture.",
        source_site=source_site,
        source_status=source_status,
        abi_status=abi_status,
        abi_covers_source=abi_covers_source,
        abi_source_failed=abi_source_failed,
        source_job_id=source_job_id,
        source_job_file=str(args.vllm_source_job_file),
        source_job_error=vllm_source_job_error,
    )

    megatron_probe_followup_pass = status_of(megatron_probe_followup, megatron_probe_followup_error) == "pass"
    add(
        rows,
        "runtime",
        "Megatron compatibility probe follow-up is guarded",
        "proven" if megatron_probe_followup_pass else ("incomplete" if megatron_probe_followup else "missing"),
        "megatron_probe_followup_validation.json PASS proving missing/bad probe reports fail closed and PASS only prints rollout unless heavy submit is explicitly allowed.",
        report_label(args.megatron_probe_followup_validation_json, megatron_probe_followup, megatron_probe_followup_error),
        "Run validate_megatron_probe_followup.py before using the probe-to-rollout helper.",
        check_status_counts=(megatron_probe_followup or {}).get("check_status_counts"),
        checks=[item.get("name") for item in (megatron_probe_followup or {}).get("checks") or [] if isinstance(item, dict)],
    )

    preflight_robustness_pass = status_of(preflight_robustness, preflight_robustness_error) == "pass"
    add(
        rows,
        "operator",
        "Local preflight failures are structured and redacted",
        "proven" if preflight_robustness_pass else ("incomplete" if preflight_robustness else "missing"),
        "eagle3_preflight_robustness_validation.json PASS proving lightweight-host preflights emit JSON/Markdown without tracebacks or token leakage.",
        report_label(args.preflight_robustness_validation_json, preflight_robustness, preflight_robustness_error),
        "Run validate_eagle3_preflight_robustness.py before relying on local dry-run evidence.",
        checks=[item.get("name") for item in (preflight_robustness or {}).get("checks") or [] if isinstance(item, dict)],
        check_status_counts=(preflight_robustness or {}).get("check_status_counts"),
    )

    pipeline_dry_run_status = status_of(pipeline_dry_run_validation, pipeline_dry_run_validation_error)
    pipeline_dry_run_checks = check_statuses(pipeline_dry_run_validation)
    pipeline_dry_run_pass = pipeline_dry_run_status == "pass"
    pipeline_already_passed = status_of(pipeline, pipeline_error) == "pass"
    add(
        rows,
        "pipeline",
        "Pipeline dry-run manifest is validated",
        "proven"
        if pipeline_dry_run_pass or pipeline_already_passed
        else ("incomplete" if pipeline_dry_run_validation or pipeline else "missing"),
        "eagle3_pipeline_dry_run_manifest_validation.json PASS, or a live pipeline analysis PASS.",
        f"dry_run={report_label(args.pipeline_dry_run_validation_json, pipeline_dry_run_validation, pipeline_dry_run_validation_error)}; analysis={report_label(args.pipeline_analysis_json, pipeline, pipeline_error)}",
        "Treat dry-run evidence as submit-plan validation only; execute the gated pipeline after runtime and rollout gates pass.",
        dry_run_status=pipeline_dry_run_status,
        dry_run_counts=(pipeline_dry_run_validation or {}).get("counts"),
        dry_run_checks=pipeline_dry_run_checks,
        pipeline_status=status_of(pipeline, pipeline_error),
        pipeline_counts=(pipeline or {}).get("counts"),
    )

    pipeline_submit_ready = status_of(pipeline_submit, pipeline_submit_error) == "pass" and pipeline_submit.get("submit_ready") is True if pipeline_submit else False
    add(
        rows,
        "pipeline",
        "Hidden-state/train/export submit preflight is ready",
        "proven" if pipeline_submit_ready else ("incomplete" if pipeline_submit else "missing"),
        "eagle3_pipeline_submit_preflight.json PASS with submit_ready=true.",
        report_label(args.pipeline_submit_preflight_json, pipeline_submit, pipeline_submit_error),
        "After container and rollout gates pass, run run_pipeline_submit_preflight from the next-action plan.",
        submit_ready=(pipeline_submit or {}).get("submit_ready"),
    )

    gated_jobs = (pipeline_gated_submit or {}).get("jobs") if isinstance((pipeline_gated_submit or {}).get("jobs"), dict) else {}
    gated_executed = (pipeline_gated_submit or {}).get("executed") is True
    gated_required_job_keys = ["dump_job", "train_job", "export_job"]
    gated_missing_jobs = [key for key in gated_required_job_keys if not gated_jobs.get(key)]
    gated_pass = status_of(pipeline_gated_submit, pipeline_gated_submit_error) == "pass" and gated_executed and not gated_missing_jobs
    add(
        rows,
        "pipeline",
        "Pipeline gated submit helper has executed",
        "proven" if gated_pass else ("incomplete" if pipeline_gated_submit else "missing"),
        "eagle3_pipeline_gated_submit.json PASS with executed=true and submitted pipeline job IDs.",
        report_label(args.pipeline_gated_submit_json, pipeline_gated_submit, pipeline_gated_submit_error),
        "After preflight reports submit_ready=true and the rollout corpus exists, run submit_eagle3_pipeline_if_ready.py with --execute --allow-heavy-gpu.",
        executed=(pipeline_gated_submit or {}).get("executed"),
        jobs=gated_jobs,
        missing_jobs=gated_missing_jobs,
        job_file=(pipeline_gated_submit or {}).get("job_file"),
        job_file_copy=(pipeline_gated_submit or {}).get("job_file_copy"),
    )

    pipeline_pass = status_of(pipeline, pipeline_error) == "pass"
    add(
        rows,
        "pipeline",
        "Slurm hidden-state dump, train, and export pipeline has passed",
        "proven" if pipeline_pass else ("incomplete" if pipeline else "missing"),
        "eagle3_pipeline_analysis.json PASS over preflight, dump, hidden validation, train, and export stages.",
        report_label(args.pipeline_analysis_json, pipeline, pipeline_error),
        "Submit the Eagle3 pilot pipeline only after pipeline submit preflight passes.",
        counts=(pipeline or {}).get("counts"),
    )

    hidden_pass = status_of(hidden, hidden_error) == "pass"
    add(
        rows,
        "training_data",
        "Verifier hidden states and answer loss masks are validated",
        "proven" if hidden_pass else ("incomplete" if hidden else "missing"),
        "Hidden-state validation PASS with aux states, positive loss masks, and ModelOpt loader check.",
        report_label(args.hidden_validation_json, hidden, hidden_error),
        "Run hidden-state dump on the target rollout corpus, then validate the dump.",
        checked_files=(hidden or {}).get("checked_files"),
        positive_loss_mask_files=(hidden or {}).get("positive_loss_mask_files"),
    )

    training_pass = status_of(training_ckpt, training_ckpt_error) == "pass"
    add(
        rows,
        "artifact",
        "ModelOpt Eagle3 training checkpoint is valid",
        "proven" if training_pass else ("incomplete" if training_ckpt else "missing"),
        "eagle3_training_checkpoint.json PASS, with HF weights, trainer step, modelopt_state.pth, and eagle mode.",
        report_label(args.training_checkpoint_json, training_ckpt, training_ckpt_error),
        "Train the offline Eagle3 draft and run validate_eagle3_training_checkpoint.py before export.",
        checkpoint_dir=(training_ckpt or {}).get("checkpoint_dir"),
        trainer_global_step=(training_ckpt or {}).get("trainer_global_step"),
        modelopt_modes=(training_ckpt or {}).get("modelopt_modes"),
    )

    export_pass = status_of(export_artifacts, export_error) == "pass"
    add(
        rows,
        "artifact",
        "HF and vLLM Eagle3 draft export artifacts are valid",
        "proven" if export_pass else ("incomplete" if export_artifacts else "missing"),
        "eagle3_export_artifacts.json PASS over HF/vLLM configs, safetensors, and one-checkpoint contract.",
        report_label(args.export_artifacts_json, export_artifacts, export_error),
        "After training checkpoint validation, export HF/vLLM draft and run post-export validators.",
        export_dir=(export_artifacts or {}).get("export_dir"),
        vllm_draft_dir=(export_artifacts or {}).get("vllm_draft_dir"),
    )

    sweep_pass = status_of(sweep, sweep_error) == "pass"
    add(
        rows,
        "rl_validation",
        "Trained draft is validated inside the RL/vLLM generation loop",
        "proven" if sweep_pass else ("incomplete" if sweep else "missing"),
        "trained_draft_spec_tokens_sweep.json PASS with execution context and at least one passing spec-token setting.",
        report_label(args.sweep_json, sweep, sweep_error),
        "Run trained-draft spec-token sweep after export artifact contracts pass.",
        recommendation=(sweep or {}).get("recommendation"),
        sweep_rows=len((sweep or {}).get("rows") or []),
    )

    plan_pass = status_of(plan_validation, plan_validation_error) == "pass"
    operator_sheet_validation_pass = status_of(operator_sheet_validation, operator_sheet_validation_error) == "pass"
    operator_followup_validation_pass = status_of(operator_followup_validation, operator_followup_validation_error) == "pass"
    operator_ready = operator is not None and len((operator.get("ready_actions") or [])) > 0
    operator_execution_status = status_of(operator_execution, operator_execution_error)
    add(
        rows,
        "operator",
        "Safe next-action plan and operator sheet exist",
        "proven"
        if plan_pass and operator_ready and operator_sheet_validation_pass and operator_followup_validation_pass
        else ("incomplete" if plan or operator else "missing"),
        "next-action validation PASS plus operator sheet/follow-up validation PASS with print-only, execute, and terminal-state guard commands.",
        f"plan={report_label(args.next_action_plan_json, plan, plan_error)}; validation={report_label(args.next_action_validation_json, plan_validation, plan_validation_error)}; operator={report_label(args.operator_sheet_json, operator, operator_error)}; sheet_validation={report_label(operator_sheet_validation_path, operator_sheet_validation, operator_sheet_validation_error)}; followup_validation={report_label(args.operator_followup_validation_json, operator_followup_validation, operator_followup_validation_error)}",
        "Use the operator sheet to execute only the currently ready gate jobs.",
        ready_actions=ready_actions(plan),
        action_summary=compact_actions(plan),
        operator_ready_actions=[item.get("id") for item in (operator or {}).get("ready_actions") or [] if isinstance(item, dict)],
        operator_sheet_validation_status=status_of(operator_sheet_validation, operator_sheet_validation_error),
        operator_sheet_validation_counts=(operator_sheet_validation or {}).get("counts"),
        operator_followup_validation_status=status_of(operator_followup_validation, operator_followup_validation_error),
        operator_followup_validation_counts=(operator_followup_validation or {}).get("counts"),
        operator_followup_state_counts=(operator_followup_validation or {}).get("followup_state_counts"),
        operator_execution_status=operator_execution_status,
        operator_execution_records=len((operator_execution or {}).get("records") or []),
    )

    add(
        rows,
        "operator",
        "Operator execution records are valid after gate execution",
        "proven"
        if operator_execution_status == "pass"
        else ("fail" if operator_execution_status == "fail" else "incomplete"),
        "eagle3_operator_execution.json PASS after operator-triggered ready actions.",
        report_label(args.operator_execution_json, operator_execution, operator_execution_error),
        "After running a ready action, refresh validate_eagle3_operator_execution.py before interpreting analyzer output.",
        latest_by_action=(operator_execution or {}).get("latest_by_action"),
        record_count=len((operator_execution or {}).get("records") or []),
    )

    completion_pass = status_of(completion, completion_error) == "pass"
    add(
        rows,
        "completion",
        "Final completion audit passes",
        "proven" if completion_pass else ("incomplete" if completion else "missing"),
        "eagle3_completion_audit.json PASS across all required final artifacts and RL validation.",
        report_label(args.completion_audit_json, completion, completion_error),
        "Keep this goal active until completion audit passes with trained artifacts and sweep evidence.",
        counts=(completion or {}).get("counts"),
    )

    return rows


def overall_status(rows: list[Requirement]) -> str:
    if any(row.status == "fail" for row in rows):
        return "fail"
    final_required = [
        "Remote execution host and Hayate path probe is recorded",
        "Qwen3 verifier static inputs are materialized and pipeline-ready",
        "Hayate/Hiso Eagle3 workflow has been analyzed and classified",
        "Hayate/Hiso SpecForge Qwen3 reference has been compared against current Qwen3-235B Thinking config",
        "Existing Hayate draft artifacts have been inventoried or access limitations are recorded",
        "ModelOpt path and Qwen3 loss-mask patch are known",
        "Training corpus source is actual Qwen3 SWE/RL rollout data",
        "Qwen3 Eagle3 training path manifest is defined",
        "Rollout capture has produced pipeline-ready conversations",
        "Selected Slurm container has passed ModelOpt preflight",
        "Source-built vLLM runtime passes native ABI probe",
        "ModelOpt Eagle3 recipe overrides match the Qwen3-235B Thinking architecture",
        "Pipeline dry-run manifest is validated",
        "Hidden-state/train/export submit preflight is ready",
        "Pipeline gated submit helper has executed",
        "Slurm hidden-state dump, train, and export pipeline has passed",
        "Verifier hidden states and answer loss masks are validated",
        "ModelOpt Eagle3 training checkpoint is valid",
        "HF and vLLM Eagle3 draft export artifacts are valid",
        "Trained draft is validated inside the RL/vLLM generation loop",
        "End-to-end Eagle3 artifact flow contract is complete",
        "Final completion audit passes",
    ]
    missing_final = [row for row in rows if row.requirement in final_required and row.status != "proven"]
    if missing_final:
        return "incomplete"
    return "pass"


def status_counts(rows: list[Requirement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def payload_from_rows(args: argparse.Namespace, rows: list[Requirement]) -> dict[str, Any]:
    proven = [row.requirement for row in rows if row.status == "proven"]
    missing = [row.requirement for row in rows if row.status in {"missing", "incomplete", "fail"}]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "artifact_root": str(args.artifact_root),
        "objective": "Create an Eagle3-based draft model training path for Qwen3-235B using ModelOpt and Hayate references",
        "overall_status": overall_status(rows),
        "counts": status_counts(rows),
        "draft_model_trained": all(
            any(row.requirement == requirement and row.status == "proven" for row in rows)
            for requirement in [
                "ModelOpt Eagle3 training checkpoint is valid",
                "HF and vLLM Eagle3 draft export artifacts are valid",
                "Trained draft is validated inside the RL/vLLM generation loop",
                "End-to-end Eagle3 artifact flow contract is complete",
            ]
        ),
        "current_ready_actions": next((row.evidence.get("ready_actions") for row in rows if row.requirement == "Safe next-action plan and operator sheet exist"), []),
        "proven_requirements": proven,
        "open_requirements": missing,
        "requirements": [
            {
                "order": row.order,
                "area": row.area,
                "requirement": row.requirement,
                "status": row.status,
                "proof_required": row.proof_required,
                "current_evidence": row.current_evidence,
                "next_step": row.next_step,
                "evidence": row.evidence,
            }
            for row in rows
        ],
    }


def md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "/").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-235B Eagle3 Goal Evidence Matrix",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Draft model trained and RL-validated: **{str(payload['draft_model_trained']).lower()}**",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "## Current Summary",
        "",
        "This report is no-submit. It records whether each requirement for a usable Qwen3-235B Eagle3 draft is proven by current artifacts.",
        "",
        f"Ready actions: `{', '.join(payload.get('current_ready_actions') or []) or '-'}`",
        "",
        "## Requirement Matrix",
        "",
        "| order | area | requirement | status | evidence | next step |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in payload["requirements"]:
        lines.append(
            f"| {row['order']} | {md_escape(row['area'])} | {md_escape(row['requirement'])} | "
            f"{md_escape(row['status']).upper()} | {md_escape(row['current_evidence'])} | {md_escape(row['next_step'])} |"
        )

    lines += [
        "",
        "## Proof Required",
        "",
        "| requirement | proof required |",
        "| --- | --- |",
    ]
    for row in payload["requirements"]:
        lines.append(f"| {md_escape(row['requirement'])} | {md_escape(row['proof_required'])} |")

    open_items = payload.get("open_requirements") or []
    lines += ["", "## Open Requirements", ""]
    if open_items:
        for item in open_items:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = with_defaults(parse_args())
    rows = build_requirements(args)
    payload = payload_from_rows(args, rows)
    markdown = render_markdown(payload)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")

    if args.fail_if_complete_missing and payload["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
