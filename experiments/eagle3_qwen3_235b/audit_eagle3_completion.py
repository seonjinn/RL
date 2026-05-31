#!/usr/bin/env python3
"""Final evidence audit for a trained Qwen3-235B Eagle3 draft path.

This script does not submit jobs or load model weights. It aggregates the
machine-readable reports produced by the operator guards, Megatron compatibility
probe follow-up, pipeline, hidden-state validator, config comparer, and
trained-draft sweep so the team can tell whether the path has actually reached
a usable Qwen3 Eagle3 draft artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_ARTIFACT_ROOT = ROOT / "outputs" / "qwen3_235b_eagle3"
EXPECTED_ARTIFACT_FLOW = [
    "rollout_conversation_corpus",
    "verifier_hidden_states",
    "modelopt_checkpoint",
    "hf_eagle3_export",
    "vllm_eagle3_draft",
    "rl_vllm_draft_validation",
]


@dataclass
class Check:
    area: str
    name: str
    status: str
    required: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--input-discovery-json", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--cluster-probe-json", type=Path)
    parser.add_argument("--pipeline-analysis-json", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--remote-host-probe-json", type=Path)
    parser.add_argument("--hayate-workflow-json", type=Path)
    parser.add_argument("--hayate-specforge-reference-json", type=Path)
    parser.add_argument("--upstream-drift-json", type=Path)
    parser.add_argument("--modelopt-recipe-overrides-json", type=Path)
    parser.add_argument("--training-path-manifest-json", type=Path)
    parser.add_argument("--training-path-manifest-validation-json", type=Path)
    parser.add_argument("--modelopt-patch-manifest", type=Path)
    parser.add_argument("--next-action-plan-json", type=Path)
    parser.add_argument("--next-action-plan-validation-json", type=Path)
    parser.add_argument("--operator-queue-transitions-json", type=Path)
    parser.add_argument("--operator-followup-validation-json", type=Path)
    parser.add_argument("--megatron-probe-followup-validation-json", type=Path)
    parser.add_argument("--preflight-robustness-validation-json", type=Path)
    parser.add_argument("--operator-submit-packet-validation-json", type=Path)
    parser.add_argument("--operator-ready-submit-preflight-json", type=Path)
    parser.add_argument("--operator-queue-json", type=Path)
    parser.add_argument("--completion-contract-json", type=Path)
    parser.add_argument("--slurm-capacity-json", type=Path)
    parser.add_argument("--resource-profile-application-json", type=Path)
    parser.add_argument("--rollout-queue-wait-json", type=Path)
    parser.add_argument("--rollout-watcher-health-json", type=Path)
    parser.add_argument("--container-preflight-json", type=Path)
    parser.add_argument("--vllm-source-build-json", type=Path)
    parser.add_argument("--vllm-abi-probe-json", type=Path)
    parser.add_argument("--vllm-source-job-file", type=Path)
    parser.add_argument("--rollout-state-json", type=Path)
    parser.add_argument("--corpus-strategy-json", type=Path)
    parser.add_argument("--pipeline-submit-preflight-json", type=Path)
    parser.add_argument("--pipeline-gated-submit-json", type=Path)
    parser.add_argument("--hidden-validation-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-checkpoint-json", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--vllm-draft-dir", type=Path)
    parser.add_argument("--export-artifacts-json", type=Path)
    parser.add_argument("--export-config-compare-json", type=Path)
    parser.add_argument("--vllm-config-compare-json", type=Path)
    parser.add_argument("--sweep-json", type=Path)
    parser.add_argument("--draft-inventory-json", type=Path)
    parser.add_argument("--hayate-inventory", type=Path)
    parser.add_argument(
        "--reference-arch",
        type=Path,
        default=EXP / "qwen3_235b_thinking_eagle3_architecture.json",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-if-not-pass", action="store_true")
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    artifact = args.artifact_root
    if args.input_discovery_json is None:
        args.input_discovery_json = artifact / "eagle3_input_discovery.json"
    if args.readiness_json is None:
        args.readiness_json = artifact / "reports" / "eagle3_readiness.json"
    if args.cluster_probe_json is None:
        args.cluster_probe_json = artifact / "reports" / "cluster_environment_probe.json"
    if args.pipeline_analysis_json is None:
        args.pipeline_analysis_json = artifact / "reports" / "eagle3_pipeline_analysis.json"
    if args.provenance_json is None:
        args.provenance_json = artifact / "reports" / "eagle3_provenance.json"
    if args.remote_host_probe_json is None:
        args.remote_host_probe_json = artifact / "reports" / "eagle3_remote_host_probe.json"
    if args.hayate_workflow_json is None:
        args.hayate_workflow_json = artifact / "reports" / "hayate_modelopt_workflow.json"
    if args.hayate_specforge_reference_json is None:
        args.hayate_specforge_reference_json = artifact / "reports" / "hayate_specforge_reference.json"
    if args.upstream_drift_json is None:
        args.upstream_drift_json = artifact / "reports" / "modelopt_upstream_drift.json"
    if args.modelopt_recipe_overrides_json is None:
        args.modelopt_recipe_overrides_json = artifact / "reports" / "modelopt_recipe_overrides_current.json"
    if args.training_path_manifest_json is None:
        args.training_path_manifest_json = artifact / "reports" / "eagle3_training_path_manifest.json"
    if args.training_path_manifest_validation_json is None:
        args.training_path_manifest_validation_json = artifact / "reports" / "eagle3_training_path_manifest_validation.json"
    if args.modelopt_patch_manifest is None:
        args.modelopt_patch_manifest = artifact / "patches" / "modelopt_eagle3_qwen3" / "manifest.json"
    if args.next_action_plan_json is None:
        args.next_action_plan_json = artifact / "reports" / "eagle3_next_actions.json"
    if args.next_action_plan_validation_json is None:
        args.next_action_plan_validation_json = artifact / "reports" / "eagle3_next_actions_validation.json"
    if args.operator_queue_transitions_json is None:
        args.operator_queue_transitions_json = artifact / "reports" / "eagle3_operator_queue_transitions.json"
    if args.operator_followup_validation_json is None:
        args.operator_followup_validation_json = artifact / "reports" / "eagle3_operator_followups_validation.json"
    if args.megatron_probe_followup_validation_json is None:
        args.megatron_probe_followup_validation_json = artifact / "reports" / "megatron_probe_followup_validation.json"
    if args.preflight_robustness_validation_json is None:
        args.preflight_robustness_validation_json = artifact / "reports" / "eagle3_preflight_robustness_validation.json"
    if args.operator_submit_packet_validation_json is None:
        args.operator_submit_packet_validation_json = artifact / "reports" / "eagle3_operator_submit_packet_validation.json"
    if args.operator_ready_submit_preflight_json is None:
        args.operator_ready_submit_preflight_json = artifact / "reports" / "eagle3_operator_ready_submit_preflight.json"
    if args.operator_queue_json is None:
        args.operator_queue_json = artifact / "reports" / "eagle3_operator_queue.json"
    if args.completion_contract_json is None:
        args.completion_contract_json = artifact / "reports" / "eagle3_completion_contract.json"
    if args.slurm_capacity_json is None:
        args.slurm_capacity_json = artifact / "reports" / "eagle3_slurm_capacity.json"
    if args.resource_profile_application_json is None:
        args.resource_profile_application_json = artifact / "reports" / "eagle3_resource_profile_application.json"
    if args.rollout_queue_wait_json is None:
        args.rollout_queue_wait_json = artifact / "reports" / "rollout_queue_wait_summary.json"
    if args.rollout_watcher_health_json is None:
        args.rollout_watcher_health_json = artifact / "reports" / "rollout_watcher_health.json"
    if args.container_preflight_json is None:
        args.container_preflight_json = artifact / "reports" / "container_preflight_analysis.json"
    if args.vllm_source_build_json is None:
        args.vllm_source_build_json = artifact / "reports" / "vllm_native_source_build.json"
    if args.vllm_abi_probe_json is None:
        args.vllm_abi_probe_json = artifact / "reports" / "vllm_native_abi_probe.json"
    if args.vllm_source_job_file is None:
        args.vllm_source_job_file = ROOT / "latest_vllm_native_source_build_job.txt"
    if args.rollout_state_json is None:
        args.rollout_state_json = select_rollout_state_report(artifact)
    if args.corpus_strategy_json is None:
        args.corpus_strategy_json = artifact / "reports" / "corpus_strategy.json"
    if args.pipeline_submit_preflight_json is None:
        args.pipeline_submit_preflight_json = artifact / "reports" / "eagle3_pipeline_submit_preflight.json"
    if args.pipeline_gated_submit_json is None:
        args.pipeline_gated_submit_json = artifact / "reports" / "eagle3_pipeline_gated_submit.json"
    if args.hidden_validation_json is None:
        args.hidden_validation_json = artifact / "hidden_states" / "validation_summary.json"
    if args.output_dir is None:
        args.output_dir = artifact / "modelopt_ckpt"
    if args.training_checkpoint_json is None:
        args.training_checkpoint_json = artifact / "reports" / "eagle3_training_checkpoint.json"
    if args.export_dir is None:
        args.export_dir = artifact / "exported_hf"
    if args.vllm_draft_dir is None:
        args.vllm_draft_dir = artifact / "vllm_draft"
    if args.export_artifacts_json is None:
        args.export_artifacts_json = artifact / "reports" / "eagle3_export_artifacts.json"
    if args.export_config_compare_json is None:
        args.export_config_compare_json = args.export_dir / "config_compare.json"
    if args.vllm_config_compare_json is None:
        args.vllm_config_compare_json = args.vllm_draft_dir / "config_compare.json"
    if args.sweep_json is None:
        args.sweep_json = artifact / "reports" / "trained_draft_spec_tokens_sweep.json"
    if args.draft_inventory_json is None:
        args.draft_inventory_json = artifact / "reports" / "eagle3_draft_config_inventory.json"
    if args.hayate_inventory is None:
        args.hayate_inventory = artifact / "reports" / "hayate_inventory.txt"
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
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid json: {exc}"


def read_job_env(path_value: Any) -> tuple[dict[str, str], str | None]:
    if not path_value:
        return {}, "not provided"
    path = Path(str(path_value))
    if not path.exists():
        return {}, f"not visible: {path}"
    jobs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            jobs[key] = value
    return jobs, None


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


def path_under_artifact_root(args: argparse.Namespace, value: str | None) -> bool:
    if not value:
        return False
    try:
        return Path(value).resolve(strict=False).is_relative_to(args.artifact_root.resolve(strict=False))
    except Exception:
        return str(value).startswith(str(args.artifact_root))


def add(
    checks: list[Check],
    area: str,
    name: str,
    status: str,
    detail: str,
    *,
    required: bool = True,
    **evidence: Any,
) -> None:
    checks.append(Check(area=area, name=name, status=status, required=required, detail=detail, evidence=evidence))


def artifact_flow_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        item
        for item in (payload or {}).get("artifact_flow") or []
        if isinstance(item, dict)
    ]


def artifact_flow_ids(payload: dict[str, Any] | None) -> list[str]:
    return [str(item.get("id")) for item in artifact_flow_rows(payload) if item.get("id")]


def artifact_flow_complete(payload: dict[str, Any] | None) -> bool:
    rows = artifact_flow_rows(payload)
    return (
        bool(rows)
        and artifact_flow_ids(payload) == EXPECTED_ARTIFACT_FLOW
        and (payload or {}).get("artifact_flow_complete") is True
        and all(item.get("proof_status") == "pass" for item in rows)
    )


def get_nested(payload: dict[str, Any] | None, keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def check_architecture(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.reference_arch)
    if error:
        add(checks, "config", "Qwen3 Eagle3 architecture reference", "missing", error, path=str(args.reference_arch))
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
        add(checks, "config", "Qwen3 Eagle3 architecture reference", "fail", "reference does not match Qwen3-235B Thinking-2507 defaults", mismatches=mismatches)
        return
    add(checks, "config", "Qwen3 Eagle3 architecture reference", "pass", "reference architecture matches expected Qwen3-235B Eagle3 fields", path=str(args.reference_arch))


def check_input_discovery(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.input_discovery_json)
    if error:
        add(checks, "reference", "cluster input discovery", "warn", error, required=False, path=str(args.input_discovery_json))
        return
    verifier = payload.get("verifier_candidates") or []
    conversations = payload.get("conversation_candidates") or []
    drafts = payload.get("draft_candidates") or []
    status = "pass" if verifier and conversations else "warn"
    detail = "discovered verifier and conversation candidates" if status == "pass" else "discovery report exists but key candidates are incomplete"
    add(
        checks,
        "reference",
        "cluster input discovery",
        status,
        detail,
        required=False,
        verifier_candidates=len(verifier),
        conversation_candidates=len(conversations),
        draft_candidates=len(drafts),
        files_scanned=payload.get("files_scanned"),
    )


def check_readiness(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.readiness_json)
    if error:
        add(checks, "planning", "readiness audit report", "warn", error, required=False, path=str(args.readiness_json))
        return
    status = payload.get("overall_status")
    if status == "fail":
        add(
            checks,
            "planning",
            "readiness audit report",
            "warn",
            "readiness audit currently reports failed checks; required gates below remain authoritative",
            required=False,
            counts=payload.get("counts"),
        )
    elif status == "pass":
        add(checks, "planning", "readiness audit report", "pass", "readiness audit passed", required=False, counts=payload.get("counts"))
    else:
        add(checks, "planning", "readiness audit report", "warn", f"readiness audit is {status!r}", required=False, counts=payload.get("counts"))


def check_cluster_probe(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.cluster_probe_json)
    if error:
        add(checks, "planning", "cluster environment probe", "warn", error, required=False, path=str(args.cluster_probe_json))
        return
    status = payload.get("overall_status")
    if status == "fail":
        check_status = "warn"
        detail = "cluster substrate probe has required failures on this host; Slurm execution gates below remain authoritative"
    elif status == "pass":
        check_status = "pass"
        detail = "cluster substrate probe passed"
    else:
        check_status = "warn"
        detail = f"cluster substrate probe is {status!r}"
    add(
        checks,
        "planning",
        "cluster environment probe",
        check_status,
        detail,
        required=False,
        host=(payload.get("host") or {}).get("hostname"),
        inputs=payload.get("inputs"),
    )


def check_provenance(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.provenance_json)
    if error:
        add(checks, "reference", "provenance capture", "warn", error, required=False, path=str(args.provenance_json))
        return
    repos = payload.get("repos") or []
    critical_files = payload.get("critical_files") or []
    missing_critical = [item.get("path") for item in critical_files if not item.get("exists")]
    local_modelopt = next((item for item in repos if item.get("label") == "local_modelopt"), {})
    if missing_critical:
        add(
            checks,
            "reference",
            "provenance capture",
            "warn",
            "provenance exists but some critical files were missing",
            required=False,
            missing_critical=missing_critical[:20],
            repo_count=len(repos),
        )
        return
    add(
        checks,
        "reference",
        "provenance capture",
        "pass",
        "repo state and critical file hashes were captured",
        required=False,
        repo_count=len(repos),
        critical_file_count=len(critical_files),
        local_modelopt_head=str(local_modelopt.get("head") or "")[:12],
    )


def check_upstream_drift(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.upstream_drift_json)
    if error:
        add(
            checks,
            "reference",
            "ModelOpt upstream drift report",
            "warn",
            error,
            required=False,
            path=str(args.upstream_drift_json),
        )
        return
    status = payload.get("overall_status")
    upstream = payload.get("upstream_probe") or {}
    local = payload.get("local") or {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    decision_status = decision.get("overall_status")
    training_source_current = decision_status == "pass"
    detail = (
        "ModelOpt drift report captured local/upstream/Hayate state; training source is current with allowed focus patch"
        if training_source_current
        else f"ModelOpt drift report is {status!r}, training-source decision is {decision_status!r}; inspect before final handoff"
    )
    add(
        checks,
        "reference",
        "ModelOpt upstream drift report",
        "pass" if training_source_current else ("pass" if status == "pass" else "warn"),
        detail,
        required=False,
        local_head=str(local.get("head") or "")[:12],
        upstream_head=str(upstream.get("head") or "")[:12],
        training_source_decision=decision_status,
        allowed_focus_diffs=decision.get("allowed_focus_diffs") or [],
        disallowed_focus_diffs=decision.get("disallowed_focus_diffs") or [],
        unrelated_dirty_file_count=decision.get("unrelated_dirty_file_count"),
        notes=payload.get("notes") or [],
    )


def check_modelopt_patch(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.modelopt_patch_manifest)
    if error:
        add(
            checks,
            "reference",
            "ModelOpt Eagle3 patch bundle",
            "warn",
            error,
            required=False,
            path=str(args.modelopt_patch_manifest),
        )
        return
    status = payload.get("overall_status")
    patch_nonempty = payload.get("patch_nonempty")
    if status == "pass" and patch_nonempty:
        check_status = "pass"
        detail = "re-applyable ModelOpt Eagle3 patch bundle was captured"
    else:
        check_status = "warn"
        detail = f"ModelOpt patch bundle is {status!r}; inspect before updating ModelOpt"
    add(
        checks,
        "reference",
        "ModelOpt Eagle3 patch bundle",
        check_status,
        detail,
        required=False,
        patch_sha256=payload.get("patch_sha256"),
        patch_paths=payload.get("patch_paths"),
        compatibility_checks=payload.get("compatibility_checks") or [],
        local_head=str(payload.get("local_head") or "")[:12],
    )


def check_remote_reference_probe(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.remote_host_probe_json)
    if error:
        add(
            checks,
            "reference_gate",
            "remote ModelOpt/Hayate host probe",
            "missing",
            error,
            path=str(args.remote_host_probe_json),
        )
        return
    status = json_status(payload)
    reachable_hosts = payload.get("reachable_hosts") or []
    reachable_records = [
        host
        for host in payload.get("hosts") or []
        if isinstance(host, dict) and host.get("reachable")
    ]
    path_records = [
        item
        for host in reachable_records
        for item in host.get("paths") or []
        if isinstance(item, dict)
    ]
    commands = {
        name
        for host in reachable_records
        for name, value in ((host.get("commands") or {}).items() if isinstance(host.get("commands"), dict) else [])
        if value
    }
    visible_paths = [
        str(item.get("path") or "")
        for item in path_records
        if item.get("exists") and item.get("readable")
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
    remote_workdir = str(payload.get("remote_workdir") or "")
    artifact_root = str(payload.get("artifact_root") or "")
    remote_workdir_ok = not remote_workdir or remote_workdir in visible_paths
    artifact_root_ok = not artifact_root or artifact_root in visible_paths
    missing_capabilities = []
    for command in ("git", "python3"):
        if command not in commands:
            missing_capabilities.append(f"command:{command}")
    if not hayate_modelopt_paths:
        missing_capabilities.append("hayate_modelopt_path")
    if not hayate_draft_paths:
        missing_capabilities.append("hayate_draft_or_specforge_path")
    if not remote_workdir_ok:
        missing_capabilities.append("remote_workdir_path")
    if not artifact_root_ok:
        missing_capabilities.append("artifact_root_path")
    if status == "pass" and reachable_hosts and not missing_capabilities:
        add(
            checks,
            "reference_gate",
            "remote ModelOpt/Hayate host probe",
            "pass",
            "remote execution host is reachable and required ModelOpt/Hayate/artifact paths are visible",
            reachable_hosts=reachable_hosts,
            counts=payload.get("counts"),
            hayate_modelopt_paths=hayate_modelopt_paths,
            hayate_draft_paths=hayate_draft_paths,
            remote_workdir=remote_workdir,
            artifact_root=artifact_root,
        )
    elif status == "unreachable":
        add(
            checks,
            "reference_gate",
            "remote ModelOpt/Hayate host probe",
            "incomplete",
            "remote host aliases are currently unreachable; remote ModelOpt/Hayate path evidence is not proven",
            reachable_hosts=reachable_hosts,
            counts=payload.get("counts"),
        )
    elif status == "pass" and reachable_hosts:
        add(
            checks,
            "reference_gate",
            "remote ModelOpt/Hayate host probe",
            "incomplete",
            "remote host is reachable but required ModelOpt/Hayate/artifact path evidence is incomplete",
            reachable_hosts=reachable_hosts,
            counts=payload.get("counts"),
            missing_capabilities=missing_capabilities,
            hayate_modelopt_paths=hayate_modelopt_paths,
            hayate_draft_paths=hayate_draft_paths,
            remote_workdir=remote_workdir,
            artifact_root=artifact_root,
        )
    else:
        add(
            checks,
            "reference_gate",
            "remote ModelOpt/Hayate host probe",
            "fail" if status == "fail" else "incomplete",
            f"remote host probe status is {status!r}",
            reachable_hosts=reachable_hosts,
            counts=payload.get("counts"),
        )


def check_hayate_workflow(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.hayate_workflow_json)
    if error:
        add(
            checks,
            "reference_gate",
            "Hayate ModelOpt workflow analysis",
            "missing",
            error,
            path=str(args.hayate_workflow_json),
        )
        return
    status = json_status(payload)
    classification = get_nested(payload, ["classification", "classification"]) or payload.get("classification")
    configs = payload.get("qwen_configs") or []
    if status in {"reference_only", "needs_review", "pass", "warn"} and (classification or configs):
        add(
            checks,
            "reference_gate",
            "Hayate ModelOpt workflow analysis",
            "pass",
            "Hayate/Hiso ModelOpt Eagle3 workflow was analyzed and classified as reference input",
            hayate_status=status,
            classification=classification,
            qwen_config_count=len(configs),
            hayate_modelopt_dir=str(payload.get("hayate_modelopt_dir") or payload.get("root") or ""),
        )
    elif status == "fail":
        add(
            checks,
            "reference_gate",
            "Hayate ModelOpt workflow analysis",
            "fail",
            "Hayate/Hiso ModelOpt workflow analysis failed",
            hayate_status=status,
            classification=classification,
        )
    else:
        add(
            checks,
            "reference_gate",
            "Hayate ModelOpt workflow analysis",
            "incomplete",
            "Hayate/Hiso ModelOpt workflow report exists but does not prove an accessible classified reference workflow",
            hayate_status=status,
            classification=classification,
        )


def check_hayate_specforge_reference(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.hayate_specforge_reference_json)
    if error:
        add(
            checks,
            "reference_gate",
            "Hayate SpecForge Qwen3 reference comparison",
            "missing",
            error,
            path=str(args.hayate_specforge_reference_json),
        )
        return
    status = json_status(payload)
    comparison = payload.get("qwen3_235b_comparison") if isinstance(payload.get("qwen3_235b_comparison"), dict) else {}
    rows = [row for row in comparison.get("rows") or [] if isinstance(row, dict)]
    if status in {"reference_only", "matches_current"} and rows:
        add(
            checks,
            "reference_gate",
            "Hayate SpecForge Qwen3 reference comparison",
            "pass",
            "SpecForge/Hayate Qwen3 reference was compared against the current Qwen3-235B Thinking architecture",
            specforge_status=status,
            matched_fields=[row.get("field") for row in rows if row.get("match") is True],
            mismatched_fields=[row.get("field") for row in rows if row.get("match") is False],
            conclusion=comparison.get("conclusion"),
        )
    elif status == "fail":
        add(
            checks,
            "reference_gate",
            "Hayate SpecForge Qwen3 reference comparison",
            "fail",
            "SpecForge/Hayate Qwen3 reference analysis failed",
            specforge_status=status,
        )
    else:
        add(
            checks,
            "reference_gate",
            "Hayate SpecForge Qwen3 reference comparison",
            "incomplete",
            "SpecForge/Hayate Qwen3 comparison report is missing required comparison rows",
            specforge_status=status,
            comparison_status=comparison.get("status"),
        )


def check_modelopt_recipe_overrides(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.modelopt_recipe_overrides_json)
    if error:
        add(
            checks,
            "training_gate",
            "ModelOpt recipe override validation",
            "missing",
            error,
            path=str(args.modelopt_recipe_overrides_json),
        )
        return
    status = payload.get("overall_status")
    mode = payload.get("training_mode")
    arch = payload.get("architecture_overrides") if isinstance(payload.get("architecture_overrides"), dict) else {}
    required_arch = {
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "use_aux_hidden_state": True,
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": True,
        "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
    }
    mismatches = {
        key: {"actual": arch.get(key), "expected": value}
        for key, value in required_arch.items()
        if arch.get(key) != value
    }
    if status == "pass" and mode == "offline" and not mismatches:
        add(
            checks,
            "training_gate",
            "ModelOpt recipe override validation",
            "pass",
            "offline training wrapper emits current ModelOpt recipe overrides for the Qwen3-235B Thinking Eagle3 architecture",
            wrapper=payload.get("wrapper"),
            recipe_config=payload.get("recipe_config"),
            override_count=payload.get("override_count"),
            counts=payload.get("counts"),
            warnings=payload.get("warnings") or [],
        )
    elif status == "fail" or mismatches:
        add(
            checks,
            "training_gate",
            "ModelOpt recipe override validation",
            "fail",
            "ModelOpt recipe override validation failed or no longer matches the Qwen3-235B Thinking Eagle3 architecture",
            overall_status=status,
            training_mode=mode,
            mismatches=mismatches,
            failures=payload.get("failures") or [],
        )
    else:
        add(
            checks,
            "training_gate",
            "ModelOpt recipe override validation",
            "incomplete",
            "ModelOpt recipe override validation exists but does not prove the offline training wrapper contract",
            overall_status=status,
            training_mode=mode,
            counts=payload.get("counts"),
        )


def check_training_path_manifest(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.training_path_manifest_json)
    validation, validation_error = load_json(args.training_path_manifest_validation_json)
    if error:
        add(
            checks,
            "training_path",
            "Qwen3 Eagle3 training path manifest",
            "warn",
            error,
            required=True,
            path=str(args.training_path_manifest_json),
        )
        return
    validation_status = None if validation_error else validation.get("overall_status")
    status = payload.get("overall_status")
    path_defined = bool(payload.get("path_defined"))
    final_artifacts_complete = payload.get("final_artifacts_complete") is True
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    required_gate_ids = {
        "reference_and_architecture",
        "remote_hayate_reference_probe",
        "modelopt_loss_and_recipe",
        "target_rollout_corpus",
        "runtime_container",
        "hidden_train_export_submit",
        "trained_artifact_contracts",
    }
    present_gate_ids = {str(item.get("id")) for item in gates if isinstance(item, dict)}
    missing_gate_ids = sorted(required_gate_ids - present_gate_ids)
    closure_contracts = {
        str(item.get("id")): item
        for item in payload.get("gate_closure_contracts") or []
        if isinstance(item, dict) and item.get("id")
    }
    target_report_labels = {
        str(item.get("label"))
        for item in (closure_contracts.get("target_rollout_corpus") or {}).get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    runtime_report_labels = {
        str(item.get("label"))
        for item in (closure_contracts.get("runtime_container") or {}).get("required_reports") or []
        if isinstance(item, dict) and item.get("label")
    }
    gate_closure_contracts_ok = (
        set(closure_contracts) == required_gate_ids
        and {"rollout_state", "corpus_strategy"}.issubset(target_report_labels)
        and {"container_preflight", "vllm_source_build", "vllm_abi_probe", "megatron_compat"}.issubset(
            runtime_report_labels
        )
    )
    reference_evidence = payload.get("reference_evidence") if isinstance(payload.get("reference_evidence"), dict) else {}
    open_gates = payload.get("open_gates") if isinstance(payload.get("open_gates"), list) else []
    remote_reference_proven = reference_evidence.get("remote_reference_proven")
    remote_reference_gate_state_ok = (
        remote_reference_proven is True
        or (remote_reference_proven is False and "remote_hayate_reference_probe" in open_gates)
    )
    reference_evidence_contract_ok = (
        isinstance(remote_reference_proven, bool)
        and remote_reference_gate_state_ok
        and isinstance(reference_evidence.get("local_modelopt"), dict)
        and isinstance(reference_evidence.get("remote_probe"), dict)
        and isinstance(reference_evidence.get("hayate_modelopt"), dict)
        and isinstance(reference_evidence.get("hayate_specforge"), dict)
    )
    reference_decisions = (
        payload.get("reference_decisions") if isinstance(payload.get("reference_decisions"), dict) else {}
    )
    training_route = (
        reference_decisions.get("training_route")
        if isinstance(reference_decisions.get("training_route"), dict)
        else {}
    )
    modelopt_source = (
        reference_decisions.get("modelopt_source")
        if isinstance(reference_decisions.get("modelopt_source"), dict)
        else {}
    )
    specforge_reference = (
        reference_decisions.get("specforge_qwen3_235b")
        if isinstance(reference_decisions.get("specforge_qwen3_235b"), dict)
        else {}
    )
    hayate_workflow_reference = (
        reference_decisions.get("hayate_workflow")
        if isinstance(reference_decisions.get("hayate_workflow"), dict)
        else {}
    )
    matched_fields = set(specforge_reference.get("matched_fields") or [])
    rejected_fields = {
        str(item.get("field"))
        for item in specforge_reference.get("rejected_fields") or []
        if isinstance(item, dict) and item.get("field")
    }
    reference_decisions_contract_ok = (
        training_route.get("primary_route") == "fixed_exported_eagle3_draft_first"
        and modelopt_source.get("source_of_truth") == "local_modelopt"
        and modelopt_source.get("upstream_drift_status") in {"pass", "warn"}
        and {"aux_layers", "hidden_size"}.issubset(matched_fields)
        and bool(rejected_fields)
        and hayate_workflow_reference.get("role") == "reference_only"
    )
    artifact_flow_ids_seen = artifact_flow_ids(payload)
    artifact_flow_ok = artifact_flow_complete(payload)
    if (
        path_defined
        and status == "pass"
        and validation_status == "pass"
        and not missing_gate_ids
        and not open_gates
        and final_artifacts_complete
        and artifact_flow_ok
        and reference_evidence_contract_ok
        and reference_decisions_contract_ok
        and gate_closure_contracts_ok
    ):
        add(
            checks,
            "training_path",
            "Qwen3 Eagle3 training path manifest",
            "pass",
            "training path manifest proves the fixed exported Eagle3 route, closed gates, reference roles, and end-to-end artifact flow",
            required=True,
            primary_route=payload.get("primary_route"),
            validation_status=validation_status,
            open_gates=open_gates,
            ready_actions=payload.get("ready_actions"),
            final_artifacts_complete=payload.get("final_artifacts_complete"),
            artifact_flow_complete=payload.get("artifact_flow_complete"),
            artifact_flow_ids=artifact_flow_ids_seen,
            remote_reference_proven=remote_reference_proven,
            reference_evidence_contract_ok=reference_evidence_contract_ok,
            reference_decisions_contract_ok=reference_decisions_contract_ok,
            gate_closure_contracts_ok=gate_closure_contracts_ok,
            artifact_flow_ok=artifact_flow_ok,
            reference_evidence=reference_evidence,
            reference_decisions=reference_decisions,
            gate_closure_contracts=closure_contracts,
            artifact_flow=artifact_flow_rows(payload),
        )
    else:
        add(
            checks,
            "training_path",
            "Qwen3 Eagle3 training path manifest",
            "incomplete",
            "training path manifest exists but does not prove all gates and the end-to-end artifact flow",
            required=True,
            overall_status=status,
            validation_status=validation_status,
            validation_error=validation_error,
            path_defined=path_defined,
            missing_gate_ids=missing_gate_ids,
            open_gates=open_gates,
            final_artifacts_complete=payload.get("final_artifacts_complete"),
            artifact_flow_complete=payload.get("artifact_flow_complete"),
            artifact_flow_ids=artifact_flow_ids_seen,
            open_artifact_flow=[
                {
                    "id": item.get("id"),
                    "proof_status": item.get("proof_status"),
                    "required_reports": item.get("required_reports"),
                    "report_statuses": item.get("report_statuses"),
                }
                for item in artifact_flow_rows(payload)
                if item.get("proof_status") != "pass"
            ],
            remote_reference_proven=remote_reference_proven,
            reference_evidence_contract_ok=reference_evidence_contract_ok,
            reference_decisions_contract_ok=reference_decisions_contract_ok,
            gate_closure_contracts_ok=gate_closure_contracts_ok,
            artifact_flow_ok=artifact_flow_ok,
        )


def check_next_action_plan(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.next_action_plan_json)
    if error:
        add(
            checks,
            "planning",
            "next-action plan",
            "warn",
            error,
            required=False,
            path=str(args.next_action_plan_json),
        )
        return
    status = payload.get("overall_status")
    actions = payload.get("next_actions") or []
    blockers = payload.get("blockers") or []
    if status == "fail":
        check_status = "fail"
        detail = "next-action plan reports a failed gate"
    elif status in {"ready_for_pipeline_submit", "pass"}:
        check_status = "pass"
        detail = "next-action plan does not block pipeline submission"
    else:
        check_status = "warn"
        detail = f"next-action plan is {status!r}; this is expected before real cluster execution"
    add(
        checks,
        "planning",
        "next-action plan",
        check_status,
        detail,
        required=False,
        actions=[
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "submits_slurm": item.get("submits_slurm"),
                "heavy_gpu": item.get("heavy_gpu"),
            }
            for item in actions[:4]
            if isinstance(item, dict)
        ],
        blockers=[
            {"id": item.get("id"), "severity": item.get("severity")}
            for item in blockers[:6]
            if isinstance(item, dict)
        ],
    )


def check_next_action_plan_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.next_action_plan_validation_json)
    if error:
        add(
            checks,
            "planning",
            "next-action plan validation",
            "warn",
            error,
            required=False,
            path=str(args.next_action_plan_validation_json),
        )
        return
    status = payload.get("overall_status")
    if status == "pass":
        add(
            checks,
            "planning",
            "next-action plan validation",
            "pass",
            "next-action plan semantic validation passed",
            required=False,
            counts=payload.get("counts"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "next-action plan validation",
            "fail",
            "next-action plan validation found unsafe or inconsistent actions",
            required=False,
            counts=payload.get("counts"),
        )
    else:
        add(
            checks,
            "planning",
            "next-action plan validation",
            "warn",
            f"next-action plan validation is {status!r}",
            required=False,
            counts=payload.get("counts"),
        )


def check_operator_queue_transitions(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.operator_queue_transitions_json)
    if error:
        add(
            checks,
            "planning",
            "operator queue transition validation",
            "warn",
            error,
            required=False,
            path=str(args.operator_queue_transitions_json),
        )
        return
    status = payload.get("overall_status")
    if status == "pass":
        add(
            checks,
            "planning",
            "operator queue transition validation",
            "pass",
            "synthetic operator queue state transitions are valid",
            required=False,
            scenarios=len(payload.get("scenarios") or []),
        )
    else:
        add(
            checks,
            "planning",
            "operator queue transition validation",
            "warn",
            f"operator queue transition validation is {status!r}",
            required=False,
            problems=payload.get("problems"),
        )


def check_operator_followup_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.operator_followup_validation_json)
    if error:
        add(
            checks,
            "planning",
            "operator Slurm follow-up validation",
            "missing",
            error,
            path=str(args.operator_followup_validation_json),
        )
        return
    status = payload.get("overall_status")
    state_counts = payload.get("followup_state_counts") or {}
    if status == "pass":
        add(
            checks,
            "planning",
            "operator Slurm follow-up validation",
            "pass",
            "Slurm follow-up guard reports preserve terminal-state safety",
            counts=payload.get("counts"),
            followup_state_counts=state_counts,
            expected_actions=payload.get("expected_actions"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "operator Slurm follow-up validation",
            "fail",
            "Slurm follow-up validation found unsafe or inconsistent guard reports",
            counts=payload.get("counts"),
            followup_state_counts=state_counts,
        )
    else:
        add(
            checks,
            "planning",
            "operator Slurm follow-up validation",
            "incomplete",
            f"Slurm follow-up validation is {status!r}",
            counts=payload.get("counts"),
            followup_state_counts=state_counts,
        )


def check_megatron_probe_followup_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.megatron_probe_followup_validation_json)
    if error:
        add(
            checks,
            "planning",
            "Megatron probe follow-up validation",
            "missing",
            error,
            path=str(args.megatron_probe_followup_validation_json),
        )
        return
    status = payload.get("overall_status")
    check_rows = payload.get("checks") or []
    counts = payload.get("counts")
    if counts is None and isinstance(check_rows, list):
        counts = {}
        for row in check_rows:
            if isinstance(row, dict):
                row_status = str(row.get("status") or "unknown")
                counts[row_status] = counts.get(row_status, 0) + 1
    if status == "pass":
        add(
            checks,
            "planning",
            "Megatron probe follow-up validation",
            "pass",
            "Megatron compatibility probe follow-up guard is fail-closed and no-submit by default",
            counts=counts,
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "Megatron probe follow-up validation",
            "fail",
            "Megatron probe follow-up validation found unsafe or inconsistent behavior",
            counts=counts,
            problems=payload.get("problems"),
        )
    else:
        add(
            checks,
            "planning",
            "Megatron probe follow-up validation",
            "incomplete",
            f"Megatron probe follow-up validation is {status!r}",
            counts=counts,
        )


def check_preflight_robustness_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.preflight_robustness_validation_json)
    if error:
        add(
            checks,
            "planning",
            "preflight robustness validation",
            "missing",
            error,
            path=str(args.preflight_robustness_validation_json),
        )
        return
    status = payload.get("overall_status")
    check_rows = payload.get("checks") or []
    counts: dict[str, int] = {}
    for row in check_rows:
        if isinstance(row, dict):
            row_status = str(row.get("status") or "unknown")
            counts[row_status] = counts.get(row_status, 0) + 1
    if status == "pass":
        add(
            checks,
            "planning",
            "preflight robustness validation",
            "pass",
            "local preflight helpers fail with structured redacted evidence on lightweight hosts",
            counts=counts,
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "preflight robustness validation",
            "fail",
            "preflight robustness validation found tracebacks, leaked values, or wrong artifact-root propagation",
            counts=counts,
            problems=payload.get("problems"),
        )
    else:
        add(
            checks,
            "planning",
            "preflight robustness validation",
            "incomplete",
            f"preflight robustness validation is {status!r}",
            counts=counts,
        )


def check_operator_submit_packet_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.operator_submit_packet_validation_json)
    if error:
        add(
            checks,
            "planning",
            "operator submit packet validation",
            "missing",
            error,
            path=str(args.operator_submit_packet_validation_json),
        )
        return
    status = payload.get("overall_status")
    if status == "pass":
        add(
            checks,
            "planning",
            "operator submit packet validation",
            "pass",
            "submit packet mirrors the operator sheet and preserves Slurm guard commands",
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
            packet_status=payload.get("packet_status"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "operator submit packet validation",
            "fail",
            "submit packet validation found stale or unsafe operator commands",
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
            packet_status=payload.get("packet_status"),
        )
    else:
        add(
            checks,
            "planning",
            "operator submit packet validation",
            "incomplete",
            f"submit packet validation is {status!r}",
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
            packet_status=payload.get("packet_status"),
        )


def check_operator_ready_submit_preflight(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.operator_ready_submit_preflight_json)
    if error:
        add(
            checks,
            "planning",
            "operator ready-submit preflight",
            "warn",
            error,
            required=False,
            path=str(args.operator_ready_submit_preflight_json),
        )
        return
    status = payload.get("overall_status")
    submit_ready = payload.get("submit_ready")
    failed_checks = [
        {
            "area": row.get("area"),
            "name": row.get("name"),
            "detail": row.get("detail"),
        }
        for row in (payload.get("checks") or [])
        if isinstance(row, dict) and row.get("status") == "fail"
    ]
    if status == "pass" and submit_ready is True:
        add(
            checks,
            "planning",
            "operator ready-submit preflight",
            "pass",
            "ready operator submit commands have visible runtime inputs and writable report paths",
            required=False,
            submit_ready=submit_ready,
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "operator ready-submit preflight",
            "warn",
            "ready operator submit preflight is not currently submit-ready; required execution gates below remain authoritative",
            required=False,
            submit_ready=submit_ready,
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
            failed_checks=failed_checks,
        )
    else:
        add(
            checks,
            "planning",
            "operator ready-submit preflight",
            "warn",
            f"operator ready-submit preflight is {status!r}",
            required=False,
            submit_ready=submit_ready,
            counts=payload.get("counts"),
            ready_actions=payload.get("ready_actions"),
        )


def check_operator_queue(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.operator_queue_json)
    if error:
        add(
            checks,
            "planning",
            "operator queue summary",
            "warn",
            error,
            required=False,
            path=str(args.operator_queue_json),
        )
        return
    status = payload.get("overall_status")
    if status in {"ready_for_operator_submit", "ready_for_followup", "waiting_for_slurm", "current_ready_set_processed", "no_ready_actions"}:
        add(
            checks,
            "planning",
            "operator queue summary",
            "pass",
            "operator queue summarizes the next concrete command for ready actions",
            required=False,
            queue_status=status,
            counts=payload.get("counts"),
            next_command_available=bool(payload.get("next_command")),
        )
    elif status == "blocked":
        add(
            checks,
            "planning",
            "operator queue summary",
            "warn",
            "operator queue reports a blocked or inspect-required action",
            required=False,
            queue_status=status,
            counts=payload.get("counts"),
        )
    else:
        add(
            checks,
            "planning",
            "operator queue summary",
            "warn",
            f"operator queue status is {status!r}",
            required=False,
            queue_status=status,
            counts=payload.get("counts"),
        )


def check_completion_contract(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.completion_contract_json)
    if error:
        add(
            checks,
            "planning",
            "completion contract self-test",
            "warn",
            error,
            required=False,
            path=str(args.completion_contract_json),
        )
        return
    status = payload.get("overall_status")
    scenarios = payload.get("scenarios") or []
    if status == "pass":
        add(
            checks,
            "planning",
            "completion contract self-test",
            "pass",
            "synthetic final artifact contracts pass and reject stale sweep evidence",
            required=False,
            scenario_count=len(scenarios),
            scenarios=[item.get("name") for item in scenarios if isinstance(item, dict)],
        )
    else:
        add(
            checks,
            "planning",
            "completion contract self-test",
            "warn",
            f"completion contract self-test is {status!r}",
            required=False,
            scenario_count=len(scenarios),
            problems=payload.get("problems"),
        )


def check_slurm_capacity(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.slurm_capacity_json)
    if error:
        add(
            checks,
            "planning",
            "Slurm capacity probe",
            "warn",
            error,
            required=False,
            path=str(args.slurm_capacity_json),
        )
        return
    status = payload.get("overall_status")
    capacity = payload.get("visible_capacity") or {}
    if status == "pass":
        add(
            checks,
            "planning",
            "Slurm capacity probe",
            "pass",
            "visible Slurm GPU shape fits current pipeline resource requests",
            required=False,
            requests=payload.get("requests"),
            max_gpu_per_node=capacity.get("max_gpu_per_node"),
            unique_gres=capacity.get("unique_gres"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "Slurm capacity probe",
            "warn",
            "visible Slurm GPU shape does not fit current pipeline resource requests",
            required=False,
            requests=payload.get("requests"),
            max_gpu_per_node=capacity.get("max_gpu_per_node"),
            recommendations=payload.get("recommendations"),
        )
    else:
        add(
            checks,
            "planning",
            "Slurm capacity probe",
            "warn",
            f"Slurm capacity probe is {status!r}",
            required=False,
            requests=payload.get("requests"),
            recommendations=payload.get("recommendations"),
        )


def check_resource_profile_application(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.resource_profile_application_json)
    if error:
        add(
            checks,
            "planning",
            "resource profile application",
            "warn",
            error,
            required=False,
            path=str(args.resource_profile_application_json),
        )
        return
    status = payload.get("overall_status")
    if status == "pass":
        add(
            checks,
            "planning",
            "resource profile application",
            "pass",
            "resource profile reaches submit_eagle3_pipeline.sh dry-run sbatch requests",
            required=False,
            profile_env=payload.get("profile_env"),
            counts=payload.get("counts"),
        )
    elif status == "fail":
        add(
            checks,
            "planning",
            "resource profile application",
            "warn",
            "resource profile did not reach the pipeline dry-run as expected",
            required=False,
            profile_env=payload.get("profile_env"),
            counts=payload.get("counts"),
        )
    else:
        add(
            checks,
            "planning",
            "resource profile application",
            "warn",
            f"resource profile application is {status!r}",
            required=False,
            profile_env=payload.get("profile_env"),
            counts=payload.get("counts"),
        )


def check_rollout_monitoring(args: argparse.Namespace, checks: list[Check]) -> None:
    queue, queue_error = load_json(args.rollout_queue_wait_json)
    health, health_error = load_json(args.rollout_watcher_health_json)

    if queue_error:
        add(
            checks,
            "planning",
            "rollout queue wait summary",
            "warn",
            queue_error,
            required=False,
            path=str(args.rollout_queue_wait_json),
        )
    else:
        queue_status = queue.get("overall_status")
        add(
            checks,
            "planning",
            "rollout queue wait summary",
            "pass" if queue_status in {"idle", "waiting", "terminal_or_unknown", "pass"} else "warn",
            f"rollout queue wait summary is {queue_status!r}",
            required=False,
            counts=queue.get("counts"),
            jobs=[
                {
                    "job_id": item.get("job_id"),
                    "state": (item.get("current_squeue") or {}).get("state"),
                    "start": (item.get("current_squeue") or {}).get("start"),
                    "sample_count": item.get("sample_count"),
                    "start_estimate_changes": item.get("start_estimate_changes"),
                }
                for item in queue.get("jobs") or []
                if isinstance(item, dict)
            ],
        )

    if health_error:
        add(
            checks,
            "planning",
            "rollout watcher health",
            "warn",
            health_error,
            required=False,
            path=str(args.rollout_watcher_health_json),
        )
    else:
        health_status = health.get("overall_status")
        add(
            checks,
            "planning",
            "rollout watcher health",
            "pass" if health_status == "pass" else "warn",
            f"rollout watcher health is {health_status!r}",
            required=False,
            dead_or_missing_required_watchers=health.get("dead_or_missing_required_watchers"),
            stale_reports=health.get("stale_reports"),
            watchers=[
                {
                    "label": item.get("label"),
                    "status": item.get("status"),
                    "pid": item.get("pid"),
                    "required_now": item.get("required_now"),
                }
                for item in health.get("watchers") or []
                if isinstance(item, dict)
            ],
        )


def check_container_preflight(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.container_preflight_json)
    if error:
        add(
            checks,
            "runtime_gate",
            "container preflight PASS",
            "missing",
            error,
            path=str(args.container_preflight_json),
        )
        return
    overall = payload.get("overall_status")
    status = payload.get("status")
    if overall == "pass" and status == "pass":
        add(
            checks,
            "runtime_gate",
            "container preflight PASS",
            "pass",
            "selected container passed ModelOpt/chat-template preflight",
            container=payload.get("container"),
            job_id=payload.get("job_id"),
            out_log=payload.get("out_log"),
            err_log=payload.get("err_log"),
        )
    elif overall == "fail" or status == "fail":
        add(
            checks,
            "runtime_gate",
            "container preflight PASS",
            "fail",
            "container preflight failed",
            overall_status=overall,
            preflight_status=status,
            preflight_detail=payload.get("detail"),
            container=payload.get("container"),
        )
    else:
        add(
            checks,
            "runtime_gate",
            "container preflight PASS",
            "incomplete",
            "container preflight has not proven the selected runtime image yet",
            overall_status=overall,
            preflight_status=status,
            preflight_detail=payload.get("detail"),
            container=payload.get("container"),
        )


def check_vllm_runtime(args: argparse.Namespace, checks: list[Check]) -> None:
    source_build, source_error = load_json(args.vllm_source_build_json)
    abi_probe, abi_error = load_json(args.vllm_abi_probe_json)
    source_job, source_job_error = read_job_env(args.vllm_source_job_file)
    source_site = source_vllm_site(args, source_build, source_job)
    source_status = None if source_error else str((source_build or {}).get("overall_status") or (source_build or {}).get("status") or "unknown")
    abi_status = None if abi_error else str((abi_probe or {}).get("overall_status") or (abi_probe or {}).get("status") or "unknown")
    abi_covers_source = abi_probe_site_passed(abi_probe, source_site)
    abi_source_failed = abi_probe_site_failed(abi_probe, source_site)
    source_job_id = source_job.get("vllm_native_source_build_job")
    runtime_relevant = bool(source_job_id or not source_error or path_under_artifact_root(args, source_job.get("output_site")))

    if source_status == "pass" and abi_status == "pass" and abi_covers_source:
        add(
            checks,
            "runtime_gate",
            "source-built vLLM native ABI PASS",
            "pass",
            "source-built vLLM site passed native import checks in the target NeMo runtime",
            source_site=source_site,
            source_build_json=str(args.vllm_source_build_json),
            abi_probe_json=str(args.vllm_abi_probe_json),
        )
    elif source_status == "fail":
        add(
            checks,
            "runtime_gate",
            "source-built vLLM native ABI PASS",
            "fail",
            "vLLM source build failed; rollout capture cannot safely proceed",
            source_build_json=str(args.vllm_source_build_json),
            source_status=source_status,
            source_site=source_site,
        )
    elif abi_source_failed:
        add(
            checks,
            "runtime_gate",
            "source-built vLLM native ABI PASS",
            "fail",
            "source-built vLLM site failed native ABI probe",
            abi_probe_json=str(args.vllm_abi_probe_json),
            source_site=source_site,
            abi_status=abi_status,
            abi_covers_source=abi_covers_source,
        )
    elif runtime_relevant:
        add(
            checks,
            "runtime_gate",
            "source-built vLLM native ABI PASS",
            "incomplete",
            "source-built vLLM site has not yet passed native ABI probe",
            source_build_json=str(args.vllm_source_build_json),
            abi_probe_json=str(args.vllm_abi_probe_json),
            source_job_file=str(args.vllm_source_job_file),
            source_job_error=source_job_error,
            source_job_id=source_job_id,
            source_status=source_status or source_error,
            abi_status=abi_status or abi_error,
            abi_covers_source=abi_covers_source,
            abi_source_failed=abi_source_failed,
            source_site=source_site,
        )
    else:
        add(
            checks,
            "runtime_gate",
            "source-built vLLM native ABI PASS",
            "incomplete",
            "vLLM source-build runtime evidence is not visible yet",
            source_build_json=str(args.vllm_source_build_json),
            abi_probe_json=str(args.vllm_abi_probe_json),
            source_job_file=str(args.vllm_source_job_file),
            source_error=source_error,
            abi_error=abi_error,
        )


def check_rollout_corpus(args: argparse.Namespace, checks: list[Check]) -> None:
    rollout, rollout_error = load_json(args.rollout_state_json)
    if rollout_error:
        add(
            checks,
            "data_gate",
            "target rollout state PASS",
            "missing",
            rollout_error,
            path=str(args.rollout_state_json),
        )
    else:
        decision = rollout.get("decision") or {}
        ok = decision.get("overall_status") == "pass" and decision.get("next_step") == "pipeline_dry_run"
        if ok:
            add(
                checks,
                "data_gate",
                "target rollout state PASS",
                "pass",
                "rollout capture state reached pipeline_dry_run",
                path=str(args.rollout_state_json),
                output_data=rollout.get("output_data"),
                rollout_log_dir=rollout.get("rollout_log_dir"),
            )
        elif decision.get("overall_status") == "fail":
            add(
                checks,
                "data_gate",
                "target rollout state PASS",
                "fail",
                "rollout capture state failed",
                path=str(args.rollout_state_json),
                decision=decision,
                output_data=rollout.get("output_data"),
            )
        else:
            add(
                checks,
                "data_gate",
                "target rollout state PASS",
                "incomplete",
                "rollout capture has not produced a pipeline-ready target corpus",
                path=str(args.rollout_state_json),
                decision=decision,
                output_data=rollout.get("output_data"),
            )

    corpus, corpus_error = load_json(args.corpus_strategy_json)
    if corpus_error:
        add(
            checks,
            "data_gate",
            "SWE/RL corpus strategy PASS",
            "missing",
            corpus_error,
            path=str(args.corpus_strategy_json),
        )
        return
    decision = corpus.get("decision") or {}
    primary_source = decision.get("primary_source")
    target_context = corpus.get("target_context")
    provenance = (
        decision.get("provenance")
        if isinstance(decision.get("provenance"), dict)
        else corpus.get("rollout_alignment")
        if isinstance(corpus.get("rollout_alignment"), dict)
        else {}
    )
    ok = (
        corpus.get("overall_status") == "pass"
        and target_context == "swe_rl"
        and primary_source == "actual_rl_rollout"
        and provenance.get("proves_actual_rollout_corpus") is True
    )
    if ok:
        add(
            checks,
            "data_gate",
            "SWE/RL corpus strategy PASS",
            "pass",
            "actual SWE/RL rollout corpus is selected as the primary Eagle3 training source",
            target_context=target_context,
            primary_source=primary_source,
            input_data=corpus.get("input_data"),
            provenance=provenance,
        )
    elif corpus.get("overall_status") == "fail":
        add(
            checks,
            "data_gate",
            "SWE/RL corpus strategy PASS",
            "fail",
            "corpus strategy failed",
            target_context=target_context,
            primary_source=primary_source,
            provenance=provenance,
            decision=decision,
        )
    else:
        add(
            checks,
            "data_gate",
            "SWE/RL corpus strategy PASS",
            "incomplete",
            "corpus strategy has not proven actual SWE/RL rollout as the primary source",
            overall_status=corpus.get("overall_status"),
            target_context=target_context,
            primary_source=primary_source,
            provenance=provenance,
            decision=decision,
        )


def check_pipeline_submit_preflight(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.pipeline_submit_preflight_json)
    if error:
        add(
            checks,
            "pipeline_gate",
            "pipeline submit preflight PASS",
            "missing",
            error,
            path=str(args.pipeline_submit_preflight_json),
        )
        return
    status = payload.get("overall_status")
    submit_ready = payload.get("submit_ready")
    if status == "pass" and submit_ready is True:
        add(
            checks,
            "pipeline_gate",
            "pipeline submit preflight PASS",
            "pass",
            "hidden-state/train/export submit preflight is ready",
            input_data=payload.get("input_data"),
            modelopt_dir=payload.get("modelopt_dir"),
            command_keys=sorted((payload.get("commands") or {}).keys()),
        )
    elif status == "fail":
        add(
            checks,
            "pipeline_gate",
            "pipeline submit preflight PASS",
            "fail",
            "hidden-state/train/export submit preflight failed",
            submit_ready=submit_ready,
            failing_checks=[
                item.get("name")
                for item in payload.get("checks") or []
                if isinstance(item, dict) and item.get("status") == "fail"
            ],
        )
    else:
        add(
            checks,
            "pipeline_gate",
            "pipeline submit preflight PASS",
            "incomplete",
            "hidden-state/train/export submit preflight is not ready",
            overall_status=status,
            submit_ready=submit_ready,
            missing_checks=[
                item.get("name")
                for item in payload.get("checks") or []
                if isinstance(item, dict) and item.get("status") in {"missing", "incomplete"}
            ][:12],
        )


def check_pipeline_gated_submit(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.pipeline_gated_submit_json)
    if error:
        add(
            checks,
            "pipeline_gate",
            "pipeline gated submit executed",
            "missing",
            error,
            path=str(args.pipeline_gated_submit_json),
        )
        return

    status = payload.get("overall_status")
    executed = payload.get("executed") is True
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), dict) else {}
    required_job_keys = ["dump_job", "train_job", "export_job"]
    missing_jobs = [key for key in required_job_keys if not jobs.get(key)]
    copied_jobs, copied_jobs_error = read_job_env(payload.get("job_file_copy"))
    missing_copied_jobs = [key for key in required_job_keys if not copied_jobs.get(key)]
    failing_checks = [
        item.get("name")
        for item in payload.get("checks") or []
        if isinstance(item, dict) and item.get("status") == "fail"
    ]

    if status == "pass" and executed and not missing_jobs and not copied_jobs_error and not missing_copied_jobs:
        add(
            checks,
            "pipeline_gate",
            "pipeline gated submit executed",
            "pass",
            "gated helper submitted the hidden-state/train/export pipeline",
            command=payload.get("command"),
            jobs=jobs,
            job_file=payload.get("job_file"),
            job_file_copy=payload.get("job_file_copy"),
        )
    elif executed:
        add(
            checks,
            "pipeline_gate",
            "pipeline gated submit executed",
            "fail",
            "gated helper ran but did not produce a complete pipeline job set",
            overall_status=status,
            missing_jobs=missing_jobs,
            job_file=payload.get("job_file"),
            job_file_copy=payload.get("job_file_copy"),
            job_file_copy_error=copied_jobs_error,
            missing_copied_jobs=missing_copied_jobs,
            failing_checks=failing_checks,
            jobs=jobs,
        )
    else:
        add(
            checks,
            "pipeline_gate",
            "pipeline gated submit executed",
            "incomplete",
            "gated helper has not submitted the hidden-state/train/export pipeline yet",
            overall_status=status,
            failing_checks=failing_checks,
            jobs=jobs,
        )


def check_pipeline(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.pipeline_analysis_json)
    if error:
        add(checks, "pipeline", "Slurm pipeline analysis", "missing", error, path=str(args.pipeline_analysis_json))
        return
    status = payload.get("overall_status")
    counts = payload.get("counts")
    if status == "pass":
        add(checks, "pipeline", "Slurm pipeline analysis", "pass", "preflight, dump, validate, train, and export stages passed", counts=counts)
    elif status == "fail":
        add(checks, "pipeline", "Slurm pipeline analysis", "fail", "one or more pipeline stages failed", counts=counts)
    else:
        add(checks, "pipeline", "Slurm pipeline analysis", "incomplete", f"pipeline is {status!r}", counts=counts)


def check_hidden_validation(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.hidden_validation_json)
    if error:
        add(checks, "data", "hidden-state validation", "missing", error, path=str(args.hidden_validation_json))
        return
    checked = int(payload.get("checked_files") or 0)
    total = int(payload.get("total_files") or 0)
    positive = int(payload.get("positive_loss_mask_files") or 0)
    loader = payload.get("modelopt_loader_validation")
    problems: list[str] = []
    if total <= 0:
        problems.append("no hidden-state .pt files")
    if checked <= 0:
        problems.append("no checked hidden-state files")
    if payload.get("expected_hidden_size") != 4096:
        problems.append("expected_hidden_size is not 4096")
    if payload.get("expected_aux_count") != 3:
        problems.append("expected_aux_count is not 3")
    if payload.get("require_loss_mask") is not True:
        problems.append("loss_mask was not required")
    if positive < checked:
        problems.append("not every checked file had a positive loss mask")
    if not isinstance(loader, dict) or int(loader.get("dataset_items_checked") or 0) <= 0:
        problems.append("ModelOpt offline loader was not validated")
    if problems:
        add(checks, "data", "hidden-state validation", "fail", "; ".join(problems), total_files=total, checked_files=checked, positive_loss_mask_files=positive)
        return
    add(checks, "data", "hidden-state validation", "pass", "hidden states, aux states, answer loss masks, and ModelOpt loader were validated", total_files=total, checked_files=checked, max_seq_len_seen=payload.get("max_seq_len_seen"))


def check_dir_nonempty(path: Path, label: str, checks: list[Check]) -> None:
    if not path.exists():
        add(checks, "artifact", label, "missing", f"path is not visible: {path}", path=str(path))
        return
    if path.is_dir() and any(path.iterdir()):
        add(checks, "artifact", label, "pass", "directory exists and is non-empty", path=str(path))
        return
    add(checks, "artifact", label, "incomplete", "path exists but has no files", path=str(path))


def check_export_artifacts(args: argparse.Namespace, checks: list[Check]) -> None:
    check_dir_nonempty(args.output_dir, "ModelOpt trained checkpoint", checks)
    export_config = args.export_dir / "config.json"
    if export_config.exists():
        add(checks, "artifact", "HF exported draft", "pass", "export config exists", path=str(args.export_dir))
    else:
        add(checks, "artifact", "HF exported draft", "missing", f"missing {export_config}", path=str(args.export_dir))
    vllm_config = args.vllm_draft_dir / "config.json"
    safetensors = sorted(args.vllm_draft_dir.glob("*.safetensors")) if args.vllm_draft_dir.exists() else []
    if vllm_config.exists() and safetensors:
        add(
            checks,
            "artifact",
            "vLLM Eagle3 draft",
            "pass",
            "vLLM config and safetensors weights exist",
            path=str(args.vllm_draft_dir),
            safetensors=[str(path.name) for path in safetensors[:8]],
        )
    elif vllm_config.exists():
        add(checks, "artifact", "vLLM Eagle3 draft", "incomplete", "config exists but safetensors weights are missing", path=str(args.vllm_draft_dir))
    else:
        add(checks, "artifact", "vLLM Eagle3 draft", "missing", f"missing {vllm_config}", path=str(args.vllm_draft_dir))


def check_training_checkpoint_report(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.training_checkpoint_json)
    if error:
        add(checks, "artifact", "ModelOpt training checkpoint contract", "missing", error, path=str(args.training_checkpoint_json))
        return
    status = payload.get("overall_status")
    recorded_checkpoint = payload.get("checkpoint_dir")
    if recorded_checkpoint and Path(recorded_checkpoint).expanduser().resolve(strict=False) != args.output_dir.expanduser().resolve(strict=False):
        add(
            checks,
            "artifact",
            "ModelOpt training checkpoint contract",
            "fail",
            "training checkpoint report validates a different checkpoint directory",
            recorded_checkpoint_dir=recorded_checkpoint,
            expected_checkpoint_dir=str(args.output_dir),
        )
        return
    check_rows = payload.get("checks") or []
    failures = [
        row
        for row in check_rows
        if isinstance(row, dict) and row.get("status") in {"fail", "missing", "incomplete"}
    ]
    modes = payload.get("modelopt_modes") or []
    if status == "pass" and not failures and "eagle" in modes:
        add(
            checks,
            "artifact",
            "ModelOpt training checkpoint contract",
            "pass",
            "trained checkpoint has HF weights, trainer state, and ModelOpt Eagle state",
            checkpoint_dir=recorded_checkpoint,
            trainer_global_step=payload.get("trainer_global_step"),
            modelopt_modes=modes,
            check_count=len(check_rows),
        )
    elif status == "pass":
        add(
            checks,
            "artifact",
            "ModelOpt training checkpoint contract",
            "fail",
            "training checkpoint validator passed but did not record an eagle mode",
            modelopt_modes=modes,
        )
    elif status == "incomplete":
        add(
            checks,
            "artifact",
            "ModelOpt training checkpoint contract",
            "incomplete",
            "training checkpoint validator found missing evidence",
            failures=failures[:8],
        )
    else:
        add(
            checks,
            "artifact",
            "ModelOpt training checkpoint contract",
            "fail",
            "training checkpoint validator did not pass",
            overall_status=status,
            failures=failures[:8],
        )


def check_config_compare(path: Path, label: str, checks: list[Check]) -> None:
    payload, error = load_json(path)
    if error:
        add(checks, "artifact", label, "missing", error, path=str(path))
        return
    failure_count = int(payload.get("failure_count") or 0)
    status = payload.get("status")
    if status == "passed" and failure_count == 0:
        add(
            checks,
            "artifact",
            label,
            "pass",
            "config comparison passed",
            config_kind=payload.get("config_kind"),
            check_count=len(payload.get("checks") or []),
        )
    else:
        add(checks, "artifact", label, "fail", "config comparison failed", status=status, failure_count=failure_count)


def check_export_artifact_report(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.export_artifacts_json)
    if error:
        add(checks, "artifact", "Eagle3 export artifact contract", "missing", error, path=str(args.export_artifacts_json))
        return
    status = payload.get("overall_status")
    check_rows = payload.get("checks") or []
    failures = [row for row in check_rows if isinstance(row, dict) and row.get("status") in {"fail", "missing"}]
    recorded_export = payload.get("export_dir")
    recorded_vllm = payload.get("vllm_draft_dir")
    if recorded_export and Path(recorded_export).expanduser().resolve(strict=False) != args.export_dir.expanduser().resolve(strict=False):
        add(
            checks,
            "artifact",
            "Eagle3 export artifact contract",
            "fail",
            "export artifact report validates a different HF export directory",
            recorded_export_dir=recorded_export,
            expected_export_dir=str(args.export_dir),
        )
        return
    if recorded_vllm and Path(recorded_vllm).expanduser().resolve(strict=False) != args.vllm_draft_dir.expanduser().resolve(strict=False):
        add(
            checks,
            "artifact",
            "Eagle3 export artifact contract",
            "fail",
            "export artifact report validates a different vLLM draft directory",
            recorded_vllm_draft_dir=recorded_vllm,
            expected_vllm_draft_dir=str(args.vllm_draft_dir),
        )
        return
    if status == "pass" and not failures:
        add(
            checks,
            "artifact",
            "Eagle3 export artifact contract",
            "pass",
            "HF/vLLM export artifact contract passed",
            export_dir=payload.get("export_dir"),
            vllm_draft_dir=payload.get("vllm_draft_dir"),
            check_count=len(check_rows),
        )
    elif status == "incomplete":
        add(
            checks,
            "artifact",
            "Eagle3 export artifact contract",
            "incomplete",
            "export artifact validator found missing evidence",
            failures=failures[:8],
        )
    else:
        add(
            checks,
            "artifact",
            "Eagle3 export artifact contract",
            "fail",
            "export artifact validator did not pass",
            overall_status=status,
            failures=failures[:8],
        )


def check_sweep(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.sweep_json)
    if error:
        add(checks, "rl_validation", "trained-draft spec-token sweep", "missing", error, path=str(args.sweep_json))
        return
    rows = payload.get("rows") or []
    recommendation = payload.get("recommendation") or {}
    passed_rows = [row for row in rows if row.get("gate_status") == "pass"]
    recorded_draft_dir = payload.get("vllm_draft_dir")
    expected_draft_dir = str(args.vllm_draft_dir)
    execution_context = payload.get("execution_context") or {}
    if not recorded_draft_dir:
        add(
            checks,
            "rl_validation",
            "trained-draft spec-token sweep",
            "incomplete",
            "sweep report does not record the vLLM draft directory it tested",
            rows=len(rows),
            recommendation=recommendation,
        )
        return
    missing_context = [
        key
        for key in ("artifact_root", "config_file", "env_file", "chat_template")
        if not execution_context.get(key)
    ]
    if not (execution_context.get("repo_root") or execution_context.get("swe_repo_root")):
        missing_context.append("repo_root_or_swe_repo_root")
    if missing_context:
        add(
            checks,
            "rl_validation",
            "trained-draft spec-token sweep",
            "incomplete",
            "sweep report does not record enough RL execution context",
            missing_context=missing_context,
            rows=len(rows),
            recommendation=recommendation,
        )
        return
    recorded_draft_path = Path(recorded_draft_dir).expanduser().resolve(strict=False)
    expected_draft_path = args.vllm_draft_dir.expanduser().resolve(strict=False)
    if recorded_draft_path != expected_draft_path:
        add(
            checks,
            "rl_validation",
            "trained-draft spec-token sweep",
            "fail",
            "sweep report tested a different vLLM draft directory than the audited artifact",
            recorded_vllm_draft_dir=recorded_draft_dir,
            expected_vllm_draft_dir=expected_draft_dir,
            recorded_resolved=str(recorded_draft_path),
            expected_resolved=str(expected_draft_path),
            rows=len(rows),
            recommendation=recommendation,
        )
        return
    if payload.get("overall_status") == "pass" and passed_rows and recommendation.get("gate_status") == "pass":
        add(
            checks,
            "rl_validation",
            "trained-draft spec-token sweep",
            "pass",
            "at least one trained-draft spec-token setting passed the RL smoke gate",
            recommendation=recommendation,
            passed_rows=len(passed_rows),
            vllm_draft_dir=recorded_draft_dir,
            execution_context=execution_context,
            spec_tokens_list=payload.get("spec_tokens_list"),
            eagle3_draft_tp=payload.get("eagle3_draft_tp"),
        )
    elif payload.get("overall_status") == "fail":
        add(checks, "rl_validation", "trained-draft spec-token sweep", "fail", "trained-draft sweep did not pass any gate", rows=len(rows))
    else:
        add(checks, "rl_validation", "trained-draft spec-token sweep", "incomplete", "sweep report exists but no passing recommendation is proven", rows=len(rows), recommendation=recommendation)


def check_reference_artifacts(args: argparse.Namespace, checks: list[Check]) -> None:
    payload, error = load_json(args.draft_inventory_json)
    if error:
        add(checks, "reference_gate", "Hayate/draft config inventory", "missing", error, path=str(args.draft_inventory_json))
    else:
        configs = payload.get("configs") or []
        warnings = payload.get("warnings") or []
        roots = payload.get("roots") or []
        root_statuses = payload.get("root_statuses") or []
        matches = [item for item in configs if item.get("matches_reference")]
        inventory_status = json_status(payload)
        if inventory_status == "fail":
            add(
                checks,
                "reference_gate",
                "Hayate/draft config inventory",
                "fail",
                "draft config inventory failed",
                configs_scanned=payload.get("configs_scanned"),
                roots=roots,
                warnings=warnings[:4],
            )
        elif configs or warnings:
            add(
                checks,
                "reference_gate",
                "Hayate/draft config inventory",
                "pass",
                "Hayate/local draft config inventory is available or access limitations are recorded",
                configs_scanned=payload.get("configs_scanned"),
                matching_reference=len(matches),
                roots=roots,
                root_statuses=root_statuses,
                warning_count=len(warnings),
                warnings=warnings[:4],
                recommendation=payload.get("recommendation"),
            )
        else:
            add(
                checks,
                "reference_gate",
                "Hayate/draft config inventory",
                "incomplete",
                "draft inventory report exists but contains no configs or access-limit warnings",
                configs_scanned=payload.get("configs_scanned"),
                roots=roots,
                root_statuses=root_statuses,
            )
    if args.hayate_inventory.exists() and args.hayate_inventory.stat().st_size > 0:
        add(checks, "reference", "Hayate filesystem inventory", "pass", "Hayate inventory text is present", required=False, path=str(args.hayate_inventory), bytes=args.hayate_inventory.stat().st_size)
    else:
        add(checks, "reference", "Hayate filesystem inventory", "warn", f"not visible or empty: {args.hayate_inventory}", required=False, path=str(args.hayate_inventory))


def overall_status(checks: list[Check]) -> str:
    required = [check for check in checks if check.required]
    if any(check.status == "fail" for check in required):
        return "fail"
    if any(check.status in {"missing", "incomplete"} for check in required):
        return "incomplete"
    if all(check.status == "pass" for check in required):
        return "pass"
    return "incomplete"


def payload(checks: list[Check], args: argparse.Namespace) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall_status(checks),
        "artifact_root": str(args.artifact_root),
        "counts": counts,
        "checks": [
            {
                "area": check.area,
                "name": check.name,
                "status": check.status,
                "required": check.required,
                "detail": check.detail,
                "evidence": check.evidence,
            }
            for check in checks
        ],
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-235B Eagle3 Completion Audit",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Generated: `{data['generated_at']}`",
        f"Artifact root: `{data['artifact_root']}`",
        "",
        "| required | area | check | status | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in data["checks"]:
        required = "yes" if check["required"] else "no"
        lines.append(
            f"| {required} | {check['area']} | {check['name']} | "
            f"{check['status'].upper()} | {check['detail'].replace('|', '/')} |"
        )
    open_required = [check for check in data["checks"] if check["required"] and check["status"] != "pass"]
    if open_required:
        lines += ["", "## Required Next Actions", ""]
        for check in open_required:
            lines.append(f"- `{check['area']} / {check['name']}`: {check['detail']}")
    return "\n".join(lines) + "\n"


def write_outputs(data: dict[str, Any], args: argparse.Namespace) -> None:
    markdown = render_markdown(data)
    print(markdown, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = with_defaults(parse_args())
    checks: list[Check] = []
    check_architecture(args, checks)
    check_input_discovery(args, checks)
    check_provenance(args, checks)
    check_remote_reference_probe(args, checks)
    check_hayate_workflow(args, checks)
    check_hayate_specforge_reference(args, checks)
    check_upstream_drift(args, checks)
    check_modelopt_patch(args, checks)
    check_modelopt_recipe_overrides(args, checks)
    check_training_path_manifest(args, checks)
    check_next_action_plan(args, checks)
    check_next_action_plan_validation(args, checks)
    check_operator_queue_transitions(args, checks)
    check_operator_followup_validation(args, checks)
    check_megatron_probe_followup_validation(args, checks)
    check_preflight_robustness_validation(args, checks)
    check_operator_submit_packet_validation(args, checks)
    check_operator_ready_submit_preflight(args, checks)
    check_operator_queue(args, checks)
    check_completion_contract(args, checks)
    check_slurm_capacity(args, checks)
    check_resource_profile_application(args, checks)
    check_rollout_monitoring(args, checks)
    check_cluster_probe(args, checks)
    check_readiness(args, checks)
    check_container_preflight(args, checks)
    check_vllm_runtime(args, checks)
    check_rollout_corpus(args, checks)
    check_pipeline_submit_preflight(args, checks)
    check_pipeline_gated_submit(args, checks)
    check_pipeline(args, checks)
    check_hidden_validation(args, checks)
    check_export_artifacts(args, checks)
    check_training_checkpoint_report(args, checks)
    check_export_artifact_report(args, checks)
    check_config_compare(args.export_config_compare_json, "HF export config comparison", checks)
    check_config_compare(args.vllm_config_compare_json, "vLLM draft config comparison", checks)
    check_sweep(args, checks)
    check_reference_artifacts(args, checks)
    data = payload(checks, args)
    write_outputs(data, args)
    if args.fail_if_not_pass and data["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
