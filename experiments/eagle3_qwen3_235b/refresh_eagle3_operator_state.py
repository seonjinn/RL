#!/usr/bin/env python3
"""Refresh Qwen3-235B Eagle3 operator state after any gate result changes.

This script is no-submit. It first refreshes state reports that the planner
reads, then reruns the planner, validators, operator sheet, submit packet, goal
evidence, and completion audit so the next ready action reflects newly finished
Slurm gate jobs.
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
DEFAULT_REMOTE_HOSTS = [
    "oci-hsg-cs-001-vscode-02",
    "oci-hsg-cs-001-vscode-01",
    "oci-hsg-cs-001-vscode-03",
    "oci-hsg-cs-001-login-01.nvidia.com",
    "oci-hsg",
]
DEFAULT_REMOTE_WORKDIR = "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap"
DEFAULT_CONTAINER = "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
DEFAULT_DRAFT_INVENTORY_ROOTS = [
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec/models",
    "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/outputs",
]
SOFT_NONZERO_STEPS = {
    "create_operator_submit_packet",
    "audit_readiness",
    "discover_run_inputs",
    "plan_next_actions",
    "preflight_rollout_capture_submit",
    "preflight_rollout_resource_profiles",
    "preflight_operator_ready_submit",
    "summarize_rollout_queue_wait",
    "validate_operator_submit_packet",
}


def parse_args() -> argparse.Namespace:
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--remote-hosts", nargs="+", default=os.environ.get("REMOTE_HOSTS", "").split() or DEFAULT_REMOTE_HOSTS)
    parser.add_argument("--remote-workdir", default=os.environ.get("REMOTE_WORKDIR", DEFAULT_REMOTE_WORKDIR))
    parser.add_argument(
        "--draft-inventory-roots",
        nargs="+",
        default=os.environ.get("DRAFT_INVENTORY_ROOTS", "").split() or DEFAULT_DRAFT_INVENTORY_ROOTS,
    )
    parser.add_argument("--skip-remote-host-probe", action="store_true")
    parser.add_argument("--skip-completion-audit", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def shell_join(command: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def read_export_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        for token in tokens:
            if "=" not in token or token.startswith("-"):
                continue
            key, value = token.split("=", 1)
            if key.replace("_", "").isalnum() and key[:1].isalpha():
                env[key] = value
    return env


def run_step(name: str, command: list[str | Path]) -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    completed = time.time()
    return {
        "name": name,
        "command": shell_join(command),
        "returncode": result.returncode,
        "duration_seconds": round(completed - started, 3),
        "output_tail": result.stdout[-8000:],
    }


def report_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid: {exc}"
    if not isinstance(payload, dict):
        return "invalid: top-level JSON is not an object"
    if path.name == "eagle3_input_discovery.json":
        verifier = payload.get("verifier_candidates") or []
        conversations = payload.get("conversation_candidates") or []
        return "pass" if verifier and conversations else "warn"
    if path.name == "eagle3_provenance.json":
        critical = payload.get("critical_files") or []
        missing = [item for item in critical if isinstance(item, dict) and not item.get("exists")]
        return "warn" if missing else "pass"
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(payload.get("overall_status") or payload.get("status") or decision.get("overall_status") or "unknown")


def text_report_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return "present" if path.stat().st_size > 0 else "empty"
    except OSError as exc:
        return f"invalid: {exc}"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def active_rollout_job_ids(root: Path) -> set[str]:
    payload = load_json(root / "reports/rollout_queue_wait_summary.json")
    if not payload:
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


def rollout_state_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(payload.get("overall_status") or payload.get("status") or decision.get("overall_status") or "unknown")


def select_rollout_state_report(root: Path) -> Path:
    reports = root / "reports"
    default = reports / "rollout_capture_state_advance.json"
    active_ids = active_rollout_job_ids(root)
    candidates: list[tuple[int, float, Path]] = []
    for path in reports.glob("rollout_capture*_state_advance.json"):
        if path.name == "rollout_capture_compact16n4g_state_advance.json":
            continue
        payload = load_json(path)
        if not payload:
            continue
        job_id = rollout_state_job_id(payload)
        status = rollout_state_status(payload)
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


def report_summary(root: Path) -> dict[str, Any]:
    reports = {
        "input_discovery": root / "eagle3_input_discovery.json",
        "provenance": root / "reports/eagle3_provenance.json",
        "cluster_environment": root / "reports/cluster_environment_probe.json",
        "readiness": root / "reports/eagle3_readiness.json",
        "next_action_plan": root / "reports/eagle3_next_actions.json",
        "next_action_plan_validation": root / "reports/eagle3_next_actions_validation.json",
        "remote_host_probe": root / "reports/eagle3_remote_host_probe.json",
        "remote_access_diagnostics": root / "reports/eagle3_remote_access_diagnostics.json",
        "next_action_transitions": root / "reports/eagle3_next_action_transitions.json",
        "operator_queue_transitions": root / "reports/eagle3_operator_queue_transitions.json",
        "completion_contract": root / "reports/eagle3_completion_contract.json",
        "slurm_capacity": root / "reports/eagle3_slurm_capacity.json",
        "resource_profile_application": root / "reports/eagle3_resource_profile_application.json",
        "rollout_queue_wait": root / "reports/rollout_queue_wait_summary.json",
        "rollout_job_arbitration": root / "reports/rollout_job_arbitration.json",
        "rollout_resource_profiles": root / "reports/rollout_resource_profiles_preflight.json",
        "rollout_submit_preflight": root / "reports/rollout_capture_submit_preflight.json",
        "qwen3_static_inputs": root / "reports/qwen3_static_inputs.json",
        "qwen3_static_inputs_materialization_validation": root / "reports/qwen3_static_inputs_materialization_validation.json",
        "preflight_robustness_validation": root / "reports/eagle3_preflight_robustness_validation.json",
        "pipeline_dry_run_manifest_validation": root / "reports/eagle3_pipeline_dry_run_manifest_validation.json",
        "modelopt_loss_mask_patch": root / "reports/modelopt_loss_mask_patch.json",
        "modelopt_upstream_drift": root / "reports/modelopt_upstream_drift.json",
        "modelopt_recipe_overrides_current": root / "reports/modelopt_recipe_overrides_current.json",
        "modelopt_recipe_overrides_online": root / "reports/modelopt_recipe_overrides_online.json",
        "nemo_rl_eagle3_drift": root / "reports/nemo_rl_eagle3_drift.json",
        "nemo_rl_specdec_integration": root / "reports/nemo_rl_specdec_integration.json",
        "specdec_rl_remote_patch_bundle": root / "reports/specdec_rl_remote_patch_bundle.json",
        "rollout_fallback_decision": root / "reports/rollout_fallback_decision.json",
        "swe_nemogym_dataset_discovery": root / "reports/swe_nemogym_dataset_discovery.json",
        "rollout_watcher_health": root / "reports/rollout_watcher_health.json",
        "rollout_watcher_health_validation": root / "reports/rollout_watcher_health_validation.json",
        "rollout_watcher_ensure": root / "reports/rollout_watcher_ensure.json",
        "rollout_watcher_ensure_validation": root / "reports/rollout_watcher_ensure_validation.json",
        "hayate_specforge_reference": root / "reports/hayate_specforge_reference.json",
        "hayate_modelopt_workflow": root / "reports/hayate_modelopt_workflow.json",
        "draft_config_inventory": root / "reports/eagle3_draft_config_inventory.json",
        "container_preflight": root / "reports/container_preflight_analysis.json",
        "pipeline_analysis": root / "reports/eagle3_pipeline_analysis.json",
        "pipeline_submit_preflight": root / "reports/eagle3_pipeline_submit_preflight.json",
        "training_path_manifest": root / "reports/eagle3_training_path_manifest.json",
        "training_path_manifest_validation": root / "reports/eagle3_training_path_manifest_validation.json",
        "pipeline_submit_preflight_contract": root / "reports/eagle3_pipeline_submit_preflight_contract.json",
        "pipeline_gated_submit": root / "reports/eagle3_pipeline_gated_submit.json",
        "pipeline_gated_submit_contract": root / "reports/eagle3_pipeline_gated_submit_contract.json",
        "full_swegym_after_smoke_gate": root / "reports/full_swegym_after_smoke_gate.json",
        "full_swegym_after_smoke_gate_validation": root / "reports/full_swegym_after_smoke_gate_validation.json",
        "rollout_submit_preflight_contract": root / "reports/rollout_submit_preflight_contract.json",
        "operator_sheet": root / "reports/eagle3_operator_sheet.json",
        "operator_sheet_validation": root / "reports/eagle3_operator_sheet_validation.json",
        "operator_execution": root / "reports/eagle3_operator_execution.json",
        "operator_followup_validation": root / "reports/eagle3_operator_followups_validation.json",
        "goal_evidence": root / "reports/eagle3_goal_evidence.json",
        "operator_submit_packet": root / "reports/eagle3_operator_submit_packet.json",
        "operator_submit_packet_validation": root / "reports/eagle3_operator_submit_packet_validation.json",
        "operator_ready_submit_preflight": root / "reports/eagle3_operator_ready_submit_preflight.json",
        "operator_safe_actions_preflight": root / "reports/eagle3_operator_safe_actions_preflight.json",
        "operator_queue": root / "reports/eagle3_operator_queue.json",
        "completion_audit": root / "reports/eagle3_completion_audit.json",
    }
    summary = {name: {"path": str(path), "status": report_status(path)} for name, path in reports.items()}
    hayate_inventory = root / "reports/hayate_inventory.txt"
    summary["hayate_inventory"] = {"path": str(hayate_inventory), "status": text_report_status(hayate_inventory)}
    return summary


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str | Path]]]:
    root = args.artifact_root
    reports = root / "reports"
    rollout_state = select_rollout_state_report(root)
    resource_env = read_export_env(reports / "eagle3_resource_profile.env")
    dump_gpus_per_node = resource_env.get("DUMP_GPUS_PER_NODE", os.environ.get("DUMP_GPUS_PER_NODE", "8"))
    train_gpus_per_node = resource_env.get("TRAIN_GPUS_PER_NODE", os.environ.get("TRAIN_GPUS_PER_NODE", "8"))
    export_gpus_per_node = resource_env.get("EXPORT_GPUS_PER_NODE", os.environ.get("EXPORT_GPUS_PER_NODE", "1"))
    tp = resource_env.get("TP", os.environ.get("TP", "8"))
    modelopt_dir = Path(os.environ.get("MODELOPT_DIR", ROOT / "Model-Optimizer"))
    container = os.environ.get("CONTAINER", DEFAULT_CONTAINER)
    mounts = os.environ.get("MOUNTS", f"/lustre:/lustre,{ROOT}:{ROOT},{root}:{root}")
    sbatch_account = os.environ.get("SBATCH_ACCOUNT", "coreai_dlalgo_nemorl")
    sbatch_partition = os.environ.get("SBATCH_PARTITION", "batch")
    draft_inventory_roots: list[str | Path] = [*args.draft_inventory_roots, root / "vllm_draft", root / "exported_hf"]
    steps: list[tuple[str, list[str | Path]]] = [
        (
            "validate_completion_contract",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_completion_contract.py",
                "--json-out",
                reports / "eagle3_completion_contract.json",
                "--markdown-out",
                reports / "eagle3_completion_contract.md",
            ],
        ),
    ]
    if not args.skip_remote_host_probe:
        steps.append(
            (
                "probe_remote_hosts",
                [
                    "python3",
                    "experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py",
                    "--include-ssh-config-hosts",
                    "--hosts",
                    *args.remote_hosts,
                    "--remote-workdir",
                    args.remote_workdir,
                    "--artifact-root",
                    root,
                    "--json-out",
                    reports / "eagle3_remote_host_probe.json",
                    "--markdown-out",
                    reports / "eagle3_remote_host_probe.md",
                ],
            )
        )
    steps.append(
        (
            "diagnose_remote_access",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/diagnose_eagle3_remote_access.py",
                "--remote-host-probe-json",
                reports / "eagle3_remote_host_probe.json",
                "--json-out",
                reports / "eagle3_remote_access_diagnostics.json",
                "--markdown-out",
                reports / "eagle3_remote_access_diagnostics.md",
            ],
        )
    )
    steps += [
        (
            "discover_run_inputs",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py",
                ROOT,
                root,
                "--artifact-root",
                root,
                "--max-depth",
                "7",
                "--max-files",
                "30000",
                "--json-out",
                root / "eagle3_input_discovery.json",
                "--markdown-out",
                reports / "eagle3_input_discovery.md",
                "--env-out",
                reports / "eagle3_input_discovery.env",
            ],
        ),
        (
            "collect_provenance",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/collect_eagle3_provenance.py",
                "--artifact-root",
                root,
                "--repo-root",
                ROOT,
                "--modelopt-dir",
                modelopt_dir,
                "--verifier-config-dir",
                root / "verifier_config",
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--hidden-states-dir",
                root / "hidden_states",
                "--output-dir",
                root / "modelopt_ckpt",
                "--export-dir",
                root / "exported_hf",
                "--vllm-draft-dir",
                root / "vllm_draft",
                "--extra-path",
                root / "patches/modelopt_eagle3_qwen3/manifest.json",
                "--json-out",
                reports / "eagle3_provenance.json",
                "--markdown-out",
                reports / "eagle3_provenance.md",
            ],
        ),
        (
            "probe_cluster_environment",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/probe_cluster_environment.py",
                "--artifact-root",
                root,
                "--modelopt-dir",
                modelopt_dir,
                "--verifier-config-dir",
                root / "verifier_config",
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--container",
                container,
                "--mounts",
                mounts,
                "--sbatch-account",
                sbatch_account,
                "--sbatch-partition",
                sbatch_partition,
                "--json-out",
                reports / "cluster_environment_probe.json",
                "--markdown-out",
                reports / "cluster_environment_probe.md",
            ],
        ),
        (
            "probe_slurm_capacity",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py",
                "--artifact-root",
                root,
                "--dump-gpus-per-node",
                dump_gpus_per_node,
                "--train-gpus-per-node",
                train_gpus_per_node,
                "--export-gpus-per-node",
                export_gpus_per_node,
                "--tp",
                tp,
                "--json-out",
                reports / "eagle3_slurm_capacity.json",
                "--markdown-out",
                reports / "eagle3_slurm_capacity.md",
                "--env-out",
                reports / "eagle3_resource_profile.env",
            ],
        ),
        (
            "validate_resource_profile_application",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_resource_profile_application.py",
                "--artifact-root",
                root,
                "--resource-profile-env",
                reports / "eagle3_resource_profile.env",
                "--json-out",
                reports / "eagle3_resource_profile_application.json",
                "--markdown-out",
                reports / "eagle3_resource_profile_application.md",
            ],
        ),
        (
            "summarize_rollout_queue_wait",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/summarize_rollout_queue_wait.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_queue_wait_summary.json",
                "--markdown-out",
                reports / "rollout_queue_wait_summary.md",
            ],
        ),
        (
            "validate_rollout_queue_wait_summary",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_rollout_queue_wait_summary.py",
                "--json-out",
                reports / "rollout_queue_wait_summary_validation.json",
                "--markdown-out",
                reports / "rollout_queue_wait_summary_validation.md",
            ],
        ),
        (
            "arbitrate_rollout_jobs",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/arbitrate_rollout_jobs.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_job_arbitration.json",
                "--markdown-out",
                reports / "rollout_job_arbitration.md",
            ],
        ),
        (
            "preflight_rollout_resource_profiles",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/preflight_rollout_resource_profiles.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_resource_profiles_preflight.json",
                "--markdown-out",
                reports / "rollout_resource_profiles_preflight.md",
            ],
        ),
        (
            "preflight_rollout_capture_submit",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/preflight_rollout_capture_submit.py",
                "--artifact-root",
                root,
                "--repo-root",
                Path("/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"),
                "--config",
                ROOT / "grpo_qwen3_235b_swe.yaml",
                "--chat-template",
                root / "templates/qwen3_generation_template.jinja2",
                "--rollout-log-dir",
                root / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke",
                "--output-conversations",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--resource-profile-env",
                reports / "eagle3_resource_profile.env",
                "--sbatch-account",
                sbatch_account,
                "--sbatch-partition",
                sbatch_partition,
                "--container",
                container,
                "--json-out",
                reports / "rollout_capture_submit_preflight.json",
                "--markdown-out",
                reports / "rollout_capture_submit_preflight.md",
            ],
        ),
        (
            "validate_preflight_robustness",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_preflight_robustness.py",
                "--json-out",
                reports / "eagle3_preflight_robustness_validation.json",
                "--markdown-out",
                reports / "eagle3_preflight_robustness_validation.md",
            ],
        ),
        (
            "validate_pipeline_dry_run_manifest",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_pipeline_dry_run_manifest.py",
                "--json-out",
                reports / "eagle3_pipeline_dry_run_manifest_validation.json",
                "--markdown-out",
                reports / "eagle3_pipeline_dry_run_manifest_validation.md",
            ],
        ),
        (
            "validate_qwen3_static_inputs_materialization",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_qwen3_static_inputs_materialization.py",
                "--json-out",
                reports / "qwen3_static_inputs_materialization_validation.json",
                "--markdown-out",
                reports / "qwen3_static_inputs_materialization_validation.md",
            ],
        ),
        (
            "validate_modelopt_loss_mask_patch",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_modelopt_loss_mask_patch.py",
                "--modelopt-dir",
                modelopt_dir,
                "--json-out",
                reports / "modelopt_loss_mask_patch.json",
                "--markdown-out",
                reports / "modelopt_loss_mask_patch.md",
            ],
        ),
        (
            "check_modelopt_upstream_drift",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/check_modelopt_upstream_drift.py",
                "--modelopt-dir",
                modelopt_dir,
                "--json-out",
                reports / "modelopt_upstream_drift.json",
                "--markdown-out",
                reports / "modelopt_upstream_drift.md",
            ],
        ),
        (
            "validate_modelopt_recipe_overrides_current",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py",
                "--wrapper",
                "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh",
                "--training-mode",
                "offline",
                "--modelopt-dir",
                modelopt_dir,
                "--json-out",
                reports / "modelopt_recipe_overrides_current.json",
                "--markdown-out",
                reports / "modelopt_recipe_overrides_current.md",
            ],
        ),
        (
            "validate_modelopt_recipe_overrides_online",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py",
                "--wrapper",
                "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh",
                "--training-mode",
                "online",
                "--modelopt-dir",
                modelopt_dir,
                "--json-out",
                reports / "modelopt_recipe_overrides_online.json",
                "--markdown-out",
                reports / "modelopt_recipe_overrides_online.md",
            ],
        ),
        (
            "check_nemo_rl_eagle3_drift",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/check_nemo_rl_eagle3_drift.py",
                "--json-out",
                reports / "nemo_rl_eagle3_drift.json",
                "--markdown-out",
                reports / "nemo_rl_eagle3_drift.md",
            ],
        ),
        (
            "validate_nemo_rl_specdec_integration",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py",
                "--json-out",
                reports / "nemo_rl_specdec_integration.json",
                "--markdown-out",
                reports / "nemo_rl_specdec_integration.md",
                "--env-out",
                reports / "nemo_rl_specdec_integration.env",
            ],
        ),
        (
            "validate_specdec_rl_remote_patch_bundle",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_specdec_rl_remote_patch_bundle.py",
                "--json-out",
                reports / "specdec_rl_remote_patch_bundle.json",
                "--markdown-out",
                reports / "specdec_rl_remote_patch_bundle.md",
            ],
        ),
        (
            "decide_rollout_fallback",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/decide_rollout_fallback.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_fallback_decision.json",
                "--markdown-out",
                reports / "rollout_fallback_decision.md",
            ],
        ),
        (
            "discover_swe_nemogym_datasets",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/discover_swe_nemogym_datasets.py",
                "--max-depth",
                "7",
                "--count-lines",
                "--json-out",
                reports / "swe_nemogym_dataset_discovery.json",
                "--markdown-out",
                reports / "swe_nemogym_dataset_discovery.md",
            ],
        ),
        (
            "summarize_rollout_watcher_health",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/summarize_rollout_watcher_health.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_watcher_health.json",
                "--markdown-out",
                reports / "rollout_watcher_health.md",
            ],
        ),
        (
            "ensure_rollout_watchers",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/ensure_rollout_watchers.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "rollout_watcher_ensure.json",
                "--markdown-out",
                reports / "rollout_watcher_ensure.md",
            ],
        ),
        (
            "validate_rollout_watcher_health",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_rollout_watcher_health.py",
                "--json-out",
                reports / "rollout_watcher_health_validation.json",
                "--markdown-out",
                reports / "rollout_watcher_health_validation.md",
            ],
        ),
        (
            "validate_rollout_watcher_ensure",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_rollout_watcher_ensure.py",
                "--json-out",
                reports / "rollout_watcher_ensure_validation.json",
                "--markdown-out",
                reports / "rollout_watcher_ensure_validation.md",
            ],
        ),
        (
            "analyze_hayate_specforge_reference",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/analyze_hayate_specforge_reference.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "hayate_specforge_reference.json",
                "--markdown-out",
                reports / "hayate_specforge_reference.md",
            ],
        ),
        (
            "analyze_hayate_modelopt_workflow",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py",
                "--json-out",
                reports / "hayate_modelopt_workflow.json",
                "--markdown-out",
                reports / "hayate_modelopt_workflow.md",
            ],
        ),
        (
            "inventory_hayate_artifacts",
            [
                "env",
                f"HAYATE_INVENTORY_OUT={reports / 'hayate_inventory.txt'}",
                "bash",
                "experiments/eagle3_qwen3_235b/inventory_hayate_eagle3_artifacts.sh",
            ],
        ),
        (
            "inventory_eagle3_draft_configs",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/inventory_eagle3_draft_configs.py",
                *draft_inventory_roots,
                "--reference-arch",
                "experiments/eagle3_qwen3_235b/qwen3_235b_thinking_eagle3_architecture.json",
                "--json-out",
                reports / "eagle3_draft_config_inventory.json",
                "--markdown-out",
                reports / "eagle3_draft_config_inventory.md",
            ],
        ),
        (
            "refresh_corpus_strategy",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py",
                "--artifact-root",
                root,
                "--target-context",
                "swe_rl",
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--rollout-capture-analysis-json",
                rollout_state,
                "--json-out",
                reports / "corpus_strategy.json",
                "--markdown-out",
                reports / "corpus_strategy.md",
            ],
        ),
        (
            "run_pipeline_submit_preflight",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py",
                "--artifact-root",
                root,
                "--dump-gpus-per-node",
                dump_gpus_per_node,
                "--train-gpus-per-node",
                train_gpus_per_node,
                "--export-gpus-per-node",
                export_gpus_per_node,
                "--tp",
                tp,
                "--json-out",
                reports / "eagle3_pipeline_submit_preflight.json",
                "--markdown-out",
                reports / "eagle3_pipeline_submit_preflight.md",
            ],
        ),
        (
            "analyze_pipeline_dry_run",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py",
                "--job-file",
                ROOT / "latest_eagle3_pipeline_jobs.txt",
                "--logs-dir",
                ROOT / "logs",
                "--base-model",
                "Qwen/Qwen3-235B-A22B-Thinking-2507",
                "--modelopt-dir",
                modelopt_dir,
                "--verifier-config-dir",
                root / "verifier_config",
                "--reference-arch",
                root / "architecture/eagle3_architecture.json",
                "--arch-env-file",
                root / "architecture/eagle3_architecture.env",
                "--chat-template",
                root / "templates/qwen3_generation_template.jinja2",
                "--container",
                container,
                "--mounts",
                mounts,
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--hidden-states-dir",
                root / "hidden_states",
                "--hidden-validation-json",
                root / "hidden_states/validation_summary.json",
                "--training-checkpoint-json",
                reports / "eagle3_training_checkpoint.json",
                "--output-dir",
                root / "modelopt_ckpt",
                "--export-dir",
                root / "exported_hf",
                "--vllm-draft-dir",
                root / "vllm_draft",
                "--export-artifacts-json",
                reports / "eagle3_export_artifacts.json",
                "--sbatch-account",
                sbatch_account,
                "--run-pilot",
                "true",
                "--sbatch-partition",
                sbatch_partition,
                "--json-out",
                reports / "eagle3_pipeline_analysis.json",
                "--markdown-out",
                reports / "eagle3_pipeline_analysis.md",
            ],
        ),
        (
            "estimate_training_scale",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py",
                "--artifact-root",
                root,
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--corpus-strategy-json",
                reports / "corpus_strategy.json",
                "--pipeline-submit-preflight-json",
                reports / "eagle3_pipeline_submit_preflight.json",
                "--gpus",
                train_gpus_per_node,
                "--target-context",
                "swe_rl",
                "--json-out",
                reports / "eagle3_training_scale.json",
                "--markdown-out",
                reports / "eagle3_training_scale.md",
            ],
        ),
        (
            "audit_readiness",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/audit_eagle3_readiness.py",
                "--input-data",
                root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
                "--hidden-states-dir",
                root / "hidden_states",
                "--output-dir",
                root / "modelopt_ckpt",
                "--trained-ckpt",
                root / "modelopt_ckpt",
                "--export-dir",
                root / "exported_hf",
                "--vllm-draft-dir",
                root / "vllm_draft",
                "--verifier-config-dir",
                root / "verifier_config",
                "--container-preflight-json",
                reports / "container_preflight_analysis.json",
                "--nemo-rl-specdec-json",
                reports / "nemo_rl_specdec_integration.json",
                "--nemo-rl-drift-json",
                reports / "nemo_rl_eagle3_drift.json",
                "--modelopt-loss-mask-json",
                reports / "modelopt_loss_mask_patch.json",
                "--rollout-capture-analysis-json",
                rollout_state,
                "--rollout-submit-preflight-json",
                reports / "rollout_capture_submit_preflight.json",
                "--corpus-strategy-json",
                reports / "corpus_strategy.json",
                "--training-scale-json",
                reports / "eagle3_training_scale.json",
                "--pipeline-submit-preflight-json",
                reports / "eagle3_pipeline_submit_preflight.json",
                "--chat-template",
                root / "templates/qwen3_generation_template.jinja2",
                "--modelopt-dir",
                modelopt_dir,
                "--reference-arch",
                root / "architecture/eagle3_architecture.json",
                "--arch-env-file",
                root / "architecture/eagle3_architecture.env",
                "--sbatch-account",
                sbatch_account,
                "--skip-dry-run",
                "--json-out",
                reports / "eagle3_readiness.json",
                "--markdown-out",
                reports / "eagle3_readiness.md",
            ],
        ),
        (
            "check_pipeline_gated_submit",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py",
                "--artifact-root",
                root,
                "--preflight-json",
                reports / "eagle3_pipeline_submit_preflight.json",
                "--json-out",
                reports / "eagle3_pipeline_gated_submit.json",
                "--markdown-out",
                reports / "eagle3_pipeline_gated_submit.md",
                "--exit-zero-if-not-ready",
            ],
        ),
        (
            "check_full_swegym_after_smoke_gate",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/submit_full_rollout_after_smoke_if_ready.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "full_swegym_after_smoke_gate.json",
                "--markdown-out",
                reports / "full_swegym_after_smoke_gate.md",
            ],
        ),
        (
            "validate_full_swegym_after_smoke_gate",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_full_rollout_gate.py",
                "--json-out",
                reports / "full_swegym_after_smoke_gate_validation.json",
                "--markdown-out",
                reports / "full_swegym_after_smoke_gate_validation.md",
            ],
        ),
        (
            "validate_pipeline_gated_submit_contract",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_pipeline_gated_submit_contract.py",
                "--json-out",
                reports / "eagle3_pipeline_gated_submit_contract.json",
                "--markdown-out",
                reports / "eagle3_pipeline_gated_submit_contract.md",
            ],
        ),
        (
            "validate_pipeline_submit_preflight_contract",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_pipeline_submit_preflight_contract.py",
                "--json-out",
                reports / "eagle3_pipeline_submit_preflight_contract.json",
                "--markdown-out",
                reports / "eagle3_pipeline_submit_preflight_contract.md",
            ],
        ),
        (
            "validate_rollout_submit_preflight_contract",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_rollout_submit_preflight_contract.py",
                "--json-out",
                reports / "rollout_submit_preflight_contract.json",
                "--markdown-out",
                reports / "rollout_submit_preflight_contract.md",
            ],
        ),
        (
            "plan_next_actions",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py",
                "--artifact-root",
                root,
                "--json-out",
                reports / "eagle3_next_actions.json",
                "--markdown-out",
                reports / "eagle3_next_actions.md",
            ],
        ),
        (
            "validate_next_action_plan",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py",
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--json-out",
                reports / "eagle3_next_actions_validation.json",
                "--markdown-out",
                reports / "eagle3_next_actions_validation.md",
            ],
        ),
        (
            "build_training_path_manifest",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/build_eagle3_training_path_manifest.py",
                "--artifact-root",
                root,
                "--modelopt-dir",
                modelopt_dir,
                "--json-out",
                reports / "eagle3_training_path_manifest.json",
                "--markdown-out",
                reports / "eagle3_training_path_manifest.md",
            ],
        ),
        (
            "validate_training_path_manifest",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_training_path_manifest.py",
                "--json-out",
                reports / "eagle3_training_path_manifest_validation.json",
                "--markdown-out",
                reports / "eagle3_training_path_manifest_validation.md",
            ],
        ),
        (
            "validate_next_action_transitions",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py",
                "--json-out",
                reports / "eagle3_next_action_transitions.json",
                "--markdown-out",
                reports / "eagle3_next_action_transitions.md",
            ],
        ),
        (
            "validate_operator_queue_transitions",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_operator_queue_transitions.py",
                "--json-out",
                reports / "eagle3_operator_queue_transitions.json",
                "--markdown-out",
                reports / "eagle3_operator_queue_transitions.md",
            ],
        ),
        (
            "validate_megatron_probe_followup",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_megatron_probe_followup.py",
                "--json-out",
                reports / "megatron_probe_followup_validation.json",
                "--markdown-out",
                reports / "megatron_probe_followup_validation.md",
            ],
        ),
        (
            "create_operator_sheet",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py",
                "--artifact-root",
                root,
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--json-out",
                reports / "eagle3_operator_sheet.json",
                "--markdown-out",
                reports / "eagle3_operator_sheet.md",
            ],
        ),
        (
            "validate_operator_sheet",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py",
                "--artifact-root",
                root,
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--json-out",
                reports / "eagle3_operator_sheet_validation.json",
                "--markdown-out",
                reports / "eagle3_operator_sheet_validation.md",
            ],
        ),
        (
            "validate_operator_execution",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py",
                "--artifact-root",
                root,
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--json-out",
                reports / "eagle3_operator_execution.json",
                "--markdown-out",
                reports / "eagle3_operator_execution.md",
            ],
        ),
        (
            "validate_operator_followups",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_operator_followups.py",
                "--artifact-root",
                root,
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--json-out",
                reports / "eagle3_operator_followups_validation.json",
                "--markdown-out",
                reports / "eagle3_operator_followups_validation.md",
            ],
        ),
        (
            "audit_goal_evidence",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py",
                "--artifact-root",
                root,
                "--rollout-state-json",
                rollout_state,
                "--megatron-probe-followup-validation-json",
                reports / "megatron_probe_followup_validation.json",
                "--preflight-robustness-validation-json",
                reports / "eagle3_preflight_robustness_validation.json",
                "--json-out",
                reports / "eagle3_goal_evidence.json",
                "--markdown-out",
                reports / "eagle3_goal_evidence.md",
            ],
        ),
        (
            "create_operator_submit_packet",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/create_eagle3_operator_submit_packet.py",
                "--artifact-root",
                root,
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--operator-sheet-validation-json",
                reports / "eagle3_operator_sheet_validation.json",
                "--operator-followup-validation-json",
                reports / "eagle3_operator_followups_validation.json",
                "--operator-execution-json",
                reports / "eagle3_operator_execution.json",
                "--goal-evidence-json",
                reports / "eagle3_goal_evidence.json",
                "--json-out",
                reports / "eagle3_operator_submit_packet.json",
                "--markdown-out",
                reports / "eagle3_operator_submit_packet.md",
            ],
        ),
        (
            "validate_operator_submit_packet",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_operator_submit_packet.py",
                "--artifact-root",
                root,
                "--operator-submit-packet-json",
                reports / "eagle3_operator_submit_packet.json",
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--operator-sheet-validation-json",
                reports / "eagle3_operator_sheet_validation.json",
                "--operator-followup-validation-json",
                reports / "eagle3_operator_followups_validation.json",
                "--operator-execution-json",
                reports / "eagle3_operator_execution.json",
                "--json-out",
                reports / "eagle3_operator_submit_packet_validation.json",
                "--markdown-out",
                reports / "eagle3_operator_submit_packet_validation.md",
            ],
        ),
        (
            "preflight_operator_ready_submit",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
                "--artifact-root",
                root,
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--operator-submit-packet-validation-json",
                reports / "eagle3_operator_submit_packet_validation.json",
                "--rollout-submit-preflight-json",
                reports / "rollout_capture_submit_preflight.json",
                "--json-out",
                reports / "eagle3_operator_ready_submit_preflight.json",
                "--markdown-out",
                reports / "eagle3_operator_ready_submit_preflight.md",
            ],
        ),
        (
            "preflight_operator_safe_actions",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py",
                "--artifact-root",
                root,
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--operator-submit-packet-validation-json",
                reports / "eagle3_operator_submit_packet_validation.json",
                "--action-ids",
                "probe_remote_hosts",
                "poll_megatron_compat_probe",
                "--allow-missing-action-ids",
                "--no-require-slurm",
                "--json-out",
                reports / "eagle3_operator_safe_actions_preflight.json",
                "--markdown-out",
                reports / "eagle3_operator_safe_actions_preflight.md",
            ],
        ),
        (
            "summarize_operator_queue",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/summarize_eagle3_operator_queue.py",
                "--artifact-root",
                root,
                "--plan-json",
                reports / "eagle3_next_actions.json",
                "--operator-sheet-json",
                reports / "eagle3_operator_sheet.json",
                "--operator-execution-json",
                reports / "eagle3_operator_execution.json",
                "--operator-followup-validation-json",
                reports / "eagle3_operator_followups_validation.json",
                "--operator-ready-submit-preflight-json",
                reports / "eagle3_operator_ready_submit_preflight.json",
                "--json-out",
                reports / "eagle3_operator_queue.json",
                "--markdown-out",
                reports / "eagle3_operator_queue.md",
            ],
        ),
    ]
    if not args.skip_completion_audit:
        steps.append(
            (
                "audit_completion",
                [
                    "python3",
                    "experiments/eagle3_qwen3_235b/audit_eagle3_completion.py",
                    "--artifact-root",
                    root,
                    "--rollout-state-json",
                    rollout_state,
                    "--markdown-out",
                    reports / "eagle3_completion_audit.md",
                    "--json-out",
                    reports / "eagle3_completion_audit.json",
                ],
            )
        )
    steps.append(
        (
            "update_status_snapshot",
            [
                "python3",
                "experiments/eagle3_qwen3_235b/update_specdec_status_snapshot.py",
                "--artifact-root",
                root,
                "--allow-missing-jobs",
            ],
        )
    )
    return steps


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Operator State Refresh",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Artifact root: `{payload['artifact_root']}`",
        "",
        "| step | returncode | seconds |",
        "| --- | ---: | ---: |",
    ]
    for step in payload["steps"]:
        lines.append(f"| {step['name']} | {step['returncode']} | {step['duration_seconds']} |")
    lines += ["", "## Report Statuses", "", "| report | status |", "| --- | --- |"]
    for name, report in payload["reports"].items():
        lines.append(f"| {name} | `{report['status']}` |")
    failures = [step for step in payload["steps"] if step["returncode"] != 0]
    if failures:
        lines += ["", "## Failures", ""]
        for step in failures:
            lines += [f"### {step['name']}", "", "```text", step.get("output_tail") or "", "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.json_out is None:
        args.json_out = args.artifact_root / "reports/eagle3_operator_state_refresh.json"
    if args.markdown_out is None:
        args.markdown_out = args.artifact_root / "reports/eagle3_operator_state_refresh.md"

    steps = [run_step(name, command) for name, command in build_steps(args)]
    hard_failures = [step for step in steps if step["returncode"] != 0 and step["name"] not in SOFT_NONZERO_STEPS]
    soft_nonzero = [step for step in steps if step["returncode"] != 0 and step["name"] in SOFT_NONZERO_STEPS]
    overall = "fail" if hard_failures else ("warn" if soft_nonzero else "pass")
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "artifact_root": str(args.artifact_root),
        "steps": steps,
        "soft_nonzero_steps": [step["name"] for step in soft_nonzero],
        "reports": report_summary(args.artifact_root),
    }
    markdown = render_markdown(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if args.fail_on_error and hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
