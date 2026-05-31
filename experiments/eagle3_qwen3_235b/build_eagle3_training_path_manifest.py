#!/usr/bin/env python3
"""Build a machine-readable Qwen3-235B Eagle3 training-path manifest.

This is a no-submit report. It ties together the verifier-derived architecture,
ModelOpt wrappers, Hayate reference reports, rollout corpus gates, runtime
preflights, and final trained-draft validation gates so the operator handoff has
one canonical path definition separate from the final completion audit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
ARTIFACT_CLOSURE_ACTIONS = {
    "rollout_conversation_corpus": [
        "submit_rollout_capture",
        "rollout_poll",
        "rollout_materialize",
        "rollout_materialize_and_refresh",
    ],
    "verifier_hidden_states": [
        "run_pipeline_submit_preflight",
        "submit_eagle3_pilot_pipeline",
        "run_post_export_artifact_validations",
    ],
    "modelopt_checkpoint": [
        "run_pipeline_submit_preflight",
        "submit_eagle3_pilot_pipeline",
        "run_post_export_artifact_validations",
    ],
    "hf_eagle3_export": [
        "run_pipeline_submit_preflight",
        "submit_eagle3_pilot_pipeline",
        "run_post_export_artifact_validations",
    ],
    "vllm_eagle3_draft": [
        "run_pipeline_submit_preflight",
        "submit_eagle3_pilot_pipeline",
        "run_post_export_artifact_validations",
    ],
    "rl_vllm_draft_validation": [
        "submit_trained_draft_spec_tokens_sweep",
    ],
}


def parse_args() -> argparse.Namespace:
    artifact = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact)
    parser.add_argument("--modelopt-dir", type=Path, default=ROOT / "Model-Optimizer")
    parser.add_argument("--playbook", type=Path, default=EXP / "EAGLE3_DRAFT_MODEL_PLAYBOOK.md")
    parser.add_argument("--reference-arch", type=Path, default=EXP / "qwen3_235b_thinking_eagle3_architecture.json")
    parser.add_argument("--static-inputs-validation-json", type=Path)
    parser.add_argument("--remote-host-probe-json", type=Path)
    parser.add_argument("--remote-access-diagnostics-json", type=Path)
    parser.add_argument("--hayate-workflow-json", type=Path)
    parser.add_argument("--hayate-specforge-reference-json", type=Path)
    parser.add_argument("--draft-inventory-json", type=Path)
    parser.add_argument("--upstream-drift-json", type=Path)
    parser.add_argument("--modelopt-loss-mask-json", type=Path)
    parser.add_argument("--modelopt-recipe-overrides-json", type=Path)
    parser.add_argument("--corpus-strategy-json", type=Path)
    parser.add_argument("--training-scale-json", type=Path)
    parser.add_argument("--next-action-plan-json", type=Path)
    parser.add_argument("--rollout-submit-preflight-json", type=Path)
    parser.add_argument("--rollout-state-json", type=Path)
    parser.add_argument("--container-preflight-json", type=Path)
    parser.add_argument("--vllm-source-build-json", type=Path)
    parser.add_argument("--vllm-abi-probe-json", type=Path)
    parser.add_argument("--megatron-compat-json", type=Path)
    parser.add_argument("--pipeline-submit-preflight-json", type=Path)
    parser.add_argument("--pipeline-analysis-json", type=Path)
    parser.add_argument("--hidden-validation-json", type=Path)
    parser.add_argument("--training-checkpoint-json", type=Path)
    parser.add_argument("--export-artifacts-json", type=Path)
    parser.add_argument("--sweep-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def with_defaults(args: argparse.Namespace) -> argparse.Namespace:
    reports = args.artifact_root / "reports"
    defaults = {
        "static_inputs_validation_json": reports / "qwen3_static_inputs_materialization_validation.json",
        "remote_host_probe_json": reports / "eagle3_remote_host_probe.json",
        "remote_access_diagnostics_json": reports / "eagle3_remote_access_diagnostics.json",
        "hayate_workflow_json": reports / "hayate_modelopt_workflow.json",
        "hayate_specforge_reference_json": reports / "hayate_specforge_reference.json",
        "draft_inventory_json": reports / "eagle3_draft_config_inventory.json",
        "upstream_drift_json": reports / "modelopt_upstream_drift.json",
        "modelopt_loss_mask_json": reports / "modelopt_loss_mask_patch.json",
        "modelopt_recipe_overrides_json": reports / "modelopt_recipe_overrides_current.json",
        "corpus_strategy_json": reports / "corpus_strategy.json",
        "training_scale_json": reports / "eagle3_training_scale.json",
        "next_action_plan_json": reports / "eagle3_next_actions.json",
        "rollout_submit_preflight_json": reports / "rollout_capture_submit_preflight.json",
        "rollout_state_json": reports / "rollout_capture_state_advance.json",
        "container_preflight_json": reports / "container_preflight_analysis.json",
        "vllm_source_build_json": reports / "vllm_native_source_build.json",
        "vllm_abi_probe_json": reports / "vllm_native_abi_probe.json",
        "megatron_compat_json": reports / "megatron_compat_probe.json",
        "pipeline_submit_preflight_json": reports / "eagle3_pipeline_submit_preflight.json",
        "pipeline_analysis_json": reports / "eagle3_pipeline_analysis.json",
        "hidden_validation_json": args.artifact_root / "hidden_states/validation_summary.json",
        "training_checkpoint_json": reports / "eagle3_training_checkpoint.json",
        "export_artifacts_json": reports / "eagle3_export_artifacts.json",
        "sweep_json": reports / "trained_draft_spec_tokens_sweep.json",
        "json_out": reports / "eagle3_training_path_manifest.json",
        "markdown_out": reports / "eagle3_training_path_manifest.md",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


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
    decision = payload.get("decision") if isinstance((payload or {}).get("decision"), dict) else {}
    return str(
        (payload or {}).get("overall_status")
        or (payload or {}).get("status")
        or decision.get("overall_status")
        or "unknown"
    )


def ready_actions(plan: dict[str, Any] | None) -> list[str]:
    return [str(item["id"]) for item in ready_operator_actions(plan) if item.get("id")]


def ready_operator_actions(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (plan or {}).get("next_actions") or []:
        if (
            not isinstance(item, dict)
            or not item.get("id")
            or item.get("status") != "ready_for_operator"
            or not item.get("command")
        ):
            continue
        rows.append(
            {
                "order": len(rows) + 1,
                "id": str(item.get("id")),
                "title": item.get("title"),
                "stage": item.get("stage"),
                "status": item.get("status"),
                "submits_slurm": item.get("submits_slurm"),
                "heavy_gpu": item.get("heavy_gpu"),
                "report": item.get("report"),
                "reason": item.get("reason"),
                "command_present": True,
            }
        )
    return rows


def plan_actions_by_id(plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in (plan or {}).get("next_actions") or []:
        if isinstance(item, dict) and item.get("id"):
            result[str(item["id"])] = item
    return result


def report_ref(path: Path, payload: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": status_of(payload, error),
        "exists": error is None,
        "error": error,
    }


def gate(
    gate_id: str,
    title: str,
    report: dict[str, Any],
    pass_statuses: set[str],
    proof_required: str,
    next_action_ids: list[str] | None = None,
) -> dict[str, Any]:
    status = "proven" if report["status"] in pass_statuses else "open"
    return {
        "id": gate_id,
        "title": title,
        "status": status,
        "report_status": report["status"],
        "report_path": report["path"],
        "proof_required": proof_required,
        "next_action_ids": next_action_ids or [],
    }


def combined_report_status(reports: list[dict[str, Any]], pass_statuses: set[str]) -> str:
    if all(report["status"] in pass_statuses for report in reports):
        return "pass"
    if any(report["exists"] for report in reports):
        return "incomplete"
    return "missing"


def combined_report(path_label: str, reports: list[dict[str, Any]], pass_statuses: set[str]) -> dict[str, Any]:
    return {
        "path": path_label,
        "status": combined_report_status(reports, pass_statuses),
        "exists": any(report["exists"] for report in reports),
        "component_statuses": {report["path"]: report["status"] for report in reports},
    }


def nested(payload: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def contract_report(
    label: str,
    report: dict[str, Any],
    accepted_statuses: set[str],
    proof: str,
) -> dict[str, Any]:
    current_status = str(report.get("status") or "unknown")
    accepted = sorted(accepted_statuses)
    return {
        "label": label,
        "path": report.get("path"),
        "exists": bool(report.get("exists")),
        "current_status": current_status,
        "accepted_statuses": accepted,
        "status": "pass" if current_status in accepted_statuses else "open",
        "proof": proof,
    }


def contract_condition(condition_id: str, value: Any, expected: Any, proof: str) -> dict[str, Any]:
    return {
        "id": condition_id,
        "current_value": value,
        "expected_value": expected,
        "status": "pass" if value == expected else "open",
        "proof": proof,
    }


def summarize_contract_actions(
    candidate_ids: list[str],
    plan: dict[str, Any] | None,
    ready_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = plan_actions_by_id(plan)
    rows: list[dict[str, Any]] = []
    for action_id in candidate_ids:
        action = by_id.get(action_id)
        if not action:
            rows.append(
                {
                    "id": action_id,
                    "status": "not_currently_selected",
                    "command_present": False,
                }
            )
            continue
        rows.append(
            {
                "id": action_id,
                "current_ready_order": (ready_by_id.get(action_id) or {}).get("order"),
                "title": action.get("title"),
                "stage": action.get("stage"),
                "status": action.get("status"),
                "submits_slurm": action.get("submits_slurm"),
                "heavy_gpu": action.get("heavy_gpu"),
                "command_present": bool(action.get("command")),
                "report": action.get("report"),
                "reason": action.get("reason"),
            }
        )
    return rows


def gate_by_id(stages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in stages if item.get("id")}


def build_gate_closure_contracts(
    stages: list[dict[str, Any]],
    refs: dict[str, dict[str, Any]],
    plan: dict[str, Any] | None,
    reference_evidence: dict[str, Any],
    rollout_state: dict[str, Any] | None,
    corpus_strategy: dict[str, Any] | None,
    pipeline_submit: dict[str, Any] | None,
    pipeline_analysis: dict[str, Any] | None,
    ready_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    stages_by_id = gate_by_id(stages)
    remote = reference_evidence.get("remote_probe") if isinstance(reference_evidence.get("remote_probe"), dict) else {}

    definitions: list[dict[str, Any]] = [
        {
            "id": "reference_and_architecture",
            "required_reports": [
                contract_report("static_inputs", refs["static_inputs"], {"pass"}, "Verifier config, tokenizer, chat template, and Eagle3 architecture are materialized."),
                contract_report("hayate_modelopt_workflow", refs["hayate_modelopt_workflow"], {"pass", "reference_only", "warn"}, "Hayate workflow evidence is present and classified as reference-only."),
                contract_report("hayate_specforge_reference", refs["hayate_specforge_reference"], {"pass", "reference_only", "warn"}, "SpecForge Qwen3-235B reference comparison is present."),
                contract_report("draft_inventory", refs["draft_inventory"], {"pass", "warn"}, "Existing draft artifacts have been inventoried for comparison only."),
                contract_report("upstream_drift", refs["upstream_drift"], {"pass", "warn"}, "ModelOpt source-of-truth and upstream drift decision is recorded."),
            ],
            "conditions": [],
            "do_not_proceed_guards": [
                "Do not import Hayate checkout changes wholesale; use local/remote ModelOpt as source of truth.",
            ],
        },
        {
            "id": "remote_hayate_reference_probe",
            "required_reports": [
                contract_report("remote_host_probe", refs["remote_host_probe"], {"pass"}, "SSH probe reaches at least one configured cluster host."),
                contract_report("remote_access_diagnostics", refs["remote_access_diagnostics"], {"pass"}, "Remote access diagnostics do not indicate local DNS or SSH blockage."),
            ],
            "conditions": [
                contract_condition("remote_reference_proven", reference_evidence.get("remote_reference_proven"), True, "Live remote Hayate ModelOpt, SpecForge, remote workdir, and artifact root paths are visible."),
                contract_condition("hayate_modelopt_remote_path_visible", (reference_evidence.get("hayate_modelopt") or {}).get("remote_path_visible"), True, "Remote host can see the Hayate ModelOpt reference path."),
                contract_condition("hayate_specforge_remote_path_visible", (reference_evidence.get("hayate_specforge") or {}).get("remote_path_visible"), True, "Remote host can see the Hayate SpecForge reference path."),
                contract_condition("remote_workdir_visible", remote.get("remote_workdir_visible"), True, "Remote host can see the target workdir."),
                contract_condition("remote_artifact_root_visible", remote.get("remote_artifact_root_visible"), True, "Remote host can see the artifact root."),
            ],
            "do_not_proceed_guards": [
                "Do not treat bundled Hayate/SpecForge snapshots as live remote proof.",
                "If diagnostics are blocked_local_dns, fix VPN/DNS or run the probe from the cluster login host before declaring remote paths absent.",
            ],
        },
        {
            "id": "modelopt_loss_and_recipe",
            "required_reports": [
                contract_report("modelopt_loss_mask", refs["modelopt_loss_mask"], {"pass"}, "TRT-LLM hidden-state dump preserves answer-only loss_mask."),
                contract_report("modelopt_recipe_overrides", refs["modelopt_recipe_overrides"], {"pass"}, "Eagle3 recipe overrides match the Qwen3-235B Thinking architecture."),
            ],
            "conditions": [],
            "do_not_proceed_guards": [
                "Do not train on unmasked thinking tokens or with mismatched aux-layer architecture overrides.",
            ],
        },
        {
            "id": "target_rollout_corpus",
            "required_reports": [
                contract_report("rollout_state", refs["rollout_state"], {"pass"}, "Rollout state advance selected a pipeline-ready materialized corpus."),
                contract_report("corpus_strategy", refs["corpus_strategy"], {"pass"}, "Corpus strategy proves actual Qwen3 SWE/RL rollout provenance."),
            ],
            "conditions": [
                contract_condition("rollout_state_next_step", nested(rollout_state, ["decision", "next_step"]) or (rollout_state or {}).get("next_step"), "pipeline_dry_run", "Rollout state should hand off to the hidden-state pipeline dry-run."),
                contract_condition("corpus_primary_source", nested(corpus_strategy, ["decision", "primary_source"]) or (corpus_strategy or {}).get("primary_source"), "actual_rl_rollout", "Training corpus must come from actual RL rollout conversations, not synthetic fallback data."),
            ],
            "do_not_proceed_guards": [
                "Do not run hidden-state dump or ModelOpt training against placeholder, synthetic, or missing conversation JSONL.",
                "Do not treat a submitted rollout job as a corpus until materialization and validation reports PASS.",
            ],
        },
        {
            "id": "runtime_container",
            "required_reports": [
                contract_report("container_preflight", refs["container_preflight"], {"pass"}, "Selected sqsh/container can import ModelOpt and render the Qwen3 chat template."),
                contract_report("vllm_source_build", refs["vllm_source_build"], {"pass"}, "Source-built vLLM completed inside the target NeMo container."),
                contract_report("vllm_abi_probe", refs["vllm_abi_probe"], {"pass"}, "Source-built vLLM passes native ABI imports for the target Torch/CUDA stack."),
                contract_report("megatron_compat", refs["megatron_compat"], {"pass"}, "Megatron-Bridge Qwen3MoE compatibility probe passes before rollout capture."),
            ],
            "conditions": [],
            "do_not_proceed_guards": [
                "Do not submit rollout capture or hidden-state dump while vLLM ABI, Megatron compatibility, or container preflight is unproven.",
            ],
        },
        {
            "id": "hidden_train_export_submit",
            "required_reports": [
                contract_report("pipeline_submit_preflight", refs["pipeline_submit_preflight"], {"pass"}, "Pipeline submit preflight reports PASS."),
                contract_report("pipeline_analysis", refs["pipeline_analysis"], {"pass"}, "Hidden-state dump, ModelOpt train, and export chain analysis reports PASS after gated submit."),
            ],
            "conditions": [
                contract_condition("pipeline_submit_ready", bool((pipeline_submit or {}).get("submit_ready")), True, "Pipeline preflight must set submit_ready=true."),
                contract_condition("gated_pilot_submit_command_present", bool(nested(pipeline_submit, ["commands", "gated_pilot_submit"])), True, "Pipeline preflight must emit commands.gated_pilot_submit for the guarded submit helper."),
                contract_condition("pipeline_analysis_pass", status_of(pipeline_analysis), "pass", "Pipeline analyzer must prove the gated Slurm chain completed successfully."),
            ],
            "do_not_proceed_guards": [
                "Do not run post-export artifact validations until the gated hidden/train/export chain has a PASS pipeline analysis.",
            ],
        },
        {
            "id": "trained_artifact_contracts",
            "required_reports": [
                contract_report("hidden_validation", refs["hidden_validation"], {"pass"}, "Hidden-state tensor validation passes."),
                contract_report("training_checkpoint", refs["training_checkpoint"], {"pass"}, "ModelOpt training checkpoint contract passes."),
                contract_report("export_artifacts", refs["export_artifacts"], {"pass"}, "HF and vLLM export artifact contracts pass."),
                contract_report("trained_draft_sweep", refs["trained_draft_sweep"], {"pass"}, "Trained draft spec-token sweep passes acceptance and speed checks."),
            ],
            "conditions": [],
            "do_not_proceed_guards": [
                "Do not claim a usable Qwen3-235B Eagle3 draft until checkpoint, export, and trained-draft sweep contracts all PASS.",
            ],
        },
    ]

    contracts: list[dict[str, Any]] = []
    for definition in definitions:
        stage = stages_by_id.get(str(definition["id"]), {})
        reports = definition["required_reports"]
        conditions = definition["conditions"]
        missing = [
            f"report:{item['label']}"
            for item in reports
            if item.get("status") != "pass"
        ] + [
            f"condition:{item['id']}"
            for item in conditions
            if item.get("status") != "pass"
        ]
        candidate_ids = list(stage.get("next_action_ids") or [])
        contracts.append(
            {
                "id": definition["id"],
                "title": stage.get("title"),
                "current_gate_status": stage.get("status"),
                "current_report_status": stage.get("report_status"),
                "closed": not missing,
                "required_reports": reports,
                "required_conditions": conditions,
                "closure_evidence_missing": missing,
                "candidate_next_action_ids": candidate_ids,
                "operator_actions": summarize_contract_actions(candidate_ids, plan, ready_by_id),
                "do_not_proceed_guards": definition["do_not_proceed_guards"],
            }
        )
    return contracts


def build_operator_gate_action_matrix(
    contracts: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ready_by_id = {str(item["id"]): item for item in ready_rows if item.get("id")}
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        candidate_ids = [str(item) for item in contract.get("candidate_next_action_ids") or [] if item]
        current_ids = [action_id for action_id in candidate_ids if action_id in ready_by_id]
        future_ids = [action_id for action_id in candidate_ids if action_id not in ready_by_id]
        if contract.get("closed") is True:
            status = "closed"
        elif current_ids:
            status = "ready_action_available"
        else:
            status = "waiting_for_prior_gate_or_report"
        rows.append(
            {
                "gate_id": contract.get("id"),
                "gate_closed": bool(contract.get("closed")),
                "status": status,
                "missing_evidence": contract.get("closure_evidence_missing") or [],
                "current_ready_action_ids": current_ids,
                "future_candidate_action_ids": future_ids,
                "current_ready_actions": [ready_by_id[action_id] for action_id in current_ids],
            }
        )
    return rows


def path_visible(path_text: str | None) -> bool:
    if not path_text:
        return False
    try:
        return Path(path_text).exists()
    except (OSError, ValueError):
        return False


def report_statuses(refs: dict[str, dict[str, Any]], labels: list[str]) -> dict[str, str]:
    return {label: str(refs.get(label, {}).get("status") or "missing") for label in labels}


def closure_action_fields(artifact_id: str, ready_action_ids: set[str]) -> dict[str, list[str]]:
    action_ids = ARTIFACT_CLOSURE_ACTIONS[artifact_id]
    return {
        "closure_action_ids": action_ids,
        "current_closure_action_ids": [action_id for action_id in action_ids if action_id in ready_action_ids],
        "future_closure_action_ids": [action_id for action_id in action_ids if action_id not in ready_action_ids],
    }


def build_artifact_flow(
    args: argparse.Namespace,
    refs: dict[str, dict[str, Any]],
    rollout_state: dict[str, Any] | None,
    corpus_strategy: dict[str, Any] | None,
    ready_action_ids: set[str],
) -> list[dict[str, Any]]:
    corpus_path = first_str(
        nested(corpus_strategy, ["decision", "provenance", "input_data_path"]),
        nested(corpus_strategy, ["provenance", "input_data_path"]),
        nested(rollout_state, ["output_data", "path"]),
        nested(rollout_state, ["artifacts", "output_data", "path"]),
        str(args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
    )
    hidden_dir = str(args.artifact_root / "hidden_states")
    modelopt_ckpt = str(args.artifact_root / "modelopt_ckpt")
    hf_export = str(args.artifact_root / "exported_hf")
    vllm_draft = str(args.artifact_root / "vllm_draft")

    corpus_ready = (
        refs["rollout_state"]["status"] == "pass"
        and refs["corpus_strategy"]["status"] == "pass"
        and (nested(corpus_strategy, ["decision", "primary_source"]) or (corpus_strategy or {}).get("primary_source"))
        == "actual_rl_rollout"
        and (nested(rollout_state, ["decision", "next_step"]) or (rollout_state or {}).get("next_step"))
        == "pipeline_dry_run"
    )

    nodes: list[dict[str, Any]] = [
        {
            "id": "rollout_conversation_corpus",
            "artifact_type": "jsonl_conversations",
            "path": corpus_path,
            "producer_gate": "target_rollout_corpus",
            "consumer_gate": "hidden_train_export_submit",
            "required_reports": ["rollout_state", "corpus_strategy"],
            "required_invariants": ["primary_source=actual_rl_rollout", "rollout_state_next_step=pipeline_dry_run"],
            **closure_action_fields("rollout_conversation_corpus", ready_action_ids),
            "report_statuses": report_statuses(refs, ["rollout_state", "corpus_strategy"]),
            "proof_status": "pass" if corpus_ready else "open",
            "path_visible": path_visible(corpus_path),
        },
        {
            "id": "verifier_hidden_states",
            "artifact_type": "modelopt_hidden_state_tensors",
            "path": hidden_dir,
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "hidden_validation"],
            "required_invariants": ["answer_only_loss_mask_preserved", "positive_loss_mask_files>0"],
            **closure_action_fields("verifier_hidden_states", ready_action_ids),
            "report_statuses": report_statuses(refs, ["pipeline_analysis", "hidden_validation"]),
            "proof_status": "pass"
            if refs["pipeline_analysis"]["status"] == "pass" and refs["hidden_validation"]["status"] == "pass"
            else "open",
            "path_visible": path_visible(hidden_dir),
        },
        {
            "id": "modelopt_checkpoint",
            "artifact_type": "modelopt_eagle3_checkpoint",
            "path": modelopt_ckpt,
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "training_checkpoint"],
            "required_invariants": ["offline_hidden_state_training", "qwen3_eagle3_recipe_overrides"],
            **closure_action_fields("modelopt_checkpoint", ready_action_ids),
            "report_statuses": report_statuses(refs, ["pipeline_analysis", "training_checkpoint"]),
            "proof_status": "pass"
            if refs["pipeline_analysis"]["status"] == "pass" and refs["training_checkpoint"]["status"] == "pass"
            else "open",
            "path_visible": path_visible(modelopt_ckpt),
        },
        {
            "id": "hf_eagle3_export",
            "artifact_type": "hf_draft_export",
            "path": hf_export,
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "export_artifacts"],
            "required_invariants": ["verifier_config_compatible", "thinking_2507_architecture_preserved"],
            **closure_action_fields("hf_eagle3_export", ready_action_ids),
            "report_statuses": report_statuses(refs, ["pipeline_analysis", "export_artifacts"]),
            "proof_status": "pass"
            if refs["pipeline_analysis"]["status"] == "pass" and refs["export_artifacts"]["status"] == "pass"
            else "open",
            "path_visible": path_visible(hf_export),
        },
        {
            "id": "vllm_eagle3_draft",
            "artifact_type": "vllm_draft_export",
            "path": vllm_draft,
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "export_artifacts"],
            "required_invariants": ["vllm_config_exists", "draft_weights_present"],
            **closure_action_fields("vllm_eagle3_draft", ready_action_ids),
            "report_statuses": report_statuses(refs, ["pipeline_analysis", "export_artifacts"]),
            "proof_status": "pass"
            if refs["pipeline_analysis"]["status"] == "pass" and refs["export_artifacts"]["status"] == "pass"
            else "open",
            "path_visible": path_visible(vllm_draft),
        },
        {
            "id": "rl_vllm_draft_validation",
            "artifact_type": "trained_draft_spec_tokens_sweep",
            "path": refs["trained_draft_sweep"]["path"],
            "producer_gate": "trained_artifact_contracts",
            "consumer_gate": "completion_audit",
            "required_reports": ["trained_draft_sweep"],
            "required_invariants": ["acceptance_gate_pass", "speed_gate_pass", "no_reward_or_malformed_regression"],
            **closure_action_fields("rl_vllm_draft_validation", ready_action_ids),
            "report_statuses": report_statuses(refs, ["trained_draft_sweep"]),
            "proof_status": "pass" if refs["trained_draft_sweep"]["status"] == "pass" else "open",
            "path_visible": path_visible(refs["trained_draft_sweep"]["path"]),
        },
    ]
    return nodes


def first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def remote_path_visible(remote_probe: dict[str, Any] | None, path: str | None) -> bool:
    if not remote_probe or not path:
        return False
    for host in remote_probe.get("hosts") or []:
        if not isinstance(host, dict) or not host.get("reachable"):
            continue
        for item in host.get("paths") or []:
            if isinstance(item, dict) and item.get("path") == path and item.get("exists"):
                return True
    return False


def build_reference_evidence(
    args: argparse.Namespace,
    refs: dict[str, dict[str, Any]],
    remote_probe: dict[str, Any] | None,
    remote_diagnostics: dict[str, Any] | None,
    hayate_workflow: dict[str, Any] | None,
    hayate_specforge: dict[str, Any] | None,
) -> dict[str, Any]:
    hayate_path = first_str(
        (hayate_workflow or {}).get("selected_path"),
        ((hayate_workflow or {}).get("path") or {}).get("chosen")
        if isinstance((hayate_workflow or {}).get("path"), dict)
        else None,
    )
    specforge_path = first_str(
        (hayate_specforge or {}).get("specforge_dir"),
        (hayate_specforge or {}).get("requested_specforge_dir"),
    )
    remote_workdir = first_str((remote_probe or {}).get("remote_workdir"))
    remote_artifact_root = first_str((remote_probe or {}).get("artifact_root"))
    hayate_path_visible = remote_path_visible(remote_probe, hayate_path)
    specforge_path_visible = remote_path_visible(remote_probe, specforge_path)
    remote_workdir_visible = remote_path_visible(remote_probe, remote_workdir)
    remote_artifact_root_visible = remote_path_visible(remote_probe, remote_artifact_root)
    remote_reference_proven = (
        refs["remote_host_probe"]["status"] == "pass"
        and hayate_path_visible
        and specforge_path_visible
        and remote_workdir_visible
        and remote_artifact_root_visible
    )
    return {
        "local_modelopt": {
            "path": str(args.modelopt_dir),
            "exists": args.modelopt_dir.exists(),
            "role": "training_source",
        },
        "remote_probe": {
            "path": refs["remote_host_probe"]["path"],
            "status": refs["remote_host_probe"]["status"],
            "diagnostics_status": refs["remote_access_diagnostics"]["status"],
            "diagnosis": (remote_diagnostics or {}).get("diagnosis"),
            "configuration_findings": (remote_diagnostics or {}).get("configuration_findings") or [],
            "reachable_hosts": (remote_probe or {}).get("reachable_hosts") or [],
            "remote_workdir": remote_workdir,
            "remote_workdir_visible": remote_workdir_visible,
            "remote_artifact_root": remote_artifact_root,
            "remote_artifact_root_visible": remote_artifact_root_visible,
        },
        "hayate_modelopt": {
            "path": hayate_path,
            "report_status": refs["hayate_modelopt_workflow"]["status"],
            "source": (hayate_workflow or {}).get("source"),
            "live_visible": bool((hayate_workflow or {}).get("live_hayate_visible")),
            "remote_path_visible": hayate_path_visible,
            "role": "reference_only",
        },
        "hayate_specforge": {
            "path": specforge_path,
            "report_status": refs["hayate_specforge_reference"]["status"],
            "source": (hayate_specforge or {}).get("source"),
            "live_visible": bool((hayate_specforge or {}).get("live_specforge_visible")),
            "remote_path_visible": specforge_path_visible,
            "qwen3_235b_comparison_status": (
                ((hayate_specforge or {}).get("qwen3_235b_comparison") or {}).get("status")
                if isinstance((hayate_specforge or {}).get("qwen3_235b_comparison"), dict)
                else None
            ),
            "role": "reference_only",
        },
        "remote_reference_proven": remote_reference_proven,
        "reference_policy": (
            "Use bundled Hayate/SpecForge snapshots only as reference evidence until a remote probe proves "
            "the live Hayate ModelOpt, SpecForge, remote workdir, and artifact root paths."
        ),
    }


def comparison_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    comparison = (payload or {}).get("qwen3_235b_comparison")
    if not isinstance(comparison, dict):
        return []
    rows = comparison.get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def build_reference_decisions(
    args: argparse.Namespace,
    refs: dict[str, dict[str, Any]],
    remote_probe: dict[str, Any] | None,
    remote_diagnostics: dict[str, Any] | None,
    hayate_workflow: dict[str, Any] | None,
    hayate_specforge: dict[str, Any] | None,
    upstream_drift: dict[str, Any] | None,
) -> dict[str, Any]:
    upstream_decision = (
        upstream_drift.get("decision")
        if isinstance((upstream_drift or {}).get("decision"), dict)
        else {}
    )
    rows = comparison_rows(hayate_specforge)
    matched_fields = [str(row.get("field")) for row in rows if row.get("match") is True and row.get("field")]
    rejected_fields = [
        {
            "field": row.get("field"),
            "current": row.get("current"),
            "reference": row.get("specforge"),
            "reason": "Thinking-2507 verifier-derived architecture is the source of truth for this field.",
        }
        for row in rows
        if row.get("match") is False and row.get("field")
    ]
    classification_basis = (
        hayate_workflow.get("classification_basis")
        if isinstance((hayate_workflow or {}).get("classification_basis"), dict)
        else {}
    )
    host_errors = [
        {
            "host": host.get("host"),
            "returncode": host.get("returncode"),
            "stderr": host.get("stderr"),
            "configured_hostname": (
                host.get("ssh_config", {}).get("hostname")
                if isinstance(host.get("ssh_config"), dict)
                else None
            ),
            "local_resolution": host.get("local_resolution")
            if isinstance(host.get("local_resolution"), dict)
            else None,
        }
        for host in (remote_probe or {}).get("hosts") or []
        if isinstance(host, dict) and not host.get("reachable")
    ]
    return {
        "training_route": {
            "primary_route": "fixed_exported_eagle3_draft_first",
            "first_training_mode": "modelopt_offline_hidden_states",
            "online_modelopt_role": "optional_scale_or_refresh_path_after_offline_pipeline_passes",
            "online_rl_draft_training_role": "future_neMo_rl_feature_after_fixed_draft_speed_reward_gate",
        },
        "modelopt_source": {
            "source_of_truth": upstream_decision.get("source_of_truth") or "local_modelopt",
            "local_modelopt_dir": str(args.modelopt_dir),
            "upstream_drift_report": refs["upstream_drift"]["path"],
            "upstream_drift_status": refs["upstream_drift"]["status"],
            "local_head": upstream_decision.get("local_head"),
            "upstream_head": upstream_decision.get("upstream_head"),
            "hayate_head": upstream_decision.get("hayate_head"),
            "allowed_focus_diffs": upstream_decision.get("allowed_focus_diffs") or [],
            "disallowed_focus_diffs": upstream_decision.get("disallowed_focus_diffs") or [],
            "recommendation": upstream_decision.get("recommendation"),
        },
        "hayate_workflow": {
            "role": "reference_only",
            "status": refs["hayate_modelopt_workflow"]["status"],
            "source": (hayate_workflow or {}).get("source"),
            "classification": (hayate_workflow or {}).get("classification"),
            "live_visible": bool((hayate_workflow or {}).get("live_hayate_visible")),
            "preserve_patterns": [
                "response aggregation",
                "hidden-state dump aggregation",
                "short-job Slurm chaining",
                "online ModelOpt training as an optional later path",
            ],
            "workflow_files_present": classification_basis.get("workflow_files_present") or [],
            "qwen_config_paths_present": classification_basis.get("qwen_config_paths_present") or [],
        },
        "specforge_qwen3_235b": {
            "role": "architecture_sanity_reference_only",
            "status": refs["hayate_specforge_reference"]["status"],
            "source": (hayate_specforge or {}).get("source"),
            "comparison_status": (
                ((hayate_specforge or {}).get("qwen3_235b_comparison") or {}).get("status")
                if isinstance((hayate_specforge or {}).get("qwen3_235b_comparison"), dict)
                else None
            ),
            "matched_fields": matched_fields,
            "rejected_fields": rejected_fields,
            "decision": (
                "Use SpecForge Qwen3-235B as aux-layer and shape sanity evidence only; "
                "keep Thinking-2507 verifier-derived fields as the training/export source of truth."
            ),
        },
        "remote_probe": {
            "status": refs["remote_host_probe"]["status"],
            "diagnostics_status": refs["remote_access_diagnostics"]["status"],
            "reachable_hosts": (remote_probe or {}).get("reachable_hosts") or [],
            "host_discovery": (remote_probe or {}).get("host_discovery") or {},
            "diagnosis": (remote_diagnostics or {}).get("diagnosis"),
            "configuration_findings": (remote_diagnostics or {}).get("configuration_findings") or [],
            "unreachable_host_errors": host_errors[:4],
            "decision": (
                "Remote reference proof remains open until SSH reaches a host and proves live Hayate "
                "ModelOpt, SpecForge, remote workdir, and artifact-root paths."
            ),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    static_inputs, static_error = load_json(args.static_inputs_validation_json)
    remote_probe, remote_probe_error = load_json(args.remote_host_probe_json)
    remote_diagnostics, remote_diagnostics_error = load_json(args.remote_access_diagnostics_json)
    hayate_workflow, hayate_workflow_error = load_json(args.hayate_workflow_json)
    hayate_specforge, hayate_specforge_error = load_json(args.hayate_specforge_reference_json)
    draft_inventory, draft_inventory_error = load_json(args.draft_inventory_json)
    upstream_drift, upstream_drift_error = load_json(args.upstream_drift_json)
    loss_mask, loss_mask_error = load_json(args.modelopt_loss_mask_json)
    recipe, recipe_error = load_json(args.modelopt_recipe_overrides_json)
    corpus, corpus_error = load_json(args.corpus_strategy_json)
    scale, scale_error = load_json(args.training_scale_json)
    plan, plan_error = load_json(args.next_action_plan_json)
    rollout_submit, rollout_submit_error = load_json(args.rollout_submit_preflight_json)
    rollout_state, rollout_state_error = load_json(args.rollout_state_json)
    container, container_error = load_json(args.container_preflight_json)
    source_build, source_build_error = load_json(args.vllm_source_build_json)
    abi_probe, abi_probe_error = load_json(args.vllm_abi_probe_json)
    megatron, megatron_error = load_json(args.megatron_compat_json)
    pipeline_submit, pipeline_submit_error = load_json(args.pipeline_submit_preflight_json)
    pipeline, pipeline_error = load_json(args.pipeline_analysis_json)
    hidden, hidden_error = load_json(args.hidden_validation_json)
    training_ckpt, training_ckpt_error = load_json(args.training_checkpoint_json)
    export_artifacts, export_error = load_json(args.export_artifacts_json)
    sweep, sweep_error = load_json(args.sweep_json)

    refs = {
        "static_inputs": report_ref(args.static_inputs_validation_json, static_inputs, static_error),
        "remote_host_probe": report_ref(args.remote_host_probe_json, remote_probe, remote_probe_error),
        "remote_access_diagnostics": report_ref(
            args.remote_access_diagnostics_json, remote_diagnostics, remote_diagnostics_error
        ),
        "hayate_modelopt_workflow": report_ref(args.hayate_workflow_json, hayate_workflow, hayate_workflow_error),
        "hayate_specforge_reference": report_ref(
            args.hayate_specforge_reference_json, hayate_specforge, hayate_specforge_error
        ),
        "draft_inventory": report_ref(args.draft_inventory_json, draft_inventory, draft_inventory_error),
        "upstream_drift": report_ref(args.upstream_drift_json, upstream_drift, upstream_drift_error),
        "modelopt_loss_mask": report_ref(args.modelopt_loss_mask_json, loss_mask, loss_mask_error),
        "modelopt_recipe_overrides": report_ref(args.modelopt_recipe_overrides_json, recipe, recipe_error),
        "corpus_strategy": report_ref(args.corpus_strategy_json, corpus, corpus_error),
        "training_scale": report_ref(args.training_scale_json, scale, scale_error),
        "next_action_plan": report_ref(args.next_action_plan_json, plan, plan_error),
        "rollout_submit_preflight": report_ref(args.rollout_submit_preflight_json, rollout_submit, rollout_submit_error),
        "rollout_state": report_ref(args.rollout_state_json, rollout_state, rollout_state_error),
        "container_preflight": report_ref(args.container_preflight_json, container, container_error),
        "vllm_source_build": report_ref(args.vllm_source_build_json, source_build, source_build_error),
        "vllm_abi_probe": report_ref(args.vllm_abi_probe_json, abi_probe, abi_probe_error),
        "megatron_compat": report_ref(args.megatron_compat_json, megatron, megatron_error),
        "pipeline_submit_preflight": report_ref(args.pipeline_submit_preflight_json, pipeline_submit, pipeline_submit_error),
        "pipeline_analysis": report_ref(args.pipeline_analysis_json, pipeline, pipeline_error),
        "hidden_validation": report_ref(args.hidden_validation_json, hidden, hidden_error),
        "training_checkpoint": report_ref(args.training_checkpoint_json, training_ckpt, training_ckpt_error),
        "export_artifacts": report_ref(args.export_artifacts_json, export_artifacts, export_error),
        "trained_draft_sweep": report_ref(args.sweep_json, sweep, sweep_error),
    }

    current_operator_actions = ready_operator_actions(plan)
    action_ids = [str(item["id"]) for item in current_operator_actions]
    ready_by_id = {str(item["id"]): item for item in current_operator_actions}
    reference_report = combined_report(
        "static_inputs + Hayate references + draft inventory",
        [
            refs["static_inputs"],
            refs["hayate_modelopt_workflow"],
            refs["hayate_specforge_reference"],
            refs["draft_inventory"],
            refs["upstream_drift"],
        ],
        {"pass", "reference_only", "warn"},
    )
    reference_evidence = build_reference_evidence(
        args, refs, remote_probe, remote_diagnostics, hayate_workflow, hayate_specforge
    )
    reference_decisions = build_reference_decisions(
        args, refs, remote_probe, remote_diagnostics, hayate_workflow, hayate_specforge, upstream_drift
    )
    remote_reference_report = {
        "path": refs["remote_host_probe"]["path"],
        "status": "pass"
        if reference_evidence["remote_reference_proven"]
        else ("incomplete" if refs["remote_host_probe"]["status"] == "pass" else refs["remote_host_probe"]["status"]),
        "exists": refs["remote_host_probe"]["exists"],
        "component_statuses": {
            "remote_host_probe": refs["remote_host_probe"]["status"],
            "hayate_modelopt_remote_path_visible": str(
                reference_evidence["hayate_modelopt"]["remote_path_visible"]
            ).lower(),
            "hayate_specforge_remote_path_visible": str(
                reference_evidence["hayate_specforge"]["remote_path_visible"]
            ).lower(),
            "remote_workdir_visible": str(
                reference_evidence["remote_probe"]["remote_workdir_visible"]
            ).lower(),
            "remote_artifact_root_visible": str(
                reference_evidence["remote_probe"]["remote_artifact_root_visible"]
            ).lower(),
        },
    }
    modelopt_report = combined_report(
        "modelopt_loss_mask + modelopt_recipe_overrides",
        [refs["modelopt_loss_mask"], refs["modelopt_recipe_overrides"]],
        {"pass"},
    )
    target_corpus_report = combined_report(
        "rollout_capture_state + corpus_strategy",
        [refs["rollout_state"], refs["corpus_strategy"]],
        {"pass"},
    )
    runtime_report = combined_report(
        "container_preflight + vllm_source_build + vllm_abi_probe + megatron_compat",
        [refs["container_preflight"], refs["vllm_source_build"], refs["vllm_abi_probe"], refs["megatron_compat"]],
        {"pass"},
    )
    hidden_submit_report = combined_report(
        "pipeline_submit_preflight + pipeline_analysis",
        [refs["pipeline_submit_preflight"], refs["pipeline_analysis"]],
        {"pass"},
    )
    final_artifact_report = combined_report(
        "hidden_validation + training_checkpoint + export_artifacts + trained_draft_sweep",
        [refs["hidden_validation"], refs["training_checkpoint"], refs["export_artifacts"], refs["trained_draft_sweep"]],
        {"pass"},
    )
    stages = [
        gate(
            "reference_and_architecture",
            "Preserve Qwen3 architecture plus ModelOpt/Hayate references",
            reference_report,
            {"pass"},
            "Static Qwen3 verifier inputs and derived Eagle3 architecture are materialized and validated.",
        ),
        gate(
            "remote_hayate_reference_probe",
            "Prove live remote ModelOpt and Hayate/SpecForge reference paths",
            remote_reference_report,
            {"pass"},
            "Remote probe must show a reachable host with visible Hayate ModelOpt, SpecForge, remote workdir, and artifact root paths.",
            ["probe_remote_hosts"],
        ),
        gate(
            "modelopt_loss_and_recipe",
            "Use current ModelOpt with Qwen3 answer-only loss and Eagle3 recipe overrides",
            modelopt_report,
            {"pass"},
            "ModelOpt loss-mask patch and recipe override reports both PASS.",
        ),
        gate(
            "target_rollout_corpus",
            "Capture actual Qwen3 SWE/RL rollout conversations",
            target_corpus_report,
            {"pass"},
            "Rollout capture state and corpus strategy both PASS with pipeline-ready ModelOpt conversation JSONL.",
            ["submit_rollout_capture", "rollout_poll", "rollout_materialize", "rollout_materialize_and_refresh"],
        ),
        gate(
            "runtime_container",
            "Prove source vLLM, Megatron compatibility, and selected container",
            runtime_report,
            {"pass"},
            "Container preflight PASS, source vLLM build/ABI PASS, and Megatron compatibility PASS.",
            [
                "probe_remote_hosts",
                "submit_vllm_source_build",
                "poll_vllm_source_build",
                "submit_source_vllm_abi_probe",
                "submit_megatron_compat_probe",
                "poll_megatron_compat_probe",
                "submit_container_preflight",
            ],
        ),
        gate(
            "hidden_train_export_submit",
            "Submit hidden-state dump, ModelOpt training, and HF/vLLM export through the guarded pipeline",
            hidden_submit_report,
            {"pass"},
            "Pipeline submit preflight submit_ready=true, gated helper is emitted, and post-submit pipeline analysis PASS.",
            ["run_pipeline_submit_preflight", "submit_eagle3_pilot_pipeline"],
        ),
        gate(
            "trained_artifact_contracts",
            "Validate hidden states, ModelOpt checkpoint, exported HF/vLLM draft, and RL sweep",
            final_artifact_report,
            {"pass"},
            "Hidden-state validation, training checkpoint, export artifacts, and trained-draft sweep all PASS.",
            ["run_post_export_artifact_validations", "submit_trained_draft_spec_tokens_sweep"],
        ),
    ]

    path_inputs_ready = (
        args.playbook.exists()
        and args.reference_arch.exists()
        and args.modelopt_dir.exists()
        and refs["static_inputs"]["status"] == "pass"
        and refs["modelopt_recipe_overrides"]["status"] == "pass"
        and refs["modelopt_loss_mask"]["status"] == "pass"
        and refs["training_scale"]["exists"]
        and len((scale or {}).get("stage_plan") or []) >= 5
        and refs["hayate_modelopt_workflow"]["exists"]
        and refs["hayate_specforge_reference"]["exists"]
        and refs["draft_inventory"]["exists"]
        and refs["upstream_drift"]["exists"]
        and refs["next_action_plan"]["exists"]
    )
    gate_closure_contracts = build_gate_closure_contracts(
        stages,
        refs,
        plan,
        reference_evidence,
        rollout_state,
        corpus,
        pipeline_submit,
        pipeline,
        ready_by_id,
    )
    operator_gate_action_matrix = build_operator_gate_action_matrix(gate_closure_contracts, current_operator_actions)
    artifact_flow = build_artifact_flow(args, refs, rollout_state, corpus, set(action_ids))
    artifact_flow_complete = all(item.get("proof_status") == "pass" for item in artifact_flow)
    final_artifacts_pass = all(
        refs[name]["status"] == "pass"
        for name in ("pipeline_analysis", "hidden_validation", "training_checkpoint", "export_artifacts", "trained_draft_sweep")
    )
    final_pass = (
        final_artifacts_pass
        and artifact_flow_complete
        and all(item["status"] == "proven" for item in stages)
    )
    if final_pass:
        overall = "pass"
    elif path_inputs_ready:
        overall = "defined"
    else:
        overall = "incomplete"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "path_defined": path_inputs_ready,
        "artifact_root": str(args.artifact_root),
        "primary_route": "fixed_exported_eagle3_draft_first",
        "target_model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "modelopt_source_of_truth": {
            "local_modelopt_dir": str(args.modelopt_dir),
            "local_modelopt_exists": args.modelopt_dir.exists(),
            "hayate_role": "reference_only",
            "reason": "Hayate workflow and draft artifacts guide comparisons, but Qwen3-235B Thinking training uses current local/remote ModelOpt plus verifier-derived architecture.",
        },
        "reference_evidence": reference_evidence,
        "reference_decisions": reference_decisions,
        "playbook": {"path": str(args.playbook), "exists": args.playbook.exists()},
        "reference_arch": {"path": str(args.reference_arch), "exists": args.reference_arch.exists()},
        "reports": refs,
        "ready_actions": action_ids,
        "stage_plan": (scale or {}).get("stage_plan") or [],
        "gates": stages,
        "gate_closure_contracts": gate_closure_contracts,
        "operator_gate_action_matrix": operator_gate_action_matrix,
        "artifact_flow": artifact_flow,
        "artifact_flow_complete": artifact_flow_complete,
        "open_gates": [item["id"] for item in stages if item["status"] != "proven"],
        "final_artifacts_complete": final_artifacts_pass,
        "next_operator_actions": current_operator_actions,
    }


def render_markdown(data: dict[str, Any]) -> str:
    def csv(values: Any) -> str:
        return ", ".join(str(item) for item in values or []) or "-"

    def statuses(values: Any) -> str:
        if not isinstance(values, dict) or not values:
            return "-"
        return ", ".join(f"{key}={value}" for key, value in values.items()) or "-"

    lines = [
        "# Eagle3 Training Path Manifest",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Path defined: **{str(data['path_defined']).lower()}**",
        f"Primary route: `{data['primary_route']}`",
        f"Target model: `{data['target_model']}`",
        "",
        "## Gate Order",
        "",
        "| gate | status | report status | proof required |",
        "| --- | --- | --- | --- |",
    ]
    for item in data["gates"]:
        lines.append(
            f"| {item['id']} | {item['status']} | {item['report_status']} | "
            f"{str(item['proof_required']).replace('|', '/')} |"
        )
    lines += ["", "## Ready Operator Actions", ""]
    if data.get("next_operator_actions"):
        for item in data["next_operator_actions"]:
            lines.append(
                f"- {item.get('order')}. `{item['id']}` ({item.get('stage') or '-'}) submits_slurm={str(item.get('submits_slurm')).lower()} heavy_gpu={str(item.get('heavy_gpu')).lower()}"
            )
    else:
        lines.append("- none")
    lines += [
        "",
        "## Operator Gate Action Matrix",
        "",
        "| gate | status | current ready actions | future candidate actions | missing evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in data.get("operator_gate_action_matrix") or []:
        ready = ", ".join(item.get("current_ready_action_ids") or []) or "-"
        future = ", ".join(item.get("future_candidate_action_ids") or []) or "-"
        missing = ", ".join(item.get("missing_evidence") or []) or "-"
        lines.append(
            f"| {item.get('gate_id')} | {item.get('status')} | {ready.replace('|', '/')} | "
            f"{future.replace('|', '/')} | {missing.replace('|', '/')} |"
        )
    lines += [
        "",
        "## Artifact Flow",
        "",
        f"Complete: **{str(data.get('artifact_flow_complete')).lower()}**",
        "",
        "| artifact | proof | current actions | future actions | reports | invariants | visible | path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in data.get("artifact_flow") or []:
        current_actions = csv(item.get("current_closure_action_ids")).replace("|", "/")
        future_actions = csv(item.get("future_closure_action_ids")).replace("|", "/")
        reports = statuses(item.get("report_statuses")).replace("|", "/")
        invariants = csv(item.get("required_invariants")).replace("|", "/")
        lines.append(
            f"| {item.get('id')} | {item.get('proof_status')} | {current_actions} | {future_actions} | {reports} | "
            f"{invariants} | {str(item.get('path_visible')).lower()} | `{item.get('path')}` |"
        )
    lines += [
        "",
        "## Gate Closure Contracts",
        "",
        "| gate | closed | missing evidence | candidate actions |",
        "| --- | --- | --- | --- |",
    ]
    for item in data.get("gate_closure_contracts") or []:
        missing = ", ".join(item.get("closure_evidence_missing") or []) or "-"
        actions = ", ".join(item.get("candidate_next_action_ids") or []) or "-"
        lines.append(
            f"| {item.get('id')} | {str(item.get('closed')).lower()} | "
            f"{missing.replace('|', '/')} | {actions.replace('|', '/')} |"
        )
    evidence = data.get("reference_evidence") or {}
    remote = evidence.get("remote_probe") if isinstance(evidence.get("remote_probe"), dict) else {}
    hayate_modelopt = evidence.get("hayate_modelopt") if isinstance(evidence.get("hayate_modelopt"), dict) else {}
    hayate_specforge = evidence.get("hayate_specforge") if isinstance(evidence.get("hayate_specforge"), dict) else {}
    lines += [
        "",
        "## Reference Evidence",
        "",
        f"- local ModelOpt: `{(evidence.get('local_modelopt') or {}).get('path')}` exists={str((evidence.get('local_modelopt') or {}).get('exists')).lower()}",
        f"- remote probe: status=`{remote.get('status')}` reachable_hosts=`{len(remote.get('reachable_hosts') or [])}` remote_reference_proven={str(evidence.get('remote_reference_proven')).lower()}",
        f"- Hayate ModelOpt: source=`{hayate_modelopt.get('source')}` live_visible={str(hayate_modelopt.get('live_visible')).lower()} remote_path_visible={str(hayate_modelopt.get('remote_path_visible')).lower()}",
        f"- Hayate SpecForge: source=`{hayate_specforge.get('source')}` live_visible={str(hayate_specforge.get('live_visible')).lower()} remote_path_visible={str(hayate_specforge.get('remote_path_visible')).lower()}",
    ]
    decisions = data.get("reference_decisions") if isinstance(data.get("reference_decisions"), dict) else {}
    modelopt = decisions.get("modelopt_source") if isinstance(decisions.get("modelopt_source"), dict) else {}
    specforge = decisions.get("specforge_qwen3_235b") if isinstance(decisions.get("specforge_qwen3_235b"), dict) else {}
    rejected = specforge.get("rejected_fields") if isinstance(specforge.get("rejected_fields"), list) else []
    lines += [
        "",
        "## Reference Decisions",
        "",
        f"- ModelOpt source of truth: `{modelopt.get('source_of_truth')}`; upstream drift status=`{modelopt.get('upstream_drift_status')}`",
        f"- SpecForge matched fields: `{', '.join(specforge.get('matched_fields') or [])}`",
        f"- SpecForge rejected fields: `{', '.join(str(item.get('field')) for item in rejected if isinstance(item, dict))}`",
    ]
    lines += ["", "## Reference Reports", "", "| report | status | path |", "| --- | --- | --- |"]
    for name, report in data["reports"].items():
        lines.append(f"| {name} | {report.get('status')} | `{report.get('path')}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = with_defaults(parse_args())
    data = build_manifest(args)
    markdown = render_markdown(data)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
