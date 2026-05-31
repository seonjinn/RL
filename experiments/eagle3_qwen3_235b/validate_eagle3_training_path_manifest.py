#!/usr/bin/env python3
"""Validate the Qwen3-235B Eagle3 training-path manifest contract.

This is a no-submit synthetic test for build_eagle3_training_path_manifest.py.
It proves the manifest can distinguish a locally defined path from an
incomplete path, and that it only reports PASS when final trained-artifact
reports are present.
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
SCRIPT = ROOT / "experiments/eagle3_qwen3_235b/build_eagle3_training_path_manifest.py"
EXPECTED_GATES = [
    "reference_and_architecture",
    "remote_hayate_reference_probe",
    "modelopt_loss_and_recipe",
    "target_rollout_corpus",
    "runtime_container",
    "hidden_train_export_submit",
    "trained_artifact_contracts",
]
EXPECTED_ARTIFACT_FLOW = [
    "rollout_conversation_corpus",
    "verifier_hidden_states",
    "modelopt_checkpoint",
    "hf_eagle3_export",
    "vllm_eagle3_draft",
    "rl_vllm_draft_validation",
]
EXPECTED_ARTIFACT_CLOSURE_ACTIONS = {
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
    "rl_vllm_draft_validation": ["submit_trained_draft_spec_tokens_sweep"],
}
HAYATE_MODELOPT = "/remote/hayate/TensorRT-Model-Optimizer"
HAYATE_SPECFORGE = "/remote/hayate/SpecForge"
REMOTE_WORKDIR = "/remote/sna/Nemo-RL_Qwen3_Roadmap"


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


def common_reports(artifact: Path, *, final_pass: bool = False) -> None:
    reports = artifact / "reports"
    write_json(reports / "qwen3_static_inputs_materialization_validation.json", {"overall_status": "pass"})
    write_json(
        reports / "hayate_modelopt_workflow.json",
        {
            "overall_status": "reference_only",
            "source": "synthetic_live" if final_pass else "synthetic_snapshot",
            "live_hayate_visible": final_pass,
            "selected_path": HAYATE_MODELOPT,
            "classification": "synthetic_hayate_reference_only",
            "classification_basis": {
                "workflow_files_present": [
                    "examples/speculative_decoding/prepare_input_conversations/generate_responses.py",
                    "examples/speculative_decoding/slurm/train_eagle3.sbatch",
                ],
                "qwen_config_paths_present": [
                    "examples/speculative_decoding/eagle_config_qwen3_30b_moe.json",
                ],
            },
        },
    )
    write_json(
        reports / "hayate_specforge_reference.json",
        {
            "overall_status": "reference_only",
            "source": "synthetic_live" if final_pass else "synthetic_snapshot",
            "live_specforge_visible": final_pass,
            "specforge_dir": HAYATE_SPECFORGE,
            "qwen3_235b_comparison": {
                "status": "reference_only",
                "rows": [
                    {"field": "aux_layers", "specforge": [1, 46, 90], "current": [1, 46, 90], "match": True},
                    {"field": "hidden_size", "specforge": 4096, "current": 4096, "match": True},
                    {"field": "rope_theta", "specforge": 1000000, "current": 5000000, "match": False},
                ],
            },
        },
    )
    remote_probe = {
        "overall_status": "unreachable",
        "reachable_hosts": [],
        "remote_workdir": REMOTE_WORKDIR,
        "artifact_root": str(artifact),
        "hosts": [{"host": "synthetic", "reachable": False, "paths": []}],
    }
    if final_pass:
        remote_probe = {
            "overall_status": "pass",
            "reachable_hosts": ["synthetic"],
            "remote_workdir": REMOTE_WORKDIR,
            "artifact_root": str(artifact),
            "hosts": [
                {
                    "host": "synthetic",
                    "reachable": True,
                    "paths": [
                        {"path": HAYATE_MODELOPT, "exists": True},
                        {"path": HAYATE_SPECFORGE, "exists": True},
                        {"path": REMOTE_WORKDIR, "exists": True},
                        {"path": str(artifact), "exists": True},
                    ],
                }
            ],
        }
    write_json(reports / "eagle3_remote_host_probe.json", remote_probe)
    write_json(
        reports / "eagle3_remote_access_diagnostics.json",
        {
            "overall_status": "pass" if final_pass else "blocked_local_dns",
            "diagnosis": "synthetic reachable remote" if final_pass else "synthetic local DNS block",
            "counts": {
                "hosts": 1,
                "resolved_hosts": 1 if final_pass else 0,
                "unresolved_hosts": 0 if final_pass else 1,
                "reachable_hosts": 1 if final_pass else 0,
            },
            "gate_interpretation": {
                "remote_hayate_reference_probe": "closed" if final_pass else "open",
                "remote_path_absence_proven": False,
            },
        },
    )
    write_json(reports / "eagle3_draft_config_inventory.json", {"overall_status": "warn", "warnings": ["synthetic"]})
    write_json(
        reports / "modelopt_upstream_drift.json",
        {
            "overall_status": "warn",
            "decision": {
                "overall_status": "warn",
                "source_of_truth": "local_modelopt",
                "local_head": "synthetic-local",
                "upstream_head": "synthetic-upstream",
                "hayate_head": "synthetic-hayate" if final_pass else None,
                "allowed_focus_diffs": ["examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py"],
                "disallowed_focus_diffs": [],
                "recommendation": "Use the local/remote ModelOpt checkout as the training source.",
            },
        },
    )
    write_json(reports / "modelopt_loss_mask_patch.json", {"overall_status": "pass"})
    write_json(reports / "modelopt_recipe_overrides_current.json", {"overall_status": "pass"})
    write_json(
        reports / "eagle3_training_scale.json",
        {
            "overall_status": "incomplete",
            "stage_plan": [
                {"name": "smoke"},
                {"name": "pilot"},
                {"name": "swegym_first_calibration"},
                {"name": "target_domain_calibration"},
                {"name": "production_candidate"},
            ],
        },
    )
    write_json(
        reports / "eagle3_next_actions.json",
        {
            "overall_status": "ready_for_operator_submit",
            "next_actions": [
                {
                    "id": "submit_vllm_source_build",
                    "title": "Submit source vLLM build",
                    "status": "ready_for_operator",
                    "stage": "runtime_gate",
                    "command": "echo source build",
                    "submits_slurm": True,
                    "heavy_gpu": False,
                }
            ],
        },
    )
    if final_pass:
        pass_reports = {
            "corpus_strategy.json": {"overall_status": "pass", "decision": {"primary_source": "actual_rl_rollout"}},
            "rollout_capture_state_advance.json": {"overall_status": "pass", "decision": {"next_step": "pipeline_dry_run"}},
            "container_preflight_analysis.json": {"overall_status": "pass"},
            "vllm_native_source_build.json": {"overall_status": "pass"},
            "vllm_native_abi_probe.json": {"overall_status": "pass"},
            "megatron_compat_probe.json": {"overall_status": "pass"},
            "eagle3_pipeline_submit_preflight.json": {
                "overall_status": "pass",
                "submit_ready": True,
                "commands": {"gated_pilot_submit": "echo submit"},
            },
            "eagle3_pipeline_analysis.json": {"overall_status": "pass"},
            "eagle3_training_checkpoint.json": {"overall_status": "pass"},
            "eagle3_export_artifacts.json": {"overall_status": "pass"},
            "trained_draft_spec_tokens_sweep.json": {"overall_status": "pass", "rows": [{"gate_status": "pass"}]},
        }
        for name, payload in pass_reports.items():
            write_json(reports / name, payload)
        write_json(artifact / "hidden_states/validation_summary.json", {"overall_status": "pass"})
    else:
        write_json(reports / "corpus_strategy.json", {"overall_status": "missing_capture"})
        write_json(reports / "rollout_capture_submit_preflight.json", {"overall_status": "fail"})
        write_json(reports / "container_preflight_analysis.json", {"overall_status": "incomplete"})
        write_json(reports / "eagle3_pipeline_submit_preflight.json", {"overall_status": "incomplete"})
        write_json(reports / "eagle3_pipeline_analysis.json", {"overall_status": "incomplete"})


def run_manifest(artifact: Path, out: Path) -> tuple[int, dict[str, Any], str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--artifact-root",
        str(artifact),
        "--modelopt-dir",
        str(ROOT / "Model-Optimizer"),
        "--upstream-drift-json",
        str(artifact / "reports/modelopt_upstream_drift.json"),
        "--remote-access-diagnostics-json",
        str(artifact / "reports/eagle3_remote_access_diagnostics.json"),
        "--json-out",
        str(out),
        "--markdown-out",
        str(out.with_suffix(".md")),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    payload = read_json(out) if out.exists() else {}
    return result.returncode, payload, result.stdout


def check_defined(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("overall_status") != "defined":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'defined'")
    if payload.get("path_defined") is not True:
        problems.append("path_defined is not true")
    gates = [item.get("id") for item in payload.get("gates") or [] if isinstance(item, dict)]
    if gates != EXPECTED_GATES:
        problems.append(f"gate order {gates!r} != {EXPECTED_GATES!r}")
    if "target_rollout_corpus" not in (payload.get("open_gates") or []):
        problems.append("target_rollout_corpus should remain open before rollout capture")
    if "remote_hayate_reference_probe" not in (payload.get("open_gates") or []):
        problems.append("remote_hayate_reference_probe should remain open before live remote reference evidence")
    reference_evidence = payload.get("reference_evidence") if isinstance(payload.get("reference_evidence"), dict) else {}
    for key in ["local_modelopt", "remote_probe", "hayate_modelopt", "hayate_specforge"]:
        if not isinstance(reference_evidence.get(key), dict):
            problems.append(f"reference_evidence.{key} is missing")
    if reference_evidence.get("remote_reference_proven") is not False:
        problems.append("remote_reference_proven should be false before live remote reference evidence")
    if "submit_vllm_source_build" not in (payload.get("ready_actions") or []):
        problems.append("ready source-build action was not preserved")
    if payload.get("final_artifacts_complete") is not False:
        problems.append("final_artifacts_complete should be false before trained artifacts")
    if payload.get("artifact_flow_complete") is not False:
        problems.append("artifact_flow_complete should be false before trained artifacts")
    problems.extend(check_reference_decisions(payload))
    problems.extend(check_gate_closure_contracts(payload, final_pass=False))
    problems.extend(check_operator_gate_action_matrix(payload, final_pass=False))
    problems.extend(check_artifact_flow(payload, final_pass=False))
    return problems


def check_incomplete(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("overall_status") != "incomplete":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'incomplete'")
    if payload.get("path_defined") is not False:
        problems.append("path_defined should be false when static input evidence is missing")
    return problems


def check_final_pass(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("overall_status") != "pass":
        problems.append(f"overall_status {payload.get('overall_status')!r} != 'pass'")
    if payload.get("path_defined") is not True:
        problems.append("path_defined is not true for final pass")
    if payload.get("final_artifacts_complete") is not True:
        problems.append("final_artifacts_complete should be true when all final reports PASS")
    if payload.get("artifact_flow_complete") is not True:
        problems.append("artifact_flow_complete should be true when all artifact-flow reports PASS")
    if payload.get("open_gates") != []:
        problems.append(f"open_gates {payload.get('open_gates')!r} != []")
    reference_evidence = payload.get("reference_evidence") if isinstance(payload.get("reference_evidence"), dict) else {}
    for key in ["local_modelopt", "remote_probe", "hayate_modelopt", "hayate_specforge"]:
        if not isinstance(reference_evidence.get(key), dict):
            problems.append(f"reference_evidence.{key} is missing")
    if reference_evidence.get("remote_reference_proven") is not True:
        problems.append("remote_reference_proven should be true for final pass")
    problems.extend(check_reference_decisions(payload))
    problems.extend(check_gate_closure_contracts(payload, final_pass=True))
    problems.extend(check_operator_gate_action_matrix(payload, final_pass=True))
    problems.extend(check_artifact_flow(payload, final_pass=True))
    return problems


def check_reference_decisions(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    decisions = payload.get("reference_decisions") if isinstance(payload.get("reference_decisions"), dict) else {}
    if not decisions:
        return ["reference_decisions is missing"]
    route = decisions.get("training_route") if isinstance(decisions.get("training_route"), dict) else {}
    if route.get("primary_route") != "fixed_exported_eagle3_draft_first":
        problems.append("reference_decisions.training_route did not preserve fixed exported draft first")
    modelopt = decisions.get("modelopt_source") if isinstance(decisions.get("modelopt_source"), dict) else {}
    if modelopt.get("source_of_truth") != "local_modelopt":
        problems.append("reference_decisions.modelopt_source did not keep local_modelopt as source_of_truth")
    if modelopt.get("upstream_drift_status") not in {"pass", "warn"}:
        problems.append("reference_decisions.modelopt_source did not include pass/warn upstream drift status")
    specforge = decisions.get("specforge_qwen3_235b") if isinstance(decisions.get("specforge_qwen3_235b"), dict) else {}
    matched = set(specforge.get("matched_fields") or [])
    if not {"aux_layers", "hidden_size"}.issubset(matched):
        problems.append("reference_decisions.specforge_qwen3_235b did not preserve matched aux/hidden fields")
    rejected = {
        str(item.get("field"))
        for item in specforge.get("rejected_fields") or []
        if isinstance(item, dict) and item.get("field")
    }
    if "rope_theta" not in rejected:
        problems.append("reference_decisions.specforge_qwen3_235b did not reject mismatched rope_theta")
    workflow = decisions.get("hayate_workflow") if isinstance(decisions.get("hayate_workflow"), dict) else {}
    if workflow.get("role") != "reference_only":
        problems.append("reference_decisions.hayate_workflow did not record reference_only role")
    remote = decisions.get("remote_probe") if isinstance(decisions.get("remote_probe"), dict) else {}
    if remote.get("diagnostics_status") not in {"pass", "blocked_local_dns"}:
        problems.append("reference_decisions.remote_probe did not include remote diagnostics status")
    return problems


def contract_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contracts = payload.get("gate_closure_contracts")
    if not isinstance(contracts, list):
        return {}
    return {
        str(item.get("id")): item
        for item in contracts
        if isinstance(item, dict) and item.get("id")
    }


def report_labels(contract: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for item in contract.get("required_reports") or []:
        if isinstance(item, dict) and item.get("label"):
            labels.add(str(item["label"]))
    return labels


def matrix_by_gate(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("operator_gate_action_matrix")
    if not isinstance(rows, list):
        return {}
    return {
        str(item.get("gate_id")): item
        for item in rows
        if isinstance(item, dict) and item.get("gate_id")
    }


def check_operator_gate_action_matrix(payload: dict[str, Any], *, final_pass: bool) -> list[str]:
    problems: list[str] = []
    matrix = matrix_by_gate(payload)
    if list(matrix.keys()) != EXPECTED_GATES:
        problems.append(f"operator_gate_action_matrix ids {list(matrix.keys())!r} != {EXPECTED_GATES!r}")
        return problems

    ready_ids = [str(item) for item in payload.get("ready_actions") or []]
    operator_ids = [
        str(item.get("id"))
        for item in payload.get("next_operator_actions") or []
        if isinstance(item, dict) and item.get("id")
    ]
    if operator_ids != ready_ids:
        problems.append(f"next_operator_actions order {operator_ids!r} != ready_actions {ready_ids!r}")

    ready_set = set(ready_ids)
    for gate_id, row in matrix.items():
        if not isinstance(row.get("missing_evidence"), list):
            problems.append(f"{gate_id} matrix row is missing missing_evidence list")
        current_ids = [str(item) for item in row.get("current_ready_action_ids") or []]
        if any(action_id not in ready_set for action_id in current_ids):
            problems.append(f"{gate_id} matrix row references non-ready actions: {current_ids!r}")
        if row.get("gate_closed") is True and row.get("status") != "closed":
            problems.append(f"{gate_id} closed matrix row did not report status=closed")
        if row.get("gate_closed") is not True and current_ids and row.get("status") != "ready_action_available":
            problems.append(f"{gate_id} matrix row has current ready actions but status {row.get('status')!r}")

    runtime = matrix.get("runtime_container", {})
    runtime_ready = set(runtime.get("current_ready_action_ids") or [])
    if not final_pass and "submit_vllm_source_build" not in runtime_ready:
        problems.append("runtime_container matrix did not map submit_vllm_source_build as a current ready action")
    if final_pass:
        not_closed = [gate_id for gate_id, row in matrix.items() if row.get("status") != "closed"]
        if not_closed:
            problems.append(f"final pass should close every operator gate-action row, still open: {not_closed}")
    return problems


def flow_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("artifact_flow")
    if not isinstance(rows, list):
        return {}
    return {
        str(item.get("id")): item
        for item in rows
        if isinstance(item, dict) and item.get("id")
    }


def check_artifact_flow(payload: dict[str, Any], *, final_pass: bool) -> list[str]:
    problems: list[str] = []
    flow = flow_by_id(payload)
    ready_set = {str(item) for item in payload.get("ready_actions") or []}
    if list(flow.keys()) != EXPECTED_ARTIFACT_FLOW:
        problems.append(f"artifact_flow ids {list(flow.keys())!r} != {EXPECTED_ARTIFACT_FLOW!r}")
        return problems

    for artifact_id, item in flow.items():
        for field in (
            "artifact_type",
            "path",
            "producer_gate",
            "consumer_gate",
            "required_reports",
            "required_invariants",
            "closure_action_ids",
            "current_closure_action_ids",
            "future_closure_action_ids",
            "report_statuses",
            "path_visible",
            "proof_status",
        ):
            if field not in item:
                problems.append(f"{artifact_id} artifact_flow row missing {field}")
        if not isinstance(item.get("required_reports"), list) or not item.get("required_reports"):
            problems.append(f"{artifact_id} artifact_flow row has no required_reports")
        if not isinstance(item.get("required_invariants"), list) or not item.get("required_invariants"):
            problems.append(f"{artifact_id} artifact_flow row has no required_invariants")
        if item.get("closure_action_ids") != EXPECTED_ARTIFACT_CLOSURE_ACTIONS.get(artifact_id):
            problems.append(
                f"{artifact_id} closure_action_ids {item.get('closure_action_ids')!r} "
                f"!= {EXPECTED_ARTIFACT_CLOSURE_ACTIONS.get(artifact_id)!r}"
            )
        current = item.get("current_closure_action_ids")
        future = item.get("future_closure_action_ids")
        if not isinstance(current, list) or not isinstance(future, list):
            problems.append(f"{artifact_id} artifact_flow row has invalid current/future closure action lists")
        elif set(current) | set(future) != set(item.get("closure_action_ids") or []):
            problems.append(f"{artifact_id} current/future closure action split does not preserve closure_action_ids")
        elif set(current) & set(future):
            problems.append(f"{artifact_id} current/future closure action split overlaps")
        elif any(str(action_id) not in ready_set for action_id in current):
            problems.append(f"{artifact_id} current_closure_action_ids includes non-ready action")
        elif any(str(action_id) in ready_set for action_id in future):
            problems.append(f"{artifact_id} future_closure_action_ids includes a currently ready action")
        if not isinstance(item.get("report_statuses"), dict) or not item.get("report_statuses"):
            problems.append(f"{artifact_id} artifact_flow row has no report_statuses")
        if not isinstance(item.get("path_visible"), bool):
            problems.append(f"{artifact_id} artifact_flow path_visible is not boolean")
        if item.get("producer_gate") not in EXPECTED_GATES and item.get("producer_gate") != "trained_artifact_contracts":
            problems.append(f"{artifact_id} producer_gate {item.get('producer_gate')!r} is not a known training gate")

    corpus = flow.get("rollout_conversation_corpus", {})
    if corpus.get("producer_gate") != "target_rollout_corpus" or corpus.get("consumer_gate") != "hidden_train_export_submit":
        problems.append("rollout_conversation_corpus does not connect target_rollout_corpus to hidden_train_export_submit")
    if "qwen3_235b_swe_rollout_conversations.jsonl" not in str(corpus.get("path") or ""):
        problems.append("rollout_conversation_corpus does not use the canonical Qwen3 SWE/RL conversation JSONL path")

    hidden = flow.get("verifier_hidden_states", {})
    if "answer_only_loss_mask_preserved" not in (hidden.get("required_invariants") or []):
        problems.append("verifier_hidden_states does not require answer_only_loss_mask_preserved")

    if final_pass:
        open_rows = [artifact_id for artifact_id, item in flow.items() if item.get("proof_status") != "pass"]
        if open_rows:
            problems.append(f"final pass should close all artifact_flow rows, still open: {open_rows}")
    else:
        for artifact_id in ("rollout_conversation_corpus", "verifier_hidden_states", "modelopt_checkpoint"):
            if flow.get(artifact_id, {}).get("proof_status") == "pass":
                problems.append(f"{artifact_id} should remain open before external execution evidence")
    return problems


def check_gate_closure_contracts(payload: dict[str, Any], *, final_pass: bool) -> list[str]:
    problems: list[str] = []
    contracts = contract_by_id(payload)
    if list(contracts.keys()) != EXPECTED_GATES:
        problems.append(f"gate_closure_contracts ids {list(contracts.keys())!r} != {EXPECTED_GATES!r}")
        return problems
    for gate_id, contract in contracts.items():
        if not isinstance(contract.get("required_reports"), list) or not contract["required_reports"]:
            problems.append(f"{gate_id} contract has no required_reports")
        if not isinstance(contract.get("closure_evidence_missing"), list):
            problems.append(f"{gate_id} contract has no closure_evidence_missing list")
        if not isinstance(contract.get("candidate_next_action_ids"), list):
            problems.append(f"{gate_id} contract has no candidate_next_action_ids list")
        if not isinstance(contract.get("operator_actions"), list):
            problems.append(f"{gate_id} contract has no operator_actions list")

    target_labels = report_labels(contracts.get("target_rollout_corpus", {}))
    if not {"rollout_state", "corpus_strategy"}.issubset(target_labels):
        problems.append("target_rollout_corpus contract does not require rollout_state and corpus_strategy")
    runtime_labels = report_labels(contracts.get("runtime_container", {}))
    if not {"container_preflight", "vllm_source_build", "vllm_abi_probe", "megatron_compat"}.issubset(runtime_labels):
        problems.append("runtime_container contract does not require container/source-vLLM/ABI/Megatron reports")
    hidden_labels = report_labels(contracts.get("hidden_train_export_submit", {}))
    if not {"pipeline_submit_preflight", "pipeline_analysis"}.issubset(hidden_labels):
        problems.append("hidden_train_export_submit contract does not require preflight and pipeline analysis")

    if final_pass:
        open_contracts = [gate_id for gate_id, contract in contracts.items() if contract.get("closed") is not True]
        if open_contracts:
            problems.append(f"final pass should close all gate contracts, still open: {open_contracts}")
    else:
        for gate_id in ("target_rollout_corpus", "runtime_container", "hidden_train_export_submit"):
            if contracts.get(gate_id, {}).get("closed") is True:
                problems.append(f"{gate_id} should remain open before external execution gates")
            if not contracts.get(gate_id, {}).get("closure_evidence_missing"):
                problems.append(f"{gate_id} should list missing closure evidence before external execution gates")
    return problems


def run_scenarios(temp_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    scenarios: list[dict[str, Any]] = []
    problems: list[str] = []

    defined_artifact = temp_root / "defined/qwen3_235b_eagle3"
    common_reports(defined_artifact, final_pass=False)
    rc, payload, output = run_manifest(defined_artifact, defined_artifact / "reports/eagle3_training_path_manifest.json")
    item_problems = ([] if rc == 0 else [f"returncode {rc} != 0"]) + check_defined(payload)
    scenarios.append({"name": "defined_before_external_gates", "status": "pass" if not item_problems else "fail", "problems": item_problems, "payload": payload, "output_tail": output[-1000:]})
    problems.extend(f"defined_before_external_gates: {problem}" for problem in item_problems)

    incomplete_artifact = temp_root / "incomplete/qwen3_235b_eagle3"
    common_reports(incomplete_artifact, final_pass=False)
    (incomplete_artifact / "reports/qwen3_static_inputs_materialization_validation.json").unlink()
    rc, payload, output = run_manifest(incomplete_artifact, incomplete_artifact / "reports/eagle3_training_path_manifest.json")
    item_problems = ([] if rc == 0 else [f"returncode {rc} != 0"]) + check_incomplete(payload)
    scenarios.append({"name": "incomplete_without_static_inputs", "status": "pass" if not item_problems else "fail", "problems": item_problems, "payload": payload, "output_tail": output[-1000:]})
    problems.extend(f"incomplete_without_static_inputs: {problem}" for problem in item_problems)

    final_artifact = temp_root / "final/qwen3_235b_eagle3"
    common_reports(final_artifact, final_pass=True)
    rc, payload, output = run_manifest(final_artifact, final_artifact / "reports/eagle3_training_path_manifest.json")
    item_problems = ([] if rc == 0 else [f"returncode {rc} != 0"]) + check_final_pass(payload)
    scenarios.append({"name": "final_artifacts_pass", "status": "pass" if not item_problems else "fail", "problems": item_problems, "payload": payload, "output_tail": output[-1000:]})
    problems.extend(f"final_artifacts_pass: {problem}" for problem in item_problems)

    return scenarios, problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Training Path Manifest Validation",
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
    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_training_path_manifest_"))
    try:
        scenarios, problems = run_scenarios(temp_root)
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "pass" if not problems else "fail",
        "script": str(SCRIPT),
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
