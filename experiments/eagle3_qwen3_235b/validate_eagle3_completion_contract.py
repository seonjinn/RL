#!/usr/bin/env python3
"""Validate the final Eagle3 completion-audit contract with synthetic artifacts.

This is a no-submit test. It creates a tiny filesystem that satisfies the same
report/file contracts expected after a real Qwen3-235B Eagle3 run, then checks
that audit_eagle3_completion.py passes. It also verifies that a sweep report
pointing at the wrong vLLM draft directory is rejected.
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
EXP = ROOT / "experiments/eagle3_qwen3_235b"
REFERENCE_ARCH = EXP / "qwen3_235b_thinking_eagle3_architecture.json"
EXPORT_VALIDATOR = EXP / "validate_eagle3_export_artifacts.py"
COMPLETION_AUDIT = EXP / "audit_eagle3_completion.py"


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


def write_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + b"\x00\x00\x00\x00")


def verifier_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "vocab_size": 151936,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "num_hidden_layers": 94,
        "intermediate_size": 12288,
        "head_dim": 128,
        "rms_norm_eps": 0.000001,
        "rope_theta": 5000000,
        "rope_scaling": {"rope_type": "default"},
    }


def hf_draft_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "vocab_size": 151936,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 12288,
        "head_dim": 128,
        "rms_norm_eps": 0.000001,
        "rope_theta": 5000000,
        "rope_scaling": {"rope_type": "default"},
        "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
    }


def vllm_draft_config(verifier_dir: Path) -> dict[str, Any]:
    layer = hf_draft_config()
    return {
        "speculators_model_type": "eagle3",
        "architectures": ["Eagle3Speculator"],
        "target_hidden_size": 4096,
        "speculators_config": {"verifier": {"name_or_path": str(verifier_dir)}},
        "transformer_layer_config": layer,
    }


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def materialize_artifacts(root: Path) -> dict[str, Path]:
    reports = root / "reports"
    hidden = root / "hidden_states"
    output = root / "modelopt_ckpt"
    export = root / "exported_hf"
    vllm = root / "vllm_draft"
    verifier = root / "verifier_config"
    data = root / "data"
    for path in (reports, hidden, output, export, vllm, verifier, data):
        path.mkdir(parents=True, exist_ok=True)

    write_json(verifier / "config.json", verifier_config())
    write_json(output / "config.json", hf_draft_config())
    write_json(export / "config.json", hf_draft_config())
    write_json(vllm / "config.json", vllm_draft_config(verifier))
    write_safetensors(output / "model.safetensors")
    write_safetensors(export / "model.safetensors")
    write_safetensors(vllm / "model.safetensors")
    (data / "qwen3_235b_swe_rollout_conversations.jsonl").write_text(
        json.dumps({"conversation_id": "synthetic", "messages": [{"role": "assistant", "content": "ok"}]}) + "\n",
        encoding="utf-8",
    )

    return {
        "reports": reports,
        "hidden": hidden,
        "output": output,
        "export": export,
        "vllm": vllm,
        "verifier": verifier,
        "data": data,
    }


def training_path_manifest(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    gates = [
        "reference_and_architecture",
        "remote_hayate_reference_probe",
        "modelopt_loss_and_recipe",
        "target_rollout_corpus",
        "runtime_container",
        "hidden_train_export_submit",
        "trained_artifact_contracts",
    ]

    def closure_contract(gate_id: str, labels: list[str]) -> dict[str, Any]:
        return {
            "id": gate_id,
            "title": gate_id,
            "current_gate_status": "proven",
            "current_report_status": "pass",
            "closed": True,
            "required_reports": [
                {
                    "label": label,
                    "path": str(paths["reports"] / f"{label}.json"),
                    "exists": True,
                    "current_status": "pass",
                    "accepted_statuses": ["pass"],
                    "status": "pass",
                    "proof": "synthetic completion fixture",
                }
                for label in labels
            ],
            "required_conditions": [],
            "closure_evidence_missing": [],
            "candidate_next_action_ids": [],
            "operator_actions": [],
            "do_not_proceed_guards": [],
        }

    artifact_flow = [
        {
            "id": "rollout_conversation_corpus",
            "artifact_type": "jsonl_conversations",
            "path": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
            "producer_gate": "target_rollout_corpus",
            "consumer_gate": "hidden_train_export_submit",
            "required_reports": ["rollout_state", "corpus_strategy"],
            "required_invariants": ["primary_source=actual_rl_rollout", "rollout_state_next_step=pipeline_dry_run"],
            "report_statuses": {"rollout_state": "pass", "corpus_strategy": "pass"},
            "proof_status": "pass",
        },
        {
            "id": "verifier_hidden_states",
            "artifact_type": "modelopt_hidden_state_tensors",
            "path": str(paths["hidden"]),
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "hidden_validation"],
            "required_invariants": ["answer_only_loss_mask_preserved", "positive_loss_mask_files>0"],
            "report_statuses": {"pipeline_analysis": "pass", "hidden_validation": "pass"},
            "proof_status": "pass",
        },
        {
            "id": "modelopt_checkpoint",
            "artifact_type": "modelopt_eagle3_checkpoint",
            "path": str(paths["output"]),
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "training_checkpoint"],
            "required_invariants": ["offline_hidden_state_training", "qwen3_eagle3_recipe_overrides"],
            "report_statuses": {"pipeline_analysis": "pass", "training_checkpoint": "pass"},
            "proof_status": "pass",
        },
        {
            "id": "hf_eagle3_export",
            "artifact_type": "hf_draft_export",
            "path": str(paths["export"]),
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "export_artifacts"],
            "required_invariants": ["verifier_config_compatible", "thinking_2507_architecture_preserved"],
            "report_statuses": {"pipeline_analysis": "pass", "export_artifacts": "pass"},
            "proof_status": "pass",
        },
        {
            "id": "vllm_eagle3_draft",
            "artifact_type": "vllm_draft_export",
            "path": str(paths["vllm"]),
            "producer_gate": "hidden_train_export_submit",
            "consumer_gate": "trained_artifact_contracts",
            "required_reports": ["pipeline_analysis", "export_artifacts"],
            "required_invariants": ["vllm_config_exists", "draft_weights_present"],
            "report_statuses": {"pipeline_analysis": "pass", "export_artifacts": "pass"},
            "proof_status": "pass",
        },
        {
            "id": "rl_vllm_draft_validation",
            "artifact_type": "trained_draft_spec_tokens_sweep",
            "path": str(paths["reports"] / "trained_draft_spec_tokens_sweep.json"),
            "producer_gate": "trained_artifact_contracts",
            "consumer_gate": "completion_audit",
            "required_reports": ["trained_draft_sweep"],
            "required_invariants": ["acceptance_gate_pass", "speed_gate_pass", "no_reward_or_malformed_regression"],
            "report_statuses": {"trained_draft_sweep": "pass"},
            "proof_status": "pass",
        },
    ]

    return {
        "overall_status": "pass",
        "path_defined": True,
        "artifact_root": str(root),
        "primary_route": "fixed_exported_eagle3_draft_first",
        "final_artifacts_complete": True,
        "artifact_flow_complete": True,
        "artifact_flow": artifact_flow,
        "open_gates": [],
        "ready_actions": [],
        "gates": [{"id": gate_id, "status": "pass"} for gate_id in gates],
        "reports": {
            "remote_host_probe": str(paths["reports"] / "eagle3_remote_host_probe.json"),
            "final_artifact_validation": str(paths["reports"] / "eagle3_export_artifacts.json"),
            "trained_draft_spec_tokens_sweep": str(paths["reports"] / "trained_draft_spec_tokens_sweep.json"),
        },
        "gate_closure_contracts": [
            closure_contract(
                "reference_and_architecture",
                ["static_inputs", "hayate_modelopt_workflow", "hayate_specforge_reference", "draft_inventory", "upstream_drift"],
            ),
            closure_contract("remote_hayate_reference_probe", ["remote_host_probe", "remote_access_diagnostics"]),
            closure_contract("modelopt_loss_and_recipe", ["modelopt_loss_mask", "modelopt_recipe_overrides"]),
            closure_contract("target_rollout_corpus", ["rollout_state", "corpus_strategy"]),
            closure_contract(
                "runtime_container",
                ["container_preflight", "vllm_source_build", "vllm_abi_probe", "megatron_compat"],
            ),
            closure_contract("hidden_train_export_submit", ["pipeline_submit_preflight", "pipeline_analysis"]),
            closure_contract(
                "trained_artifact_contracts",
                ["hidden_validation", "training_checkpoint", "export_artifacts", "trained_draft_sweep"],
            ),
        ],
        "reference_evidence": {
            "remote_reference_proven": True,
            "local_modelopt": {
                "source": "Model-Optimizer",
                "role": "training_source",
            },
            "remote_probe": {
                "status": "pass",
                "reachable_hosts": ["synthetic-remote"],
                "remote_workdir": str(root / "remote_workdir"),
                "artifact_root_visible": True,
            },
            "hayate_modelopt": {
                "source": "reports/hayate_modelopt_workflow.json",
                "role": "reference_only",
                "remote_path_visible": True,
            },
            "hayate_specforge": {
                "source": "reports/hayate_specforge_reference.json",
                "role": "reference_only",
                "remote_path_visible": True,
            },
        },
        "reference_decisions": {
            "training_route": {
                "primary_route": "fixed_exported_eagle3_draft_first",
                "first_training_mode": "modelopt_offline_hidden_states",
            },
            "modelopt_source": {
                "source_of_truth": "local_modelopt",
                "upstream_drift_status": "pass",
            },
            "hayate_workflow": {
                "role": "reference_only",
                "status": "reference_only",
            },
            "specforge_qwen3_235b": {
                "role": "architecture_sanity_reference_only",
                "matched_fields": ["aux_layers", "hidden_size"],
                "rejected_fields": [{"field": "rope_theta", "reason": "synthetic mismatch"}],
            },
        },
    }


def write_required_reports(root: Path, paths: dict[str, Path]) -> None:
    reports = paths["reports"]
    write_json(reports / "eagle3_next_actions_validation.json", {"overall_status": "pass", "counts": {"pass": 1}})
    write_json(reports / "eagle3_operator_queue_transitions.json", {"overall_status": "pass", "scenarios": [{"status": "pass"}]})
    write_json(
        reports / "eagle3_operator_followups_validation.json",
        {"overall_status": "pass", "counts": {"pass": 1}, "followup_state_counts": {"pass": 2}},
    )
    write_json(
        reports / "megatron_probe_followup_validation.json",
        {
            "overall_status": "pass",
            "checks": [{"name": "synthetic guarded probe follow-up", "status": "pass"}],
            "counts": {"pass": 1},
        },
    )
    write_json(
        reports / "eagle3_preflight_robustness_validation.json",
        {
            "overall_status": "pass",
            "checks": [{"name": "synthetic clean preflight failure", "status": "pass"}],
        },
    )
    write_json(
        reports / "eagle3_remote_host_probe.json",
        {
            "overall_status": "pass",
            "reachable_hosts": ["synthetic-remote"],
            "counts": {"reachable": 1, "unreachable": 0, "requested": 1},
            "remote_workdir": str(root / "remote_workdir"),
            "artifact_root": str(root),
            "hosts": [
                {
                    "host": "synthetic-remote",
                    "reachable": True,
                    "status": "pass",
                    "commands": {"git": "/usr/bin/git", "python3": "/usr/bin/python3", "sbatch": "/usr/bin/sbatch"},
                    "paths": [
                        {"path": str(root / "remote_workdir"), "exists": True, "readable": True, "executable": True},
                        {"path": str(root), "exists": True, "readable": True, "executable": True},
                        {
                            "path": "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer",
                            "exists": True,
                            "readable": True,
                            "executable": True,
                        },
                        {
                            "path": "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/outputs",
                            "exists": True,
                            "readable": True,
                            "executable": True,
                        },
                    ],
                    "checks": {
                        "remote_workdir": {"status": "pass"},
                        "artifact_root": {"status": "pass"},
                        "hayate_modelopt": {"status": "pass"},
                    },
                }
            ],
        },
    )
    write_json(
        reports / "hayate_modelopt_workflow.json",
        {
            "overall_status": "reference_only",
            "hayate_modelopt_dir": "/synthetic/hayate/TensorRT-Model-Optimizer",
            "classification": {
                "classification": "reference_only",
                "summary": "Synthetic Hayate workflow is reference input, not a drop-in Qwen3-235B SWE recipe.",
            },
            "qwen_configs": [{"path": "examples/speculative_decoding/eagle_config_qwen3_30b.json"}],
        },
    )
    write_json(
        reports / "hayate_specforge_reference.json",
        {
            "overall_status": "reference_only",
            "qwen3_235b_comparison": {
                "status": "reference_only",
                "rows": [
                    {"field": "aux_layers", "specforge": [1, 46, 90], "current": [1, 46, 90], "match": True},
                    {"field": "rope_theta", "specforge": 1000000, "current": 5000000, "match": False},
                ],
                "conclusion": "Synthetic SpecForge config is a shape reference, not the Qwen3 Thinking source of truth.",
            },
        },
    )
    write_json(
        reports / "eagle3_draft_config_inventory.json",
        {
            "overall_status": "pass",
            "configs_scanned": 1,
            "roots": ["/synthetic/hayate/SpecForge/outputs"],
            "configs": [{"path": "/synthetic/hayate/SpecForge/outputs/qwen3/config.json", "matches_reference": True}],
            "warnings": [],
            "recommendation": "Use inventoried Hayate draft configs as reference evidence only.",
        },
    )
    (reports / "hayate_inventory.txt").write_text("synthetic Hayate inventory\n", encoding="utf-8")
    write_json(
        reports / "modelopt_recipe_overrides_current.json",
        {
            "overall_status": "pass",
            "wrapper": "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh",
            "training_mode": "offline",
            "recipe_config": str(ROOT / "Model-Optimizer/modelopt_recipes/general/speculative_decoding/eagle3.yaml"),
            "override_count": 28,
            "counts": {"pass": 8, "warn": 1},
            "warnings": ["synthetic ModelOpt import skipped outside training container"],
            "architecture_overrides": {
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "intermediate_size": 12288,
                "head_dim": 128,
                "rms_norm_eps": 0.000001,
                "rope_theta": 5000000,
                "use_aux_hidden_state": True,
                "use_input_layernorm_in_first_layer": True,
                "use_last_layernorm": True,
                "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
            },
        },
    )
    write_json(
        reports / "eagle3_operator_submit_packet_validation.json",
        {"overall_status": "pass", "counts": {"pass": 1}, "packet_status": "ready_for_operator_submit"},
    )
    write_json(
        reports / "eagle3_operator_ready_submit_preflight.json",
        {"overall_status": "pass", "submit_ready": True, "counts": {"pass": 1}, "ready_actions": []},
    )
    write_json(
        reports / "eagle3_operator_queue.json",
        {"overall_status": "current_ready_set_processed", "counts": {"ready_actions": 0}, "queue": [], "next_command": None},
    )
    write_json(
        reports / "container_preflight_analysis.json",
        {"overall_status": "pass", "status": "pass", "container": "synthetic.sqsh", "job_id": "12345"},
    )
    source_site = str(root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312")
    write_json(
        reports / "vllm_native_source_build.json",
        {
            "overall_status": "pass",
            "output_site": source_site,
            "vllm_source_spec": "vllm-0.10.2.tar.gz",
        },
    )
    write_json(
        reports / "vllm_native_abi_probe.json",
        {
            "overall_status": "pass",
            "results": [
                {
                    "site": source_site,
                    "returncode": 0,
                    "parsed": {
                        "vllm_c_ok": True,
                        "vllm_version": "0.10.2",
                    },
                }
            ],
        },
    )
    write_json(
        reports / "rollout_capture_state_advance.json",
        {"decision": {"overall_status": "pass", "next_step": "pipeline_dry_run"}, "output_data": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl")},
    )
    write_json(
        reports / "corpus_strategy.json",
        {
            "overall_status": "pass",
            "target_context": "swe_rl",
            "input_data": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
            "rollout_alignment": {
                "input_data_path": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
                "input_valid": True,
                "output_matches_input": True,
                "proves_actual_rollout_corpus": True,
                "rollout_output_path": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
                "rollout_status": "pass",
            },
            "decision": {
                "primary_source": "actual_rl_rollout",
                "provenance": {
                    "input_data_path": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
                    "input_valid": True,
                    "output_matches_input": True,
                    "proves_actual_rollout_corpus": True,
                    "rollout_output_path": str(paths["data"] / "qwen3_235b_swe_rollout_conversations.jsonl"),
                    "rollout_status": "pass",
                },
            },
        },
    )
    write_json(
        reports / "eagle3_pipeline_submit_preflight.json",
        {"overall_status": "pass", "submit_ready": True, "input_data": str(paths["data"]), "commands": {"pilot_submit": "true"}},
    )
    pipeline_jobs = reports / "eagle3_pipeline_jobs.env"
    pipeline_jobs.write_text(
        "\n".join(
            [
                "preflight_job=111",
                "dump_job=112",
                "validate_hiddens_job=113",
                "train_job=114",
                "export_job=115",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        reports / "eagle3_pipeline_gated_submit.json",
        {
            "overall_status": "pass",
            "executed": True,
            "command": "SUBMIT=true bash experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh",
            "checks": [{"name": "synthetic gated submit", "status": "pass", "detail": "synthetic"}],
            "job_file": "latest_eagle3_pipeline_jobs.txt",
            "job_file_copy": str(pipeline_jobs),
            "required_job_keys": ["dump_job", "train_job", "export_job"],
            "jobs": {
                "preflight_job": "111",
                "dump_job": "112",
                "validate_hiddens_job": "113",
                "train_job": "114",
                "export_job": "115",
            },
        },
    )
    write_json(reports / "eagle3_pipeline_analysis.json", {"overall_status": "pass", "counts": {"pass": 5}})
    write_json(
        paths["hidden"] / "validation_summary.json",
        {
            "overall_status": "pass",
            "total_files": 1,
            "checked_files": 1,
            "positive_loss_mask_files": 1,
            "expected_hidden_size": 4096,
            "expected_aux_count": 3,
            "require_loss_mask": True,
            "modelopt_loader_validation": {"dataset_items_checked": 1},
        },
    )
    write_json(
        reports / "eagle3_training_checkpoint.json",
        {
            "overall_status": "pass",
            "checkpoint_dir": str(paths["output"]),
            "trainer_global_step": 20,
            "modelopt_modes": ["eagle"],
            "checks": [{"status": "pass", "name": "synthetic"}],
        },
    )
    write_json(
        reports / "trained_draft_spec_tokens_sweep.json",
        {
            "overall_status": "pass",
            "vllm_draft_dir": str(paths["vllm"]),
            "rows": [{"gate_status": "pass", "spec_tokens": 3}],
            "recommendation": {"gate_status": "pass", "spec_tokens": 3},
            "execution_context": {
                "artifact_root": str(root),
                "config_file": str(ROOT / "grpo_qwen3_235b_swe.yaml"),
                "env_file": str(ROOT / "env.sh"),
                "chat_template": str(root / "templates/qwen3_generation_template.jinja2"),
                "repo_root": str(ROOT),
            },
        },
    )
    write_json(reports / "eagle3_training_path_manifest.json", training_path_manifest(root, paths))
    write_json(
        reports / "eagle3_training_path_manifest_validation.json",
        {"overall_status": "pass", "checks": [{"name": "synthetic training path manifest", "status": "pass"}]},
    )


def run_export_validator(paths: dict[str, Path]) -> dict[str, Any]:
    report = paths["reports"] / "eagle3_export_artifacts.json"
    command = [
        sys.executable,
        str(EXPORT_VALIDATOR),
        "--export-dir",
        str(paths["export"]),
        "--vllm-draft-dir",
        str(paths["vllm"]),
        "--verifier-config-dir",
        str(paths["verifier"]),
        "--reference-arch",
        str(REFERENCE_ARCH),
        "--export-config-compare-json",
        str(paths["export"] / "config_compare.json"),
        "--vllm-config-compare-json",
        str(paths["vllm"] / "config_compare.json"),
        "--json-out",
        str(report),
        "--markdown-out",
        str(paths["reports"] / "eagle3_export_artifacts.md"),
        "--fail-on-error",
    ]
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"export artifact validator failed:\n{result.stdout}")
    return read_json(report)


def run_completion(root: Path, paths: dict[str, Path], out_name: str = "completion.json") -> dict[str, Any]:
    out = paths["reports"] / out_name
    command = [
        sys.executable,
        str(COMPLETION_AUDIT),
        "--artifact-root",
        str(root),
        "--reference-arch",
        str(REFERENCE_ARCH),
        "--megatron-probe-followup-validation-json",
        str(paths["reports"] / "megatron_probe_followup_validation.json"),
        "--preflight-robustness-validation-json",
        str(paths["reports"] / "eagle3_preflight_robustness_validation.json"),
        "--json-out",
        str(out),
        "--markdown-out",
        str(out.with_suffix(".md")),
    ]
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"completion audit returned nonzero:\n{result.stdout}")
    return read_json(out)


def scenario_pass_contract(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    export_report = run_export_validator(paths)
    completion = run_completion(root, paths)
    problems: list[str] = []
    if export_report.get("overall_status") != "pass":
        problems.append(f"export validator status is {export_report.get('overall_status')!r}")
    if completion.get("overall_status") != "pass":
        open_required = [
            f"{row.get('area')}/{row.get('name')}={row.get('status')}"
            for row in completion.get("checks", [])
            if row.get("required") and row.get("status") != "pass"
        ]
        problems.append(f"completion status is {completion.get('overall_status')!r}; open={open_required[:8]}")
    return {
        "name": "complete_contract_passes",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "export_status": export_report.get("overall_status"),
        "problems": problems,
    }


def scenario_wrong_sweep_draft_fails(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    sweep = read_json(paths["reports"] / "trained_draft_spec_tokens_sweep.json")
    sweep["vllm_draft_dir"] = str(root / "other_vllm_draft")
    write_json(paths["reports"] / "trained_draft_spec_tokens_sweep.json", sweep)
    completion = run_completion(root, paths, out_name="completion_bad_sweep.json")
    failed = completion.get("overall_status") == "fail"
    matching_failures = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "trained-draft spec-token sweep" and row.get("status") == "fail"
    ]
    problems: list[str] = []
    if not failed or not matching_failures:
        problems.append("completion audit did not reject a sweep report for the wrong vLLM draft directory")
    return {
        "name": "wrong_sweep_draft_fails",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_failure_count": len(matching_failures),
        "problems": problems,
    }


def scenario_missing_training_path_is_incomplete(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    (paths["reports"] / "eagle3_training_path_manifest.json").unlink()
    completion = run_completion(root, paths, out_name="completion_missing_training_path.json")
    matching_required = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "Qwen3 Eagle3 training path manifest"
        and row.get("required")
        and row.get("status") == "warn"
    ]
    problems: list[str] = []
    if completion.get("overall_status") == "pass" or not matching_required:
        problems.append("completion audit accepted a missing required training path manifest")
    return {
        "name": "missing_training_path_is_incomplete",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_required_count": len(matching_required),
        "problems": problems,
    }


def scenario_incomplete_artifact_flow_is_incomplete(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    manifest_path = paths["reports"] / "eagle3_training_path_manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifact_flow_complete"] = False
    for item in manifest.get("artifact_flow") or []:
        if isinstance(item, dict) and item.get("id") == "vllm_eagle3_draft":
            item["proof_status"] = "open"
            item["report_statuses"] = {"pipeline_analysis": "pass", "export_artifacts": "missing"}
    write_json(manifest_path, manifest)
    completion = run_completion(root, paths, out_name="completion_incomplete_artifact_flow.json")
    matching_required = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "Qwen3 Eagle3 training path manifest"
        and row.get("required")
        and row.get("status") == "incomplete"
    ]
    problems: list[str] = []
    if completion.get("overall_status") == "pass" or not matching_required:
        problems.append("completion audit accepted an incomplete artifact_flow in the training path manifest")
    return {
        "name": "incomplete_artifact_flow_is_incomplete",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_required_count": len(matching_required),
        "problems": problems,
    }


def scenario_abi_other_site_pass_source_site_fails(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    source_site = str(root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312")
    write_json(
        paths["reports"] / "vllm_native_abi_probe.json",
        {
            "overall_status": "pass",
            "results": [
                {
                    "site": source_site,
                    "returncode": 1,
                    "parsed": {
                        "vllm_c_ok": False,
                        "vllm_c_error": "ImportError: undefined symbol: synthetic",
                        "vllm_version": "0.10.2",
                    },
                },
                {
                    "site": str(root / "python_site/other_vllm_site"),
                    "returncode": 0,
                    "parsed": {
                        "compilation_config_ok": True,
                        "vllm_c_ok": True,
                        "vllm_version": "0.10.2",
                    },
                },
            ],
        },
    )
    completion = run_completion(root, paths, out_name="completion_bad_abi_source_site.json")
    matching_failures = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "source-built vLLM native ABI PASS" and row.get("status") == "fail"
    ]
    problems: list[str] = []
    if completion.get("overall_status") == "pass" or not matching_failures:
        problems.append("completion audit accepted ABI PASS from a different site while the source-built site failed")
    return {
        "name": "abi_other_site_pass_source_site_fails",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_failure_count": len(matching_failures),
        "problems": problems,
    }


def scenario_non_required_planning_failures_are_warnings(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    write_json(
        paths["reports"] / "cluster_environment_probe.json",
        {
            "overall_status": "fail",
            "host": {"hostname": "synthetic-lightweight-host"},
            "inputs": {"artifact_root": str(root)},
            "checks": [{"name": "sbatch available", "status": "fail", "required": True}],
        },
    )
    write_json(
        paths["reports"] / "eagle3_readiness.json",
        {
            "overall_status": "fail",
            "counts": {"fail": 2, "missing": 3, "pass": 5},
            "checks": [{"name": "synthetic readiness gap", "status": "fail"}],
        },
    )
    write_json(
        paths["reports"] / "eagle3_operator_ready_submit_preflight.json",
        {
            "overall_status": "fail",
            "submit_ready": False,
            "counts": {"fail": 2, "pass": 3},
            "ready_actions": [{"id": "submit_container_preflight", "submits_slurm": True}],
            "checks": [
                {"area": "slurm", "name": "sbatch", "status": "fail", "detail": "sbatch is not on PATH"},
                {
                    "area": "path",
                    "name": "submit_container_preflight input data",
                    "status": "fail",
                    "detail": "input data is not visible on this lightweight host",
                },
            ],
        },
    )
    completion = run_completion(root, paths, out_name="completion_non_required_planning_failures.json")
    by_name = {row.get("name"): row for row in completion.get("checks", [])}
    cluster_status = (by_name.get("cluster environment probe") or {}).get("status")
    readiness_status = (by_name.get("readiness audit report") or {}).get("status")
    operator_ready_status = (by_name.get("operator ready-submit preflight") or {}).get("status")
    fail_count = int((completion.get("counts") or {}).get("fail") or 0)
    problems: list[str] = []
    if completion.get("overall_status") != "pass":
        problems.append(f"completion status is {completion.get('overall_status')!r}")
    if cluster_status != "warn":
        problems.append(f"cluster environment probe status is {cluster_status!r}, expected warn")
    if readiness_status != "warn":
        problems.append(f"readiness audit report status is {readiness_status!r}, expected warn")
    if operator_ready_status != "warn":
        problems.append(f"operator ready-submit preflight status is {operator_ready_status!r}, expected warn")
    if fail_count:
        problems.append(f"non-required planning failures leaked into fail count: {fail_count}")
    return {
        "name": "non_required_planning_failures_are_warnings",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "cluster_status": cluster_status,
        "readiness_status": readiness_status,
        "operator_ready_status": operator_ready_status,
        "problems": problems,
    }


def scenario_empty_inventory_is_incomplete(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    write_json(
        paths["reports"] / "eagle3_draft_config_inventory.json",
        {
            "overall_status": "missing",
            "configs_scanned": 0,
            "roots": ["/synthetic/hayate/empty"],
            "root_statuses": [{"path": "/synthetic/hayate/empty", "status": "missing"}],
            "warnings": [],
            "configs": [],
            "recommendation": "Synthetic empty inventory lacks access-limit evidence.",
        },
    )
    completion = run_completion(root, paths, out_name="completion_empty_inventory.json")
    matching_incomplete = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "Hayate/draft config inventory" and row.get("status") == "incomplete"
    ]
    problems: list[str] = []
    if completion.get("overall_status") == "pass" or not matching_incomplete:
        problems.append("completion audit accepted an empty draft inventory without warnings or configs")
    return {
        "name": "empty_inventory_is_incomplete",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_incomplete_count": len(matching_incomplete),
        "problems": problems,
    }


def scenario_missing_references_are_incomplete(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    write_json(
        paths["reports"] / "eagle3_remote_host_probe.json",
        {
            "overall_status": "unreachable",
            "reachable_hosts": [],
            "counts": {"reachable": 0, "unreachable": 4, "requested": 4},
        },
    )
    write_json(
        paths["reports"] / "hayate_modelopt_workflow.json",
        {
            "overall_status": "missing_reference",
            "classification": "not_accessible",
            "detail": "synthetic missing Hayate ModelOpt checkout",
            "path": {"exists": False, "requested": "/synthetic/missing"},
            "qwen_configs": [],
        },
    )
    write_json(
        paths["reports"] / "hayate_specforge_reference.json",
        {
            "overall_status": "missing_reference",
            "specforge_dir": "/synthetic/missing-specforge",
            "qwen3_235b_comparison": {"status": "missing_inputs", "rows": []},
        },
    )
    completion = run_completion(root, paths, out_name="completion_missing_references.json")
    expected_incomplete = {
        "remote ModelOpt/Hayate host probe",
        "Hayate ModelOpt workflow analysis",
        "Hayate SpecForge Qwen3 reference comparison",
    }
    observed = {
        row.get("name")
        for row in completion.get("checks", [])
        if row.get("required") and row.get("status") == "incomplete"
    }
    problems: list[str] = []
    missing = sorted(expected_incomplete - observed)
    if completion.get("overall_status") == "pass":
        problems.append("completion audit accepted missing remote/Hayate reference evidence")
    if missing:
        problems.append(f"missing incomplete reference checks: {missing}")
    return {
        "name": "missing_references_are_incomplete",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_incomplete_count": len(expected_incomplete & observed),
        "problems": problems,
    }


def scenario_reachable_probe_missing_paths_is_incomplete(root: Path) -> dict[str, Any]:
    paths = materialize_artifacts(root)
    write_required_reports(root, paths)
    run_export_validator(paths)
    write_json(
        paths["reports"] / "eagle3_remote_host_probe.json",
        {
            "overall_status": "pass",
            "reachable_hosts": ["synthetic-remote"],
            "counts": {"reachable": 1, "unreachable": 0, "requested": 1},
            "remote_workdir": str(root / "remote_workdir"),
            "artifact_root": str(root),
            "hosts": [
                {
                    "host": "synthetic-remote",
                    "reachable": True,
                    "commands": {"python3": "/usr/bin/python3"},
                    "paths": [
                        {"path": "/lustre", "exists": True, "readable": True, "executable": True},
                    ],
                }
            ],
        },
    )
    completion = run_completion(root, paths, out_name="completion_bad_remote_probe.json")
    matching_incomplete = [
        row
        for row in completion.get("checks", [])
        if row.get("name") == "remote ModelOpt/Hayate host probe" and row.get("status") == "incomplete"
    ]
    problems: list[str] = []
    if completion.get("overall_status") == "pass" or not matching_incomplete:
        problems.append("completion audit accepted a reachable remote probe without required ModelOpt/Hayate/artifact paths")
    return {
        "name": "reachable_probe_missing_paths_is_incomplete",
        "status": "pass" if not problems else "fail",
        "completion_status": completion.get("overall_status"),
        "matching_incomplete_count": len(matching_incomplete),
        "problems": problems,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Completion Contract Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | completion | detail |",
        "| --- | --- | --- | --- |",
    ]
    for scenario in payload["scenarios"]:
        detail = "; ".join(scenario.get("problems") or []) or "-"
        lines.append(
            f"| {scenario['name']} | {scenario['status']} | `{scenario.get('completion_status')}` | {detail.replace('|', '/')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_completion_contract_"))
    scenarios: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        scenarios.append(scenario_pass_contract(temp_root / "pass" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_wrong_sweep_draft_fails(temp_root / "bad_sweep" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_missing_training_path_is_incomplete(temp_root / "missing_training_path" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_incomplete_artifact_flow_is_incomplete(temp_root / "incomplete_artifact_flow" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_abi_other_site_pass_source_site_fails(temp_root / "bad_abi_source_site" / "qwen3_235b_eagle3"))
        scenarios.append(
            scenario_non_required_planning_failures_are_warnings(
                temp_root / "non_required_planning_failures" / "qwen3_235b_eagle3"
            )
        )
        scenarios.append(scenario_empty_inventory_is_incomplete(temp_root / "empty_inventory" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_missing_references_are_incomplete(temp_root / "missing_refs" / "qwen3_235b_eagle3"))
        scenarios.append(scenario_reachable_probe_missing_paths_is_incomplete(temp_root / "bad_remote_probe" / "qwen3_235b_eagle3"))
        for scenario in scenarios:
            problems.extend(f"{scenario['name']}: {problem}" for problem in scenario.get("problems") or [])
    except Exception as exc:
        problems.append(str(exc))
    finally:
        if args.keep_temp:
            print(f"Kept temp artifacts under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    overall = "pass" if not problems and all(item.get("status") == "pass" for item in scenarios) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
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
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
