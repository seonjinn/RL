#!/usr/bin/env python3
"""Validate Eagle3 next-action state transitions with synthetic reports.

This is a no-submit test for the operator state machine. It builds temporary
report sets that represent each major Qwen3-235B Eagle3 gate and verifies that
plan_eagle3_next_actions.py promotes exactly the expected ready action.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "experiments/eagle3_qwen3_235b"
PLAN_SCRIPT = SCRIPT_DIR / "plan_eagle3_next_actions.py"
VALIDATE_SCRIPT = SCRIPT_DIR / "validate_eagle3_next_action_plan.py"
SHEET_SCRIPT = SCRIPT_DIR / "create_eagle3_operator_sheet.py"
SHEET_VALIDATOR = SCRIPT_DIR / "validate_eagle3_operator_sheet.py"
EXECUTION_VALIDATOR = SCRIPT_DIR / "validate_eagle3_operator_execution.py"
FOLLOWUP_VALIDATOR = SCRIPT_DIR / "validate_eagle3_operator_followups.py"
SUBMIT_PACKET_SCRIPT = SCRIPT_DIR / "create_eagle3_operator_submit_packet.py"
SUBMIT_PACKET_VALIDATOR = SCRIPT_DIR / "validate_eagle3_operator_submit_packet.py"
KNOWN_READY_ACTIONS = {
    "probe_remote_hosts",
    "submit_vllm_source_build",
    "poll_vllm_source_build",
    "submit_source_vllm_abi_probe",
    "submit_container_preflight",
    "submit_megatron_compat_probe",
    "poll_megatron_compat_probe",
    "submit_rollout_capture",
    "submit_rollout_fallback",
    "submit_full_swegym_rollout",
    "run_pipeline_submit_preflight",
    "submit_eagle3_pilot_pipeline",
    "run_post_export_artifact_validations",
    "submit_trained_draft_spec_tokens_sweep",
}


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


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def base_commands(artifact: Path) -> dict[str, str]:
    return {
        "container_submit": shell_join(
            [
                f"SUBMIT=true",
                f"ARTIFACT_ROOT={artifact}",
                "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
                "SBATCH_PARTITION=batch",
                "PREFLIGHT_GPUS_PER_NODE=4",
                f"MODELOPT_DIR={ROOT / 'Model-Optimizer'}",
                f"VERIFIER_CONFIG_DIR={artifact / 'verifier_config'}",
                f"INPUT_DATA={artifact / 'data/pilot_existing_chat_content_64.jsonl'}",
                f"CHAT_TEMPLATE={artifact / 'templates/qwen3_generation_template.jinja2'}",
                "CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh",
                "MOUNTS=/lustre:/lustre",
                "bash",
                "experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh",
            ]
        ),
        "rollout_submit": shell_join(
            [
                f"ARTIFACT_ROOT={artifact}",
                "SWE_REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                "REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                f"CONFIG_FILE={ROOT / 'grpo_qwen3_235b_swe.yaml'}",
                f"ENV_FILE={ROOT / 'env.sh'}",
                f"CHAT_TEMPLATE={artifact / 'templates/qwen3_generation_template.jinja2'}",
                f"ROLLOUT_LOG_DIR={artifact / 'rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke'}",
                f"OUTPUT_CONVERSATIONS={artifact / 'data/qwen3_235b_swe_rollout_conversations.jsonl'}",
                "DRY_RUN=false",
                "MAX_NUM_STEPS=1",
                "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
                "SBATCH_PARTITION=batch",
                "bash",
                "experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh",
            ]
        ),
        "pipeline_dry_run": shell_join(
            [
                "SUBMIT=false",
                "RUN_PILOT=true",
                f"ARTIFACT_ROOT={artifact}",
                f"INPUT_DATA={artifact / 'data/qwen3_235b_swe_rollout_conversations.jsonl'}",
                "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
                "SBATCH_PARTITION=batch",
                "bash",
                "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh",
            ]
        ),
        "pipeline_submit": shell_join(
            [
                "SUBMIT=true",
                "RUN_PILOT=true",
                f"ARTIFACT_ROOT={artifact}",
                f"INPUT_DATA={artifact / 'data/qwen3_235b_swe_rollout_conversations.jsonl'}",
                f"HIDDEN_STATES_DIR={artifact / 'hidden_states'}",
                f"OUTPUT_DIR={artifact / 'modelopt_ckpt'}",
                f"EXPORT_DIR={artifact / 'exported_hf'}",
                f"VLLM_DRAFT_DIR={artifact / 'vllm_draft'}",
                "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
                "SBATCH_PARTITION=batch",
                "bash",
                "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh",
            ]
        ),
        "pipeline_gated_submit": shell_join(
            [
                sys.executable,
                "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py",
                "--artifact-root",
                str(artifact),
                "--preflight-json",
                str(artifact / "reports/eagle3_pipeline_submit_preflight.json"),
                "--json-out",
                str(artifact / "reports/eagle3_pipeline_gated_submit.json"),
                "--markdown-out",
                str(artifact / "reports/eagle3_pipeline_gated_submit.md"),
                "--execute",
                "--allow-heavy-gpu",
            ]
        ),
        "fallback_submit": shell_join(
            [
                f"ARTIFACT_ROOT={artifact}",
                "SWE_REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                "REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                f"CONFIG_FILE={ROOT / 'grpo_qwen3_235b_swe.yaml'}",
                f"ENV_FILE={ROOT / 'env.sh'}",
                f"CHAT_TEMPLATE={artifact / 'templates/qwen3_generation_template.jinja2'}",
                f"ROLLOUT_LOG_DIR={artifact / 'rl_rollout_capture_logs/qwen3_235b_swe_capture_balanced24n4g'}",
                f"OUTPUT_CONVERSATIONS={artifact / 'data/qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl'}",
                "DRY_RUN=false",
                "MAX_NUM_STEPS=1",
                "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
                "SBATCH_PARTITION=batch",
                "NUM_NODES=24",
                "NUM_GEN_NODES=8",
                "bash",
                "experiments/eagle3_qwen3_235b/submit_source_vllm_rollout_smoke.sh",
            ]
        ),
    }


def common_reports(artifact: Path) -> dict[str, dict[str, Any]]:
    source_site = artifact / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"
    return {
        "eagle3_training_scale.json": {
            "overall_status": "pass",
            "training_defaults": {"effective_global_batch": 8, "epochs": 1, "max_seq_len": 16384},
            "stage_plan": [
                {"name": "smoke", "examples": 5, "max_steps": 0, "nominal_epoch_steps": 1},
                {"name": "pilot", "examples": 8, "max_steps": 20, "nominal_epoch_steps": 1},
                {"name": "swegym_first_calibration", "examples": 2438, "max_steps": 1000, "nominal_epoch_steps": 305},
                {"name": "target_domain_calibration", "examples": 50000, "max_steps": 2000, "nominal_epoch_steps": 6250},
                {"name": "production_candidate", "examples": 100000, "max_steps": None, "nominal_epoch_steps": 12500},
                {"name": "generic_optional", "examples": 500000, "max_steps": None, "nominal_epoch_steps": 62500},
            ],
        },
        "modelopt_loss_mask_patch.json": {"overall_status": "pass"},
        "nemo_rl_eagle3_drift.json": {"overall_status": "pass"},
        "megatron_compat_probe.json": {
            "overall_status": "pass",
            "api": {
                "tpaware_grouped_linear_detection": {
                    "TEColumnParallelGroupedLinear": "replicated",
                    "TERowParallelGroupedLinear": "replicated",
                },
                "grouped_linear_temporary_weight_attr": {
                    "has_weight": True,
                    "weight_is_weight0": True,
                },
                "community_import_save_compat": {
                    "helper_available": True,
                    "checkpoint_fallback_available": True,
                    "model_load_save_available": False,
                    "checkpointing_save_available": True,
                },
            },
            "errors": [],
        },
        "vllm_native_source_build.json": {
            "overall_status": "pass",
            "output_site": str(source_site),
        },
        "vllm_native_abi_probe.json": {
            "overall_status": "pass",
            "results": [
                {
                    "site": str(source_site),
                    "returncode": 0,
                    "parsed": {"vllm_c_ok": True, "compilation_config_ok": True},
                }
            ],
        },
        "eagle3_readiness.json": {"overall_status": "incomplete"},
        "eagle3_remote_host_probe.json": {
            "overall_status": "pass",
            "reachable_hosts": ["synthetic-remote"],
            "counts": {"reachable": 1, "unreachable": 0, "requested": 1},
            "hosts": [
                {
                    "host": "synthetic-remote",
                    "reachable": True,
                    "commands": {"git": "/usr/bin/git", "python3": "/usr/bin/python3"},
                    "paths": [
                        {"path": str(artifact), "exists": True, "readable": True},
                        {
                            "path": "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer",
                            "exists": True,
                            "readable": True,
                        },
                        {
                            "path": "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge/outputs",
                            "exists": True,
                            "readable": True,
                        },
                    ],
                }
            ],
        },
    }


def report_set(artifact: Path, state: str) -> dict[str, dict[str, Any]]:
    commands = base_commands(artifact)
    reports = common_reports(artifact)

    container_incomplete = {
        "overall_status": "incomplete",
        "next_action": {"submit_command": commands["container_submit"]},
    }
    container_pass = {"overall_status": "pass"}
    rollout_submit_pass = {
        "overall_status": "pass",
        "submit_ready": True,
        "commands": {
            "submit": commands["rollout_submit"],
            "analyze_job": shell_join(
                [
                    sys.executable,
                    "experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py",
                    "--artifact-root",
                    str(artifact),
                    "--repo-root",
                    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                    "--rollout-log-dir",
                    str(artifact / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke"),
                    "--output-data",
                    str(artifact / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
                ]
            ),
        },
    }
    rollout_submit_fail = {
        "overall_status": "fail",
        "submit_ready": False,
        "checks": [
            {
                "status": "fail",
                "name": "prerequisite_visibility",
                "detail": "synthetic cluster-visible inputs are not proven yet",
            }
        ],
    }
    rollout_not_submitted = {
        "decision": {"overall_status": "not_submitted", "next_step": "submit_capture"},
        "commands": {"submit_capture": commands["rollout_submit"]},
    }
    rollout_pipeline_ready = {
        "overall_status": "pass",
        "decision": {"overall_status": "pass", "next_step": "pipeline_dry_run"},
        "output_data": str(artifact / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
        "commands": {"pipeline_dry_run": commands["pipeline_dry_run"]},
    }
    rollout_pending_poll = {
        "overall_status": "running",
        "decision": {"overall_status": "running", "next_step": "poll"},
        "commands": {
            "poll": shell_join(
                [
                    sys.executable,
                    "experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py",
                    "--job-id",
                    "12345",
                    "--json-out",
                    str(artifact / "reports/rollout_capture_state_advance.json"),
                    "--markdown-out",
                    str(artifact / "reports/rollout_capture_state_advance.md"),
                ]
            )
        },
    }
    fallback_ready = {
        "overall_status": "fallback_ready",
        "recommendation": "submit_balanced_24n4g_smoke",
        "detail": "official job estimated start delay 240.0 min exceeds 120 min; selected fallback balanced_24n4g_smoke",
        "next_command": commands["fallback_submit"],
        "selected_fallback": {
            "id": "balanced_24n4g_smoke",
            "output": str(artifact / "data/qwen3_235b_swe_rollout_conversations_balanced24n4g.jsonl"),
        },
    }
    full_rollout_gate_ready = {
        "decision": {"overall_status": "ready", "next_step": "submit_full_rollout"},
        "smoke_state_json": str(artifact / "reports/rollout_capture_state_advance.json"),
        "full_preflight_json": str(artifact / "reports/rollout_capture_swegym_full_submit_preflight.json"),
    }
    full_rollout_submit_preflight = {
        "overall_status": "pass",
        "submit_ready": True,
        "output_conversations": str(artifact / "data/qwen3_235b_swe_rollout_conversations_vllm0102src_swegym_full.jsonl"),
        "rollout_log_dir": str(artifact / "rl_rollout_capture_logs/qwen3_235b_swe_capture_vllm0102src_swegym_full"),
        "commands": {"submit": "echo Submitted batch job 123456"},
    }
    pipeline_submit_ready = {
        "overall_status": "pass",
        "submit_ready": True,
        "commands": {
            "dry_run": commands["pipeline_dry_run"],
            "pilot_submit": commands["pipeline_submit"],
            "gated_pilot_submit": commands["pipeline_gated_submit"],
            "analyze_pipeline": shell_join(
                [
                    sys.executable,
                    "experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py",
                    "--job-file",
                    "latest_eagle3_pipeline_jobs.txt",
                    "--logs-dir",
                    "logs",
                    "--input-data",
                    str(artifact / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
                    "--hidden-states-dir",
                    str(artifact / "hidden_states"),
                    "--hidden-validation-json",
                    str(artifact / "hidden_states/validation_summary.json"),
                    "--training-checkpoint-json",
                    str(artifact / "reports/eagle3_training_checkpoint.json"),
                    "--output-dir",
                    str(artifact / "modelopt_ckpt"),
                    "--export-dir",
                    str(artifact / "exported_hf"),
                    "--vllm-draft-dir",
                    str(artifact / "vllm_draft"),
                    "--export-artifacts-json",
                    str(artifact / "reports/eagle3_export_artifacts.json"),
                    "--markdown-out",
                    str(artifact / "reports/eagle3_pipeline_analysis.md"),
                    "--json-out",
                    str(artifact / "reports/eagle3_pipeline_analysis.json"),
                ]
            ),
        },
    }
    pipeline_analysis_pass = {
        "overall_status": "pass",
        "commands": {"pilot_submit": commands["pipeline_submit"]},
        "next_action": {
            "resume_command": shell_join(
                [
                    f"ARTIFACT_ROOT={artifact}",
                    "SWE_REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                    "REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
                    f"CONFIG_FILE={ROOT / 'grpo_qwen3_235b_swe.yaml'}",
                    f"ENV_FILE={ROOT / 'env.sh'}",
                    f"CHAT_TEMPLATE={artifact / 'templates/qwen3_generation_template.jinja2'}",
                    "bash",
                    "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh",
                ]
            )
        },
    }
    training_checkpoint_pass = {
        "overall_status": "pass",
        "checkpoint_dir": str(artifact / "modelopt_ckpt"),
        "modelopt_modes": ["eagle"],
        "trainer_global_step": 20,
        "checks": [{"status": "pass"}],
    }
    export_artifacts_pass = {
        "overall_status": "pass",
        "export_dir": str(artifact / "exported_hf"),
        "vllm_draft_dir": str(artifact / "vllm_draft"),
        "checks": [{"status": "pass"}],
    }
    sweep_pass = {
        "overall_status": "pass",
        "vllm_draft_dir": str(artifact / "vllm_draft"),
        "execution_context": {
            "artifact_root": str(artifact),
            "repo_root": "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
            "swe_repo_root": "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
            "config_file": str(ROOT / "grpo_qwen3_235b_swe.yaml"),
            "env_file": str(ROOT / "env.sh"),
            "chat_template": str(artifact / "templates/qwen3_generation_template.jinja2"),
        },
        "rows": [{"tokens": 3, "gate_status": "pass"}],
        "recommendation": {"tokens": 3, "gate_status": "pass"},
    }

    if state == "source_vllm_missing":
        reports.pop("vllm_native_source_build.json", None)
        reports.pop("vllm_native_abi_probe.json", None)
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_not_submitted,
            }
        )
    elif state == "frontier":
        reports.update(
            {
                "container_preflight_analysis.json": container_incomplete,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_not_submitted,
            }
        )
    elif state == "rollout_submit_failed_prereq":
        reports.pop("vllm_native_source_build.json", None)
        reports.pop("vllm_native_abi_probe.json", None)
        reports.update(
            {
                "eagle3_remote_host_probe.json": {
                    "overall_status": "unreachable",
                    "reachable_hosts": [],
                    "counts": {"reachable": 0, "unreachable": 4, "requested": 4},
                },
                "container_preflight_analysis.json": container_incomplete,
                "rollout_capture_submit_preflight.json": rollout_submit_fail,
                "rollout_capture_state_advance.json": rollout_not_submitted,
            }
        )
    elif state == "megatron_probe_pending":
        reports.pop("megatron_compat_probe.json", None)
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_not_submitted,
            }
        )
    elif state == "rollout_ready":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
            }
        )
    elif state == "rollout_pending_fallback_ready":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pending_poll,
                "rollout_fallback_decision.json": fallback_ready,
            }
        )
    elif state == "full_rollout_ready":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "full_swegym_after_smoke_gate.json": full_rollout_gate_ready,
                "rollout_capture_swegym_full_submit_preflight.json": full_rollout_submit_preflight,
            }
        )
    elif state == "pipeline_ready":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "eagle3_pipeline_submit_preflight.json": pipeline_submit_ready,
            }
        )
    elif state == "pipeline_artifacts_missing":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "eagle3_pipeline_submit_preflight.json": pipeline_submit_ready,
                "eagle3_pipeline_analysis.json": pipeline_analysis_pass,
            }
        )
    elif state == "pipeline_passed":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "eagle3_pipeline_submit_preflight.json": pipeline_submit_ready,
                "eagle3_pipeline_analysis.json": pipeline_analysis_pass,
                "eagle3_training_checkpoint.json": training_checkpoint_pass,
                "eagle3_export_artifacts.json": export_artifacts_pass,
            }
        )
    elif state == "sweep_passed":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "eagle3_pipeline_submit_preflight.json": pipeline_submit_ready,
                "eagle3_pipeline_analysis.json": pipeline_analysis_pass,
                "eagle3_training_checkpoint.json": training_checkpoint_pass,
                "eagle3_export_artifacts.json": export_artifacts_pass,
                "trained_draft_spec_tokens_sweep.json": sweep_pass,
            }
        )
    elif state == "remote_probe_unreachable":
        reports.update(
            {
                "container_preflight_analysis.json": container_pass,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_pipeline_ready,
                "eagle3_pipeline_submit_preflight.json": pipeline_submit_ready,
                "eagle3_pipeline_analysis.json": pipeline_analysis_pass,
                "eagle3_training_checkpoint.json": training_checkpoint_pass,
                "eagle3_export_artifacts.json": export_artifacts_pass,
                "trained_draft_spec_tokens_sweep.json": sweep_pass,
                "eagle3_remote_host_probe.json": {
                    "overall_status": "unreachable",
                    "reachable_hosts": [],
                    "counts": {"reachable": 0, "unreachable": 4, "requested": 4},
                },
            }
        )
    elif state == "readiness_failed_frontier":
        reports["eagle3_readiness.json"] = {
            "overall_status": "fail",
            "counts": {"fail": 3, "missing": 4, "pass": 2},
        }
        reports.update(
            {
                "container_preflight_analysis.json": container_incomplete,
                "rollout_capture_submit_preflight.json": rollout_submit_pass,
                "rollout_capture_state_advance.json": rollout_not_submitted,
            }
        )
    else:
        raise ValueError(f"unknown transition state: {state}")

    return reports


def materialize_reports(artifact: Path, state: str) -> None:
    for name, payload in report_set(artifact, state).items():
        write_json(artifact / "reports" / name, payload)
    (artifact / "reports").mkdir(parents=True, exist_ok=True)
    if state == "megatron_probe_pending":
        (artifact / "reports/megatron_compat_probe_job.env").write_text(
            "\n".join(
                [
                    "megatron_compat_probe_job=2867766",
                    f"json={artifact / 'reports/megatron_compat_probe.json'}",
                    f"markdown={artifact / 'reports/megatron_compat_probe.md'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    (artifact / "reports/eagle3_resource_profile.env").write_text(
        "\n".join(
            [
                "export DUMP_GPUS_PER_NODE=4",
                "export TRAIN_GPUS_PER_NODE=4",
                "export EXPORT_GPUS_PER_NODE=1",
                "export TP=4",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def run_checked(command: list[str], label: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    result = run_command(command)
    if expect_success and result.returncode:
        raise RuntimeError(
            f"{label} failed with return code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly passed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def run_plan(artifact: Path, out_path: Path) -> dict[str, Any]:
    run_checked(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--artifact-root",
            str(artifact),
            "--json-out",
            str(out_path),
            "--markdown-out",
            str(out_path.with_suffix(".md")),
        ],
        "planner",
    )
    return read_json(out_path)


def run_validator(plan_path: Path, expected_ready: set[str]) -> None:
    command = [sys.executable, str(VALIDATE_SCRIPT), "--plan-json", str(plan_path), "--fail-on-warn"]
    for action_id in sorted(expected_ready):
        command.extend(["--expect-ready-action", action_id])
    for action_id in sorted(KNOWN_READY_ACTIONS - expected_ready):
        command.extend(["--forbid-ready-action", action_id])
    run_checked(command, "plan validator")


def ready_action_ids(plan: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id"))
        for item in plan.get("next_actions", [])
        if isinstance(item, dict) and item.get("status") == "ready_for_operator" and item.get("command")
    }


def check_action_shape(plan: dict[str, Any], expected_ready: set[str]) -> list[str]:
    problems: list[str] = []
    ready = ready_action_ids(plan)
    if ready != expected_ready:
        problems.append(f"ready actions {sorted(ready)} != expected {sorted(expected_ready)}")

    by_id = {str(item.get("id")): item for item in plan.get("next_actions", []) if isinstance(item, dict)}
    for action_id in sorted(expected_ready):
        action = by_id.get(action_id) or {}
        after_text = "\n".join(str(item) for item in action.get("after_commands") or [])
        if after_text and "refresh_eagle3_operator_state.py" not in after_text:
            problems.append(f"{action_id} after_commands must refresh full operator state")

    container = by_id.get("submit_container_preflight")
    if container and container.get("status") == "ready_for_operator":
        command = str(container.get("command") or "")
        if "PREFLIGHT_GPUS_PER_NODE=4" not in command:
            problems.append("container preflight command must request PREFLIGHT_GPUS_PER_NODE=4")

    rollout_poll = by_id.get("rollout_poll")
    if rollout_poll and rollout_poll.get("status") == "ready_for_operator":
        tokens = shlex.split(str(rollout_poll.get("command") or ""))
        if not any(token.endswith("advance_rollout_capture_state.py") for token in tokens):
            problems.append("rollout poll command must run advance_rollout_capture_state.py")
        if "--json-out" not in tokens or "--markdown-out" not in tokens:
            problems.append("rollout poll command must write JSON and Markdown reports")
        report = str(rollout_poll.get("report") or "")
        if "--json-out" in tokens:
            idx = tokens.index("--json-out")
            observed = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if report and observed != report:
                problems.append("rollout poll command --json-out must match the selected report path")

    megatron_poll = by_id.get("poll_megatron_compat_probe")
    if megatron_poll and megatron_poll.get("status") == "ready_for_operator":
        command = str(megatron_poll.get("command") or "")
        required_snippets = [
            "ARTIFACT_ROOT=",
            "PROBE_JOB_ID=2867766",
            "SUBMIT_ROLLOUT=false",
            "followup_megatron_probe_to_rollout.sh",
        ]
        missing = [snippet for snippet in required_snippets if snippet not in command]
        if missing:
            problems.append(f"megatron poll command missing snippets: {missing}")

    source_build = by_id.get("submit_vllm_source_build")
    if source_build and source_build.get("status") == "ready_for_operator":
        command = str(source_build.get("command") or "")
        required_snippets = [
            "ARTIFACT_ROOT=",
            "SUBMIT=true",
            "submit_vllm_native_source_build.sh",
        ]
        missing = [snippet for snippet in required_snippets if snippet not in command]
        if missing:
            problems.append(f"source vLLM build command missing snippets: {missing}")
        after_text = "\n".join(str(item) for item in source_build.get("after_commands") or [])
        if "analyze_vllm_source_build_job.py" not in after_text:
            problems.append("source vLLM build after_commands must analyze the build job")

    pipeline_preflight = by_id.get("run_pipeline_submit_preflight")
    if pipeline_preflight and pipeline_preflight.get("status") == "ready_for_operator":
        command = str(pipeline_preflight.get("command") or "")
        tokens = shlex.split(str(pipeline_preflight.get("command") or ""))
        if tokens.count("--json-out") != 1:
            problems.append("pipeline submit preflight command must contain exactly one --json-out")
        artifact_root = Path(str(plan.get("artifact_root") or ""))
        if (artifact_root / "reports/eagle3_resource_profile.env").exists():
            required_snippets = [
                "--dump-gpus-per-node 4",
                "--train-gpus-per-node 4",
                "--export-gpus-per-node 1",
                "--tp 4",
                f"--slurm-capacity-env {artifact_root / 'reports/eagle3_resource_profile.env'}",
            ]
            missing = [snippet for snippet in required_snippets if snippet not in command]
            if missing:
                problems.append(f"pipeline submit preflight command missing resource profile snippets: {missing}")

    sweep = by_id.get("submit_trained_draft_spec_tokens_sweep")
    if sweep and sweep.get("status") == "ready_for_operator":
        command = str(sweep.get("command") or "")
        required_snippets = [
            "ARTIFACT_ROOT=",
            "REPO_ROOT=",
            "SWE_REPO_ROOT=",
            "CONFIG_FILE=",
            "ENV_FILE=",
            "CHAT_TEMPLATE=",
            "VLLM_DRAFT_DIR=",
            "submit_trained_draft_spec_tokens_sweep.sh",
        ]
        missing = [snippet for snippet in required_snippets if snippet not in command]
        if missing:
            problems.append(f"sweep command missing execution-context snippets: {missing}")
        after_text = "\n".join(str(item) for item in sweep.get("after_commands") or [])
        if "--fail-on-missing-spec-metrics" not in after_text:
            problems.append("sweep after_commands must use --fail-on-missing-spec-metrics")

    return problems


def rewrite_ready_action_command(path: Path, action_id: str, key: str, command: str) -> None:
    payload = read_json(path)
    rewritten = False
    for item in payload.get("ready_actions", []):
        if isinstance(item, dict) and item.get("id") == action_id:
            item[key] = command
            rewritten = True
    if not rewritten:
        raise RuntimeError(f"could not rewrite {key} for {action_id} in {path}")
    write_json(path, payload)


def check_pipeline_operator_handoff(artifact: Path, plan_path: Path) -> list[str]:
    problems: list[str] = []
    action_id = "submit_eagle3_pilot_pipeline"
    reports = artifact / "reports"
    sheet_path = reports / "eagle3_operator_sheet.json"
    sheet_validation_path = reports / "eagle3_operator_sheet_validation.json"
    execution_path = reports / "eagle3_operator_execution.json"
    followup_validation_path = reports / "eagle3_operator_followups_validation.json"
    packet_path = reports / "eagle3_operator_submit_packet.json"
    packet_validation_path = reports / "eagle3_operator_submit_packet_validation.json"

    try:
        run_checked(
            [
                sys.executable,
                str(SHEET_SCRIPT),
                "--artifact-root",
                str(artifact),
                "--plan-json",
                str(plan_path),
                "--json-out",
                str(sheet_path),
                "--markdown-out",
                str(sheet_path.with_suffix(".md")),
            ],
            "operator sheet creation",
        )
        run_checked(
            [
                sys.executable,
                str(SHEET_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--plan-json",
                str(plan_path),
                "--operator-sheet-json",
                str(sheet_path),
                "--json-out",
                str(sheet_validation_path),
                "--markdown-out",
                str(sheet_validation_path.with_suffix(".md")),
                "--expect-ready-action",
                action_id,
                "--fail-on-warn",
            ],
            "operator sheet validation",
        )
        run_checked(
            [
                sys.executable,
                str(EXECUTION_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--plan-json",
                str(plan_path),
                "--operator-sheet-json",
                str(sheet_path),
                "--json-out",
                str(execution_path),
                "--markdown-out",
                str(execution_path.with_suffix(".md")),
            ],
            "operator execution validation",
        )
        run_checked(
            [
                sys.executable,
                str(FOLLOWUP_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--plan-json",
                str(plan_path),
                "--operator-sheet-json",
                str(sheet_path),
                "--json-out",
                str(followup_validation_path),
                "--markdown-out",
                str(followup_validation_path.with_suffix(".md")),
                "--expect-action",
                action_id,
                "--fail-on-warn",
            ],
            "operator follow-up validation",
        )
        run_checked(
            [
                sys.executable,
                str(SUBMIT_PACKET_SCRIPT),
                "--artifact-root",
                str(artifact),
                "--operator-sheet-json",
                str(sheet_path),
                "--operator-sheet-validation-json",
                str(sheet_validation_path),
                "--operator-followup-validation-json",
                str(followup_validation_path),
                "--operator-execution-json",
                str(execution_path),
                "--json-out",
                str(packet_path),
                "--markdown-out",
                str(packet_path.with_suffix(".md")),
            ],
            "operator submit packet creation",
        )
        run_checked(
            [
                sys.executable,
                str(SUBMIT_PACKET_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--operator-submit-packet-json",
                str(packet_path),
                "--operator-sheet-json",
                str(sheet_path),
                "--operator-sheet-validation-json",
                str(sheet_validation_path),
                "--operator-followup-validation-json",
                str(followup_validation_path),
                "--operator-execution-json",
                str(execution_path),
                "--json-out",
                str(packet_validation_path),
                "--markdown-out",
                str(packet_validation_path.with_suffix(".md")),
                "--expect-ready-action",
                action_id,
                "--fail-on-warn",
            ],
            "operator submit packet validation",
        )

        direct_submit = base_commands(artifact)["pipeline_submit"]
        bad_sheet_path = reports / "eagle3_operator_sheet_direct_submit.json"
        shutil.copyfile(sheet_path, bad_sheet_path)
        rewrite_ready_action_command(bad_sheet_path, action_id, "raw_command", direct_submit)
        run_checked(
            [
                sys.executable,
                str(SHEET_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--plan-json",
                str(plan_path),
                "--operator-sheet-json",
                str(bad_sheet_path),
                "--json-out",
                str(reports / "eagle3_operator_sheet_direct_submit_validation.json"),
                "--markdown-out",
                str(reports / "eagle3_operator_sheet_direct_submit_validation.md"),
                "--expect-ready-action",
                action_id,
            ],
            "direct pipeline submit sheet validation",
            expect_success=False,
        )

        bad_packet_path = reports / "eagle3_operator_submit_packet_direct_submit.json"
        shutil.copyfile(packet_path, bad_packet_path)
        rewrite_ready_action_command(bad_packet_path, action_id, "planner_command", direct_submit)
        run_checked(
            [
                sys.executable,
                str(SUBMIT_PACKET_VALIDATOR),
                "--artifact-root",
                str(artifact),
                "--operator-submit-packet-json",
                str(bad_packet_path),
                "--operator-sheet-json",
                str(bad_sheet_path),
                "--operator-sheet-validation-json",
                str(sheet_validation_path),
                "--operator-followup-validation-json",
                str(followup_validation_path),
                "--operator-execution-json",
                str(execution_path),
                "--json-out",
                str(reports / "eagle3_operator_submit_packet_direct_submit_validation.json"),
                "--markdown-out",
                str(reports / "eagle3_operator_submit_packet_direct_submit_validation.md"),
                "--expect-ready-action",
                action_id,
            ],
            "direct pipeline submit packet validation",
            expect_success=False,
        )
    except Exception as exc:
        problems.append(str(exc))

    return problems


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Next-Action Transition Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| scenario | status | expected ready actions | observed ready actions |",
        "| --- | --- | --- | --- |",
    ]
    for item in payload["scenarios"]:
        lines.append(
            f"| {item['name']} | {item['status']} | `{', '.join(item['expected_ready_actions']) or '-'}` | "
            f"`{', '.join(item['observed_ready_actions']) or '-'}` |"
        )
    if payload.get("problems"):
        lines += ["", "## Problems", ""]
        lines.extend(f"- {problem}" for problem in payload["problems"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    scenarios = [
        ("source_vllm_missing", {"submit_vllm_source_build"}),
        ("frontier", {"submit_container_preflight", "submit_rollout_capture"}),
        ("rollout_submit_failed_prereq", {"probe_remote_hosts", "submit_container_preflight", "submit_vllm_source_build"}),
        ("megatron_probe_pending", {"poll_megatron_compat_probe"}),
        ("rollout_pending_fallback_ready", {"rollout_poll", "submit_rollout_fallback"}),
        ("full_rollout_ready", {"submit_full_swegym_rollout"}),
        ("rollout_ready", {"run_pipeline_submit_preflight"}),
        ("pipeline_ready", {"submit_eagle3_pilot_pipeline"}),
        ("pipeline_artifacts_missing", {"run_post_export_artifact_validations"}),
        ("pipeline_passed", {"submit_trained_draft_spec_tokens_sweep"}),
        ("sweep_passed", set()),
        ("remote_probe_unreachable", {"probe_remote_hosts"}),
        ("readiness_failed_frontier", {"submit_container_preflight", "submit_rollout_capture"}),
    ]
    results: list[dict[str, Any]] = []
    problems: list[str] = []

    temp_root = Path(tempfile.mkdtemp(prefix="eagle3_transition_validation_"))
    try:
        for name, expected_ready in scenarios:
            artifact = temp_root / name / "qwen3_235b_eagle3"
            materialize_reports(artifact, name)
            plan_path = artifact / "reports/eagle3_next_actions.json"
            plan = run_plan(artifact, plan_path)
            run_validator(plan_path, expected_ready)
            scenario_problems = check_action_shape(plan, expected_ready)
            if name == "pipeline_ready":
                scenario_problems.extend(check_pipeline_operator_handoff(artifact, plan_path))
            problems.extend(f"{name}: {problem}" for problem in scenario_problems)
            results.append(
                {
                    "name": name,
                    "status": "pass" if not scenario_problems else "fail",
                    "expected_ready_actions": sorted(expected_ready),
                    "observed_ready_actions": sorted(ready_action_ids(plan)),
                    "plan_status": plan.get("overall_status"),
                    "plan_json": str(plan_path),
                }
            )
    except Exception as exc:
        problems.append(str(exc))
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    overall = "pass" if not problems and all(item["status"] == "pass" for item in results) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "scenarios": results,
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
