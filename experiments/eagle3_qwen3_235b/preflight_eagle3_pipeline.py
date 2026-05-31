#!/usr/bin/env python3
"""Preflight checks for the Qwen3-235B Eagle3 draft pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"


REQUIRED_FILES = [
    "EAGLE3_DRAFT_MODEL_PLAYBOOK.md",
    "REMOTE_CLUSTER_STATUS.md",
    "REMOTE_EXECUTION_INPUTS.md",
    "audit_eagle3_readiness.py",
    "audit_eagle3_completion.py",
    "audit_eagle3_goal_evidence.py",
    "collect_eagle3_provenance.py",
    "check_modelopt_upstream_drift.py",
    "check_nemo_rl_eagle3_drift.py",
    "export_modelopt_eagle3_patch_bundle.py",
    "probe_cluster_environment.py",
    "probe_eagle3_remote_host.py",
    "probe_eagle3_slurm_capacity.py",
    "validate_eagle3_resource_profile_application.py",
    "discover_rollout_conversation_sources.py",
    "discover_eagle3_run_inputs.py",
    "create_eagle3_handoff_bundle.py",
    "derive_eagle3_architecture.py",
    "estimate_eagle3_training_scale.py",
    "plan_eagle3_next_actions.py",
    "create_eagle3_operator_sheet.py",
    "create_eagle3_operator_submit_packet.py",
    "refresh_eagle3_operator_state.py",
    "summarize_eagle3_operator_queue.py",
    "run_eagle3_next_action.py",
    "run_eagle3_slurm_followups.py",
    "validate_eagle3_operator_sheet.py",
    "validate_eagle3_operator_execution.py",
    "validate_eagle3_operator_followups.py",
    "validate_eagle3_operator_submit_packet.py",
    "validate_eagle3_next_action_plan.py",
    "validate_eagle3_next_action_transitions.py",
    "validate_eagle3_operator_queue_transitions.py",
    "validate_eagle3_preflight_robustness.py",
    "validate_megatron_probe_followup.py",
    "validate_eagle3_completion_contract.py",
    "validate_eagle3_export_artifacts.py",
    "generate_training_conversations_openai.py",
    "inventory_eagle3_draft_configs.py",
    "normalize_rl_rollouts_to_conversations.py",
    "nemo_rl_specdec_overlay.yaml",
    "nemo_rl_eagle3_online_draft_overlay.yaml",
    "run_baseline_smoke.sh",
    "run_rollout_capture_smoke.sh",
    "run_eagle3_cluster_pilot.sh",
    "run_eagle3_remote_cluster_pilot.sh",
    "followup_megatron_probe_to_rollout.sh",
    "run_static_specdec_smoke.sh",
    "submit_megatron_compat_probe.sh",
    "submit_source_vllm_rollout_smoke.sh",
    "submit_static_specdec_smoke_pair.sh",
    "submit_trained_draft_smoke_pair.sh",
    "submit_trained_draft_spec_tokens_sweep.sh",
    "submit_eagle3_container_preflight.sh",
    "analyze_container_preflight.py",
    "analyze_specforge_reference.py",
    "analyze_hayate_modelopt_workflow.py",
    "analyze_rollout_capture.py",
    "analyze_rollout_capture_job.py",
    "advance_rollout_capture_state.py",
    "analyze_corpus_strategy.py",
    "analyze_static_specdec_smoke_pair.py",
    "analyze_eagle3_pipeline.py",
    "analyze_spec_tokens_sweep.py",
    "bootstrap_eagle3_path.sh",
    "modelopt_qwen3_235b_dump_hidden_states.sh",
    "modelopt_qwen3_235b_offline_train.sh",
    "modelopt_qwen3_235b_online_train.sh",
    "modelopt_qwen3_235b_export_vllm.sh",
    "prepare_qwen3_chat_template.sh",
    "prepare_training_conversations.sh",
    "materialize_rollout_capture_corpus.sh",
    "prepare_qwen3_generation_template.py",
    "submit_eagle3_pipeline.sh",
    "slurm_megatron_compat_probe.sbatch",
    "slurm_preflight.sbatch",
    "slurm_dump_hidden_states.sbatch",
    "slurm_validate_hidden_states.sbatch",
    "slurm_offline_train.sbatch",
    "slurm_online_train.sbatch",
    "slurm_export_vllm.sbatch",
    "qwen3_235b_thinking_eagle3_architecture.json",
    "validate_hidden_state_dump.py",
    "validate_chat_template_loss_mask.py",
    "validate_training_conversations.py",
    "validate_modelopt_loss_mask_patch.py",
    "validate_modelopt_recipe_overrides.py",
    "validate_nemo_rl_specdec_integration.py",
    "validate_rollout_capture_config.py",
    "preflight_rollout_capture_submit.py",
    "preflight_eagle3_operator_ready_submit.py",
    "preflight_eagle3_pipeline_submit.py",
    "specdec_rl_rollout_role_logging.patch",
    "apply_specdec_rl_rollout_role_logging_patch.sh",
    "analyze_specdec_smoke.py",
]


CHECKS: list[dict[str, str]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get("ARTIFACT_ROOT", "/tmp/eagle3_container_preflight"),
        help="Artifact root passed to nested dry-run helpers that derive report paths.",
    )
    parser.add_argument("--input-data", default=os.environ.get("INPUT_DATA"))
    parser.add_argument("--hidden-states-dir", default=os.environ.get("HIDDEN_STATES_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--trained-ckpt", default=os.environ.get("TRAINED_CKPT"))
    parser.add_argument("--export-dir", default=os.environ.get("EXPORT_DIR"))
    parser.add_argument("--vllm-draft-dir", default=os.environ.get("VLLM_DRAFT_DIR"))
    parser.add_argument("--verifier-config-dir", default=os.environ.get("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER"))
    parser.add_argument("--chat-template", default=os.environ.get("CHAT_TEMPLATE"))
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-235B-A22B-Thinking-2507"))
    parser.add_argument("--modelopt-dir", default=os.environ.get("MODELOPT_DIR"))
    parser.add_argument(
        "--reference-arch",
        default=os.environ.get("REFERENCE_ARCH", str(EXP / "qwen3_235b_thinking_eagle3_architecture.json")),
    )
    parser.add_argument("--skip-existing-path-checks", action="store_true")
    parser.add_argument(
        "--require-modelopt-import",
        action="store_true",
        help="Require real ModelOpt recipe import/validation inside the current Python environment.",
    )
    parser.add_argument(
        "--require-chat-template-mask",
        action="store_true",
        help="Require Transformers to prove CHAT_TEMPLATE produces a positive assistant token mask.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional structured preflight report path.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown preflight report path.")
    return parser.parse_args()


def ok(msg: str) -> None:
    print(f"OK   {msg}")
    CHECKS.append({"status": "pass", "detail": msg})


def warn(msg: str) -> None:
    print(f"WARN {msg}")
    CHECKS.append({"status": "warn", "detail": msg})


def fail(msg: str, failures: list[str]) -> None:
    print(f"FAIL {msg}")
    failures.append(msg)
    CHECKS.append({"status": "fail", "detail": msg})


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    timeout = int(os.environ.get("EAGLE3_PREFLIGHT_COMMAND_TIMEOUT", "120"))
    try:
        return subprocess.run(
            cmd,
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=f"{output}\nTIMEOUT after {timeout}s: {' '.join(cmd)}",
            stderr=None,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            cmd,
            127,
            stdout=f"{exc.__class__.__name__}: {exc}",
            stderr=None,
        )


def check_file(path: Path, failures: list[str], executable: bool = False) -> None:
    if not path.exists():
        fail(f"missing {path}", failures)
        return
    if executable and not os.access(path, os.X_OK):
        fail(f"not executable {path}", failures)
        return
    try:
        label = path.resolve().relative_to(ROOT)
    except ValueError:
        label = path
    ok(f"found {label}")


def build_payload(args: argparse.Namespace, failures: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for check in CHECKS:
        status = check["status"]
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "fail" if failures else "pass",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cwd": str(ROOT),
        "modelopt_dir": str((Path(args.modelopt_dir) if args.modelopt_dir else ROOT / "Model-Optimizer").resolve()),
        "base_model": args.base_model,
        "container": args.container,
        "input_data": args.input_data,
        "verifier_config_dir": args.verifier_config_dir,
        "chat_template": args.chat_template,
        "reference_arch": args.reference_arch,
        "counts": counts,
        "failures": failures,
        "checks": CHECKS,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-235B Eagle3 Pipeline Preflight",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Slurm job id: `{payload.get('slurm_job_id') or '-'}`",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| modelopt dir | `{payload.get('modelopt_dir')}` |",
        f"| base model | `{payload.get('base_model')}` |",
        f"| container | `{payload.get('container') or '-'}` |",
        f"| input data | `{payload.get('input_data') or '-'}` |",
        f"| verifier config | `{payload.get('verifier_config_dir') or '-'}` |",
        f"| chat template | `{payload.get('chat_template') or '-'}` |",
        "",
        "## Checks",
        "",
        "| status | detail |",
        "| --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", "<br>")
        lines.append(f"| {check['status'].upper()} | {detail} |")
    if payload["failures"]:
        lines += ["", "## Failures", ""]
        lines.extend(f"- {item}" for item in payload["failures"])
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(args: argparse.Namespace, failures: list[str]) -> None:
    if not args.json_out and not args.markdown_out:
        return
    payload = build_payload(args, failures)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        check_file(EXP / rel, failures, executable=rel.endswith((".sh", ".py")))

    modelopt = (Path(args.modelopt_dir) if args.modelopt_dir else ROOT / "Model-Optimizer").resolve()
    check_file(modelopt / "examples/speculative_decoding/launch_train.sh", failures, executable=True)
    check_file(
        modelopt / "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
        failures,
    )

    arch_path = Path(args.reference_arch)
    try:
        arch = json.loads(arch_path.read_text())
        cfg = arch["eagle_architecture_config"]
        required_arch_keys = [
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "head_dim",
            "use_aux_hidden_state",
            "eagle_aux_hidden_state_layer_ids",
        ]
        for key in required_arch_keys:
            if key not in cfg:
                fail(f"architecture reference missing {key}: {arch_path}", failures)
        if cfg.get("num_hidden_layers") != 1:
            fail(f"architecture num_hidden_layers={cfg.get('num_hidden_layers')!r}, expected 1", failures)
        if cfg.get("use_aux_hidden_state") is not True:
            fail("architecture use_aux_hidden_state must be true for Eagle3", failures)
        aux_layers = cfg.get("eagle_aux_hidden_state_layer_ids")
        if not isinstance(aux_layers, list) or not aux_layers:
            fail("architecture eagle_aux_hidden_state_layer_ids must be a non-empty list", failures)
        if arch_path.name == "qwen3_235b_thinking_eagle3_architecture.json":
            expected = {
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "intermediate_size": 12288,
                "use_aux_hidden_state": True,
                "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
                "rope_theta": 5000000,
            }
            for key, value in expected.items():
                if cfg.get(key) != value:
                    fail(f"architecture {key}={cfg.get(key)!r}, expected {value!r}", failures)
        ok(f"architecture reference is valid: {arch_path}")
    except Exception as exc:
        fail(f"invalid architecture json: {exc}", failures)

    if not args.skip_existing_path_checks:
        if args.input_data:
            path = Path(args.input_data)
            if path.exists():
                ok(f"INPUT_DATA exists: {path}")
                result = run(
                    [
                        "python3",
                        "experiments/eagle3_qwen3_235b/validate_training_conversations.py",
                        str(path),
                        "--limit",
                        "200",
                        "--max-seq-len",
                        "16384",
                    ]
                )
                if result.returncode == 0:
                    ok("INPUT_DATA conversation schema sample passed")
                else:
                    fail(f"INPUT_DATA conversation schema sample failed\n{result.stdout}", failures)
            else:
                warn(f"INPUT_DATA does not exist from this host: {path}")
        else:
            warn("INPUT_DATA not provided")

        if args.verifier_config_dir:
            path = Path(args.verifier_config_dir) / "config.json"
            if path.exists():
                ok(f"verifier config exists: {path}")
            else:
                warn(f"verifier config not visible from this host: {path}")
        else:
            warn("VERIFIER_CONFIG_DIR not provided")

        if args.chat_template:
            path = Path(args.chat_template)
            if path.exists():
                text = path.read_text(errors="ignore")
                if "generation" in text and "endgeneration" in text:
                    ok("chat template has generation tags")
                else:
                    warn("chat template lacks generation/endgeneration tags; answer-only loss will fail")
                if args.require_chat_template_mask:
                    result = run(
                        [
                            "python3",
                            "experiments/eagle3_qwen3_235b/validate_chat_template_loss_mask.py",
                            "--model-or-tokenizer",
                            args.base_model,
                            "--chat-template",
                            str(path),
                        ]
                    )
                    if result.returncode == 0:
                        ok("chat template assistant mask validation passed")
                    else:
                        fail(f"chat template assistant mask validation failed\n{result.stdout}", failures)
            else:
                warn(f"CHAT_TEMPLATE not visible from this host: {path}")
                if args.require_chat_template_mask:
                    fail(f"CHAT_TEMPLATE required but not visible: {path}", failures)
        elif args.require_chat_template_mask:
            fail("CHAT_TEMPLATE is required for assistant mask validation", failures)

    default_output_dir = args.output_dir or "/tmp/modelopt_ckpt"
    trained_ckpt = args.trained_ckpt or default_output_dir
    if not trained_ckpt:
        fail("TRAINED_CKPT or OUTPUT_DIR must be set for export", failures)
    else:
        ok(f"export TRAINED_CKPT will be: {trained_ckpt}")

    tmp_chat_template = Path(tempfile.gettempdir()) / "eagle3_qwen3_generation_template.jinja2"
    if not tmp_chat_template.exists():
        tmp_chat_template.write_text(
            "{% for message in messages %}"
            "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] }}{% endgeneration %}"
            "{% else %}{{ message['content'] }}{% endif %}"
            "{% endfor %}\n",
            encoding="utf-8",
        )

    synthetic_rollout_data = Path(tempfile.gettempdir()) / "eagle3_qwen3_synthetic_swe_nemogym.jsonl"
    synthetic_rollout_record = {
        "responses_create_params": {
            "metadata": {
                "problem_statement": "synthetic Eagle3 rollout preflight problem",
                "instance_id": "synthetic-0",
                "base_commit": "0000000",
                "dataset_name": "synthetic_swe",
                "split": "train",
                "instance_dict": json.dumps(
                    {
                        "instance_id": "synthetic-0",
                        "repo": "synthetic/repo",
                        "base_commit": "0000000",
                    },
                    sort_keys=True,
                ),
            }
        }
    }
    synthetic_rollout_data.write_text(json.dumps(synthetic_rollout_record, sort_keys=True) + "\n", encoding="utf-8")

    dry_env = {
        "INPUT_DATA": args.input_data or "/tmp/conversations.jsonl",
        "HIDDEN_STATES_DIR": args.hidden_states_dir or "/tmp/hiddens",
        "OUTPUT_DIR": default_output_dir,
        "TRAINED_CKPT": trained_ckpt or "/tmp/modelopt_ckpt",
        "EXPORT_DIR": args.export_dir or "/tmp/exported",
        "VLLM_DRAFT_DIR": args.vllm_draft_dir or "/tmp/vllm",
        "VERIFIER_CONFIG_DIR": args.verifier_config_dir or "/tmp/verifier",
        "SBATCH_ACCOUNT": args.sbatch_account or "dummy",
        "CONTAINER": args.container or "",
        "MOUNTS": "/lustre:/lustre",
        "ARTIFACT_ROOT": str(args.artifact_root),
        "CHAT_TEMPLATE": args.chat_template or str(tmp_chat_template),
        "ANSWER_ONLY_LOSS": "true",
        "DRY_RUN": "true",
        "SUBMIT": "false",
        "MODE": "rollout",
        "INPUT_PATHS": args.input_data or "/tmp/rollouts.jsonl",
        "OUTPUT_DATA": args.input_data or "/tmp/conversations.jsonl",
        "OUTPUT_TEMPLATE": args.chat_template or "/tmp/qwen3_generation_template.jinja2",
        "ALLOW_MISSING_TRANSFORMERS": "true",
        "REFERENCE_ARCH": str(arch_path),
        "ARCH_ENV_FILE": os.environ.get("ARCH_ENV_FILE", ""),
        "JOB_FILE": "/tmp/eagle3_preflight_jobs.txt",
        "SMOKE_JOB_FILE": "/tmp/eagle3_preflight_smoke_jobs.txt",
        "SWEEP_JOB_FILE": "/tmp/eagle3_preflight_sweep_jobs.txt",
    }

    recipe_check_offline = [
        "python3",
        "experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py",
        "--wrapper",
        "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh",
        "--training-mode",
        "offline",
        "--modelopt-dir",
        str(modelopt),
        "--reference-arch",
        str(arch_path),
    ]
    recipe_check_online = [
        "python3",
        "experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py",
        "--wrapper",
        "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh",
        "--training-mode",
        "online",
        "--modelopt-dir",
        str(modelopt),
        "--reference-arch",
        str(arch_path),
    ]
    if args.require_modelopt_import:
        recipe_check_offline.append("--require-modelopt-import")
        recipe_check_online.append("--require-modelopt-import")

    checks = [
        recipe_check_offline,
        recipe_check_online,
        ["python3", "experiments/eagle3_qwen3_235b/audit_eagle3_completion.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/audit_eagle3_goal_evidence.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/collect_eagle3_provenance.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/check_modelopt_upstream_drift.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/check_nemo_rl_eagle3_drift.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/estimate_eagle3_training_scale.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/create_eagle3_operator_sheet.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/create_eagle3_operator_submit_packet.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/summarize_eagle3_operator_queue.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/run_eagle3_next_action.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/run_eagle3_slurm_followups.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_operator_sheet.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_operator_execution.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_operator_followups.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_operator_submit_packet.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_plan.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_next_action_transitions.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_operator_queue_transitions.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_preflight_robustness.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_megatron_probe_followup.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_completion_contract.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_export_artifacts.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/export_modelopt_eagle3_patch_bundle.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/probe_cluster_environment.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_resource_profile_application.py", "--help"],
        [
            "python3",
            "experiments/eagle3_qwen3_235b/probe_cluster_environment.py",
            "--artifact-root",
            "/tmp/eagle3_cluster_probe",
            "--modelopt-dir",
            str(modelopt),
        ],
        [
            "python3",
            "experiments/eagle3_qwen3_235b/check_modelopt_upstream_drift.py",
            "--no-probe-upstream",
            "--modelopt-dir",
            str(modelopt),
        ],
        ["python3", "experiments/eagle3_qwen3_235b/discover_eagle3_run_inputs.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/create_eagle3_handoff_bundle.py", "--help"],
        ["bash", "experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh", "--help"],
        ["bash", "experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh", "--help"],
        ["bash", "experiments/eagle3_qwen3_235b/run_baseline_smoke.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/run_static_specdec_smoke.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/submit_static_specdec_smoke_pair.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/submit_trained_draft_smoke_pair.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_container_preflight.sh", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_container_preflight.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_specforge_reference.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_hayate_modelopt_workflow.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_rollout_capture.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_corpus_strategy.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/preflight_rollout_capture_submit.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/preflight_eagle3_operator_ready_submit.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py", "--help"],
        ["python3", "experiments/eagle3_qwen3_235b/validate_eagle3_training_checkpoint.py", "--help"],
        [
            "python3",
            "experiments/eagle3_qwen3_235b/validate_modelopt_loss_mask_patch.py",
            "--modelopt-dir",
            str(modelopt),
        ],
        [
            "python3",
            "experiments/eagle3_qwen3_235b/validate_nemo_rl_specdec_integration.py",
            "--config",
            "grpo_qwen3_235b_swe.yaml",
            "--draft-model",
            "nvidia/Qwen3-235B-A22B-Eagle3",
        ],
        [
            "python3",
            "experiments/eagle3_qwen3_235b/validate_rollout_capture_config.py",
            "--config",
            "grpo_qwen3_235b_swe.yaml",
            "--artifact-root",
            "/tmp/eagle3_rollout_capture",
            "--train-data-path",
            str(synthetic_rollout_data),
            "--val-data-path",
            str(synthetic_rollout_data),
            "--chat-template",
            str(tmp_chat_template),
        ],
        ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/prepare_qwen3_chat_template.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/prepare_training_conversations.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_export_vllm.sh"],
        ["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"],
        ["env", "RUN_TRAINED_DRAFT_SMOKE=true", "bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"],
        ["env", "RUN_TRAINED_DRAFT_SWEEP=true", "bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py"],
        ["python3", "experiments/eagle3_qwen3_235b/analyze_spec_tokens_sweep.py", "--help"],
        [
            "env",
            "RUN_TEMPLATE_PREP=false",
            "RUN_DATA_PREP=false",
            "RUN_PREFLIGHT=false",
            "RUN_PIPELINE=false",
            "RUN_PIPELINE_SUBMIT_PREFLIGHT=false",
            "RUN_TRAINING_SCALE_PLAN=false",
            "RUN_NEXT_ACTION_PLAN=false",
            "RUN_AUDIT=false",
            "RUN_PROVENANCE=false",
            "bash",
            "experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh",
        ],
    ]
    for cmd in checks:
        result = run(cmd, dry_env)
        if result.returncode == 0:
            ok(f"dry-run passed: {' '.join(cmd)}")
        else:
            fail(f"dry-run failed: {' '.join(cmd)}\n{result.stdout}", failures)

    with tempfile.TemporaryDirectory(prefix="eagle3_handoff_preflight_") as tmp:
        tmp_path = Path(tmp)
        patch_dir = tmp_path / "patch_bundle"
        patch_cmd = [
            "python3",
            "experiments/eagle3_qwen3_235b/export_modelopt_eagle3_patch_bundle.py",
            "--modelopt-dir",
            str(modelopt),
            "--out-dir",
            str(patch_dir),
            "--compat-modelopt-dir",
            str(modelopt),
        ]
        result = run(patch_cmd, dry_env)
        if result.returncode != 0:
            fail(f"ModelOpt patch bundle export failed\n{result.stdout}", failures)
        else:
            try:
                patch_manifest = json.loads((patch_dir / "manifest.json").read_text(encoding="utf-8"))
                if patch_manifest.get("overall_status") == "pass" and patch_manifest.get("patch_nonempty"):
                    ok("ModelOpt patch bundle export passed")
                else:
                    fail(f"ModelOpt patch bundle manifest is not pass: {patch_manifest.get('overall_status')}", failures)
            except Exception as exc:
                fail(f"ModelOpt patch bundle manifest is invalid: {exc}", failures)

        handoff_dir = tmp_path / "handoff"
        handoff_cmd = [
            "python3",
            "experiments/eagle3_qwen3_235b/create_eagle3_handoff_bundle.py",
            "--out-dir",
            str(handoff_dir),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--sbatch-account",
            args.sbatch_account or "dummy",
            "--modelopt-patch-manifest",
            str(patch_dir / "manifest.json"),
        ]
        result = run(handoff_cmd, dry_env)
        if result.returncode != 0:
            fail(f"handoff bundle generation failed\n{result.stdout}", failures)
        else:
            commands_path = handoff_dir / "commands.sh"
            manifest_path = handoff_dir / "manifest.json"
            syntax = run(["bash", "-n", str(commands_path)], dry_env)
            if syntax.returncode != 0:
                fail(f"generated handoff commands.sh has invalid shell syntax\n{syntax.stdout}", failures)
            else:
                ok("generated handoff commands.sh syntax passed")
            try:
                json.loads(manifest_path.read_text(encoding="utf-8"))
                ok("generated handoff manifest is valid JSON")
            except Exception as exc:
                fail(f"generated handoff manifest is invalid: {exc}", failures)
            command_text = commands_path.read_text(encoding="utf-8", errors="replace")
            required_snippets = [
                "--verifier-config-dir \"$VERIFIER_CONFIG_DIR\"",
                "--sbatch-account \"$SBATCH_ACCOUNT\"",
                "--run-pilot true",
                "SBATCH_PARTITION=\"$SBATCH_PARTITION\"",
                "run_eagle3_cluster_pilot.sh",
                "modelopt_eagle3_qwen3",
                "create_eagle3_operator_sheet.py",
                "validate_eagle3_operator_sheet.py",
                "summarize_eagle3_operator_queue.py",
                "validate_eagle3_operator_queue_transitions.py",
                "validate_eagle3_completion_contract.py",
                "probe_eagle3_slurm_capacity.py",
                "validate_eagle3_resource_profile_application.py",
                "audit_eagle3_goal_evidence.py",
            ]
            missing = [snippet for snippet in required_snippets if snippet not in command_text]
            if missing:
                fail(f"generated handoff commands.sh missing expected snippets: {missing}", failures)
            else:
                ok("generated handoff commands.sh contains resume/pilot fields")

    if failures:
        write_outputs(args, failures)
        print("\nPreflight failed:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    write_outputs(args, failures)
    print("\nPreflight passed.")


if __name__ == "__main__":
    main()
