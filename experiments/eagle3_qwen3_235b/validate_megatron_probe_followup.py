#!/usr/bin/env python3
"""Validate the Megatron-probe to rollout follow-up helper without Slurm submit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh"
REMOTE_WRAPPER = ROOT / "experiments/eagle3_qwen3_235b/run_eagle3_remote_cluster_pilot.sh"
HANDOFF_BUNDLE = ROOT / "experiments/eagle3_qwen3_235b/create_eagle3_handoff_bundle.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def pass_probe_payload() -> dict[str, Any]:
    return {
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
    }


def bad_probe_payload() -> dict[str, Any]:
    payload = pass_probe_payload()
    payload["api"]["grouped_linear_temporary_weight_attr"]["weight_is_weight0"] = False
    return payload


def prepare_case(root: Path, payload: dict[str, Any] | None) -> dict[str, Path]:
    artifact = root / "artifact"
    report_dir = artifact / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    job_file = root / "latest_megatron_compat_probe_job.txt"
    write_text(
        job_file,
        "\n".join(
            [
                "megatron_compat_probe_job=2867766",
                f"json={report_dir / 'megatron_compat_probe.json'}",
                f"markdown={report_dir / 'megatron_compat_probe.md'}",
                "",
            ]
        ),
    )
    report = report_dir / "megatron_compat_probe.json"
    if payload is not None:
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"artifact": artifact, "job_file": job_file, "report": report}


def fake_slurm_bin(root: Path) -> Path:
    bin_dir = root / "bin"
    write_text(
        bin_dir / "squeue",
        "#!/usr/bin/env bash\n"
        "echo \"2867766|COMPLETED|00:00|1|(None)|2026-05-23T01:00:00\"\n",
        executable=True,
    )
    write_text(
        bin_dir / "sacct",
        "#!/usr/bin/env bash\n"
        "echo \"2867766|COMPLETED|0:0|00:01:00|2026-05-23T01:00:00|2026-05-23T01:01:00\"\n",
        executable=True,
    )
    return bin_dir


def run_helper(root: Path, payload: dict[str, Any] | None, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    paths = prepare_case(root, payload)
    bin_dir = fake_slurm_bin(root)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "ARTIFACT_ROOT": str(paths["artifact"]),
            "SWE_REPO_ROOT": "/remote/SpecDec-RL",
            "JOB_FILE": str(paths["job_file"]),
            "REPORT_JOB_FILE": str(paths["artifact"] / "reports/megatron_compat_probe_job.env"),
            "JSON_OUT": str(paths["report"]),
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HELPER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def run_remote_wrapper_print_only() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PRINT_ONLY": "true",
            "SYNC_EXPERIMENTS": "true",
            "SYNC_PROBE_JOB_FILE": "true",
            "REMOTE_HOST": "oci-hsg-cs-001-vscode-02",
            "REMOTE_WORKDIR": "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap",
            "REMOTE_ARTIFACT_ROOT": "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3",
            "REMOTE_ENTRYPOINT": "experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh",
            "PROBE_JOB_ID": "2867766",
            "SUBMIT_ROLLOUT": "false",
        }
    )
    return subprocess.run(
        ["bash", str(REMOTE_WRAPPER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def run_remote_resume_wrapper_print_only() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PRINT_ONLY": "true",
            "SYNC_EXPERIMENTS": "true",
            "SYNC_PROBE_JOB_FILE": "true",
            "REMOTE_HOST": "oci-hsg-cs-001-vscode-02",
            "REMOTE_WORKDIR": "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap",
            "REMOTE_ARTIFACT_ROOT": "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3",
            "REMOTE_ENTRYPOINT": "experiments/eagle3_qwen3_235b/resume_eagle3_operator_state.sh",
            "PROBE_JOB_ID": "2867766",
            "EXECUTE_SAFE_ACTIONS": "true",
            "SAFE_ACTION_IDS": "probe_remote_hosts poll_megatron_compat_probe",
            "RUN_AFTER_SAFE_ACTIONS": "false",
            "RUN_FULL_REFRESH": "true",
        }
    )
    return subprocess.run(
        ["bash", str(REMOTE_WRAPPER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def validate_handoff_default_include(root: Path, checks: list[dict[str, Any]]) -> None:
    artifact = root / "artifact"
    report = artifact / "reports" / "megatron_probe_followup_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"overall_status": "pass", "checks": [{"name": "x", "status": "pass"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    default_reports = {
        "eagle3_next_actions.json": {"overall_status": "ready_for_operator_submit", "next_actions": []},
        "eagle3_operator_state_refresh.json": {"overall_status": "warn", "reports": {}},
        "eagle3_operator_state_refresh_validation.json": {"overall_status": "pass", "checks": []},
        "eagle3_goal_evidence.json": {"overall_status": "incomplete", "open_requirements": []},
        "eagle3_remote_access_diagnostics.json": {
            "overall_status": "blocked_local_dns",
            "diagnosis": "synthetic DNS block",
            "counts": {"hosts": 1, "resolved_hosts": 0, "reachable_hosts": 0},
            "gate_interpretation": {"remote_path_absence_proven": False},
        },
        "eagle3_training_path_manifest.json": {
            "overall_status": "defined",
            "path_defined": True,
            "open_gates": ["remote_hayate_reference_probe", "target_rollout_corpus", "runtime_container"],
            "ready_actions": ["probe_remote_hosts", "submit_vllm_source_build", "poll_megatron_compat_probe", "submit_container_preflight"],
            "gate_closure_contracts": [
                {
                    "id": "remote_hayate_reference_probe",
                    "closed": False,
                    "closure_evidence_missing": ["report:remote_host_probe"],
                    "candidate_next_action_ids": ["probe_remote_hosts"],
                },
                {
                    "id": "target_rollout_corpus",
                    "closed": False,
                    "closure_evidence_missing": ["report:rollout_state", "report:corpus_strategy"],
                    "candidate_next_action_ids": ["submit_rollout_capture", "rollout_poll", "rollout_materialize"],
                },
                {
                    "id": "runtime_container",
                    "closed": False,
                    "closure_evidence_missing": ["report:container_preflight", "report:vllm_source_build", "report:vllm_abi_probe", "report:megatron_compat"],
                    "candidate_next_action_ids": [
                        "submit_vllm_source_build",
                        "poll_megatron_compat_probe",
                        "submit_container_preflight",
                    ],
                },
            ],
            "reference_evidence": {
                "remote_reference_proven": False,
                "remote_probe": {"status": "unreachable", "reachable_hosts": []},
                "hayate_modelopt": {
                    "source": "synthetic_snapshot",
                    "remote_path_visible": False,
                },
                "hayate_specforge": {
                    "source": "synthetic_snapshot",
                    "remote_path_visible": False,
                },
            },
            "reference_decisions": {
                "modelopt_source": {"source_of_truth": "local_modelopt", "upstream_drift_status": "warn"},
                "specforge_qwen3_235b": {
                    "matched_fields": ["aux_layers", "hidden_size"],
                    "rejected_fields": [{"field": "rope_theta"}],
                },
            },
        },
        "eagle3_training_path_manifest_validation.json": {
            "overall_status": "pass",
            "scenarios": [{"name": "defined_before_external_gates", "status": "pass"}],
        },
        "qwen3_static_inputs.json": {
            "overall_status": "warn",
            "outputs": {"verifier_config_dir": str(artifact / "verifier_config")},
        },
        "qwen3_static_inputs_materialization_validation.json": {"overall_status": "pass", "checks": []},
    }
    for name, payload in default_reports.items():
        (report.parent / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (report.parent / "eagle3_training_path_manifest.md").write_text(
        "# Eagle3 Training Path Manifest\n\n"
        "| gate | status |\n"
        "| --- | --- |\n"
        "| remote_hayate_reference_probe | open |\n",
        encoding="utf-8",
    )
    (report.parent / "eagle3_training_path_manifest_validation.md").write_text(
        "# Eagle3 Training Path Manifest Validation\n\n"
        "Overall: **PASS**\n",
        encoding="utf-8",
    )
    static_files = {
        "verifier_config/config.json": {"model_type": "qwen3_moe"},
        "verifier_config/generation_config.json": {"eos_token_id": 151645},
        "verifier_config/tokenizer_config.json": {"chat_template": "{% generation %}x{% endgeneration %}"},
        "templates/qwen3_generation_template.jinja2": "{% generation %}{{ messages[-1]['content'] }}{% endgeneration %}\n",
        "templates/qwen3_generation_template.mask_validation.json": {"overall_status": "warn"},
        "architecture/eagle3_architecture.json": {"eagle_architecture_config": {"num_hidden_layers": 1}},
        "architecture/eagle3_architecture.env": "EAGLE_NUM_LAYERS=1\n",
        "architecture/eagle3_architecture.dotlist": "model.config.num_hidden_layers=1\n",
    }
    for rel, payload in static_files.items():
        path = artifact / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    patch_manifest = artifact / "patches" / "modelopt_eagle3_qwen3" / "manifest.json"
    patch_manifest.parent.mkdir(parents=True, exist_ok=True)
    patch_manifest.write_text(
        json.dumps({"overall_status": "pass", "patch_nonempty": True, "patch_sha256": "abc123"}, indent=2) + "\n",
        encoding="utf-8",
    )
    out_dir = root / "handoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_files = [
        out_dir / "next_action_plan_eagle3_next_actions.json",
        out_dir / "repo_readme_README.md",
    ]
    for path in stale_files:
        path.write_text("stale\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(HANDOFF_BUNDLE),
            "--out-dir",
            str(out_dir),
            "--artifact-root",
            str(artifact),
            "--sbatch-account",
            "coreai_dlalgo_nemorl",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    manifest_path = out_dir / "manifest.json"
    copied_report = out_dir / "megatron_probe_followup_validation.json"
    job_file = out_dir / "latest_megatron_compat_probe_job.txt"
    commands = out_dir / "commands.sh"
    runbook = out_dir / "RUNBOOK.md"
    gate_closure = out_dir / "GATE_CLOSURE.md"
    if result.returncode != 0 or not manifest_path.exists():
        add(
            checks,
            "handoff bundle includes probe follow-up report",
            "fail",
            "handoff bundle generation failed",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("inputs", {}).get("megatron_probe_followup_validation", {})
    patch_entry = manifest.get("inputs", {}).get("modelopt_patch", {})
    gate_closure_entry = manifest.get("inputs", {}).get("gate_closure", {})
    default_entries = {
        key: (manifest.get("inputs", {}).get(key) or {}).get("status")
        for key in [
            "next_action_plan",
            "operator_state_refresh",
            "operator_state_refresh_validation",
            "goal_evidence",
            "remote_access_diagnostics",
            "training_path_manifest",
            "training_path_manifest_markdown",
            "training_path_manifest_validation",
            "training_path_manifest_validation_markdown",
            "static_inputs",
            "static_inputs_materialization_validation",
        ]
    }
    materialized_entries = {
        key: manifest.get("inputs", {}).get(key) or {}
        for key in [
            "materialized_verifier_config",
            "materialized_generation_config",
            "materialized_tokenizer_config",
            "materialized_chat_template",
            "materialized_chat_template_mask_validation",
            "materialized_architecture_json",
            "materialized_architecture_env",
            "materialized_architecture_dotlist",
        ]
    }
    command_text = commands.read_text(encoding="utf-8") if commands.exists() else ""
    runbook_text = runbook.read_text(encoding="utf-8") if runbook.exists() else ""
    gate_closure_text = gate_closure.read_text(encoding="utf-8") if gate_closure.exists() else ""
    problems = []
    if entry.get("status") != "copied":
        problems.append("manifest entry is not copied")
    if not copied_report.exists():
        problems.append("copied validation report is missing")
    if not job_file.exists():
        problems.append("probe job file is missing from bundle")
    if gate_closure_entry.get("status") != "generated":
        problems.append("gate closure handoff entry is not generated")
    if not gate_closure.exists():
        problems.append("GATE_CLOSURE.md is missing from bundle")
    stale_remaining = [path.name for path in stale_files if path.exists()]
    if stale_remaining:
        problems.append(f"stale handoff files were not cleaned: {stale_remaining}")
    if "next_action_plan_eagle3_next_actions.json" not in (manifest.get("stale_files_removed") or []):
        problems.append("manifest did not record stale file cleanup")
    patch_bundle = patch_entry.get("bundle_path")
    if patch_entry.get("status") != "copied":
        problems.append("default ModelOpt patch manifest was not copied")
    elif Path(str(patch_bundle)).name == "manifest.json":
        problems.append("ModelOpt patch manifest collides with handoff manifest.json")
    elif not Path(str(patch_bundle)).exists():
        problems.append("renamed ModelOpt patch manifest is missing")
    for key, status in default_entries.items():
        if status != "copied":
            problems.append(f"default report {key} was not copied")
    summaries = manifest.get("summaries") if isinstance(manifest.get("summaries"), dict) else {}
    training_summary = summaries.get("training_path_manifest") if isinstance(summaries.get("training_path_manifest"), dict) else {}
    reference_decisions = (
        training_summary.get("reference_decisions")
        if isinstance(training_summary.get("reference_decisions"), dict)
        else {}
    )
    if reference_decisions.get("modelopt_source_of_truth") != "local_modelopt":
        problems.append("training path manifest summary did not preserve ModelOpt source-of-truth decision")
    if "rope_theta" not in (reference_decisions.get("specforge_rejected_fields") or []):
        problems.append("training path manifest summary did not preserve SpecForge rejected fields")
    for key, expected_heading in {
        "training_path_manifest_markdown": "# Eagle3 Training Path Manifest",
        "training_path_manifest_validation_markdown": "# Eagle3 Training Path Manifest Validation",
    }.items():
        summary = summaries.get(key) if isinstance(summaries.get(key), dict) else {}
        if summary.get("format") != "text":
            problems.append(f"{key} summary did not record format=text")
        if summary.get("first_heading") != expected_heading:
            problems.append(f"{key} summary did not record first heading")
        if "parse_error" in summary:
            problems.append(f"{key} summary leaked parse_error for text input")
        bundle_path = Path(str((manifest.get("inputs", {}).get(key) or {}).get("bundle_path") or ""))
        if not bundle_path.exists():
            problems.append(f"{key} bundle file is missing")
    for key, static_entry in materialized_entries.items():
        bundle_path = Path(str(static_entry.get("bundle_path") or ""))
        if static_entry.get("status") != "copied":
            problems.append(f"materialized static input {key} was not copied")
        elif not bundle_path.exists():
            problems.append(f"materialized static input {key} bundle file is missing")
        elif not bundle_path.name.startswith(f"{key}_"):
            problems.append(f"materialized static input {key} did not get a collision-safe bundle name")
    for snippet in [
        "LOCAL_ARTIFACT_ROOT=",
        'HANDOFF_DIR="${HANDOFF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
        'ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REMOTE_ARTIFACT_ROOT}"',
        'INPUT_DATA="${INPUT_DATA:-$ARTIFACT_ROOT/data/qwen3_235b_swe_rollout_conversations.jsonl}"',
        "0_restore_materialized_static_inputs",
        'copy_static_input materialized_verifier_config_config.json "$VERIFIER_CONFIG_DIR/config.json"',
        'copy_static_input materialized_chat_template_qwen3_generation_template.jinja2 "$CHAT_TEMPLATE"',
        'copy_static_input materialized_architecture_json_eagle3_architecture.json "$REFERENCE_ARCH"',
        "validate_megatron_probe_followup.py",
        "followup_megatron_probe_to_rollout.sh",
        "MEGATRON_PROBE_FOLLOWUP_VALIDATION_JSON",
        'REMOTE_WORKDIR="${REMOTE_WORKDIR:-/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap}"',
        'REMOTE_ARTIFACT_ROOT="${REMOTE_ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"',
        "--include-ssh-config-hosts",
        "diagnose_eagle3_remote_access.py",
        "eagle3_remote_access_diagnostics.json",
        '--remote-access-diagnostics-json "$ARTIFACT_ROOT/reports/eagle3_remote_access_diagnostics.json"',
        '--draft-inventory-json "$DRAFT_INVENTORY_JSON"',
        '--upstream-drift-json "$ARTIFACT_ROOT/reports/modelopt_upstream_drift.json"',
        "resume_eagle3_operator_state.sh",
        "1j_operator_safe_actions",
        "OPERATOR_SAFE_ACTIONS_PREFLIGHT_JSON",
        "REQUIRE_SLURM=false",
        "RUN_FULL_REFRESH=false",
        "RUN_FULL_REFRESH=true",
        "--action-ids",
        'SAFE_ACTION_IDS="probe_remote_hosts poll_megatron_compat_probe"',
        "validate_eagle3_operator_state_refresh.py",
        "OPERATOR_STATE_REFRESH_VALIDATION_JSON",
        "TRAINING_PATH_MANIFEST_MARKDOWN",
        "TRAINING_PATH_MANIFEST_VALIDATION_MARKDOWN",
        "submit_vllm_source_build",
        'operator_followups/${action_id}.json',
    ]:
        if snippet not in command_text:
            problems.append(f"commands.sh missing {snippet}")
    if "experiments/eagle3_qwen3_b/" in command_text:
        problems.append("commands.sh contains stale eagle3_qwen3_b typo")
    if "validate_megatron_probe_followup.py" not in runbook_text:
        problems.append("RUNBOOK missing follow-up validator step")
    if "validate_eagle3_operator_state_refresh.py" not in runbook_text:
        problems.append("RUNBOOK missing operator state refresh validator step")
    if "visible ModelOpt, Hayate/SpecForge, artifact-root, `git`, and `python3` evidence" not in runbook_text:
        problems.append("RUNBOOK missing remote path evidence gate")
    if "0_restore_materialized_static_inputs" not in runbook_text:
        problems.append("RUNBOOK missing materialized static input restore step")
    if "training_path_manifest_markdown" not in runbook_text:
        problems.append("RUNBOOK missing training path markdown handoff entry")
    if "## Gate Closure Checklist" not in runbook_text:
        problems.append("RUNBOOK missing readable gate closure checklist")
    for snippet in [
        "# Eagle3 Gate Closure Checklist",
        "target_rollout_corpus",
        "runtime_container",
        "submit_vllm_source_build",
        "submit_container_preflight",
        "poll_megatron_compat_probe",
    ]:
        if snippet not in gate_closure_text:
            problems.append(f"GATE_CLOSURE.md missing {snippet}")
    if problems:
        add(
            checks,
            "handoff bundle includes probe follow-up report",
            "fail",
            "handoff bundle did not include the probe follow-up contract",
            problems=problems,
            manifest_entry=entry,
            patch_entry=patch_entry,
            default_entries=default_entries,
            materialized_entries=materialized_entries,
            stale_remaining=stale_remaining,
            stale_files_removed=manifest.get("stale_files_removed"),
        )
        return
    add(
        checks,
        "handoff bundle includes probe follow-up report",
        "pass",
        "handoff bundle copied the probe follow-up report and records the remote path evidence contract",
        manifest_entry=entry,
        patch_entry=patch_entry,
        default_entries=default_entries,
        materialized_entries=materialized_entries,
        stale_files_removed=manifest.get("stale_files_removed"),
    )


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, "evidence": evidence})


def expect(
    checks: list[dict[str, Any]],
    name: str,
    result: subprocess.CompletedProcess[str],
    *,
    returncode: int,
    contains: list[str],
    excludes: list[str] | None = None,
) -> None:
    text = result.stdout + result.stderr
    missing = [item for item in contains if item not in text]
    present_forbidden = [item for item in (excludes or []) if item in text]
    if result.returncode != returncode or missing or present_forbidden:
        add(
            checks,
            name,
            "fail",
            "helper output did not match expected contract",
            expected_returncode=returncode,
            returncode=result.returncode,
            missing=missing,
            present_forbidden=present_forbidden,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return
    add(
        checks,
        name,
        "pass",
        "helper output matched expected contract",
        returncode=result.returncode,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Megatron Probe Follow-Up Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['name']} | {item['status'].upper()} | {item['detail']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="megatron_probe_followup_"))
    checks: list[dict[str, Any]] = []
    try:
        expect(
            checks,
            "missing report is no-submit",
            run_helper(temp_root / "missing", None),
            returncode=0,
            contains=["Megatron compatibility probe is not PASS yet: missing"],
            excludes=["submit_source_vllm_rollout_smoke.sh"],
        )
        expect(
            checks,
            "missing report can fail closed",
            run_helper(temp_root / "missing_fail_closed", None, {"FAIL_ON_NOT_READY": "true"}),
            returncode=1,
            contains=["Megatron compatibility probe is not PASS yet: missing"],
            excludes=["submit_source_vllm_rollout_smoke.sh"],
        )
        expect(
            checks,
            "pass report prints rollout command",
            run_helper(temp_root / "pass_print", pass_probe_payload()),
            returncode=0,
            contains=[
                "Megatron compatibility probe PASS",
                "submit_source_vllm_rollout_smoke.sh",
                "DRY_RUN=false",
                "automapgroupedweight",
            ],
            excludes=["Source-built vLLM site is not proven PASS yet"],
        )
        expect(
            checks,
            "heavy submit requires explicit allow flag",
            run_helper(temp_root / "heavy_guard", pass_probe_payload(), {"SUBMIT_ROLLOUT": "true"}),
            returncode=1,
            contains=["Refusing to submit rollout without ALLOW_HEAVY_GPU=true"],
            excludes=["Source-built vLLM site is not proven PASS yet"],
        )
        expect(
            checks,
            "bad grouped weight check fails closed",
            run_helper(temp_root / "bad_grouped", bad_probe_payload(), {"FAIL_ON_NOT_READY": "true"}),
            returncode=1,
            contains=[
                "Megatron compatibility probe is not PASS yet: not_pass",
                "grouped_linear_temporary_weight_attr",
            ],
            excludes=["submit_source_vllm_rollout_smoke.sh"],
        )
        expect(
            checks,
            "remote wrapper prints probe resume path",
            run_remote_wrapper_print_only(),
            returncode=0,
            contains=[
                "# sync experiments",
                "# sync Megatron compatibility probe job file",
                "latest_megatron_compat_probe_job.txt",
                "followup_megatron_probe_to_rollout.sh",
                "PROBE_JOB_ID=2867766",
                "SUBMIT_ROLLOUT=false",
            ],
            excludes=["ALLOW_HEAVY_GPU=true"],
        )
        expect(
            checks,
            "remote wrapper prints operator resume path",
            run_remote_resume_wrapper_print_only(),
            returncode=0,
            contains=[
                "# sync experiments",
                "# sync Megatron compatibility probe job file",
                "latest_megatron_compat_probe_job.txt",
                "resume_eagle3_operator_state.sh",
                "PROBE_JOB_ID=2867766",
                "EXECUTE_SAFE_ACTIONS=true",
                "SAFE_ACTION_IDS=probe_remote_hosts",
                "RUN_FULL_REFRESH=true",
            ],
            excludes=["ALLOW_HEAVY_GPU=true", "SUBMIT_ROLLOUT=true"],
        )
        validate_handoff_default_include(temp_root / "handoff_default", checks)
    finally:
        if args.keep_temp:
            print(f"Kept temp reports under: {temp_root}", file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    overall = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "helper": str(HELPER),
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
