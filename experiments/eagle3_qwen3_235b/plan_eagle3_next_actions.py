#!/usr/bin/env python3
"""Plan the next Qwen3-235B Eagle3 actions from existing no-submit reports.

This script is deliberately read-only with respect to Slurm/GPU work. It reads
the current rollout/container/pipeline reports, chooses the next operator
actions, and writes one concise JSON/Markdown decision report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_HOSTS = [
    "oci-hsg-cs-001-vscode-02",
    "oci-hsg-cs-001-vscode-01",
    "oci-hsg-cs-001-vscode-03",
    "oci-hsg-cs-001-login-01.nvidia.com",
    "oci-hsg",
]
DEFAULT_REMOTE_WORKDIR = "/lustre/fsw/portfolios/coreai/users/sna/Nemo-RL_Qwen3_Roadmap"


def default_report_path(artifact_root: Path, filename: str) -> Path:
    return artifact_root / "reports" / filename


def parse_args() -> argparse.Namespace:
    artifact_default = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_default)
    parser.add_argument(
        "--container-preflight-json",
        type=Path,
        default=Path(os.environ["CONTAINER_PREFLIGHT_JSON"])
        if os.environ.get("CONTAINER_PREFLIGHT_JSON")
        else None,
    )
    parser.add_argument(
        "--rollout-submit-preflight-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_SUBMIT_PREFLIGHT_JSON"])
        if os.environ.get("ROLLOUT_SUBMIT_PREFLIGHT_JSON")
        else None,
    )
    parser.add_argument(
        "--rollout-state-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_STATE_ADVANCE_JSON"])
        if os.environ.get("ROLLOUT_STATE_ADVANCE_JSON")
        else None,
    )
    parser.add_argument(
        "--pipeline-submit-preflight-json",
        type=Path,
        default=Path(os.environ["PIPELINE_SUBMIT_PREFLIGHT_JSON"])
        if os.environ.get("PIPELINE_SUBMIT_PREFLIGHT_JSON")
        else None,
    )
    parser.add_argument(
        "--full-rollout-gate-json",
        type=Path,
        default=Path(os.environ["FULL_ROLLOUT_GATE_JSON"])
        if os.environ.get("FULL_ROLLOUT_GATE_JSON")
        else None,
    )
    parser.add_argument(
        "--megatron-compat-json",
        type=Path,
        default=Path(os.environ["MEGATRON_COMPAT_JSON"]) if os.environ.get("MEGATRON_COMPAT_JSON") else None,
    )
    parser.add_argument(
        "--megatron-compat-job-file",
        type=Path,
        default=Path(os.environ["MEGATRON_COMPAT_JOB_FILE"])
        if os.environ.get("MEGATRON_COMPAT_JOB_FILE")
        else None,
    )
    parser.add_argument(
        "--pipeline-analysis-json",
        type=Path,
        default=Path(os.environ["PIPELINE_ANALYSIS_JSON"]) if os.environ.get("PIPELINE_ANALYSIS_JSON") else None,
    )
    parser.add_argument(
        "--training-checkpoint-json",
        type=Path,
        default=Path(os.environ["TRAINING_CKPT_VALIDATION_JSON"])
        if os.environ.get("TRAINING_CKPT_VALIDATION_JSON")
        else None,
    )
    parser.add_argument(
        "--export-artifacts-json",
        type=Path,
        default=Path(os.environ["EXPORT_ARTIFACTS_JSON"]) if os.environ.get("EXPORT_ARTIFACTS_JSON") else None,
    )
    parser.add_argument(
        "--sweep-json",
        type=Path,
        default=Path(os.environ["SWEEP_JSON"]) if os.environ.get("SWEEP_JSON") else None,
    )
    parser.add_argument(
        "--training-scale-json",
        type=Path,
        default=Path(os.environ["TRAINING_SCALE_JSON"]) if os.environ.get("TRAINING_SCALE_JSON") else None,
    )
    parser.add_argument(
        "--modelopt-loss-mask-json",
        type=Path,
        default=Path(os.environ["MODELOPT_LOSS_MASK_JSON"]) if os.environ.get("MODELOPT_LOSS_MASK_JSON") else None,
    )
    parser.add_argument(
        "--nemo-rl-drift-json",
        type=Path,
        default=Path(os.environ["NEMO_RL_DRIFT_JSON"]) if os.environ.get("NEMO_RL_DRIFT_JSON") else None,
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=Path(os.environ["READINESS_JSON"]) if os.environ.get("READINESS_JSON") else None,
    )
    parser.add_argument(
        "--remote-host-probe-json",
        type=Path,
        default=Path(os.environ["REMOTE_HOST_PROBE_JSON"])
        if os.environ.get("REMOTE_HOST_PROBE_JSON")
        else None,
    )
    parser.add_argument(
        "--require-source-vllm-runtime",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("REQUIRE_SOURCE_VLLM_RUNTIME", "true").lower() not in {"false", "0", "no"},
        help="Require source-built vLLM + native ABI evidence before rollout capture.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def simple_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def simple_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def rollout_state_job_id(payload: dict[str, Any]) -> str:
    for keys in (
        ["job", "job_id"],
        ["job", "job", "job_id"],
        ["job", "slurm", "job_id"],
        ["job_id"],
    ):
        value = simple_nested(payload, keys)
        if value:
            return str(value)
    return ""


def select_rollout_state_report(artifact_root: Path) -> Path:
    reports = artifact_root / "reports"
    default = reports / "rollout_capture_state_advance.json"
    queue = simple_load_json(reports / "rollout_queue_wait_summary.json")
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
    active_ids: set[str] = set()
    for job in queue.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        snapshot = job.get("current_squeue") if isinstance(job.get("current_squeue"), dict) else {}
        if str(snapshot.get("state") or "").upper() not in active_states:
            continue
        job_id = str(job.get("job_id") or snapshot.get("job_id") or "")
        if job_id:
            active_ids.add(job_id)
    if not active_ids:
        return default

    candidates: list[tuple[float, Path]] = []
    for path in reports.glob("rollout_capture_*_state_advance.json"):
        if path.name in {"rollout_capture_state_advance.json", "rollout_capture_compact16n4g_state_advance.json"}:
            continue
        payload = simple_load_json(path)
        if rollout_state_job_id(payload) in active_ids:
            candidates.append((path.stat().st_mtime, path))
    if not candidates:
        return default
    return sorted(candidates)[-1][1]


def resolve_report_args(args: argparse.Namespace) -> dict[str, Path]:
    root = args.artifact_root
    return {
        "container_preflight": args.container_preflight_json
        or default_report_path(root, "container_preflight_analysis.json"),
        "rollout_submit_preflight": args.rollout_submit_preflight_json
        or default_report_path(root, "rollout_capture_submit_preflight.json"),
        "rollout_state": args.rollout_state_json or select_rollout_state_report(root),
        "pipeline_submit_preflight": args.pipeline_submit_preflight_json
        or default_report_path(root, "eagle3_pipeline_submit_preflight.json"),
        "megatron_compat": args.megatron_compat_json or default_report_path(root, "megatron_compat_probe.json"),
        "full_rollout_gate": args.full_rollout_gate_json
        or default_report_path(root, "full_swegym_after_smoke_gate.json"),
        "rollout_fallback_decision": default_report_path(root, "rollout_fallback_decision.json"),
        "pipeline_analysis": args.pipeline_analysis_json or default_report_path(root, "eagle3_pipeline_analysis.json"),
        "training_checkpoint": args.training_checkpoint_json or default_report_path(root, "eagle3_training_checkpoint.json"),
        "export_artifacts": args.export_artifacts_json or default_report_path(root, "eagle3_export_artifacts.json"),
        "trained_draft_sweep": args.sweep_json or default_report_path(root, "trained_draft_spec_tokens_sweep.json"),
        "training_scale": args.training_scale_json or default_report_path(root, "eagle3_training_scale.json"),
        "modelopt_loss_mask": args.modelopt_loss_mask_json or default_report_path(root, "modelopt_loss_mask_patch.json"),
        "nemo_rl_drift": args.nemo_rl_drift_json or default_report_path(root, "nemo_rl_eagle3_drift.json"),
        "readiness": args.readiness_json or default_report_path(root, "eagle3_readiness.json"),
        "remote_host_probe": args.remote_host_probe_json or default_report_path(root, "eagle3_remote_host_probe.json"),
        "vllm_source_build": default_report_path(root, "vllm_native_source_build.json"),
        "vllm_source_build_analysis": default_report_path(root, "vllm_source_build_job_analysis.json"),
        "vllm_abi_probe": default_report_path(root, "vllm_native_abi_probe.json"),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc), "_path": str(path)}
    if isinstance(payload, dict):
        payload["_path"] = str(path)
        return payload
    return {"_parse_error": "top-level JSON is not an object", "_path": str(path)}


def get_nested(payload: dict[str, Any] | None, keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def report_status(payload: dict[str, Any]) -> str:
    if payload.get("_missing"):
        return "missing"
    if payload.get("_parse_error"):
        return "invalid"
    return str(
        payload.get("overall_status")
        or get_nested(payload, ["decision", "overall_status"])
        or payload.get("status")
        or "unknown"
    )


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "pass", "ready"}
    return bool(value)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_env_assignments(command: str | None) -> dict[str, str]:
    if not command:
        return {}
    env: dict[str, str] = {}
    try:
        tokens = shlex.split(command)
    except ValueError:
        return env
    for token in tokens:
        if "=" not in token or token.startswith("-"):
            break
        key, value = token.split("=", 1)
        if not key.replace("_", "").isalnum() or not key[:1].isalpha():
            break
        env[key] = value
    return env


def first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


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


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def refresh_plan_command(artifact_root: Path) -> str:
    return shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/plan_eagle3_next_actions.py",
            "--artifact-root",
            str(artifact_root),
            "--json-out",
            str(artifact_root / "reports/eagle3_next_actions.json"),
            "--markdown-out",
            str(artifact_root / "reports/eagle3_next_actions.md"),
        ]
    )


def refresh_operator_state_command(artifact_root: Path) -> str:
    return shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py",
            "--artifact-root",
            str(artifact_root),
            "--json-out",
            str(artifact_root / "reports/eagle3_operator_state_refresh.json"),
            "--markdown-out",
            str(artifact_root / "reports/eagle3_operator_state_refresh.md"),
        ]
    )


def refresh_after_commands(artifact_root: Path) -> list[str]:
    return [refresh_plan_command(artifact_root), refresh_operator_state_command(artifact_root)]


def remote_host_probe_command(artifact_root: Path) -> str:
    return shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/probe_eagle3_remote_host.py",
            "--include-ssh-config-hosts",
            "--hosts",
            *DEFAULT_REMOTE_HOSTS,
            "--remote-workdir",
            os.environ.get("REMOTE_WORKDIR", DEFAULT_REMOTE_WORKDIR),
            "--artifact-root",
            str(artifact_root),
            "--json-out",
            str(artifact_root / "reports/eagle3_remote_host_probe.json"),
            "--markdown-out",
            str(artifact_root / "reports/eagle3_remote_host_probe.md"),
        ]
    )


def source_build_info(artifact_root: Path) -> dict[str, str]:
    candidates = [artifact_root / "latest_vllm_native_source_build_job.txt"]
    if artifact_root == DEFAULT_ARTIFACT_ROOT:
        candidates.append(ROOT / "latest_vllm_native_source_build_job.txt")
    for path in candidates:
        values = read_key_values(path)
        if values:
            values["_path"] = str(path)
            return values
    return {}


def megatron_compat_info(args: argparse.Namespace) -> dict[str, str]:
    candidates = []
    if args.megatron_compat_job_file:
        candidates.append(args.megatron_compat_job_file)
    candidates.extend(
        [
            ROOT / "latest_megatron_compat_probe_job.txt",
            args.artifact_root / "reports/megatron_compat_probe_job.env",
            args.artifact_root / "latest_megatron_compat_probe_job.txt",
        ]
    )
    for path in candidates:
        values = read_key_values(path)
        job_id = values.get("megatron_compat_probe_job")
        if job_id and "MEGATRON_COMPAT_PROBE_JOB_ID" not in job_id:
            values["_path"] = str(path)
            return values
    env_job = os.environ.get("MEGATRON_COMPAT_PROBE_JOB_ID")
    if env_job:
        return {"megatron_compat_probe_job": env_job, "_path": "MEGATRON_COMPAT_PROBE_JOB_ID"}
    return {}


def megatron_compat_submit_command(artifact_root: Path) -> str:
    return " ".join(
        [
            f"ARTIFACT_ROOT={shlex.quote(str(artifact_root))}",
            "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
            "SBATCH_PARTITION=batch",
            "SUBMIT=true",
            "bash",
            "experiments/eagle3_qwen3_235b/submit_megatron_compat_probe.sh",
        ]
    )


def megatron_compat_poll_command(artifact_root: Path, info: dict[str, str]) -> str:
    job_id = info.get("megatron_compat_probe_job") or "MEGATRON_COMPAT_PROBE_JOB_ID"
    return " ".join(
        [
            f"ARTIFACT_ROOT={shlex.quote(str(artifact_root))}",
            f"PROBE_JOB_ID={shlex.quote(job_id)}",
            "SUBMIT_ROLLOUT=false",
            "bash",
            "experiments/eagle3_qwen3_235b/followup_megatron_probe_to_rollout.sh",
        ]
    )


def source_vllm_site(artifact_root: Path, source_build: dict[str, Any], info: dict[str, str]) -> str:
    return str(
        source_build.get("output_site")
        or info.get("output_site")
        or artifact_root / "python_site/vllm_0_10_2_cu129_torch28nv_source_py312"
    )


def path_under_artifact_root(artifact_root: Path, value: str | None) -> bool:
    if not value:
        return False
    try:
        return Path(value).resolve().is_relative_to(artifact_root.resolve())
    except Exception:
        return str(value).startswith(str(artifact_root))


def abi_probe_site_result(abi_probe: dict[str, Any], source_site: str) -> dict[str, Any] | None:
    for item in abi_probe.get("results") or []:
        if isinstance(item, dict) and str(item.get("site") or "") == source_site:
            return item
    return None


def abi_probe_site_passed(abi_probe: dict[str, Any], source_site: str) -> bool:
    item = abi_probe_site_result(abi_probe, source_site)
    parsed = item.get("parsed") if isinstance((item or {}).get("parsed"), dict) else {}
    return bool(
        item
        and item.get("returncode") == 0
        and parsed.get("vllm_c_ok") is True
        and parsed.get("compilation_config_ok") is not False
    )


def abi_probe_site_failed(abi_probe: dict[str, Any], source_site: str) -> bool:
    item = abi_probe_site_result(abi_probe, source_site)
    return bool(item and not abi_probe_site_passed(abi_probe, source_site))


def source_build_poll_command(artifact_root: Path, info: dict[str, str]) -> str:
    job_id = info.get("vllm_native_source_build_job") or "VLLM_SOURCE_BUILD_JOB_ID"
    return (
        f"JOB={shlex.quote(job_id)} ARTIFACT_ROOT={shlex.quote(str(artifact_root))} "
        "bash -lc 'date; "
        "squeue -j \"$JOB\" -h -o \"%i|%T|%M|%D|%R|%S\" || true; "
        "sacct -j \"$JOB\" --format=JobID,JobName,State,Elapsed,Start,End,ExitCode -P -n 2>/dev/null | tail -40 || true; "
        "test -e \"$ARTIFACT_ROOT/reports/vllm_native_source_build.md\" "
        "&& cat \"$ARTIFACT_ROOT/reports/vllm_native_source_build.md\" "
        "|| echo source_report_missing'"
    )


def source_build_submit_command(artifact_root: Path) -> str:
    return " ".join(
        [
            f"ARTIFACT_ROOT={shlex.quote(str(artifact_root))}",
            "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
            "SBATCH_PARTITION=batch",
            "SUBMIT=true",
            "bash",
            "experiments/eagle3_qwen3_235b/submit_vllm_native_source_build.sh",
        ]
    )


def source_build_after_commands(artifact_root: Path) -> list[str]:
    analyze_cmd = shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/analyze_vllm_source_build_job.py",
            "--artifact-root",
            str(artifact_root),
            "--job-file",
            "latest_vllm_native_source_build_job.txt",
            "--logs-dir",
            "logs",
            "--json-out",
            str(artifact_root / "reports/vllm_source_build_job_analysis.json"),
            "--markdown-out",
            str(artifact_root / "reports/vllm_source_build_job_analysis.md"),
        ]
    )
    return [analyze_cmd, *refresh_after_commands(artifact_root)]


def source_abi_probe_command(artifact_root: Path, source_site: str) -> str:
    return " ".join(
        [
            f"ARTIFACT_ROOT={shlex.quote(str(artifact_root))}",
            "SBATCH_ACCOUNT=coreai_dlalgo_nemorl",
            "SBATCH_PARTITION=batch",
            f"VLLM_SITE_CANDIDATES={shlex.quote(source_site)}",
            "SUBMIT=true",
            "bash",
            "experiments/eagle3_qwen3_235b/submit_vllm_native_abi_probe.sh",
        ]
    )


def source_build_analysis_sentence(analysis: dict[str, Any]) -> str:
    if analysis.get("_missing") or analysis.get("_parse_error"):
        return ""
    status = str(analysis.get("overall_status") or "unknown").upper()
    detail = str(analysis.get("detail") or "").strip()
    slurm = analysis.get("slurm") if isinstance(analysis.get("slurm"), dict) else {}
    tmp_site = get_nested(analysis, ["paths", "tmp_site"], {})
    sstat = analysis.get("sstat") if isinstance(analysis.get("sstat"), dict) else {}
    tmp_size = tmp_site.get("du") if isinstance(tmp_site, dict) else None
    parts = [f"latest source-build analysis is `{status}`"]
    if detail:
        parts.append(detail)
    if slurm.get("elapsed"):
        parts.append(f"elapsed `{slurm.get('elapsed')}`")
    if slurm.get("time_limit"):
        parts.append(f"time limit `{slurm.get('time_limit')}`")
    if tmp_size:
        parts.append(f"tmp site `{tmp_size}`")
    if sstat.get("available"):
        if sstat.get("AveCPU"):
            parts.append(f"AveCPU `{sstat.get('AveCPU')}`")
        if sstat.get("MaxRSS"):
            parts.append(f"MaxRSS `{sstat.get('MaxRSS')}`")
    return "; ".join(parts) + "."


def ensure_rollout_state_outputs(command: str | None, artifact_root: Path, report_path: str | None = None) -> str | None:
    if not command or "advance_rollout_capture_state.py" not in command:
        return command
    suffix: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    json_out = Path(report_path) if report_path else artifact_root / "reports/rollout_capture_state_advance.json"
    if json_out.suffix != ".json":
        json_out = artifact_root / "reports/rollout_capture_state_advance.json"
    markdown_out = json_out.with_suffix(".md")
    if "--markdown-out" not in tokens:
        suffix.extend(["--markdown-out", str(markdown_out)])
    if "--json-out" not in tokens:
        suffix.extend(["--json-out", str(json_out)])
    if not suffix:
        return command
    return f"{command} {shell_join(suffix)}"


def container_after_commands(command: str | None, artifact_root: Path) -> list[str]:
    env = parse_env_assignments(command)
    root = Path(env.get("ARTIFACT_ROOT", str(artifact_root)))
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/analyze_container_preflight.py",
        "--job-file",
        "latest_eagle3_container_preflight_job.txt",
        "--logs-dir",
        "logs",
        "--cluster-probe-json",
        str(root / "reports/container_preflight_cluster_probe.json"),
        "--pipeline-preflight-json",
        str(root / "reports/container_preflight_pipeline.json"),
        "--pipeline-preflight-markdown",
        str(root / "reports/container_preflight_pipeline.md"),
        "--artifact-root",
        str(root),
    ]
    if env.get("CONTAINER"):
        cmd += ["--container", env["CONTAINER"]]
    if env.get("MODELOPT_DIR"):
        cmd += ["--modelopt-dir", env["MODELOPT_DIR"]]
    if env.get("VERIFIER_CONFIG_DIR"):
        cmd += ["--verifier-config-dir", env["VERIFIER_CONFIG_DIR"]]
    if env.get("INPUT_DATA"):
        cmd += ["--input-data", env["INPUT_DATA"]]
    if env.get("CHAT_TEMPLATE"):
        cmd += ["--chat-template", env["CHAT_TEMPLATE"]]
    if env.get("MOUNTS"):
        cmd += ["--mounts", env["MOUNTS"]]
    if env.get("SBATCH_ACCOUNT"):
        cmd += ["--sbatch-account", env["SBATCH_ACCOUNT"]]
    if env.get("SBATCH_PARTITION"):
        cmd += ["--sbatch-partition", env["SBATCH_PARTITION"]]
    cmd += [
        "--markdown-out",
        str(root / "reports/container_preflight_analysis.md"),
        "--json-out",
        str(root / "reports/container_preflight_analysis.json"),
    ]
    return [shell_join(cmd), *refresh_after_commands(root)]


def rollout_after_commands(command: str | None, rollout_submit: dict[str, Any], artifact_root: Path) -> list[str]:
    env = parse_env_assignments(command)
    root = Path(env.get("ARTIFACT_ROOT", str(artifact_root)))
    repo = env.get("SWE_REPO_ROOT") or env.get("REPO_ROOT")
    rollout_log_dir = env.get("ROLLOUT_LOG_DIR", str(root / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke"))
    output = env.get("OUTPUT_CONVERSATIONS", str(root / "data/qwen3_235b_swe_rollout_conversations.jsonl"))
    commands: list[str] = []
    analyze_job = get_nested(rollout_submit, ["commands", "analyze_job"])
    if analyze_job:
        commands.append(str(analyze_job))
    state_cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/advance_rollout_capture_state.py",
        "--artifact-root",
        str(root),
        "--rollout-log-dir",
        rollout_log_dir,
        "--output-data",
        output,
        "--markdown-out",
        str(root / "reports/rollout_capture_state_advance.md"),
        "--json-out",
        str(root / "reports/rollout_capture_state_advance.json"),
    ]
    if repo:
        state_cmd += ["--repo-root", repo]
    commands.append(shell_join(state_cmd))
    commands.extend(refresh_after_commands(root))
    return commands


def pipeline_after_commands(command: str | None, pipeline: dict[str, Any], artifact_root: Path) -> list[str]:
    env = parse_env_assignments(command)
    root = Path(env.get("ARTIFACT_ROOT", str(artifact_root)))
    analyze_pipeline = get_nested(pipeline, ["commands", "analyze_pipeline"])
    commands = [str(analyze_pipeline)] if analyze_pipeline else []
    commands.extend(refresh_after_commands(root))
    return commands


def post_export_artifact_validation_command(args: argparse.Namespace, pipeline_analysis: dict[str, Any]) -> str:
    root = args.artifact_root
    env: dict[str, str] = {}
    for command in [
        get_nested(pipeline_analysis, ["next_action", "resume_command"]),
        get_nested(pipeline_analysis, ["commands", "pilot_submit"]),
    ]:
        env.update(parse_env_assignments(str(command) if command else None))

    modelopt_dir = first_nonempty(env.get("MODELOPT_DIR"), os.environ.get("MODELOPT_DIR"), ROOT / "Model-Optimizer")
    verifier_config = first_nonempty(env.get("VERIFIER_CONFIG_DIR"), os.environ.get("VERIFIER_CONFIG_DIR"), root / "verifier_config")
    reference_arch = first_nonempty(env.get("REFERENCE_ARCH"), os.environ.get("REFERENCE_ARCH"), root / "architecture/eagle3_architecture.json")
    output_dir = first_nonempty(env.get("OUTPUT_DIR"), os.environ.get("OUTPUT_DIR"), root / "modelopt_ckpt")
    export_dir = first_nonempty(env.get("EXPORT_DIR"), os.environ.get("EXPORT_DIR"), root / "exported_hf")
    vllm_draft_dir = first_nonempty(env.get("VLLM_DRAFT_DIR"), os.environ.get("VLLM_DRAFT_DIR"), root / "vllm_draft")
    training_json = root / "reports/eagle3_training_checkpoint.json"
    training_md = root / "reports/eagle3_training_checkpoint.md"
    export_compare_json = Path(str(export_dir)) / "config_compare.json"
    vllm_compare_json = Path(str(vllm_draft_dir)) / "config_compare.json"
    export_artifacts_json = root / "reports/eagle3_export_artifacts.json"
    export_artifacts_md = root / "reports/eagle3_export_artifacts.md"

    commands = [
        shell_join(
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_training_checkpoint.py",
                "--checkpoint-dir",
                str(output_dir),
                "--modelopt-dir",
                str(modelopt_dir),
                "--reference-arch",
                str(reference_arch),
                "--json-out",
                str(training_json),
                "--markdown-out",
                str(training_md),
                "--require-modelopt-state-load",
                "--fail-on-error",
            ]
        ),
        shell_join(
            [
                "python3",
                "experiments/eagle3_qwen3_235b/compare_eagle3_configs.py",
                "--draft-config",
                str(export_dir),
                "--verifier-config",
                str(verifier_config),
                "--reference-arch",
                str(reference_arch),
                "--json-out",
                str(export_compare_json),
            ]
        ),
        shell_join(
            [
                "python3",
                "experiments/eagle3_qwen3_235b/compare_eagle3_configs.py",
                "--draft-config",
                str(vllm_draft_dir),
                "--verifier-config",
                str(verifier_config),
                "--reference-arch",
                str(reference_arch),
                "--json-out",
                str(vllm_compare_json),
            ]
        ),
        shell_join(
            [
                "python3",
                "experiments/eagle3_qwen3_235b/validate_eagle3_export_artifacts.py",
                "--export-dir",
                str(export_dir),
                "--vllm-draft-dir",
                str(vllm_draft_dir),
                "--verifier-config-dir",
                str(verifier_config),
                "--reference-arch",
                str(reference_arch),
                "--export-config-compare-json",
                str(export_compare_json),
                "--vllm-config-compare-json",
                str(vllm_compare_json),
                "--json-out",
                str(export_artifacts_json),
                "--markdown-out",
                str(export_artifacts_md),
                "--fail-on-error",
            ]
        ),
    ]
    return " && \\\n".join(commands)


def completion_audit_command(artifact_root: Path) -> str:
    return shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/audit_eagle3_completion.py",
            "--artifact-root",
            str(artifact_root),
            "--markdown-out",
            str(artifact_root / "reports/eagle3_completion_audit.md"),
            "--json-out",
            str(artifact_root / "reports/eagle3_completion_audit.json"),
        ]
    )


def trained_draft_sweep_command(args: argparse.Namespace, pipeline: dict[str, Any]) -> str:
    root = args.artifact_root
    env: dict[str, str] = {}
    for command in [
        get_nested(pipeline, ["next_action", "resume_command"]),
        get_nested(pipeline, ["commands", "pilot_submit"]),
    ]:
        env.update(parse_env_assignments(str(command) if command else None))
    swe_repo = first_nonempty(
        env.get("SWE_REPO_ROOT"),
        env.get("REPO_ROOT"),
        os.environ.get("SWE_REPO_ROOT"),
        os.environ.get("REPO_ROOT"),
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
    )
    command_env = {
        "SUBMIT": "true",
        "ARTIFACT_ROOT": str(root),
        "SWE_REPO_ROOT": str(swe_repo),
        "REPO_ROOT": str(swe_repo),
        "CONFIG_FILE": str(
            first_nonempty(env.get("CONFIG_FILE"), os.environ.get("CONFIG_FILE"), ROOT / "grpo_qwen3_235b_swe.yaml")
        ),
        "ENV_FILE": str(first_nonempty(env.get("ENV_FILE"), os.environ.get("ENV_FILE"), ROOT / "env.sh")),
        "CHAT_TEMPLATE": str(
            first_nonempty(
                env.get("CHAT_TEMPLATE"),
                os.environ.get("CHAT_TEMPLATE"),
                root / "templates/qwen3_generation_template.jinja2",
            )
        ),
        "VLLM_DRAFT_DIR": str(root / "vllm_draft"),
        "JOB_FILE": "latest_trained_draft_spec_tokens_sweep_jobs.txt",
        "MAX_NUM_STEPS": env.get("SWEEP_MAX_NUM_STEPS", os.environ.get("SWEEP_MAX_NUM_STEPS", "2")),
        "SPEC_TOKENS_LIST": env.get("SWEEP_SPEC_TOKENS_LIST", os.environ.get("SWEEP_SPEC_TOKENS_LIST", "2 3 4")),
        "EAGLE3_DRAFT_TP": env.get("EAGLE3_DRAFT_TP", os.environ.get("EAGLE3_DRAFT_TP", "1")),
    }
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in command_env.items()) + (
        " bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh"
    )


def trained_draft_sweep_after_commands(artifact_root: Path) -> list[str]:
    sweep_json = artifact_root / "reports/trained_draft_spec_tokens_sweep.json"
    sweep_md = artifact_root / "reports/trained_draft_spec_tokens_sweep.md"
    analyze_cmd = shell_join(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/analyze_spec_tokens_sweep.py",
            "--job-file",
            "latest_trained_draft_spec_tokens_sweep_jobs.txt",
            "--repo-root",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
            "--markdown-out",
            str(sweep_md),
            "--json-out",
            str(sweep_json),
            "--fail-on-missing-spec-metrics",
        ]
    )
    return [analyze_cmd, completion_audit_command(artifact_root), *refresh_after_commands(artifact_root)]


def pipeline_submit_preflight_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    container: dict[str, Any],
    rollout_submit: dict[str, Any],
    rollout_state: dict[str, Any],
    pipeline: dict[str, Any],
) -> str:
    root = args.artifact_root
    env: dict[str, str] = {}
    for command in [
        get_nested(container, ["next_action", "submit_command"]),
        get_nested(rollout_submit, ["commands", "submit"]),
        get_nested(rollout_state, ["commands", "pipeline_dry_run"]),
        get_nested(pipeline, ["commands", "dry_run"]),
        get_nested(pipeline, ["commands", "pilot_submit"]),
    ]:
        env.update(parse_env_assignments(str(command) if command else None))

    input_data = first_nonempty(
        rollout_state.get("output_data"),
        env.get("INPUT_DATA"),
        env.get("ROLLOUT_CONVERSATIONS"),
        env.get("OUTPUT_DATA"),
        root / "data/qwen3_235b_swe_rollout_conversations.jsonl",
    )
    modelopt_dir = first_nonempty(env.get("MODELOPT_DIR"), os.environ.get("MODELOPT_DIR"), ROOT / "Model-Optimizer")
    sbatch_account = first_nonempty(env.get("SBATCH_ACCOUNT"), os.environ.get("SBATCH_ACCOUNT"), "coreai_dlalgo_nemorl")
    sbatch_partition = first_nonempty(env.get("SBATCH_PARTITION"), os.environ.get("SBATCH_PARTITION"), "batch")
    container_path = first_nonempty(container.get("container"), env.get("CONTAINER"), os.environ.get("CONTAINER"))
    mounts = first_nonempty(env.get("MOUNTS"), os.environ.get("MOUNTS"))
    base_model = first_nonempty(env.get("BASE_MODEL"), os.environ.get("BASE_MODEL"), "Qwen/Qwen3-235B-A22B-Thinking-2507")
    resource_profile_env = root / "reports/eagle3_resource_profile.env"
    resource_env = read_export_env(resource_profile_env)
    dump_gpus_per_node = first_nonempty(
        env.get("DUMP_GPUS_PER_NODE"),
        resource_env.get("DUMP_GPUS_PER_NODE"),
        os.environ.get("DUMP_GPUS_PER_NODE"),
        "8",
    )
    train_gpus_per_node = first_nonempty(
        env.get("TRAIN_GPUS_PER_NODE"),
        resource_env.get("TRAIN_GPUS_PER_NODE"),
        os.environ.get("TRAIN_GPUS_PER_NODE"),
        "8",
    )
    export_gpus_per_node = first_nonempty(
        env.get("EXPORT_GPUS_PER_NODE"),
        resource_env.get("EXPORT_GPUS_PER_NODE"),
        os.environ.get("EXPORT_GPUS_PER_NODE"),
        "1",
    )
    tp = first_nonempty(env.get("TP"), resource_env.get("TP"), os.environ.get("TP"), "8")

    command = [
        "python3",
        "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py",
        "--artifact-root",
        str(root),
        "--input-data",
        str(input_data),
        "--hidden-states-dir",
        str(root / "hidden_states"),
        "--output-dir",
        str(root / "modelopt_ckpt"),
        "--trained-ckpt",
        str(root / "modelopt_ckpt"),
        "--export-dir",
        str(root / "exported_hf"),
        "--vllm-draft-dir",
        str(root / "vllm_draft"),
        "--verifier-config-dir",
        str(root / "verifier_config"),
        "--chat-template",
        str(root / "templates/qwen3_generation_template.jinja2"),
        "--modelopt-dir",
        str(modelopt_dir),
        "--reference-arch",
        str(root / "architecture/eagle3_architecture.json"),
        "--arch-env-file",
        str(root / "architecture/eagle3_architecture.env"),
        "--container-preflight-json",
        str(paths["container_preflight"]),
        "--corpus-strategy-json",
        str(root / "reports/corpus_strategy.json"),
        "--rollout-state-json",
        str(paths["rollout_state"]),
        "--base-model",
        str(base_model),
        "--sbatch-account",
        str(sbatch_account),
        "--sbatch-partition",
        str(sbatch_partition),
        "--run-pilot",
        "true",
        "--dump-gpus-per-node",
        str(dump_gpus_per_node),
        "--train-gpus-per-node",
        str(train_gpus_per_node),
        "--export-gpus-per-node",
        str(export_gpus_per_node),
        "--tp",
        str(tp),
        "--slurm-capacity-env",
        str(resource_profile_env),
        "--target-context",
        "swe_rl",
        "--markdown-out",
        str(root / "reports/eagle3_pipeline_submit_preflight.md"),
        "--json-out",
        str(root / "reports/eagle3_pipeline_submit_preflight.json"),
    ]
    if container_path:
        command.extend(["--container", str(container_path)])
    if mounts:
        command.extend(["--mounts", str(mounts)])
    return shell_join(command)


def full_rollout_gate_execute_command(
    args: argparse.Namespace,
    full_gate: dict[str, Any],
) -> str:
    root = args.artifact_root
    command = [
        "python3",
        "experiments/eagle3_qwen3_235b/submit_full_rollout_after_smoke_if_ready.py",
        "--artifact-root",
        str(root),
    ]
    smoke_state_json = full_gate.get("smoke_state_json")
    if smoke_state_json:
        command.extend(["--smoke-state-json", str(smoke_state_json)])
    full_preflight_json = full_gate.get("full_preflight_json") or full_gate.get("full_preflight_path")
    if full_preflight_json:
        command.extend(["--full-preflight-json", str(full_preflight_json)])
    command.extend(
        [
            "--execute",
            "--allow-heavy-gpu",
            "--start-watcher",
            "--allow-background",
            "--json-out",
            str(root / "reports/full_swegym_after_smoke_gate.json"),
            "--markdown-out",
            str(root / "reports/full_swegym_after_smoke_gate.md"),
        ]
    )
    return shell_join(command)


def add_action(
    actions: list[dict[str, Any]],
    action_id: str,
    title: str,
    status: str,
    reason: str,
    *,
    command: str | None = None,
    report: str | None = None,
    priority: int | None = None,
    stage: str = "gate",
    submits_slurm: bool = False,
    heavy_gpu: bool = False,
    after_commands: list[str] | None = None,
) -> None:
    actions.append(
        {
            "id": action_id,
            "priority": priority or len(actions) + 1,
            "stage": stage,
            "title": title,
            "status": status,
            "reason": reason,
            "report": report,
            "command": command,
            "after_commands": after_commands or [],
            "submits_slurm": submits_slurm,
            "heavy_gpu": heavy_gpu,
        }
    )


def add_blocker(
    blockers: list[dict[str, Any]],
    blocker_id: str,
    severity: str,
    summary: str,
    *,
    report: str | None = None,
) -> None:
    blockers.append({"id": blocker_id, "severity": severity, "summary": summary, "report": report})


def training_summary(training: dict[str, Any]) -> dict[str, Any]:
    if training.get("_missing") or training.get("_parse_error"):
        return {
            "status": report_status(training),
            "defaults": {
                "effective_global_batch": 8,
                "epochs": 1,
                "max_seq_len": 16384,
            },
            "stages": [
                {
                    "name": "pilot",
                    "examples": 8,
                    "max_steps": 20,
                    "nominal_epoch_steps": 1,
                    "purpose": "pipeline wiring only",
                },
                {
                    "name": "swegym_first_calibration",
                    "examples": 2438,
                    "max_steps": 1000,
                    "nominal_epoch_steps": 305,
                    "purpose": "first acceptance/speed signal on the materialized SWE-Gym train split",
                },
                {
                    "name": "target_domain_calibration",
                    "examples": 50000,
                    "max_steps": 2000,
                    "nominal_epoch_steps": 6250,
                    "purpose": "larger target-domain SWE/RL calibration if the 2.4k run improves acceptance",
                },
                {
                    "name": "production_candidate",
                    "examples": 100000,
                    "max_steps": None,
                    "nominal_epoch_steps": 12500,
                    "purpose": "candidate draft for longer SWE/RL runs",
                },
                {
                    "name": "generic_optional",
                    "examples": 500000,
                    "max_steps": None,
                    "nominal_epoch_steps": 62500,
                    "purpose": "broad reusable Qwen3-235B draft outside the SWE/RL target",
                },
            ],
            "recommendation": {
                "status": "needs_rollout_corpus",
                "summary": "training-scale report is missing; use the default staged plan until rollout corpus exists",
            },
        }

    stages = []
    for item in training.get("stage_plan") or []:
        if not isinstance(item, dict) or item.get("name") == "scenarios":
            continue
        stages.append(
            {
                "name": item.get("name"),
                "examples": item.get("examples"),
                "max_steps": item.get("max_steps"),
                "nominal_epoch_steps": item.get("nominal_epoch_steps"),
                "purpose": item.get("purpose"),
                "gate": item.get("gate"),
                "hidden_state_storage_gib_avg_tokens": item.get("hidden_state_storage_gib_avg_tokens"),
            }
        )
    return {
        "status": report_status(training),
        "defaults": training.get("training_defaults") or {},
        "corpus": training.get("corpus") or {},
        "stages": stages,
        "recommendation": training.get("recommendation") or {},
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_report_args(args)
    reports = {label: load_json(path) for label, path in paths.items()}

    actions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    container = reports["container_preflight"]
    container_status = report_status(container)
    container_command = get_nested(container, ["next_action", "submit_command"])
    artifact_root = args.artifact_root

    remote_host = reports["remote_host_probe"]
    remote_host_status = report_status(remote_host)
    remote_counts = remote_host.get("counts") if isinstance(remote_host.get("counts"), dict) else {}
    remote_reachable = int(remote_counts.get("reachable") or 0)
    remote_gate_unproven = remote_host_status in {"missing", "unreachable", "incomplete", "invalid", "unknown"} or (
        remote_host_status == "pass" and remote_reachable < 1
    )
    if remote_gate_unproven:
        add_action(
            actions,
            "probe_remote_hosts",
            "Probe remote host aliases and Hayate paths",
            "ready_for_operator",
            "Remote execution/Hayate path evidence is missing or not PASS; refresh it before interpreting remote follow-ups.",
            command=remote_host_probe_command(artifact_root),
            report=remote_host.get("_path"),
            priority=-70,
            stage="reference_gate",
            submits_slurm=False,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "remote_host_probe_not_passed",
            "blocking",
            "Remote execution and Hayate/SpecForge path evidence is not proven.",
            report=remote_host.get("_path"),
        )

    source_build = reports["vllm_source_build"]
    source_build_analysis = reports["vllm_source_build_analysis"]
    abi_probe = reports["vllm_abi_probe"]
    source_info = source_build_info(artifact_root)
    source_site = source_vllm_site(artifact_root, source_build, source_info)
    source_job_id = source_info.get("vllm_native_source_build_job")
    source_status = report_status(source_build)
    abi_status = report_status(abi_probe)
    runtime_relevant = (
        args.require_source_vllm_runtime
        or not source_build.get("_missing")
        or not abi_probe.get("_missing")
        or path_under_artifact_root(artifact_root, source_info.get("output_site"))
    )
    abi_covers_source = abi_probe_site_passed(abi_probe, source_site)
    abi_source_failed = abi_probe_site_failed(abi_probe, source_site)
    source_abi_pass = abi_status == "pass" and (abi_covers_source or not source_site)
    source_abi_fail = abi_source_failed
    runtime_ready = not runtime_relevant
    source_analysis_sentence = source_build_analysis_sentence(source_build_analysis)
    source_runtime_report = (
        source_build_analysis.get("_path") if not source_build_analysis.get("_missing") else source_build.get("_path")
    )

    if runtime_relevant:
        if source_status == "pass":
            if source_abi_pass:
                runtime_ready = True
            elif source_abi_fail:
                add_blocker(
                    blockers,
                    "source_built_vllm_abi_failed",
                    "blocking",
                    "The source-built vLLM site failed the native ABI probe; do not submit Qwen3 rollout capture.",
                    report=abi_probe.get("_path"),
                )
            else:
                add_action(
                    actions,
                    "submit_source_vllm_abi_probe",
                    "Submit native ABI probe for the source-built vLLM site",
                    "ready_for_operator",
                    "The source build reports PASS, but the ABI probe has not yet proven the source-built site.",
                    command=source_abi_probe_command(artifact_root, source_site),
                    report=abi_probe.get("_path"),
                    priority=-40,
                    stage="runtime_gate",
                    submits_slurm=True,
                    heavy_gpu=False,
                    after_commands=refresh_after_commands(artifact_root),
                )
                add_blocker(
                    blockers,
                    "source_built_vllm_abi_unproven",
                    "blocking",
                    "Qwen3 rollout capture waits for `vllm._C` and `CompilationConfig` imports from the source-built site.",
                    report=abi_probe.get("_path"),
                )
        elif source_status == "fail":
            add_action(
                actions,
                "submit_vllm_source_build",
                "Resubmit vLLM source build after inspecting the failed build log",
                "blocked",
                "The latest vLLM source-build report failed; inspect logs and patch the build wrapper before resubmitting.",
                command=source_build_submit_command(artifact_root),
                report=source_build.get("_path"),
                priority=-50,
                stage="runtime_gate",
                submits_slurm=True,
                heavy_gpu=False,
                after_commands=source_build_after_commands(artifact_root),
            )
            add_blocker(
                blockers,
                "vllm_source_build_failed",
                "blocking",
                "Source-built vLLM is required because shared pip wheels fail the target Torch/CUDA ABI.",
                report=source_build.get("_path"),
            )
        elif source_job_id:
            add_action(
                actions,
                "poll_vllm_source_build",
                "Poll the vLLM source-build job",
                "ready_for_operator",
                "The active runtime gate is still compiling vLLM against the target NeMo container Torch/CUDA ABI."
                + (f" {source_analysis_sentence}" if source_analysis_sentence else ""),
                command=source_build_poll_command(artifact_root, source_info),
                report=source_runtime_report,
                priority=-50,
                stage="runtime_gate",
                submits_slurm=False,
                heavy_gpu=False,
                after_commands=refresh_after_commands(artifact_root),
            )
            add_blocker(
                blockers,
                "vllm_source_build_running",
                "blocking",
                "Rollout capture and Eagle3 training should wait until the source-built vLLM site passes native ABI probe."
                + (f" {source_analysis_sentence}" if source_analysis_sentence else ""),
                report=source_runtime_report,
            )
        else:
            add_action(
                actions,
                "submit_vllm_source_build",
                "Submit vLLM source build in the target NeMo container",
                "ready_for_operator",
                "Shared vLLM wheels failed native ABI probes and no source-build job/report is visible.",
                command=source_build_submit_command(artifact_root),
                report=source_build.get("_path"),
                priority=-50,
                stage="runtime_gate",
                submits_slurm=True,
                heavy_gpu=False,
                after_commands=source_build_after_commands(artifact_root),
            )
            add_blocker(
                blockers,
                "vllm_source_build_missing",
                "blocking",
                "A source-built vLLM site is required before Qwen3 rollout capture can produce Eagle3 data.",
                report=source_build.get("_path"),
            )

    if container_status == "pass":
        pass
    elif container_command:
        add_action(
            actions,
            "submit_container_preflight",
            "Submit the container-only ModelOpt preflight",
            "ready_for_operator",
            "The selected sqsh/container has not passed the ModelOpt/chat-template gate yet.",
            command=str(container_command),
            report=container.get("_path"),
            stage="container_gate",
            submits_slurm=True,
            heavy_gpu=False,
            after_commands=container_after_commands(str(container_command), artifact_root),
        )
        add_blocker(
            blockers,
            "container_preflight_not_passed",
            "blocking",
            "Do not run hidden-state dump or training until container preflight reports PASS.",
            report=container.get("_path"),
        )
    else:
        add_blocker(
            blockers,
            "container_preflight_missing",
            "blocking",
            "Container preflight report is missing or has no submit command.",
            report=container.get("_path"),
        )

    rollout_submit = reports["rollout_submit_preflight"]
    rollout_state = reports["rollout_state"]
    rollout_submit_status = report_status(rollout_submit)
    rollout_submit_ready = boolish(rollout_submit.get("submit_ready"))
    rollout_state_status = report_status(rollout_state)
    rollout_next_step = get_nested(rollout_state, ["decision", "next_step"]) or rollout_state.get("next_step")
    rollout_state_command = None
    if rollout_next_step:
        rollout_state_command = get_nested(rollout_state, ["commands", str(rollout_next_step)])
    rollout_submit_command = get_nested(rollout_submit, ["commands", "submit"])
    megatron_compat = reports["megatron_compat"]
    megatron_compat_status = report_status(megatron_compat)
    megatron_info = megatron_compat_info(args)
    megatron_job_id = megatron_info.get("megatron_compat_probe_job")
    megatron_ready = megatron_compat_status == "pass"
    rollout_submit_prereqs_unproven = (
        remote_gate_unproven or not runtime_ready or not megatron_ready or container_status != "pass"
    )
    rollout_submit_fail_deferred = rollout_submit_status == "fail" and rollout_submit_prereqs_unproven

    if megatron_ready:
        pass
    elif megatron_compat_status == "fail":
        add_action(
            actions,
            "submit_megatron_compat_probe",
            "Resubmit the Megatron compatibility probe after inspecting the failed report",
            "blocked",
            "The latest Megatron compatibility probe failed; inspect the report before spending another rollout slot.",
            command=megatron_compat_submit_command(artifact_root),
            report=megatron_compat.get("_path"),
            priority=-30,
            stage="runtime_gate",
            submits_slurm=True,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "megatron_compat_probe_failed",
            "blocking",
            "Qwen3 rollout capture waits for the Megatron-Bridge Qwen3MoE compatibility probe to pass.",
            report=megatron_compat.get("_path"),
        )
    elif megatron_job_id:
        add_action(
            actions,
            "poll_megatron_compat_probe",
            "Poll the Megatron compatibility probe",
            "ready_for_operator",
            "A Megatron compatibility probe job is recorded, but its PASS report has not been observed yet.",
            command=megatron_compat_poll_command(artifact_root, megatron_info),
            report=megatron_compat.get("_path"),
            priority=-30,
            stage="runtime_gate",
            submits_slurm=False,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "megatron_compat_probe_pending",
            "blocking",
            "Do not submit the next rollout until the grouped-expert Megatron compatibility probe reports PASS.",
            report=megatron_compat.get("_path"),
        )
    else:
        add_action(
            actions,
            "submit_megatron_compat_probe",
            "Submit the Megatron compatibility probe",
            "ready_for_operator",
            "No PASS report is visible for the Megatron-Bridge Qwen3MoE compatibility shims used by rollout capture.",
            command=megatron_compat_submit_command(artifact_root),
            report=megatron_compat.get("_path"),
            priority=-30,
            stage="runtime_gate",
            submits_slurm=True,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "megatron_compat_probe_missing",
            "blocking",
            "Run the Megatron compatibility probe before spending another multi-node Qwen3 rollout attempt.",
            report=megatron_compat.get("_path"),
        )

    if rollout_submit_fail_deferred:
        add_blocker(
            blockers,
            "rollout_submit_preflight_deferred",
            "blocking",
            "Rollout submit preflight failed before remote/runtime/Megatron/container prerequisites were proven; rerun it after those gates pass.",
            report=rollout_submit.get("_path"),
        )

    if not runtime_ready:
        add_blocker(
            blockers,
            "vllm_runtime_gate_not_ready",
            "blocking",
            "The source-built vLLM runtime gate is ahead of rollout capture in the current execution order.",
            report=source_build.get("_path"),
        )
    elif not megatron_ready:
        add_blocker(
            blockers,
            "megatron_compat_gate_not_ready",
            "blocking",
            "The Megatron compatibility gate is ahead of rollout capture in the current execution order.",
            report=megatron_compat.get("_path"),
        )
    elif rollout_submit.get("_missing"):
        add_action(
            actions,
            "run_rollout_submit_preflight",
            "Run rollout-capture submit preflight",
            "needs_report",
            "The rollout capture gate has not been generated yet.",
            report=rollout_submit.get("_path"),
            stage="rollout_gate",
        )
        add_blocker(
            blockers,
            "rollout_submit_preflight_missing",
            "blocking",
            "Need rollout submit preflight before any capture job submission.",
            report=rollout_submit.get("_path"),
        )
    elif rollout_submit_ready and rollout_next_step in {None, "submit_capture"} and rollout_state_status in {
        "missing",
        "not_submitted",
        "missing_capture",
        "unknown",
    }:
        add_action(
            actions,
            "submit_rollout_capture",
            "Submit the 1-step Qwen3 SWE rollout-capture smoke",
            "ready_for_operator",
            "The no-submit gate is ready, and no usable target-domain rollout corpus exists yet.",
            command=str(rollout_submit_command) if rollout_submit_command else None,
            report=rollout_submit.get("_path"),
            stage="rollout_capture",
            submits_slurm=True,
            heavy_gpu=True,
            after_commands=rollout_after_commands(str(rollout_submit_command), rollout_submit, artifact_root)
            if rollout_submit_command
            else [],
        )
        add_blocker(
            blockers,
            "rollout_corpus_missing",
            "blocking",
            "Non-pilot Eagle3 training needs actual Qwen3 SWE/RL rollout conversations.",
            report=rollout_state.get("_path"),
        )
    elif rollout_next_step in {"poll", "materialize", "materialize_and_refresh", "pipeline_dry_run"}:
        title_by_step = {
            "poll": "Poll the rollout-capture job",
            "materialize": "Materialize rollout train_data into ModelOpt conversations",
            "materialize_and_refresh": "Materialize rollout corpus and refresh reports",
            "pipeline_dry_run": "Run hidden-state pipeline dry-run/preflight",
        }
        if rollout_next_step != "pipeline_dry_run":
            add_action(
                actions,
                f"rollout_{rollout_next_step}",
                title_by_step[str(rollout_next_step)],
                "ready_for_operator",
                "Rollout state report selected this as the next step.",
                command=ensure_rollout_state_outputs(str(rollout_state_command), artifact_root, rollout_state.get("_path"))
                if rollout_state_command
                else None,
                report=rollout_state.get("_path"),
                stage="rollout_capture",
                submits_slurm=False,
                heavy_gpu=False,
                after_commands=refresh_after_commands(artifact_root),
            )
            add_blocker(
                blockers,
                "rollout_corpus_not_ready",
                "blocking",
                "Rollout corpus is not ready for hidden-state dump yet.",
                report=rollout_state.get("_path"),
            )

        fallback = reports["rollout_fallback_decision"]
        fallback_status = report_status(fallback)
        fallback_command = fallback.get("next_command")
        fallback_recommendation = str(fallback.get("recommendation") or "")
        if (
            rollout_next_step == "poll"
            and fallback_status == "fallback_ready"
            and fallback_command
            and fallback_recommendation.startswith("submit_")
        ):
            selected = fallback.get("selected_fallback") if isinstance(fallback.get("selected_fallback"), dict) else {}
            profile_id = str(selected.get("id") or fallback_recommendation.removeprefix("submit_") or "fallback")
            add_action(
                actions,
                "submit_rollout_fallback",
                f"Submit fallback rollout-capture smoke ({profile_id})",
                "ready_for_operator",
                str(fallback.get("detail") or "A smaller prevalidated rollout profile may start sooner than the current queued smoke."),
                command=str(fallback_command),
                report=fallback.get("_path"),
                stage="rollout_capture",
                submits_slurm=True,
                heavy_gpu=True,
                after_commands=refresh_after_commands(artifact_root),
            )
    elif not rollout_submit_ready:
        add_blocker(
            blockers,
            "rollout_submit_preflight_not_ready",
            "blocking",
            "Rollout capture submit preflight is not ready; inspect its failing checks.",
            report=rollout_submit.get("_path"),
        )

    pipeline = reports["pipeline_submit_preflight"]
    full_rollout_gate = reports["full_rollout_gate"]
    pipeline_analysis = reports["pipeline_analysis"]
    training_checkpoint = reports["training_checkpoint"]
    export_artifacts = reports["export_artifacts"]
    trained_draft_sweep = reports["trained_draft_sweep"]
    pipeline_status = report_status(pipeline)
    full_rollout_gate_status = report_status(full_rollout_gate)
    full_rollout_gate_next_step = get_nested(full_rollout_gate, ["decision", "next_step"]) or full_rollout_gate.get("next_step")
    full_rollout_submit_ready = (
        full_rollout_gate_status == "ready" and full_rollout_gate_next_step == "submit_full_rollout"
    )
    pipeline_ready = boolish(pipeline.get("submit_ready"))
    pipeline_analysis_pass = report_status(pipeline_analysis) == "pass"
    training_checkpoint_pass = report_status(training_checkpoint) == "pass"
    export_artifacts_pass = report_status(export_artifacts) == "pass"
    pipeline_complete = pipeline_analysis_pass and training_checkpoint_pass and export_artifacts_pass
    sweep_status = report_status(trained_draft_sweep)
    container_ready = container_status == "pass"
    rollout_corpus_ready = runtime_ready and rollout_state_status == "pass" and rollout_next_step == "pipeline_dry_run"
    if full_rollout_submit_ready:
        add_action(
            actions,
            "submit_full_swegym_rollout",
            "Submit the full SWE-Gym Qwen3 rollout-capture job",
            "ready_for_operator",
            "Smoke passed and full SWE-Gym submit preflight is ready; collect the 2,438-row calibration rollout before Eagle3 training.",
            command=full_rollout_gate_execute_command(args, full_rollout_gate),
            report=full_rollout_gate.get("_path"),
            stage="rollout_capture",
            submits_slurm=True,
            heavy_gpu=True,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "full_swegym_rollout_not_captured",
            "blocking",
            "Run the full SWE-Gym rollout before treating the smoke output as an Eagle3 training corpus.",
            report=full_rollout_gate.get("_path"),
        )
    if pipeline_complete:
        pass
    elif pipeline_analysis_pass:
        missing_contracts = []
        if not training_checkpoint_pass:
            missing_contracts.append("training checkpoint")
            add_blocker(
                blockers,
                "training_checkpoint_contract_missing",
                "blocking",
                "Pipeline logs passed, but the ModelOpt training checkpoint contract has not passed.",
                report=training_checkpoint.get("_path"),
            )
        if not export_artifacts_pass:
            missing_contracts.append("export artifacts")
            add_blocker(
                blockers,
                "export_artifact_contract_missing",
                "blocking",
                "Pipeline logs passed, but the HF/vLLM export artifact contract has not passed.",
                report=export_artifacts.get("_path"),
            )
        add_action(
            actions,
            "run_post_export_artifact_validations",
            "Run post-export artifact contract validations",
            "ready_for_operator",
            f"Pipeline logs passed, but {', '.join(missing_contracts)} evidence is missing or not PASS.",
            command=post_export_artifact_validation_command(args, pipeline_analysis),
            report=pipeline_analysis.get("_path"),
            stage="artifact_validation",
            submits_slurm=False,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
    elif pipeline_ready:
        gated_submit = get_nested(pipeline, ["commands", "gated_pilot_submit"])
        if not gated_submit:
            add_blocker(
                blockers,
                "pipeline_gated_submit_command_missing",
                "blocking",
                "Pipeline submit preflight is ready, but it did not emit commands.gated_pilot_submit; rerun preflight_eagle3_pipeline_submit.py before submitting.",
                report=pipeline.get("_path"),
            )
        else:
            add_action(
                actions,
                "submit_eagle3_pilot_pipeline",
                "Submit the Eagle3 hidden-state/train/export pilot pipeline",
                "ready_for_operator",
                "The pipeline submit preflight reports submit_ready=true and emitted a gated submit command.",
                command=gated_submit,
                report=pipeline.get("_path"),
                stage="eagle3_pipeline",
                submits_slurm=True,
                heavy_gpu=True,
                after_commands=pipeline_after_commands(gated_submit, pipeline, artifact_root),
            )
    elif pipeline_status == "fail":
        add_blocker(
            blockers,
            "pipeline_submit_preflight_failed",
            "blocking",
            "Hidden-state dump/train/export pipeline preflight failed; inspect its failing checks before rerunning.",
            report=pipeline.get("_path"),
        )
    elif container_ready and rollout_corpus_ready and not full_rollout_submit_ready:
        command = pipeline_submit_preflight_command(args, paths, container, rollout_submit, rollout_state, pipeline)
        add_action(
            actions,
            "run_pipeline_submit_preflight",
            "Run the Eagle3 pipeline submit preflight",
            "ready_for_operator",
            "Container preflight and target rollout corpus are ready; prove the hidden-state/train/export chain before submitting it.",
            command=command,
            report=pipeline.get("_path"),
            stage="eagle3_pipeline",
            submits_slurm=False,
            heavy_gpu=False,
            after_commands=refresh_after_commands(artifact_root),
        )
        add_blocker(
            blockers,
            "pipeline_submit_not_ready",
            "blocking",
            "Run the no-submit pipeline submit preflight before the pilot hidden-state/train/export Slurm chain.",
            report=pipeline.get("_path"),
        )
    elif pipeline_status in {"missing", "unknown"}:
        add_action(
            actions,
            "run_pipeline_submit_preflight",
            "Run the Eagle3 pipeline submit preflight after rollout/container gates pass",
            "blocked",
            "The expensive pipeline is not ready; first pass rollout corpus and container preflight.",
            report=pipeline.get("_path"),
            stage="eagle3_pipeline",
        )
    else:
        add_blocker(
            blockers,
            "pipeline_submit_not_ready",
            "blocking",
            "Hidden-state dump/train/export pipeline preflight is not submit-ready yet.",
            report=pipeline.get("_path"),
        )

    if pipeline_complete:
        if sweep_status == "pass":
            pass
        elif sweep_status == "fail":
            add_blocker(
                blockers,
                "trained_draft_sweep_failed",
                "blocking",
                "Trained-draft spec-token sweep failed; inspect acceptance/speedup rows before longer RL runs.",
                report=trained_draft_sweep.get("_path"),
            )
        else:
            add_action(
                actions,
                "submit_trained_draft_spec_tokens_sweep",
                "Submit the trained Eagle3 draft spec-token sweep",
                "ready_for_operator",
                "The hidden-state/train/export pipeline passed, but trained-draft RL sweep evidence is missing.",
                command=trained_draft_sweep_command(args, pipeline_analysis),
                report=trained_draft_sweep.get("_path"),
                stage="rl_validation",
                submits_slurm=True,
                heavy_gpu=True,
                after_commands=trained_draft_sweep_after_commands(artifact_root),
            )
            add_blocker(
                blockers,
                "trained_draft_sweep_missing",
                "blocking",
                "Run the trained-draft spec-token sweep before claiming a usable Qwen3-235B Eagle3 draft.",
                report=trained_draft_sweep.get("_path"),
            )

    loss_mask = reports["modelopt_loss_mask"]
    if report_status(loss_mask) == "fail":
        add_blocker(
            blockers,
            "modelopt_loss_mask_patch_failed",
            "blocking",
            "ModelOpt TRT-LLM loss-mask validation failed; answer-only Eagle3 loss is unsafe.",
            report=loss_mask.get("_path"),
        )
    elif report_status(loss_mask) in {"missing", "invalid"}:
        add_blocker(
            blockers,
            "modelopt_loss_mask_patch_unproven",
            "warning",
            "ModelOpt loss-mask patch validation has not been generated in this report set.",
            report=loss_mask.get("_path"),
        )

    readiness = reports["readiness"]
    if report_status(readiness) == "fail":
        add_blocker(
            blockers,
            "readiness_audit_not_ready",
            "warning",
            "Readiness audit has failed checks; this records the current missing gates but should not hide actionable operator steps.",
            report=readiness.get("_path"),
        )

    nemo_drift = reports["nemo_rl_drift"]
    nemo_status = report_status(nemo_drift)
    if nemo_status in {"warn", "incomplete"}:
        recommendation = nemo_drift.get("recommendation")
        add_blocker(
            blockers,
            "online_draft_training_not_primary",
            "warning",
            str(recommendation)
            if recommendation
            else "Fixed exported draft is the primary path; online draft training remains a later gate.",
            report=nemo_drift.get("_path"),
        )

    failed_report_labels = []
    for label, payload in reports.items():
        if report_status(payload) != "fail":
            continue
        if label == "readiness":
            # Readiness is an aggregate audit of gates that the planner is
            # already turning into ordered blockers/actions. A FAIL readiness
            # report should be visible, but it should not suppress those
            # actionable next steps by making the whole plan hard-fail.
            continue
        if label == "vllm_abi_probe" and runtime_relevant and not source_abi_fail:
            # Existing wheel-site ABI probes may be FAIL while a source-built
            # vLLM site is still compiling. Treat that as the reason for the
            # source-build path, not as failure of the current runtime gate.
            continue
        if label == "rollout_submit_preflight" and rollout_submit_fail_deferred:
            # The no-submit rollout gate checks cluster-visible runtime inputs.
            # Before the preceding gates have passed, a FAIL report is useful
            # evidence of the next blocked gate, not a reason to suppress
            # currently actionable remote/runtime/container steps.
            continue
        failed_report_labels.append(label)

    has_fail = bool(failed_report_labels) or any(
        item["severity"] == "blocking" and item["id"].endswith("_failed") for item in blockers
    )
    any_ready_submit = any(action["status"] == "ready_for_operator" and action["command"] for action in actions)
    if has_fail:
        overall = "fail"
    elif any_ready_submit:
        overall = "ready_for_operator_submit"
    elif pipeline_ready and not pipeline_complete:
        overall = "ready_for_pipeline_submit"
    else:
        overall = "incomplete"

    report_table = {
        label: {
            "path": str(paths[label]),
            "status": report_status(payload),
            "submit_ready": payload.get("submit_ready"),
            "next_step": get_nested(payload, ["decision", "next_step"]) or payload.get("next_step"),
        }
        for label, payload in reports.items()
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "artifact_root": str(args.artifact_root),
        "build_path": {
            "mode": "fixed_exported_eagle3_draft_first",
            "summary": (
                "Capture actual Qwen3 SWE/RL rollout conversations, dump Qwen3-235B verifier hidden states, "
                "train an offline ModelOpt Eagle3 draft, export it for vLLM, then validate inside NeMo-RL."
            ),
            "online_training": "later_gate_after_fixed_draft_speed_and_reward_smoke",
        },
        "reports": report_table,
        "training": training_summary(reports["training_scale"]),
        "next_actions": sorted(actions, key=lambda item: item["priority"]),
        "blockers": blockers,
    }


def render_training_table(training: dict[str, Any]) -> list[str]:
    defaults = training.get("defaults") or {}
    lines = [
        "## Training Scale",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| effective global batch | {defaults.get('effective_global_batch', '-')} |",
        f"| epochs | {defaults.get('epochs', '-')} |",
        f"| max sequence length | {defaults.get('max_seq_len', '-')} |",
        "",
        "| stage | examples | max steps | nominal steps | purpose |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for stage in training.get("stages") or []:
        max_steps = stage.get("max_steps")
        if max_steps is None:
            max_steps = "epoch"
        lines.append(
            f"| {stage.get('name')} | {stage.get('examples')} | {max_steps} | "
            f"{stage.get('nominal_epoch_steps')} | {str(stage.get('purpose') or '').replace('|', '/')} |"
        )
    rec = training.get("recommendation") or {}
    if rec:
        lines += ["", f"Recommendation: {rec.get('summary') or rec.get('status') or '-'}"]
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Next-Action Plan",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        "## Summary",
        "",
        data["build_path"]["summary"],
        "",
        "The current route is fixed exported Eagle3 draft first. Online draft training remains a later gate after the fixed draft proves rollout speedup without reward or malformed-output regression.",
        "",
        "## Report State",
        "",
        "| report | status | submit ready | next step | path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, report in data["reports"].items():
        lines.append(
            f"| {label} | {report.get('status')} | {report.get('submit_ready')} | "
            f"{report.get('next_step') or '-'} | `{report.get('path')}` |"
        )

    lines += ["", "## Ordered Next Actions", "", "| order | action | status | submits Slurm | heavy GPU | reason |"]
    lines += ["| ---: | --- | --- | --- | --- | --- |"]
    for idx, action in enumerate(data["next_actions"], 1):
        lines.append(
            f"| {idx} | {action['title']} | {action['status']} | {str(action['submits_slurm']).lower()} | "
            f"{str(action['heavy_gpu']).lower()} | {action['reason'].replace('|', '/')} |"
        )
    if not data["next_actions"]:
        lines.append("| - | no automatic action | incomplete | false | false | inspect blockers |")

    blockers = data.get("blockers") or []
    lines += ["", "## Blockers And Warnings", "", "| severity | item | summary |"]
    lines += ["| --- | --- | --- |"]
    if blockers:
        for item in blockers:
            lines.append(f"| {item['severity']} | {item['id']} | {item['summary'].replace('|', '/')} |")
    else:
        lines.append("| - | none | no blockers found in the supplied reports |")

    lines += ["", *render_training_table(data["training"])]

    command_actions = [action for action in data["next_actions"] if action.get("command")]
    if command_actions:
        lines += ["", "## Commands", ""]
        for action in command_actions:
            lines += [f"### {action['title']}", "", "```bash", str(action["command"]), "```", ""]
            after = action.get("after_commands") or []
            if after:
                if action.get("submits_slurm"):
                    lines += ["After the submitted Slurm job reaches a terminal state:", ""]
                else:
                    lines += ["After the action completes:", ""]
                for command in after:
                    lines += ["```bash", str(command), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    data = build_plan(args)
    markdown = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")
    return 1 if data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
