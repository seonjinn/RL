#!/usr/bin/env python3
"""No-submit gate for the Qwen3 SWE rollout-capture job.

This verifies the exact pieces used by run_rollout_capture_smoke.sh and
run_grpo_qwen3_235b_swe.sh without submitting Slurm work.
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
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_SPECDEC_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)
DEFAULT_ROLLOUT_CONTAINER = Path(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"
)
DEFAULT_SWEGYM_EXAMPLE_DATA = Path(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/ultra/"
    "tk-nemo-gym/responses_api_agents/swe_agents/data/example.jsonl"
)
DEFAULT_SOURCE_VLLM_PIP_SPEC = (
    "https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/"
    "vllm-0.10.2.tar.gz"
)
PASSTHROUGH_ENV_KEYS = (
    "INSTALL_VLLM_IN_SYSTEM",
    "SHARED_VLLM_SITE",
    "VLLM_PIP_SPEC",
    "VLLM_WHEEL_LOCATION",
    "VLLM_ENFORCE_EAGER",
    "VLLM_COMPILATION_LEVEL",
    "VLLM_USE_INDUCTOR",
    "MEGATRON_BRIDGE_PLUGIN_DIR",
    "MEGATRON_BRIDGE_QWEN3MOE_PLUGIN",
    "MEGATRON_BRIDGE_SRC",
    "MEGATRON_LM_SRC",
)
SOURCE_VLLM_REQUIRED_ENV_KEYS = (
    "SHARED_VLLM_SITE",
    "VLLM_PIP_SPEC",
    "VLLM_ENFORCE_EAGER",
    "VLLM_COMPILATION_LEVEL",
    "VLLM_USE_INDUCTOR",
)
REQUIRED_SWE_METADATA_KEYS = (
    "problem_statement",
    "instance_id",
    "base_commit",
    "dataset_name",
    "split",
    "instance_dict",
)


def parse_args() -> argparse.Namespace:
    artifact_default = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_default)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("SWE_REPO_ROOT") or os.environ.get("REPO_ROOT") or DEFAULT_SPECDEC_RL_DIR),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("CONFIG_FILE") or os.environ.get("NEMO_RL_CONFIG") or ROOT / "grpo_qwen3_235b_swe.yaml"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("ENV_FILE", ROOT / "env.sh")))
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(
            os.environ.get(
                "CHAT_TEMPLATE",
                artifact_default / "templates/qwen3_generation_template.jinja2",
            )
        ),
    )
    parser.add_argument("--rollout-log-dir", type=Path)
    parser.add_argument("--output-conversations", type=Path)
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=Path(os.environ["TRAIN_DATA_PATH"]) if os.environ.get("TRAIN_DATA_PATH") else None,
        help="Effective NemoGym/SWE JSONL passed as data.train.data_path.",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=Path(os.environ["VAL_DATA_PATH"]) if os.environ.get("VAL_DATA_PATH") else None,
        help="Effective NemoGym/SWE JSONL passed as data.validation.data_path.",
    )
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "coreai_dlalgo_nemorl"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--max-num-steps", type=int, default=int(os.environ.get("MAX_NUM_STEPS", "1")))
    parser.add_argument(
        "--wandb-name",
        default=os.environ.get("WANDB_NAME", "qwen3-235b-swe-rollout-capture-smoke"),
        help="Slurm/W&B experiment name passed through the rollout wrapper.",
    )
    parser.add_argument("--container", type=Path, default=Path(os.environ.get("CONTAINER", DEFAULT_ROLLOUT_CONTAINER)))
    parser.add_argument(
        "--require-source-vllm-env",
        action="store_true",
        default=os.environ.get("REQUIRE_SOURCE_VLLM_ENV", "").lower() in {"1", "true", "yes", "on"},
        help="Require source-built vLLM passthrough env in the emitted submit command.",
    )
    parser.add_argument(
        "--resource-profile-env",
        type=Path,
        default=Path(os.environ.get("RESOURCE_PROFILE_ENV", artifact_default / "reports/eagle3_resource_profile.env")),
    )
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def shell_join(env: dict[str, str], command: list[str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    body = " ".join(shlex.quote(part) for part in command)
    return f"{prefix} {body}".strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def proven_source_vllm_site(artifact_root: Path) -> str | None:
    source_build = load_json(artifact_root / "reports/vllm_native_source_build.json")
    abi_probe = load_json(artifact_root / "reports/vllm_native_abi_probe.json")
    source_site = str(source_build.get("output_site") or "")
    if source_build.get("overall_status") != "pass" or not source_site:
        return None
    if abi_probe.get("overall_status") != "pass":
        return None
    for result in abi_probe.get("results") or []:
        if not isinstance(result, dict) or str(result.get("site") or "") != source_site:
            continue
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        if result.get("returncode") == 0 and parsed.get("vllm_c_ok") and parsed.get("compilation_config_ok"):
            return source_site
    return None


def runtime_passthrough_env(artifact_root: Path | None = None) -> dict[str, str]:
    env = {key: value for key in PASSTHROUGH_ENV_KEYS if (value := os.environ.get(key))}
    if artifact_root is not None and "SHARED_VLLM_SITE" not in env:
        source_site = proven_source_vllm_site(artifact_root)
        if source_site:
            env.setdefault("INSTALL_VLLM_IN_SYSTEM", "true")
            env["SHARED_VLLM_SITE"] = source_site
            env.setdefault("VLLM_PIP_SPEC", DEFAULT_SOURCE_VLLM_PIP_SPEC)
            env.setdefault("VLLM_ENFORCE_EAGER", "True")
            env.setdefault("VLLM_COMPILATION_LEVEL", "0")
            env.setdefault("VLLM_USE_INDUCTOR", "False")
    return env


def source_vllm_env_required(args: argparse.Namespace, rollout_log_dir: Path, output_conversations: Path) -> bool:
    if args.require_source_vllm_env:
        return True
    if proven_source_vllm_site(args.artifact_root):
        return True
    text = " ".join(
        [
            str(args.wandb_name),
            str(rollout_log_dir),
            str(output_conversations),
        ]
    ).lower()
    normalized = text.replace("_", "-")
    return any(marker in normalized for marker in ("vllm0102src", "source-vllm", "source-built-vllm"))


def check_source_vllm_env(
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
    rollout_log_dir: Path,
    output_conversations: Path,
) -> dict[str, Any]:
    required = source_vllm_env_required(args, rollout_log_dir, output_conversations)
    env = runtime_passthrough_env(args.artifact_root)
    if not required:
        add(
            checks,
            "runtime",
            "source-built vLLM env",
            "pass",
            "source-built vLLM passthrough env is not required for this rollout name",
            required=False,
            runtime_passthrough_env=env,
        )
        return {"required": False, "runtime_passthrough_env": env, "missing": [], "invalid": []}

    missing = [key for key in SOURCE_VLLM_REQUIRED_ENV_KEYS if not env.get(key)]
    invalid: list[str] = []
    if env.get("VLLM_ENFORCE_EAGER", "").lower() not in {"1", "true", "yes", "on"}:
        invalid.append("VLLM_ENFORCE_EAGER must be true")
    if env.get("VLLM_COMPILATION_LEVEL") != "0":
        invalid.append("VLLM_COMPILATION_LEVEL must be 0")
    if env.get("VLLM_USE_INDUCTOR", "").lower() not in {"0", "false", "no", "off"}:
        invalid.append("VLLM_USE_INDUCTOR must be false")
    source_site = Path(env["SHARED_VLLM_SITE"]) if env.get("SHARED_VLLM_SITE") else None
    if source_site is not None and not source_site.exists():
        invalid.append(f"SHARED_VLLM_SITE is not visible: {source_site}")

    ok = not missing and not invalid
    add(
        checks,
        "runtime",
        "source-built vLLM env",
        "pass" if ok else "fail",
        "source-built vLLM env is present and suitable for the rollout submit command"
        if ok
        else "source-built vLLM env is required but incomplete or invalid",
        required=True,
        missing=missing,
        invalid=invalid,
        runtime_passthrough_env=env,
    )
    return {"required": True, "runtime_passthrough_env": env, "missing": missing, "invalid": invalid}


def read_export_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def rollout_resource_env(resource_env: dict[str, str]) -> dict[str, str]:
    gpus = positive_int(
        resource_env.get("ROLLOUT_GPUS_PER_NODE")
        or resource_env.get("TRAIN_GPUS_PER_NODE")
        or resource_env.get("DUMP_GPUS_PER_NODE")
    )
    if not gpus:
        return {}
    env = {"NUM_GPU": str(gpus)}
    actor_total = positive_int(resource_env.get("ROLLOUT_TOTAL_ACTOR_GPUS")) or 128
    generation_total = positive_int(resource_env.get("ROLLOUT_TOTAL_GENERATION_GPUS")) or 64
    env["NUM_NODES"] = str((actor_total + gpus - 1) // gpus)
    env["NUM_GEN_NODES"] = str((generation_total + gpus - 1) // gpus)
    return env


def apply_runtime_resource_overrides(resource_overrides: dict[str, str]) -> dict[str, str]:
    result = dict(resource_overrides)
    for key in ("NUM_GPU", "NUM_NODES", "NUM_GEN_NODES"):
        value = os.environ.get(key)
        if positive_int(value):
            result[key] = str(value)
    return result


def effective_data_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, str]:
    train = args.train_data_path
    source = "explicit"
    fixed_swegym_example = args.artifact_root / "data/swegym_example_for_sweagent_with_instance_dict.jsonl"
    if train is None and fixed_swegym_example.exists():
        train = fixed_swegym_example
        source = "generated_swegym_example"
    elif train is None and DEFAULT_SWEGYM_EXAMPLE_DATA.exists():
        train = DEFAULT_SWEGYM_EXAMPLE_DATA
        source = "default_swegym_example"
    elif train is None:
        source = "launcher_default"
    val = args.val_data_path or train
    return train, val, source


def validate_swe_metadata(metadata: Any) -> tuple[list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    warnings: list[str] = []
    if not isinstance(metadata, dict):
        return ["metadata is not an object"], warnings, {}
    for key in REQUIRED_SWE_METADATA_KEYS:
        if metadata.get(key) in (None, ""):
            failures.append(f"missing metadata.{key}")
    instance_raw = metadata.get("instance_dict")
    instance_keys: list[str] = []
    if isinstance(instance_raw, str) and instance_raw.strip():
        try:
            instance = json.loads(instance_raw)
            if not isinstance(instance, dict):
                failures.append("metadata.instance_dict is not a JSON object string")
            else:
                instance_keys = sorted(instance.keys())
                for key in ("instance_id", "repo", "base_commit"):
                    if instance.get(key) in (None, ""):
                        warnings.append(f"instance_dict missing {key}")
        except json.JSONDecodeError as exc:
            failures.append(f"metadata.instance_dict is not valid JSON: {exc}")
    elif "instance_dict" in metadata:
        failures.append("metadata.instance_dict must be a nonempty JSON string")
    return failures, warnings, {"instance_dict_keys": instance_keys}


def probe_nemogym_jsonl(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"ok": False, "detail": "no path provided"}
    if not path.exists():
        return {"ok": False, "detail": f"not visible: {path}"}
    if not path.is_file():
        return {"ok": False, "detail": f"not a file: {path}"}
    if path.stat().st_size == 0:
        return {"ok": False, "detail": f"empty file: {path}"}
    try:
        first = path.open(encoding="utf-8", errors="replace").readline()
        record = json.loads(first)
    except Exception as exc:
        return {"ok": False, "detail": f"first line is not valid JSON: {exc}"}
    responses_create_params = record.get("responses_create_params") if isinstance(record, dict) else None
    has_swe_payload = isinstance(responses_create_params, dict)
    metadata = responses_create_params.get("metadata", {}) if has_swe_payload else {}
    metadata_failures, metadata_warnings, metadata_details = validate_swe_metadata(metadata) if has_swe_payload else (
        [],
        [],
        {},
    )
    return {
        "ok": bool(has_swe_payload and not metadata_failures),
        "detail": "first record is SWE NemoGym JSONL"
        if has_swe_payload and not metadata_failures
        else f"first record has invalid SWE metadata: {', '.join(metadata_failures)}"
        if has_swe_payload
        else "first record does not contain responses_create_params",
        "size_bytes": path.stat().st_size,
        "instance_id": metadata.get("instance_id") if isinstance(metadata, dict) else None,
        "dataset_name": metadata.get("dataset_name") if isinstance(metadata, dict) else None,
        "metadata_failures": metadata_failures,
        "metadata_warnings": metadata_warnings,
        **metadata_details,
    }


def check_dataset_paths(
    checks: list[dict[str, Any]],
    train_data_path: Path | None,
    val_data_path: Path | None,
    source: str,
) -> None:
    for label, path in (("train", train_data_path), ("validation", val_data_path)):
        probe = probe_nemogym_jsonl(path)
        add(
            checks,
            "data",
            f"{label} NemoGym JSONL",
            "pass" if probe["ok"] else "fail",
            probe["detail"],
            path=str(path) if path else None,
            source=source if label == "train" else ("explicit" if val_data_path != train_data_path else "train_data_path"),
            **{key: value for key, value in probe.items() if key not in {"ok", "detail"}},
        )


def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 90) -> dict[str, Any]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": shell_join(env or {}, cmd),
            "returncode": result.returncode,
            "output": result.stdout,
            "output_tail": result.stdout[-6000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "command": shell_join(env or {}, cmd),
            "returncode": 124,
            "timeout": timeout,
            "output": output,
            "output_tail": output[-6000:],
        }


def check_paths(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    required = [
        ("roadmap run_grpo launcher", ROOT / "run_grpo_qwen3_235b_swe.sh"),
        ("rollout capture wrapper", EXP / "run_rollout_capture_smoke.sh"),
        ("rollout capture validator", EXP / "validate_rollout_capture_config.py"),
        ("Qwen3MoE Megatron-Bridge plugin", EXP / "megatron_bridge_qwen3moe" / "sitecustomize.py"),
        ("Qwen3MoE bridge registration module", EXP / "megatron_bridge_qwen3moe" / "qwen3_moe_bridge_plugin.py"),
        ("Qwen3 SWE config", args.config),
        ("Qwen3 chat template", args.chat_template),
        ("SpecDec-RL repo root", args.repo_root),
        ("SpecDec-RL ray.sub", args.repo_root / "ray.sub"),
        ("SpecDec-RL NeMo-Gym entrypoint", args.repo_root / "examples/nemo_gym/run_grpo_nemo_gym.py"),
        ("SpecDec-RL GRPO source", args.repo_root / "nemo_rl/algorithms/grpo.py"),
    ]
    for name, path in required:
        add(
            checks,
            "paths",
            name,
            "pass" if path.exists() else "fail",
            f"visible: {path}" if path.exists() else f"not visible: {path}",
            path=str(path),
        )
    launcher = ROOT / "run_grpo_qwen3_235b_swe.sh"
    wrapper = EXP / "run_rollout_capture_smoke.sh"
    launcher_uses_bash = "exec bash \"$ROOT_DIR/run_grpo_qwen3_235b_swe.sh\"" in wrapper.read_text(
        encoding="utf-8",
        errors="replace",
    ) if wrapper.exists() else False
    add(
        checks,
        "paths",
        "rollout launcher invocation",
        "pass" if launcher_uses_bash else "fail",
        "rollout wrapper invokes run_grpo launcher through bash"
        if launcher_uses_bash
        else "rollout wrapper must not require executable bit on run_grpo launcher",
        launcher=str(launcher),
        wrapper=str(wrapper),
    )
    add(
        checks,
        "paths",
        "env file",
        "pass" if args.env_file.exists() else "warn",
        f"visible: {args.env_file}"
        if args.env_file.exists()
        else "env file is not visible; submit can still work if tokens/env are provided by caller",
        path=str(args.env_file),
    )


def check_existing_capture(checks: list[dict[str, Any]], rollout_log_dir: Path, output_conversations: Path) -> None:
    train_files = sorted(rollout_log_dir.glob("train_data_step*.jsonl"))
    if train_files:
        add(
            checks,
            "state",
            "existing rollout train_data",
            "warn",
            "train_data_step*.jsonl already exists; materialization may be the next step instead of another submit",
            file_count=len(train_files),
            files=[str(path) for path in train_files[:5]],
        )
    else:
        add(
            checks,
            "state",
            "existing rollout train_data",
            "pass",
            "no existing train_data_step*.jsonl found for this capture target",
            rollout_log_dir=str(rollout_log_dir),
        )
    add(
        checks,
        "state",
        "materialized rollout conversations",
        "warn" if output_conversations.exists() else "pass",
        f"output conversations already exists: {output_conversations}"
        if output_conversations.exists()
        else f"output conversations not present yet: {output_conversations}",
        path=str(output_conversations),
    )


def check_rollout_validator(
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
    train_data_path: Path | None,
    val_data_path: Path | None,
) -> dict[str, Any]:
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/validate_rollout_capture_config.py",
        "--config",
        str(args.config),
        "--specdec-rl-dir",
        str(args.repo_root),
        "--artifact-root",
        str(args.artifact_root),
        "--chat-template",
        str(args.chat_template),
    ]
    if train_data_path:
        cmd.extend(["--train-data-path", str(train_data_path)])
    if val_data_path:
        cmd.extend(["--val-data-path", str(val_data_path)])
    result = run(cmd)
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(result["output"])
    except Exception as exc:
        result["parse_error"] = str(exc)
    status = (parsed or {}).get("overall_status")
    ok = result["returncode"] == 0 and status == "pass"
    add(
        checks,
        "validation",
        "rollout capture config gate",
        "pass" if ok else "fail",
        "validate_rollout_capture_config.py passed"
        if ok
        else "validate_rollout_capture_config.py did not prove submit readiness",
        returncode=result["returncode"],
        overall_status=status,
        output_tail=result["output_tail"],
    )
    return {"result": result, "parsed": parsed}


def check_capture_wrapper_dry_run(
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
    rollout_log_dir: Path,
    output_conversations: Path,
    train_data_path: Path | None,
    val_data_path: Path | None,
) -> dict[str, Any]:
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "ROLLOUT_LOG_DIR": str(rollout_log_dir),
        "OUTPUT_CONVERSATIONS": str(output_conversations),
        "SWE_REPO_ROOT": str(args.repo_root),
        "CONFIG_FILE": str(args.config),
        "ENV_FILE": str(args.env_file),
        "CHAT_TEMPLATE": str(args.chat_template),
        "RESOURCE_PROFILE_ENV": str(args.resource_profile_env),
        "DRY_RUN": "true",
        "MAX_NUM_STEPS": str(args.max_num_steps),
        "WANDB_NAME": args.wandb_name,
        "EXP_SUFFIX_OVERRIDE": args.wandb_name,
        "CHECKPOINT_SUBDIR": args.wandb_name,
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
    }
    env.update(runtime_passthrough_env(args.artifact_root))
    if train_data_path:
        env["TRAIN_DATA_PATH"] = str(train_data_path)
    if val_data_path:
        env["VAL_DATA_PATH"] = str(val_data_path)
    result = run(["bash", "experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"], env=env)
    output = result["output"]
    needles = [
        "DRY_RUN=true",
        f"SWE_REPO_ROOT={args.repo_root}",
        f"CONFIG_FILE={args.config}",
        f"CHAT_TEMPLATE={args.chat_template}",
        f"RESOURCE_PROFILE_ENV={args.resource_profile_env}",
        "run_grpo_qwen3_235b_swe.sh",
    ]
    if train_data_path:
        needles.append(f"TRAIN_DATA_PATH={train_data_path}")
    if val_data_path:
        needles.append(f"VAL_DATA_PATH={val_data_path}")
    missing = [needle for needle in needles if needle not in output]
    ok = result["returncode"] == 0 and not missing
    add(
        checks,
        "dry_run",
        "rollout capture wrapper",
        "pass" if ok else "fail",
        "run_rollout_capture_smoke.sh dry-run uses the expected repo/config/template"
        if ok
        else "run_rollout_capture_smoke.sh dry-run missed expected evidence",
        returncode=result["returncode"],
        missing=missing,
        output_tail=result["output_tail"],
    )
    return result


def check_run_grpo_dry_run(
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
    resource_overrides: dict[str, str],
    train_data_path: Path | None,
    val_data_path: Path | None,
) -> dict[str, Any]:
    env_file = str(args.env_file if args.env_file.exists() else Path("/dev/null"))
    env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "REPO_ROOT": str(args.repo_root),
        "CONFIG_FILE": str(args.config),
        "ENV_FILE": env_file,
        "CHAT_TEMPLATE": str(args.chat_template),
        "DRY_RUN": "true",
        "MAX_NUM_STEPS": str(args.max_num_steps),
        "WANDB_NAME": args.wandb_name,
        "EXP_SUFFIX_OVERRIDE": args.wandb_name,
        "CHECKPOINT_SUBDIR": args.wandb_name,
        "SAVE_PERIOD": "1000000",
        "VAL_PERIOD": "1000000",
        "KEEP_TOP_K": "1",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "CONTAINER": str(args.container),
        "WANDB_API_KEY": "redacted",
        "HUGGINGFACE_TOKEN": "redacted",
        "GITHUB_TOKEN": "redacted",
        "GITLAB_TOKEN": "redacted",
        "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf_home"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets"),
    }
    env.update(runtime_passthrough_env(args.artifact_root))
    if train_data_path:
        env["TRAIN_DATA_PATH"] = str(train_data_path)
    if val_data_path:
        env["VAL_DATA_PATH"] = str(val_data_path)
    env.update(resource_overrides)
    result = run(["bash", "run_grpo_qwen3_235b_swe.sh"], env=env)
    output = result["output"]
    expected_gpu = resource_overrides.get("NUM_GPU")
    needles = [
        f"Repo root: {args.repo_root}",
        f"Config: {args.config}",
        f"Chat template: {args.chat_template}",
        f"Container: {args.container}",
        f"Experiment: {args.wandb_name}",
        "Megatron bridge plugin:",
        "Qwen3MoE bridge plugin enabled: 1",
        f"Train data: {train_data_path}" if train_data_path else "Train data:",
        f"Val data: {val_data_path}" if val_data_path else "Val data:",
        "[DRY-RUN] COMMAND:",
        "[DRY-RUN] sbatch:",
        f"--account={args.sbatch_account}",
        "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=",
    ]
    if train_data_path:
        needles.append(f"++data.train.data_path={train_data_path}")
    if val_data_path:
        needles.append(f"++data.validation.data_path={val_data_path}")
    if expected_gpu:
        needles.extend([f"GPUs/node: {expected_gpu}", f"--gres=gpu:{expected_gpu}"])
    missing = [needle for needle in needles if needle not in output]
    ok = result["returncode"] == 0 and not missing
    add(
        checks,
        "dry_run",
        "run_grpo launcher",
        "pass" if ok else "fail",
        "run_grpo_qwen3_235b_swe.sh dry-run reached the expected Slurm command"
        if ok
        else "run_grpo_qwen3_235b_swe.sh dry-run failed or missed expected evidence",
        returncode=result["returncode"],
        missing=missing,
        output_tail=result["output_tail"],
    )
    return result


def overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Rollout Capture Submit Preflight",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Submit ready: **{str(data['submit_ready']).lower()}**",
        "",
        "Submit command:",
        "",
        "```bash",
        data["commands"]["submit"],
        "```",
        "",
        "Post-submit status command:",
        "",
        "```bash",
        data["commands"]["analyze_job"],
        "```",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in data["checks"]:
        lines.append(
            f"| {check['area']} | {check['name']} | {check['status'].upper()} | "
            f"{check['detail'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rollout_log_dir = args.rollout_log_dir or args.artifact_root / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke"
    output_conversations = args.output_conversations or args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"
    checks: list[dict[str, Any]] = []

    check_paths(checks, args)
    check_existing_capture(checks, rollout_log_dir, output_conversations)
    train_data_path, val_data_path, data_path_source = effective_data_paths(args)
    check_dataset_paths(checks, train_data_path, val_data_path, data_path_source)
    resource_env = read_export_env(args.resource_profile_env)
    resource_overrides = apply_runtime_resource_overrides(rollout_resource_env(resource_env))
    expected_gpu = resource_overrides.get("NUM_GPU")
    dryrun_name = "dryrun" in args.wandb_name.lower().replace("_", "-")
    add(
        checks,
        "naming",
        "non-dry-run submit name",
        "fail" if dryrun_name else "pass",
        "WANDB_NAME is suitable for a real Slurm submit command"
        if not dryrun_name
        else "WANDB_NAME contains dryrun but the emitted submit command uses DRY_RUN=false",
        wandb_name=args.wandb_name,
    )
    source_vllm_env = check_source_vllm_env(checks, args, rollout_log_dir, output_conversations)
    add(
        checks,
        "resources",
        "rollout resource profile",
        "pass" if expected_gpu else "warn",
        f"using NUM_GPU={expected_gpu}, NUM_NODES={resource_overrides.get('NUM_NODES')}, "
        f"NUM_GEN_NODES={resource_overrides.get('NUM_GEN_NODES')}"
        if expected_gpu
        else "no rollout GPU override found; launcher default will be used",
        path=str(args.resource_profile_env),
        parsed=resource_env,
        overrides=resource_overrides,
    )
    validator = check_rollout_validator(checks, args, train_data_path, val_data_path)
    add(
        checks,
        "paths",
        "rollout container",
        "pass" if args.container.exists() and args.container.is_file() else "fail",
        "rollout launcher container image is visible"
        if args.container.exists() and args.container.is_file()
        else "rollout launcher container image is missing",
        path=str(args.container),
    )
    wrapper = check_capture_wrapper_dry_run(
        checks,
        args,
        rollout_log_dir,
        output_conversations,
        train_data_path,
        val_data_path,
    )
    launcher = check_run_grpo_dry_run(checks, args, resource_overrides, train_data_path, val_data_path)

    status = overall_status(checks)
    submit_env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "SWE_REPO_ROOT": str(args.repo_root),
        "REPO_ROOT": str(args.repo_root),
        "CONFIG_FILE": str(args.config),
        "ENV_FILE": str(args.env_file),
        "CHAT_TEMPLATE": str(args.chat_template),
        "RESOURCE_PROFILE_ENV": str(args.resource_profile_env),
        "ROLLOUT_LOG_DIR": str(rollout_log_dir),
        "OUTPUT_CONVERSATIONS": str(output_conversations),
        "DRY_RUN": "false",
        "MAX_NUM_STEPS": str(args.max_num_steps),
        "WANDB_NAME": args.wandb_name,
        "EXP_SUFFIX_OVERRIDE": args.wandb_name,
        "CHECKPOINT_SUBDIR": args.wandb_name,
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "CONTAINER": str(args.container),
    }
    submit_env.update(runtime_passthrough_env(args.artifact_root))
    if train_data_path:
        submit_env["TRAIN_DATA_PATH"] = str(train_data_path)
    if val_data_path:
        submit_env["VAL_DATA_PATH"] = str(val_data_path)
    commands = {
        "submit": shell_join(submit_env, ["bash", "experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"]),
        "dry_run": shell_join({**submit_env, "DRY_RUN": "true"}, ["bash", "experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"]),
        "analyze_job": (
            "python3 experiments/eagle3_qwen3_235b/analyze_rollout_capture_job.py "
            f"--artifact-root {shlex.quote(str(args.artifact_root))} "
            f"--repo-root {shlex.quote(str(args.repo_root))} "
            f"--rollout-log-dir {shlex.quote(str(rollout_log_dir))} "
            f"--output-data {shlex.quote(str(output_conversations))}"
        ),
    }
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "submit_ready": status in {"pass", "warn"} and not any(check["status"] == "fail" for check in checks),
        "artifact_root": str(args.artifact_root),
        "repo_root": str(args.repo_root),
        "config": str(args.config),
        "chat_template": str(args.chat_template),
        "resource_profile_env": str(args.resource_profile_env),
        "container": str(args.container),
        "wandb_name": args.wandb_name,
        "train_data_path": str(train_data_path) if train_data_path else None,
        "val_data_path": str(val_data_path) if val_data_path else None,
        "data_path_source": data_path_source,
        "runtime_passthrough_env": runtime_passthrough_env(args.artifact_root),
        "source_vllm_env": source_vllm_env,
        "resource_profile": resource_env,
        "resource_overrides": resource_overrides,
        "rollout_log_dir": str(rollout_log_dir),
        "output_conversations": str(output_conversations),
        "commands": commands,
        "checks": checks,
        "validator": validator,
        "wrapper_dry_run": wrapper,
        "run_grpo_dry_run": launcher,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(data))
    print(render_markdown(data))
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
