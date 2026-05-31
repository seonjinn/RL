#!/usr/bin/env python3
"""Validate that a NeMo-RL run can produce Eagle3 training conversations.

This is a no-submit, no-GPU gate. It checks the target Qwen3 RL config, the
SpecDec-RL logging source, and the local normalizer so that the final
`train_data_step*.jsonl` files can be converted into ModelOpt Eagle3
conversation JSONL.
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
DEFAULT_CONFIG = ROOT / "grpo_qwen3_235b_swe.yaml"
DEFAULT_SPECDEC_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)
DEFAULT_ARTIFACT_ROOT = ROOT / "outputs/qwen3_235b_eagle3"
DEFAULT_PATCH_FILE = EXP / "specdec_rl_rollout_role_logging.patch"
REQUIRED_SWE_METADATA_KEYS = (
    "problem_statement",
    "instance_id",
    "base_commit",
    "dataset_name",
    "split",
    "instance_dict",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("NEMO_RL_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--specdec-rl-dir", type=Path, default=Path(os.environ.get("SPECDEC_RL_DIR", DEFAULT_SPECDEC_RL_DIR)))
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
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
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(os.environ["CHAT_TEMPLATE"]) if os.environ.get("CHAT_TEMPLATE") else None,
        help="Effective Qwen3 chat template override for the rollout run.",
    )
    parser.add_argument("--patch-file", type=Path, default=Path(os.environ.get("ROLLOUT_ROLE_PATCH", DEFAULT_PATCH_FILE)))
    parser.add_argument("--require-role-logging", action="store_true")
    parser.add_argument("--env-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to parse {path}: {exc}") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def nested_get(value: Any, keys: list[str], default: Any = None) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def rollout_globs(log_dir: str | None) -> dict[str, str | None]:
    if not log_dir:
        return {"train_data_glob": None, "val_data_glob": None}
    base = Path(log_dir)
    return {
        "train_data_glob": str(base / "train_data_step*.jsonl"),
        "val_data_glob": str(base / "val_data_step*.jsonl"),
    }


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


def validate_config(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {"config": str(args.config)}
    if not args.config.exists():
        add(checks, "config", "NeMo-RL config", "fail", f"config not visible: {args.config}")
        return data
    try:
        config = load_yaml(args.config)
    except Exception as exc:
        add(checks, "config", "NeMo-RL config", "fail", f"cannot parse config: {exc}", path=str(args.config))
        return data

    should_use_nemo_gym = bool(nested_get(config, ["env", "should_use_nemo_gym"], False))
    should_log_full = bool(nested_get(config, ["env", "should_log_nemo_gym_responses"], False))
    async_grpo = bool(nested_get(config, ["grpo", "async_grpo", "enabled"], False))
    backend = nested_get(config, ["policy", "generation", "backend"])
    async_engine = nested_get(config, ["policy", "generation", "vllm_cfg", "async_engine"])
    expose_http = nested_get(config, ["policy", "generation", "vllm_cfg", "expose_http_server"])
    configured_chat_template = nested_get(
        config,
        ["policy", "generation", "vllm_cfg", "http_server_serving_chat_kwargs", "chat_template"],
    )
    log_dir = nested_get(config, ["logger", "log_dir"])
    effective_chat_template = str(args.chat_template) if args.chat_template else configured_chat_template
    configured_train_data_path = nested_get(config, ["data", "train", "data_path"])
    configured_val_data_path = nested_get(config, ["data", "validation", "data_path"])
    effective_train_data_path = args.train_data_path or (
        Path(configured_train_data_path) if isinstance(configured_train_data_path, str) else None
    )
    effective_val_data_path = args.val_data_path or (
        Path(configured_val_data_path) if isinstance(configured_val_data_path, str) else effective_train_data_path
    )

    data.update(
        {
            "should_use_nemo_gym": should_use_nemo_gym,
            "should_log_nemo_gym_responses": should_log_full,
            "async_grpo": async_grpo,
            "generation_backend": backend,
            "vllm_async_engine": async_engine,
            "vllm_expose_http_server": expose_http,
            "configured_chat_template": configured_chat_template,
            "chat_template_override": str(args.chat_template) if args.chat_template else None,
            "effective_chat_template": effective_chat_template,
            "configured_train_data_path": configured_train_data_path,
            "configured_val_data_path": configured_val_data_path,
            "train_data_path_override": str(args.train_data_path) if args.train_data_path else None,
            "val_data_path_override": str(args.val_data_path) if args.val_data_path else None,
            "effective_train_data_path": str(effective_train_data_path) if effective_train_data_path else None,
            "effective_val_data_path": str(effective_val_data_path) if effective_val_data_path else None,
            "log_dir": log_dir,
        }
    )
    data.update(rollout_globs(log_dir if isinstance(log_dir, str) else None))

    add(
        checks,
        "config",
        "NeMo-Gym rollout path",
        "pass" if should_use_nemo_gym else "warn",
        "env.should_use_nemo_gym is enabled" if should_use_nemo_gym else "env.should_use_nemo_gym is not enabled",
        should_use_nemo_gym=should_use_nemo_gym,
    )
    add(
        checks,
        "config",
        "compact train_data JSONL",
        "pass" if not should_log_full else "warn",
        "env.should_log_nemo_gym_responses=false keeps compact train_data_step*.jsonl logging in SpecDec-RL"
        if not should_log_full
        else "env.should_log_nemo_gym_responses=true may keep full_result metrics and skip compact train_data JSONL in the sync path",
        should_log_nemo_gym_responses=should_log_full,
    )
    add(
        checks,
        "config",
        "generation backend",
        "pass" if backend == "vllm" else "fail",
        "policy.generation.backend is vllm" if backend == "vllm" else "policy.generation.backend is not vllm",
        backend=backend,
    )
    add(
        checks,
        "config",
        "async engine",
        "pass" if async_engine is True else "warn",
        "vLLM async engine is enabled" if async_engine is True else "vLLM async engine is not explicitly enabled",
        async_engine=async_engine,
    )
    add(
        checks,
        "config",
        "HTTP server exposure",
        "pass" if (not should_use_nemo_gym or expose_http is True) else "fail",
        "vLLM HTTP server exposure is compatible with NeMo-Gym"
        if (not should_use_nemo_gym or expose_http is True)
        else "NeMo-Gym requires policy.generation.vllm_cfg.expose_http_server=true",
        expose_http_server=expose_http,
    )
    chat_path = Path(effective_chat_template) if isinstance(effective_chat_template, str) and effective_chat_template else None
    chat_exists = chat_path.exists() if chat_path else False
    add(
        checks,
        "config",
        "Qwen3 chat template",
        "pass" if chat_path and chat_exists else "fail",
        f"effective chat template is visible: {chat_path}"
        if chat_path and chat_exists
        else (
            f"effective chat template is not visible: {chat_path}"
            if chat_path
            else "no effective chat template configured"
        ),
        configured_chat_template=configured_chat_template,
        chat_template_override=str(args.chat_template) if args.chat_template else None,
        effective_chat_template=str(chat_path) if chat_path else None,
    )
    add(
        checks,
        "config",
        "logger.log_dir",
        "pass" if isinstance(log_dir, str) and bool(log_dir) else "fail",
        f"train/val JSONL will be written under {log_dir}" if isinstance(log_dir, str) and log_dir else "logger.log_dir is missing",
        log_dir=log_dir,
    )
    for label, path in (("train", effective_train_data_path), ("validation", effective_val_data_path)):
        probe = probe_nemogym_jsonl(path)
        add(
            checks,
            "data",
            f"{label} NemoGym JSONL",
            "pass" if probe["ok"] else "fail",
            probe["detail"],
            path=str(path) if path else None,
            **{key: value for key, value in probe.items() if key not in {"ok", "detail"}},
        )
    add(
        checks,
        "config",
        "async GRPO",
        "pass" if async_grpo else "warn",
        "Qwen3 SWE recipe uses async GRPO" if async_grpo else "async GRPO is not enabled",
        async_grpo=async_grpo,
    )
    return data


def validate_source(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    root = args.specdec_rl_dir
    grpo = root / "nemo_rl/algorithms/grpo.py"
    logger = root / "nemo_rl/utils/logger.py"
    data = {"specdec_rl_dir": str(root), "grpo_py": str(grpo), "logger_py": str(logger)}

    if not root.exists():
        add(checks, "source", "SpecDec-RL checkout", "warn", f"SpecDec-RL checkout not visible: {root}")
        return data
    if not grpo.exists() or not logger.exists():
        add(
            checks,
            "source",
            "rollout logging source files",
            "warn",
            "expected grpo.py/logger.py not visible",
            grpo_exists=grpo.exists(),
            logger_exists=logger.exists(),
        )
        return data

    grpo_text = grpo.read_text(encoding="utf-8", errors="replace")
    logger_text = logger.read_text(encoding="utf-8", errors="replace")
    train_jsonl_ok = "train_data_step" in grpo_text and "log_batched_dict_as_jsonl" in grpo_text
    compact_guard_ok = "if not _should_log_nemo_gym_responses(master_config)" in grpo_text
    role_logging_ok = (
        'log_data["role"]' in grpo_text
        or '"role": flat_messages' in grpo_text
        or "flat_messages_role" in grpo_text
        or "metrics_logging_data[\"role\"]" in grpo_text
    )
    logger_path_ok = "filepath = os.path.join(self.base_log_dir, filename)" in logger_text
    patch_applicable = None
    patch_output = ""
    if not role_logging_ok and args.patch_file.exists():
        result = subprocess.run(
            ["git", "-C", str(root), "apply", "--check", str(args.patch_file)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        patch_applicable = result.returncode == 0
        patch_output = result.stdout[-2000:]

    data.update(
        {
            "train_jsonl_logging": train_jsonl_ok,
            "compact_guard": compact_guard_ok,
            "role_logging": role_logging_ok,
            "role_logging_patch": str(args.patch_file),
            "role_logging_patch_applicable": patch_applicable,
            "logger_writes_to_base_log_dir": logger_path_ok,
        }
    )
    add(
        checks,
        "source",
        "train_data_step JSONL logging",
        "pass" if train_jsonl_ok else "warn",
        "SpecDec-RL writes train_data_step*.jsonl" if train_jsonl_ok else "could not prove train_data_step*.jsonl logging",
        file=str(grpo),
    )
    if not role_logging_ok:
        if not args.patch_file.exists():
            add(
                checks,
                "source",
                "role logging patch",
                "warn",
                f"role logging patch file is not visible: {args.patch_file}",
            )
        elif patch_applicable:
            add(
                checks,
                "source",
                "role logging patch",
                "pass",
                "role logging patch applies cleanly to the SpecDec-RL checkout",
                patch_file=str(args.patch_file),
            )
        else:
            add(
                checks,
                "source",
                "role logging patch",
                "fail" if args.require_role_logging else "warn",
                "role logging patch did not apply cleanly",
                patch_file=str(args.patch_file),
                output=patch_output,
            )
    add(
        checks,
        "source",
        "compact logging guard",
        "pass" if compact_guard_ok else "warn",
        "compact train_data logging is guarded by should_log_nemo_gym_responses=false"
        if compact_guard_ok
        else "could not prove compact logging guard",
        file=str(grpo),
    )
    add(
        checks,
        "source",
        "role-aware train_data JSONL",
        "pass" if role_logging_ok else ("fail" if args.require_role_logging else "warn"),
        "train_data JSONL includes role arrays"
        if role_logging_ok
        else "train_data JSONL appears to include content only; apply the role logging patch or use lossy --infer-flat-content-roles",
        file=str(grpo),
    )
    add(
        checks,
        "source",
        "logger output path",
        "pass" if logger_path_ok else "warn",
        "logger writes JSONL under logger.log_dir" if logger_path_ok else "could not prove logger JSONL output path",
        file=str(logger),
    )
    return data


def validate_normalizer(checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = EXP / "normalize_rl_rollouts_to_conversations.py"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    flat_role_ok = "FLAT_ROLE_KEYS" in text and "flat_content_messages" in text
    infer_ok = "--infer-flat-content-roles" in text
    add(
        checks,
        "normalizer",
        "flat content+role support",
        "pass" if flat_role_ok else "fail",
        "normalizer can read SpecDec-RL flat content+role arrays"
        if flat_role_ok
        else "normalizer lacks SpecDec-RL flat content+role support",
        file=str(path),
    )
    add(
        checks,
        "normalizer",
        "lossy role inference fallback",
        "pass" if infer_ok else "warn",
        "normalizer has explicit --infer-flat-content-roles fallback"
        if infer_ok
        else "normalizer has no explicit fallback for content-only logs",
        file=str(path),
    )
    return {"normalizer": str(path), "flat_role_support": flat_role_ok, "infer_fallback": infer_ok}


def overall(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "fail" for item in checks):
        return "fail"
    if any(item["status"] in {"warn", "missing"} for item in checks):
        return "warn"
    return "pass"


def recommended_outputs(args: argparse.Namespace, config_data: dict[str, Any], source_data: dict[str, Any]) -> dict[str, Any]:
    output = args.output_conversations or args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"
    train_glob = config_data.get("train_data_glob") or "<logger.log_dir>/train_data_step*.jsonl"
    normalizer_args = ["--include-metadata"]
    if not source_data.get("role_logging"):
        normalizer_args.append("--infer-flat-content-roles")
    normalize_command = (
        "python3 experiments/eagle3_qwen3_235b/normalize_rl_rollouts_to_conversations.py "
        f"--input {shell_quote(str(train_glob))} "
        f"--output {shell_quote(str(output))} "
        f"{' '.join(normalizer_args)}"
    )
    validate_command = (
        "python3 experiments/eagle3_qwen3_235b/validate_training_conversations.py "
        f"{shell_quote(str(output))} --max-seq-len 16384"
    )
    hydra_overrides: list[str] = []
    if args.train_data_path:
        hydra_overrides.append(f"data.train.data_path={args.train_data_path}")
    if args.val_data_path:
        hydra_overrides.append(f"data.validation.data_path={args.val_data_path}")
    if config_data.get("should_log_nemo_gym_responses"):
        hydra_overrides.append("env.should_log_nemo_gym_responses=false")
    if not config_data.get("log_dir"):
        hydra_overrides.append(f"logger.log_dir={args.artifact_root / 'rl_logs'}")
    configured_chat_template = config_data.get("configured_chat_template")
    configured_chat_template_exists = (
        Path(configured_chat_template).exists()
        if isinstance(configured_chat_template, str) and configured_chat_template
        else False
    )
    if args.chat_template:
        hydra_overrides.append(
            "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="
            f"{args.chat_template}"
        )
    elif not configured_chat_template_exists:
        default_chat_template = args.artifact_root / "templates/qwen3_generation_template.jinja2"
        if default_chat_template.exists():
            hydra_overrides.append(
                "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="
                f"{default_chat_template}"
            )
    return {
        "output_conversations": str(output),
        "normalizer_args": normalizer_args,
        "normalize_command": normalize_command,
        "validate_command": validate_command,
        "recommended_hydra_overrides": hydra_overrides,
        "role_logging_patch": str(args.patch_file),
        "role_logging_patch_applicable": source_data.get("role_logging_patch_applicable"),
    }


def render_env(data: dict[str, Any]) -> str:
    rec = data["recommendation"]
    lines = [
        f"export RL_LOG_DIR={shell_quote(str(data['config_data'].get('log_dir') or ''))}",
        f"export RL_TRAIN_DATA_PATH={shell_quote(str(data['config_data'].get('effective_train_data_path') or ''))}",
        f"export RL_VAL_DATA_PATH={shell_quote(str(data['config_data'].get('effective_val_data_path') or ''))}",
        f"export RL_TRAIN_DATA_GLOB={shell_quote(str(data['config_data'].get('train_data_glob') or ''))}",
        f"export RL_VAL_DATA_GLOB={shell_quote(str(data['config_data'].get('val_data_glob') or ''))}",
        f"export RL_OUTPUT_CONVERSATIONS={shell_quote(rec['output_conversations'])}",
        f"export RL_NORMALIZER_ARGS={shell_quote(' '.join(rec['normalizer_args']))}",
    ]
    if rec["recommended_hydra_overrides"]:
        lines.append(f"export RL_CAPTURE_HYDRA_OVERRIDES={shell_quote(' '.join(rec['recommended_hydra_overrides']))}")
    return "\n".join(lines) + "\n"


def render_markdown(data: dict[str, Any]) -> str:
    rec = data["recommendation"]
    lines = [
        "# RL Rollout Capture Validation",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        "Recommended commands:",
        "",
        "```bash",
        rec["normalize_command"],
        rec["validate_command"],
        "```",
        "",
        f"Role logging patch: `{rec['role_logging_patch']}`",
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
    checks: list[dict[str, Any]] = []
    config_data = validate_config(checks, args)
    source_data = validate_source(checks, args)
    normalizer_data = validate_normalizer(checks)
    recommendation = recommended_outputs(args, config_data, source_data)
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall(checks),
        "config_data": config_data,
        "source_data": source_data,
        "normalizer_data": normalizer_data,
        "recommendation": recommendation,
        "checks": checks,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(data))
    if args.env_out:
        args.env_out.parent.mkdir(parents=True, exist_ok=True)
        args.env_out.write_text(render_env(data))

    print(json.dumps(data, indent=2, sort_keys=True))
    return 1 if data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
