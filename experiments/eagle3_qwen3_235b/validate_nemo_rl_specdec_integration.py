#!/usr/bin/env python3
"""Validate NeMo-RL/vLLM speculative decoding integration overrides.

This is a configuration-side gate for the RL validation phase. It does not
launch vLLM or submit Slurm jobs; it proves that the target NeMo-RL config has a
vLLM generation section, that the Eagle3 draft override lands under
policy.generation.vllm_kwargs.speculative_config, and that the referenced
SpecDec-RL checkout contains the expected speculative decoding load-format and
metric plumbing.

Two RL integration modes are supported:

- generation-only: vLLM uses a fixed exported Eagle3 draft model during rollout.
- online-draft-training: NeMo-RL owns the draft model during policy training and
  refits policy + draft into vLLM. This needs Megatron, DTensor disabled, and
  sequence packing disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "grpo_qwen3_235b_swe.yaml"
DEFAULT_SPECDEC_RL_DIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("NEMO_RL_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--draft-model", default=os.environ.get("EAGLE3_DRAFT_MODEL", "nvidia/Qwen3-235B-A22B-Eagle3"))
    parser.add_argument("--num-speculative-tokens", type=int, default=int(os.environ.get("EAGLE3_NUM_SPEC_TOKENS", "3")))
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=int(os.environ.get("EAGLE3_DRAFT_TP", "1")))
    parser.add_argument("--method", default=os.environ.get("EAGLE3_METHOD", "eagle3"))
    parser.add_argument(
        "--integration-mode",
        choices=("generation-only", "online-draft-training"),
        default=os.environ.get("EAGLE3_INTEGRATION_MODE", "generation-only"),
    )
    parser.add_argument("--draft-loss-weight", type=float, default=float(os.environ.get("EAGLE3_DRAFT_LOSS_WEIGHT", "1.0")))
    parser.add_argument("--specdec-rl-dir", type=Path, default=Path(os.environ.get("SPECDEC_RL_DIR", DEFAULT_SPECDEC_RL_DIR)))
    parser.add_argument("--allow-existing-speculative-config", action="store_true")
    parser.add_argument("--require-draft-files", action="store_true")
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


def hydra_overrides(args: argparse.Namespace) -> list[str]:
    prefix = "++policy.generation.vllm_kwargs.speculative_config"
    overrides = [
        f"{prefix}.method={args.method}",
        f"{prefix}.model={args.draft_model}",
        f"{prefix}.num_speculative_tokens={args.num_speculative_tokens}",
        f"{prefix}.draft_tensor_parallel_size={args.draft_tensor_parallel_size}",
    ]
    if args.integration_mode == "online-draft-training":
        overrides.extend(
            [
                "policy.megatron_cfg.enabled=true",
                "policy.dtensor_cfg.enabled=false",
                "policy.sequence_packing.enabled=false",
                "++policy.draft.enabled=true",
                f"++policy.draft.model_name={args.draft_model}",
                f"++policy.draft.loss_weight={args.draft_loss_weight}",
            ]
        )
    return overrides


def shell_override(overrides: list[str]) -> str:
    return " ".join(shell_quote(item) for item in overrides)


def validate_config(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.config.exists():
        add(checks, "config", "NeMo-RL config", "fail", f"config not visible: {args.config}")
        return None
    try:
        config = load_yaml(args.config)
    except Exception as exc:
        add(checks, "config", "NeMo-RL config", "fail", f"cannot parse config: {exc}", path=str(args.config))
        return None

    generation = nested_get(config, ["policy", "generation"], {})
    if not isinstance(generation, dict):
        add(checks, "config", "generation section", "fail", "policy.generation is missing or not a map")
        return config

    backend = generation.get("backend")
    add(
        checks,
        "config",
        "generation backend",
        "pass" if backend == "vllm" else "fail",
        "policy.generation.backend is vllm" if backend == "vllm" else "policy.generation.backend is not vllm",
        backend=backend,
    )

    vllm_cfg = generation.get("vllm_cfg")
    add(
        checks,
        "config",
        "vllm_cfg",
        "pass" if isinstance(vllm_cfg, dict) else "fail",
        "policy.generation.vllm_cfg is present" if isinstance(vllm_cfg, dict) else "policy.generation.vllm_cfg is missing",
    )
    if isinstance(vllm_cfg, dict):
        add(
            checks,
            "config",
            "async engine",
            "pass" if vllm_cfg.get("async_engine") is True else "warn",
            "vLLM async engine is enabled" if vllm_cfg.get("async_engine") is True else "vLLM async engine is not explicitly enabled",
            async_engine=vllm_cfg.get("async_engine"),
        )
        add(
            checks,
            "config",
            "metrics logger",
            "pass" if vllm_cfg.get("enable_vllm_metrics_logger") is True else "warn",
            "vLLM metrics logger is enabled" if vllm_cfg.get("enable_vllm_metrics_logger") is True else "vLLM metrics logger is not enabled",
            enable_vllm_metrics_logger=vllm_cfg.get("enable_vllm_metrics_logger"),
        )

    if args.integration_mode == "online-draft-training":
        megatron_enabled = nested_get(config, ["policy", "megatron_cfg", "enabled"])
        dtensor_enabled = nested_get(config, ["policy", "dtensor_cfg", "enabled"])
        sequence_packing_enabled = nested_get(config, ["policy", "sequence_packing", "enabled"])
        add(
            checks,
            "online_draft",
            "Megatron backend",
            "pass" if megatron_enabled is True else "fail",
            "online draft training can use the Megatron policy backend"
            if megatron_enabled is True
            else "online draft training requires policy.megatron_cfg.enabled=true",
            current_value=megatron_enabled,
        )
        add(
            checks,
            "online_draft",
            "DTensor disabled",
            "pass" if dtensor_enabled is False else "fail",
            "online draft training has DTensor disabled"
            if dtensor_enabled is False
            else "online draft training requires policy.dtensor_cfg.enabled=false",
            current_value=dtensor_enabled,
        )
        add(
            checks,
            "online_draft",
            "sequence packing disabled",
            "warn" if sequence_packing_enabled is True else "pass",
            "current config enables sequence packing; generated online overrides disable it"
            if sequence_packing_enabled is True
            else "sequence packing is already disabled or absent",
            current_value=sequence_packing_enabled,
            planned_override="policy.sequence_packing.enabled=false",
        )

    vllm_kwargs = generation.get("vllm_kwargs")
    add(
        checks,
        "config",
        "vllm_kwargs",
        "pass" if isinstance(vllm_kwargs, dict) else "fail",
        "policy.generation.vllm_kwargs is present" if isinstance(vllm_kwargs, dict) else "policy.generation.vllm_kwargs is missing",
    )
    existing_spec = vllm_kwargs.get("speculative_config") if isinstance(vllm_kwargs, dict) else None
    if existing_spec and not args.allow_existing_speculative_config:
        add(
            checks,
            "config",
            "existing speculative_config",
            "warn",
            "config already contains speculative_config; Hydra ++ overrides may need explicit replacement semantics",
            speculative_config=existing_spec,
        )
    else:
        add(
            checks,
            "config",
            "speculative override target",
            "pass",
            "speculative_config can be added under policy.generation.vllm_kwargs",
        )

    return config


def validate_draft(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    draft = Path(args.draft_model)
    looks_like_path = args.draft_model.startswith("/") or args.draft_model.startswith(".")
    if not looks_like_path:
        add(
            checks,
            "draft",
            "draft model reference",
            "pass",
            "draft model is a model id, not a local path",
            draft_model=args.draft_model,
        )
        return
    config = draft / "config.json"
    safetensors = sorted(draft.glob("*.safetensors")) if draft.exists() else []
    if config.exists() and safetensors:
        add(
            checks,
            "draft",
            "draft model files",
            "pass",
            "vLLM draft config and safetensors weights are visible",
            path=str(draft),
            safetensors=[path.name for path in safetensors[:8]],
        )
    elif args.require_draft_files:
        add(
            checks,
            "draft",
            "draft model files",
            "fail",
            "local draft path must contain config.json and at least one .safetensors weight",
            path=str(draft),
            config_exists=config.exists(),
            safetensors_count=len(safetensors),
        )
    else:
        add(
            checks,
            "draft",
            "draft model files",
            "warn",
            "local draft files are not complete yet; this is expected before export",
            path=str(draft),
            config_exists=config.exists(),
            safetensors_count=len(safetensors),
        )


def validate_specdec_rl_source(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    root = args.specdec_rl_dir
    if not root.exists():
        add(checks, "source", "SpecDec-RL checkout", "warn", f"SpecDec-RL checkout not visible: {root}")
        return

    generation_init = root / "nemo_rl/models/generation/__init__.py"
    vllm_utils = root / "nemo_rl/models/generation/vllm/utils.py"
    vllm_worker = root / "nemo_rl/models/generation/vllm/vllm_worker.py"
    missing = [str(path) for path in (generation_init, vllm_utils, vllm_worker) if not path.exists()]
    if missing:
        add(checks, "source", "SpecDec-RL source files", "warn", "expected source files missing", missing=missing)
        return

    init_text = generation_init.read_text(encoding="utf-8", errors="replace")
    utils_text = vllm_utils.read_text(encoding="utf-8", errors="replace")
    worker_text = vllm_worker.read_text(encoding="utf-8", errors="replace")

    load_format_ok = "speculative_config" in init_text and "load_format" in init_text and '"auto"' in init_text
    metrics_ok = "aggregate_spec_decode_counters" in utils_text and "spec_acceptance_rate" in utils_text
    worker_patch_ok = "speculative decoding post_step" in worker_text and "post_step" in worker_text

    add(
        checks,
        "source",
        "load_format auto hook",
        "pass" if load_format_ok else "warn",
        "SpecDec-RL sets vLLM load_format=auto when speculative_config is present"
        if load_format_ok
        else "could not prove load_format auto hook",
        file=str(generation_init),
    )
    add(
        checks,
        "source",
        "spec decode metrics",
        "pass" if metrics_ok else "warn",
        "SpecDec-RL aggregates spec decode acceptance metrics" if metrics_ok else "could not prove spec decode metric aggregation",
        file=str(vllm_utils),
    )
    add(
        checks,
        "source",
        "vLLM post_step patch",
        "pass" if worker_patch_ok else "warn",
        "SpecDec-RL contains speculative decoding post_step patch" if worker_patch_ok else "could not prove vLLM post_step patch",
        file=str(vllm_worker),
    )

    if args.integration_mode == "online-draft-training":
        policy_init = root / "nemo_rl/models/policy/__init__.py"
        lm_policy = root / "nemo_rl/models/policy/lm_policy.py"
        grpo = root / "nemo_rl/algorithms/grpo.py"
        source_text = ""
        missing_online_files: list[str] = []
        for path in (policy_init, lm_policy, grpo):
            if path.exists():
                source_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
            else:
                missing_online_files.append(str(path))
        online_tokens_ok = (
            "draft" in source_text
            and "loss_weight" in source_text
            and ("policy.draft" in source_text or '"draft"' in source_text or "'draft'" in source_text)
        )
        add(
            checks,
            "source",
            "online draft trainer support",
            "pass" if online_tokens_ok else "fail",
            "SpecDec-RL checkout appears to support trainer-owned draft loss"
            if online_tokens_ok
            else "could not prove policy.draft/loss_weight support in this checkout; use fixed-draft generation or update NeMo-RL before online draft training",
            missing_files=missing_online_files,
            files=[str(policy_init), str(lm_policy), str(grpo)],
        )


def overall(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "fail" for item in checks):
        return "fail"
    if any(item["status"] in {"warn", "missing"} for item in checks):
        return "warn"
    return "pass"


def payload(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    overrides = hydra_overrides(args)
    return {
        "overall_status": overall(checks),
        "config": str(args.config),
        "draft_model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        "method": args.method,
        "integration_mode": args.integration_mode,
        "draft_loss_weight": args.draft_loss_weight,
        "hydra_overrides": overrides,
        "extra_hydra_overrides": shell_override(overrides),
        "checks": checks,
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# NeMo-RL Eagle3 SpecDec Integration Validation",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        f"Integration mode: `{data['integration_mode']}`",
        "",
        "Hydra overrides:",
        "",
        "```bash",
        f"EXTRA_HYDRA_OVERRIDES={shell_quote(data['extra_hydra_overrides'])}",
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


def write_outputs(data: dict[str, Any], args: argparse.Namespace) -> None:
    if args.env_out:
        args.env_out.parent.mkdir(parents=True, exist_ok=True)
        args.env_out.write_text(
            "EXTRA_HYDRA_OVERRIDES="
            + shell_quote(data["extra_hydra_overrides"])
            + "\n"
            + f"EAGLE3_DRAFT_MODEL={shell_quote(args.draft_model)}\n"
            + f"EAGLE3_NUM_SPEC_TOKENS={args.num_speculative_tokens}\n"
            + f"EAGLE3_DRAFT_TP={args.draft_tensor_parallel_size}\n"
            + f"EAGLE3_INTEGRATION_MODE={shell_quote(args.integration_mode)}\n"
            + f"EAGLE3_DRAFT_LOSS_WEIGHT={args.draft_loss_weight}\n"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    validate_config(checks, args)
    validate_draft(checks, args)
    validate_specdec_rl_source(checks, args)
    data = payload(checks, args)
    write_outputs(data, args)
    return 1 if data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
