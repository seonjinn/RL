#!/usr/bin/env python3
"""Validate Eagle3 ModelOpt recipe overrides before launching training.

The training wrapper ultimately calls ModelOpt's recipe loader with OmegaConf
dotlist overrides. A shell dry-run can prove the command is assembled, but it
does not prove that the override keys are valid or that critical Eagle3
architecture fields match the selected verifier model.

This script performs a dependency-light static validation everywhere. When the
intended ModelOpt Python environment is available, it also imports
``modelopt.recipe.load_recipe`` and validates the dotlist against the real
Pydantic schema.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"
DEFAULT_WRAPPER = EXP / "modelopt_qwen3_235b_offline_train.sh"
DEFAULT_MODELOPT = Path(os.environ.get("MODELOPT_DIR", ROOT / "Model-Optimizer"))
DEFAULT_RECIPE = DEFAULT_MODELOPT / "modelopt_recipes/general/speculative_decoding/eagle3.yaml"
DEFAULT_REFERENCE_ARCH = EXP / "qwen3_235b_thinking_eagle3_architecture.json"
TrainingMode = str

REQUIRED_TOP_LEVEL = {"model", "data", "training", "eagle"}
EXPECTED_ARCH_KEYS = [
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "use_aux_hidden_state",
    "use_input_layernorm_in_first_layer",
    "use_last_layernorm",
    "eagle_aux_hidden_state_layer_ids",
]
CHECKS: list[dict[str, str]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--modelopt-dir", type=Path, default=DEFAULT_MODELOPT)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--reference-arch",
        type=Path,
        default=DEFAULT_REFERENCE_ARCH,
        help=(
            "Architecture JSON to compare wrapper overrides against. Use "
            "derive_eagle3_architecture.py to create this for non-Qwen3 models."
        ),
    )
    parser.add_argument(
        "--training-mode",
        choices=("auto", "offline", "online"),
        default="auto",
        help="Validate mode-specific data overrides. Auto derives the mode from wrapper output.",
    )
    parser.add_argument(
        "--require-modelopt-import",
        action="store_true",
        help="Fail if modelopt.recipe.load_recipe cannot be imported and executed.",
    )
    parser.add_argument(
        "--allow-modelopt-metadata-shim",
        action="store_true",
        help=(
            "Patch importlib.metadata.version('nvidia-modelopt') to 0+local before import. "
            "Useful for source checkouts that are not pip-installed."
        ),
    )
    parser.add_argument("--json-out", type=Path, help="Optional structured validation report path.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown validation report path.")
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


def status_counts(checks: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = check.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_payload(
    args: argparse.Namespace,
    failures: list[str],
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    mode: TrainingMode | None = None,
    expected_arch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_overrides = overrides or {}
    arch_prefix = "eagle.eagle_architecture_config."
    arch_overrides = {
        key.removeprefix(arch_prefix): value
        for key, value in parsed_overrides.items()
        if key.startswith(arch_prefix)
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "fail" if failures else "pass",
        "wrapper": str(args.wrapper),
        "modelopt_dir": str(args.modelopt_dir),
        "recipe_arg": str(args.recipe),
        "recipe_config": str(config_path) if config_path else None,
        "reference_arch": str(args.reference_arch),
        "training_mode": mode or args.training_mode,
        "override_count": len(parsed_overrides),
        "override_keys": sorted(parsed_overrides),
        "architecture_overrides": arch_overrides,
        "expected_architecture": expected_arch or {},
        "counts": status_counts(CHECKS),
        "checks": CHECKS,
        "warnings": [check["detail"] for check in CHECKS if check.get("status") == "warn"],
        "failures": failures,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ModelOpt Eagle3 Recipe Override Validation",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        f"Generated: `{payload['generated_at']}`",
        f"Wrapper: `{payload['wrapper']}`",
        f"Training mode: `{payload['training_mode']}`",
        f"Recipe config: `{payload.get('recipe_config') or '-'}`",
        "",
        "| status | detail |",
        "| --- | --- |",
    ]
    for check in payload["checks"]:
        detail = str(check.get("detail") or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {str(check.get('status') or 'unknown').upper()} | {detail} |")
    if payload.get("failures"):
        lines += ["", "## Failures", ""]
        for item in payload["failures"]:
            lines.append(f"- {str(item).replace(chr(10), ' ')}")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(payload), encoding="utf-8")


def parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    if text == "{}":
        return {}
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_expected_arch(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arch = payload.get("eagle_architecture_config", payload)
    return {key: arch[key] for key in EXPECTED_ARCH_KEYS if key in arch}


def collect_wrapper_overrides(wrapper: Path, modelopt_dir: Path) -> tuple[Path, list[str]]:
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "INPUT_DATA": os.environ.get("INPUT_DATA", "/tmp/conversations.jsonl"),
        "HIDDEN_STATES_DIR": os.environ.get("HIDDEN_STATES_DIR", "/tmp/hiddens"),
        "OUTPUT_DIR": os.environ.get("OUTPUT_DIR", "/tmp/modelopt_ckpt"),
        "CHAT_TEMPLATE": os.environ.get("CHAT_TEMPLATE") or "/tmp/qwen3_generation_template.jinja2",
        "MODELOPT_DIR": str(modelopt_dir),
    }
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wrapper dry-run failed:\n{result.stdout}")
    command_line = next(
        (line for line in result.stdout.splitlines() if line.strip().startswith("./launch_train.sh")),
        "",
    )
    if not command_line:
        raise RuntimeError(f"could not find launch_train command in wrapper output:\n{result.stdout}")

    tokens = shlex.split(command_line)
    config_path: Path | None = None
    overrides: list[str] = []
    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--config":
            config_path = Path(tokens[idx + 1])
            idx += 2
            continue
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
            idx += 1
            continue
        if token in {"--num_nodes", "--head_node_ip"}:
            idx += 2
            continue
        if token.startswith("--num_nodes=") or token.startswith("--head_node_ip="):
            idx += 1
            continue
        overrides.append(token)
        idx += 1

    if config_path is None:
        raise RuntimeError(f"--config missing in wrapper command: {command_line}")
    if not config_path.is_absolute():
        config_path = modelopt_dir / "examples/speculative_decoding" / config_path
    return config_path.resolve(), overrides


def derive_training_mode(overrides: dict[str, Any], requested: TrainingMode) -> TrainingMode:
    if requested != "auto":
        return requested
    if overrides.get("data.offline_data_path") not in {"", None}:
        return "offline"
    if overrides.get("data.data_path") not in {"", None}:
        return "online"
    raise ValueError("could not derive training mode from wrapper overrides")


def override_map(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override lacks '=': {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = parse_scalar(value)
    return parsed


def static_validate(
    config_path: Path,
    overrides: dict[str, Any],
    mode: TrainingMode,
    expected_arch: dict[str, Any],
    failures: list[str],
) -> None:
    if not config_path.exists():
        fail(f"recipe config does not exist: {config_path}", failures)
    else:
        text = config_path.read_text(errors="ignore")
        if "recipe_type: speculative_eagle" not in text:
            fail(f"recipe is not marked speculative_eagle: {config_path}", failures)
        if "eagle_architecture_config" not in text:
            fail(f"recipe lacks eagle_architecture_config section: {config_path}", failures)
        ok(f"recipe file found: {config_path.relative_to(ROOT)}")

    for key in overrides:
        top = key.split(".", 1)[0]
        if top not in REQUIRED_TOP_LEVEL:
            fail(f"unsupported top-level override namespace {top!r} in {key!r}", failures)

    required_overrides = {
        "model.model_name_or_path",
        "training.output_dir",
        "training.training_seq_len",
        "training.answer_only_loss",
        "eagle.eagle_decoder_type",
        "eagle.eagle_ttt_steps",
        "eagle.eagle_architecture_config.use_aux_hidden_state",
        "eagle.eagle_architecture_config.eagle_aux_hidden_state_layer_ids",
    }
    for key in sorted(required_overrides):
        if key not in overrides:
            fail(f"missing required override: {key}", failures)

    if mode == "offline":
        if "data.offline_data_path" not in overrides:
            fail("missing required offline override: data.offline_data_path", failures)
        if overrides.get("data.offline_data_path") in {"", None}:
            fail("data.offline_data_path is empty", failures)
        else:
            ok("offline hidden-state path override present")
        if "data.data_path" in overrides:
            warn("offline wrapper also overrides data.data_path; ModelOpt will still derive offline mode")
        if overrides.get("model.use_fake_base_for_offline") is not True:
            warn(
                "offline wrapper does not enable model.use_fake_base_for_offline; "
                "235B training may spend extra memory/loading time on base embeddings/lm_head"
            )
        else:
            ok("offline fake-base loading enabled")
    elif mode == "online":
        if "data.data_path" not in overrides:
            fail("missing required online override: data.data_path", failures)
        if overrides.get("data.data_path") in {"", None}:
            fail("data.data_path is empty", failures)
        else:
            ok("online conversation data path override present")
        if overrides.get("data.offline_data_path") not in {None, ""}:
            fail("online wrapper must not set data.offline_data_path", failures)
        if overrides.get("training.answer_only_loss") is True:
            if overrides.get("data.chat_template") in {"", None}:
                fail("online answer-only loss requires data.chat_template", failures)
            else:
                ok("online answer-only chat template override present")
    else:
        fail(f"unsupported training mode: {mode}", failures)

    for key, expected in expected_arch.items():
        override_key = f"eagle.eagle_architecture_config.{key}"
        actual = overrides.get(override_key)
        if actual != expected:
            fail(f"{override_key}={actual!r}, expected {expected!r}", failures)

    if overrides.get("eagle.eagle_export_rope_scaling") != {}:
        warn("eagle_export_rope_scaling is not disabled; verify long-context export intent")
    else:
        ok("export rope scaling injection disabled for Thinking-2507 draft")

    if overrides.get("training.answer_only_loss") is not True:
        fail("training.answer_only_loss must be true for RL/SWE assistant-token loss", failures)
    else:
        ok("answer-only loss enabled")


def try_modelopt_validation(
    modelopt_dir: Path,
    config_path: Path,
    overrides: list[str],
    mode: TrainingMode,
    expected_arch: dict[str, Any],
    require: bool,
    metadata_shim: bool,
    failures: list[str],
) -> None:
    if metadata_shim:
        real_version = importlib.metadata.version

        def patched_version(name: str) -> str:
            if name == "nvidia-modelopt":
                return "0+local"
            return real_version(name)

        importlib.metadata.version = patched_version

    sys.path.insert(0, str(modelopt_dir))
    try:
        from modelopt.recipe import load_recipe

        recipe = load_recipe(str(config_path), overrides=overrides)
    except Exception as exc:
        message = f"ModelOpt load_recipe validation unavailable/failed: {exc}"
        if require:
            fail(message, failures)
        else:
            warn(message)
            warn("Run this validator inside the training container with --require-modelopt-import.")
        return

    arch = recipe.eagle.eagle_architecture_config
    expected_offline = mode == "offline"
    if recipe.eagle.eagle_offline is not expected_offline:
        fail(
            f"ModelOpt recipe derived eagle_offline={recipe.eagle.eagle_offline}, "
            f"expected {expected_offline} for {mode} training",
            failures,
        )
    if arch.get("use_aux_hidden_state") is not True:
        fail("ModelOpt recipe has use_aux_hidden_state != true", failures)
    expected_aux_layers = expected_arch.get("eagle_aux_hidden_state_layer_ids")
    if expected_aux_layers is not None and arch.get("eagle_aux_hidden_state_layer_ids") != expected_aux_layers:
        fail(
            "ModelOpt recipe has wrong aux layer ids: "
            f"{arch.get('eagle_aux_hidden_state_layer_ids')!r}",
            failures,
        )
    ok("ModelOpt load_recipe accepted wrapper overrides")


def main() -> int:
    args = parse_args()
    CHECKS.clear()
    failures: list[str] = []
    config_path: Path | None = None
    parsed: dict[str, Any] | None = None
    mode: TrainingMode | None = None
    expected_arch: dict[str, Any] | None = None

    if not args.wrapper.exists():
        fail(f"missing wrapper: {args.wrapper}", failures)
    if not args.modelopt_dir.exists():
        fail(f"missing ModelOpt dir: {args.modelopt_dir}", failures)
    if not args.reference_arch.exists():
        fail(f"missing reference architecture: {args.reference_arch}", failures)
    if failures:
        write_outputs(args, build_payload(args, failures))
        return 1

    try:
        expected_arch = load_expected_arch(args.reference_arch)
        config_path, override_items = collect_wrapper_overrides(args.wrapper, args.modelopt_dir)
        parsed = override_map(override_items)
        mode = derive_training_mode(parsed, args.training_mode)
    except Exception as exc:
        fail(str(exc), failures)
        write_outputs(
            args,
            build_payload(args, failures, config_path=config_path, overrides=parsed, mode=mode, expected_arch=expected_arch),
        )
        return 1

    ok(f"collected {len(override_items)} ModelOpt overrides from wrapper dry-run")
    ok(f"training mode: {mode}")
    ok(f"loaded {len(expected_arch)} expected Eagle3 arch fields from {args.reference_arch}")
    static_validate(config_path, parsed, mode, expected_arch, failures)
    try_modelopt_validation(
        args.modelopt_dir,
        config_path,
        override_items,
        mode,
        expected_arch,
        args.require_modelopt_import,
        args.allow_modelopt_metadata_shim,
        failures,
    )

    if failures:
        print("\nRecipe override validation failed:")
        for item in failures:
            print(f"- {item}")
        write_outputs(
            args,
            build_payload(args, failures, config_path=config_path, overrides=parsed, mode=mode, expected_arch=expected_arch),
        )
        return 1
    print("\nRecipe override validation passed.")
    write_outputs(
        args,
        build_payload(args, failures, config_path=config_path, overrides=parsed, mode=mode, expected_arch=expected_arch),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
