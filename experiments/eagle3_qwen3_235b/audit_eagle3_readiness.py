#!/usr/bin/env python3
"""Audit readiness of the Qwen3-235B Eagle3 draft-model pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preflight_eagle3_pipeline import EXP, REQUIRED_FILES, ROOT


@dataclass
class Check:
    area: str
    name: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", default=os.environ.get("INPUT_DATA"))
    parser.add_argument("--hidden-states-dir", default=os.environ.get("HIDDEN_STATES_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--trained-ckpt", default=os.environ.get("TRAINED_CKPT"))
    parser.add_argument("--export-dir", default=os.environ.get("EXPORT_DIR"))
    parser.add_argument("--vllm-draft-dir", default=os.environ.get("VLLM_DRAFT_DIR"))
    parser.add_argument("--verifier-config-dir", default=os.environ.get("VERIFIER_CONFIG_DIR"))
    parser.add_argument("--smoke-json", type=Path, default=None)
    parser.add_argument(
        "--container-preflight-json",
        type=Path,
        default=Path(os.environ["CONTAINER_PREFLIGHT_JSON"]) if os.environ.get("CONTAINER_PREFLIGHT_JSON") else None,
    )
    parser.add_argument(
        "--nemo-rl-specdec-json",
        type=Path,
        default=Path(os.environ["NEMO_RL_SPECDEC_JSON"]) if os.environ.get("NEMO_RL_SPECDEC_JSON") else None,
    )
    parser.add_argument(
        "--nemo-rl-drift-json",
        type=Path,
        default=Path(os.environ["NEMO_RL_DRIFT_JSON"]) if os.environ.get("NEMO_RL_DRIFT_JSON") else None,
    )
    parser.add_argument(
        "--modelopt-loss-mask-json",
        type=Path,
        default=Path(os.environ["MODELOPT_LOSS_MASK_JSON"]) if os.environ.get("MODELOPT_LOSS_MASK_JSON") else None,
    )
    parser.add_argument(
        "--rollout-capture-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_CAPTURE_JSON"]) if os.environ.get("ROLLOUT_CAPTURE_JSON") else None,
    )
    parser.add_argument(
        "--rollout-capture-analysis-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_CAPTURE_ANALYSIS_JSON"]) if os.environ.get("ROLLOUT_CAPTURE_ANALYSIS_JSON") else None,
    )
    parser.add_argument(
        "--rollout-capture-job-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_CAPTURE_JOB_JSON"]) if os.environ.get("ROLLOUT_CAPTURE_JOB_JSON") else None,
    )
    parser.add_argument(
        "--rollout-submit-preflight-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_SUBMIT_PREFLIGHT_JSON"]) if os.environ.get("ROLLOUT_SUBMIT_PREFLIGHT_JSON") else None,
    )
    parser.add_argument(
        "--corpus-strategy-json",
        type=Path,
        default=Path(os.environ["CORPUS_STRATEGY_JSON"]) if os.environ.get("CORPUS_STRATEGY_JSON") else None,
    )
    parser.add_argument(
        "--training-scale-json",
        type=Path,
        default=Path(os.environ["TRAINING_SCALE_JSON"]) if os.environ.get("TRAINING_SCALE_JSON") else None,
    )
    parser.add_argument(
        "--pipeline-submit-preflight-json",
        type=Path,
        default=Path(os.environ["PIPELINE_SUBMIT_PREFLIGHT_JSON"]) if os.environ.get("PIPELINE_SUBMIT_PREFLIGHT_JSON") else None,
    )
    parser.add_argument("--chat-template", default=os.environ.get("CHAT_TEMPLATE"))
    parser.add_argument("--modelopt-dir", default=os.environ.get("MODELOPT_DIR"))
    parser.add_argument(
        "--reference-arch",
        default=os.environ.get("REFERENCE_ARCH", str(EXP / "qwen3_235b_thinking_eagle3_architecture.json")),
    )
    parser.add_argument("--arch-env-file", default=os.environ.get("ARCH_ENV_FILE"))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "dummy"))
    parser.add_argument("--conversation-sample-limit", type=int, default=200)
    parser.add_argument("--hidden-state-sample-limit", type=int, default=16)
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero for missing optional runtime artifacts, not just broken checks.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add(checks: list[Check], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append(Check(area=area, name=name, status=status, detail=detail, evidence=evidence))


def path_from(value: str | None) -> Path | None:
    return Path(value) if value else None


def nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def load_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            values[key] = value.strip().strip("'\"")
    return values


def check_local_tooling(checks: list[Check], args: argparse.Namespace) -> None:
    missing: list[str] = []
    not_executable: list[str] = []
    for item in dict.fromkeys(REQUIRED_FILES + ["audit_eagle3_readiness.py"]):
        path = EXP / item
        if not path.exists():
            missing.append(rel(path))
        elif item.endswith((".py", ".sh")) and not os.access(path, os.X_OK):
            not_executable.append(rel(path))
    if missing or not_executable:
        add(
            checks,
            "tooling",
            "local artifacts",
            "fail",
            "required local Eagle3 scripts are missing or not executable",
            missing=missing,
            not_executable=not_executable,
        )
    else:
        add(checks, "tooling", "local artifacts", "pass", "all required local scripts exist")

    modelopt = Path(args.modelopt_dir) if args.modelopt_dir else ROOT / "Model-Optimizer"
    required = [
        modelopt / "examples/speculative_decoding/launch_train.sh",
        modelopt / "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
        modelopt / "modelopt_recipes/general/speculative_decoding/eagle3.yaml",
    ]
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        add(checks, "tooling", "ModelOpt checkout", "fail", "ModelOpt files missing", missing=absent)
    else:
        add(checks, "tooling", "ModelOpt checkout", "pass", f"found {modelopt}")


def check_architecture(checks: list[Check], args: argparse.Namespace) -> dict[str, Any] | None:
    arch_path = Path(args.reference_arch)
    expected = {
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": [1, 46, 90],
        "rope_theta": 5000000,
    }
    try:
        cfg = load_json(arch_path)["eagle_architecture_config"]
    except Exception as exc:
        add(checks, "config", "architecture reference", "fail", f"cannot read reference: {exc}")
        return None

    required = [
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "head_dim",
        "use_aux_hidden_state",
        "eagle_aux_hidden_state_layer_ids",
    ]
    missing = [key for key in required if key not in cfg]
    malformed_aux = not isinstance(cfg.get("eagle_aux_hidden_state_layer_ids"), list) or not cfg.get(
        "eagle_aux_hidden_state_layer_ids"
    )
    if missing or cfg.get("num_hidden_layers") != 1 or cfg.get("use_aux_hidden_state") is not True or malformed_aux:
        add(
            checks,
            "config",
            "architecture reference",
            "fail",
            "architecture reference is missing required Eagle3 fields",
            path=str(arch_path),
            missing=missing,
            num_hidden_layers=cfg.get("num_hidden_layers"),
            use_aux_hidden_state=cfg.get("use_aux_hidden_state"),
            aux_layers=cfg.get("eagle_aux_hidden_state_layer_ids"),
        )
        return cfg

    if arch_path.name == "qwen3_235b_thinking_eagle3_architecture.json":
        mismatches = {
            key: {"actual": cfg.get(key), "expected": value}
            for key, value in expected.items()
            if cfg.get(key) != value
        }
        if mismatches:
            add(
                checks,
                "config",
                "architecture reference",
                "fail",
                "Qwen3-235B arch mismatch",
                path=str(arch_path),
                mismatches=mismatches,
            )
            return cfg
        detail = "Qwen3-235B Thinking Eagle3 fields match reference"
    else:
        detail = "custom Eagle3 architecture reference has required fields"

    add(
        checks,
        "config",
        "architecture reference",
        "pass",
        detail,
        path=str(arch_path),
        aux_layers=cfg.get("eagle_aux_hidden_state_layer_ids"),
        hidden_size=load_json(arch_path).get("verifier_summary", {}).get("hidden_size"),
    )
    return cfg


def check_architecture_derivation(
    checks: list[Check], args: argparse.Namespace, reference_arch: dict[str, Any] | None
) -> None:
    verifier = path_from(args.verifier_config_dir)
    if verifier is None:
        add(checks, "config", "architecture derivation", "missing", "VERIFIER_CONFIG_DIR is not set")
        return
    config_path = verifier / "config.json" if verifier.is_dir() else verifier
    if not config_path.exists():
        add(checks, "config", "architecture derivation", "missing", f"verifier config not visible: {config_path}")
        return
    if reference_arch is None:
        add(checks, "config", "architecture derivation", "warn", "reference architecture is unavailable")
        return

    with tempfile.TemporaryDirectory() as tmp:
        derived = Path(tmp) / "derived_arch.json"
        result = run(
            [
                sys.executable,
                "experiments/eagle3_qwen3_235b/derive_eagle3_architecture.py",
                "--verifier-config",
                str(verifier),
                "--json-out",
                str(derived),
            ]
        )
        evidence: dict[str, Any] = {
            "verifier_config": str(config_path),
            "reference_arch": str(args.reference_arch),
            "returncode": result.returncode,
        }
        if result.returncode != 0 or not derived.exists():
            evidence["output"] = result.stdout[-4000:]
            add(checks, "config", "architecture derivation", "fail", "derivation from verifier config failed", **evidence)
            return
        derived_arch = load_json(derived)["eagle_architecture_config"]
        keys = [
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "use_aux_hidden_state",
            "eagle_aux_hidden_state_layer_ids",
        ]
        mismatches = {
            key: {"derived": derived_arch.get(key), "reference": reference_arch.get(key)}
            for key in keys
            if key in reference_arch and derived_arch.get(key) != reference_arch.get(key)
        }
        evidence["aux_layers"] = derived_arch.get("eagle_aux_hidden_state_layer_ids")
        if mismatches:
            add(
                checks,
                "config",
                "architecture derivation",
                "fail",
                "derived verifier architecture does not match REFERENCE_ARCH",
                mismatches=mismatches,
                **evidence,
            )
        else:
            add(
                checks,
                "config",
                "architecture derivation",
                "pass",
                "REFERENCE_ARCH is reproducible from verifier config",
                **evidence,
            )


def check_arch_env(checks: list[Check], args: argparse.Namespace, reference_arch: dict[str, Any] | None) -> None:
    if not args.arch_env_file:
        add(
            checks,
            "config",
            "architecture env file",
            "warn",
            "ARCH_ENV_FILE is not set; Qwen3 defaults are used unless wrappers receive explicit env overrides",
        )
        return
    path = Path(args.arch_env_file)
    if not path.exists():
        add(checks, "config", "architecture env file", "fail", f"ARCH_ENV_FILE not visible: {path}")
        return
    values = parse_env_file(path)
    required = [
        "EAGLE_TRAIN_AUX_LAYERS",
        "EAGLE_DUMP_AUX_LAYERS",
        "EXPECTED_HIDDEN_SIZE",
        "EXPECTED_AUX_COUNT",
        "NUM_ATTENTION_HEADS",
        "NUM_KEY_VALUE_HEADS",
        "INTERMEDIATE_SIZE",
        "HEAD_DIM",
    ]
    missing = [key for key in required if key not in values]
    evidence: dict[str, Any] = {"path": str(path), "values": {key: values.get(key) for key in required}}
    if reference_arch:
        expected_aux = reference_arch.get("eagle_aux_hidden_state_layer_ids") or []
        expected_train_aux = "[" + ",".join(str(item) for item in expected_aux) + "]"
        expected_dump_aux = ",".join(str(item) for item in expected_aux)
        expected_aux_count = str(len(expected_aux))
        mismatches = {}
        if values.get("EAGLE_TRAIN_AUX_LAYERS") != expected_train_aux:
            mismatches["EAGLE_TRAIN_AUX_LAYERS"] = {
                "actual": values.get("EAGLE_TRAIN_AUX_LAYERS"),
                "expected": expected_train_aux,
            }
        if values.get("EAGLE_DUMP_AUX_LAYERS") != expected_dump_aux:
            mismatches["EAGLE_DUMP_AUX_LAYERS"] = {
                "actual": values.get("EAGLE_DUMP_AUX_LAYERS"),
                "expected": expected_dump_aux,
            }
        if values.get("EXPECTED_AUX_COUNT") != expected_aux_count:
            mismatches["EXPECTED_AUX_COUNT"] = {
                "actual": values.get("EXPECTED_AUX_COUNT"),
                "expected": expected_aux_count,
            }
        evidence["mismatches"] = mismatches
    else:
        mismatches = {}
    if missing or mismatches:
        add(
            checks,
            "config",
            "architecture env file",
            "fail",
            "ARCH_ENV_FILE is missing fields or disagrees with REFERENCE_ARCH",
            missing=missing,
            **evidence,
        )
    else:
        add(checks, "config", "architecture env file", "pass", "ARCH_ENV_FILE is aligned with REFERENCE_ARCH", **evidence)


def check_recipe_wrappers(checks: list[Check], args: argparse.Namespace) -> None:
    modelopt = Path(args.modelopt_dir) if args.modelopt_dir else ROOT / "Model-Optimizer"
    env = {"ARCH_ENV_FILE": args.arch_env_file or "", "REFERENCE_ARCH": str(args.reference_arch)}
    for mode, wrapper in (
        ("offline", "modelopt_qwen3_235b_offline_train.sh"),
        ("online", "modelopt_qwen3_235b_online_train.sh"),
    ):
        result = run(
            [
                sys.executable,
                "experiments/eagle3_qwen3_235b/validate_modelopt_recipe_overrides.py",
                "--wrapper",
                f"experiments/eagle3_qwen3_235b/{wrapper}",
                "--training-mode",
                mode,
                "--modelopt-dir",
                str(modelopt),
                "--reference-arch",
                str(args.reference_arch),
            ],
            env=env,
        )
        evidence = {"wrapper": wrapper, "mode": mode, "returncode": result.returncode}
        if result.returncode == 0:
            add(
                checks,
                "config",
                f"ModelOpt {mode} recipe overrides",
                "pass",
                "wrapper dry-run produced valid Eagle3 overrides for REFERENCE_ARCH",
                **evidence,
            )
        else:
            evidence["output"] = result.stdout[-4000:]
            add(
                checks,
                "config",
                f"ModelOpt {mode} recipe overrides",
                "fail",
                "wrapper recipe override validation failed",
                **evidence,
            )


def check_conversations(checks: list[Check], args: argparse.Namespace) -> None:
    path = path_from(args.input_data)
    if path is None:
        add(checks, "data", "training conversations", "missing", "INPUT_DATA is not set")
        return
    if not path.exists():
        add(checks, "data", "training conversations", "missing", f"INPUT_DATA not visible: {path}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        summary = Path(tmp) / "conversation_validation.json"
        result = run(
            [
                sys.executable,
                "experiments/eagle3_qwen3_235b/validate_training_conversations.py",
                str(path),
                "--limit",
                str(args.conversation_sample_limit),
                "--max-seq-len",
                "16384",
                "--json-out",
                str(summary),
            ]
        )
        evidence: dict[str, Any] = {"path": str(path), "returncode": result.returncode}
        if summary.exists():
            evidence["summary"] = load_json(summary)
        if result.returncode == 0:
            add(checks, "data", "training conversations", "pass", "conversation sample validates", **evidence)
        else:
            evidence["output"] = result.stdout[-4000:]
            add(checks, "data", "training conversations", "fail", "conversation validation failed", **evidence)


def check_chat_template(checks: list[Check], args: argparse.Namespace) -> None:
    if not args.chat_template:
        add(
            checks,
            "data",
            "answer-only chat template",
            "warn",
            "CHAT_TEMPLATE is not set; use prepare_qwen3_chat_template.sh before ANSWER_ONLY_LOSS=true",
        )
        return
    path = Path(args.chat_template)
    if not path.exists():
        add(checks, "data", "answer-only chat template", "warn", f"CHAT_TEMPLATE not visible: {path}")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "generation" not in text or "endgeneration" not in text:
        add(checks, "data", "answer-only chat template", "warn", "generation/endgeneration tags not found")
        return

    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/validate_chat_template_loss_mask.py",
            "--model-or-tokenizer",
            os.environ.get("BASE_MODEL", "Qwen/Qwen3-235B-A22B-Thinking-2507"),
            "--chat-template",
            str(path),
            "--allow-missing-transformers",
        ]
    )
    evidence = {"path": str(path), "returncode": result.returncode}
    if result.returncode == 0 and "WARN Transformers unavailable" not in result.stdout:
        add(checks, "data", "answer-only chat template", "pass", "assistant mask validation passed", **evidence)
    elif result.returncode == 0:
        add(
            checks,
            "data",
            "answer-only chat template",
            "warn",
            "generation tags found, but Transformers is unavailable for assistant mask validation",
            **evidence,
        )
    else:
        evidence["output"] = result.stdout[-4000:]
        add(checks, "data", "answer-only chat template", "fail", "assistant mask validation failed", **evidence)


def check_hidden_states(checks: list[Check], args: argparse.Namespace) -> None:
    path = path_from(args.hidden_states_dir)
    if path is None:
        add(checks, "hidden_states", "dump output", "missing", "HIDDEN_STATES_DIR is not set")
        return
    if not path.exists():
        add(checks, "hidden_states", "dump output", "missing", f"HIDDEN_STATES_DIR not visible: {path}")
        return
    files = sorted(path.glob("*.pt"))
    if not files:
        add(checks, "hidden_states", "dump output", "missing", f"no .pt files found under {path}")
        return

    expected_hidden_size = os.environ.get("EXPECTED_HIDDEN_SIZE", "4096")
    expected_aux_count = os.environ.get("EXPECTED_AUX_COUNT", "3")
    if args.arch_env_file and Path(args.arch_env_file).exists():
        env_values = parse_env_file(Path(args.arch_env_file))
        expected_hidden_size = env_values.get("EXPECTED_HIDDEN_SIZE", expected_hidden_size)
        expected_aux_count = env_values.get("EXPECTED_AUX_COUNT", expected_aux_count)

    result = run(
        [
            sys.executable,
            "experiments/eagle3_qwen3_235b/validate_hidden_state_dump.py",
            str(path),
            "--limit",
            str(args.hidden_state_sample_limit),
            "--require-loss-mask",
            "--require-positive-loss-mask",
            "--expected-hidden-size",
            expected_hidden_size,
            "--expected-aux-count",
            expected_aux_count,
            "--max-seq-len",
            "16384",
        ]
    )
    evidence = {"path": str(path), "pt_files": len(files), "returncode": result.returncode}
    if result.returncode == 0:
        add(checks, "hidden_states", "dump output", "pass", "hidden-state shard sample validates", **evidence)
    elif "No module named 'torch'" in result.stdout:
        add(
            checks,
            "hidden_states",
            "dump output",
            "warn",
            "found .pt files but torch is unavailable in this environment for shape validation",
            **evidence,
        )
    else:
        evidence["output"] = result.stdout[-4000:]
        add(checks, "hidden_states", "dump output", "fail", "hidden-state validation failed", **evidence)


def check_training_and_exports(checks: list[Check], args: argparse.Namespace) -> None:
    output = path_from(args.output_dir)
    trained = path_from(args.trained_ckpt) or output
    export = path_from(args.export_dir)
    vllm = path_from(args.vllm_draft_dir)
    verifier = path_from(args.verifier_config_dir)

    for area, name, path in (
        ("training", "ModelOpt checkpoint", trained),
        ("export", "HF export", export),
        ("export", "vLLM draft", vllm),
    ):
        if path is None:
            add(checks, area, name, "missing", f"{name} path is not set")
        elif nonempty_dir(path):
            add(checks, area, name, "pass", f"{name} directory is non-empty", path=str(path))
        elif path.exists():
            add(checks, area, name, "warn", f"{name} path exists but is empty", path=str(path))
        else:
            add(checks, area, name, "missing", f"{name} path not visible: {path}")

    draft_config = None
    if export and (export / "config.json").exists():
        draft_config = export
    elif vllm and (vllm / "config.json").exists():
        draft_config = vllm

    if draft_config and verifier and (verifier / "config.json").exists():
        result = run(
            [
                sys.executable,
                "experiments/eagle3_qwen3_235b/compare_eagle3_configs.py",
                "--draft-config",
                str(draft_config),
                "--verifier-config",
                str(verifier),
                "--reference-arch",
                str(args.reference_arch),
            ]
        )
        evidence = {"draft_config": str(draft_config), "verifier_config": str(verifier), "returncode": result.returncode}
        if result.returncode == 0:
            add(checks, "export", "draft config compatibility", "pass", "draft config matches verifier/reference", **evidence)
        else:
            evidence["output"] = result.stdout[-4000:]
            add(checks, "export", "draft config compatibility", "fail", "draft config comparison failed", **evidence)
    else:
        add(
            checks,
            "export",
            "draft config compatibility",
            "missing",
            "exported draft config and local verifier config are not both visible",
        )


def check_smoke(checks: list[Check], args: argparse.Namespace) -> None:
    if args.smoke_json is None:
        add(checks, "rl_validation", "static draft smoke", "missing", "smoke analyzer JSON is not provided")
        return
    if not args.smoke_json.exists():
        add(checks, "rl_validation", "static draft smoke", "missing", f"smoke JSON not visible: {args.smoke_json}")
        return
    try:
        payload = load_json(args.smoke_json)
    except Exception as exc:
        add(checks, "rl_validation", "static draft smoke", "fail", f"cannot parse smoke JSON: {exc}")
        return
    gate = payload.get("current", {}).get("gate", {})
    status = gate.get("status")
    if status == "pass":
        add(checks, "rl_validation", "static draft smoke", "pass", "smoke gate passed", gate=gate)
    elif status == "fail":
        add(checks, "rl_validation", "static draft smoke", "fail", "smoke gate failed", gate=gate)
    else:
        add(checks, "rl_validation", "static draft smoke", "warn", "smoke JSON has no pass/fail gate", gate=gate)


def check_container_preflight(checks: list[Check], args: argparse.Namespace) -> None:
    if args.container_preflight_json is None:
        add(
            checks,
            "execution",
            "container preflight",
            "missing",
            "container preflight analysis JSON is not provided",
        )
        return
    if not args.container_preflight_json.exists():
        add(
            checks,
            "execution",
            "container preflight",
            "missing",
            f"container preflight JSON not visible: {args.container_preflight_json}",
        )
        return
    try:
        payload = load_json(args.container_preflight_json)
    except Exception as exc:
        add(checks, "execution", "container preflight", "fail", f"cannot parse container preflight JSON: {exc}")
        return

    overall = payload.get("overall_status")
    preflight_status = payload.get("status")
    evidence = {
        "overall_status": overall,
        "preflight_status": preflight_status,
        "job_id": payload.get("job_id"),
        "container": payload.get("container"),
        "out_log": payload.get("out_log"),
        "err_log": payload.get("err_log"),
    }
    if overall == "pass":
        add(checks, "execution", "container preflight", "pass", "container preflight passed", **evidence)
    elif overall == "fail":
        add(checks, "execution", "container preflight", "fail", "container preflight failed", **evidence)
    else:
        add(
            checks,
            "execution",
            "container preflight",
            "warn",
            "container preflight has not passed yet",
            **evidence,
        )


def check_nemo_rl_specdec(checks: list[Check], args: argparse.Namespace) -> None:
    if args.nemo_rl_specdec_json is None:
        add(
            checks,
            "rl_validation",
            "NeMo-RL specdec integration",
            "missing",
            "NeMo-RL specdec integration JSON is not provided",
        )
        return
    if not args.nemo_rl_specdec_json.exists():
        add(
            checks,
            "rl_validation",
            "NeMo-RL specdec integration",
            "missing",
            f"NeMo-RL specdec integration JSON not visible: {args.nemo_rl_specdec_json}",
        )
        return
    try:
        payload = load_json(args.nemo_rl_specdec_json)
    except Exception as exc:
        add(checks, "rl_validation", "NeMo-RL specdec integration", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    evidence = {
        "overall_status": overall,
        "config": payload.get("config"),
        "draft_model": payload.get("draft_model"),
        "hydra_overrides": payload.get("hydra_overrides"),
    }
    if overall == "pass":
        add(
            checks,
            "rl_validation",
            "NeMo-RL specdec integration",
            "pass",
            "RL config and SpecDec-RL source accept Eagle3 speculative_config",
            **evidence,
        )
    elif overall == "fail":
        add(checks, "rl_validation", "NeMo-RL specdec integration", "fail", "integration validation failed", **evidence)
    else:
        add(
            checks,
            "rl_validation",
            "NeMo-RL specdec integration",
            "warn",
            "integration validation has warnings",
            **evidence,
        )


def check_nemo_rl_drift(checks: list[Check], args: argparse.Namespace) -> None:
    if args.nemo_rl_drift_json is None:
        add(
            checks,
            "rl_validation",
            "NeMo-RL Eagle3 drift/support",
            "missing",
            "NeMo-RL Eagle3 drift/support JSON is not provided",
        )
        return
    if not args.nemo_rl_drift_json.exists():
        add(
            checks,
            "rl_validation",
            "NeMo-RL Eagle3 drift/support",
            "missing",
            f"NeMo-RL Eagle3 drift/support JSON not visible: {args.nemo_rl_drift_json}",
        )
        return
    try:
        payload = load_json(args.nemo_rl_drift_json)
    except Exception as exc:
        add(checks, "rl_validation", "NeMo-RL Eagle3 drift/support", "fail", f"cannot parse JSON: {exc}")
        return

    fixed = (payload.get("support") or {}).get("fixed_generation") or {}
    online = (payload.get("support") or {}).get("online_draft_training") or {}
    evidence = {
        "overall_status": payload.get("overall_status"),
        "fixed_generation_status": fixed.get("status"),
        "online_draft_training_status": online.get("status"),
        "recommendation": payload.get("recommendation"),
        "repo": {
            "path": (payload.get("repo") or {}).get("path"),
            "branch": (payload.get("repo") or {}).get("branch"),
            "short_head": (payload.get("repo") or {}).get("short_head"),
        },
        "notes": payload.get("notes"),
    }

    if payload.get("overall_status") == "fail" or fixed.get("status") != "pass":
        add(
            checks,
            "rl_validation",
            "NeMo-RL Eagle3 drift/support",
            "fail",
            "fixed Eagle3 rollout support is not proven in the target checkout",
            **evidence,
        )
    elif online.get("status") != "pass" or payload.get("overall_status") == "warn":
        add(
            checks,
            "rl_validation",
            "NeMo-RL Eagle3 drift/support",
            "warn",
            "fixed Eagle3 rollout support is present; online draft training is not fully proven",
            **evidence,
        )
    else:
        add(
            checks,
            "rl_validation",
            "NeMo-RL Eagle3 drift/support",
            "pass",
            "fixed and online Eagle3 source markers are present",
            **evidence,
        )


def check_modelopt_loss_mask_report(checks: list[Check], args: argparse.Namespace) -> None:
    if args.modelopt_loss_mask_json is None:
        add(checks, "config", "ModelOpt loss-mask patch", "missing", "ModelOpt loss-mask patch JSON is not provided")
        return
    if not args.modelopt_loss_mask_json.exists():
        add(
            checks,
            "config",
            "ModelOpt loss-mask patch",
            "missing",
            f"ModelOpt loss-mask patch JSON not visible: {args.modelopt_loss_mask_json}",
        )
        return
    try:
        payload = load_json(args.modelopt_loss_mask_json)
    except Exception as exc:
        add(checks, "config", "ModelOpt loss-mask patch", "fail", f"cannot parse JSON: {exc}")
        return
    evidence = {
        "overall_status": payload.get("overall_status"),
        "modelopt_dir": payload.get("modelopt_dir"),
        "check_statuses": {
            key: value.get("status")
            for key, value in (payload.get("checks") or {}).items()
            if isinstance(value, dict)
        },
        "recommendation": payload.get("recommendation"),
    }
    if payload.get("overall_status") == "pass":
        add(
            checks,
            "config",
            "ModelOpt loss-mask patch",
            "pass",
            "TRT-LLM hidden-state dumper preserves answer-only loss_mask",
            **evidence,
        )
    else:
        add(
            checks,
            "config",
            "ModelOpt loss-mask patch",
            "fail",
            "TRT-LLM hidden-state loss_mask patch is not proven",
            **evidence,
        )


def check_rollout_capture(checks: list[Check], args: argparse.Namespace) -> None:
    if args.rollout_capture_json is None:
        add(
            checks,
            "data",
            "RL rollout capture",
            "missing",
            "RL rollout capture validation JSON is not provided",
        )
        return
    if not args.rollout_capture_json.exists():
        add(
            checks,
            "data",
            "RL rollout capture",
            "missing",
            f"RL rollout capture validation JSON not visible: {args.rollout_capture_json}",
        )
        return
    try:
        payload = load_json(args.rollout_capture_json)
    except Exception as exc:
        add(checks, "data", "RL rollout capture", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    recommendation = payload.get("recommendation") or {}
    evidence = {
        "overall_status": overall,
        "train_data_glob": (payload.get("config_data") or {}).get("train_data_glob"),
        "output_conversations": recommendation.get("output_conversations"),
        "role_logging_patch": recommendation.get("role_logging_patch"),
    }
    if overall == "pass":
        add(checks, "data", "RL rollout capture", "pass", "RL rollout logs can be normalized into Eagle3 conversations", **evidence)
    elif overall == "fail":
        add(checks, "data", "RL rollout capture", "fail", "RL rollout capture validation failed", **evidence)
    else:
        add(
            checks,
            "data",
            "RL rollout capture",
            "warn",
            "RL rollout capture path has warnings; inspect role logging and normalizer fallback",
            **evidence,
        )


def check_rollout_capture_analysis(checks: list[Check], args: argparse.Namespace) -> None:
    if args.rollout_capture_analysis_json is None:
        add(
            checks,
            "data",
            "rollout capture artifacts",
            "missing",
            "rollout capture artifact analysis JSON is not provided",
        )
        return
    if not args.rollout_capture_analysis_json.exists():
        add(
            checks,
            "data",
            "rollout capture artifacts",
            "missing",
            f"rollout capture artifact analysis JSON not visible: {args.rollout_capture_analysis_json}",
        )
        return
    try:
        payload = load_json(args.rollout_capture_analysis_json)
    except Exception as exc:
        add(checks, "data", "rollout capture artifacts", "fail", f"cannot parse JSON: {exc}")
        return

    output_data = payload.get("output_data")
    if isinstance(output_data, dict):
        output_data_path = output_data.get("path")
    elif isinstance(output_data, str):
        output_data_path = output_data
    else:
        output_data_path = None

    overall = payload.get("overall_status")
    evidence = {
        "overall_status": overall,
        "rollout_log_dir": payload.get("rollout_log_dir"),
        "train_file_count": (payload.get("train_data") or {}).get("file_count"),
        "extractable_conversations": (payload.get("train_data") or {}).get("extractable_conversations"),
        "output_data": output_data_path,
    }
    if overall == "pass":
        add(checks, "data", "rollout capture artifacts", "pass", "materialized rollout corpus validates", **evidence)
    elif overall == "needs_materialize":
        add(checks, "data", "rollout capture artifacts", "warn", "rollout train_data exists and needs corpus materialization", **evidence)
    elif overall == "missing_capture":
        add(checks, "data", "rollout capture artifacts", "missing", "rollout capture train_data is not present yet", **evidence)
    else:
        add(checks, "data", "rollout capture artifacts", "fail", "rollout capture artifact analysis failed", **evidence)


def check_rollout_capture_job(checks: list[Check], args: argparse.Namespace) -> None:
    if args.rollout_capture_job_json is None:
        add(checks, "data", "rollout capture job", "missing", "rollout capture job analysis JSON is not provided")
        return
    if not args.rollout_capture_job_json.exists():
        add(
            checks,
            "data",
            "rollout capture job",
            "missing",
            f"rollout capture job analysis JSON not visible: {args.rollout_capture_job_json}",
        )
        return
    try:
        payload = load_json(args.rollout_capture_job_json)
    except Exception as exc:
        add(checks, "data", "rollout capture job", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    artifacts = payload.get("artifacts") or {}
    evidence = {
        "overall_status": overall,
        "job_detail": payload.get("detail"),
        "job_id": (payload.get("job") or {}).get("job_id"),
        "slurm_status": (payload.get("slurm") or {}).get("status"),
        "train_file_count": ((artifacts.get("train_data") or {}).get("file_count")),
        "extractable_conversations": ((artifacts.get("train_data") or {}).get("extractable_conversations")),
        "output_status": ((artifacts.get("output_data") or {}).get("status")),
    }
    if overall == "pass":
        add(checks, "data", "rollout capture job", "pass", "rollout capture job produced a validated corpus", **evidence)
    elif overall == "needs_materialize":
        add(checks, "data", "rollout capture job", "warn", "rollout job produced train_data and needs materialization", **evidence)
    elif overall == "running":
        add(checks, "data", "rollout capture job", "warn", "rollout capture job is still running or queued", **evidence)
    elif overall in {"not_submitted", "missing_capture"}:
        add(checks, "data", "rollout capture job", "missing", "rollout capture job/corpus is not available yet", **evidence)
    else:
        add(checks, "data", "rollout capture job", "fail", "rollout capture job analysis failed or is unknown", **evidence)


def check_rollout_submit_preflight(checks: list[Check], args: argparse.Namespace) -> None:
    if args.rollout_submit_preflight_json is None:
        add(checks, "execution", "rollout submit preflight", "missing", "rollout submit preflight JSON is not provided")
        return
    if not args.rollout_submit_preflight_json.exists():
        add(
            checks,
            "execution",
            "rollout submit preflight",
            "missing",
            f"rollout submit preflight JSON not visible: {args.rollout_submit_preflight_json}",
        )
        return
    try:
        payload = load_json(args.rollout_submit_preflight_json)
    except Exception as exc:
        add(checks, "execution", "rollout submit preflight", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    evidence = {
        "overall_status": overall,
        "submit_ready": payload.get("submit_ready"),
        "repo_root": payload.get("repo_root"),
        "config": payload.get("config"),
        "chat_template": payload.get("chat_template"),
        "rollout_log_dir": payload.get("rollout_log_dir"),
        "output_conversations": payload.get("output_conversations"),
    }
    if overall in {"pass", "warn"} and payload.get("submit_ready") is True:
        add(
            checks,
            "execution",
            "rollout submit preflight",
            "pass" if overall == "pass" else "warn",
            "rollout capture submit path dry-runs successfully",
            **evidence,
        )
    elif overall == "fail":
        add(checks, "execution", "rollout submit preflight", "fail", "rollout submit preflight failed", **evidence)
    else:
        add(
            checks,
            "execution",
            "rollout submit preflight",
            "warn",
            "rollout submit preflight did not prove submit readiness",
            **evidence,
        )


def check_corpus_strategy(checks: list[Check], args: argparse.Namespace) -> None:
    if args.corpus_strategy_json is None:
        add(checks, "data", "corpus strategy", "missing", "corpus strategy JSON is not provided")
        return
    if not args.corpus_strategy_json.exists():
        add(
            checks,
            "data",
            "corpus strategy",
            "missing",
            f"corpus strategy JSON not visible: {args.corpus_strategy_json}",
        )
        return
    try:
        payload = load_json(args.corpus_strategy_json)
    except Exception as exc:
        add(checks, "data", "corpus strategy", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    decision = payload.get("decision") or {}
    evidence = {
        "overall_status": overall,
        "target_context": payload.get("target_context"),
        "primary_source": decision.get("primary_source"),
        "next_action": decision.get("next_action"),
        "input_data_status": (payload.get("input_data") or {}).get("status"),
        "rollout_status": (payload.get("rollout_capture_analysis") or {}).get("overall_status"),
    }
    if overall == "pass":
        add(checks, "data", "corpus strategy", "pass", "corpus source is aligned with target context", **evidence)
    elif overall in {"needs_materialize", "bootstrap_data_only"}:
        add(checks, "data", "corpus strategy", "warn", "corpus path needs the indicated next action", **evidence)
    elif overall in {"missing_capture", "missing_math_corpus", "missing_corpus"}:
        add(checks, "data", "corpus strategy", "missing", "primary corpus for target context is not available yet", **evidence)
    else:
        add(checks, "data", "corpus strategy", "fail", "corpus strategy report indicates failure or unknown state", **evidence)


def check_training_scale(checks: list[Check], args: argparse.Namespace) -> None:
    if args.training_scale_json is None:
        add(checks, "training", "Eagle3 training scale plan", "missing", "training scale JSON is not provided")
        return
    if not args.training_scale_json.exists():
        add(
            checks,
            "training",
            "Eagle3 training scale plan",
            "missing",
            f"training scale JSON not visible: {args.training_scale_json}",
        )
        return
    try:
        payload = load_json(args.training_scale_json)
    except Exception as exc:
        add(checks, "training", "Eagle3 training scale plan", "fail", f"cannot parse JSON: {exc}")
        return

    rec = payload.get("recommendation") or {}
    defaults = payload.get("training_defaults") or {}
    corpus = payload.get("corpus") or {}
    evidence = {
        "overall_status": payload.get("overall_status"),
        "recommendation_status": rec.get("status"),
        "recommendation": rec.get("summary"),
        "effective_global_batch": defaults.get("effective_global_batch"),
        "corpus_rows": corpus.get("total_rows"),
        "avg_estimated_tokens": (corpus.get("estimated_tokens") or {}).get("avg"),
    }
    if payload.get("overall_status") == "pass":
        add(checks, "training", "Eagle3 training scale plan", "pass", "training scale is ready for submit planning", **evidence)
    elif payload.get("overall_status") in {"planning", "incomplete"}:
        add(checks, "training", "Eagle3 training scale plan", "warn", "training scale plan exists but still needs the indicated gate", **evidence)
    else:
        add(checks, "training", "Eagle3 training scale plan", "fail", "training scale report has unknown status", **evidence)


def check_pipeline_submit_preflight(checks: list[Check], args: argparse.Namespace) -> None:
    if args.pipeline_submit_preflight_json is None:
        add(checks, "execution", "pipeline submit preflight", "missing", "pipeline submit preflight JSON is not provided")
        return
    if not args.pipeline_submit_preflight_json.exists():
        add(
            checks,
            "execution",
            "pipeline submit preflight",
            "missing",
            f"pipeline submit preflight JSON not visible: {args.pipeline_submit_preflight_json}",
        )
        return
    try:
        payload = load_json(args.pipeline_submit_preflight_json)
    except Exception as exc:
        add(checks, "execution", "pipeline submit preflight", "fail", f"cannot parse JSON: {exc}")
        return

    overall = payload.get("overall_status")
    submit_ready = payload.get("submit_ready")
    evidence = {
        "overall_status": overall,
        "submit_ready": submit_ready,
        "input_data": payload.get("input_data"),
        "preflight_check_statuses": {
            item.get("name"): item.get("status")
            for item in payload.get("checks", [])
            if item.get("name") in {
                "training conversation validation",
                "local ModelOpt pipeline preflight",
                "Slurm pipeline dry-run",
                "container preflight",
                "corpus strategy",
                "rollout state advance",
            }
        },
    }
    if overall == "pass" and submit_ready is True:
        add(
            checks,
            "execution",
            "pipeline submit preflight",
            "pass",
            "hidden-state/train/export pipeline submit preflight passed",
            **evidence,
        )
    elif overall == "fail":
        add(checks, "execution", "pipeline submit preflight", "fail", "pipeline submit preflight failed", **evidence)
    else:
        add(
            checks,
            "execution",
            "pipeline submit preflight",
            "missing",
            "pipeline submit preflight has not reached submit readiness",
            **evidence,
        )


def check_dry_run(checks: list[Check], args: argparse.Namespace) -> None:
    if args.skip_dry_run:
        add(checks, "execution", "pipeline dry-run", "warn", "dry-run skipped by request")
        return
    env = {
        "SUBMIT": "false",
        "SBATCH_ACCOUNT": args.sbatch_account,
        "INPUT_DATA": args.input_data or "/tmp/conversations.jsonl",
        "HIDDEN_STATES_DIR": args.hidden_states_dir or "/tmp/hiddens",
        "OUTPUT_DIR": args.output_dir or "/tmp/modelopt_ckpt",
        "TRAINED_CKPT": args.trained_ckpt or args.output_dir or "/tmp/modelopt_ckpt",
        "EXPORT_DIR": args.export_dir or "/tmp/exported",
        "VLLM_DRAFT_DIR": args.vllm_draft_dir or "/tmp/vllm",
        "VERIFIER_CONFIG_DIR": args.verifier_config_dir or "/tmp/verifier",
        "CONTAINER": "",
        "ARCH_ENV_FILE": args.arch_env_file or "",
        "REFERENCE_ARCH": str(args.reference_arch),
    }
    result = run(["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"], env)
    has_validate_gate = "slurm_validate_hidden_states.sbatch" in result.stdout
    has_dependency = "--dependency=afterok:VALIDATE_HIDDENS_JOB_ID" in result.stdout
    evidence = {"returncode": result.returncode, "has_validate_gate": has_validate_gate, "has_train_dependency": has_dependency}
    if result.returncode == 0 and has_validate_gate and has_dependency:
        add(checks, "execution", "pipeline dry-run", "pass", "Slurm plan includes post-dump validation gate", **evidence)
    else:
        evidence["output"] = result.stdout[-4000:]
        add(checks, "execution", "pipeline dry-run", "fail", "pipeline dry-run missing expected validation dependency", **evidence)

    pilot_env = {**env, "RUN_PILOT": "true"}
    pilot = run(["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"], pilot_env)
    pilot_markers = {
        "run_pilot": "RUN_PILOT=true" in pilot.stdout,
        "dump_limit": "DEBUG_MAX_NUM_CONVERSATIONS=8" in pilot.stdout,
        "sample_size": "DATA_SAMPLE_SIZE=8" in pilot.stdout,
        "max_steps": "MAX_STEPS=20" in pilot.stdout,
        "short_dump_time": "--time=02:00:00" in pilot.stdout,
        "short_export_time": "--time=01:00:00" in pilot.stdout,
    }
    pilot_evidence = {"returncode": pilot.returncode, **pilot_markers}
    if pilot.returncode == 0 and all(pilot_markers.values()):
        add(checks, "execution", "pilot dry-run", "pass", "pilot plan limits dump/train/export before full run", **pilot_evidence)
    else:
        pilot_evidence["output"] = pilot.stdout[-4000:]
        add(checks, "execution", "pilot dry-run", "fail", "pilot dry-run is missing expected limit markers", **pilot_evidence)


def status_counts(checks: list[Check]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def overall_status(checks: list[Check], strict: bool) -> str:
    counts = status_counts(checks)
    if counts.get("fail", 0):
        return "fail"
    if strict and counts.get("missing", 0):
        return "fail"
    if counts.get("missing", 0) or counts.get("warn", 0):
        return "incomplete"
    return "pass"


def to_payload(checks: list[Check], strict: bool) -> dict[str, Any]:
    return {
        "overall_status": overall_status(checks, strict),
        "strict": strict,
        "counts": status_counts(checks),
        "checks": [
            {
                "area": check.area,
                "name": check.name,
                "status": check.status,
                "detail": check.detail,
                "evidence": check.evidence,
            }
            for check in checks
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-235B Eagle3 Readiness Audit",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        "",
        "| area | check | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['area']} | {check['name']} | {check['status'].upper()} | "
            f"{check['detail'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    checks: list[Check] = []
    check_local_tooling(checks, args)
    reference_arch = check_architecture(checks, args)
    check_architecture_derivation(checks, args, reference_arch)
    check_arch_env(checks, args, reference_arch)
    check_recipe_wrappers(checks, args)
    check_conversations(checks, args)
    check_chat_template(checks, args)
    check_hidden_states(checks, args)
    check_training_and_exports(checks, args)
    check_smoke(checks, args)
    check_nemo_rl_specdec(checks, args)
    check_nemo_rl_drift(checks, args)
    check_modelopt_loss_mask_report(checks, args)
    check_rollout_capture(checks, args)
    check_rollout_capture_analysis(checks, args)
    check_rollout_capture_job(checks, args)
    check_rollout_submit_preflight(checks, args)
    check_corpus_strategy(checks, args)
    check_training_scale(checks, args)
    check_pipeline_submit_preflight(checks, args)
    check_container_preflight(checks, args)
    check_dry_run(checks, args)

    payload = to_payload(checks, args.strict)
    text = render_markdown(payload)
    print(text, end="")

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return 1 if payload["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
