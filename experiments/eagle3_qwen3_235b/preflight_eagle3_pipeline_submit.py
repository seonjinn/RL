#!/usr/bin/env python3
"""No-submit gate for the Qwen3-235B Eagle3 hidden-state/train/export pipeline.

This is the gate to run after the RL rollout corpus has been materialized. It
does not submit Slurm jobs. It proves the selected corpus, verifier config,
chat-template masking path, ModelOpt wrappers, and Slurm dependency chain are
coherent enough to submit the pilot hidden-state pipeline.
"""

from __future__ import annotations

import argparse
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
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_CONTAINER = "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_llm/containers/nemo_25.07.01.sqsh"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


def parse_args() -> argparse.Namespace:
    artifact_default = Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_default)
    parser.add_argument(
        "--input-data",
        type=Path,
        default=env_path("INPUT_DATA", artifact_default / "data/qwen3_235b_swe_rollout_conversations.jsonl"),
    )
    parser.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=env_path("HIDDEN_STATES_DIR", artifact_default / "hidden_states"),
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("OUTPUT_DIR", artifact_default / "modelopt_ckpt"))
    parser.add_argument("--trained-ckpt", type=Path, default=env_path("TRAINED_CKPT", artifact_default / "modelopt_ckpt"))
    parser.add_argument("--export-dir", type=Path, default=env_path("EXPORT_DIR", artifact_default / "exported_hf"))
    parser.add_argument("--vllm-draft-dir", type=Path, default=env_path("VLLM_DRAFT_DIR", artifact_default / "vllm_draft"))
    parser.add_argument(
        "--verifier-config-dir",
        type=Path,
        default=env_path("VERIFIER_CONFIG_DIR", artifact_default / "verifier_config"),
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=env_path("CHAT_TEMPLATE", artifact_default / "templates/qwen3_generation_template.jinja2"),
    )
    parser.add_argument(
        "--modelopt-dir",
        type=Path,
        default=Path(os.environ.get("MODELOPT_DIR", ROOT / "Model-Optimizer")),
    )
    parser.add_argument(
        "--reference-arch",
        type=Path,
        default=env_path("REFERENCE_ARCH", artifact_default / "architecture/eagle3_architecture.json"),
    )
    parser.add_argument(
        "--arch-env-file",
        type=Path,
        default=env_path("ARCH_ENV_FILE", artifact_default / "architecture/eagle3_architecture.env"),
    )
    parser.add_argument(
        "--container-preflight-json",
        type=Path,
        default=env_path("CONTAINER_PREFLIGHT_JSON", artifact_default / "reports/container_preflight_analysis.json"),
    )
    parser.add_argument(
        "--corpus-strategy-json",
        type=Path,
        default=env_path("CORPUS_STRATEGY_JSON", artifact_default / "reports/corpus_strategy.json"),
    )
    parser.add_argument(
        "--rollout-state-json",
        type=Path,
        default=env_path("ROLLOUT_STATE_ADVANCE_JSON", artifact_default / "reports/rollout_capture_state_advance.json"),
    )
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-235B-A22B-Thinking-2507"))
    parser.add_argument("--training-seq-len", type=int, default=int(os.environ.get("TRAINING_SEQ_LEN", "16384")))
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=int(os.environ.get("MAX_SEQ_LEN", os.environ.get("TRAINING_SEQ_LEN", "16384"))),
    )
    parser.add_argument("--answer-only-loss", default=os.environ.get("ANSWER_ONLY_LOSS", "true"))
    parser.add_argument("--sbatch-account", default=os.environ.get("SBATCH_ACCOUNT", "coreai_dlalgo_nemorl"))
    parser.add_argument("--sbatch-partition", default=os.environ.get("SBATCH_PARTITION", "batch"))
    parser.add_argument("--container", default=os.environ.get("CONTAINER") or DEFAULT_CONTAINER)
    parser.add_argument("--mounts", default=os.environ.get("MOUNTS", f"/lustre:/lustre,{ROOT}:{ROOT},{artifact_default}:{artifact_default}"))
    parser.add_argument("--run-pilot", default=os.environ.get("RUN_PILOT", "true"))
    parser.add_argument(
        "--min-pilot-rows",
        type=int,
        default=int(os.environ.get("MIN_PIPELINE_PILOT_ROWS") or os.environ.get("DATA_SAMPLE_SIZE", "8") or "8"),
        help="Minimum valid conversation rows required before RUN_PILOT=true can be submit-ready.",
    )
    parser.add_argument("--dump-gpus-per-node", type=int, default=int(os.environ.get("DUMP_GPUS_PER_NODE", "8")))
    parser.add_argument("--train-gpus-per-node", type=int, default=int(os.environ.get("TRAIN_GPUS_PER_NODE", "8")))
    parser.add_argument("--export-gpus-per-node", type=int, default=int(os.environ.get("EXPORT_GPUS_PER_NODE", "1")))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    parser.add_argument(
        "--slurm-capacity-json",
        type=Path,
        default=env_path("SLURM_CAPACITY_JSON", artifact_default / "reports/eagle3_slurm_capacity.json"),
    )
    parser.add_argument(
        "--slurm-capacity-markdown",
        type=Path,
        default=env_path("SLURM_CAPACITY_MARKDOWN", artifact_default / "reports/eagle3_slurm_capacity.md"),
    )
    parser.add_argument(
        "--slurm-capacity-env",
        type=Path,
        default=env_path("SLURM_CAPACITY_ENV", artifact_default / "reports/eagle3_resource_profile.env"),
    )
    parser.add_argument("--target-context", choices=("swe_rl", "math", "general"), default=os.environ.get("EAGLE3_TARGET_CONTEXT", "swe_rl"))
    parser.add_argument("--require-container-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-rollout-corpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fail-if-not-ready",
        action="store_true",
        help="Return nonzero unless every required submit-readiness check passes.",
    )
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def read_export_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def cli_arg_present(*names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:] for name in names)


def set_artifact_default(
    args: argparse.Namespace,
    attr: str,
    env_name: str,
    cli_name: str,
    default: Path | str,
) -> None:
    if env_name not in os.environ and not cli_arg_present(cli_name):
        setattr(args, attr, default)


def apply_artifact_path_defaults(args: argparse.Namespace) -> None:
    root = args.artifact_root
    set_artifact_default(args, "input_data", "INPUT_DATA", "--input-data", root / "data/qwen3_235b_swe_rollout_conversations.jsonl")
    set_artifact_default(args, "hidden_states_dir", "HIDDEN_STATES_DIR", "--hidden-states-dir", root / "hidden_states")
    set_artifact_default(args, "output_dir", "OUTPUT_DIR", "--output-dir", root / "modelopt_ckpt")
    set_artifact_default(args, "trained_ckpt", "TRAINED_CKPT", "--trained-ckpt", root / "modelopt_ckpt")
    set_artifact_default(args, "export_dir", "EXPORT_DIR", "--export-dir", root / "exported_hf")
    set_artifact_default(args, "vllm_draft_dir", "VLLM_DRAFT_DIR", "--vllm-draft-dir", root / "vllm_draft")
    set_artifact_default(args, "verifier_config_dir", "VERIFIER_CONFIG_DIR", "--verifier-config-dir", root / "verifier_config")
    set_artifact_default(args, "chat_template", "CHAT_TEMPLATE", "--chat-template", root / "templates/qwen3_generation_template.jinja2")
    set_artifact_default(args, "reference_arch", "REFERENCE_ARCH", "--reference-arch", root / "architecture/eagle3_architecture.json")
    set_artifact_default(args, "arch_env_file", "ARCH_ENV_FILE", "--arch-env-file", root / "architecture/eagle3_architecture.env")
    set_artifact_default(
        args,
        "container_preflight_json",
        "CONTAINER_PREFLIGHT_JSON",
        "--container-preflight-json",
        root / "reports/container_preflight_analysis.json",
    )
    set_artifact_default(
        args,
        "corpus_strategy_json",
        "CORPUS_STRATEGY_JSON",
        "--corpus-strategy-json",
        root / "reports/corpus_strategy.json",
    )
    set_artifact_default(
        args,
        "rollout_state_json",
        "ROLLOUT_STATE_ADVANCE_JSON",
        "--rollout-state-json",
        root / "reports/rollout_capture_state_advance.json",
    )
    set_artifact_default(
        args,
        "slurm_capacity_json",
        "SLURM_CAPACITY_JSON",
        "--slurm-capacity-json",
        root / "reports/eagle3_slurm_capacity.json",
    )
    set_artifact_default(
        args,
        "slurm_capacity_markdown",
        "SLURM_CAPACITY_MARKDOWN",
        "--slurm-capacity-markdown",
        root / "reports/eagle3_slurm_capacity.md",
    )
    set_artifact_default(
        args,
        "slurm_capacity_env",
        "SLURM_CAPACITY_ENV",
        "--slurm-capacity-env",
        root / "reports/eagle3_resource_profile.env",
    )
    set_artifact_default(args, "mounts", "MOUNTS", "--mounts", f"/lustre:/lustre,{ROOT}:{ROOT},{root}:{root}")


def apply_resource_profile_defaults(args: argparse.Namespace) -> None:
    if "SLURM_CAPACITY_ENV" not in os.environ and not cli_arg_present("--slurm-capacity-env"):
        args.slurm_capacity_env = args.artifact_root / "reports/eagle3_resource_profile.env"
    profile = read_export_env(args.slurm_capacity_env)
    if not profile:
        return
    mappings = [
        ("DUMP_GPUS_PER_NODE", "dump_gpus_per_node", ("--dump-gpus-per-node",)),
        ("TRAIN_GPUS_PER_NODE", "train_gpus_per_node", ("--train-gpus-per-node",)),
        ("EXPORT_GPUS_PER_NODE", "export_gpus_per_node", ("--export-gpus-per-node",)),
        ("TP", "tp", ("--tp",)),
    ]
    for env_name, attr, cli_names in mappings:
        if env_name in os.environ or cli_arg_present(*cli_names):
            continue
        value = profile.get(env_name)
        if value is None:
            continue
        try:
            setattr(args, attr, int(value))
        except ValueError:
            continue


def shell_join(env: dict[str, str], command: list[str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    body = " ".join(shlex.quote(part) for part in command)
    return f"{prefix} {body}".strip()


def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
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
            "output_tail": result.stdout[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "command": shell_join(env or {}, cmd),
            "returncode": 124,
            "timeout": timeout,
            "output": output,
            "output_tail": output[-8000:],
        }


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"not visible: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid json: {exc}"


def add(checks: list[dict[str, Any]], area: str, name: str, status: str, detail: str, **evidence: Any) -> None:
    checks.append({"area": area, "name": name, "status": status, "detail": detail, "evidence": evidence})


def is_true(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def path_check(checks: list[dict[str, Any]], name: str, path: Path, *, required: bool = True, nonempty: bool = False) -> None:
    exists = path.exists()
    ok = exists and (not nonempty or path.stat().st_size > 0)
    if ok:
        add(checks, "paths", name, "pass", f"visible: {path}", path=str(path), size_bytes=path.stat().st_size if path.is_file() else None)
    elif required:
        detail = f"not visible: {path}" if not exists else f"empty file: {path}"
        status = "missing" if not exists else "fail"
        add(checks, "paths", name, status, detail, path=str(path), exists=exists)
    else:
        detail = f"not visible yet: {path}" if not exists else f"empty file: {path}"
        add(checks, "paths", name, "warn", detail, path=str(path), exists=exists)


def check_paths(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path_check(checks, "submit_eagle3_pipeline.sh", EXP / "submit_eagle3_pipeline.sh")
    path_check(checks, "slurm_preflight.sbatch", EXP / "slurm_preflight.sbatch")
    path_check(checks, "slurm_dump_hidden_states.sbatch", EXP / "slurm_dump_hidden_states.sbatch")
    path_check(checks, "slurm_validate_hidden_states.sbatch", EXP / "slurm_validate_hidden_states.sbatch")
    path_check(checks, "slurm_offline_train.sbatch", EXP / "slurm_offline_train.sbatch")
    path_check(checks, "slurm_export_vllm.sbatch", EXP / "slurm_export_vllm.sbatch")
    path_check(checks, "ModelOpt launch_train.sh", args.modelopt_dir / "examples/speculative_decoding/launch_train.sh")
    path_check(
        checks,
        "ModelOpt TRT-LLM hidden-state dumper",
        args.modelopt_dir / "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py",
    )
    path_check(checks, "input conversation JSONL", args.input_data, nonempty=True)
    path_check(checks, "chat template", args.chat_template, nonempty=True)
    path_check(checks, "verifier config.json", args.verifier_config_dir / "config.json", nonempty=True)
    path_check(checks, "Eagle3 architecture JSON", args.reference_arch, nonempty=True)
    path_check(checks, "Eagle3 architecture env", args.arch_env_file, nonempty=True)
    if args.sbatch_account in {"", "dummy", "<account>"}:
        add(checks, "paths", "Slurm account", "fail", "SBATCH_ACCOUNT must be a real account", sbatch_account=args.sbatch_account)
    else:
        add(checks, "paths", "Slurm account", "pass", "SBATCH_ACCOUNT is set", sbatch_account=args.sbatch_account)
    if args.container:
        path_check(checks, "container image", Path(args.container), required=args.require_container_preflight, nonempty=False)
    elif args.require_container_preflight:
        add(checks, "paths", "container image", "fail", "container image is required for Slurm pipeline submit")


def check_chat_template(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.chat_template.exists():
        return
    text = args.chat_template.read_text(encoding="utf-8", errors="replace")
    has_tags = "generation" in text and "endgeneration" in text
    add(
        checks,
        "data",
        "answer-only chat template tags",
        "pass" if has_tags else "fail",
        "generation/endgeneration tags present" if has_tags else "generation/endgeneration tags missing",
        path=str(args.chat_template),
    )


def check_modelopt_loss_mask_patch(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    result = run(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/validate_modelopt_loss_mask_patch.py",
            "--modelopt-dir",
            str(args.modelopt_dir),
        ],
        timeout=120,
    )
    status = "pass" if result["returncode"] == 0 else "fail"
    add(
        checks,
        "modelopt",
        "TRT-LLM loss-mask patch",
        status,
        "ModelOpt TRT-LLM dumper and wrapper preserve answer-only loss_mask"
        if status == "pass"
        else "ModelOpt TRT-LLM dumper loss_mask patch validation failed",
        command=result["command"],
        returncode=result["returncode"],
        output_tail=result["output_tail"],
    )


def check_corpus_reports(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    corpus, corpus_error = load_json(args.corpus_strategy_json)
    if corpus_error:
        status = "missing" if args.require_rollout_corpus else "warn"
        add(checks, "data", "corpus strategy", status, corpus_error, path=str(args.corpus_strategy_json))
    else:
        overall = corpus.get("overall_status")
        decision = corpus.get("decision") or {}
        alignment = corpus.get("rollout_alignment") if isinstance(corpus.get("rollout_alignment"), dict) else {}
        provenance = decision.get("provenance") if isinstance(decision.get("provenance"), dict) else alignment
        expected_primary = "actual_rl_rollout" if args.target_context == "swe_rl" else decision.get("primary_source")
        rollout_proven = (
            args.target_context != "swe_rl"
            or (
                decision.get("primary_source") == expected_primary
                and provenance.get("proves_actual_rollout_corpus") is True
                and provenance.get("output_matches_input") is True
                and provenance.get("input_valid") is True
            )
        )
        ok = overall == "pass" and rollout_proven
        add(
            checks,
            "data",
            "corpus strategy",
            "pass" if ok else ("missing" if args.require_rollout_corpus and overall != "fail" else ("fail" if args.require_rollout_corpus else "warn")),
            "corpus strategy proves target-aligned rollout corpus"
            if ok
            else "corpus strategy does not yet prove the required target corpus",
            overall_status=overall,
            target_context=corpus.get("target_context"),
            primary_source=decision.get("primary_source"),
            rollout_provenance=provenance,
            next_action=decision.get("next_action"),
        )

    rollout, rollout_error = load_json(args.rollout_state_json)
    if rollout_error:
        status = "missing" if args.require_rollout_corpus else "warn"
        add(checks, "data", "rollout state advance", status, rollout_error, path=str(args.rollout_state_json))
    else:
        decision = rollout.get("decision") or {}
        ok = decision.get("overall_status") == "pass" and decision.get("next_step") == "pipeline_dry_run"
        add(
            checks,
            "data",
            "rollout state advance",
            "pass" if ok else ("missing" if args.require_rollout_corpus and decision.get("overall_status") != "fail" else ("fail" if args.require_rollout_corpus else "warn")),
            "rollout state is ready for hidden-state pipeline"
            if ok
            else "rollout state has not reached pipeline_dry_run",
            decision=decision,
        )


def check_container_preflight(checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload, error = load_json(args.container_preflight_json)
    if error:
        status = "missing" if args.require_container_preflight else "warn"
        add(checks, "execution", "container preflight", status, error, path=str(args.container_preflight_json))
        return
    ok = payload.get("overall_status") == "pass" and payload.get("status") == "pass"
    add(
        checks,
        "execution",
        "container preflight",
        "pass" if ok else ("missing" if args.require_container_preflight and payload.get("overall_status") != "fail" else ("fail" if args.require_container_preflight else "warn")),
        "container preflight passed"
        if ok
        else "container preflight has not proved the selected runtime image",
        overall_status=payload.get("overall_status"),
        preflight_status=payload.get("status"),
        preflight_detail=payload.get("detail"),
        container=payload.get("container"),
    )


def check_training_conversations(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not args.input_data.exists():
        result = {"returncode": 1, "output_tail": "input data missing"}
        add(
            checks,
            "data",
            "training conversation validation",
            "missing",
            "skipped because input conversation JSONL is missing",
            path=str(args.input_data),
        )
        return result
    validation_json = args.input_data.with_suffix(".submit_preflight_validation.json")
    result = run(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/validate_training_conversations.py",
            str(args.input_data),
            "--max-seq-len",
            "16384",
            "--json-out",
            str(validation_json),
        ]
    )
    parsed = None
    if validation_json.exists():
        try:
            parsed = json.loads(validation_json.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
    ok = result["returncode"] == 0
    add(
        checks,
        "data",
        "training conversation validation",
        "pass" if ok else "fail",
        "conversation JSONL validates for ModelOpt hidden-state dump"
        if ok
        else "conversation JSONL validation failed",
        returncode=result["returncode"],
        validation_json=str(validation_json),
        summary=parsed,
        output_tail=result["output_tail"],
    )
    valid_rows = int((parsed or {}).get("valid_rows") or 0)
    if is_true(args.run_pilot):
        row_ok = ok and valid_rows >= args.min_pilot_rows
        add(
            checks,
            "data",
            "pilot minimum rows",
            "pass" if row_ok else "fail",
            f"RUN_PILOT=true requires at least {args.min_pilot_rows} valid conversations"
            if row_ok
            else f"RUN_PILOT=true input has {valid_rows} valid conversations, below required {args.min_pilot_rows}",
            valid_rows=valid_rows,
            min_pilot_rows=args.min_pilot_rows,
            run_pilot=args.run_pilot,
        )
    return result


def check_modelopt_local_preflight(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline.py",
        "--artifact-root",
        str(args.artifact_root),
        "--input-data",
        str(args.input_data),
        "--hidden-states-dir",
        str(args.hidden_states_dir),
        "--output-dir",
        str(args.output_dir),
        "--trained-ckpt",
        str(args.trained_ckpt),
        "--export-dir",
        str(args.export_dir),
        "--vllm-draft-dir",
        str(args.vllm_draft_dir),
        "--verifier-config-dir",
        str(args.verifier_config_dir),
        "--chat-template",
        str(args.chat_template),
        "--base-model",
        args.base_model,
        "--modelopt-dir",
        str(args.modelopt_dir),
        "--reference-arch",
        str(args.reference_arch),
        "--sbatch-account",
        args.sbatch_account,
    ]
    env = {"ARCH_ENV_FILE": str(args.arch_env_file)}
    result = run(cmd, env=env)
    ok = result["returncode"] == 0 and "Preflight passed." in result["output"]
    add(
        checks,
        "validation",
        "local ModelOpt pipeline preflight",
        "pass" if ok else "fail",
        "preflight_eagle3_pipeline.py passed with the intended paths"
        if ok
        else "preflight_eagle3_pipeline.py did not pass with the intended paths",
        returncode=result["returncode"],
        output_tail=result["output_tail"],
    )
    return result


def check_slurm_capacity(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    result = run(
        [
            "python3",
            "experiments/eagle3_qwen3_235b/probe_eagle3_slurm_capacity.py",
            "--artifact-root",
            str(args.artifact_root),
            "--sbatch-partition",
            args.sbatch_partition,
            "--dump-gpus-per-node",
            str(args.dump_gpus_per_node),
            "--train-gpus-per-node",
            str(args.train_gpus_per_node),
            "--export-gpus-per-node",
            str(args.export_gpus_per_node),
            "--tp",
            str(args.tp),
            "--json-out",
            str(args.slurm_capacity_json),
            "--markdown-out",
            str(args.slurm_capacity_markdown),
            "--env-out",
            str(args.slurm_capacity_env),
        ]
    )
    payload, error = load_json(args.slurm_capacity_json)
    probe_status = "unknown" if error else str((payload or {}).get("overall_status") or "unknown")
    if probe_status == "pass":
        status = "pass"
        detail = "Slurm partition GPU shape fits hidden-state/train/export requests"
    elif probe_status == "fail":
        status = "fail"
        detail = "Slurm partition GPU shape does not fit the requested pipeline resources"
    else:
        status = "warn"
        detail = "Slurm partition GPU shape could not be fully proven"
    add(
        checks,
        "slurm",
        "GPU capacity vs pipeline requests",
        status,
        detail,
        probe_status=probe_status,
        probe_json=str(args.slurm_capacity_json),
        returncode=result["returncode"],
        recommendations=(payload or {}).get("recommendations") if payload else None,
        output_tail=result["output_tail"],
    )
    return result


def check_wrapper_dry_runs(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "DRY_RUN": "true",
        "MODELOPT_DIR": str(args.modelopt_dir),
        "ARCH_ENV_FILE": str(args.arch_env_file),
        "BASE_MODEL": args.base_model,
        "INPUT_DATA": str(args.input_data),
        "HIDDEN_STATES_DIR": str(args.hidden_states_dir),
        "OUTPUT_DIR": str(args.output_dir),
        "TRAINED_CKPT": str(args.trained_ckpt),
        "EXPORT_DIR": str(args.export_dir),
        "VLLM_DRAFT_DIR": str(args.vllm_draft_dir),
        "TRAINING_CKPT_VALIDATION_JSON": str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
        "TRAINING_CKPT_VALIDATION_MARKDOWN": str(args.artifact_root / "reports/eagle3_training_checkpoint.md"),
        "EXPORT_ARTIFACTS_JSON": str(args.artifact_root / "reports/eagle3_export_artifacts.json"),
        "EXPORT_ARTIFACTS_MARKDOWN": str(args.artifact_root / "reports/eagle3_export_artifacts.md"),
        "VERIFIER_CONFIG_DIR": str(args.verifier_config_dir),
        "TRAINING_SEQ_LEN": str(args.training_seq_len),
        "MAX_SEQ_LEN": str(args.max_seq_len),
        "CHAT_TEMPLATE": str(args.chat_template),
        "ANSWER_ONLY_LOSS": args.answer_only_loss,
        "RUN_CONFIG_COMPARE": "true",
        "REFERENCE_ARCH": str(args.reference_arch),
    }
    commands = {
        "dump_hidden_states": ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh"],
        "offline_train": ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh"],
        "export_vllm": ["bash", "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_export_vllm.sh"],
    }
    results: dict[str, Any] = {}
    for name, cmd in commands.items():
        result = run(cmd, env=env)
        results[name] = result
        output = result["output"]
        if name == "dump_hidden_states":
            markers = ["--answer-only-loss", "--aux-layers", "--chat-template", f"--max-seq-len {args.max_seq_len}"]
        elif name == "offline_train":
            markers = [
                "data.offline_data_path=",
                f"training.answer_only_loss={args.answer_only_loss}",
                f"training.training_seq_len={args.training_seq_len}",
                "eagle.eagle_architecture_config.eagle_aux_hidden_state_layer_ids=",
            ]
        else:
            markers = ["validate_eagle3_training_checkpoint.py", "export_hf_checkpoint.py", "convert_to_vllm_ckpt.py", "compare_eagle3_configs.py"]
        missing = [marker for marker in markers if marker not in output]
        ok = result["returncode"] == 0 and not missing
        add(
            checks,
            "dry_run",
            name,
            "pass" if ok else "fail",
            f"{name} wrapper dry-run produced expected command"
            if ok
            else f"{name} wrapper dry-run failed or missed expected markers",
            returncode=result["returncode"],
            missing=missing,
            output_tail=result["output_tail"],
        )
    return results


def check_pipeline_dry_run(checks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "SUBMIT": "false",
        "ARTIFACT_ROOT": str(args.artifact_root),
        "RUN_PILOT": args.run_pilot,
        "MIN_PIPELINE_PILOT_ROWS": str(args.min_pilot_rows),
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "DUMP_GPUS_PER_NODE": str(args.dump_gpus_per_node),
        "TRAIN_GPUS_PER_NODE": str(args.train_gpus_per_node),
        "EXPORT_GPUS_PER_NODE": str(args.export_gpus_per_node),
        "TP": str(args.tp),
        "INPUT_DATA": str(args.input_data),
        "HIDDEN_STATES_DIR": str(args.hidden_states_dir),
        "OUTPUT_DIR": str(args.output_dir),
        "TRAINED_CKPT": str(args.trained_ckpt),
        "EXPORT_DIR": str(args.export_dir),
        "VLLM_DRAFT_DIR": str(args.vllm_draft_dir),
        "TRAINING_CKPT_VALIDATION_JSON": str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
        "TRAINING_CKPT_VALIDATION_MARKDOWN": str(args.artifact_root / "reports/eagle3_training_checkpoint.md"),
        "EXPORT_ARTIFACTS_JSON": str(args.artifact_root / "reports/eagle3_export_artifacts.json"),
        "EXPORT_ARTIFACTS_MARKDOWN": str(args.artifact_root / "reports/eagle3_export_artifacts.md"),
        "VERIFIER_CONFIG_DIR": str(args.verifier_config_dir),
        "BASE_MODEL": args.base_model,
        "TRAINING_SEQ_LEN": str(args.training_seq_len),
        "MAX_SEQ_LEN": str(args.max_seq_len),
        "ANSWER_ONLY_LOSS": args.answer_only_loss,
        "CHAT_TEMPLATE": str(args.chat_template),
        "MODELOPT_DIR": str(args.modelopt_dir),
        "ARCH_ENV_FILE": str(args.arch_env_file),
        "REFERENCE_ARCH": str(args.reference_arch),
        "CONTAINER": args.container,
        "MOUNTS": args.mounts,
    }
    result = run(["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"], env=env)
    output = result["output"]
    markers = [
        "slurm_preflight.sbatch",
        "slurm_dump_hidden_states.sbatch",
        "slurm_validate_hidden_states.sbatch",
        "slurm_offline_train.sbatch",
        "slurm_export_vllm.sbatch",
        "--dependency=afterok:VALIDATE_HIDDENS_JOB_ID",
        f"ARTIFACT_ROOT={args.artifact_root}",
        f"DUMP_GPUS_PER_NODE={args.dump_gpus_per_node}",
        f"TRAIN_GPUS_PER_NODE={args.train_gpus_per_node}",
        f"EXPORT_GPUS_PER_NODE={args.export_gpus_per_node}",
        f"TP={args.tp}",
        f"--gres=gpu:{args.dump_gpus_per_node}",
        f"--gres=gpu:{args.train_gpus_per_node}",
        f"--gres=gpu:{args.export_gpus_per_node}",
        "HIDDEN_STATES_VALIDATION_JSON=",
        "TRAINING_CKPT_VALIDATION_JSON=",
        "EXPORT_ARTIFACTS_JSON=",
        f"TRAINING_SEQ_LEN={args.training_seq_len}",
        f"MAX_SEQ_LEN={args.max_seq_len}",
        f"ANSWER_ONLY_LOSS={args.answer_only_loss}",
        f"CHAT_TEMPLATE={args.chat_template}",
        f"REFERENCE_ARCH={args.reference_arch}",
    ]
    if is_true(args.run_pilot):
        markers.extend(["RUN_PILOT=true", "DEBUG_MAX_NUM_CONVERSATIONS=8", "DATA_SAMPLE_SIZE=8", "MAX_STEPS=20"])
    missing = [marker for marker in markers if marker not in output]
    ok = result["returncode"] == 0 and not missing
    add(
        checks,
        "dry_run",
        "Slurm pipeline dry-run",
        "pass" if ok else "fail",
        "submit_eagle3_pipeline.sh dry-run includes preflight, dump, hidden validation, train, export, and dependencies"
        if ok
        else "submit_eagle3_pipeline.sh dry-run missed expected pipeline evidence",
        returncode=result["returncode"],
        missing=missing,
        output_tail=result["output_tail"],
    )
    return result


def overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "missing" for check in checks):
        return "incomplete"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def command_env(args: argparse.Namespace, submit: bool) -> dict[str, str]:
    return {
        "SUBMIT": "true" if submit else "false",
        "ARTIFACT_ROOT": str(args.artifact_root),
        "RUN_PILOT": args.run_pilot,
        "SBATCH_ACCOUNT": args.sbatch_account,
        "SBATCH_PARTITION": args.sbatch_partition,
        "DUMP_GPUS_PER_NODE": str(args.dump_gpus_per_node),
        "TRAIN_GPUS_PER_NODE": str(args.train_gpus_per_node),
        "EXPORT_GPUS_PER_NODE": str(args.export_gpus_per_node),
        "TP": str(args.tp),
        "INPUT_DATA": str(args.input_data),
        "HIDDEN_STATES_DIR": str(args.hidden_states_dir),
        "OUTPUT_DIR": str(args.output_dir),
        "TRAINED_CKPT": str(args.trained_ckpt),
        "EXPORT_DIR": str(args.export_dir),
        "VLLM_DRAFT_DIR": str(args.vllm_draft_dir),
        "TRAINING_CKPT_VALIDATION_JSON": str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
        "TRAINING_CKPT_VALIDATION_MARKDOWN": str(args.artifact_root / "reports/eagle3_training_checkpoint.md"),
        "EXPORT_ARTIFACTS_JSON": str(args.artifact_root / "reports/eagle3_export_artifacts.json"),
        "EXPORT_ARTIFACTS_MARKDOWN": str(args.artifact_root / "reports/eagle3_export_artifacts.md"),
        "VERIFIER_CONFIG_DIR": str(args.verifier_config_dir),
        "BASE_MODEL": args.base_model,
        "TRAINING_SEQ_LEN": str(args.training_seq_len),
        "MAX_SEQ_LEN": str(args.max_seq_len),
        "ANSWER_ONLY_LOSS": args.answer_only_loss,
        "CHAT_TEMPLATE": str(args.chat_template),
        "MODELOPT_DIR": str(args.modelopt_dir),
        "ARCH_ENV_FILE": str(args.arch_env_file),
        "REFERENCE_ARCH": str(args.reference_arch),
        "CONTAINER": args.container,
        "MOUNTS": args.mounts,
    }


def gated_submit_command(args: argparse.Namespace, *, execute: bool) -> str:
    command = [
        "python3",
        "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline_if_ready.py",
        "--artifact-root",
        str(args.artifact_root),
        "--preflight-json",
        str(args.artifact_root / "reports/eagle3_pipeline_submit_preflight.json"),
        "--json-out",
        str(args.artifact_root / "reports/eagle3_pipeline_gated_submit.json"),
        "--markdown-out",
        str(args.artifact_root / "reports/eagle3_pipeline_gated_submit.md"),
    ]
    if execute:
        command.extend(["--execute", "--allow-heavy-gpu"])
    return shell_join({}, command)


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Eagle3 Pipeline Submit Preflight",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Submit ready: **{str(data['submit_ready']).lower()}**",
        (
            "Resource request: "
            f"dump={data['resource_request']['dump_gpus_per_node']} GPU/node, "
            f"train={data['resource_request']['train_gpus_per_node']} GPU/node, "
            f"export={data['resource_request']['export_gpus_per_node']} GPU/node, "
            f"TP={data['resource_request']['tp']}"
        ),
        f"Minimum pilot rows: **{data['min_pilot_rows']}**",
        "",
        "Pilot submit command:",
        "",
        "```bash",
        data["commands"]["pilot_submit"],
        "```",
        "",
        "Gated submit command:",
        "",
        "```bash",
        data["commands"]["gated_pilot_submit"],
        "```",
        "",
        "Gated readiness check:",
        "",
        "```bash",
        data["commands"]["gated_submit_check"],
        "```",
        "",
        "Dry-run command:",
        "",
        "```bash",
        data["commands"]["dry_run"],
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
    apply_artifact_path_defaults(args)
    apply_resource_profile_defaults(args)
    checks: list[dict[str, Any]] = []

    check_paths(checks, args)
    check_chat_template(checks, args)
    check_modelopt_loss_mask_patch(checks, args)
    check_corpus_reports(checks, args)
    check_container_preflight(checks, args)
    conversation_validation = check_training_conversations(checks, args)
    slurm_capacity = check_slurm_capacity(checks, args)
    core_inputs_visible = all(
        path.exists()
        for path in [
            args.input_data,
            args.chat_template,
            args.verifier_config_dir / "config.json",
            args.reference_arch,
            args.arch_env_file,
        ]
    )
    if core_inputs_visible:
        local_preflight = check_modelopt_local_preflight(checks, args)
        wrapper_dry_runs = check_wrapper_dry_runs(checks, args)
        pipeline_dry_run = check_pipeline_dry_run(checks, args)
    else:
        add(
            checks,
            "validation",
            "local ModelOpt pipeline preflight",
            "missing",
            "skipped until corpus, chat template, verifier config, and architecture files are visible",
        )
        add(
            checks,
            "dry_run",
            "ModelOpt wrapper dry-runs",
            "missing",
            "skipped until core pipeline inputs are visible",
        )
        add(
            checks,
            "dry_run",
            "Slurm pipeline dry-run",
            "missing",
            "skipped until core pipeline inputs are visible",
        )
        local_preflight = {"returncode": 1, "output_tail": "skipped: core inputs missing"}
        wrapper_dry_runs = {}
        pipeline_dry_run = {"returncode": 1, "output_tail": "skipped: core inputs missing"}

    status = overall_status(checks)
    submit_ready = status == "pass"
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "submit_ready": submit_ready,
        "artifact_root": str(args.artifact_root),
        "input_data": str(args.input_data),
        "hidden_states_dir": str(args.hidden_states_dir),
        "output_dir": str(args.output_dir),
        "resource_request": {
            "dump_gpus_per_node": args.dump_gpus_per_node,
            "train_gpus_per_node": args.train_gpus_per_node,
            "export_gpus_per_node": args.export_gpus_per_node,
            "tp": args.tp,
        },
        "min_pilot_rows": args.min_pilot_rows,
        "base_model": args.base_model,
        "training_seq_len": args.training_seq_len,
        "max_seq_len": args.max_seq_len,
        "answer_only_loss": args.answer_only_loss,
        "training_checkpoint_json": str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
        "export_dir": str(args.export_dir),
        "vllm_draft_dir": str(args.vllm_draft_dir),
        "verifier_config_dir": str(args.verifier_config_dir),
        "chat_template": str(args.chat_template),
        "modelopt_dir": str(args.modelopt_dir),
        "reference_arch": str(args.reference_arch),
        "arch_env_file": str(args.arch_env_file),
        "require_container_preflight": args.require_container_preflight,
        "require_rollout_corpus": args.require_rollout_corpus,
        "slurm_capacity_json": str(args.slurm_capacity_json),
        "slurm_capacity_env": str(args.slurm_capacity_env),
        "checks": checks,
        "commands": {
            "dry_run": shell_join(command_env(args, submit=False), ["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"]),
            "pilot_submit": shell_join(command_env(args, submit=True), ["bash", "experiments/eagle3_qwen3_235b/submit_eagle3_pipeline.sh"]),
            "gated_submit_check": gated_submit_command(args, execute=False),
            "gated_pilot_submit": gated_submit_command(args, execute=True),
            "analyze_pipeline": shell_join(
                {
                    "ARTIFACT_ROOT": str(args.artifact_root),
                    "INPUT_DATA": str(args.input_data),
                    "HIDDEN_STATES_DIR": str(args.hidden_states_dir),
                    "HIDDEN_STATES_VALIDATION_JSON": str(args.hidden_states_dir / "validation_summary.json"),
                    "OUTPUT_DIR": str(args.output_dir),
                    "TRAINING_CKPT_VALIDATION_JSON": str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
                    "EXPORT_DIR": str(args.export_dir),
                    "VLLM_DRAFT_DIR": str(args.vllm_draft_dir),
                    "EXPORT_ARTIFACTS_JSON": str(args.artifact_root / "reports/eagle3_export_artifacts.json"),
                    "VERIFIER_CONFIG_DIR": str(args.verifier_config_dir),
                },
                [
                    "python3",
                    "experiments/eagle3_qwen3_235b/analyze_eagle3_pipeline.py",
                    "--job-file",
                    "latest_eagle3_pipeline_jobs.txt",
                    "--logs-dir",
                    "logs",
                    "--base-model",
                    args.base_model,
                    "--modelopt-dir",
                    str(args.modelopt_dir),
                    "--verifier-config-dir",
                    str(args.verifier_config_dir),
                    "--reference-arch",
                    str(args.reference_arch),
                    "--arch-env-file",
                    str(args.arch_env_file),
                    "--chat-template",
                    str(args.chat_template),
                    "--container",
                    args.container,
                    "--mounts",
                    args.mounts,
                    "--input-data",
                    str(args.input_data),
                    "--hidden-states-dir",
                    str(args.hidden_states_dir),
                    "--hidden-validation-json",
                    str(args.hidden_states_dir / "validation_summary.json"),
                    "--training-checkpoint-json",
                    str(args.artifact_root / "reports/eagle3_training_checkpoint.json"),
                    "--output-dir",
                    str(args.output_dir),
                    "--export-dir",
                    str(args.export_dir),
                    "--vllm-draft-dir",
                    str(args.vllm_draft_dir),
                    "--export-artifacts-json",
                    str(args.artifact_root / "reports/eagle3_export_artifacts.json"),
                    "--sbatch-account",
                    args.sbatch_account,
                    "--sbatch-partition",
                    args.sbatch_partition,
                    "--run-pilot",
                    args.run_pilot,
                    "--markdown-out",
                    str(args.artifact_root / "reports/eagle3_pipeline_analysis.md"),
                    "--json-out",
                    str(args.artifact_root / "reports/eagle3_pipeline_analysis.json"),
                ],
            ),
        },
        "conversation_validation": conversation_validation,
        "slurm_capacity": slurm_capacity,
        "local_preflight": local_preflight,
        "wrapper_dry_runs": wrapper_dry_runs,
        "pipeline_dry_run": pipeline_dry_run,
    }

    markdown = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown)
    if status == "fail":
        return 1
    if args.fail_if_not_ready and not submit_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
