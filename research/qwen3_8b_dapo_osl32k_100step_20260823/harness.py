"""Fail-closed orchestration for Qwen3-8B DAPO OSL32K segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


EXPERIMENT = "qwen3_8b_dapo_osl32k_100step_20260823"
PILOT_EXPERIMENT = "qwen3_8b_dapo_osl32k_pilot_20260823"
ARMS = ("baseline-k0", "dflash-k5", "dspark-k5")
ENDPOINTS = (25, 50, 75, 100)
HARNESS_BASE_SHA = "afac9bec73067a81141af5dbdb7a5a972d2ee24d"
PRODUCT_BASE_SHA = "3020cf42c4ec416c83ba2cd78ec5b26ca142c412"
PINNED_PRODUCT_SHA = "ff4dc0f44154c4c3a678b33a8e83c2e71f41628e"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DFLASH_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
DSPARK_REVISION = "03326e5043815da1f81b109078b2889737c26017"
DATASET_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
CONTAINER_SHA256 = "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
USER_ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna"
CONTAINER = f"{USER_ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh"
DATA_SOURCE = (
    f"{USER_ROOT}/hf_home/hub/datasets--BytedTsinghua-SIA--DAPO-Math-17k/"
    f"snapshots/{DATASET_REVISION}/data/dapo-math-17k.parquet"
)
DATASET = (
    f"{USER_ROOT}/experiments/qwen3_8b_dapo_osl32k_pilot_20260823/data/"
    "dapo-math-17k-r658770-first64.jsonl"
)
TARGET = f"{USER_ROOT}/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/{TARGET_REVISION}"
DFLASH = (
    f"{USER_ROOT}/hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/"
    f"snapshots/{DFLASH_REVISION}"
)
DSPARK = (
    f"{USER_ROOT}/hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/"
    f"snapshots/{DSPARK_REVISION}"
)
ACCOUNT = "nemotron_n3_post"


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def repository_root() -> Path:
    return experiment_dir().parents[1]


def pilot_dir() -> Path:
    return repository_root() / "research" / PILOT_EXPERIMENT


def require_full_sha(value: str, *, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must be a full lowercase 40-character SHA")


def require_pinned_product_sha(value: str) -> None:
    require_full_sha(value, label="product SHA")
    if value != PINNED_PRODUCT_SHA:
        raise ValueError(
            f"product SHA must equal pinned product commit {PINNED_PRODUCT_SHA}"
        )


def validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")


def validate_endpoint(endpoint: int) -> None:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}: {endpoint}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, *, excluded: frozenset[str] = frozenset()) -> str:
    if not root.is_dir():
        raise ValueError(f"missing tree: {root}")
    digest = hashlib.sha256()
    members = sorted(path for path in root.rglob("*") if path.is_file())
    for member in members:
        relative = str(member.relative_to(root))
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(member)))
    return digest.hexdigest()


def config_path(arm: str) -> Path:
    validate_arm(arm)
    return experiment_dir() / "configs" / f"{arm}.yaml"


def load_config(arm: str) -> dict[str, Any]:
    config = read_json(config_path(arm))
    grpo = config.get("grpo")
    checkpointing = config.get("checkpointing")
    cadence = config.get("cadence_runtime")
    data = config.get("data")
    policy = config.get("policy")
    if not all(
        isinstance(value, dict)
        for value in (grpo, checkpointing, cadence, data, policy)
    ):
        raise ValueError("config is missing a required object")
    if (
        grpo.get("max_num_steps") != 100
        or grpo.get("max_num_epochs") != 4
        or grpo.get("segment_stop_step") is not None
    ):
        raise ValueError("config must keep the global 100-step/4-epoch horizon")
    if checkpointing != {
        "enabled": True,
        "save_period": 25,
        "save_optimizer": True,
        "keep_top_k": None,
        "metric_name": None,
    }:
        raise ValueError(
            "checkpointing must preserve full optimizer state every 25 steps"
        )
    if cadence != {
        "enabled": True,
        "required_checkpoint_steps": list(ENDPOINTS),
    }:
        raise ValueError("cadence runtime must require every segment endpoint")
    if config.get("data_plane") != {"enabled": True}:
        raise ValueError("data_plane.enabled=true is required")
    train = data.get("train")
    if (
        data.get("shuffle") is not False
        or not isinstance(train, dict)
        or train.get("seed") != 42
        or train.get("data_path") != DATASET
    ):
        raise ValueError("immutable first-64 DAPO ordering is not configured")
    draft = policy.get("draft")
    if not isinstance(draft, dict):
        raise ValueError("policy.draft config is missing")
    if arm == "baseline-k0":
        if draft != {"enabled": False}:
            raise ValueError("baseline must disable draft training")
    elif draft.get("enabled") is not True or draft.get("update_schedule") != {
        "mode": "always"
    }:
        raise ValueError("draft arms must use always update/refit cadence")
    return config


def arm_checkpoint(arm: str) -> str | None:
    validate_arm(arm)
    if arm == "dflash-k5":
        return DFLASH
    if arm == "dspark-k5":
        return DSPARK
    return None


def manifest(arm: str, *, harness_sha: str, product_sha: str) -> dict[str, Any]:
    require_full_sha(harness_sha, label="harness SHA")
    require_pinned_product_sha(product_sha)
    config = load_config(arm)
    dataset_identity = read_json(pilot_dir() / "dataset_identity.json")
    checkpoint_identity = read_json(pilot_dir() / "checkpoint_identity.json")
    draft_checkpoint = arm_checkpoint(arm)
    return {
        "schema_version": 1,
        "arm": arm,
        "harness_base_sha": HARNESS_BASE_SHA,
        "harness_sha": harness_sha,
        "product_base_sha": PRODUCT_BASE_SHA,
        "product_sha": product_sha,
        "config_sha256": sha256_file(config_path(arm)),
        "max_num_steps": config["grpo"]["max_num_steps"],
        "max_num_epochs": config["grpo"]["max_num_epochs"],
        "segment_endpoints": list(ENDPOINTS),
        "target": TARGET,
        "target_revision": TARGET_REVISION,
        "target_identity": checkpoint_identity["target"],
        "draft_checkpoint": draft_checkpoint,
        "draft_revision": None
        if draft_checkpoint is None
        else Path(draft_checkpoint).name,
        "dataset_source": DATA_SOURCE,
        "dataset_path": DATASET,
        "dataset_revision": dataset_identity["source"]["revision"],
        "dataset_source_sha256": dataset_identity["source"]["sha256"],
        "dataset_slice_sha256": dataset_identity["slice"]["sha256"],
        "dataset_rows": dataset_identity["slice"]["rows"],
        "dataset_source_order": dataset_identity["slice"]["source_order"],
        "dataset_seed": dataset_identity["slice"]["seed"],
        "container": CONTAINER,
        "container_sha256": CONTAINER_SHA256,
        "topology": {
            "nodes": 1,
            "gpus_per_node": 4,
            "tp": 2,
            "pp": 1,
            "cp": 1,
            "dp": 2,
        },
        "max_input_length": 2048,
        "max_output_length": 32768,
        "max_model_len": 40960,
        "global_batch_size": 8,
    }


def stable_wandb_id(arm: str, *, harness_sha: str, product_sha: str) -> str:
    payload = manifest(arm, harness_sha=harness_sha, product_sha=product_sha)
    identity = canonical_sha256(
        {
            "arm": arm,
            "harness_sha": harness_sha,
            "product_sha": product_sha,
            "config_sha256": payload["config_sha256"],
        }
    )
    return f"q8-dapo-osl32k-100-{arm}-{identity[:20]}"


def chain_plan(arm: str, *, harness_sha: str, product_sha: str) -> list[dict[str, Any]]:
    run_id = stable_wandb_id(arm, harness_sha=harness_sha, product_sha=product_sha)
    result = []
    for index, endpoint in enumerate(ENDPOINTS):
        predecessor = None if index == 0 else ENDPOINTS[index - 1]
        result.append(
            {
                "arm": arm,
                "endpoint": endpoint,
                "predecessor_endpoint": predecessor,
                "dependency_type": None if predecessor is None else "afterok",
                "wandb_run_id": run_id,
                "wandb_resume": "never" if predecessor is None else "must",
                "max_num_steps": 100,
                "segment_stop_step": endpoint,
            }
        )
    return result


def materialize_config(
    arm: str, *, product_root: Path, result_dir: Path, output: Path
) -> None:
    config = load_config(arm)
    recipe = config["defaults"]
    if not isinstance(recipe, str) or not recipe.startswith("examples/configs/"):
        raise ValueError("committed defaults must be a product-relative recipe path")
    config["defaults"] = str(product_root / recipe)
    config["checkpointing"] = {
        **config["checkpointing"],
        "checkpoint_dir": str(result_dir / "checkpoints"),
    }
    config["cadence_runtime"] = {
        **config["cadence_runtime"],
        "result_dir": str(result_dir),
    }
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def write_if_identical(path: Path, payload: dict[str, Any]) -> None:
    contents = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != contents:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def render_chain(
    arm: str,
    *,
    output_root: Path,
    harness_sha: str,
    product_sha: str,
    product_root: Path | None,
) -> list[dict[str, Any]]:
    plan = chain_plan(arm, harness_sha=harness_sha, product_sha=product_sha)
    source_root = product_root or Path(
        f"/home/sna/nemorl-q8-dapo32k-100step-{arm}-{product_sha[:12]}"
    )
    result_dir = output_root / arm
    result_dir.mkdir(parents=True, exist_ok=True)
    arm_manifest = manifest(arm, harness_sha=harness_sha, product_sha=product_sha)
    arm_manifest["product_root"] = str(source_root)
    arm_manifest["result_dir"] = str(result_dir)
    arm_manifest["wandb_run_id"] = plan[0]["wandb_run_id"]
    write_if_identical(result_dir / "manifest.json", arm_manifest)
    rendered = []
    assets = (
        "verify_dapo_slice.py",
        "verify_model_identity.py",
        "check_checkpoint_state_dict.py",
        "summarize_output_lengths.py",
        "dataset_identity.json",
        "checkpoint_identity.json",
    )
    for segment in plan:
        endpoint = segment["endpoint"]
        artifact_dir = result_dir / "jobs" / f"step_{endpoint}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = artifact_dir / f"resolved-{arm}.yaml"
        materialize_config(
            arm,
            product_root=source_root,
            result_dir=result_dir,
            output=resolved_config,
        )
        shutil.copy2(experiment_dir() / "harness.py", artifact_dir / "harness.py")
        shutil.copy2(
            experiment_dir() / "run_segment.sh", artifact_dir / "run_segment.sh"
        )
        (artifact_dir / "configs").mkdir(exist_ok=True)
        shutil.copy2(config_path(arm), artifact_dir / "configs" / f"{arm}.yaml")
        for asset in assets:
            shutil.copy2(pilot_dir() / asset, artifact_dir / asset)
        sbatch_path = artifact_dir / "job.sbatch"
        checkpoint = arm_checkpoint(arm) or ""
        method = "" if arm == "baseline-k0" else arm.split("-", 1)[0]
        sbatch_path.write_text(
            "\n".join(
                (
                    "#!/usr/bin/env bash",
                    f"#SBATCH --job-name={ACCOUNT}.q8-32k-{arm}-{endpoint}",
                    f"#SBATCH --account={ACCOUNT}",
                    "#SBATCH --partition=batch",
                    "#SBATCH --qos=normal",
                    "#SBATCH --time=04:00:00",
                    "#SBATCH --nodes=1",
                    "#SBATCH --gpus-per-node=4",
                    f"#SBATCH --output={artifact_dir}/slurm-%j.out",
                    f"#SBATCH --error={artifact_dir}/slurm-%j.err",
                    "set -euo pipefail",
                    f'export CONTAINER="{CONTAINER}"',
                    'export MOUNTS="/lustre:/lustre,/home:/home,/tmp:/tmp"',
                    'export GPUS_PER_NODE="4"',
                    f'export SOURCE_ROOT="{source_root}"',
                    f'export PRODUCT_SHA="{product_sha}"',
                    f'export HARNESS_SHA="{harness_sha}"',
                    f'export ARTIFACT_DIR="{artifact_dir}"',
                    f'export RESULT_DIR="{result_dir}"',
                    f'export CONFIG="{resolved_config}"',
                    f'export DATA_SOURCE="{DATA_SOURCE}"',
                    f'export DATASET="{DATASET}"',
                    f'export TARGET="{TARGET}"',
                    f'export VARIANT="{arm}"',
                    f'export METHOD="{method}"',
                    f'export CHECKPOINT="{checkpoint}"',
                    f'export SEGMENT_STOP_STEP="{endpoint}"',
                    'export MAX_NUM_STEPS="100"',
                    f'export WANDB_RUN_ID="{segment["wandb_run_id"]}"',
                    f'export WANDB_RESUME="{segment["wandb_resume"]}"',
                    f'export BASE_LOG_DIR="{artifact_dir}"',
                    'export NRL_FORCE_REBUILD_VENVS="true"',
                    "export TMPDIR=/tmp",
                    f"export COMMAND='bash \"{artifact_dir}/run_segment.sh\"'",
                    f'exec bash "{source_root}/ray.sub"',
                    "",
                )
            )
        )
        sbatch_path.chmod(0o700)
        rendered.append(
            {
                **segment,
                "artifact_dir": str(artifact_dir),
                "sbatch_path": str(sbatch_path),
            }
        )
    return rendered


def runtime_gates(arm: str, *, endpoint: int, log: Path) -> dict[str, Any]:
    validate_arm(arm)
    validate_endpoint(endpoint)
    text = log.read_text()
    if re.search(
        r"CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|(^|[^A-Za-z])nan([^A-Za-z]|$)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        raise ValueError("fatal OOM, traceback, or NaN pattern in segment log")
    start = 1 if endpoint == ENDPOINTS[0] else endpoint - 24
    cuda_graph = (
        re.search(
            r"CUDAGRAPH_CAPTURE_COMPLETE|Capturing CUDA graphs.*100%|Graph capturing finished",
            text,
        )
        is not None
    )
    first_step = (
        re.search(rf"Step\s+{start}\s*/\s*100", text) is not None
        and re.search(rf"Logged data to .*train_data_step{start}\.jsonl", text)
        is not None
    )
    last_step = (
        re.search(rf"Step\s+{endpoint}\s*/\s*100", text) is not None
        and re.search(rf"Logged data to .*train_data_step{endpoint}\.jsonl", text)
        is not None
    )
    if not cuda_graph or not first_step or not last_step:
        raise ValueError("CUDA Graph or segment boundary step evidence is missing")
    wake_refit = arm == "baseline-k0" or all(
        pattern in text
        for pattern in (
            "wake up tags ['weights']",
            "GPU Memory after refit complete",
            "wake up tags ['kv_cache']",
        )
    )
    if not wake_refit:
        raise ValueError("draft wake/refit evidence is missing")
    return {
        "arm": arm,
        "segment_start_step": start,
        "segment_stop_step": endpoint,
        "cuda_graph": True,
        "first_step_complete": True,
        "last_step_complete": True,
        "wake_refit": True,
        "no_fatal": True,
    }


def decision_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid decision ledger: {path}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("decision ledger contains a non-object row")
    return rows


def validate_checkpoint(result_dir: Path, *, arm: str, endpoint: int) -> dict[str, Any]:
    checkpoint = (result_dir / "checkpoints" / f"step_{endpoint}").resolve()
    receipt = read_json(checkpoint / "cadence-checkpoint-receipt.json")
    decision_count = 0 if arm == "baseline-k0" else endpoint
    required = {
        "schema_version",
        "successful",
        "checkpoint_id",
        "checkpoint_path",
        "completed_policy_steps",
        "current_step",
        "checkpoint_tree_sha256",
        "components",
        "scheduler_state_sha256",
        "draft_update_schedule",
        "applied_draft_snapshot",
        "decision_ledger",
        "decision_ledger_prefixes",
        "ledger_high_water",
        "resumed_from",
        "cadence_terminal_evidence",
    }
    expected_resumed_from = (
        None
        if endpoint == ENDPOINTS[0]
        else str((result_dir / "checkpoints" / f"step_{endpoint - 25}").resolve())
    )
    if (
        set(receipt) != required
        or receipt.get("schema_version") != 1
        or receipt.get("successful") is not True
        or receipt.get("checkpoint_id") != f"step_{endpoint}"
        or receipt.get("current_step") != endpoint
        or receipt.get("completed_policy_steps") != endpoint
        or receipt.get("checkpoint_path") != str(checkpoint)
        or receipt.get("resumed_from") != expected_resumed_from
    ):
        raise ValueError("product checkpoint receipt identity is inconsistent")
    components = receipt.get("components")
    expected_component_paths = {
        "model": "policy/weights",
        "optimizer": "policy/optimizer",
        "dataloader_rng": "train_dataloader.pt",
    }
    if not isinstance(components, dict) or set(components) != set(
        expected_component_paths
    ):
        raise ValueError("product checkpoint component schema is inconsistent")
    component_paths: dict[str, Path] = {}
    for name, expected_relative in expected_component_paths.items():
        binding = components.get(name)
        if not isinstance(binding, dict) or set(binding) != {
            "relative_path",
            "sha256",
        }:
            raise ValueError(f"invalid {name} checkpoint binding")
        relative = binding.get("relative_path")
        if relative != expected_relative:
            raise ValueError(f"unexpected {name} checkpoint path")
        member = (checkpoint / relative).resolve()
        try:
            member.relative_to(checkpoint)
        except ValueError as error:
            raise ValueError(f"{name} checkpoint path escapes checkpoint") from error
        actual_digest = tree_sha256(member) if member.is_dir() else sha256_file(member)
        if binding.get("sha256") != actual_digest:
            raise ValueError(f"{name} checkpoint digest mismatch")
        component_paths[name] = member
    binding = receipt.get("decision_ledger")
    ledger_keys = {
        "relative_path",
        "size_bytes",
        "sha256",
        "first_decision_id",
        "last_decision_id",
        "entry_count",
    }
    if not isinstance(binding, dict) or set(binding) != ledger_keys:
        raise ValueError("product checkpoint decision-ledger binding is missing")
    relative = binding.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("product checkpoint decision-ledger path is invalid")
    ledger = (checkpoint / relative).resolve()
    try:
        ledger.relative_to(checkpoint)
    except ValueError as error:
        raise ValueError("decision ledger escapes checkpoint") from error
    raw = ledger.read_bytes()
    rows = decision_rows(ledger)
    expected_ids = list(range(1, decision_count + 1))
    if (
        binding.get("size_bytes") != len(raw)
        or binding.get("sha256") != hashlib.sha256(raw).hexdigest()
        or binding.get("first_decision_id") != (1 if decision_count else None)
        or binding.get("last_decision_id") != decision_count
        or binding.get("entry_count") != decision_count
        or receipt.get("decision_ledger_prefixes") != [binding]
        or receipt.get("ledger_high_water") != decision_count
        or [row.get("decision_id") for row in rows] != expected_ids
        or [row.get("global_step") for row in rows] != expected_ids
    ):
        raise ValueError("checkpoint decision ledger is not a contiguous bound prefix")
    schedule = receipt.get("draft_update_schedule")
    if not isinstance(schedule, dict) or receipt.get(
        "scheduler_state_sha256"
    ) != canonical_sha256(schedule):
        raise ValueError("checkpoint scheduler digest is invalid")
    state = schedule.get("state")
    if (
        not isinstance(state, dict)
        or state.get("next_decision_id") != decision_count + 1
        or state.get("decisions") != decision_count
        or (arm == "baseline-k0" and schedule.get("mode") != "disabled")
        or (arm != "baseline-k0" and schedule.get("mode") != "always")
    ):
        raise ValueError("checkpoint scheduler/ledger high-water mismatch")
    applied_snapshot = receipt.get("applied_draft_snapshot")
    if arm == "baseline-k0":
        if applied_snapshot is not None:
            raise ValueError("baseline checkpoint must not bind a draft snapshot")
    else:
        if not isinstance(applied_snapshot, dict) or set(applied_snapshot) != {
            "version",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("draft checkpoint applied snapshot is missing")
        snapshot_path = Path(str(applied_snapshot["path"]))
        if (
            applied_snapshot.get("version") != decision_count
            or not snapshot_path.is_file()
            or applied_snapshot.get("size_bytes") != snapshot_path.stat().st_size
            or applied_snapshot.get("sha256") != sha256_file(snapshot_path)
        ):
            raise ValueError("draft checkpoint applied snapshot is invalid")
    if not isinstance(receipt.get("cadence_terminal_evidence"), dict):
        raise ValueError("checkpoint cadence terminal evidence is missing")
    actual_tree = tree_sha256(
        checkpoint, excluded=frozenset({"cadence-checkpoint-receipt.json"})
    )
    if receipt.get("checkpoint_tree_sha256") != actual_tree:
        raise ValueError("checkpoint tree digest is invalid")
    checkpoint_runtime_path = result_dir / f"checkpoint-runtime-step_{endpoint}.json"
    checkpoint_runtime = read_json(checkpoint_runtime_path)
    if checkpoint_runtime != receipt:
        raise ValueError("checkpoint runtime receipt differs from checkpoint receipt")
    return {
        "path": str(checkpoint),
        "tree_sha256": actual_tree,
        "checkpoint_runtime_sha256": sha256_file(checkpoint_runtime_path),
        "decision_ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "decision_count": decision_count,
        "dataloader_sha256": sha256_file(component_paths["dataloader_rng"]),
        "optimizer_tree_sha256": tree_sha256(component_paths["optimizer"]),
    }


def segment_identity(
    arm: str, *, endpoint: int, harness_sha: str, product_sha: str
) -> dict[str, Any]:
    validate_arm(arm)
    validate_endpoint(endpoint)
    require_full_sha(harness_sha, label="harness SHA")
    require_pinned_product_sha(product_sha)
    return {
        "arm": arm,
        "endpoint": endpoint,
        "harness_sha": harness_sha,
        "product_sha": product_sha,
        "config_sha256": sha256_file(config_path(arm)),
    }


def segment_receipt_path(
    result_dir: Path,
    *,
    arm: str,
    endpoint: int,
    harness_sha: str,
    product_sha: str,
) -> Path:
    identity = segment_identity(
        arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    key = canonical_sha256(identity)
    return result_dir / "segment-receipts" / arm / f"step_{endpoint}" / f"{key}.json"


def validate_terminal(
    result_dir: Path, *, arm: str, endpoint: int
) -> dict[str, Any] | None:
    checkpoint_runtime_path = result_dir / "checkpoint-runtime.json"
    schedule_runtime_path = result_dir / "schedule-runtime.json"
    if (result_dir / "terminal.json").exists():
        raise ValueError("unexpected legacy terminal.json artifact")
    if endpoint != ENDPOINTS[-1]:
        if checkpoint_runtime_path.exists() or schedule_runtime_path.exists():
            raise ValueError("intermediate segment must not emit terminal artifacts")
        return None
    checkpoint_runtime = read_json(checkpoint_runtime_path)
    final_checkpoint_receipt = read_json(
        result_dir / "checkpoints" / "step_100" / "cadence-checkpoint-receipt.json"
    )
    if checkpoint_runtime != final_checkpoint_receipt:
        raise ValueError("terminal checkpoint runtime differs from step_100 receipt")
    schedule = read_json(schedule_runtime_path)
    common = {
        "current_step": 100,
        "policy_refit_count": 100,
        "successful_target_refits": 100,
    }
    if any(schedule.get(key) != value for key, value in common.items()):
        raise ValueError("terminal schedule does not prove 100 target refits")
    if arm == "baseline-k0":
        expected = {
            "mode": "disabled",
            "decision_count": 0,
            "successful_updates": 0,
            "successful_draft_refits": 0,
            "updated_steps": [],
            "refit_steps": [],
        }
    else:
        expected = {
            "mode": "always",
            "decision_count": 100,
            "successful_updates": 100,
            "successful_draft_refits": 100,
            "updated_steps": list(range(1, 101)),
            "refit_steps": list(range(1, 101)),
        }
    if any(schedule.get(key) != value for key, value in expected.items()):
        raise ValueError("terminal schedule does not match the selected arm")
    return schedule


def read_segment_receipt(
    result_dir: Path,
    *,
    arm: str,
    endpoint: int,
    harness_sha: str,
    product_sha: str,
) -> tuple[Path, dict[str, Any]]:
    path = segment_receipt_path(
        result_dir,
        arm=arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    receipt = read_json(path)
    identity = segment_identity(
        arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError("segment receipt identity mismatch")
    if receipt.get("receipt_key") != canonical_sha256(identity):
        raise ValueError("segment receipt key mismatch")
    return path, receipt


def segment_preflight(
    result_dir: Path,
    *,
    arm: str,
    endpoint: int,
    harness_sha: str,
    product_sha: str,
) -> dict[str, Any] | None:
    validate_endpoint(endpoint)
    current_receipt = segment_receipt_path(
        result_dir,
        arm=arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    if current_receipt.exists():
        raise ValueError(f"segment already completed exactly once: {current_receipt}")
    index = ENDPOINTS.index(endpoint)
    if index == 0:
        return None
    predecessor = ENDPOINTS[index - 1]
    predecessor_path, receipt = read_segment_receipt(
        result_dir,
        arm=arm,
        endpoint=predecessor,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    checkpoint = validate_checkpoint(result_dir, arm=arm, endpoint=predecessor)
    for key in (
        "tree_sha256",
        "checkpoint_runtime_sha256",
        "decision_ledger_sha256",
        "dataloader_sha256",
        "optimizer_tree_sha256",
    ):
        if receipt["checkpoint"].get(key) != checkpoint[key]:
            raise ValueError(f"predecessor checkpoint {key} drift")
    return {
        "predecessor_endpoint": predecessor,
        "predecessor_receipt": str(predecessor_path),
        "predecessor_receipt_sha256": sha256_file(predecessor_path),
    }


def segment_finalize(
    result_dir: Path,
    *,
    arm: str,
    endpoint: int,
    harness_sha: str,
    product_sha: str,
) -> Path:
    predecessor = segment_preflight(
        result_dir,
        arm=arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    checkpoint = validate_checkpoint(result_dir, arm=arm, endpoint=endpoint)
    gates_path = result_dir / "runtime-gates" / f"step_{endpoint}.json"
    gates = read_json(gates_path)
    expected_gates = {
        "arm": arm,
        "segment_start_step": 1 if endpoint == 25 else endpoint - 24,
        "segment_stop_step": endpoint,
        "cuda_graph": True,
        "first_step_complete": True,
        "last_step_complete": True,
        "wake_refit": True,
        "no_fatal": True,
    }
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise ValueError("runtime gate evidence is incomplete")
    terminal = validate_terminal(result_dir, arm=arm, endpoint=endpoint)
    identity = segment_identity(
        arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    receipt = {
        **identity,
        "receipt_key": canonical_sha256(identity),
        "checkpoint": checkpoint,
        "runtime_gates_path": str(gates_path.resolve()),
        "runtime_gates_sha256": sha256_file(gates_path),
        "predecessor": predecessor,
        "terminal": terminal is not None,
    }
    path = segment_receipt_path(
        result_dir,
        arm=arm,
        endpoint=endpoint,
        harness_sha=harness_sha,
        product_sha=product_sha,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def build_report(
    result_dir: Path, *, arm: str, harness_sha: str, product_sha: str
) -> dict[str, Any]:
    receipt_paths = []
    for endpoint in ENDPOINTS:
        path, _ = read_segment_receipt(
            result_dir,
            arm=arm,
            endpoint=endpoint,
            harness_sha=harness_sha,
            product_sha=product_sha,
        )
        receipt_paths.append(str(path))
    terminal = validate_terminal(result_dir, arm=arm, endpoint=100)
    assert terminal is not None
    return {
        "arm": arm,
        "completed_segment_endpoints": list(ENDPOINTS),
        "completed_policy_steps": terminal["current_step"],
        "terminal": True,
        "segment_receipts": receipt_paths,
        "wandb_run_id": stable_wandb_id(
            arm, harness_sha=harness_sha, product_sha=product_sha
        ),
    }


def submission_identity(arm: str, *, harness_sha: str, product_sha: str) -> str:
    return canonical_sha256(
        {
            "arm": arm,
            "harness_sha": harness_sha,
            "product_sha": product_sha,
            "config_sha256": sha256_file(config_path(arm)),
            "endpoints": ENDPOINTS,
        }
    )


def submit_chain(
    arm: str,
    *,
    output_root: Path,
    harness_sha: str,
    product_sha: str,
    scheduler: Path,
    test_only: bool,
) -> dict[str, Any]:
    rendered = render_chain(
        arm,
        output_root=output_root,
        harness_sha=harness_sha,
        product_sha=product_sha,
        product_root=None,
    )
    key = submission_identity(arm, harness_sha=harness_sha, product_sha=product_sha)
    preflight = output_root / "preflight" / f"{key}.json"
    record = output_root / "submissions" / f"{key}.json"
    if test_only:
        outputs = []
        for segment in rendered:
            result = subprocess.run(
                (str(scheduler), "--test-only", segment["sbatch_path"]),
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(result.stdout.strip())
        payload = {
            "submission_key": key,
            "arm": arm,
            "harness_sha": harness_sha,
            "product_sha": product_sha,
            "config_sha256": sha256_file(config_path(arm)),
            "scheduler_outputs": outputs,
        }
        write_if_identical(preflight, payload)
        return payload
    expected_preflight = read_json(preflight)
    if expected_preflight.get("submission_key") != key:
        raise ValueError("test-only receipt does not match actual chain")
    if record.exists() or Path(f"{record}.lock").exists():
        raise ValueError("actual chain already exists or is in progress")
    record.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(f"{record}.lock")
    with lock.open("x"):
        pass
    job_ids = []
    for segment in rendered:
        argv = [str(scheduler), "--parsable"]
        if job_ids:
            argv.append(f"--dependency=afterok:{job_ids[-1]}")
        argv.append(segment["sbatch_path"])
        result = subprocess.run(argv, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise ValueError(f"scheduler returned an invalid job ID: {job_id!r}")
        job_ids.append(job_id)
    payload = {
        "submission_key": key,
        "arm": arm,
        "harness_sha": harness_sha,
        "product_sha": product_sha,
        "job_ids": job_ids,
        "segment_endpoints": list(ENDPOINTS),
        "dependency_type": "afterok",
    }
    with record.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    lock.unlink()
    return payload


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument("--product-sha", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("manifest", "plan"):
        add_identity_arguments(commands.add_parser(name))
    render = commands.add_parser("render-chain")
    add_identity_arguments(render)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--product-root", type=Path)
    gates = commands.add_parser("runtime-gates")
    gates.add_argument("--arm", choices=ARMS, required=True)
    gates.add_argument("--endpoint", type=int, choices=ENDPOINTS, required=True)
    gates.add_argument("--log", type=Path, required=True)
    gates.add_argument("--output", type=Path)
    for name in ("segment-preflight", "segment-finalize"):
        command = commands.add_parser(name)
        add_identity_arguments(command)
        command.add_argument("--endpoint", type=int, choices=ENDPOINTS, required=True)
        command.add_argument("--result-dir", type=Path, required=True)
    report = commands.add_parser("report")
    add_identity_arguments(report)
    report.add_argument("--result-dir", type=Path, required=True)
    submit = commands.add_parser("submit")
    add_identity_arguments(submit)
    submit.add_argument("--output-root", type=Path, required=True)
    submit.add_argument("--scheduler", type=Path, default=Path("sbatch"))
    mode = submit.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--actual", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        payload: object = manifest(
            args.arm, harness_sha=args.harness_sha, product_sha=args.product_sha
        )
    elif args.command == "plan":
        payload = chain_plan(
            args.arm, harness_sha=args.harness_sha, product_sha=args.product_sha
        )
    elif args.command == "render-chain":
        payload = render_chain(
            args.arm,
            output_root=args.output_root,
            harness_sha=args.harness_sha,
            product_sha=args.product_sha,
            product_root=args.product_root,
        )
    elif args.command == "runtime-gates":
        payload = runtime_gates(args.arm, endpoint=args.endpoint, log=args.log)
        if args.output is not None:
            write_if_identical(args.output, payload)
    elif args.command == "segment-preflight":
        payload = segment_preflight(
            args.result_dir,
            arm=args.arm,
            endpoint=args.endpoint,
            harness_sha=args.harness_sha,
            product_sha=args.product_sha,
        ) or {"predecessor_endpoint": None}
    elif args.command == "segment-finalize":
        path = segment_finalize(
            args.result_dir,
            arm=args.arm,
            endpoint=args.endpoint,
            harness_sha=args.harness_sha,
            product_sha=args.product_sha,
        )
        print(path)
        return 0
    elif args.command == "report":
        payload = build_report(
            args.result_dir,
            arm=args.arm,
            harness_sha=args.harness_sha,
            product_sha=args.product_sha,
        )
    elif args.command == "submit":
        payload = submit_chain(
            args.arm,
            output_root=args.output_root,
            harness_sha=args.harness_sha,
            product_sha=args.product_sha,
            scheduler=args.scheduler,
            test_only=args.test_only,
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"Q8_DAPO32K_100STEP_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
