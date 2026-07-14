#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free model-profile preflight for cluster login nodes."""

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any


PROFILE_FIELDS = {
    "schema_version",
    "profile_id",
    "display_name",
    "recipe",
    "default_contexts",
    "artifacts",
    "topology",
    "workload",
    "runtime",
    "provenance",
}
ARTIFACT_FIELDS = {
    "model_repo_id",
    "dataset_repo_id",
    "dataset_repo_type",
    "dataset_split",
    "dataset_num_rows",
}
TOPOLOGY_FIELDS = {
    "num_nodes",
    "gpus_per_node",
    "segment_size",
    "tp",
    "pp",
    "vpp",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "cp",
    "ep",
    "etp",
}
WORKLOAD_FIELDS = {
    "train_global_batch_size",
    "train_micro_batch_size",
    "logprob_batch_size",
    "max_total_sequence_length",
    "sequence_packing_enabled",
    "num_prompts_per_step",
    "num_generations_per_prompt",
}
RUNTIME_FIELDS = {
    "policy_precision",
    "rollout_precision",
    "generation_tensor_parallel_size",
    "generation_gpu_memory_utilization",
    "generation_colocated",
    "generation_num_nodes",
    "generation_gpus_per_node",
    "policy_training_gpu_count",
    "activation_checkpointing",
    "recompute_granularity",
    "recompute_method",
    "recompute_modules",
    "allow_full_cg",
    "allow_a2a",
}
PROVENANCE_FIELDS = {
    "triton_cache_scope",
    "megatron_checkpoint_scope",
    "megatron_checkpoint_root_name",
    "megatron_checkpoint_marker",
}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], label: str
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"missing {label} fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"unexpected {label} fields: {', '.join(unexpected)}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, label)


def _validate_profile(profile: Any, *, path: Path) -> dict[str, Any]:
    root = _require_mapping(profile, "profile")
    _require_exact_fields(root, PROFILE_FIELDS, "profile")
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise ValueError("schema_version must be integer 1")
    profile_id = _require_string(root["profile_id"], "profile_id")
    if re.fullmatch(r"[a-z0-9_]+", profile_id) is None:
        raise ValueError("profile_id must match [a-z0-9_]+")
    display_name = _require_string(root["display_name"], "display_name")
    if len(display_name) > 128:
        raise ValueError("display_name must contain at most 128 characters")
    recipe = _require_string(root["recipe"], "recipe")
    recipe_path = PurePosixPath(recipe)
    if (
        not recipe.startswith("examples/configs/recipes/")
        or recipe_path.is_absolute()
        or ".." in recipe_path.parts
    ):
        raise ValueError("recipe must be a contained recipe path")
    contexts = _require_string(root["default_contexts"], "default_contexts").split(",")
    allowed_contexts = {"g0a0", "g1a0", "g0a1", "g1a1"}
    if len(contexts) != len(set(contexts)) or any(
        context not in allowed_contexts for context in contexts
    ):
        raise ValueError("default_contexts must contain unique factorial contexts")

    artifacts = _require_mapping(root["artifacts"], "artifacts")
    _require_exact_fields(artifacts, ARTIFACT_FIELDS, "artifact")
    for key in ("model_repo_id", "dataset_repo_id", "dataset_split"):
        _require_string(artifacts[key], f"artifacts.{key}")
    if artifacts["dataset_repo_type"] != "dataset":
        raise ValueError("artifacts.dataset_repo_type must be dataset")
    _require_positive_int(artifacts["dataset_num_rows"], "artifacts.dataset_num_rows")

    topology = _require_mapping(root["topology"], "topology")
    _require_exact_fields(topology, TOPOLOGY_FIELDS, "topology")
    for key in ("num_nodes", "gpus_per_node", "tp", "pp", "cp", "ep", "etp"):
        _require_positive_int(topology[key], f"topology.{key}")
    for key in (
        "segment_size",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
    ):
        _require_optional_positive_int(topology[key], f"topology.{key}")
    vpp = topology["vpp"]
    if vpp is not None and (type(vpp) is not int or vpp < 2):
        raise ValueError("topology.vpp must be null or an integer greater than one")
    stage_layers = (
        topology["num_layers_in_first_pipeline_stage"],
        topology["num_layers_in_last_pipeline_stage"],
    )
    if topology["pp"] == 1 and (vpp is not None or any(stage_layers)):
        raise ValueError("PP1 profiles cannot define VPP or uneven stage layers")
    if vpp is not None and any(value is None for value in stage_layers):
        raise ValueError("VPP profiles must bind first and last stage layer counts")

    workload = _require_mapping(root["workload"], "workload")
    _require_exact_fields(workload, WORKLOAD_FIELDS, "workload")
    for key in WORKLOAD_FIELDS - {"sequence_packing_enabled"}:
        _require_positive_int(workload[key], f"workload.{key}")
    _require_bool(
        workload["sequence_packing_enabled"],
        "workload.sequence_packing_enabled",
    )

    runtime = _require_mapping(root["runtime"], "runtime")
    _require_exact_fields(runtime, RUNTIME_FIELDS, "runtime")
    for key in ("policy_precision", "rollout_precision"):
        _require_string(runtime[key], f"runtime.{key}")
    for key in ("generation_tensor_parallel_size", "policy_training_gpu_count"):
        _require_positive_int(runtime[key], f"runtime.{key}")
    for key in ("generation_num_nodes", "generation_gpus_per_node"):
        _require_optional_positive_int(runtime[key], f"runtime.{key}")
    utilization = runtime["generation_gpu_memory_utilization"]
    if type(utilization) not in (int, float) or type(utilization) is bool:
        raise ValueError("runtime.generation_gpu_memory_utilization must be numeric")
    if not 0.0 < utilization <= 1.0:
        raise ValueError("runtime.generation_gpu_memory_utilization must be in (0, 1]")
    for key in (
        "generation_colocated",
        "activation_checkpointing",
        "allow_full_cg",
        "allow_a2a",
    ):
        _require_bool(runtime[key], f"runtime.{key}")
    if runtime["recompute_granularity"] not in {"full", "selective"}:
        raise ValueError("runtime.recompute_granularity must be full or selective")
    recompute_method = runtime["recompute_method"]
    if recompute_method is not None and not isinstance(recompute_method, str):
        raise ValueError("runtime.recompute_method must be null or a string")
    recompute_modules = runtime["recompute_modules"]
    if recompute_modules is not None and (
        not isinstance(recompute_modules, list)
        or any(not isinstance(module, str) for module in recompute_modules)
    ):
        raise ValueError("runtime.recompute_modules must be null or a string list")

    provenance = _require_mapping(root["provenance"], "provenance")
    _require_exact_fields(provenance, PROVENANCE_FIELDS, "provenance")
    expected_provenance = {
        "triton_cache_scope": "job_node_local",
        "megatron_checkpoint_scope": "job_shared",
        "megatron_checkpoint_root_name": "megatron_checkpoints",
        "megatron_checkpoint_marker": "iter_0000000/run_config.yaml",
    }
    if provenance != expected_provenance:
        raise ValueError("profile provenance contract does not match the harness")
    if not path.is_file():
        raise ValueError(f"profile does not exist: {path}")
    return root


def load_profile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid model profile {path}: {error}") from error
    return _validate_profile(payload, path=path)


def profile_sha256(profile: dict[str, Any]) -> str:
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def shell_exports(profile: dict[str, Any], profile_path: Path) -> dict[str, str]:
    artifacts = profile["artifacts"]
    topology = profile["topology"]
    workload = profile["workload"]
    runtime = profile["runtime"]
    provenance = profile["provenance"]
    return {
        "CUTEDSL_MODEL_PROFILE_PATH": str(profile_path.resolve()),
        "CUTEDSL_MODEL_PROFILE_ID": profile["profile_id"],
        "CUTEDSL_MODEL_PROFILE_SHA256": profile_sha256(profile),
        "CUTEDSL_PROFILE_RECIPE": profile["recipe"],
        "CUTEDSL_PROFILE_DEFAULT_CONTEXTS": profile["default_contexts"],
        "CUTEDSL_MODEL_REPO_ID": artifacts["model_repo_id"],
        "CUTEDSL_DATASET_REPO_ID": artifacts["dataset_repo_id"],
        "CUTEDSL_DATASET_REPO_TYPE": artifacts["dataset_repo_type"],
        "CUTEDSL_DATASET_SPLIT": artifacts["dataset_split"],
        "CUTEDSL_DATASET_NUM_ROWS": str(artifacts["dataset_num_rows"]),
        "CUTEDSL_MEGATRON_CHECKPOINT_SCOPE": provenance["megatron_checkpoint_scope"],
        "CUTEDSL_MEGATRON_CHECKPOINT_ROOT_NAME": provenance[
            "megatron_checkpoint_root_name"
        ],
        "CUTEDSL_MEGATRON_CHECKPOINT_MARKER": provenance["megatron_checkpoint_marker"],
        "CUTEDSL_EXPECTED_TRITON_CACHE_SCOPE": provenance["triton_cache_scope"],
        "CUTEDSL_PROFILE_NUM_NODES": str(topology["num_nodes"]),
        "CUTEDSL_PROFILE_GPUS_PER_NODE": str(topology["gpus_per_node"]),
        "CUTEDSL_PROFILE_SEGMENT_SIZE": str(
            topology["segment_size"] or topology["num_nodes"]
        ),
        "CUTEDSL_PROFILE_CONFIG_SEGMENT_SIZE": (
            "null"
            if topology["segment_size"] is None
            else str(topology["segment_size"])
        ),
        "CUTEDSL_PROFILE_TP": str(topology["tp"]),
        "CUTEDSL_PROFILE_PP": str(topology["pp"]),
        "CUTEDSL_PROFILE_VPP": (
            "null" if topology["vpp"] is None else str(topology["vpp"])
        ),
        "CUTEDSL_PROFILE_FIRST_STAGE_LAYERS": (
            "null"
            if topology["num_layers_in_first_pipeline_stage"] is None
            else str(topology["num_layers_in_first_pipeline_stage"])
        ),
        "CUTEDSL_PROFILE_LAST_STAGE_LAYERS": (
            "null"
            if topology["num_layers_in_last_pipeline_stage"] is None
            else str(topology["num_layers_in_last_pipeline_stage"])
        ),
        "CUTEDSL_PROFILE_CP": str(topology["cp"]),
        "CUTEDSL_PROFILE_EP": str(topology["ep"]),
        "CUTEDSL_PROFILE_ETP": str(topology["etp"]),
        "CUTEDSL_PROFILE_TRAIN_GLOBAL_BATCH_SIZE": str(
            workload["train_global_batch_size"]
        ),
        "CUTEDSL_PROFILE_TRAIN_MICRO_BATCH_SIZE": str(
            workload["train_micro_batch_size"]
        ),
        "CUTEDSL_PROFILE_LOGPROB_BATCH_SIZE": str(workload["logprob_batch_size"]),
        "CUTEDSL_PROFILE_MAX_TOTAL_SEQUENCE_LENGTH": str(
            workload["max_total_sequence_length"]
        ),
        "CUTEDSL_PROFILE_SEQUENCE_PACKING_ENABLED": str(
            workload["sequence_packing_enabled"]
        ).lower(),
        "CUTEDSL_PROFILE_NUM_PROMPTS_PER_STEP": str(workload["num_prompts_per_step"]),
        "CUTEDSL_PROFILE_NUM_GENERATIONS_PER_PROMPT": str(
            workload["num_generations_per_prompt"]
        ),
        "CUTEDSL_PROFILE_ALLOW_FULL_CG": str(runtime["allow_full_cg"]).lower(),
        "CUTEDSL_PROFILE_ALLOW_A2A": str(runtime["allow_a2a"]).lower(),
        "CUTEDSL_PROFILE_POLICY_TRAINING_GPU_COUNT": str(
            runtime["policy_training_gpu_count"]
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("shell", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--profile", type=Path, required=True)
    subparsers.choices["validate"].add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile = load_profile(args.profile)
    if args.command == "shell":
        for key, value in shell_exports(profile, args.profile).items():
            print(f"export {key}={shlex.quote(value)}")
        return 0
    repo_root = args.repo_root.resolve()
    recipe_path = (repo_root / profile["recipe"]).resolve()
    if repo_root not in recipe_path.parents or not recipe_path.is_file():
        raise ValueError(
            f"profile recipe is missing or escapes repository: {recipe_path}"
        )
    print(json.dumps({"profile_sha256": profile_sha256(profile)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
