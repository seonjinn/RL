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

"""Strict model profiles for the CuTeDSL performance harness."""

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)


class StrictProfile(BaseModel):
    """Base class for immutable, fail-closed profile records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactProfile(StrictProfile):
    """Hugging Face inputs whose resolved revisions bind run evidence."""

    model_repo_id: str
    dataset_repo_id: str
    dataset_repo_type: Literal["dataset"]
    dataset_split: str
    dataset_num_rows: int

    @field_validator("model_repo_id", "dataset_repo_id", "dataset_split")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact identifiers must not be empty")
        return value

    @field_validator("dataset_num_rows")
    @classmethod
    def require_positive_rows(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("dataset_num_rows must be a positive integer")
        return value


class TopologyProfile(StrictProfile):
    """Fixed allocation and model-parallel topology."""

    num_nodes: int
    gpus_per_node: int
    segment_size: int | None
    tp: int
    pp: int
    vpp: int | None
    num_layers_in_first_pipeline_stage: int | None
    num_layers_in_last_pipeline_stage: int | None
    cp: int
    ep: int
    etp: int

    @field_validator(
        "num_nodes",
        "gpus_per_node",
        "tp",
        "pp",
        "cp",
        "ep",
        "etp",
    )
    @classmethod
    def require_positive(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("topology values must be positive integers")
        return value

    @field_validator(
        "segment_size",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
    )
    @classmethod
    def require_positive_segment(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value < 1):
            raise ValueError("segment_size must be null or a positive integer")
        return value

    @field_validator("vpp")
    @classmethod
    def require_valid_vpp(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value < 2):
            raise ValueError("vpp must be null or an integer greater than one")
        return value

    @model_validator(mode="after")
    def require_consistent_pipeline_layout(self) -> "TopologyProfile":
        stage_layers = (
            self.num_layers_in_first_pipeline_stage,
            self.num_layers_in_last_pipeline_stage,
        )
        if self.pp == 1 and (self.vpp is not None or any(stage_layers)):
            raise ValueError("PP1 profiles cannot define VPP or uneven stage layers")
        if self.vpp is not None and any(value is None for value in stage_layers):
            raise ValueError("VPP profiles must bind first and last stage layer counts")
        return self


class WorkloadProfile(StrictProfile):
    """Fixed training and rollout workload shape."""

    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: int
    max_total_sequence_length: int
    sequence_packing_enabled: bool
    num_prompts_per_step: int
    num_generations_per_prompt: int

    @field_validator(
        "train_global_batch_size",
        "train_micro_batch_size",
        "logprob_batch_size",
        "max_total_sequence_length",
        "num_prompts_per_step",
        "num_generations_per_prompt",
    )
    @classmethod
    def require_positive(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("workload values must be positive integers")
        return value


class RuntimeProfile(StrictProfile):
    """Precision and implementation invariants shared by timing arms."""

    policy_precision: str
    rollout_precision: str
    generation_tensor_parallel_size: int
    generation_gpu_memory_utilization: float
    generation_colocated: bool
    generation_num_nodes: int | None
    generation_gpus_per_node: int | None
    policy_training_gpu_count: int
    activation_checkpointing: bool
    recompute_granularity: Literal["full", "selective"]
    recompute_method: str | None
    recompute_modules: list[str] | None
    allow_full_cg: bool
    allow_a2a: bool

    @field_validator("policy_precision", "rollout_precision")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime precision must not be empty")
        return value

    @field_validator("generation_tensor_parallel_size", "policy_training_gpu_count")
    @classmethod
    def require_positive_tp(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("generation tensor parallel size must be positive")
        return value

    @field_validator("generation_num_nodes", "generation_gpus_per_node")
    @classmethod
    def require_positive_optional_allocation(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value < 1):
            raise ValueError("generation allocation values must be positive")
        return value

    @field_validator("generation_gpu_memory_utilization")
    @classmethod
    def require_gpu_memory_fraction(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("generation GPU memory utilization must be in (0, 1]")
        return value


class ProvenanceProfile(StrictProfile):
    """Required cache and converted-checkpoint provenance contract."""

    triton_cache_scope: Literal["job_node_local"]
    megatron_checkpoint_scope: Literal["job_shared"]
    megatron_checkpoint_root_name: Literal["megatron_checkpoints"]
    megatron_checkpoint_marker: Literal["iter_0000000/run_config.yaml"]


class ModelProfile(StrictProfile):
    """Complete compatibility contract for one performance recipe."""

    schema_version: Literal[1]
    profile_id: str
    display_name: str
    recipe: str
    default_contexts: str
    artifacts: ArtifactProfile
    topology: TopologyProfile
    workload: WorkloadProfile
    runtime: RuntimeProfile
    provenance: ProvenanceProfile

    def canonical_json(self) -> str:
        """Return this profile's stable compatibility serialization."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        """Return this profile's compatibility digest."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @field_validator("profile_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9_]+", value) is None:
            raise ValueError("profile_id must match [a-z0-9_]+")
        return value

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        if not value.strip() or len(value) > 128:
            raise ValueError("display_name must contain 1 to 128 characters")
        return value

    @field_validator("default_contexts")
    @classmethod
    def validate_contexts(cls, value: str) -> str:
        contexts = value.split(",")
        if (
            not contexts
            or len(contexts) != len(set(contexts))
            or any(
                context not in {"g0a0", "g1a0", "g0a1", "g1a1"} for context in contexts
            )
        ):
            raise ValueError("default_contexts must contain unique factorial contexts")
        return value

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.startswith("examples/configs/recipes/")
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("recipe must be a contained recipe path")
        return value


def canonical_profile_json(profile: ModelProfile) -> str:
    """Return the stable compatibility identity serialization."""
    return profile.canonical_json()


def profile_sha256(profile: ModelProfile) -> str:
    """Return the digest used to bind submissions and evidence."""
    return profile.sha256()


def load_model_profile(path: Path) -> ModelProfile:
    """Load a strict profile without coercing unknown or malformed fields."""
    try:
        return ModelProfile.model_validate_json(path.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError(f"Invalid model profile {path}: {error}") from error


def _resolved_contract(config: dict[str, Any]) -> dict[str, Any]:
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    vllm = policy["generation"]["vllm_cfg"]
    generation = policy["generation"]["colocated"]
    generation_resources = generation["resources"]
    allocation_gpu_count = (
        config["cluster"]["num_nodes"] * config["cluster"]["gpus_per_node"]
    )
    generation_gpu_count = 0
    if not generation["enabled"]:
        generation_gpu_count = (
            generation_resources["num_nodes"] * generation_resources["gpus_per_node"]
        )
    return {
        "model_name": policy["model_name"],
        "topology": {
            "num_nodes": config["cluster"]["num_nodes"],
            "gpus_per_node": config["cluster"]["gpus_per_node"],
            "segment_size": config["cluster"]["segment_size"],
            "tp": megatron["tensor_model_parallel_size"],
            "pp": megatron["pipeline_model_parallel_size"],
            "vpp": megatron.get("virtual_pipeline_model_parallel_size"),
            "num_layers_in_first_pipeline_stage": megatron.get(
                "num_layers_in_first_pipeline_stage"
            ),
            "num_layers_in_last_pipeline_stage": megatron.get(
                "num_layers_in_last_pipeline_stage"
            ),
            "cp": megatron["context_parallel_size"],
            "ep": megatron["expert_model_parallel_size"],
            "etp": megatron["expert_tensor_parallel_size"],
        },
        "workload": {
            "train_global_batch_size": policy["train_global_batch_size"],
            "train_micro_batch_size": policy["train_micro_batch_size"],
            "logprob_batch_size": policy["logprob_batch_size"],
            "max_total_sequence_length": policy["max_total_sequence_length"],
            "sequence_packing_enabled": policy["sequence_packing"]["enabled"],
            "num_prompts_per_step": config["grpo"]["num_prompts_per_step"],
            "num_generations_per_prompt": config["grpo"]["num_generations_per_prompt"],
        },
        "runtime": {
            "policy_precision": policy["precision"],
            "rollout_precision": vllm["precision"],
            "generation_tensor_parallel_size": vllm["tensor_parallel_size"],
            "generation_gpu_memory_utilization": vllm["gpu_memory_utilization"],
            "generation_colocated": generation["enabled"],
            "generation_num_nodes": generation_resources["num_nodes"],
            "generation_gpus_per_node": generation_resources["gpus_per_node"],
            "policy_training_gpu_count": allocation_gpu_count - generation_gpu_count,
            "activation_checkpointing": megatron["activation_checkpointing"],
            "recompute_granularity": megatron["recompute_granularity"],
            "recompute_method": megatron.get("recompute_method"),
            "recompute_modules": megatron.get("recompute_modules"),
            "moe_grouped_gemm": megatron["moe_grouped_gemm"],
            "moe_router_dtype": megatron["moe_router_dtype"],
            "use_transformer_engine_op_fuser": megatron[
                "use_transformer_engine_op_fuser"
            ],
            "moe_mlp_glu_interleave_size": megatron["moe_mlp_glu_interleave_size"],
            "fp8_enabled": megatron["fp8_cfg"]["enabled"],
            "fp8_format": megatron["fp8_cfg"]["fp8"],
            "fp8_recipe": megatron["fp8_cfg"]["fp8_recipe"],
            "fp8_param": megatron["fp8_cfg"]["fp8_param"],
            "cuda_graph_impl": megatron.get("cuda_graph_impl"),
            "overlap_moe_expert_parallel_comm": megatron.get(
                "overlap_moe_expert_parallel_comm"
            ),
            "high_priority_a2a_comm_stream": megatron.get(
                "high_priority_a2a_comm_stream"
            ),
            "delay_wgrad_compute": megatron.get("delay_wgrad_compute"),
        },
    }


def validate_resolved_recipe(
    profile: ModelProfile,
    repo_root: Path,
    overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve the real recipe offline and compare every profile-fixed field."""
    sys.path.insert(0, str(repo_root.resolve()))
    # NeMo-RL config imports are deferred so the shell-export command stays lightweight.
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    recipe_path = (repo_root / profile.recipe).resolve()
    resolved_root = repo_root.resolve()
    if resolved_root not in recipe_path.parents or not recipe_path.is_file():
        raise ValueError(
            f"profile recipe is missing or escapes repository: {profile.recipe}"
        )
    config = parse_hydra_overrides(load_config(recipe_path), list(overrides))
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("resolved recipe must be a mapping")
    contract = _resolved_contract(cast(dict[str, Any], value))

    if contract["model_name"] != profile.artifacts.model_repo_id:
        raise ValueError("resolved model identity does not match profile")
    if contract["topology"] != profile.topology.model_dump():
        raise ValueError("resolved topology does not match profile")
    if contract["workload"] != profile.workload.model_dump():
        raise ValueError("resolved workload does not match profile")
    runtime = contract["runtime"]
    runtime_profile = profile.runtime
    expected_runtime = {
        "policy_precision": runtime_profile.policy_precision,
        "rollout_precision": runtime_profile.rollout_precision,
        "generation_tensor_parallel_size": (
            runtime_profile.generation_tensor_parallel_size
        ),
        "generation_gpu_memory_utilization": (
            runtime_profile.generation_gpu_memory_utilization
        ),
        "generation_colocated": runtime_profile.generation_colocated,
        "generation_num_nodes": runtime_profile.generation_num_nodes,
        "generation_gpus_per_node": runtime_profile.generation_gpus_per_node,
        "policy_training_gpu_count": runtime_profile.policy_training_gpu_count,
        "activation_checkpointing": runtime_profile.activation_checkpointing,
        "recompute_granularity": runtime_profile.recompute_granularity,
        "recompute_method": runtime_profile.recompute_method,
        "recompute_modules": runtime_profile.recompute_modules,
    }
    if any(runtime[key] != expected for key, expected in expected_runtime.items()):
        raise ValueError(
            "resolved precision or generation runtime does not match profile"
        )
    fixed_runtime = {
        "moe_grouped_gemm": True,
        "moe_router_dtype": "fp32",
        "use_transformer_engine_op_fuser": True,
        "moe_mlp_glu_interleave_size": 32,
        "fp8_enabled": True,
        "fp8_format": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
        "cuda_graph_impl": "none",
        "overlap_moe_expert_parallel_comm": False,
        "high_priority_a2a_comm_stream": False,
        "delay_wgrad_compute": False,
    }
    if any(runtime[key] != expected for key, expected in fixed_runtime.items()):
        raise ValueError(
            "resolved CuTeDSL prerequisite contract does not match profile"
        )
    return contract


def shell_exports(profile: ModelProfile, profile_path: Path) -> dict[str, str]:
    """Return profile fields required by the shell submitter and payload."""
    return {
        "CUTEDSL_MODEL_PROFILE_PATH": str(profile_path.resolve()),
        "CUTEDSL_MODEL_PROFILE_ID": profile.profile_id,
        "CUTEDSL_MODEL_PROFILE_SHA256": profile_sha256(profile),
        "CUTEDSL_PROFILE_RECIPE": profile.recipe,
        "CUTEDSL_PROFILE_DEFAULT_CONTEXTS": profile.default_contexts,
        "CUTEDSL_MODEL_REPO_ID": profile.artifacts.model_repo_id,
        "CUTEDSL_DATASET_REPO_ID": profile.artifacts.dataset_repo_id,
        "CUTEDSL_DATASET_REPO_TYPE": profile.artifacts.dataset_repo_type,
        "CUTEDSL_DATASET_SPLIT": profile.artifacts.dataset_split,
        "CUTEDSL_DATASET_NUM_ROWS": str(profile.artifacts.dataset_num_rows),
        "CUTEDSL_MEGATRON_CHECKPOINT_SCOPE": (
            profile.provenance.megatron_checkpoint_scope
        ),
        "CUTEDSL_MEGATRON_CHECKPOINT_ROOT_NAME": (
            profile.provenance.megatron_checkpoint_root_name
        ),
        "CUTEDSL_MEGATRON_CHECKPOINT_MARKER": (
            profile.provenance.megatron_checkpoint_marker
        ),
        "CUTEDSL_EXPECTED_TRITON_CACHE_SCOPE": (profile.provenance.triton_cache_scope),
        "CUTEDSL_PROFILE_NUM_NODES": str(profile.topology.num_nodes),
        "CUTEDSL_PROFILE_GPUS_PER_NODE": str(profile.topology.gpus_per_node),
        "CUTEDSL_PROFILE_SEGMENT_SIZE": str(
            profile.topology.segment_size or profile.topology.num_nodes
        ),
        "CUTEDSL_PROFILE_CONFIG_SEGMENT_SIZE": (
            "null"
            if profile.topology.segment_size is None
            else str(profile.topology.segment_size)
        ),
        "CUTEDSL_PROFILE_TP": str(profile.topology.tp),
        "CUTEDSL_PROFILE_PP": str(profile.topology.pp),
        "CUTEDSL_PROFILE_VPP": (
            "null" if profile.topology.vpp is None else str(profile.topology.vpp)
        ),
        "CUTEDSL_PROFILE_FIRST_STAGE_LAYERS": (
            "null"
            if profile.topology.num_layers_in_first_pipeline_stage is None
            else str(profile.topology.num_layers_in_first_pipeline_stage)
        ),
        "CUTEDSL_PROFILE_LAST_STAGE_LAYERS": (
            "null"
            if profile.topology.num_layers_in_last_pipeline_stage is None
            else str(profile.topology.num_layers_in_last_pipeline_stage)
        ),
        "CUTEDSL_PROFILE_CP": str(profile.topology.cp),
        "CUTEDSL_PROFILE_EP": str(profile.topology.ep),
        "CUTEDSL_PROFILE_ETP": str(profile.topology.etp),
        "CUTEDSL_PROFILE_TRAIN_GLOBAL_BATCH_SIZE": str(
            profile.workload.train_global_batch_size
        ),
        "CUTEDSL_PROFILE_TRAIN_MICRO_BATCH_SIZE": str(
            profile.workload.train_micro_batch_size
        ),
        "CUTEDSL_PROFILE_LOGPROB_BATCH_SIZE": str(profile.workload.logprob_batch_size),
        "CUTEDSL_PROFILE_MAX_TOTAL_SEQUENCE_LENGTH": str(
            profile.workload.max_total_sequence_length
        ),
        "CUTEDSL_PROFILE_SEQUENCE_PACKING_ENABLED": str(
            profile.workload.sequence_packing_enabled
        ).lower(),
        "CUTEDSL_PROFILE_NUM_PROMPTS_PER_STEP": str(
            profile.workload.num_prompts_per_step
        ),
        "CUTEDSL_PROFILE_NUM_GENERATIONS_PER_PROMPT": str(
            profile.workload.num_generations_per_prompt
        ),
        "CUTEDSL_PROFILE_ALLOW_FULL_CG": str(profile.runtime.allow_full_cg).lower(),
        "CUTEDSL_PROFILE_ALLOW_A2A": str(profile.runtime.allow_a2a).lower(),
        "CUTEDSL_PROFILE_POLICY_TRAINING_GPU_COUNT": str(
            profile.runtime.policy_training_gpu_count
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
    profile = load_model_profile(args.profile)
    if args.command == "shell":
        for key, value in shell_exports(profile, args.profile).items():
            print(f"export {key}={shlex.quote(value)}")
        return 0
    contract = validate_resolved_recipe(profile, args.repo_root)
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
