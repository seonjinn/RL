# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-loud contract validation for the fixed DFlash experiment."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


SAFE_STAGES = frozenset({1, 10, 100})
SWEEP_K_VALUES = frozenset({3, 5, 7, 9})
MAX_NUM_SEQS = 8
CUDAGRAPH_CAPTURE_SIZES = (
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    16,
    18,
    20,
    24,
    28,
    30,
    32,
    36,
    40,
    42,
    48,
    50,
    56,
    60,
    64,
    70,
    80,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
)
TARGET_REPO = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFTER_REPO = "z-lab/Qwen3-8B-DFlash-b16"
DRAFTER_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
WANDB_PROJECT = "sna-nemo-rl-fixed-drafter"
WANDB_GROUP = "qwen3-8b-dflash-fixed-drafter-k-sweep"


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}; got {actual!r}")


def validate_stage(steps: int) -> int:
    """Return a safe staged step count or fail loudly."""
    if steps not in SAFE_STAGES:
        raise ValueError(f"stage steps must be 1, 10, or 100; got {steps}")
    return steps


def validate_sweep_k(k: int) -> int:
    """Return a supported K-sweep arm or fail loudly."""
    if k not in SWEEP_K_VALUES:
        raise ValueError(f"sweep K must be 3, 5, 7, or 9; got {k}")
    return k


def validate_k_stage(k: int, steps: int) -> tuple[int, int]:
    """Keep new K-sweep arms at the one-step gate until both are green."""
    validate_sweep_k(k)
    validate_stage(steps)
    if steps != 1:
        raise ValueError(f"K={k} is currently allowed only the 1-step gate")
    return k, steps


def _merge_config(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and value.get("_override_") is True:
            merged[key] = {
                nested_key: copy.deepcopy(nested_value)
                for nested_key, nested_value in value.items()
                if nested_key != "_override_"
            }
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the experiment's simple YAML inheritance for contract validation."""
    raw_config = yaml.safe_load(config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    defaults = raw_config.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: dict[str, Any] = {}
    for default in defaults:
        merged = _merge_config(merged, load_config(config_path.parent / default))
    return _merge_config(merged, raw_config)


def _expected_wandb_tags(k: int) -> list[str]:
    return [
        "fixed-drafter",
        "dflash",
        "qwen3-8b",
        f"k{k}",
        "cudagraph",
        "target-only-grpo",
        "seed42",
        "step001",
    ]


def _expected_wandb_config(k: int) -> dict[str, Any]:
    return {
        "experiment": "fixed-drafter-qwen3-8b-dflash-k-sweep",
        "git_sha": "${oc.env:EXPECTED_HEAD}",
        "target_repo": TARGET_REPO,
        "target_revision": TARGET_REVISION,
        "drafter_repo": DRAFTER_REPO,
        "drafter_revision": DRAFTER_REVISION,
        "drafter_config_sha256": (
            "9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102"
        ),
        "container_sha256": (
            "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
        ),
        "runtime_vllm_version": "0.25.1",
        "k": k,
        "compilation_mode": 3,
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": list(CUDAGRAPH_CAPTURE_SIZES),
        "max_num_seqs": MAX_NUM_SEQS,
        "max_dflash_decode_query_tokens": MAX_NUM_SEQS * (k + 1),
        "per_position_acceptance_positions": list(range(1, k + 1)),
        "seed": 42,
        "stage_steps": 1,
        "training_tp": 2,
        "training_dp": 2,
        "target_tp": 1,
        "draft_tp": 1,
        "draft_training_enabled": False,
        "draft_refit_enabled": False,
    }


def validate_config(
    config: Mapping[str, Any],
    *,
    expected_k: int = 15,
    require_wandb: bool = False,
) -> dict[str, Any]:
    """Validate and summarize the immutable cross-arm experiment contract."""
    experiment = config["experiment"]
    grpo = config["grpo"]
    policy = config["policy"]
    generation = policy["generation"]
    vllm_cfg = generation["vllm_cfg"]
    vllm_kwargs = generation["vllm_kwargs"]
    speculative = vllm_kwargs["speculative_config"]
    compilation = vllm_kwargs.get("compilation_config", {})
    data = config["data"]
    logger = config["logger"]
    megatron_cfg = policy["megatron_cfg"]
    cluster = config["cluster"]

    expected_values = {
        "experiment.target_repo": (experiment["target_repo"], TARGET_REPO),
        "experiment.target_revision": (
            experiment["target_revision"],
            TARGET_REVISION,
        ),
        "experiment.tokenizer_revision": (
            experiment["tokenizer_revision"],
            TARGET_REVISION,
        ),
        "experiment.drafter_repo": (experiment["drafter_repo"], DRAFTER_REPO),
        "experiment.drafter_revision": (
            experiment["drafter_revision"],
            DRAFTER_REVISION,
        ),
        "grpo.seed": (grpo["seed"], 42),
        "grpo.num_prompts_per_step": (grpo["num_prompts_per_step"], 8),
        "grpo.num_generations_per_prompt": (
            grpo["num_generations_per_prompt"],
            4,
        ),
        "policy.train_global_batch_size": (policy["train_global_batch_size"], 32),
        "policy.train_micro_batch_size": (policy["train_micro_batch_size"], 1),
        "policy.precision": (policy["precision"], "bfloat16"),
        "policy.max_total_sequence_length": (
            policy["max_total_sequence_length"],
            4096,
        ),
        "data.max_input_seq_length": (data["max_input_seq_length"], 2048),
        "policy.generation.max_new_tokens": (generation["max_new_tokens"], 1024),
        "policy.generation.temperature": (generation["temperature"], 1.0),
        "policy.generation.top_p": (generation["top_p"], 1.0),
        "policy.generation.top_k": (generation["top_k"], None),
        "policy.megatron_cfg.optimizer.lr": (
            megatron_cfg["optimizer"]["lr"],
            1.0e-6,
        ),
        "policy.megatron_cfg.scheduler.lr_warmup_iters": (
            megatron_cfg["scheduler"]["lr_warmup_iters"],
            10,
        ),
        "policy.megatron_cfg.tensor_model_parallel_size": (
            megatron_cfg["tensor_model_parallel_size"],
            2,
        ),
        "policy.megatron_cfg.pipeline_model_parallel_size": (
            megatron_cfg["pipeline_model_parallel_size"],
            1,
        ),
        "policy.megatron_cfg.context_parallel_size": (
            megatron_cfg["context_parallel_size"],
            1,
        ),
        "policy.megatron_cfg.sequence_parallel": (
            megatron_cfg["sequence_parallel"],
            True,
        ),
        "policy.draft.enabled": (policy["draft"]["enabled"], False),
        "policy.generation.vllm_cfg.precision": (
            vllm_cfg["precision"],
            "bfloat16",
        ),
        "policy.generation.vllm_cfg.kv_cache_dtype": (
            vllm_cfg["kv_cache_dtype"],
            "auto",
        ),
        "policy.generation.vllm_cfg.tensor_parallel_size": (
            vllm_cfg["tensor_parallel_size"],
            1,
        ),
        "speculative_config.method": (speculative["method"], "dflash"),
        "speculative_config.num_speculative_tokens": (
            speculative["num_speculative_tokens"],
            expected_k,
        ),
        "speculative_config.draft_tensor_parallel_size": (
            speculative["draft_tensor_parallel_size"],
            1,
        ),
        "speculative_config.rejection_sample_method": (
            speculative["rejection_sample_method"],
            "standard",
        ),
        "speculative_config.draft_load_config.load_format": (
            speculative["draft_load_config"]["load_format"],
            "auto",
        ),
        "data.train.dataset_name": (data["train"]["dataset_name"], "DAPOMath17K"),
        "vllm metrics": (vllm_cfg["enable_vllm_metrics_logger"], True),
        "fixed prompt panel": (experiment["fixed_prompt_panel"], True),
        "tensorboard": (logger["tensorboard_enabled"], True),
        "wandb": (logger["wandb_enabled"], require_wandb),
        "cluster.num_nodes": (cluster["num_nodes"], 1),
        "cluster.gpus_per_node": (cluster["gpus_per_node"], 4),
    }
    if require_wandb:
        expected_values.update(
            {
                "policy.generation.vllm_cfg.enforce_eager": (
                    vllm_cfg["enforce_eager"],
                    False,
                ),
                "policy.generation.vllm_kwargs.max_num_seqs": (
                    vllm_kwargs["max_num_seqs"],
                    MAX_NUM_SEQS,
                ),
                "compilation_config.backend": (
                    compilation["backend"],
                    "eager",
                ),
                "compilation_config.mode": (compilation["mode"], 3),
                "compilation_config.cudagraph_mode": (
                    compilation["cudagraph_mode"],
                    "PIECEWISE",
                ),
                "compilation_config.cudagraph_capture_sizes": (
                    compilation["cudagraph_capture_sizes"],
                    list(CUDAGRAPH_CAPTURE_SIZES),
                ),
            }
        )
    for name, (actual, expected) in expected_values.items():
        _require_equal(actual, expected, name=name)

    if policy["model_name"].split("/")[-2:] != ["snapshots", TARGET_REVISION]:
        raise ValueError("policy.model_name must use the exact target snapshot")
    if policy["tokenizer"]["name"] != policy["model_name"]:
        raise ValueError("tokenizer must use the exact target snapshot")
    if speculative["model"].split("/")[-2:] != ["snapshots", DRAFTER_REVISION]:
        raise ValueError("speculative_config.model must use the exact draft snapshot")
    max_query_tokens = MAX_NUM_SEQS * (expected_k + 1)
    if require_wandb and max_query_tokens > CUDAGRAPH_CAPTURE_SIZES[-1]:
        raise ValueError(
            "CUDA graph capture sizes do not cover the maximum DFlash decode query"
        )

    wandb_config = logger.get("wandb", {})
    if require_wandb:
        validate_sweep_k(expected_k)
        expected_wandb_values = {
            "logger.wandb.project": (wandb_config.get("project"), WANDB_PROJECT),
            "logger.wandb.group": (wandb_config.get("group"), WANDB_GROUP),
            "logger.wandb.name": (
                wandb_config.get("name"),
                f"qwen3-8b-dflash-fixed-k{expected_k}-cudagraph-step001-seed42",
            ),
            "logger.wandb.tags": (
                wandb_config.get("tags"),
                _expected_wandb_tags(expected_k),
            ),
            "logger.wandb.config": (
                wandb_config.get("config"),
                _expected_wandb_config(expected_k),
            ),
        }
        for name, (actual, expected) in expected_wandb_values.items():
            _require_equal(actual, expected, name=name)

    return {
        "target_repo": experiment["target_repo"],
        "target_revision": experiment["target_revision"],
        "tokenizer_revision": experiment["tokenizer_revision"],
        "drafter_repo": experiment["drafter_repo"],
        "drafter_revision": experiment["drafter_revision"],
        "num_speculative_tokens": speculative["num_speculative_tokens"],
        "draft_training_enabled": policy["draft"]["enabled"],
        "draft_refit_enabled": False,
        "dataset": data["train"]["dataset_name"],
        "seed": grpo["seed"],
        "num_prompts_per_step": grpo["num_prompts_per_step"],
        "num_generations_per_prompt": grpo["num_generations_per_prompt"],
        "train_global_batch_size": policy["train_global_batch_size"],
        "train_micro_batch_size": policy["train_micro_batch_size"],
        "max_input_seq_length": data["max_input_seq_length"],
        "max_new_tokens": generation["max_new_tokens"],
        "max_total_sequence_length": policy["max_total_sequence_length"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "learning_rate": megatron_cfg["optimizer"]["lr"],
        "warmup_iters": megatron_cfg["scheduler"]["lr_warmup_iters"],
        "training_tp": megatron_cfg["tensor_model_parallel_size"],
        "training_pp": megatron_cfg["pipeline_model_parallel_size"],
        "training_cp": megatron_cfg["context_parallel_size"],
        "training_dp": 2,
        "sequence_parallel": megatron_cfg["sequence_parallel"],
        "target_tp": vllm_cfg["tensor_parallel_size"],
        "draft_tp": speculative["draft_tensor_parallel_size"],
        "precision": policy["precision"],
        "kv_cache_dtype": vllm_cfg["kv_cache_dtype"],
        "enforce_eager": vllm_cfg["enforce_eager"],
        "compilation_mode": compilation.get("mode"),
        "cudagraph_backend": compilation.get("backend"),
        "cudagraph_mode": compilation.get("cudagraph_mode"),
        "cudagraph_metrics": None,
        "cudagraph_capture_sizes": compilation.get("cudagraph_capture_sizes"),
        "max_num_seqs": vllm_kwargs.get("max_num_seqs"),
        "max_dflash_decode_query_tokens": max_query_tokens,
        "per_position_acceptance_positions": list(range(1, expected_k + 1)),
        "acceptance_metrics_enabled": vllm_cfg["enable_vllm_metrics_logger"],
        "fixed_prompt_panel_enabled": experiment["fixed_prompt_panel"],
        "wandb_enabled": logger["wandb_enabled"],
        "wandb_project": wandb_config.get("project"),
        "wandb_group": wandb_config.get("group"),
        "wandb_name": wandb_config.get("name"),
        "wandb_tags": wandb_config.get("tags"),
        "wandb_config": wandb_config.get("config"),
    }


def main() -> None:
    """Validate one config and print its normalized contract as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    validate_stage(args.steps)
    if args.k in SWEEP_K_VALUES:
        validate_k_stage(args.k, args.steps)
    else:
        raise ValueError(f"K must be 3, 5, 7, or 9; got {args.k}")
    config = load_config(args.config)
    print(
        json.dumps(
            validate_config(
                config,
                expected_k=args.k,
                require_wandb=args.k in SWEEP_K_VALUES,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
