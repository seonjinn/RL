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
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


SAFE_STAGES = frozenset({1, 10, 100})
TARGET_REPO = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFTER_REPO = "z-lab/Qwen3-8B-DFlash-b16"
DRAFTER_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}; got {actual!r}")


def validate_stage(steps: int) -> int:
    """Return a safe staged step count or fail loudly."""
    if steps not in SAFE_STAGES:
        raise ValueError(f"stage steps must be 1, 10, or 100; got {steps}")
    return steps


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the immutable cross-arm experiment contract."""
    experiment = config["experiment"]
    grpo = config["grpo"]
    policy = config["policy"]
    generation = policy["generation"]
    vllm_cfg = generation["vllm_cfg"]
    speculative = generation["vllm_kwargs"]["speculative_config"]
    data = config["data"]
    logger = config["logger"]
    megatron_cfg = policy["megatron_cfg"]

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
        "policy.draft.enabled": (policy["draft"]["enabled"], False),
        "speculative_config.method": (speculative["method"], "dflash"),
        "speculative_config.num_speculative_tokens": (
            speculative["num_speculative_tokens"],
            15,
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
    }
    for name, (actual, expected) in expected_values.items():
        _require_equal(actual, expected, name=name)

    if policy["model_name"].split("/")[-2:] != ["snapshots", TARGET_REVISION]:
        raise ValueError("policy.model_name must use the exact target snapshot")
    if policy["tokenizer"]["name"] != policy["model_name"]:
        raise ValueError("tokenizer must use the exact target snapshot")
    if speculative["model"].split("/")[-2:] != ["snapshots", DRAFTER_REVISION]:
        raise ValueError("speculative_config.model must use the exact draft snapshot")

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
        "acceptance_metrics_enabled": vllm_cfg["enable_vllm_metrics_logger"],
        "fixed_prompt_panel_enabled": experiment["fixed_prompt_panel"],
    }


def main() -> None:
    """Validate one config and print its normalized contract as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    validate_stage(args.steps)
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(validate_config(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
