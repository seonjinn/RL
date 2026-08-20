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

"""Resolved-config parity validation for the no-SpecDec control arm."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, NoReturn, cast


TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
WANDB_PROJECT = "sna-nemo-rl-fixed-drafter"
WANDB_GROUP = "qwen3-8b-dflash-fixed-drafter-k-sweep"
WANDB_NAME = "qwen3-8b-no-specdec-k0"
WANDB_TAGS = ["no-specdec", "k0"]

_SPECULATIVE_PATH = (
    "policy",
    "generation",
    "vllm_kwargs",
    "speculative_config",
)
_IDENTITY_PATHS = (
    ("experiment", "arm"),
    ("experiment", "draft_k"),
    ("grpo", "max_num_steps"),
    ("policy", "megatron_cfg", "train_iters"),
    ("checkpointing", "checkpoint_dir"),
    ("checkpointing", "save_period"),
    ("logger", "log_dir"),
    ("logger", "wandb_enabled"),
    ("logger", "wandb"),
)


class ConfigParityError(ValueError):
    """Raised when the baseline is not a controlled DFlash ablation."""


def _fail(path: str, actual: Any, expected: Any) -> NoReturn:
    raise ConfigParityError(f"{path} must be {expected!r}; got {actual!r}")


def _get(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigParityError(f"{'.'.join(path)} is missing")
        value = value[key]
    return value


def _require(config: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _get(config, path.split("."))
    if actual != expected:
        _fail(path, actual, expected)


def _remove_path(config: MutableMapping[str, Any], path: Sequence[str]) -> None:
    parent: Any = config
    for key in path[:-1]:
        if not isinstance(parent, MutableMapping) or key not in parent:
            return
        parent = parent[key]
    if isinstance(parent, MutableMapping):
        parent.pop(path[-1], None)


def _diff(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child_path = (*path, str(key))
            if key not in left or key not in right:
                differences.append(".".join(child_path))
            else:
                differences.extend(_diff(left[key], right[key], child_path))
        return differences
    if left != right:
        return [".".join(path)]
    return []


def _train_dataset_name(config: Mapping[str, Any]) -> str:
    train = _get(config, ("data", "train"))
    if isinstance(train, Mapping):
        return cast(str, train["dataset_name"])
    if isinstance(train, list) and len(train) == 1 and isinstance(train[0], Mapping):
        return cast(str, train[0]["dataset_name"])
    raise ConfigParityError("data.train must contain exactly one dataset")


def _validate_shared_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "experiment.target_repo": "Qwen/Qwen3-8B",
        "experiment.target_revision": TARGET_REVISION,
        "experiment.tokenizer_revision": TARGET_REVISION,
        "cluster.num_nodes": 1,
        "cluster.gpus_per_node": 4,
        "data.max_input_seq_length": 2048,
        "grpo.seed": 42,
        "grpo.num_prompts_per_step": 8,
        "grpo.num_generations_per_prompt": 4,
        "policy.precision": "bfloat16",
        "policy.train_global_batch_size": 32,
        "policy.train_micro_batch_size": 1,
        "policy.max_total_sequence_length": 4096,
        "policy.draft.enabled": False,
        "policy.megatron_cfg.enabled": True,
        "policy.megatron_cfg.tensor_model_parallel_size": 2,
        "policy.megatron_cfg.pipeline_model_parallel_size": 1,
        "policy.megatron_cfg.context_parallel_size": 1,
        "policy.megatron_cfg.sequence_parallel": True,
        "policy.megatron_cfg.optimizer.bf16": True,
        "policy.megatron_cfg.optimizer.fp16": False,
        "policy.megatron_cfg.optimizer.lr": 1.0e-6,
        "policy.megatron_cfg.scheduler.lr_warmup_iters": 10,
        "policy.generation.backend": "vllm",
        "policy.generation.max_new_tokens": 1024,
        "policy.generation.temperature": 1.0,
        "policy.generation.top_p": 1.0,
        "policy.generation.top_k": None,
        "policy.generation.vllm_cfg.tensor_parallel_size": 1,
        "policy.generation.vllm_cfg.pipeline_parallel_size": 1,
        "policy.generation.vllm_cfg.precision": "bfloat16",
        "policy.generation.vllm_cfg.kv_cache_dtype": "auto",
        "policy.generation.vllm_cfg.max_model_len": 4096,
    }
    for path, expected_value in expected.items():
        _require(config, path, expected_value)
    dataset = _train_dataset_name(config)
    if dataset != "DAPOMath17K":
        _fail("data.train.dataset_name", dataset, "DAPOMath17K")
    model_name = cast(str, _get(config, ("policy", "model_name")))
    if Path(model_name).name != TARGET_REVISION:
        _fail("policy.model_name revision", Path(model_name).name, TARGET_REVISION)
    tokenizer_name = cast(str, _get(config, ("policy", "tokenizer", "name")))
    if tokenizer_name != model_name:
        _fail("policy.tokenizer.name", tokenizer_name, model_name)


def _normalize_baseline_speculative_config(config: MutableMapping[str, Any]) -> None:
    generation = cast(MutableMapping[str, Any], _get(config, ("policy", "generation")))
    vllm_kwargs = generation.get("vllm_kwargs")
    if not isinstance(vllm_kwargs, MutableMapping):
        return
    speculative = vllm_kwargs.get("speculative_config")
    disabled = (
        speculative is None
        or speculative is False
        or (isinstance(speculative, Mapping) and speculative.get("enabled") is False)
    )
    if speculative is not None and not disabled:
        raise ConfigParityError(
            "policy.generation.vllm_kwargs.speculative_config must be absent or disabled"
        )
    vllm_kwargs.pop("speculative_config", None)


def validate_parity(
    *, baseline: Mapping[str, Any], dflash: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one resolved baseline as a controlled DFlash K=15 ablation."""
    _validate_shared_contract(baseline)
    _validate_shared_contract(dflash)
    _require(baseline, "grpo.max_num_steps", 100)
    _require(baseline, "experiment.arm", "no-specdec")
    _require(baseline, "experiment.draft_k", 0)
    _require(baseline, "logger.tensorboard_enabled", True)
    _require(baseline, "logger.wandb_enabled", True)
    _require(baseline, "logger.wandb.project", WANDB_PROJECT)
    _require(baseline, "logger.wandb.group", WANDB_GROUP)
    _require(baseline, "logger.wandb.name", WANDB_NAME)
    _require(baseline, "logger.wandb.tags", WANDB_TAGS)
    _require(
        dflash, "policy.generation.vllm_kwargs.speculative_config.method", "dflash"
    )
    _require(
        dflash,
        "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens",
        15,
    )

    allowed_differences = _diff(baseline, dflash)
    normalized_baseline = cast(dict[str, Any], deepcopy(baseline))
    normalized_dflash = cast(dict[str, Any], deepcopy(dflash))
    _normalize_baseline_speculative_config(normalized_baseline)
    _remove_path(normalized_dflash, _SPECULATIVE_PATH)
    for path in _IDENTITY_PATHS:
        _remove_path(normalized_baseline, path)
        _remove_path(normalized_dflash, path)
    differences = _diff(normalized_baseline, normalized_dflash)
    if differences:
        raise ConfigParityError(
            "undeclared baseline differences: " + ", ".join(differences)
        )

    return {
        "allowed_differences": allowed_differences,
        "target_revision": TARGET_REVISION,
        "speculative_decoding_enabled": False,
        "num_speculative_tokens": 0,
        "policy_topology": {"tp": 2, "pp": 1, "cp": 1, "sp": True},
        "generation_topology": {"tp": 1, "precision": "bfloat16"},
        "wandb": {
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "name": WANDB_NAME,
            "tags": WANDB_TAGS,
        },
    }


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Load a NeMo-RL config after resolving inheritance and interpolation."""
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(path), resolve=True)
    if not isinstance(resolved, dict):
        raise ConfigParityError(f"{path} must resolve to a mapping")
    return cast(dict[str, Any], resolved)


def main() -> None:
    """Validate two resolved arms and print the normalized contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
    args = parser.parse_args()
    result = validate_parity(
        baseline=load_resolved_config(args.baseline),
        dflash=load_resolved_config(args.dflash),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
