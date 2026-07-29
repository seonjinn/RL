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

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


register_omegaconf_resolvers()


MOE_8G_RECIPE_PAIRS = (
    ("grpo-deepseek-v3-32n8g.yaml", "grpo-deepseek-v3-32n8g-alltoall.yaml"),
    ("grpo-deepseek-v3-64n8g.yaml", "grpo-deepseek-v3-64n8g-alltoall.yaml"),
    (
        "grpo-deepseek-v3-64n8g-async-1off.yaml",
        "grpo-deepseek-v3-64n8g-async-1off-alltoall.yaml",
    ),
    (
        "grpo-deepseek-v3-64n8g-fp8-async-1off.yaml",
        "grpo-deepseek-v3-64n8g-fp8-async-1off-alltoall.yaml",
    ),
    (
        "grpo-nemotron3-super-120BA12B-32n8g.yaml",
        "grpo-nemotron3-super-120BA12B-32n8g-alltoall.yaml",
    ),
    (
        "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml",
        "grpo-nemotron3-super-120BA12B-32n8g-async-1off-alltoall.yaml",
    ),
    (
        "grpo-qwen3-235b-16n8g.yaml",
        "grpo-qwen3-235b-16n8g-alltoall.yaml",
    ),
    (
        "grpo-qwen3-235b-32n8g.yaml",
        "grpo-qwen3-235b-32n8g-alltoall.yaml",
    ),
    (
        "grpo-qwen3-235b-32n8g-async-1off.yaml",
        "grpo-qwen3-235b-32n8g-async-1off-alltoall.yaml",
    ),
    (
        "grpo-qwen3-30ba3b-4n8g.yaml",
        "grpo-qwen3-30ba3b-4n8g-alltoall.yaml",
    ),
    (
        "grpo-qwen3-30ba3b-4n8g-async-1off.yaml",
        "grpo-qwen3-30ba3b-4n8g-async-1off-alltoall.yaml",
    ),
    (
        "grpo-qwen3-30ba3b-24n8g-async-8off.yaml",
        "grpo-qwen3-30ba3b-24n8g-async-8off-alltoall.yaml",
    ),
    (
        "grpo-qwen3-30ba3b-4n8g-40K.yaml",
        "grpo-qwen3-30ba3b-4n8g-40K-alltoall.yaml",
    ),
)

X86_HYBRIDEP_ENVIRONMENT = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "8",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": "128",
    "NVLINK_DOMAIN_SIZE": "8",
    "USE_MNNVL": "0",
}

X86_HYBRIDEP_ENVIRONMENT_KEYS = set(X86_HYBRIDEP_ENVIRONMENT)

DENSE_8G_RECIPES = (
    "grpo-llama3.1-8b-instruct-2n8g.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-async-1off.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-fp8-async-1off.yaml",
    "grpo-qwen3-32b-4n8g.yaml",
    "grpo-qwen3-32b-8n8g-async-1off.yaml",
)

FOUR_GPU_DISPATCHER_CONTRACT = {
    "grpo-deepseek-v3-32n4g.yaml": ("flex", "hybridep", 32),
    "grpo-deepseek-v3-64n4g-async-1off.yaml": ("alltoall", None, None),
    "grpo-qwen3-235b-16n4g.yaml": ("flex", "hybridep", 32),
    "grpo-qwen3-235b-32n4g-async-1off.yaml": ("alltoall", None, None),
}


def _recipe_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
    )


def _resolve_recipe(recipe_name: str) -> dict[str, Any]:
    recipe_path = _recipe_dir() / recipe_name
    assert recipe_path.is_file(), f"Missing recipe: {recipe_path}"
    resolved = OmegaConf.to_container(load_config(recipe_path), resolve=True)
    assert isinstance(resolved, dict)
    return cast(dict[str, Any], resolved)


def _megatron_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = config["policy"]
    assert isinstance(policy, dict)
    megatron_cfg = policy["megatron_cfg"]
    assert isinstance(megatron_cfg, dict)
    return megatron_cfg


def _environment(megatron_cfg: dict[str, Any]) -> dict[str, Any]:
    env_vars = megatron_cfg.get("env_vars", {})
    assert isinstance(env_vars, dict)
    return env_vars


def _without_dispatcher_contract(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(config)
    megatron_cfg = _megatron_config(cleaned)
    for key in (
        "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
        "moe_hybridep_num_sms",
    ):
        megatron_cfg.pop(key, None)

    env_vars = megatron_cfg.get("env_vars")
    if env_vars is not None:
        assert isinstance(env_vars, dict)
        for key in X86_HYBRIDEP_ENVIRONMENT_KEYS:
            env_vars.pop(key, None)
        if not env_vars:
            megatron_cfg.pop("env_vars")

    return cleaned


@pytest.mark.parametrize("canonical, _baseline", MOE_8G_RECIPE_PAIRS)
def test_moe_8g_canonical_recipes_default_to_x86_hybridep(
    canonical: str, _baseline: str
) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(canonical))

    assert megatron_cfg["moe_token_dispatcher_type"] == "flex"
    assert megatron_cfg["moe_flex_dispatcher_backend"] == "hybridep"
    assert megatron_cfg["moe_hybridep_num_sms"] == 32
    assert _environment(megatron_cfg) >= X86_HYBRIDEP_ENVIRONMENT


@pytest.mark.parametrize("_canonical, baseline", MOE_8G_RECIPE_PAIRS)
def test_moe_8g_alltoall_baselines_exclude_x86_hybridep_settings(
    _canonical: str, baseline: str
) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(baseline))

    assert megatron_cfg["moe_token_dispatcher_type"] == "alltoall"
    assert "moe_flex_dispatcher_backend" not in megatron_cfg
    assert "moe_hybridep_num_sms" not in megatron_cfg
    assert not X86_HYBRIDEP_ENVIRONMENT_KEYS.intersection(_environment(megatron_cfg))


@pytest.mark.parametrize("canonical, baseline", MOE_8G_RECIPE_PAIRS)
def test_moe_8g_hybridep_and_alltoall_pairs_only_differ_by_dispatcher_contract(
    canonical: str, baseline: str
) -> None:
    assert _without_dispatcher_contract(_resolve_recipe(canonical)) == (
        _without_dispatcher_contract(_resolve_recipe(baseline))
    )


@pytest.mark.parametrize("recipe_name", DENSE_8G_RECIPES)
def test_dense_8g_recipes_do_not_select_hybridep(recipe_name: str) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(recipe_name))

    assert megatron_cfg["moe_token_dispatcher_type"] == "alltoall"
    assert "moe_flex_dispatcher_backend" not in megatron_cfg
    assert "moe_hybridep_num_sms" not in megatron_cfg
    assert not X86_HYBRIDEP_ENVIRONMENT_KEYS.intersection(_environment(megatron_cfg))


@pytest.mark.parametrize(
    "recipe_name, dispatcher, backend, num_sms",
    [
        (recipe_name, *contract)
        for recipe_name, contract in FOUR_GPU_DISPATCHER_CONTRACT.items()
    ],
)
def test_4g_descendants_preserve_their_dispatcher_without_x86_environment(
    recipe_name: str, dispatcher: str, backend: str | None, num_sms: int | None
) -> None:
    megatron_cfg = _megatron_config(_resolve_recipe(recipe_name))

    assert megatron_cfg["moe_token_dispatcher_type"] == dispatcher
    assert megatron_cfg.get("moe_flex_dispatcher_backend") == backend
    assert megatron_cfg.get("moe_hybridep_num_sms") == num_sms
    assert not X86_HYBRIDEP_ENVIRONMENT_KEYS.intersection(_environment(megatron_cfg))
