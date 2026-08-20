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

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


EXPERIMENT_DIR = Path(__file__).parents[1]
DFLASH_RESOLVED_CONFIG = EXPERIMENT_DIR / "source_dflash_k15_step10_resolved.yaml"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _load_parity_module() -> ModuleType:
    module_path = EXPERIMENT_DIR / "parity.py"
    assert module_path.is_file(), "the no-SpecDec parity validator is missing"
    spec = importlib.util.spec_from_file_location("no_specdec_parity", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolved_pair(module: ModuleType) -> tuple[dict, dict]:
    baseline = module.load_resolved_config(EXPERIMENT_DIR / "config.yaml")
    dflash = module.load_resolved_config(DFLASH_RESOLVED_CONFIG)
    return baseline, dflash


def test_resolved_baseline_diff_is_limited_to_the_declared_arm_identity() -> None:
    parity = _load_parity_module()
    baseline, dflash = _resolved_pair(parity)

    result = parity.validate_parity(baseline=baseline, dflash=dflash)

    assert result["target_revision"] == TARGET_REVISION
    assert result["speculative_decoding_enabled"] is False
    assert result["num_speculative_tokens"] == 0
    assert result["policy_topology"] == {"tp": 2, "pp": 1, "cp": 1, "sp": True}
    assert result["generation_topology"] == {"tp": 1, "precision": "bfloat16"}
    assert result["wandb"] == {
        "project": "sna-nemo-rl-fixed-drafter",
        "group": "qwen3-8b-dflash-fixed-drafter-k-sweep",
        "name": "qwen3-8b-no-specdec-k0",
        "tags": ["no-specdec", "k0"],
    }
    assert result["allowed_differences"] == [
        "checkpointing.checkpoint_dir",
        "checkpointing.save_period",
        "experiment.arm",
        "experiment.draft_k",
        "grpo.max_num_steps",
        "logger.log_dir",
        "logger.wandb.group",
        "logger.wandb.name",
        "logger.wandb.project",
        "logger.wandb.tags",
        "logger.wandb_enabled",
        "policy.generation.vllm_kwargs.speculative_config",
        "policy.megatron_cfg.train_iters",
    ]


def test_validator_rejects_non_base_target_drift_even_when_both_arms_match() -> None:
    parity = _load_parity_module()
    baseline, dflash = _resolved_pair(parity)
    baseline["experiment"]["target_repo"] = "Qwen/Qwen3-8B-Base"
    dflash["experiment"]["target_repo"] = "Qwen/Qwen3-8B-Base"

    with pytest.raises(parity.ConfigParityError, match="experiment.target_repo"):
        parity.validate_parity(baseline=baseline, dflash=dflash)


def test_validator_rejects_shared_topology_drift() -> None:
    parity = _load_parity_module()
    baseline, dflash = _resolved_pair(parity)
    baseline["policy"]["megatron_cfg"]["tensor_model_parallel_size"] = 4
    dflash["policy"]["megatron_cfg"]["tensor_model_parallel_size"] = 4

    with pytest.raises(
        parity.ConfigParityError,
        match="policy.megatron_cfg.tensor_model_parallel_size",
    ):
        parity.validate_parity(baseline=baseline, dflash=dflash)


def test_validator_rejects_an_undeclared_cross_arm_difference() -> None:
    parity = _load_parity_module()
    baseline, dflash = _resolved_pair(parity)
    baseline["policy"]["megatron_cfg"]["optimizer"]["lr"] = 2.0e-6

    with pytest.raises(
        parity.ConfigParityError,
        match="policy.megatron_cfg.optimizer.lr",
    ):
        parity.validate_parity(baseline=baseline, dflash=dflash)


def test_explicitly_disabled_speculative_config_is_equivalent_to_absent() -> None:
    parity = _load_parity_module()
    baseline, dflash = _resolved_pair(parity)
    baseline = deepcopy(baseline)
    baseline["policy"]["generation"]["vllm_kwargs"]["speculative_config"] = {
        "enabled": False
    }

    result = parity.validate_parity(baseline=baseline, dflash=dflash)

    assert result["speculative_decoding_enabled"] is False
    assert result["num_speculative_tokens"] == 0
