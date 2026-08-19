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

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml


EXPERIMENT_DIR = Path(__file__).parents[1]


def _load_contract_module() -> ModuleType:
    module_path = EXPERIMENT_DIR / "contract.py"
    assert module_path.is_file(), "the DFlash experiment contract is not implemented"
    spec = importlib.util.spec_from_file_location(
        "dflash_experiment_contract", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_pins_fixed_public_drafter_and_matching_target() -> None:
    contract = _load_contract_module()
    raw_config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())

    result = contract.validate_config(raw_config)

    assert result["target_repo"] == "Qwen/Qwen3-8B"
    assert result["target_revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
    assert result["tokenizer_revision"] == result["target_revision"]
    assert result["drafter_repo"] == "z-lab/Qwen3-8B-DFlash-b16"
    assert result["drafter_revision"] == "9b41424b7109f9c5413454f481b09a82b85333f4"
    assert result["num_speculative_tokens"] == 15
    assert result["draft_training_enabled"] is False
    assert result["draft_refit_enabled"] is False


def test_config_preserves_shared_arm_schedule_and_metrics() -> None:
    contract = _load_contract_module()
    raw_config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())

    result = contract.validate_config(raw_config)

    assert result["dataset"] == "DAPOMath17K"
    assert result["seed"] == 42
    assert result["num_prompts_per_step"] == 8
    assert result["num_generations_per_prompt"] == 4
    assert result["train_global_batch_size"] == 32
    assert result["train_micro_batch_size"] == 1
    assert result["max_input_seq_length"] == 2048
    assert result["max_new_tokens"] == 1024
    assert result["max_total_sequence_length"] == 4096
    assert result["temperature"] == 1.0
    assert result["top_p"] == 1.0
    assert result["top_k"] is None
    assert result["learning_rate"] == 1.0e-6
    assert result["warmup_iters"] == 10
    assert result["acceptance_metrics_enabled"] is True
    assert result["fixed_prompt_panel_enabled"] is True


def test_runner_uses_a_short_job_local_ray_temp_path() -> None:
    runner = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert "export TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'" in runner
    assert "export RAY_TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'" in runner
    assert "export TMPDIR='${RUN_DIR}/tmp'" not in runner


@pytest.mark.parametrize("steps", [1, 10, 100])
def test_only_safe_stage_lengths_are_accepted(steps: int) -> None:
    contract = _load_contract_module()

    assert contract.validate_stage(steps) == steps


@pytest.mark.parametrize("steps", [0, 2, 99, 101])
def test_unsafe_stage_lengths_fail_loudly(steps: int) -> None:
    contract = _load_contract_module()

    with pytest.raises(ValueError, match="1, 10, or 100"):
        contract.validate_stage(steps)
