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
WANDB_PROJECT = "sna-nemo-rl-fixed-drafter"
WANDB_GROUP = "qwen3-8b-dflash-fixed-drafter-k-sweep"


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
    assert result["training_tp"] == 2
    assert result["training_pp"] == 1
    assert result["training_cp"] == 1
    assert result["sequence_parallel"] is True
    assert result["target_tp"] == 1
    assert result["draft_tp"] == 1
    assert result["precision"] == "bfloat16"
    assert result["kv_cache_dtype"] == "auto"
    assert result["acceptance_metrics_enabled"] is True
    assert result["fixed_prompt_panel_enabled"] is True


def test_runner_uses_a_short_job_local_ray_temp_path() -> None:
    runner = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert "export TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'" in runner
    assert "export RAY_TMPDIR='/tmp/nrl-${SLURM_JOB_ID}'" in runner
    assert "export TMPDIR='${RUN_DIR}/tmp'" not in runner


def test_runner_selects_only_pinned_k_configs_and_requires_wandb_key() -> None:
    runner = (EXPERIMENT_DIR / "run_oci_hsg.sbatch").read_text()

    assert 'dflash_k="${DFLASH_K:-15}"' in runner
    assert "config_k${dflash_k}.yaml" in runner
    assert 'if [[ -z "${WANDB_API_KEY:-}" ]]' in runner
    assert "--k '${dflash_k}'" in runner
    assert "DFLASH_K must be 3, 5, 7, or 9" in runner


def test_standard_vllm_panel_reuses_train_sampling_parameters() -> None:
    raw_config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
    generation = raw_config["policy"]["generation"]

    assert generation["val_temperature"] == generation["temperature"]
    assert generation["val_top_p"] == generation["top_p"]
    assert generation["val_top_k"] == generation["top_k"]


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_k_sweep_config_has_deterministic_wandb_provenance(k: int) -> None:
    contract = _load_contract_module()
    config_path = EXPERIMENT_DIR / f"config_k{k}.yaml"

    raw_config = contract.load_config(config_path)
    result = contract.validate_config(raw_config, expected_k=k, require_wandb=True)

    assert result["num_speculative_tokens"] == k
    assert result["wandb_enabled"] is True
    assert result["wandb_project"] == WANDB_PROJECT
    assert result["wandb_group"] == WANDB_GROUP
    assert result["wandb_name"] == (
        f"qwen3-8b-dflash-fixed-k{k}-cudagraph-step001-seed42"
    )
    assert result["wandb_tags"] == [
        "fixed-drafter",
        "dflash",
        "qwen3-8b",
        f"k{k}",
        "cudagraph",
        "target-only-grpo",
        "seed42",
        "step001",
    ]
    assert result["wandb_config"] == {
        "experiment": "fixed-drafter-qwen3-8b-dflash-k-sweep",
        "git_sha": "${oc.env:EXPECTED_HEAD}",
        "target_repo": "Qwen/Qwen3-8B",
        "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "drafter_repo": "z-lab/Qwen3-8B-DFlash-b16",
        "drafter_revision": "9b41424b7109f9c5413454f481b09a82b85333f4",
        "drafter_config_sha256": (
            "9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102"
        ),
        "container_sha256": (
            "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
        ),
        "runtime_vllm_version": "0.25.1",
        "k": k,
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": [
            1,
            2,
            4,
            *range(8, 256, 8),
            256,
            272,
            288,
            304,
            320,
        ],
        "max_num_seqs": 32,
        "max_dflash_decode_query_tokens": 32 * (k + 1),
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


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_cudagraph_config_covers_every_dflash_sweep_arm(k: int) -> None:
    contract = _load_contract_module()
    raw_config = contract.load_config(EXPERIMENT_DIR / f"config_k{k}.yaml")

    result = contract.validate_config(raw_config, expected_k=k, require_wandb=True)

    assert result["enforce_eager"] is False
    assert result["cudagraph_backend"] == "eager"
    assert result["cudagraph_mode"] == "PIECEWISE"
    assert result["max_num_seqs"] == 32
    assert result["max_dflash_decode_query_tokens"] == 32 * (k + 1)
    assert result["cudagraph_capture_sizes"][-1] == 320
    assert result["max_dflash_decode_query_tokens"] <= 320
    assert result["per_position_acceptance_positions"] == list(range(1, k + 1))


@pytest.mark.parametrize("k", [0, 1, 2, 4, 6, 8, 10, 15])
def test_non_sweep_k_fails_loudly(k: int) -> None:
    contract = _load_contract_module()

    with pytest.raises(ValueError, match="3, 5, 7, or 9"):
        contract.validate_sweep_k(k)


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_sweep_k_is_gated_to_one_step(k: int) -> None:
    contract = _load_contract_module()

    assert contract.validate_k_stage(k, 1) == (k, 1)
    for unsafe_steps in (10, 100):
        with pytest.raises(ValueError, match="only the 1-step gate"):
            contract.validate_k_stage(k, unsafe_steps)


@pytest.mark.parametrize("steps", [1, 10, 100])
def test_only_safe_stage_lengths_are_accepted(steps: int) -> None:
    contract = _load_contract_module()

    assert contract.validate_stage(steps) == steps


@pytest.mark.parametrize("steps", [0, 2, 99, 101])
def test_unsafe_stage_lengths_fail_loudly(steps: int) -> None:
    contract = _load_contract_module()

    with pytest.raises(ValueError, match="1, 10, or 100"):
        contract.validate_stage(steps)
