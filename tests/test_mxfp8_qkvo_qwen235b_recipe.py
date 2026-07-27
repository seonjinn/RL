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

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_CONFIG_DIR = PROJECT_ROOT / "examples/configs/recipes/llm/performance"
BASE_RECIPE = PERF_CONFIG_DIR / "grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml"
QKVO_RECIPE = PERF_CONFIG_DIR / "grpo-qwen3-235b-16n4g-mxfp8-qkvo-rollout.yaml"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/mxfp8_qkvo_qwen235b"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_resolved_yaml(path: Path, seen: set[Path] | None = None) -> dict:
    if seen is None:
        seen = set()
    path = path.resolve()
    assert path not in seen

    config = _load_yaml(path)
    defaults = config.get("defaults")
    if not isinstance(defaults, str):
        return config

    parent = (path.parent / defaults).resolve()
    base_config = _load_resolved_yaml(parent, seen | {path})
    return _deep_merge(base_config, config)


def test_qkvo_recipe_only_changes_quantization_scope() -> None:
    base_config = _load_resolved_yaml(BASE_RECIPE)
    qkvo_config = _load_resolved_yaml(QKVO_RECIPE)

    base_vllm_cfg = base_config["policy"]["generation"]["vllm_cfg"]
    qkvo_vllm_cfg = qkvo_config["policy"]["generation"]["vllm_cfg"]

    assert base_vllm_cfg["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert qkvo_vllm_cfg["quantization_ignored_layer_kws"] == []

    base_vllm_cfg.pop("quantization_ignored_layer_kws")
    qkvo_vllm_cfg.pop("quantization_ignored_layer_kws")
    base_config.pop("defaults")
    qkvo_config.pop("defaults")
    assert qkvo_config == base_config


def test_submitter_defines_the_five_arm_64_gpu_matrix() -> None:
    submitter = (EXPERIMENT_DIR / "submit_suite.sh").read_text(encoding="utf-8")

    expected_arms = {
        '"bf16:grpo-qwen3-235b-16n4g:0"',
        '"moe-baseline:grpo-qwen3-235b-16n4g-mxfp8-rollout:0"',
        '"moe-optimized:grpo-qwen3-235b-16n4g-mxfp8-rollout:1"',
        '"qkvo-baseline:grpo-qwen3-235b-16n4g-mxfp8-qkvo-rollout:0"',
        '"qkvo-optimized:grpo-qwen3-235b-16n4g-mxfp8-qkvo-rollout:1"',
    }
    for arm in expected_arms:
        assert arm in submitter
    assert 'NUM_NODES=${NUM_NODES:-16}' in submitter
    assert 'GPUS_PER_NODE=${GPUS_PER_NODE:-4}' in submitter
    assert "requires NUM_NODES=16 and GPUS_PER_NODE=4" in submitter
    assert "EXPECTED_REPO_SHA=$REPO_SHA" in submitter
    assert 'CONTAINER=$(readlink -f "$CONTAINER")' in submitter
    assert "--ignore-submodules=dirty" in submitter


def test_launcher_matches_the_validated_235b_workload() -> None:
    launcher = (EXPERIMENT_DIR / "run_arm.sbatch").read_text(encoding="utf-8")

    assert "policy.train_global_batch_size=512" in launcher
    assert "loss_fn.force_on_policy_ratio=false" in launcher
    assert "loss_fn.use_importance_sampling_correction=true" in launcher
    assert "policy.megatron_cfg.moe_token_dispatcher_type=alltoall" in launcher
    assert "policy.megatron_cfg.moe_flex_dispatcher_backend=deepep" in launcher
    assert "export NCCL_NVLS_ENABLE=0" in launcher
    assert "export RAY_CGRAPH_get_timeout=2400" in launcher
    assert "+policy.generation.vllm_kwargs.distributed_timeout_seconds=2400" in launcher
    assert "checkpointing.enabled=false" in launcher
    assert "logger.tensorboard_enabled=False" in launcher
    assert "export NRL_FORCE_REBUILD_VENVS=false" in launcher
    assert ': "${EXPECTED_REPO_SHA:?EXPECTED_REPO_SHA is required}"' in launcher
    assert 'if [[ "$ACTUAL_REPO_SHA" != "$EXPECTED_REPO_SHA" ]]' in launcher
    assert 'if [[ -f "$HOME/.netrc" ]]' in launcher
    assert 'export WANDB_API_KEY=$WANDB_NETRC_KEY' in launcher
    assert "export WANDB_ENTITY=${WANDB_ENTITY:-nvidia}" in launcher
    assert "/opt/nemo_rl_venv/bin/python examples/run_grpo.py" in launcher
