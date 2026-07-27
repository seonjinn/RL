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

import os
import subprocess
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


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


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
    assert "requires 64 GPUs total" in submitter
    assert "TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))" in submitter
    assert '--gpus-per-node="$GPUS_PER_NODE"' in submitter
    assert "GPU_REQUEST_MODE" in submitter
    assert "INIT_SUBMODULES" in submitter
    assert 'if [[ "$INIT_SUBMODULES" == "1" ]]' in submitter
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
    assert 'git -C "$REPO" rev-parse --git-dir' in launcher
    assert 'test -d "$REPO/.git"' not in launcher
    assert "cluster.num_nodes='$NUM_NODES'" in launcher
    assert "cluster.gpus_per_node='$GPUS_PER_NODE'" in launcher
    assert "cluster.segment_size='$NUM_NODES'" in launcher
    assert 'echo "nodes=$NUM_NODES"' in launcher


def test_gcp_nrt_profile_uses_eight_b200_gpus_per_node() -> None:
    profile = (EXPERIMENT_DIR / "submit_gcp_nrt.sh").read_text(encoding="utf-8")

    assert "coreai_chef_posttrain" in profile
    assert "PARTITION=${PARTITION:-batch}" in profile
    assert "NUM_NODES=8" in profile
    assert "GPUS_PER_NODE=8" in profile
    assert "GPU_REQUEST_MODE=gpus-per-node" in profile
    assert "INIT_SUBMODULES=0" in profile
    assert "SLURM_NETWORK=" in profile
    assert "CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre}" in profile
    assert "/.cache/huggingface" in profile
    assert "nemo-rl-nightly-main-20260705.sqsh" in profile
    assert "experiments/refit-opt-qwen30b/nemo-rl-refit-opt-r2" in profile
    assert "OccupiedIdleGPUsJobReaper" in profile


def test_gcp_nrt_profile_emits_the_expected_sbatch_shape(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls.txt"
    container = tmp_path / "container.sqsh"
    container.touch()

    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
if [[ "$*" == *"rev-parse"* ]]; then
  echo deadbeef
fi
""",
    )
    _write_executable(
        fake_bin / "readlink",
        """#!/bin/bash
printf '%s\n' "$2"
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\n' "$*" >"$SBATCH_CALLS"
echo "Submitted batch job 12345"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "ACTION": "test-only",
            "ARM_FILTER": "qkvo-optimized",
            "BASE": str(tmp_path / "base"),
            "CONTAINER": str(container),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO": str(PROJECT_ROOT),
            "RUN_SUFFIX": "pytest-gcp",
            "SBATCH_CALLS": str(sbatch_calls),
            "WORK": str(tmp_path / "work"),
        }
    )
    subprocess.run(
        [str(EXPERIMENT_DIR / "submit_gcp_nrt.sh")],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    call = sbatch_calls.read_text(encoding="utf-8")
    assert "--account=coreai_chef_posttrain" in call
    assert "--partition=batch" in call
    assert "--nodes=8" in call
    assert "--gpus-per-node=8" in call
    assert "--network=" not in call
    assert "--segment=" not in call
    assert "NUM_NODES=8" in call
    assert "GPUS_PER_NODE=8" in call
    assert "EXPERIMENT_CLUSTER=gcp-nrt-b200" in call
    assert "OccupiedIdleGPUsJobReaper" in call
