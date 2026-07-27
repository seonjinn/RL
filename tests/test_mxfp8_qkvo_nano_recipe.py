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
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = PROJECT_ROOT / "examples/configs/recipes/llm"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/mxfp8_qkvo_nano"
CANONICAL_RECIPE = RECIPE_DIR / "grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
BASE_RECIPE = RECIPE_DIR / "grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-rollout.yaml"
QKVO_RECIPE = (
    RECIPE_DIR / "grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-qkvo-rollout.yaml"
)
DEFAULT_MODEL_PATH = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/models/nemotron-nano3/"
    "Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_resolved_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
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


def _without_qkvo_identity(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _deep_merge({}, config)
    normalized.pop("defaults", None)
    normalized["checkpointing"].pop("checkpoint_dir")
    normalized["logger"].pop("log_dir")
    normalized["logger"]["wandb"].pop("name")
    normalized["policy"]["generation"]["vllm_cfg"].pop("quantization_ignored_layer_kws")
    return normalized


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_mxfp8_recipe_resolves_with_canonical_nano_trainer_topology() -> None:
    canonical = _load_resolved_yaml(CANONICAL_RECIPE)
    config = _load_resolved_yaml(BASE_RECIPE)

    canonical_megatron = canonical["policy"]["megatron_cfg"]
    megatron = config["policy"]["megatron_cfg"]
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "sequence_parallel",
        "bias_activation_fusion",
    ):
        assert megatron[key] == canonical_megatron[key]

    assert config["grpo"]["num_prompts_per_step"] == 2
    assert config["grpo"]["num_generations_per_prompt"] == 8
    assert config["grpo"]["seed"] == 42
    assert config["policy"]["train_global_batch_size"] == 16
    assert config["loss_fn"]["force_on_policy_ratio"] is False
    assert config["loss_fn"]["use_importance_sampling_correction"] is True
    assert config["checkpointing"]["enabled"] is False
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["num_nodes"] == 4
    assert config["cluster"]["segment_size"] == 4

    vllm_cfg = config["policy"]["generation"]["vllm_cfg"]
    assert vllm_cfg["tensor_parallel_size"] == 1
    assert vllm_cfg["gpu_memory_utilization"] == 0.5
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    recipe_text = BASE_RECIPE.read_text(encoding="utf-8")
    assert "1856 -> 1920" in recipe_text
    assert "TP1" in recipe_text


def test_qkvo_overlay_only_changes_scope_and_output_identity() -> None:
    base_config = _load_resolved_yaml(BASE_RECIPE)
    qkvo_config = _load_resolved_yaml(QKVO_RECIPE)

    assert (
        qkvo_config["policy"]["generation"]["vllm_cfg"][
            "quantization_ignored_layer_kws"
        ]
        == []
    )
    assert qkvo_config["checkpointing"]["checkpoint_dir"].endswith(
        "-mxfp8-qkvo-rollout"
    )
    assert qkvo_config["logger"]["log_dir"].endswith("-mxfp8-qkvo-rollout")
    assert qkvo_config["logger"]["wandb"]["name"].endswith("-mxfp8-qkvo-rollout")
    assert _without_qkvo_identity(qkvo_config) == _without_qkvo_identity(base_config)


def test_launcher_enforces_matched_arm_settings_and_nightly_preflight() -> None:
    launcher = (EXPERIMENT_DIR / "run_arm.sbatch").read_text(encoding="utf-8")

    assert f"NANO_MODEL_PATH=${{NANO_MODEL_PATH:-{DEFAULT_MODEL_PATH}}}" in launcher
    assert "REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}" in launcher
    assert "policy.model_name='$NANO_MODEL_PATH'" in launcher
    assert "policy.tokenizer.name='$NANO_MODEL_PATH'" in launcher
    assert ': "${NUM_NODES:?NUM_NODES is required}"' in launcher
    assert ': "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"' in launcher
    assert ': "${EXPECTED_REPO_SHA:?EXPECTED_REPO_SHA is required}"' in launcher
    assert "cluster.num_nodes='$NUM_NODES'" in launcher
    assert "cluster.gpus_per_node='$GPUS_PER_NODE'" in launcher
    assert "cluster.segment_size='$NUM_NODES'" in launcher
    assert "policy.train_global_batch_size=16" in launcher
    assert "grpo.seed=42" in launcher
    assert "loss_fn.force_on_policy_ratio=false" in launcher
    assert "loss_fn.use_importance_sampling_correction=true" in launcher
    assert "policy.generation.vllm_cfg.tensor_parallel_size=1" in launcher
    assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.5" in launcher
    assert 'case "$ARM" in' in launcher
    assert "ROLLOUT_PRECISION=bf16" in launcher
    assert "QUANTIZATION_SCOPE=moe+qkvo" in launcher
    assert "++policy.generation.vllm_kwargs.moe_backend=triton" in launcher
    assert "$VLLM_BACKEND_OVERRIDE" in launcher
    assert "checkpointing.enabled=false" in launcher
    assert "policy.megatron_cfg.pinned_reference_swap=false" in launcher
    assert "logger.tensorboard_enabled='$TENSORBOARD_ENABLED'" in launcher

    assert "refit_prequantize='$PREQUANTIZE'" in launcher
    assert "refit_persistent_ipc_buffers='$PERSISTENT_IPC'" in launcher
    assert "refit_slim_offload_after='$SLIM_OFFLOAD'" in launcher
    assert "NRL_MXFP8_BATCHED_SHUFFLE='$NRL_MXFP8_BATCHED_SHUFFLE'" in launcher
    assert "NRL_REFIT_CACHED_LOADERS='$NRL_REFIT_CACHED_LOADERS'" in launcher

    assert "export PYTHONPATH='$REPO':" in launcher
    assert "export NRL_FORCE_REBUILD_VENVS=false" in launcher
    assert "/opt/nemo_rl_venv/bin/python examples/run_grpo.py" in launcher
    assert "Ray version mismatch before driver launch" in launcher
    assert "Driver did not import NeMo-RL from the experiment checkout" in launcher
    assert "VLLM_WORKER_VERSION" in launcher
    assert (
        "VLLM_WORKER_PYTHON=/opt/ray_venvs/"
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/"
        "bin/python" in launcher
    )
    assert 'test -x \\"\\$VLLM_WORKER_PYTHON\\"' in launcher
    assert "uv run --locked --extra vllm" not in launcher
    assert "ModelOptMxFp8FusedMoE" in launcher
    assert "ModelOptMxFp8LinearMethod" in launcher
    assert "WANDB_AUTH_SOURCE=netrc-host" in launcher
    assert "wandb.login(verify=True)" in launcher
    assert 'git -C "$REPO" rev-parse --git-dir' in launcher
    assert 'test -d "$REPO/.git"' not in launcher
    assert 'if [[ "$ACTUAL_REPO_SHA" != "$EXPECTED_REPO_SHA" ]]' in launcher
    assert "export WANDB_ENTITY=${WANDB_ENTITY:-nvidia}" in launcher
    assert "+logger.wandb.entity='$WANDB_ENTITY'" in launcher
    assert 'echo "cluster=$EXPERIMENT_CLUSTER"' in launcher
    assert 'echo "nodes=$NUM_NODES"' in launcher


def test_submitter_defaults_to_lyris_and_declares_five_arm_16_gpu_matrix() -> None:
    submitter = (EXPERIMENT_DIR / "submit_suite.sh").read_text(encoding="utf-8")

    assert "SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}" in submitter
    assert "PARTITION=${PARTITION:-gb200}" in submitter
    assert "NUM_NODES=${NUM_NODES:-4}" in submitter
    assert "GPUS_PER_NODE=${GPUS_PER_NODE:-4}" in submitter
    assert "GPU_REQUEST_MODE=${GPU_REQUEST_MODE:-none}" in submitter
    assert "SLURM_NETWORK=${SLURM_NETWORK-sharp}" in submitter
    assert "SLURM_SEGMENT=${SLURM_SEGMENT-$NUM_NODES}" in submitter
    assert "MAX_STEPS=${MAX_STEPS:-20}" in submitter
    assert "REPO=${REPO:-$BASE/RL-mxfp8-qkvo-pr3294-ab}" in submitter
    assert "TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))" in submitter
    assert "Nano suite requires 16 GPUs total" in submitter
    assert "Nano suite supports only 4x4 or 2x8 topology" in submitter
    assert '--gpus-per-node="$GPUS_PER_NODE"' in submitter
    assert "INIT_SUBMODULES=${INIT_SUBMODULES:-1}" in submitter
    assert 'if [[ "$INIT_SUBMODULES" == "1" ]]' in submitter
    assert "SUBMODULE_STATUS_MODE=all" in submitter
    assert '--ignore-submodules="$SUBMODULE_STATUS_MODE"' in submitter
    assert 'test -f "$NANO_MODEL_PATH/config.json"' in submitter
    assert "SUBMIT_SUITE_REEXEC" in submitter
    assert "EXPECTED_REPO_SHA=$REPO_SHA" in submitter
    assert 'CONTAINER=$(readlink -f "$CONTAINER")' in submitter
    assert "Repository HEAD does not match its upstream" in submitter
    assert (
        "CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre,/project:/project}"
        in (submitter)
    )
    assert "NANO_MODEL_PATH=${NANO_MODEL_PATH:-" in submitter

    arm_lines = [
        line.strip()
        for line in submitter.splitlines()
        if line.strip().startswith(('"bf16:', '"moe-', '"qkvo-'))
    ]
    assert arm_lines == [
        '"bf16:grpo-nanov3-30BA3B-2n8g-megatron-pack-cp:0"',
        ('"moe-baseline:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-rollout:0"'),
        ('"moe-optimized:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-mxfp8-rollout:1"'),
        (
            '"qkvo-baseline:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-'
            'mxfp8-qkvo-rollout:0"'
        ),
        (
            '"qkvo-optimized:grpo-nanov3-30BA3B-4n4g-megatron-pack-cp-'
            'mxfp8-qkvo-rollout:1"'
        ),
    ]


def test_submitter_applies_comma_separated_arm_filter(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls.txt"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
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
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\n' "$*" >>"$SBATCH_CALLS"
echo "Submitted batch job 12345"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "ACTION": "test-only",
            "ARM_FILTER": "bf16,qkvo-optimized",
            "CONTAINER": str(container),
            "NANO_MODEL_PATH": str(model_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO": str(PROJECT_ROOT),
            "RUN_SUFFIX": "pytest",
            "SBATCH_CALLS": str(sbatch_calls),
            "WORK": str(tmp_path / "work"),
        }
    )
    subprocess.run(
        [str(EXPERIMENT_DIR / "submit_suite.sh")],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    calls = sbatch_calls.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "ARM=bf16" in calls[0]
    assert "ARM=qkvo-optimized" in calls[1]


def test_submitter_rejects_non_16_gpu_topology() -> None:
    cases = (
        ("2", "4", "Nano suite requires 16 GPUs total, got 8"),
        ("8", "2", "Nano suite supports only 4x4 or 2x8 topology"),
    )
    for num_nodes, gpus_per_node, expected_error in cases:
        env = os.environ.copy()
        env.update(
            {
                "ACTION": "test-only",
                "NUM_NODES": num_nodes,
                "GPUS_PER_NODE": gpus_per_node,
            }
        )

        result = subprocess.run(
            [str(EXPERIMENT_DIR / "submit_suite.sh")],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )

        assert result.returncode == 2
        assert expected_error in result.stderr


def test_gcp_nrt_profile_uses_two_b200_nodes() -> None:
    profile = (EXPERIMENT_DIR / "submit_gcp_nrt.sh").read_text(encoding="utf-8")

    assert "coreai_chef_posttrain" in profile
    assert "PARTITION=${PARTITION:-batch}" in profile
    assert "NUM_NODES=2" in profile
    assert "GPUS_PER_NODE=8" in profile
    assert "GPU_REQUEST_MODE=gpus-per-node" in profile
    assert "INIT_SUBMODULES=0" in profile
    assert "SLURM_NETWORK=" in profile
    assert "SLURM_SEGMENT=" in profile
    assert "MAX_STEPS=20" in profile
    assert "CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre}" in profile
    assert "/.cache/huggingface" in profile
    assert "nemo-rl-nightly-main-20260705.sqsh" in profile
    assert "experiments/refit-opt-qwen30b/nemo-rl-refit-opt-r2" in profile
    assert "curriculum" not in profile
    assert "sna-mxfp8-qkvo-nano-gcp-nrt" in profile
    assert "OccupiedIdleGPUsJobReaper" in profile
    assert 'exemptIdleTimeMins":"120"' in profile


def test_gcp_nrt_profile_emits_expected_sbatch_shape(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls.txt"
    git_calls = tmp_path / "git-calls.txt"
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    container = tmp_path / "container.sqsh"
    container.touch()

    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
printf '%s\n' "$*" >>"$GIT_CALLS"
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
printf '%s\n' "$*" >>"$SBATCH_CALLS"
echo "Submitted batch job 12345"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "ACTION": "test-only",
            "BASE": str(tmp_path / "base"),
            "CONTAINER": str(container),
            "GIT_CALLS": str(git_calls),
            "MAX_STEPS": "3",
            "NANO_MODEL_PATH": str(model_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPO": str(PROJECT_ROOT),
            "RUN_SUFFIX": "pytest-gcp",
            "SBATCH_CALLS": str(sbatch_calls),
            "WORK": str(tmp_path / "work"),
        }
    )
    env.pop("ARM_FILTER", None)
    subprocess.run(
        [str(EXPERIMENT_DIR / "submit_gcp_nrt.sh")],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    calls = sbatch_calls.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5
    for arm, call in zip(
        (
            "bf16",
            "moe-baseline",
            "moe-optimized",
            "qkvo-baseline",
            "qkvo-optimized",
        ),
        calls,
        strict=True,
    ):
        assert "--account=coreai_chef_posttrain" in call
        assert "--partition=batch" in call
        assert "--nodes=2" in call
        assert "--gpus-per-node=8" in call
        assert "--time=4:00:00" in call
        assert "--network=" not in call
        assert "--segment=" not in call
        assert f"ARM={arm}" in call
        assert "MAX_STEPS=20" in call
        assert "NUM_NODES=2" in call
        assert "GPUS_PER_NODE=8" in call
        assert "EXPECTED_REPO_SHA=deadbeef" in call
        assert "WANDB_PROJECT=sna-mxfp8-qkvo-nano-gcp-nrt" in call
        assert "WANDB_ENTITY=nvidia" in call
        assert "EXPERIMENT_CLUSTER=gcp-nrt-b200" in call
        assert "OccupiedIdleGPUsJobReaper" in call

    git_call_text = git_calls.read_text(encoding="utf-8")
    assert "submodule update" not in git_call_text


def test_readme_scopes_qkvo_and_records_validation_caveat() -> None:
    readme = (EXPERIMENT_DIR / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert (
        "QKVO enables q/k/v/o relative to the standard MXFP8 scope, while all "
        "other ModelOpt-eligible layers stay unchanged."
    ) in normalized_readme
    assert (
        "A prior QKV run showed probability-ratio outliers, so this is "
        "performance/correctness validation, not a recommended default."
    ) in normalized_readme
