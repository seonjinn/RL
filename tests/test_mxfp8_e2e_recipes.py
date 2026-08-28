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

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_CONFIG_DIR = PROJECT_ROOT / "examples/configs/recipes/llm/performance"
SUBMIT_SCRIPT = PROJECT_ROOT / "research/mxfp8_training_rl/submit_oci_hsg.sh"
NANO_TE_PRECISION_CONFIG = (
    PROJECT_ROOT / "examples/nemo_gym/nemotron-3.5-nano/te_mxfp8_nano_v2.yaml"
)
NANO_BF16_TRAIN_MXFP8_ROLLOUT_RECIPE = (
    PERF_CONFIG_DIR
    / "grpo-nanov3-30ba3b-8n4g-bf16-train-mxfp8-rollout.yaml"
)
QWEN_BF16_TRAIN_MXFP8_ROLLOUT_RECIPE = (
    PERF_CONFIG_DIR
    / "grpo-qwen3-30ba3b-4n4g-async-1off-bf16-train-mxfp8-rollout.yaml"
)

BF16_ROLLOUT_CASES = {
    "grpo-qwen3-30ba3b-4n4g-async-1off-bf16-rollout": {
        "training_fp8": False,
        "async_grpo": True,
        "async_engine": True,
    },
    "grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-train-bf16-rollout": {
        "training_fp8": True,
        "async_grpo": True,
        "async_engine": True,
    },
    "grpo-nanov3-30ba3b-8n4g-bf16-rollout": {
        "training_fp8": False,
        "async_grpo": False,
        "async_engine": False,
    },
    "grpo-nanov3-30ba3b-8n4g-mxfp8-train-bf16-rollout": {
        "training_fp8": True,
        "async_grpo": False,
        "async_engine": False,
    },
}

MXFP8_E2E_CASES = {
    "grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-e2e-fp8param-false": {
        "nodes": 4,
        "generation_nodes": 2,
        "async_engine": True,
    },
    "grpo-nanov3-30ba3b-8n4g-mxfp8-e2e-fp8param-false": {
        "nodes": 8,
        "generation_nodes": 4,
        "async_engine": False,
        "te_precision_config_file": str(
            NANO_TE_PRECISION_CONFIG.relative_to(PROJECT_ROOT)
        ),
    },
}


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
    return _deep_merge(_load_resolved_yaml(parent, seen | {path}), config)


@pytest.mark.parametrize(("case_name", "expected"), MXFP8_E2E_CASES.items())
def test_mxfp8_e2e_fp8param_false_recipe(case_name: str, expected: dict) -> None:
    config_path = PERF_CONFIG_DIR / f"{case_name}.yaml"
    assert config_path.is_file()

    config = _load_resolved_yaml(config_path)
    fp8_cfg = config["policy"]["megatron_cfg"]["fp8_cfg"]
    generation = config["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]

    assert fp8_cfg == {
        "enabled": True,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "fp8_param": False,
    }
    assert config["policy"]["megatron_cfg"]["moe_router_dtype"] == "fp32"
    te_precision_config_file = expected.get("te_precision_config_file")
    assert (
        config["policy"]["megatron_cfg"].get("te_precision_config_file")
        == te_precision_config_file
    )
    if te_precision_config_file is not None:
        assert config["policy"]["megatron_cfg"]["first_last_layers_bf16"] is True
        assert config["policy"]["megatron_cfg"]["num_layers_at_start_in_bf16"] == 0
        assert config["policy"]["megatron_cfg"]["num_layers_at_end_in_bf16"] == 8
    assert generation["refit_transport"] == "nccl_reshard"
    assert generation["colocated"]["enabled"] is False
    assert generation["colocated"]["resources"]["num_nodes"] == expected[
        "generation_nodes"
    ]
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["async_engine"] is expected["async_engine"]
    assert vllm_cfg["enforce_eager"] is False
    assert config["cluster"]["num_nodes"] == expected["nodes"]
    assert config["cluster"]["gpus_per_node"] == 4


def test_nano_te_precision_config_quantizes_only_routed_experts() -> None:
    config = _load_yaml(NANO_TE_PRECISION_CONFIG)

    assert config["configs"] == {
        "bf16": {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": {"override_quantized_autocast": True},
        },
        "mxfp8": {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": {
                "fp8_quantization_recipe": "mxfp8",
                "override_quantized_autocast": True,
            },
        },
    }
    assert list(config["matchers"]) == [
        "routed_experts_fc1_mxfp8",
        "routed_experts_fc2_mxfp8",
        "all_other_modules_bf16",
    ]
    assert config["matchers"]["routed_experts_fc1_mxfp8"] == {
        "config": "mxfp8",
        "type": "glob",
        "pattern": "*mlp.experts.linear_fc1",
        "enabled": True,
    }
    assert config["matchers"]["routed_experts_fc2_mxfp8"] == {
        "config": "mxfp8",
        "type": "glob",
        "pattern": "*mlp.experts.linear_fc2",
        "enabled": True,
    }
    assert config["matchers"]["all_other_modules_bf16"] == {
        "config": "bf16",
        "type": "glob",
        "pattern": "*",
        "enabled": True,
    }


def test_nano_bf16_training_mxfp8_rollout_recipe() -> None:
    config = _load_resolved_yaml(NANO_BF16_TRAIN_MXFP8_ROLLOUT_RECIPE)
    megatron_cfg = config["policy"]["megatron_cfg"]
    vllm_cfg = config["policy"]["generation"]["vllm_cfg"]

    assert megatron_cfg["fp8_cfg"]["enabled"] is False
    assert megatron_cfg["moe_router_dtype"] == "fp32"
    assert "te_precision_config_file" not in megatron_cfg
    assert megatron_cfg.get("first_last_layers_bf16", False) is False
    assert megatron_cfg.get("num_layers_at_start_in_bf16", 0) == 0
    assert megatron_cfg.get("num_layers_at_end_in_bf16", 0) == 0
    assert config["policy"]["generation"]["refit_transport"] == "nccl_reshard"
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["enforce_eager"] is False
    assert config["policy"]["generation"]["vllm_kwargs"]["moe_backend"] == (
        "flashinfer_trtllm"
    )


def test_qwen_bf16_training_mxfp8_rollout_recipe() -> None:
    config = _load_resolved_yaml(QWEN_BF16_TRAIN_MXFP8_ROLLOUT_RECIPE)
    megatron_cfg = config["policy"]["megatron_cfg"]
    generation = config["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]

    assert megatron_cfg["fp8_cfg"]["enabled"] is False
    assert megatron_cfg["moe_router_dtype"] == "fp32"
    assert "te_precision_config_file" not in megatron_cfg
    assert generation["refit_transport"] == "nccl_reshard"
    assert generation["colocated"]["enabled"] is False
    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["async_engine"] is True
    assert vllm_cfg["enforce_eager"] is False
    assert generation["vllm_kwargs"]["moe_backend"] == "flashinfer_trtllm"


@pytest.mark.parametrize(("case_name", "expected"), BF16_ROLLOUT_CASES.items())
def test_bf16_rollout_comparison_recipe(case_name: str, expected: dict) -> None:
    config_path = PERF_CONFIG_DIR / f"{case_name}.yaml"
    assert config_path.is_file()

    config = _load_resolved_yaml(config_path)
    megatron_cfg = config["policy"]["megatron_cfg"]
    generation = config["policy"]["generation"]
    vllm_cfg = generation["vllm_cfg"]

    assert megatron_cfg["fp8_cfg"]["enabled"] is expected["training_fp8"]
    assert megatron_cfg["moe_router_dtype"] == "fp32"
    if expected["training_fp8"]:
        assert megatron_cfg["fp8_cfg"]["fp8"] == "e4m3"
        assert megatron_cfg["fp8_cfg"]["fp8_recipe"] == "mxfp8"
        assert megatron_cfg["fp8_cfg"]["fp8_param"] is False
        assert megatron_cfg["moe_router_dtype"] == "fp32"
    assert config["grpo"]["async_grpo"]["enabled"] is expected["async_grpo"]
    assert config["loss_fn"]["force_on_policy_ratio"] is False
    assert generation["refit_transport"] == "nccl_reshard"
    assert generation["colocated"]["enabled"] is False
    assert vllm_cfg["precision"] == "bfloat16"
    assert vllm_cfg["async_engine"] is expected["async_engine"]
    assert vllm_cfg["enforce_eager"] is False
    assert "is_mx" not in vllm_cfg
    assert "quantization_ignore_patterns" not in vllm_cfg
    assert generation["vllm_kwargs"]["moe_backend"] == "flashinfer_trtllm"


def test_oci_launcher_exports_cpu_count_before_sbatch() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    cpu_export = "export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}"
    assert cpu_export in script
    assert script.index(cpu_export) < script.index("exec sbatch")


def test_oci_launcher_exports_resolved_slurm_bin_before_sbatch() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    resolve_slurm_bin = 'SLURM_BIN_DIR=$(dirname "$(readlink -f "$(command -v scontrol)")")'
    export_slurm_bin = 'export PATH="${SLURM_BIN_DIR}:${PATH}"'
    assert resolve_slurm_bin in script
    assert export_slurm_bin in script
    assert script.index(resolve_slurm_bin) < script.index(export_slurm_bin)
    assert script.index(export_slurm_bin) < script.index("exec sbatch")


def test_oci_launcher_does_not_sync_ray_session_small_files_to_lustre() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "RAY_LOG_SYNC_FREQUENCY" not in script


def test_oci_launcher_reuses_sha_keyed_node_local_venv_by_default() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}" in script
    assert "export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS}" in script
    assert "nemo-rl-worker-cache/${LOCAL_HEAD}" in script
    assert "export NRL_FORCE_REBUILD_VENVS=true" not in script


def test_oci_launcher_limits_transformer_engine_build_to_gb200() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-100}" in script
    assert "export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}" in script


def test_oci_launcher_selects_training_precision_recipe() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "TRAINING_PRECISION=${TRAINING_PRECISION:-mxfp8}" in script
    assert "TRAINING_PRECISION must be bf16 or mxfp8" in script
    assert "grpo-nanov3-30ba3b-8n4g-bf16-train-mxfp8-rollout.yaml" in script
    assert (
        "grpo-qwen3-30ba3b-4n4g-async-1off-bf16-train-mxfp8-rollout.yaml"
        in script
    )


def test_oci_launcher_selects_rollout_precision_recipe() -> None:
    script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "ROLLOUT_PRECISION=${ROLLOUT_PRECISION:-mxfp8}" in script
    assert "ROLLOUT_PRECISION must be bf16 or mxfp8" in script
    for case_name in BF16_ROLLOUT_CASES:
        assert f"{case_name}.yaml" in script
    assert (
        "${TRAINING_PRECISION}-training-${MODEL}-${ROLLOUT_PRECISION}-rollout"
        in script
    )
