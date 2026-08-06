# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from typing import cast

import pytest

from nemo_rl.models.generation.vllm.config import (
    VllmConfig,
    validate_vllm_quantization_config,
)


def _config(**vllm_overrides: object) -> VllmConfig:
    return cast(
        VllmConfig,
        {
            "vllm_cfg": {
                "precision": "fp8",
                "is_mx": True,
                **vllm_overrides,
            }
        },
    )


def _nvfp4_config(**overrides: object) -> VllmConfig:
    return cast(
        VllmConfig,
        {
            "vllm_cfg": {
                "precision": "bfloat16",
                "refit_prequantize": True,
            },
            "quant_cfg": (
                "examples/modelopt/quant_configs/nvfp4_experts_weightonly.yaml"
            ),
            "real_quant": True,
            "refit_transport": None,
            "colocated": {"enabled": True, "resources": {}},
            **overrides,
        },
    )


@pytest.mark.parametrize(
    "field",
    [
        "refit_prequantize",
        "refit_batched_moe_shuffle",
        "refit_cache_loader_routes",
    ],
)
def test_refit_optimization_flags_require_boolean(field):
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        validate_vllm_quantization_config(_config(**{field: "yes"}))


def test_refit_prequantize_rejects_non_mxfp8_non_real_quant_rollout():
    with pytest.raises(ValueError, match="real_quant=true"):
        validate_vllm_quantization_config(
            _config(precision="bf16", is_mx=False, refit_prequantize=True)
        )


def test_refit_prequantize_rejects_ambiguous_mxfp8_real_quant_rollout():
    config = _config(refit_prequantize=True)
    config["real_quant"] = True

    with pytest.raises(
        ValueError,
        match="cannot combine real_quant=true with precision='fp8' and is_mx=true",
    ):
        validate_vllm_quantization_config(config)


def test_refit_prequantize_accepts_mxfp8_with_real_quant_false():
    config = _config(refit_prequantize=True)
    config["real_quant"] = False

    validate_vllm_quantization_config(config)


def test_refit_prequantize_accepts_colocated_nvfp4_w4a16(monkeypatch):
    monkeypatch.setattr(
        "nemo_rl.modelopt.utils.resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )

    validate_vllm_quantization_config(_nvfp4_config())


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"real_quant": False}, "real_quant=true"),
        ({"quant_cfg": None}, "non-empty quant_cfg"),
        (
            {"colocated": {"enabled": False, "resources": {}}},
            "colocated.enabled=true",
        ),
        ({"refit_transport": "nccl_reshard"}, "refit_transport=null"),
    ],
    ids=("fake-quant", "missing-quant-config", "non-colocated", "nccl-reshard"),
)
def test_refit_prequantize_rejects_unsupported_nvfp4_topology(
    monkeypatch, overrides, error_match
):
    monkeypatch.setattr(
        "nemo_rl.modelopt.utils.resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )

    with pytest.raises(ValueError, match=error_match):
        validate_vllm_quantization_config(_nvfp4_config(**overrides))


def test_refit_prequantize_rejects_non_nvfp4_quant_config(monkeypatch):
    def reject_quant_config(_quant_cfg):
        raise ValueError("supports only block-16 NVFP4 weights")

    monkeypatch.setattr(
        "nemo_rl.modelopt.utils.resolve_nvfp4_real_quant_mode",
        reject_quant_config,
    )

    with pytest.raises(ValueError, match="supports only block-16 NVFP4"):
        validate_vllm_quantization_config(_nvfp4_config())


def test_refit_prequantize_accepts_w4a4_with_frozen_artifact_provenance(monkeypatch):
    monkeypatch.setattr(
        "nemo_rl.modelopt.utils.resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a4",
    )

    validate_vllm_quantization_config(
        _nvfp4_config(
            quant_cfg="examples/modelopt/quant_configs/nvfp4_experts.yaml",
            model_name="org/model",
            vllm_kwargs={"revision": "0123456789abcdef"},
            real_quant_calibration_path="/artifacts/calibration.safetensors",
        )
    )


@pytest.mark.parametrize(
    ("overrides", "missing_field"),
    [
        ({"model_name": ""}, "model_name"),
        ({"vllm_kwargs": {}}, "vllm_kwargs.revision"),
        ({"real_quant_calibration_path": ""}, "real_quant_calibration_path"),
    ],
    ids=("model", "revision", "artifact"),
)
def test_refit_prequantize_w4a4_requires_frozen_artifact_provenance(
    monkeypatch, overrides, missing_field
):
    monkeypatch.setattr(
        "nemo_rl.modelopt.utils.resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a4",
    )
    config = _nvfp4_config(
        quant_cfg="examples/modelopt/quant_configs/nvfp4_experts.yaml",
        model_name="org/model",
        vllm_kwargs={"revision": "0123456789abcdef"},
        real_quant_calibration_path="/artifacts/calibration.safetensors",
    )
    config.update(cast(VllmConfig, overrides))

    with pytest.raises(ValueError, match=missing_field):
        validate_vllm_quantization_config(config)


def test_valid_refit_optimization_flags():
    validate_vllm_quantization_config(
        _config(
            refit_prequantize=True,
            refit_batched_moe_shuffle=True,
            refit_cache_loader_routes=True,
        )
    )
