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

import pytest

from nemo_rl.models.generation.vllm.config import validate_vllm_quantization_config


def _config(**vllm_overrides: object) -> dict[str, object]:
    return {
        "vllm_cfg": {
            "precision": "fp8",
            "is_mx": True,
            **vllm_overrides,
        }
    }


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


def test_refit_prequantize_requires_mxfp8_rollout():
    with pytest.raises(ValueError, match="requires precision='fp8' and is_mx=true"):
        validate_vllm_quantization_config(
            {"vllm_cfg": {"precision": "bf16", "refit_prequantize": True}}
        )


def test_valid_refit_optimization_flags():
    validate_vllm_quantization_config(
        _config(
            refit_prequantize=True,
            refit_batched_moe_shuffle=True,
            refit_cache_loader_routes=True,
        )
    )
