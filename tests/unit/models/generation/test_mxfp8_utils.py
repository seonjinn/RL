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

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_mxfp8_utils():
    source_path = (
        Path(__file__).parents[4]
        / "nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py"
    )
    spec = importlib.util.spec_from_file_location("mxfp8_utils", source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refit_scale_name_uses_native_scale_for_refit_safe_linear_kernel():
    utils = _load_mxfp8_utils()
    module = SimpleNamespace(
        quant_method=SimpleNamespace(
            kernel=SimpleNamespace(preserves_checkpoint_weight_scale_for_refit=True)
        )
    )

    assert (
        utils.mxfp8_refit_scale_name("layers.0.self_attn.q_proj.weight", module)
        == "layers.0.self_attn.q_proj.weight_scale"
    )


def test_refit_scale_name_uses_native_scale_for_refit_safe_moe_base_method():
    utils = _load_mxfp8_utils()
    module = SimpleNamespace(
        quant_method=SimpleNamespace(),
        base_quant_method=SimpleNamespace(
            preserves_checkpoint_weight_scale_for_refit=True
        ),
    )

    assert (
        utils.mxfp8_refit_scale_name("layers.0.mlp.experts.0.gate_proj.weight", module)
        == "layers.0.mlp.experts.0.gate_proj.weight_scale"
    )


def test_refit_scale_name_keeps_legacy_checkpoint_scale_for_stock_vllm():
    utils = _load_mxfp8_utils()
    module = SimpleNamespace(quant_method=SimpleNamespace())

    assert (
        utils.mxfp8_refit_scale_name("layers.0.self_attn.q_proj.weight", module)
        == "layers.0.self_attn.q_proj.weight_scale_from_checkpoint"
    )
