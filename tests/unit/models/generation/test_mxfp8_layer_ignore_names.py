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

"""Exercise boundary-name conversion without importing the GPU runtime."""

import ast
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest


@pytest.fixture
def layer_names() -> Callable[[Sequence[str], Sequence[int]], list[str]]:
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/quantization/fp8.py"
    )
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_params_in_layers"
    )
    namespace = {"Sequence": Sequence}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_get_params_in_layers"]


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("language_model.backbone.layers", "language_model.backbone.layers"),
        ("backbone.layers", "backbone.layers"),
        ("layers", "model.layers"),
        ("language_model.layers", "model.language_model.layers"),
    ],
)
def test_boundary_names_preserve_model_mapper_prefix(
    layer_names: Callable, prefix: str, expected: str
) -> None:
    names = [f"{prefix}.{i}.mixer.up_proj.weight" for i in range(10)]
    assert layer_names(names, [8, 9]) == [
        f"{expected}.{i}.mixer.up_proj" for i in [8, 9]
    ]


def test_supervl_boundary_maps_to_runtime_module(layer_names: Callable) -> None:
    ignored = layer_names(
        ["language_model.backbone.layers.8.mixer.up_proj.weight"], [8]
    )
    # NanoNemotronVL's hf_to_vllm_mapper maps this exact prefix.
    old, new = "language_model.backbone", "language_model.model"
    mapped = [
        new + name[len(old) :] if name.startswith(old) else name for name in ignored
    ]
    assert mapped == ["language_model.model.layers.8.mixer.up_proj"]


def test_boundary_selection_excludes_other_layers_and_bias(
    layer_names: Callable,
) -> None:
    prefix = "language_model.backbone.layers"
    assert layer_names(
        [
            f"{prefix}.8.mixer.up_proj.weight",
            f"{prefix}.18.mixer.up_proj.weight",
            f"{prefix}.8.mixer.up_proj.bias",
            f"{prefix}.8.layernorm.weight",
        ],
        [8],
    ) == [f"{prefix}.8.mixer.up_proj"]
