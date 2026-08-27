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
