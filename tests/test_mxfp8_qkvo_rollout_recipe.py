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
BASE_RECIPE = PERF_CONFIG_DIR / "grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml"
QKVO_RECIPE = PERF_CONFIG_DIR / "grpo-qwen3-30ba3b-4n4g-mxfp8-qkvo-rollout.yaml"


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


def test_lyris_launcher_reuses_container_runtime_with_local_source() -> None:
    launcher = (
        PROJECT_ROOT / "experiments/mxfp8_qkvo_pr3294/run_arm.sbatch"
    ).read_text(encoding="utf-8")

    assert "export PYTHONPATH='$REPO':" in launcher
    assert "export NRL_FORCE_REBUILD_VENVS=false" in launcher
    assert "/opt/nemo_rl_venv/bin/python examples/run_grpo.py" in launcher
    assert "uv run examples/run_grpo.py" not in launcher
    assert "Ray version mismatch before driver launch" in launcher
    assert "WANDB_AUTH_SOURCE=netrc-host" in launcher
    assert "export WANDB_API_KEY=$WANDB_NETRC_KEY" in launcher
    assert "wandb.login(verify=True)" in launcher


def test_submitter_pulls_branch_without_fetching_unrelated_submodule_refs() -> None:
    submitter = (
        PROJECT_ROOT / "experiments/mxfp8_qkvo_pr3294/submit_suite.sh"
    ).read_text(encoding="utf-8")

    assert "fetch.recurseSubmodules=false" in submitter
    assert "pull --ff-only --recurse-submodules=no" in submitter
    assert "submodule update --init --recursive" in submitter


def test_submitter_includes_selectable_bf16_baseline() -> None:
    submitter = (
        PROJECT_ROOT / "experiments/mxfp8_qkvo_pr3294/submit_suite.sh"
    ).read_text(encoding="utf-8")

    assert '"bf16:grpo-qwen3-30ba3b-4n4g:0"' in submitter
    assert "ARM_FILTER=${ARM_FILTER:-}" in submitter
    assert 'arm_is_selected "$ARM"' in submitter
