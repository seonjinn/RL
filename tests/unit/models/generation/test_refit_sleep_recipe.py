# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the dedicated GB200 functional lane."""

import ast
import os
from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config

ROOT = Path(__file__).resolve().parents[4]
NAME = "vllm-destructive-refit-qwen3-30ba3b-4n4g"
CONTROL = "vllm-preserving-refit-qwen3.5-35ba3b-6n4g-bf16-trtllm"


@pytest.mark.parametrize(
    "name,base",
    [
        (NAME, "performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout"),
        (CONTROL, "grpo-qwen3.5-35ba3b-6n4g-async-1off-bf16-trtllm"),
    ],
)
def test_gate_recipe_inherits_unchanged_runtime(name: str, base: str) -> None:
    recipes = ROOT / "examples/configs/recipes/llm"
    assert (recipes / f"{name}.yaml").is_file(), "missing dedicated recipe"
    actual = load_config(recipes / f"{name}.yaml")
    expected = load_config(recipes / f"{base}.yaml")
    assert OmegaConf.to_container(actual, resolve=False) == OmegaConf.to_container(
        expected, resolve=False
    )


@pytest.mark.parametrize("name", [NAME, CONTROL])
def test_wrapper_matches_common_env_and_nightly(name: str) -> None:
    wrapper = ROOT / f"tests/test_suites/llm/{name}.sh"
    assert wrapper.is_file(), "missing functional wrapper"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    source = wrapper.read_text()
    assert "common.env" in source
    assert "NUM_MINUTES=240" in source
    assert (
        str(wrapper.relative_to(ROOT))
        in (ROOT / "tests/test_suites/nightly_gb200.txt").read_text().splitlines()
    )


def test_only_new_functional_nodes_are_invoked() -> None:
    path = ROOT / "tests/functional/test_vllm_refit_sleep.py"
    assert path.is_file(), "missing dedicated functional pytest"
    nodes = {
        n.name
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    assert nodes == {
        "test_qwen3_mxfp8_destructive_refit_abc",
        "test_qwen3_mxfp8_missing_manifest_after_discard",
        "test_qwen35_bf16_nccl_reshard_preserving_control",
    }
    wrappers = "\n".join(
        (ROOT / f"tests/test_suites/llm/{name}.sh").read_text()
        for name in [NAME, CONTROL]
    )
    for node in nodes:
        assert f"tests/functional/test_vllm_refit_sleep.py::{node}" in wrappers
    assert "tests/unit/" not in wrappers


def test_common_env_accepts_node_local_test_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"\nprintf "%s" "$EXP_DIR"',
            "bash",
            str(ROOT / "tests/test_suites/llm/common.env"),
        ],
        env={
            **os.environ,
            "EXP_NAME": "grpo-qwen3.5-35ba3b-6n4g-async-1off-bf16-trtllm",
            "NRL_TEST_RUN_DIR": str(tmp_path / "node-local"),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == str(tmp_path / "node-local")
