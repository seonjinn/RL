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

"""Behavioral contracts for Qwen3-30B-A3B CUDA Graph submissions."""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_ROOT = REPO_ROOT / "experiments/cuda_graph/qwen3_30ba3b_4n4g"
SCOPE_ROOT = SCRIPT_ROOT / "scopes"
QWEN_SNAPSHOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
    "models--Qwen--Qwen3-30B-A3B/snapshots/"
    "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)


def _install_fake_commands(tmp_path: Path) -> None:
    uv_path = tmp_path / "uv"
    uv_path.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  printf 'FAKE_UV_ARG=%s\\n' \"${arg}\"\n"
        "done\n"
    )
    uv_path.chmod(0o755)

    sbatch_path = tmp_path / "sbatch"
    sbatch_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'FAKE_SBATCH'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\nCOMMAND=%s\\n' \"${COMMAND:-}\"\n"
        "printf 'BASE_LOG_DIR=%s\\n' \"${BASE_LOG_DIR:-}\"\n"
        "printf 'WANDB_MODE=%s\\n' \"${WANDB_MODE:-}\"\n"
        "bash -c \"${COMMAND}\"\n"
    )
    sbatch_path.chmod(0o755)


def _run_script(
    script: Path,
    tmp_path: Path,
    args: tuple[str, ...] = (),
    **extra_env: str,
) -> subprocess.CompletedProcess[str]:
    _install_fake_commands(tmp_path)
    env = (
        os.environ
        | {
        "CLUSTER": "ptyche",
        "LOG_ROOT_OVERRIDE": str(tmp_path / "logs"),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PHASE": "performance",
        "STEPS": "20",
        "TEST_ONLY": "1",
        }
        | extra_env
    )
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("script_name", "expected_scope", "expected_impl"),
    [
        ("00_nocg.sh", "[]", "none"),
        ("01_attn.sh", "[attn]", "transformer_engine"),
        (
            "02_moe_router_preprocess.sh",
            "[moe_router,moe_preprocess]",
            "transformer_engine",
        ),
        (
            "03_attn_moe_router_preprocess.sh",
            "[attn,moe_router,moe_preprocess]",
            "transformer_engine",
        ),
    ],
)
def test_scope_wrapper_submits_the_controlled_20_step_workload(
    script_name: str,
    expected_scope: str,
    expected_impl: str,
    tmp_path: Path,
) -> None:
    """A wrong topology, recipe, packing cap, or graph scope changes the experiment."""
    result = _run_script(SCOPE_ROOT / script_name, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "FAKE_SBATCH --test-only" in result.stdout
    assert "--nodes=8" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--time=04:00:00" in result.stdout
    assert "--segment=4" in result.stdout
    assert "--dependency" not in result.stdout
    run_log_dir = tmp_path / "logs" / f"qwen3-30ba3b-{script_name.removesuffix('.sh').removeprefix('00_').removeprefix('01_').removeprefix('02_').removeprefix('03_').replace('_', '-')}-performance"
    assert run_log_dir.is_dir()
    assert f"--output={run_log_dir}/slurm-%j.out" in result.stdout
    assert f"--error={run_log_dir}/slurm-%j.out" in result.stdout
    assert (
        "FAKE_UV_ARG=examples/configs/recipes/llm/performance/"
        "grpo-qwen3-30ba3b-4n4g.yaml"
    ) in result.stdout
    for expected_arg in (
        f"policy.model_name={QWEN_SNAPSHOT}",
        f"policy.tokenizer.name={QWEN_SNAPSHOT}",
        "cluster.num_nodes=8",
        "cluster.gpus_per_node=4",
        "policy.generation.colocated.enabled=false",
        "policy.generation.colocated.resources.num_nodes=4",
        "grpo.max_num_steps=20",
        "checkpointing.enabled=false",
        "policy.megatron_cfg.cuda_graph_max_packed_seqs=16",
        "policy.megatron_cfg.cuda_graph_warmup_steps=3",
        f"policy.megatron_cfg.cuda_graph_impl={expected_impl}",
        f"policy.megatron_cfg.cuda_graph_scope={expected_scope}",
    ):
        assert f"FAKE_UV_ARG={expected_arg}" in result.stdout
    assert "WANDB_MODE=offline" in result.stdout


def test_matrix_driver_submits_four_independent_jobs(tmp_path: Path) -> None:
    """The reusable matrix launches every approved row without serial dependencies."""
    result = _run_script(SCRIPT_ROOT / "submit_performance_matrix.sh", tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("FAKE_SBATCH --test-only") == 4
    assert "--dependency" not in result.stdout
    for run_name in (
        "qwen3-30ba3b-nocg-performance",
        "qwen3-30ba3b-attn-performance",
        "qwen3-30ba3b-moe-router-preprocess-performance",
        "qwen3-30ba3b-attn-moe-router-preprocess-performance",
    ):
        assert f"{tmp_path}/logs/{run_name}" in result.stdout


def test_launcher_rejects_non_numeric_steps_before_sbatch(tmp_path: Path) -> None:
    """An environment override cannot append a shell command to the training command."""
    marker = tmp_path / "injected"
    result = _run_script(
        SCOPE_ROOT / "01_attn.sh",
        tmp_path,
        STEPS=f"20; touch {marker}",
    )

    assert result.returncode == 2
    assert "STEPS must be a positive integer" in result.stderr
    assert "FAKE_SBATCH" not in result.stdout
    assert not marker.exists()


def test_logger_path_metacharacters_remain_one_driver_argument(tmp_path: Path) -> None:
    """A shell metacharacter in a log root stays data when ray.sub executes COMMAND."""
    marker = tmp_path / "injected"
    log_root = f"{tmp_path}/logs with spaces; touch {marker}"
    result = _run_script(
        SCOPE_ROOT / "01_attn.sh",
        tmp_path,
        LOG_ROOT_OVERRIDE=log_root,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "FAKE_UV_ARG=logger.log_dir="
        f"{log_root}/qwen3-30ba3b-attn-performance"
    ) in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize(
    "args",
    [
        ("attn/../../escape", "[attn]", "transformer_engine"),
        ("attn", "[mamba]", "transformer_engine"),
        ("nocg", "[attn]", "none"),
    ],
)
def test_common_launcher_rejects_unapproved_scope_inputs(
    args: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Direct callers cannot create arbitrary paths or unreviewed graph scopes."""
    result = _run_script(SCRIPT_ROOT / "run_scope.sh", tmp_path, args=args)

    assert result.returncode == 2
    assert "Unsupported Qwen CUDA Graph scope request" in result.stderr
    assert "FAKE_SBATCH" not in result.stdout
