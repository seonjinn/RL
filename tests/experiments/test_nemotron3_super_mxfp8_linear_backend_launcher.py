from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "nemotron3_super_mxfp8_linear_backends"
    / "submit_ptyche.sh"
)
MATRIX_LAUNCHER = LAUNCHER.with_name("submit_matrix_ptyche.sh")


def _dry_run(tmp_path: Path, backend: str) -> str:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": backend,
        "EXPERIMENT_ROOT": str(tmp_path / backend),
        "RUN_ID": "test-run",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _grpo_overrides(output: str) -> list[str]:
    command_lines: list[str] = []
    collecting = False
    for line in output.splitlines():
        if line.startswith("uv run --frozen --extra vllm examples/run_grpo.py"):
            collecting = True
        if collecting:
            command_lines.append(line)
        if line.endswith("logger.tensorboard_enabled=true"):
            break
    else:
        raise AssertionError("dry-run output did not contain the GRPO command")

    command = "\n".join(command_lines)
    tokens = shlex.split(command)
    config_index = tokens.index("--config")
    return tokens[config_index + 2 :]


def test_mxfp8_cuda_graph_arms_differ_only_by_linear_backend(tmp_path: Path) -> None:
    outputs = {
        backend: _dry_run(tmp_path, backend)
        for backend in ("flashinfer_cutedsl", "flashinfer_cutlass")
    }

    for backend, output in outputs.items():
        assert "grpo-nemotron3-super-120BA12B-32n4g.yaml" in output
        assert f"linear_backend={backend}" in output
        assert "cluster.num_nodes=32" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=8" in output
        assert "grpo.num_prompts_per_step=32" in output
        assert "grpo.num_generations_per_prompt=8" in output
        assert "policy.train_global_batch_size=256" in output
        assert "policy.max_total_sequence_length=8192" in output
        assert "policy.generation.vllm_cfg.tensor_parallel_size=4" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "policy.generation.vllm_cfg.precision=fp8" in output
        assert "policy.generation.vllm_cfg.is_mx=true" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "logger.wandb_enabled=false" in output
        assert "logger.tensorboard_enabled=true" in output
        assert "checkpointing.enabled=false" in output
        assert "nccl_reshard" not in output
        assert "--dependency=" not in output

    normalized = {
        backend: output.replace(backend, "LINEAR_BACKEND").replace(
            backend.removeprefix("flashinfer_"), "LINEAR_BACKEND"
        )
        for backend, output in outputs.items()
    }
    assert normalized["flashinfer_cutedsl"] == normalized["flashinfer_cutlass"]


def test_emitted_mxfp8_overrides_compose_with_the_super_recipe(tmp_path: Path) -> None:
    overrides = _grpo_overrides(_dry_run(tmp_path, "flashinfer_cutedsl"))

    assert (
        "++policy.generation.vllm_cfg.quantization_ignored_layer_kws=[lm_head,mlp.gate]"
        in overrides
    )

    pytest.importorskip("hydra")
    from nemo_rl.utils.config import load_config, parse_hydra_overrides

    config = load_config(
        "examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml"
    )
    composed = parse_hydra_overrides(config, overrides)

    assert composed.policy.generation.vllm_cfg.quantization_ignored_layer_kws == [
        "lm_head",
        "mlp.gate",
    ]


def test_dry_run_validates_custom_vllm_runtime_provenance(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl")
    custom_vllm_root = REPO_ROOT / "3rdparty" / "vllm"

    assert f"source {custom_vllm_root}/nemo-rl.env" in output
    assert "vllm_path = Path(vllm.__file__).resolve()" in output
    assert f'custom_vllm_root = Path("{custom_vllm_root}").resolve()' in output
    assert "vllm_path.is_relative_to(custom_vllm_root)" in output


def test_matrix_dry_run_launches_independent_arms(tmp_path: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "EXPERIMENT_ROOT": str(tmp_path / "runs"),
        "RUN_ID": "test-run",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(MATRIX_LAUNCHER)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines().count("backend=flashinfer_cutedsl") == 1
    assert result.stdout.splitlines().count("backend=flashinfer_cutlass") == 1
    assert "--dependency=" not in result.stdout
    assert result.stdout.splitlines().count(
        f"experiment_root={tmp_path / 'runs' / 'flashinfer_cutedsl'}"
    ) == 1
    assert result.stdout.splitlines().count(
        f"experiment_root={tmp_path / 'runs' / 'flashinfer_cutlass'}"
    ) == 1
