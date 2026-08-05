from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "qwen30b_mxfp8_linear_backends"
    / "submit_ptyche.sh"
)
PREPARE_SCRIPT = LAUNCHER.with_name("prepare_custom_vllm_ptyche.sh")
BUILD_CUSTOM_VLLM_SCRIPT = REPO_ROOT / "tools" / "build-custom-vllm.sh"


def _dry_run(tmp_path: Path, backend: str) -> str:
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()
    custom_vllm = tmp_path / "vllm"
    custom_vllm.mkdir(exist_ok=True)
    (custom_vllm / ".git").mkdir(exist_ok=True)

    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": backend,
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPERIMENT_ROOT": str(tmp_path / backend),
        "WANDB_MODE": "disabled",
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


def test_dry_run_changes_only_backend(tmp_path: Path) -> None:
    outputs = {
        backend: _dry_run(tmp_path, backend)
        for backend in (
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
            "flashinfer_trtllm",
        )
    }

    for backend, output in outputs.items():
        assert f"linear_backend={backend}" in output
        assert "policy.train_global_batch_size=2048" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "cluster.num_nodes=4" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=4" in output
        assert "grpo.max_num_steps=8" in output


def test_rejects_unknown_backend(tmp_path: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": "auto",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported BACKEND" in result.stderr


def test_custom_vllm_build_is_recoverable() -> None:
    prepare_text = PREPARE_SCRIPT.read_text()
    build_text = BUILD_CUSTOM_VLLM_SCRIPT.read_text()

    assert "3rdparty/vllm/nemo-rl.env" in prepare_text
    assert "vllm.incomplete" in prepare_text
    assert "git submodule update --init --recursive --depth 1" in prepare_text
    assert "3rdparty/vllm/.venv/bin/python -c 'import vllm'" in prepare_text
    assert "uv lock" in prepare_text
    assert "setuptools_rust" in build_text
