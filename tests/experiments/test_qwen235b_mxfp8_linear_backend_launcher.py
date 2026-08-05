from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT / "experiments" / "qwen235b_mxfp8_linear_backends" / "submit_cluster.sh"
)


def _dry_run(tmp_path: Path, backend: str, dependency_job_id: str = "") -> str:
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
        "WORK_ROOT": str(tmp_path),
        "DEPENDENCY_JOB_ID": dependency_job_id,
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


def test_qkvo_scope_changes_only_linear_backend(tmp_path: Path) -> None:
    outputs = {
        backend: _dry_run(tmp_path, backend)
        for backend in ("flashinfer_cutedsl", "flashinfer_cutlass")
    }

    for backend, output in outputs.items():
        assert "grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml" in output
        assert f"linear_backend={backend}" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "cluster.num_nodes=16" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=16" in output
        assert "grpo.max_num_steps=8" in output

    normalized = {
        backend: output.replace(backend, "LINEAR_BACKEND").replace(
            backend.removeprefix("flashinfer_"), "LINEAR_BACKEND"
        )
        for backend, output in outputs.items()
    }
    assert normalized["flashinfer_cutedsl"] == normalized["flashinfer_cutlass"]


def test_rejects_non_baseline_backend(tmp_path: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": "flashinfer_trtllm",
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


def test_adds_afterok_dependency_when_requested(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl", dependency_job_id="12345")

    assert "--dependency=afterok:12345" in output
