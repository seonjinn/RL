from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from nemo_rl.utils.config import load_config, parse_hydra_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "nemotron3_super_mxfp8_linear_backends"
    / "submit_ptyche.sh"
)
MATRIX_LAUNCHER = LAUNCHER.with_name("submit_matrix_ptyche.sh")
ALL_LAUNCHERS = (
    REPO_ROOT / "experiments" / "qwen30b_mxfp8_linear_backends" / "submit_ptyche.sh",
    REPO_ROOT / "experiments" / "qwen235b_mxfp8_linear_backends" / "submit_cluster.sh",
    LAUNCHER,
)


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
    expected_nemo_rl_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    assert f"source {custom_vllm_root}/nemo-rl.env" in output
    assert "vllm_path = Path(vllm.__file__).resolve()" in output
    assert f'custom_vllm_root = Path("{custom_vllm_root}").resolve()' in output
    assert "vllm_path.is_relative_to(custom_vllm_root)" in output
    assert "runtime_nemo_rl_commit=$(git rev-parse HEAD)" in output
    assert expected_nemo_rl_commit in output
    assert "git status --porcelain --untracked-files=all" in output
    assert "runtime_vllm_commit=$(git -C" in output
    assert "run_manifest.json" in output
    assert '"model": "nemotron3-super"' in output
    assert '"linear_backend": "flashinfer_cutedsl"' in output


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
    assert (
        result.stdout.splitlines().count(
            f"experiment_root={tmp_path / 'runs' / 'flashinfer_cutedsl'}"
        )
        == 1
    )
    assert (
        result.stdout.splitlines().count(
            f"experiment_root={tmp_path / 'runs' / 'flashinfer_cutlass'}"
        )
        == 1
    )


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_submit_rejects_dirty_nemo_rl_source(tmp_path: Path, launcher: Path) -> None:
    source_root = tmp_path / "nemo-rl"
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    (source_root / "dirty.py").write_text("dirty = True\n")

    custom_vllm = tmp_path / "vllm"
    subprocess.run(["git", "init", "-q", str(custom_vllm)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(custom_vllm),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (custom_vllm / "nemo-rl.env").write_text("# test environment\n")
    vllm_commit = subprocess.check_output(
        ["git", "-C", str(custom_vllm), "rev-parse", "HEAD"], text=True
    ).strip()
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    env = os.environ | {
        "ACTION": "test-only",
        "BACKEND": "flashinfer_cutedsl",
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPECTED_NEMO_RL_BASE_COMMIT": source_commit,
        "EXPECTED_VLLM_COMMIT": vllm_commit,
        "REPO_DIR_OVERRIDE": str(source_root),
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "NeMo-RL source is not clean" in result.stderr


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_submit_allows_pinned_custom_vllm_inside_clean_source(
    tmp_path: Path, launcher: Path
) -> None:
    source_root = tmp_path / "nemo-rl"
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    custom_vllm = source_root / "3rdparty" / "vllm"
    custom_vllm.parent.mkdir()
    subprocess.run(["git", "init", "-q", str(custom_vllm)], check=True)
    (custom_vllm / "nemo-rl.env").write_text("# test environment\n")
    subprocess.run(
        ["git", "-C", str(custom_vllm), "add", "nemo-rl.env"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(custom_vllm),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    vllm_commit = subprocess.check_output(
        ["git", "-C", str(custom_vllm), "rev-parse", "HEAD"], text=True
    ).strip()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_sbatch.chmod(0o755)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    env = os.environ | {
        "ACTION": "test-only",
        "BACKEND": "flashinfer_cutedsl",
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
        "EXPECTED_VLLM_COMMIT": vllm_commit,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REPO_DIR_OVERRIDE": str(source_root),
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
def test_emitted_job_command_is_valid_bash(tmp_path: Path, launcher: Path) -> None:
    env = os.environ | {
        "ACTION": "dry-run",
        "BACKEND": "flashinfer_cutedsl",
        "RUN_ID": "test-run",
        "WORK_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(launcher)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    command_start = result.stdout.index("set -euo pipefail\n")
    command = result.stdout[command_start:]

    syntax_check = subprocess.run(
        ["bash", "-n"],
        input=command,
        capture_output=True,
        text=True,
    )

    assert syntax_check.returncode == 0, syntax_check.stderr
