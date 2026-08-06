from __future__ import annotations

import hashlib
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
RECIPE_PATHS = (
    "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml",
    "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml",
    "examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml",
)


def _initialize_source_repo(source_root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    (source_root / "pyproject.toml").write_text("[project]\nname = 'nemo-rl'\n")
    (source_root / "uv.lock").write_text("version = 1\n")
    for recipe_path in RECIPE_PATHS:
        full_path = source_root / recipe_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"recipe: {full_path.name}\n")
    subprocess.run(["git", "-C", str(source_root), "add", "."], check=True)
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
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _initialize_custom_vllm(custom_vllm: Path) -> str:
    custom_vllm.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(custom_vllm)], check=True)
    (custom_vllm / "kernel.py").write_text("BACKEND = 'mxfp8'\n")
    (custom_vllm / "nemo-rl.env").write_text("# test environment\n")
    requirements = custom_vllm / "requirements"
    requirements.mkdir()
    (requirements / "cuda.txt").write_text("torch==2.11.0\nxformers==0.0.30\n")
    subprocess.run(
        ["git", "-C", str(custom_vllm), "add", "."],
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
    return subprocess.check_output(
        ["git", "-C", str(custom_vllm), "rev-parse", "HEAD"], text=True
    ).strip()


def _dependency_state_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("pyproject.toml", "uv.lock"):
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update((source_root / filename).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _vllm_source_sha256(custom_vllm: Path) -> str:
    archive = subprocess.check_output(
        ["git", "-C", str(custom_vllm), "archive", "--format=tar", "HEAD"]
    )
    return hashlib.sha256(archive).hexdigest()


def _vllm_dependency_state_sha256(custom_vllm: Path) -> str:
    dependency_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(custom_vllm),
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-renames",
            "--diff-algorithm=myers",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "HEAD",
            "--",
            "requirements/",
        ]
    )
    return hashlib.sha256(dependency_diff).hexdigest()


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
        assert "policy.generation.max_new_tokens=8192" in output
        assert "policy.generation.vllm_cfg.max_model_len=8192" in output
        assert "data.max_input_seq_length=8192" in output
        assert "policy.generation.vllm_cfg.tensor_parallel_size=4" in output
        assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.7" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "policy.generation.vllm_cfg.precision=fp8" in output
        assert "policy.generation.vllm_cfg.is_mx=true" in output
        assert "policy.logprob_batch_size=1" in output
        assert "policy.logprob_chunk_size=2048" in output
        assert "policy.megatron_cfg.activation_checkpointing=true" in output
        assert "policy.megatron_cfg.defer_fp32_logits=true" in output
        assert "policy.sequence_packing.enabled=true" in output
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
    assert '"model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"' in output
    assert '"dependency_state_sha256"' in output
    assert '"vllm_source_sha256"' in output
    assert '"vllm_dependency_state_sha256"' in output
    assert '"vllm_source_files_clean": True' in output
    assert '"recipe_sha256"' in output
    assert '"precision": "fp8"' in output
    assert '"is_mx": True' in output
    assert '"num_nodes": 32' in output
    assert '"gpus_per_node": 4' in output
    assert '"segment_size": 8' in output
    assert '"num_prompts_per_step": 32' in output
    assert '"num_generations_per_prompt": 8' in output
    assert '"train_global_batch_size": 256' in output
    assert '"max_total_sequence_length": 8192' in output
    assert '"max_input_sequence_length": 8192' in output
    assert '"max_new_tokens": 8192' in output
    assert '"max_model_len": 8192' in output
    assert '"generation_tensor_parallel_size": 4' in output
    assert '"max_steps": 8' in output
    assert '"gpu_memory_utilization": 0.7' in output
    assert '"logprob_batch_size": 1' in output
    assert '"logprob_chunk_size": 2048' in output
    assert '"activation_checkpointing": True' in output
    assert '"defer_fp32_logits": True' in output
    assert '"sequence_packing": True' in output
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
    _initialize_source_repo(source_root)
    custom_vllm = source_root / "3rdparty" / "vllm"
    vllm_commit = _initialize_custom_vllm(custom_vllm)

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
def test_submit_accepts_only_preparation_dependency_mutations(
    tmp_path: Path, launcher: Path
) -> None:
    source_root = tmp_path / "nemo-rl"
    _initialize_source_repo(source_root)
    (source_root / "pyproject.toml").write_text(
        "[project]\nname = 'nemo-rl'\ndependencies = ['setuptools_scm']\n"
    )
    (source_root / "uv.lock").write_text(
        "version = 1\n[[package]]\nname = 'vllm'\nsource = '3rdparty/vllm'\n"
    )
    expected_dependency_sha = _dependency_state_sha256(source_root)

    custom_vllm = source_root / "3rdparty" / "vllm"
    vllm_commit = _initialize_custom_vllm(custom_vllm)
    expected_vllm_source_sha = _vllm_source_sha256(custom_vllm)
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
    assert expected_dependency_sha in result.stdout
    assert expected_vllm_source_sha in result.stdout
    assert "runtime_dependency_state_sha256=" in result.stdout
    assert "runtime_vllm_source_sha256=" in result.stdout
    assert '"vllm_source_files_clean": True' in result.stdout


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
@pytest.mark.parametrize("staged", (False, True))
def test_submit_allows_fingerprinted_vllm_requirements_rewrites(
    tmp_path: Path, launcher: Path, staged: bool
) -> None:
    source_root = tmp_path / "nemo-rl"
    _initialize_source_repo(source_root)
    custom_vllm = source_root / "3rdparty" / "vllm"
    vllm_commit = _initialize_custom_vllm(custom_vllm)
    requirements_path = custom_vllm / "requirements/cuda.txt"
    requirements_path.write_text("torch==2.11.0\nxformers==0.0.32.post1\n")
    if staged:
        subprocess.run(
            ["git", "-C", str(custom_vllm), "add", "requirements/cuda.txt"],
            check=True,
        )
    expected_dependency_sha = _vllm_dependency_state_sha256(custom_vllm)

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
    assert expected_dependency_sha in result.stdout
    assert requirements_path.read_text().endswith("xformers==0.0.32.post1\n")
    assert "runtime_vllm_dependency_state_sha256=" in result.stdout


@pytest.mark.parametrize("launcher", ALL_LAUNCHERS)
@pytest.mark.parametrize("staged", (False, True))
def test_submit_rejects_dirty_tracked_custom_vllm_source(
    tmp_path: Path, launcher: Path, staged: bool
) -> None:
    source_root = tmp_path / "nemo-rl"
    _initialize_source_repo(source_root)
    custom_vllm = source_root / "3rdparty" / "vllm"
    vllm_commit = _initialize_custom_vllm(custom_vllm)
    (custom_vllm / "kernel.py").write_text("BACKEND = 'modified'\n")
    if staged:
        subprocess.run(["git", "-C", str(custom_vllm), "add", "kernel.py"], check=True)
    container = tmp_path / "nemo-rl.sqsh"
    container.touch()

    env = os.environ | {
        "ACTION": "test-only",
        "BACKEND": "flashinfer_cutedsl",
        "CONTAINER": str(container),
        "CUSTOM_VLLM_ROOT": str(custom_vllm),
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
    assert "Custom vLLM tracked files are not clean" in result.stderr


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
