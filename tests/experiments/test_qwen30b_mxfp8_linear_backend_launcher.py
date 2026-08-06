from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT / "experiments" / "qwen30b_mxfp8_linear_backends" / "submit_ptyche.sh"
)
MATRIX_LAUNCHER = LAUNCHER.with_name("submit_matrix_ptyche.sh")
PREPARE_SCRIPT = LAUNCHER.with_name("prepare_custom_vllm_ptyche.sh")
BUILD_CUSTOM_VLLM_SCRIPT = REPO_ROOT / "tools" / "build-custom-vllm.sh"
PROVENANCE_HELPER = (
    REPO_ROOT / "experiments/mxfp8_linear_backend_model_matrix/provenance.sh"
)
VLLM_BUILD_STATE_MARKER = "nemo-rl-build-state.sha256"


def _dry_run(
    tmp_path: Path, backend: str, extra_env: dict[str, str] | None = None
) -> str:
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
        "QOS": "interactive",
        "RUN_ID": "test-run",
        "WANDB_MODE": "disabled",
        "WORK_ROOT": str(tmp_path),
    }
    if extra_env is not None:
        env.update(extra_env)
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
            "flashinfer_trtllm_adaptive",
        )
    }

    for backend, output in outputs.items():
        effective_backend = (
            "flashinfer_trtllm" if backend == "flashinfer_trtllm_adaptive" else backend
        )
        assert f"linear_backend={effective_backend}" in output
        assert "policy.train_global_batch_size=2048" in output
        assert "policy.generation.vllm_cfg.tensor_parallel_size=1" in output
        assert "policy.generation.vllm_cfg.enforce_eager=false" in output
        assert "policy.generation.vllm_cfg.precision=fp8" in output
        assert "policy.generation.vllm_cfg.is_mx=true" in output
        assert "quantization_ignored_layer_kws=[lm_head,mlp.gate]" in output
        assert "moe_backend=flashinfer_trtllm" in output
        assert "cluster.num_nodes=4" in output
        assert "cluster.gpus_per_node=4" in output
        assert "cluster.segment_size=4" in output
        assert "grpo.max_num_steps=8" in output
        assert (
            "NRL_VENV_BOOTSTRAP_PACKAGES='--torch-backend cu130 "
            "torch==2.11.0 numpy setuptools setuptools-rust setuptools-scm'" in output
        )
        assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1" in output
        assert "--qos=interactive" in output
        assert "uv run --frozen --extra vllm" not in output
        assert "uv venv" not in output
        assert "uv pip install --python" not in output
        assert output.count("/bin/python") >= 3
        assert "/.cache/nemo-rl-vllm0251-worker-venvs" in output
        assert "export NRL_FORCE_REBUILD_VENVS=false" in output

    adaptive_output = outputs["flashinfer_trtllm_adaptive"]
    assert "VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK=1" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT=adaptive" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M=256" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE=" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_SHA256=" in adaptive_output
    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64=" in adaptive_output

    for backend in (
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
        "flashinfer_trtllm",
    ):
        assert "VLLM_MXFP8_DENSE_TRTLLM_EXACT_TACTIC_FILE=" not in outputs[backend]


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


def test_long_context_overrides_are_forwarded(tmp_path: Path) -> None:
    output = _dry_run(
        tmp_path,
        "flashinfer_trtllm_adaptive",
        {
            "MAX_STEPS": "20",
            "MAX_TOTAL_SEQUENCE_LENGTH": "34816",
            "MAX_NEW_TOKENS": "32768",
            "MAX_INPUT_SEQUENCE_LENGTH": "2048",
            "NUM_PROMPTS_PER_STEP": "48",
            "NUM_GENERATIONS_PER_PROMPT": "4",
            "TRAIN_GLOBAL_BATCH_SIZE": "192",
            "ACTIVATION_CHECKPOINTING": "true",
            "SEQUENCE_PACKING": "false",
            "LOGPROB_BATCH_SIZE": "1",
            "LOGPROB_CHUNK_SIZE": "2048",
            "DEFER_FP32_LOGITS": "true",
            "GPU_MEMORY_UTILIZATION": "0.5",
        },
    )

    assert "grpo.max_num_steps=20" in output
    assert "policy.max_total_sequence_length=34816" in output
    assert "policy.generation.max_new_tokens=32768" in output
    assert "policy.generation.vllm_cfg.max_model_len=34816" in output
    assert "data.max_input_seq_length=2048" in output
    assert "grpo.num_prompts_per_step=48" in output
    assert "grpo.num_generations_per_prompt=4" in output
    assert "policy.train_global_batch_size=192" in output
    assert "policy.megatron_cfg.activation_checkpointing=true" in output
    assert "policy.sequence_packing.enabled=false" in output
    assert "policy.logprob_batch_size=1" in output
    assert "policy.logprob_chunk_size=2048" in output
    assert "policy.megatron_cfg.defer_fp32_logits=true" in output
    assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.5" in output
    assert '"logprob_batch_size": 1' in output
    assert '"logprob_chunk_size": 2048' in output
    assert '"activation_checkpointing": True' in output
    assert '"defer_fp32_logits": True' in output
    assert '"sequence_packing": False' in output


def test_dry_run_captures_runtime_provenance_and_manifest(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "flashinfer_cutedsl")
    expected_nemo_rl_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    custom_vllm_root = tmp_path / "vllm"

    assert f"source {custom_vllm_root}/nemo-rl.env" in output
    assert "vllm_path = Path(vllm.__file__).resolve()" in output
    assert f'custom_vllm_root = Path("{custom_vllm_root}").resolve()' in output
    assert "vllm_path.is_relative_to(custom_vllm_root)" in output
    assert "runtime_nemo_rl_commit=$(git rev-parse HEAD)" in output
    assert expected_nemo_rl_commit in output
    assert "git status --porcelain --untracked-files=all" in output
    assert "runtime_vllm_commit=$(git -C" in output
    assert "run_manifest.json" in output
    assert '"model": "Qwen/Qwen3-30B-A3B"' in output
    assert '"nemo_rl_commit"' in output
    assert '"dependency_state_sha256"' in output
    assert '"vllm_commit"' in output
    assert '"vllm_source_sha256"' in output
    assert '"vllm_dependency_state_sha256"' in output
    assert '"vllm_source_files_clean": True' in output
    assert '"container"' in output
    assert '"recipe"' in output
    assert '"recipe_sha256"' in output
    assert '"cuda_graph": True' in output
    assert '"precision": "fp8"' in output
    assert '"is_mx": True' in output
    assert '"quantization_ignored_layer_kws": ["lm_head", "mlp.gate"]' in output
    assert '"moe_backend": "flashinfer_trtllm"' in output
    assert '"num_nodes": 4' in output
    assert '"gpus_per_node": 4' in output
    assert '"segment_size": 4' in output
    assert '"num_prompts_per_step": 64' in output
    assert '"num_generations_per_prompt": 32' in output
    assert '"train_global_batch_size": 2048' in output
    assert '"max_total_sequence_length": 4096' in output
    assert '"max_input_sequence_length": 4096' in output
    assert '"max_new_tokens": 4096' in output
    assert '"max_model_len": 4096' in output
    assert '"generation_tensor_parallel_size": 1' in output
    assert '"max_steps": 8' in output
    assert '"gpu_memory_utilization": 0.6' in output
    assert '"logprob_batch_size": 2' in output
    assert '"logprob_chunk_size": None' in output
    assert '"activation_checkpointing": False' in output
    assert '"defer_fp32_logits": False' in output
    assert '"sequence_packing": True' in output
    assert '"linear_backend": "flashinfer_cutedsl"' in output


def test_model_matrix_defaults_to_two_isolated_backend_roots(tmp_path: Path) -> None:
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
    assert result.stdout.splitlines().count("backend=flashinfer_trtllm") == 0
    assert result.stdout.splitlines().count("backend=flashinfer_trtllm_adaptive") == 0
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


def test_custom_vllm_build_is_recoverable() -> None:
    prepare_text = PREPARE_SCRIPT.read_text()
    build_text = BUILD_CUSTOM_VLLM_SCRIPT.read_text()
    provenance_text = PROVENANCE_HELPER.read_text()
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()

    assert "3rdparty/vllm/nemo-rl.env" in prepare_text
    assert "incomplete=${PREP_ROOT}/vllm.incomplete" in prepare_text
    assert "incomplete=3rdparty/vllm.incomplete" not in prepare_text
    assert "git submodule update --init --recursive --depth 1" not in prepare_text
    assert "assert_preparation_scope_clean" in prepare_text
    assert "git submodule update --init --recursive" in prepare_text
    assert '":(exclude)pyproject.toml"' in prepare_text
    assert '":(exclude)uv.lock"' in prepare_text
    assert '":(exclude)3rdparty/vllm"' in prepare_text
    assert "Preparation found disallowed NeMo-RL source changes" in prepare_text
    assert "\"${vllm_root}/.venv/bin/python\" -c 'import vllm'" in provenance_text
    assert "3rdparty/vllm/.venv/bin/python - <<'PY'" in prepare_text
    assert "uv run --frozen python - <<'PY'" not in prepare_text
    assert "3rdparty/vllm/.venv uv lock" in prepare_text
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1" in prepare_text
    assert "setuptools_rust" in build_text
    assert "existing_vllm_valid=false" in prepare_text
    assert "mxfp8_vllm_reuse_state_valid" in prepare_text
    assert "Preserving polluted custom vLLM checkout" in prepare_text
    polluted_branch = prepare_text.split(
        "elif ! mxfp8_assert_vllm_tracked_state 3rdparty/vllm; then", 1
    )[1].split("else", 1)[0]
    assert "exit 1" not in polluted_branch
    assert VLLM_BUILD_STATE_MARKER in build_text
    assert build_text.index("cat <<EOF >$BUILD_DIR/nemo-rl.env") < build_text.index(
        'echo "Updating repo pyproject.toml to point vLLM to local clone..."'
    )
    assert "3rdparty/vllm/.venv/bin/python -c 'import vllm'" in prepare_text
    assert "Replacing custom vLLM commit" in prepare_text
    assert "mxfp8_vllm_build_state_matches 3rdparty/vllm" in prepare_text
    assert 'SBATCH_ARGS+=(--qos="${QOS}")' in prepare_text
    assert "TORCH_REQUIREMENT=$(sed -nE" in build_text
    assert "VLLM_TORCH_BACKEND:-cu130" in build_text
    assert "torch==2.10.0" not in build_text
    assert 'git restore --source="$GIT_REF" --worktree -- .' not in build_text
    assert "pyproject.toml|requirements/*" in build_text
    assert "Disallowed tracked vLLM changes after build" in build_text
    assert 'vllm = ["setuptools", "setuptools-rust"]' in pyproject_text


def test_preparation_pins_ray_to_the_container_version() -> None:
    prepare_text = PREPARE_SCRIPT.read_text()

    detect = "CONTAINER_RAY_VERSION=\\$(python3 -c 'import ray; print(ray.__version__)')"
    pin = 'uv add --frozen --bounds exact "ray[default]==\\${CONTAINER_RAY_VERSION}"'
    build_dependency = (
        "uv pip install --python 3rdparty/vllm/.venv/bin/python "
        "poetry-dynamic-versioning poetry pybind11"
    )
    cache_ray = (
        'uv pip install --python 3rdparty/vllm/.venv/bin/python '
        '"ray[default]==\\${CONTAINER_RAY_VERSION}"'
    )
    limit_lock_environment = (
        'uv_config["environments"] = ["python_version == \'3.13\' and '
        'sys_platform == \'linux\' and platform_machine == \'aarch64\'"]'
    )
    lock = (
        "UV_PROJECT_ENVIRONMENT=${REPO_DIR}/3rdparty/vllm/.venv "
        "uv lock --no-build-isolation --refresh-package ray"
    )

    assert detect in prepare_text
    assert pin in prepare_text
    assert build_dependency in prepare_text
    assert cache_ray in prepare_text
    assert limit_lock_environment in prepare_text
    assert "uv lock --offline" not in prepare_text
    assert "Ray lock mismatch" in prepare_text
    assert (
        prepare_text.index(detect)
        < prepare_text.index(pin)
        < prepare_text.index(build_dependency)
        < prepare_text.index(cache_ray)
        < prepare_text.index(limit_lock_environment)
        < prepare_text.index(lock)
    )


def test_launchers_reject_ray_version_drift_before_the_workload() -> None:
    launchers = (
        LAUNCHER,
        REPO_ROOT
        / "experiments/qwen235b_mxfp8_linear_backends/submit_cluster.sh",
        REPO_ROOT
        / "experiments/nemotron3_super_mxfp8_linear_backends/submit_ptyche.sh",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert "MXFP8_CONTAINER_RAY_VERSION" in text
        assert "Ray version mismatch before workload launch" in text


def test_launchers_reuse_prebuilt_vllm_environment_without_syncing() -> None:
    launchers = (
        LAUNCHER,
        REPO_ROOT
        / "experiments/qwen235b_mxfp8_linear_backends/submit_cluster.sh",
        REPO_ROOT
        / "experiments/nemotron3_super_mxfp8_linear_backends/submit_ptyche.sh",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert "PREBUILT_VLLM_VENV" in text
        assert "DRIVER_VENV=${DRIVER_VENV:-${PREBUILT_VLLM_VENV}}" in text
        assert "WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-${SHARED_VENV_ROOT}}" in text
        assert "uv venv ${DRIVER_VENV}" not in text
        assert "uv pip install --python ${DRIVER_VENV}" not in text
        assert "uv run --frozen --extra vllm python" not in text
        assert "uv run --frozen --extra vllm examples/run_grpo.py" not in text
        assert "${DRIVER_VENV}/bin/python examples/run_grpo.py" in text


def test_preparation_builds_shared_worker_venv_once() -> None:
    prepare_text = PREPARE_SCRIPT.read_text()

    assert "SHARED_WORKER_VENV_ROOT" in prepare_text
    assert "nemo-rl-vllm0251-worker-venvs" in prepare_text
    assert "create_local_venv(" in prepare_text
    assert "VllmAsyncGenerationWorker" in prepare_text
    assert "force_rebuild=True" in prepare_text


def test_custom_vllm_build_preserves_compatibility_requirements_for_lock(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "vllm-origin"
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    requirements = origin / "requirements"
    requirements.mkdir()
    (requirements / "cuda.txt").write_text(
        "torch==2.11.0 # build torch\nxformers==0.0.30; platform_system == 'Linux'\n"
    )
    (origin / "use_existing_torch.py").write_text("# fixture\n")
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(origin),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    git_ref = subprocess.check_output(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], text=True
    ).strip()

    repo = tmp_path / "nemo-rl"
    tools_dir = repo / "tools"
    tools_dir.mkdir(parents=True)
    (repo / "3rdparty").mkdir()
    shutil.copy2(BUILD_CUSTOM_VLLM_SCRIPT, tools_dir / BUILD_CUSTOM_VLLM_SCRIPT.name)
    (repo / "pyproject.toml").write_text("[project]\nname = 'nemo-rl'\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == lock ]]; then\n'
        "  cp 3rdparty/vllm/requirements/cuda.txt uv.lock\n"
        "elif [[ \"$*\" == *'python -'* ]]; then\n"
        "  cat >/dev/null\n"
        "fi\n"
    )
    fake_uv.chmod(0o755)
    fake_find = fake_bin / "find"
    fake_find.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "for file in requirements/*.txt; do\n"
        "  perl -0pi -e 's/#.*$//mg; s/^[ \\t]*\\n//mg; "
        's/^(xformers)==[^;\\s]*/$1==0.0.32.post1/mg\' "$file"\n'
        "done\n"
    )
    fake_find.chmod(0o755)
    fake_realpath = fake_bin / "realpath"
    fake_realpath.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "target=$1\n"
        'cd "$(dirname "$target")"\n'
        'printf \'%s/%s\\n\' "$PWD" "$(basename "$target")"\n'
    )
    fake_realpath.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(tools_dir / BUILD_CUSTOM_VLLM_SCRIPT.name),
            str(origin),
            git_ref,
            "https://example.invalid/vllm.whl",
        ],
        cwd=repo,
        env=os.environ
        | {
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "UV_PROJECT_ENVIRONMENT": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    built_requirements = repo / "3rdparty/vllm/requirements/cuda.txt"
    requirement_text = built_requirements.read_text()
    assert "# build torch" not in requirement_text
    assert "xformers==0.0.32.post1" in requirement_text
    assert (repo / "uv.lock").read_text() == requirement_text
    changed_paths = subprocess.check_output(
        ["git", "-C", str(repo / "3rdparty/vllm"), "diff", "--name-only"],
        text=True,
    ).splitlines()
    assert changed_paths == ["requirements/cuda.txt"]
    dependency_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo / "3rdparty/vllm"),
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
            "pyproject.toml",
            "requirements/",
        ]
    )
    expected_fingerprint = hashlib.sha256(dependency_diff).hexdigest()
    marker = repo / "3rdparty/vllm" / VLLM_BUILD_STATE_MARKER
    assert marker.read_text() == f"{expected_fingerprint}\n"


def _reuse_gate_result(
    tmp_path: Path, marker_state: str
) -> tuple[subprocess.CompletedProcess[str], Path, str]:
    custom_vllm = tmp_path / "vllm"
    subprocess.run(["git", "init", "-q", str(custom_vllm)], check=True)
    requirements = custom_vllm / "requirements"
    requirements.mkdir()
    requirements_path = requirements / "cuda.txt"
    requirements_path.write_text("torch==2.11.0\nxformers==0.0.30\n")
    requirements_test = requirements / "test"
    requirements_test.mkdir()
    requirements_input_path = requirements_test / "cuda.in"
    requirements_input_path.write_text("torch==2.11.0\n")
    pyproject_path = custom_vllm / "pyproject.toml"
    pyproject_path.write_text("[build-system]\nrequires = ['torch == 2.11.0']\n")
    subprocess.run(["git", "-C", str(custom_vllm), "add", "."], check=True)
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
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(custom_vllm), "rev-parse", "HEAD"], text=True
    ).strip()
    requirements_path.write_text("torch==2.11.0\nxformers==0.0.32.post1\n")
    requirements_input_path.write_text("torch==2.11.0\n# prepared\n")
    pyproject_path.write_text("[build-system]\nrequires = []\n")
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
            "pyproject.toml",
            "requirements/",
        ]
    )
    expected_fingerprint = hashlib.sha256(dependency_diff).hexdigest()
    if marker_state == "valid":
        (custom_vllm / VLLM_BUILD_STATE_MARKER).write_text(f"{expected_fingerprint}\n")
    elif marker_state == "stale":
        (custom_vllm / VLLM_BUILD_STATE_MARKER).write_text(f"{'0' * 64}\n")

    venv_python = custom_vllm / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    venv_python.chmod(0o755)
    wheel = "https://example.invalid/vllm.whl"
    (custom_vllm / "nemo-rl.env").write_text(
        f"export VLLM_GIT_REF={commit}\n"
        f"export VLLM_PRECOMPILED_WHEEL_LOCATION={wheel}\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; mxfp8_vllm_reuse_state_valid "$2" "$3" "$4"',
            "bash",
            str(PROVENANCE_HELPER),
            str(custom_vllm),
            commit,
            wheel,
        ],
        capture_output=True,
        text=True,
    )
    return result, requirements_path, expected_fingerprint


def test_prepare_reuse_gate_rejects_missing_and_stale_build_state(
    tmp_path: Path,
) -> None:
    missing, _, _ = _reuse_gate_result(tmp_path / "missing", "missing")
    stale, _, _ = _reuse_gate_result(tmp_path / "stale", "stale")

    assert missing.returncode == 1
    assert "Missing vLLM build-state marker" in missing.stderr
    assert stale.returncode == 1
    assert "Stale vLLM build-state marker" in stale.stderr


def test_prepare_reuse_gate_accepts_matching_build_state(tmp_path: Path) -> None:
    result, requirements_path, expected_fingerprint = _reuse_gate_result(
        tmp_path, "valid"
    )

    assert result.returncode == 0, result.stderr
    assert requirements_path.read_text().endswith("xformers==0.0.32.post1\n")
    assert (requirements_path.parent.parent / VLLM_BUILD_STATE_MARKER).read_text() == (
        f"{expected_fingerprint}\n"
    )


def test_prepare_emits_valid_scoped_job_command(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(PREPARE_SCRIPT)],
        check=True,
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "ACTION": "dry-run",
            "PREP_ROOT": str(tmp_path / "prepare"),
            "WORK_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    command_start = result.stdout.index("set -euo pipefail\n")
    command = result.stdout[command_start:]

    syntax_check = subprocess.run(
        ["bash", "-n"], input=command, capture_output=True, text=True
    )

    assert syntax_check.returncode == 0, syntax_check.stderr
    assert str(tmp_path / "prepare" / "vllm.incomplete") in command
