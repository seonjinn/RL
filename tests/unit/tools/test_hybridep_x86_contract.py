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

import ast
import os
from pathlib import Path
import subprocess

import pytest


DEEPEP_COMMIT = "f725d29699f5bda9ba789456bb9579af69844685"
OLD_X86_COMMIT = "29d31c095796f3c8ece47ee9cdcc167051bbeed9"


def test_linux_x86_uses_latest_hybridep_commit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pyproject = (project_root / "pyproject.toml").read_text()
    x86_dependencies = [
        line
        for line in pyproject.splitlines()
        if "deep_ep @ git+" in line and "platform_machine == 'x86_64'" in line
    ]

    assert len(x86_dependencies) == 4
    assert all(DEEPEP_COMMIT in line for line in x86_dependencies)
    assert OLD_X86_COMMIT not in pyproject


def test_lock_does_not_retain_the_pre_hybridep_x86_commit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    lock = (project_root / "uv.lock").read_text()

    assert OLD_X86_COMMIT not in lock
    assert f"DeepEP.git?rev={DEEPEP_COMMIT}" in lock


def test_create_local_venv_does_not_set_a_per_actor_uv_cache() -> None:
    project_root = Path(__file__).resolve().parents[3]
    source = (project_root / "nemo_rl" / "utils" / "venvs.py").read_text()
    module = ast.parse(source)
    create_local_venv = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_local_venv"
    )
    uv_cache_assignments = [
        node
        for node in ast.walk(create_local_venv)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "UV_CACHE_DIR"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]

    assert not uv_cache_assignments


def test_x86_profiles_are_matched() -> None:
    project_root = Path(__file__).resolve().parents[3]
    profile_dir = (
        project_root / "scripts" / "experiments" / "oci-hsg" / "hybridep" / "models"
    )
    pairs = (
        (
            "qwen3-30ba3b-4n8g-x86",
            "grpo-qwen3-30ba3b-4n8g",
            4,
            4,
            None,
        ),
        (
            "qwen3-235b-16n8g-x86",
            "grpo-qwen3-235b-16n8g",
            16,
            16,
            None,
        ),
        (
            "nemotron3-super-120ba12b-32n8g-sync-x86",
            "grpo-nemotron3-super-120BA12B-32n8g",
            32,
            16,
            None,
        ),
        (
            "deepseek-v3-32n8g-x86",
            "grpo-deepseek-v3-32n8g",
            32,
            16,
            "MODEL_TOKENIZER_OVERRIDE_ENV=NRL_DEEPSEEK_V3_BF16_CKPT",
        ),
    )

    for profile_name, recipe_name, nodes, segment_size, required_line in pairs:
        baseline = (profile_dir / f"{profile_name}.env").read_text()
        hybridep = (profile_dir / f"{profile_name}-hybridep.env").read_text()
        common_lines = {
            "export NCCL_NVLS_ENABLE=0",
            "DISPATCHER_MODE=recipe",
            "NRL_FORCE_REBUILD_VENVS=false",
            "REQUIRE_PREBUILT_ACTOR_VENVS=true",
            "REQUIRE_DEEPEP_WHEEL=true",
            f"NUM_ACTOR_NODES=${{NUM_ACTOR_NODES:-{nodes}}}",
            "GPUS_PER_NODE=${GPUS_PER_NODE:-8}",
            f"SEGMENT_SIZE=${{SEGMENT_SIZE:-{segment_size}}}",
            "MAX_STEPS=${MAX_STEPS:-20}",
            "TIME_LIMIT=${TIME_LIMIT:-04:00:00}",
            f"DEFAULT_DEEPEP_COMMIT={DEEPEP_COMMIT}",
        }
        if required_line is not None:
            common_lines.add(required_line)

        assert common_lines <= set(baseline.splitlines())
        assert common_lines <= set(hybridep.splitlines())
        assert (
            "CONFIG_PATH=examples/configs/recipes/llm/performance/"
            f"{recipe_name}-alltoall.yaml"
        ) in baseline
        assert (
            f"CONFIG_PATH=examples/configs/recipes/llm/performance/{recipe_name}.yaml"
        ) in hybridep


def test_recursive_checkout_uses_the_branch_that_exposes_the_bridge_gitlink() -> None:
    project_root = Path(__file__).resolve().parents[3]
    gitmodules = (project_root / ".gitmodules").read_text()
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    ).read_text()

    assert "branch = sna/super-autobridge-config-reuse-20260727" in gitmodules
    assert (
        "git -c fetch.recurseSubmodules=false pull --ff-only --recurse-submodules=no"
    ) in launcher
    assert "git submodule update --init --recursive" in launcher


def test_launcher_allows_shared_non_lustre_logs_without_changing_defaults() -> None:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    ).read_text()

    assert "EXTRA_MOUNTS=${EXTRA_MOUNTS:-}" in launcher
    assert 'MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"' in launcher
    assert "NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}" in launcher
    assert "DRIVER_VENV=${DRIVER_VENV:-}" in launcher
    assert "RAY_VENV=${RAY_VENV:-}" in launcher
    assert "UV_LOCK_TIMEOUT=${UV_LOCK_TIMEOUT:-1800}" in launcher
    assert "export UV_LOCK_TIMEOUT" in launcher
    assert "uv_lock_timeout=%q" in launcher
    assert 'driver_args=(env "UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}"' in launcher
    assert 'PATH="${RAY_VENV}/bin:${PATH}"' in launcher
    assert "export PATH" in launcher
    assert "export RAY_VENV" in launcher
    assert "ray_venv=%q" in launcher

    ray_submit = (project_root / "ray.sub").read_text()
    assert 'RAY_BIN="${RAY_VENV}/bin/ray"' in ray_submit
    assert '"$RAY_BIN" status' in ray_submit
    assert '"$RAY_BIN" stop' in ray_submit
    assert '"$RAY_BIN" start --head' in ray_submit
    assert '"$RAY_BIN" start --address' in ray_submit


def test_x86_driver_venv_preparation_allows_time_for_actor_prefetch() -> None:
    project_root = Path(__file__).resolve().parents[3]
    submitter = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "submit_driver_venv.sh"
    ).read_text()

    assert "TIME_LIMIT=${TIME_LIMIT:-02:00:00}" in submitter


def _write_fake_command(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize(
    "traversal_variable",
    ("DRIVER_VENV", "UV_CACHE_DIR", "NEMO_RL_VENV_DIR"),
)
def test_driver_venv_submitter_rejects_paths_that_traverse_outside_lustre(
    tmp_path: Path, traversal_variable: str
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    submitter = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "submit_driver_venv.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_command(
        fake_bin / "git",
        """#!/bin/sh
case "$*" in
  *"rev-parse --show-toplevel"*) printf '%s\\n' "$FAKE_PROJECT_ROOT" ;;
esac
""",
    )
    _write_fake_command(
        fake_bin / "sshare",
        """#!/bin/sh
printf '%s|%s|%s|\\n' 'nemotron_sw_pre' "$FAKE_USER" '0.900000'
""",
    )
    _write_fake_command(
        fake_bin / "sbatch",
        """#!/bin/sh
if [ "$1" != "--test-only" ]; then printf '%s\\n' '999999'; fi
""",
    )
    paths = {
        "DRIVER_VENV": "/lustre/driver-venv",
        "UV_CACHE_DIR": "/lustre/uv-cache",
        "NEMO_RL_VENV_DIR": "/lustre/actor-venvs",
    }
    paths[traversal_variable] = str(
        Path("/lustre") / ".." / tmp_path.relative_to("/") / traversal_variable
    )
    result = subprocess.run(
        ["bash", str(submitter)],
        check=False,
        cwd=project_root,
        env={
            **os.environ,
            **paths,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_PROJECT_ROOT": str(project_root),
            "FAKE_USER": subprocess.check_output(["id", "-un"], text=True).strip(),
            "CONTAINER": "/lustre/container.sqsh",
            "VENV_LOG_DIR": str(tmp_path / "logs"),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{traversal_variable} must be on shared /lustre storage" in result.stderr


@pytest.mark.parametrize(
    "traversal_variable",
    ("DRIVER_VENV", "UV_CACHE_DIR", "NEMO_RL_VENV_DIR"),
)
def test_driver_venv_preparer_rejects_paths_that_traverse_outside_lustre(
    tmp_path: Path, traversal_variable: str
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    preparer = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "prepare_driver_venv.sbatch"
    )
    paths = {
        "DRIVER_VENV": "/lustre/driver-venv",
        "UV_CACHE_DIR": "/lustre/uv-cache",
        "NEMO_RL_VENV_DIR": "/lustre/actor-venvs",
    }
    paths[traversal_variable] = str(
        Path("/lustre") / ".." / tmp_path.relative_to("/") / traversal_variable
    )
    result = subprocess.run(
        ["bash", str(preparer)],
        check=False,
        cwd=project_root,
        env={
            **os.environ,
            **paths,
            "CONTAINER": "/lustre/container.sqsh",
            "HYBRID_EP_PROJECT_ROOT": str(project_root),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{traversal_variable} must be on shared /lustre storage" in result.stderr


def _prepare_payload(project_root: Path) -> str:
    source = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "prepare_driver_venv.sbatch"
    ).read_text()
    return source.split("    bash -lc '\n", maxsplit=1)[1].rsplit(
        "\n  '\n", maxsplit=1
    )[0]


def test_driver_venv_host_paths_override_container_image_defaults() -> None:
    project_root = Path(__file__).resolve().parents[3]
    source = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "prepare_driver_venv.sbatch"
    ).read_text()
    injection = (
        "  env \\\n"
        '    DRIVER_VENV="${DRIVER_VENV}" \\\n'
        '    UV_CACHE_DIR="${UV_CACHE_DIR}" \\\n'
        '    NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR}" \\\n'
        '    UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR}" \\\n'
        "    UV_MANAGED_PYTHON=1 \\\n"
        "    bash -lc '"
    )
    diagnostics = (
        'printf "Effective DRIVER_VENV=%s\\\\n" "${DRIVER_VENV}"',
        'printf "Effective UV_CACHE_DIR=%s\\\\n" "${UV_CACHE_DIR}"',
        'printf "Effective NEMO_RL_VENV_DIR=%s\\\\n" "${NEMO_RL_VENV_DIR}"',
        'printf "Effective UV_PYTHON_INSTALL_DIR=%s\\\\n" "${UV_PYTHON_INSTALL_DIR}"',
        'printf "Effective UV_MANAGED_PYTHON=%s\\\\n" "${UV_MANAGED_PYTHON}"',
    )

    assert 'UV_PYTHON_INSTALL_DIR="${NEMO_RL_VENV_DIR}/.uv-python"' in source
    assert injection in source
    assert "export UV_MANAGED_PYTHON" in source
    assert source.index("--container-workdir=") < source.index(injection)
    assert all(diagnostic in source for diagnostic in diagnostics)
    assert all(
        source.index(diagnostic) < source.index("uv sync --frozen")
        for diagnostic in diagnostics
    )


def _run_prepare_payload(
    tmp_path: Path,
    *,
    prefetch_reports_failure: bool,
    actor_import_exit: int,
    actor_python_outside_managed: bool = False,
    precreate_stale_actor: bool = False,
) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[3]
    prepare_source = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "prepare_driver_venv.sbatch"
    ).read_text()
    uv_managed_python_assignment = next(
        line.strip().removesuffix("\\").strip()
        for line in prepare_source.splitlines()
        if line.strip().startswith("UV_MANAGED_PYTHON=")
    )
    uv_managed_python = uv_managed_python_assignment.split("=", maxsplit=1)[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_command(
        fake_bin / "uv",
        """#!/bin/bash
if [[ "${UV_MANAGED_PYTHON:-}" != "1" ]]; then
  printf '%s\\n' 'UV_MANAGED_PYTHON=1 was not enforced' >&2
  exit 86
fi
if [[ ! -d "${UV_PYTHON_INSTALL_DIR:-}" ]]; then
  printf '%s\\n' 'UV_PYTHON_INSTALL_DIR was not initialized' >&2
  exit 87
fi
if find "${UV_PYTHON_INSTALL_DIR}" -mindepth 1 -print -quit | grep -q .; then
  printf '%s\\n' 'UV_PYTHON_INSTALL_DIR was not initially empty' >&2
  exit 88
fi
printf '%s\\n' 'managed-python-contract-ok'
""",
    )
    _write_fake_command(fake_bin / "sed", "#!/bin/sh\nexit 0\n")
    driver_venv = tmp_path / "driver-venv"
    nsight_patch_target = (
        driver_venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / "ray"
        / "_private"
        / "runtime_env"
        / "nsight.py"
    )
    nsight_patch_target.parent.mkdir(parents=True)
    nsight_patch_target.write_text(
        'context.py_executable = " ".join(self.nsight_cmd) + " python"\n'
    )
    driver_bin = driver_venv / "bin"
    driver_bin.mkdir()
    _write_fake_command(driver_bin / "ray", "#!/bin/sh\nexit 0\n")
    _write_fake_command(
        driver_bin / "python",
        """#!/bin/bash
if [[ "$1" == "-m" ]]; then
  actor_fqn=$3
  actor_python="${NEMO_RL_VENV_DIR}/${actor_fqn}/bin/python"
  if [[ "${ACTOR_PYTHON_OUTSIDE_MANAGED:-0}" == "1" ]]; then
    managed_python="${NEMO_RL_VENV_DIR}/unmanaged-python/${actor_fqn}/python"
  else
    managed_python="${UV_PYTHON_INSTALL_DIR}/cpython-3.13/bin/python3.13"
  fi
  mkdir -p "$(dirname -- "${managed_python}")"
  printf '%s\\n' '#!/bin/bash' 'exit "${ACTOR_IMPORT_EXIT:-0}"' > "${managed_python}"
  chmod +x "${managed_python}"
  mkdir -p "$(dirname -- "${actor_python}")"
  rm -f -- "${actor_python}"
  ln -s "${managed_python}" "${actor_python}"
  if [[ "${PREFETCH_REPORTS_FAILURE:-0}" == "1" ]]; then
    printf '%s\\n' '  Failed: 1'
  else
    printf '%s\\n' '  Failed: 0'
  fi
fi
exit 0
""",
    )
    actor_venv_dir = tmp_path / "actor-venvs"
    uv_python_install_dir = actor_venv_dir / ".uv-python"
    uv_python_install_dir.mkdir(parents=True)
    if precreate_stale_actor:
        stale_python = tmp_path / "container-root" / "bin" / "python3.13"
        stale_python.parent.mkdir(parents=True)
        _write_fake_command(stale_python, "#!/bin/sh\nexit 0\n")
        stale_actor_python = (
            actor_venv_dir
            / "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
            / "bin"
            / "python"
        )
        stale_actor_python.parent.mkdir(parents=True)
        stale_actor_python.symlink_to(stale_python)
    return subprocess.run(
        ["bash", "-c", _prepare_payload(project_root)],
        check=False,
        cwd=project_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "DRIVER_VENV": str(driver_venv),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
            "UV_PYTHON_INSTALL_DIR": str(uv_python_install_dir),
            "UV_MANAGED_PYTHON": uv_managed_python,
            "PREFETCH_REPORTS_FAILURE": "1" if prefetch_reports_failure else "0",
            "ACTOR_IMPORT_EXIT": str(actor_import_exit),
            "ACTOR_PYTHON_OUTSIDE_MANAGED": (
                "1" if actor_python_outside_managed else "0"
            ),
        },
        capture_output=True,
        text=True,
    )


def test_actor_prefetch_summary_failure_is_fatal(tmp_path: Path) -> None:
    result = _run_prepare_payload(
        tmp_path, prefetch_reports_failure=True, actor_import_exit=0
    )

    assert result.returncode == 2
    assert "Actor environment prefetch failed" in result.stderr


def test_actor_prefetch_forces_managed_python_from_an_empty_store(
    tmp_path: Path,
) -> None:
    result = _run_prepare_payload(
        tmp_path, prefetch_reports_failure=False, actor_import_exit=0
    )

    assert result.returncode == 0
    assert "managed-python-contract-ok" in result.stdout


def test_actor_prefetch_rejects_an_interpreter_without_required_imports(
    tmp_path: Path,
) -> None:
    result = _run_prepare_payload(
        tmp_path, prefetch_reports_failure=False, actor_import_exit=1
    )

    assert result.returncode == 2
    assert "Actor environment validation failed" in result.stderr


def test_actor_prefetch_rejects_a_new_unmanaged_python_target(
    tmp_path: Path,
) -> None:
    result = _run_prepare_payload(
        tmp_path,
        prefetch_reports_failure=False,
        actor_import_exit=0,
        actor_python_outside_managed=True,
    )

    assert result.returncode == 2
    assert "Actor environment uses unmanaged Python" in result.stderr


def test_actor_prefetch_rejects_a_stale_existing_python_target(
    tmp_path: Path,
) -> None:
    result = _run_prepare_payload(
        tmp_path,
        prefetch_reports_failure=False,
        actor_import_exit=0,
        precreate_stale_actor=True,
    )

    assert result.returncode == 2
    assert "Stale actor environment uses unmanaged Python" in result.stderr
    assert "Quarantine or remove the actor environment" in result.stderr


def test_launcher_rejects_an_invalid_uv_lock_timeout() -> None:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    ).read_text()

    assert '[[ ! "${UV_LOCK_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]' in launcher
    assert "UV_LOCK_TIMEOUT must be a positive integer number of seconds." in launcher


def test_deepep_setup_probe_uses_the_ray_runtime_python() -> None:
    project_root = Path(__file__).resolve().parents[3]
    renderer = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "render_deepep_setup_command.sh"
    )
    env = {
        **os.environ,
        "DEEPEP_OVERLAY": "/tmp/deepep-overlay",
        "DEEPEP_WHEEL": "/lustre/deep_ep.whl",
        "DEEPEP_WHEEL_SHA256": "a" * 64,
        "RAY_VENV": "/lustre/driver-venv",
    }

    result = subprocess.run(
        ["bash", str(renderer)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "runtime_python=/lustre/driver-venv/bin/python" in result.stdout
    assert (
        'PYTHONPATH="${overlay}" "${runtime_python}" -c '
        '"import importlib.metadata as md, os;'
    ) in result.stdout


def test_x86_wheel_build_job_is_arch_specific_and_reproducible() -> None:
    project_root = Path(__file__).resolve().parents[3]
    build_script = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "build_deepep_wheel.sbatch"
    )
    source = build_script.read_text()

    required_snippets = {
        ': "${CONTAINER:?CONTAINER is required}"',
        ': "${OUTPUT_DIR:?OUTPUT_DIR is required}"',
        ': "${GPU_ARCH:?GPU_ARCH is required}"',
        ': "${DEEPEP_COMMIT:?DEEPEP_COMMIT is required}"',
        ': "${HYBRID_EP_SCRIPT_PATH:?HYBRID_EP_SCRIPT_PATH is required}"',
        "LOCAL_SCRATCH_ROOT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}",
        'BUILD_ROOT=$(mktemp -d "${LOCAL_SCRATCH_ROOT}/nemo-rl-hybridep-${SLURM_JOB_ID}.XXXXXX")',
        '[[ "${DEEPEP_COMMIT}" =~ ^[0-9a-f]{40}$ ]]',
        "9.0 | 10.0",
        "export HYBRID_EP_MULTINODE=1",
        'export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"',
        'SCRIPT_PATH=$(readlink -f -- "${HYBRID_EP_SCRIPT_PATH}")',
        'srun --container-image="${CONTAINER}"',
        "--no-container-mount-home",
        'container_mounts="${container_mounts},${BUILD_ROOT}:${BUILD_ROOT}"',
        'rmdir -- "${BUILD_ROOT}"',
        "git clone --filter=blob:none --recurse-submodules",
        'git -C "${source_dir}" checkout --detach "${DEEPEP_COMMIT}"',
        'git -C "${source_dir}" submodule update --init --recursive',
        "uv build --wheel --no-build-isolation",
        "import deep_ep, deep_ep_cpp, hybrid_ep_cpp",
        'sha256sum "${staged_wheel}"',
        "container_sha256=",
        "SLURM_JOB_ID",
        'if [[ -e "${final_wheel}" ]]',
        'mv "${artifact_stage}" "${artifact_dir}"',
    }
    missing = sorted(snippet for snippet in required_snippets if snippet not in source)

    assert not missing


def test_x86_driver_venv_job_prepares_the_shared_ray_runtime() -> None:
    project_root = Path(__file__).resolve().parents[3]
    submit_script = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "submit_driver_venv.sh"
    ).read_text()
    prepare_script = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "prepare_driver_venv.sbatch"
    )
    source = prepare_script.read_text()

    required_snippets = {
        ': "${CONTAINER:?CONTAINER is required}"',
        ': "${DRIVER_VENV:?DRIVER_VENV is required}"',
        ': "${UV_CACHE_DIR:?UV_CACHE_DIR is required}"',
        ': "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR is required}"',
        "--no-container-mount-home",
        'UV_PROJECT_ENVIRONMENT="${DRIVER_VENV}"',
        "export NEMO_RL_VENV_DIR",
        "uv sync --frozen",
        "site-packages/ray/_private/runtime_env/nsight.py",
        "ray --version",
        'python -c "import ray; print(ray.__version__)"',
        "python -m nemo_rl.utils.prefetch_venvs",
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
        "Missing prefetched actor interpreter",
    }
    missing = sorted(snippet for snippet in required_snippets if snippet not in source)

    assert not missing
    assert "git pull --ff-only --recurse-submodules=no" in submit_script
    assert "git submodule update --init --recursive" in submit_script
    assert ': "${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR is required}"' in submit_script
    assert "TIME_LIMIT=${TIME_LIMIT:-02:00:00}" in submit_script
    assert 'sbatch --test-only "${sbatch_args[@]}"' in submit_script


def test_cw_h100_profile_pins_the_hopper_build_and_nvl8_topology() -> None:
    project_root = Path(__file__).resolve().parents[3]
    profile = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "clusters"
        / "cw-dfw-h100.env"
    ).read_text()

    required_lines = {
        "export CLUSTER_ID=cw-dfw-h100",
        "export ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}",
        "export PARTITION=${PARTITION:-batch}",
        "export GPU_ARCH=9.0",
        "export GPUS_PER_NODE=${GPUS_PER_NODE:-8}",
        "export NCCL_NVLS_ENABLE=0",
        "export NVLINK_DOMAIN_SIZE=8",
        "export USE_MNNVL=0",
        "export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8",
        "export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128",
    }

    assert required_lines <= set(profile.splitlines())


def test_cw_h100_profile_exports_values_to_slurm_jobs() -> None:
    project_root = Path(__file__).resolve().parents[3]
    profile = (
        project_root
        / "scripts"
        / "experiments"
        / "x86"
        / "hybridep"
        / "clusters"
        / "cw-dfw-h100.env"
    ).read_text()

    assignments = [
        line
        for line in profile.splitlines()
        if line and not line.startswith("#") and "=" in line
    ]

    assert assignments
    assert all(line.startswith("export ") for line in assignments)
