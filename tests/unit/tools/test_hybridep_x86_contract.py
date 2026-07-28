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


def test_qwen_4n8g_hybridep_only_adds_x86_dispatcher_settings() -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_path = (
        project_root
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
        / "grpo-qwen3-30ba3b-4n8g-hybridep.yaml"
    )

    assert config_path.read_text() == (
        "defaults: grpo-qwen3-30ba3b-4n8g.yaml\n"
        "\n"
        "policy:\n"
        "  megatron_cfg:\n"
        "    moe_token_dispatcher_type: flex\n"
        "    moe_flex_dispatcher_backend: hybridep\n"
        "    moe_hybridep_num_sms: 32\n"
        "    env_vars:\n"
        '      NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: "8"\n'
        '      NUM_OF_TOKENS_PER_CHUNK_COMBINE_API: "128"\n'
        '      NVLINK_DOMAIN_SIZE: "8"\n'
        '      USE_MNNVL: "0"\n'
    )


def test_qwen_4n8g_x86_profiles_are_matched() -> None:
    project_root = Path(__file__).resolve().parents[3]
    profile_dir = (
        project_root / "scripts" / "experiments" / "oci-hsg" / "hybridep" / "models"
    )
    baseline = (profile_dir / "qwen3-30ba3b-4n8g-x86.env").read_text()
    hybridep = (profile_dir / "qwen3-30ba3b-4n8g-x86-hybridep.env").read_text()

    common_lines = {
        "export NCCL_NVLS_ENABLE=0",
        "NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}",
        "GPUS_PER_NODE=${GPUS_PER_NODE:-8}",
        "SEGMENT_SIZE=${SEGMENT_SIZE:-4}",
        "MAX_STEPS=${MAX_STEPS:-20}",
        "TIME_LIMIT=${TIME_LIMIT:-04:00:00}",
        f"DEFAULT_DEEPEP_COMMIT={DEEPEP_COMMIT}",
    }
    assert common_lines <= set(baseline.splitlines())
    assert common_lines <= set(hybridep.splitlines())
    assert (
        "CONFIG_PATH=examples/configs/recipes/llm/performance/"
        "grpo-qwen3-30ba3b-4n8g.yaml"
    ) in baseline
    assert (
        "CONFIG_PATH=examples/configs/recipes/llm/performance/"
        "grpo-qwen3-30ba3b-4n8g-hybridep.yaml"
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
    assert 'driver_args=(env "UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}"' in launcher
    assert 'PATH="${RAY_VENV}/bin:${PATH}"' in launcher
    assert "export PATH" in launcher
    assert "ray_venv=%q" in launcher


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
        "--no-container-mount-home",
        'UV_PROJECT_ENVIRONMENT="${DRIVER_VENV}"',
        "uv sync --frozen",
        "ray --version",
        'python -c "import ray; print(ray.__version__)"',
    }
    missing = sorted(snippet for snippet in required_snippets if snippet not in source)

    assert not missing
    assert "git pull --ff-only --recurse-submodules=no" in submit_script
    assert "git submodule update --init --recursive" in submit_script
    assert 'sbatch --test-only "${sbatch_args[@]}"' in submit_script
