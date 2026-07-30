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

import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest


ACTOR_FQNS = (
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
)
DEEPEP_COMMIT = "f725d29699f5bda9ba789456bb9579af69844685"
STANDARD_DEEPEP_COMMIT = "dd758caf451848bd150e1046af3d0a73e5fff38d"
NCCL_VERSION = "2.30.4"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PERFORMANCE_RECIPE_DIR = (
    PROJECT_ROOT / "examples" / "configs" / "recipes" / "llm" / "performance"
)
RECIPE_TOPOLOGY_RESOLVER = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "oci-hsg"
    / "hybridep"
    / "resolve_recipe_topology.py"
)


@pytest.mark.parametrize(
    "recipe_name",
    (
        "grpo-qwen3-30ba3b-4n8g-alltoall.yaml",
        "grpo-qwen3-30ba3b-4n8g.yaml",
    ),
)
def test_qwen30_performance_recipe_resolves_native_topology(
    recipe_name: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RECIPE_TOPOLOGY_RESOLVER),
            str(PERFORMANCE_RECIPE_DIR / recipe_name),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nodes, gpus, config_segment, resolved_sha = result.stdout.strip().split("\t")

    assert (nodes, gpus, config_segment) == ("4", "8", "null")
    assert len(resolved_sha) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_nodes", "0"),
        ("gpus_per_node", "true"),
        ("segment_size", "-1"),
    ),
)
def test_recipe_topology_rejects_nonpositive_or_boolean_values(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    topology = {
        "num_nodes": "4",
        "gpus_per_node": "8",
        "segment_size": "null",
    }
    topology[field] = value
    recipe = tmp_path / "invalid-topology.yaml"
    recipe.write_text(
        "\n".join(
            (
                "cluster:",
                f"  num_nodes: {topology['num_nodes']}",
                f"  gpus_per_node: {topology['gpus_per_node']}",
                f"  segment_size: {topology['segment_size']}",
                "",
            )
        )
    )

    result = subprocess.run(
        [sys.executable, str(RECIPE_TOPOLOGY_RESOLVER), str(recipe)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"cluster.{field} must be a positive integer" in result.stderr


def _run_launcher(
    tmp_path: Path,
    *,
    dispatcher_mode: str,
    model_config_name: str = "qwen3-30ba3b-4n4g.env",
    model_config_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    result, command_capture = _run_launcher_result(
        tmp_path,
        dispatcher_mode=dispatcher_mode,
        model_config_name=model_config_name,
        model_config_path=model_config_path,
        extra_env=extra_env,
    )
    result.check_returncode()
    return shlex.split(command_capture.read_text())


def _run_launcher_result(
    tmp_path: Path,
    *,
    dispatcher_mode: str,
    model_config_name: str = "qwen3-30ba3b-4n4g.env",
    model_config_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    )
    model_config = model_config_path or launcher.parent / "models" / model_config_name
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/sh
case "$*" in
  *"rev-parse --show-toplevel"*) printf '%s\\n' "$FAKE_PROJECT_ROOT" ;;
  *"rev-parse HEAD"*) printf '%s\\n' "1342aa09a0e0903f4299390e5d9fc1d88f3b75eb" ;;
esac
"""
    )
    fake_git.chmod(0o755)

    fake_sshare = fake_bin / "sshare"
    fake_sshare.write_text(
        """#!/bin/sh
printf '%s|%s|%s|\\n' 'nemotron_sw_pre' "$FAKE_USER" '0.900000'
"""
    )
    fake_sshare.chmod(0o755)

    command_capture = tmp_path / "command.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/bin/sh
if [ "$1" = "--test-only" ]; then
  printf '%s\\n' 'test-only accepted'
  exit 0
fi
printf '%s\\n' "$COMMAND" > "$COMMAND_CAPTURE"
printf '%s\\n' "${SETUP_COMMAND:-}" > "$SETUP_COMMAND_CAPTURE"
printf '%s\\n' "${LD_LIBRARY_PATH:-}" > "$RAY_PARENT_LD_LIBRARY_PATH_CAPTURE"
printf '%s\\n' "${NEMO_RL_VENV_DIR:-}" > "$VENV_DIR_CAPTURE"
printf '%s\\n' '999999'
"""
    )
    fake_sbatch.chmod(0o755)

    container = tmp_path / "nightly.sqsh"
    container.touch()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "TOPOLOGY_PYTHON": sys.executable,
        "FAKE_PROJECT_ROOT": str(project_root),
        "FAKE_USER": subprocess.check_output(["id", "-un"], text=True).strip(),
        "COMMAND_CAPTURE": str(command_capture),
        "SETUP_COMMAND_CAPTURE": str(tmp_path / "setup-command.txt"),
        "RAY_PARENT_LD_LIBRARY_PATH_CAPTURE": str(
            tmp_path / "ray-parent-ld-library-path.txt"
        ),
        "VENV_DIR_CAPTURE": str(tmp_path / "venv-dir.txt"),
        "CONTAINER": str(container),
        "RUN_ROOT": str(tmp_path / "run"),
        "RUN_SUFFIX": "test",
        "DISPATCHER_MODE": dispatcher_mode,
    }
    env.pop("NEMO_RL_VENV_DIR", None)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(launcher), str(model_config)],
        check=False,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, command_capture


def _write_qwen30_native_topology_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "qwen3-30ba3b-native-topology.env"
    profile.write_text(
        "\n".join(
            (
                "MODEL_ID=qwen3-30ba3b-native-topology",
                (
                    "CONFIG_PATH=examples/configs/recipes/llm/performance/"
                    "grpo-qwen3-30ba3b-4n8g-alltoall.yaml"
                ),
                "NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}",
                "GPUS_PER_NODE=${GPUS_PER_NODE:-8}",
                "SEGMENT_SIZE=${SEGMENT_SIZE:-4}",
                "REQUIRE_RECIPE_TOPOLOGY_MATCH=true",
                "MAX_STEPS=3",
                "TIME_LIMIT=04:00:00",
                f"DEFAULT_DEEPEP_COMMIT={DEEPEP_COMMIT}",
                "",
            )
        )
    )
    return profile


def test_generic_profile_does_not_require_topology_python(tmp_path: Path) -> None:
    reject_bin = tmp_path / "reject-bin"
    reject_bin.mkdir()
    reject_python = reject_bin / "python3"
    reject_python.write_text("#!/bin/sh\nexit 88\n")
    reject_python.chmod(0o755)

    args = _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        extra_env={
            "PATH": f"{tmp_path / 'bin'}:{reject_bin}:{os.environ['PATH']}",
            "TOPOLOGY_PYTHON": "",
        },
    )

    assert "examples/run_grpo.py" in args


def test_topology_matched_profile_requires_a_managed_python(tmp_path: Path) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_path=_write_qwen30_native_topology_profile(tmp_path),
        extra_env={"TOPOLOGY_PYTHON": ""},
    )

    assert result.returncode == 2
    assert (
        "DRIVER_VENV or TOPOLOGY_PYTHON is required for recipe topology validation"
        in result.stderr
    )


def test_launcher_preserves_recipe_topology(tmp_path: Path) -> None:
    args = _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_path=_write_qwen30_native_topology_profile(tmp_path),
    )

    assert not any(arg.startswith("cluster.") for arg in args)


def test_submission_records_recipe_and_scheduler_topology(tmp_path: Path) -> None:
    _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_path=_write_qwen30_native_topology_profile(tmp_path),
    )

    metadata = (tmp_path / "run" / "submission.env").read_text()
    assert "config_num_nodes=4\n" in metadata
    assert "config_gpus_per_node=8\n" in metadata
    assert "config_segment_size=null\n" in metadata
    assert "scheduler_segment_size=4\n" in metadata
    resolved_hash = next(
        line.removeprefix("resolved_config_sha256=")
        for line in metadata.splitlines()
        if line.startswith("resolved_config_sha256=")
    )
    assert len(resolved_hash) == 64


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"NUM_ACTOR_NODES": "2"}, "scheduler nodes 2 != recipe nodes 4"),
        (
            {"GPUS_PER_NODE": "4"},
            "scheduler GPUs per node 4 != recipe GPUs per node 8",
        ),
    ),
)
def test_qwen30_launcher_rejects_scheduler_topology_mismatch(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_path=_write_qwen30_native_topology_profile(tmp_path),
        extra_env=override,
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("NUM_ACTOR_NODES", "0"),
        ("GPUS_PER_NODE", "false"),
        ("SEGMENT_SIZE", "-1"),
    ),
)
def test_launcher_rejects_invalid_scheduler_topology(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_path=_write_qwen30_native_topology_profile(tmp_path),
        extra_env={field: value},
    )

    assert result.returncode == 2
    assert f"{field} must be a positive integer: {value}" in result.stderr


def _x86_actor_venv_dir(tmp_path: Path) -> Path:
    return tmp_path / "actor-venvs"


def _resolve_lustre_test_root(configured_root: str | None) -> Path:
    lustre_root = Path(configured_root or "/lustre").resolve()
    try:
        lustre_root.relative_to("/lustre")
    except ValueError as error:
        raise ValueError(
            "NEMO_RL_TEST_LUSTRE_ROOT must resolve under /lustre: "
            f"{lustre_root}"
        ) from error
    return lustre_root


def test_lustre_test_root_accepts_a_configured_lustre_subtree() -> None:
    assert _resolve_lustre_test_root("/lustre/project/users/tester") == Path(
        "/lustre/project/users/tester"
    )


def test_lustre_test_root_rejects_a_path_outside_lustre() -> None:
    with pytest.raises(ValueError, match="must resolve under /lustre"):
        _resolve_lustre_test_root("/lustre/../tmp/nemo-rl-test")


@pytest.fixture
def lustre_tmp_path() -> Iterator[Path]:
    try:
        lustre_root = _resolve_lustre_test_root(
            os.environ.get("NEMO_RL_TEST_LUSTRE_ROOT")
        )
    except ValueError as error:
        pytest.fail(str(error))
    if not lustre_root.is_dir() or not os.access(lustre_root, os.W_OK):
        pytest.skip(f"requires a writable /lustre test root: {lustre_root}")
    path = Path(tempfile.mkdtemp(prefix="nemo-rl-test-", dir=lustre_root))
    try:
        yield path
    finally:
        if path.parent != lustre_root:
            pytest.fail(f"refusing to clean a path outside the Lustre test root: {path}")
        shutil.rmtree(path)


def _write_actor_python(actor_venv_dir: Path, actor_fqn: str) -> None:
    actor_root = actor_venv_dir / actor_fqn
    python_path = actor_root / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)
    nvidia_library_dir = (
        actor_root / "lib" / "python3.13" / "site-packages" / "nvidia" / "cudnn" / "lib"
    )
    nvidia_library_dir.mkdir(parents=True)
    (nvidia_library_dir / "libcudnn.so.9").touch()


def _write_prefetched_actor_pythons(actor_venv_dir: Path) -> None:
    for actor_fqn in ACTOR_FQNS:
        _write_actor_python(actor_venv_dir, actor_fqn)


def _synthetic_x86_shared_env() -> dict[str, str]:
    return {
        "CONTAINER": "/lustre/nemo-rl-test/nightly.sqsh",
        "DEEPEP_COMMIT": DEEPEP_COMMIT,
        "DEEPEP_VARIANT": "hybridep",
        "DEEPEP_WHEEL": "/lustre/nemo-rl-test/deep-ep.whl",
        "HF_HOME": "/lustre/nemo-rl-test/hf-home",
        "HF_DATASETS_CACHE": "/lustre/nemo-rl-test/hf-home/cache",
        "NCCL_WHEEL": "/lustre/nemo-rl-test/nvidia-nccl-cu13.whl",
        "RUN_ROOT": "/lustre/nemo-rl-test/run",
        "UV_CACHE_DIR": "/lustre/nemo-rl-test/uv-cache",
    }


def _x86_shared_env(shared_root: Path) -> dict[str, str]:
    container = shared_root / "nightly.sqsh"
    container.touch()
    wheel = shared_root / "deep-ep.whl"
    wheel.touch()
    return {
        "CONTAINER": str(container),
        "DEEPEP_COMMIT": DEEPEP_COMMIT,
        "DEEPEP_WHEEL": str(wheel),
        "HF_HOME": str(shared_root / "hf-home"),
        "HF_DATASETS_CACHE": str(shared_root / "hf-home" / "cache"),
        "RUN_ROOT": str(shared_root / "run"),
        "UV_CACHE_DIR": str(shared_root / "uv-cache"),
    }


def _write_nccl_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "nvidia_nccl_cu13-2.30.4.dist-info/METADATA",
            "Name: nvidia-nccl-cu13\nVersion: 2.30.4\n",
        )


def _write_metadata(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def test_standard_artifact_helpers_write_separate_metadata_lines(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.env"
    _write_metadata(
        metadata_path,
        {"package": "nvidia-nccl-cu13", "version": NCCL_VERSION},
    )
    assert metadata_path.read_text().splitlines() == [
        "package=nvidia-nccl-cu13",
        f"version={NCCL_VERSION}",
    ]

    wheel_path = tmp_path / "nvidia_nccl_cu13.whl"
    _write_nccl_wheel(wheel_path)
    with zipfile.ZipFile(wheel_path) as archive:
        metadata = archive.read(
            "nvidia_nccl_cu13-2.30.4.dist-info/METADATA"
        ).decode()
    assert metadata.splitlines() == [
        "Name: nvidia-nccl-cu13",
        "Version: 2.30.4",
    ]


def standard_deepep_artifact_env(shared_root: Path) -> dict[str, str]:
    deepep_dir = shared_root / "deepep-artifact"
    nccl_dir = shared_root / "nccl-artifact"
    deepep_dir.mkdir()
    nccl_dir.mkdir()
    deepep_wheel = deepep_dir / "deep_ep-standard.whl"
    deepep_wheel.write_bytes(b"standard-deepep-wheel")
    nccl_wheel = nccl_dir / "nvidia_nccl_cu13-2.30.4.whl"
    _write_nccl_wheel(nccl_wheel)
    deepep_sha256 = subprocess.check_output(
        ["sha256sum", str(deepep_wheel)], text=True
    ).split()[0]
    nccl_sha256 = subprocess.check_output(
        ["sha256sum", str(nccl_wheel)], text=True
    ).split()[0]
    _write_metadata(
        deepep_dir / "metadata.env",
        {
            "deepep_variant": "deepep",
            "deepep_branch": "main",
            "deepep_commit": STANDARD_DEEPEP_COMMIT,
            "deepep_wheel_sha256": deepep_sha256,
            "nccl_version": NCCL_VERSION,
            "nccl_wheel_sha256": nccl_sha256,
            "wheel": str(deepep_wheel),
            "wheel_sha256": deepep_sha256,
        },
    )
    _write_metadata(
        nccl_dir / "metadata.env",
        {
            "package": "nvidia-nccl-cu13",
            "version": NCCL_VERSION,
            "wheel": str(nccl_wheel),
            "wheel_sha256": nccl_sha256,
        },
    )
    container = shared_root / "nightly.sqsh"
    container.touch()
    return {
        "CONTAINER": str(container),
        "DEEPEP_COMMIT": STANDARD_DEEPEP_COMMIT,
        "DEEPEP_WHEEL": str(deepep_wheel),
        "DEEPEP_VARIANT": "deepep",
        "HF_HOME": str(shared_root / "hf-home"),
        "HF_DATASETS_CACHE": str(shared_root / "hf-home" / "cache"),
        "NCCL_WHEEL": str(nccl_wheel),
        "RUN_ROOT": str(shared_root / "run"),
        "UV_CACHE_DIR": str(shared_root / "uv-cache"),
    }


def test_recipe_dispatcher_preserves_recipe_default(tmp_path: Path) -> None:
    driver_args = _run_launcher(tmp_path, dispatcher_mode="recipe")

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" not in driver_args
    assert (
        "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" not in driver_args
    )
    assert "++policy.megatron_cfg.moe_hybridep_num_sms=32" not in driver_args


def test_deepep_dispatcher_applies_standard_backend(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    args = _run_launcher(
        tmp_path,
        dispatcher_mode="deepep",
        extra_env=standard_deepep_artifact_env(lustre_tmp_path),
    )

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" in args
    assert "++policy.megatron_cfg.moe_flex_dispatcher_backend=deepep" in args
    assert "++policy.megatron_cfg.moe_deepep_num_sms=20" in args
    assert not any("moe_hybridep_num_sms" in arg for arg in args)


def test_deepep_dispatcher_rejects_an_invalid_variant(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="deepep",
        extra_env={
            **standard_deepep_artifact_env(lustre_tmp_path),
            "DEEPEP_VARIANT": "invalid",
        },
    )

    assert result.returncode == 2
    assert "DEEPEP_VARIANT must be deepep or hybridep" in result.stderr


def test_deepep_dispatcher_rejects_branch_variant_mismatch(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    metadata_path = Path(env["DEEPEP_WHEEL"]).parent / "metadata.env"
    _write_metadata(
        metadata_path,
        {
            **dict(
                line.split("=", maxsplit=1)
                for line in metadata_path.read_text().splitlines()
            ),
            "deepep_branch": "hybrid-ep",
        },
    )
    result, _ = _run_launcher_result(
        tmp_path, dispatcher_mode="deepep", extra_env=env
    )

    assert result.returncode == 2
    assert "DeepEP artifact branch does not match variant deepep" in result.stderr


def test_deepep_dispatcher_requires_the_nccl_wheel(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    env.pop("NCCL_WHEEL")
    result, _ = _run_launcher_result(
        tmp_path, dispatcher_mode="deepep", extra_env=env
    )

    assert result.returncode == 2
    assert "NCCL_WHEEL is required for model profile" in result.stderr


def test_deepep_dispatcher_rejects_the_wrong_nccl_version(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    metadata_path = Path(env["NCCL_WHEEL"]).parent / "metadata.env"
    _write_metadata(
        metadata_path,
        {
            **dict(
                line.split("=", maxsplit=1)
                for line in metadata_path.read_text().splitlines()
            ),
            "version": "2.30.5",
        },
    )
    result, _ = _run_launcher_result(
        tmp_path, dispatcher_mode="deepep", extra_env=env
    )

    assert result.returncode == 2
    assert "NCCL artifact version must be 2.30.4" in result.stderr


def test_deepep_dispatcher_rejects_the_wrong_nccl_checksum(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    metadata_path = Path(env["NCCL_WHEEL"]).parent / "metadata.env"
    _write_metadata(
        metadata_path,
        {
            **dict(
                line.split("=", maxsplit=1)
                for line in metadata_path.read_text().splitlines()
            ),
            "wheel_sha256": "0" * 64,
        },
    )
    result, _ = _run_launcher_result(
        tmp_path, dispatcher_mode="deepep", extra_env=env
    )

    assert result.returncode == 2
    assert "NCCL artifact wheel checksum mismatch" in result.stderr


def test_deepep_dispatcher_rejects_mismatched_deepep_metadata(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    metadata_path = Path(env["DEEPEP_WHEEL"]).parent / "metadata.env"
    _write_metadata(
        metadata_path,
        {
            **dict(
                line.split("=", maxsplit=1)
                for line in metadata_path.read_text().splitlines()
            ),
            "deepep_commit": DEEPEP_COMMIT,
        },
    )
    result, _ = _run_launcher_result(tmp_path, dispatcher_mode="deepep", extra_env=env)

    assert result.returncode == 2
    assert "DeepEP artifact commit does not match DEEPEP_COMMIT" in result.stderr


def test_deepep_dispatcher_wires_both_wheels_and_ray_parent_loader_path(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    env = standard_deepep_artifact_env(lustre_tmp_path)
    _run_launcher(tmp_path, dispatcher_mode="deepep", extra_env=env)

    overlay = "/tmp/nemo-rl-deepep-dd758caf4518-test"
    setup_command = (tmp_path / "setup-command.txt").read_text()
    ray_parent_loader_path = (tmp_path / "ray-parent-ld-library-path.txt").read_text()
    metadata = (Path(env["RUN_ROOT"]) / "submission.env").read_text()

    assert "deepep_variant=deepep" in setup_command
    assert f"nccl_wheel={env['NCCL_WHEEL']}" in setup_command
    assert "expected_nccl_wheel_sha256=" in setup_command
    assert ray_parent_loader_path.startswith(f"{overlay}/nvidia/nccl/lib:")
    assert f"deepep_variant=deepep\n" in metadata
    assert f"deepep_wheel={env['DEEPEP_WHEEL']}\n" in metadata
    assert f"nccl_wheel={env['NCCL_WHEEL']}\n" in metadata
    assert f"nccl_version={NCCL_VERSION}\n" in metadata
    project_root = Path(__file__).resolve().parents[3]
    expected_config_sha256 = subprocess.check_output(
        [
            "sha256sum",
            str(
                project_root
                / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
            ),
        ],
        text=True,
    ).split()[0]
    assert f"config_sha256={expected_config_sha256}\n" in metadata


def test_unknown_dispatcher_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run_launcher(tmp_path, dispatcher_mode="unknown")


def test_deepseek_profile_requires_a_checkpoint_path(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    actor_venv_dir = lustre_tmp_path / "actor-venvs"
    _write_prefetched_actor_pythons(actor_venv_dir)

    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="deepseek-v3-32n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
        },
    )

    assert result.returncode == 2
    assert "NRL_DEEPSEEK_V3_BF16_CKPT must be set for model profile" in result.stderr


def test_deepseek_profile_applies_a_verified_checkpoint_to_model_and_tokenizer(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    checkpoint = lustre_tmp_path / "deepseek-v3-bf16"
    checkpoint.mkdir()
    actor_venv_dir = lustre_tmp_path / "actor-venvs"
    _write_prefetched_actor_pythons(actor_venv_dir)
    shared_env = _x86_shared_env(lustre_tmp_path)

    driver_args = _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="deepseek-v3-32n8g-x86.env",
        extra_env={
            **shared_env,
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
            "NRL_DEEPSEEK_V3_BF16_CKPT": str(checkpoint),
        },
    )

    assert f"policy.model_name={checkpoint}" in driver_args
    assert f"policy.tokenizer.name={checkpoint}" in driver_args

    metadata = (Path(shared_env["RUN_ROOT"]) / "submission.env").read_text()
    assert f"model_name_override={checkpoint}\n" in metadata
    assert f"tokenizer_name_override={checkpoint}\n" in metadata


def test_x86_profile_requires_a_prefetched_actor_venv_directory(tmp_path: Path) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
        },
    )

    assert result.returncode == 2
    assert "NEMO_RL_VENV_DIR is required for model profile" in result.stderr


def test_x86_profile_requires_an_explicit_deepep_wheel(tmp_path: Path) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            "HF_HOME": "/lustre/nemo-rl-test/hf-home",
            "HF_DATASETS_CACHE": "/lustre/nemo-rl-test/hf-home/cache",
            "RUN_ROOT": "/lustre/nemo-rl-test/run",
        },
    )

    assert result.returncode == 2
    assert "DEEPEP_WHEEL is required for model profile" in result.stderr


def test_x86_profile_requires_an_explicit_uv_cache_directory(tmp_path: Path) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
            "UV_CACHE_DIR": "",
        },
    )

    assert result.returncode == 2
    assert "UV_CACHE_DIR is required for model profile" in result.stderr


@pytest.mark.parametrize(
    "path_variable",
    (
        "CONTAINER",
        "DRIVER_VENV",
        "RAY_VENV",
        "UV_CACHE_DIR",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "RUN_ROOT",
        "DEEPEP_WHEEL",
    ),
)
def test_x86_profile_rejects_shared_paths_that_traverse_outside_lustre(
    tmp_path: Path, path_variable: str
) -> None:
    escaped_path = tmp_path / path_variable.lower()
    if path_variable in {"CONTAINER", "DEEPEP_WHEEL"}:
        escaped_path.touch()
    traversal_path = Path("/lustre") / ".." / escaped_path.relative_to("/")
    extra_env = {
        **_synthetic_x86_shared_env(),
        "DRIVER_VENV": "/lustre/driver-venv",
        "RAY_VENV": "/lustre/driver-venv",
        "UV_CACHE_DIR": "/lustre/uv-cache",
    }
    extra_env[path_variable] = str(traversal_path)

    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env=extra_env,
    )

    assert result.returncode == 2
    assert f"{path_variable} must resolve under shared /lustre storage" in result.stderr


def test_x86_profile_rejects_a_home_backed_container(tmp_path: Path) -> None:
    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
            "CONTAINER": "/home/sna/nemo_rl_nightly.sqsh",
        },
    )

    assert result.returncode == 2
    assert "CONTAINER must resolve under shared /lustre storage" in result.stderr


def test_deepseek_profile_rejects_a_checkpoint_that_traverses_outside_lustre(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "deepseek-v3-bf16"
    checkpoint.mkdir()
    traversal_path = Path("/lustre") / ".." / checkpoint.relative_to("/")

    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="deepseek-v3-32n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
            "NRL_DEEPSEEK_V3_BF16_CKPT": str(traversal_path),
        },
    )

    assert result.returncode == 2
    assert (
        "NRL_DEEPSEEK_V3_BF16_CKPT must resolve under shared /lustre storage"
        in result.stderr
    )


def test_x86_profile_rejects_an_incomplete_prefetched_actor_venv_directory(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    actor_venv_dir = lustre_tmp_path / "actor-venvs"
    _write_actor_python(
        actor_venv_dir,
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    )

    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **_x86_shared_env(lustre_tmp_path),
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
        },
    )

    assert result.returncode == 2
    assert "Missing prebuilt actor interpreter" in result.stderr
    assert "MegatronPolicyWorker/bin/python" in result.stderr


def test_x86_profile_exports_prefetched_actor_venv_directory_to_ray(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    actor_venv_dir = lustre_tmp_path / "actor-venvs"
    _write_prefetched_actor_pythons(actor_venv_dir)
    shared_env = _x86_shared_env(lustre_tmp_path)

    _run_launcher(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **shared_env,
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
        },
    )

    assert (tmp_path / "venv-dir.txt").read_text().strip() == str(actor_venv_dir)
    metadata = (Path(shared_env["RUN_ROOT"]) / "submission.env").read_text()
    assert f"nemo_rl_venv_dir={actor_venv_dir}\n" in metadata
    assert "prebuilt_actor_venvs_required=true\n" in metadata


def test_x86_profile_prepends_actor_libraries_and_preserves_container_paths(
    tmp_path: Path, lustre_tmp_path: Path
) -> None:
    actor_venv_dir = lustre_tmp_path / "actor-venvs"
    _write_prefetched_actor_pythons(actor_venv_dir)
    shared_env = _x86_shared_env(lustre_tmp_path)

    result, command_capture = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **shared_env,
            "NEMO_RL_VENV_DIR": str(actor_venv_dir),
            "LD_LIBRARY_PATH": "/container/cuda/lib",
        },
    )

    result.check_returncode()
    policy_library_dir = (
        actor_venv_dir
        / ACTOR_FQNS[1]
        / "lib"
        / "python3.13"
        / "site-packages"
        / "nvidia"
        / "cudnn"
        / "lib"
    )
    command = command_capture.read_text()
    assert command.startswith(f"export LD_LIBRARY_PATH={policy_library_dir}:")
    assert "${LD_LIBRARY_PATH:-};" in command

    metadata = (Path(shared_env["RUN_ROOT"]) / "submission.env").read_text()
    assert f"prebuilt_actor_library_path={policy_library_dir}\n" in metadata


def test_x86_profile_rejects_a_lustre_path_that_traverses_outside_lustre(
    tmp_path: Path,
) -> None:
    actor_venv_dir = _x86_actor_venv_dir(tmp_path)
    _write_prefetched_actor_pythons(actor_venv_dir)
    traversal_path = Path("/lustre") / ".." / actor_venv_dir.relative_to("/")

    result, _ = _run_launcher_result(
        tmp_path,
        dispatcher_mode="recipe",
        model_config_name="qwen3-30ba3b-4n8g-x86.env",
        extra_env={
            **_synthetic_x86_shared_env(),
            "NEMO_RL_VENV_DIR": str(traversal_path),
        },
    )

    assert result.returncode == 2
    assert "NEMO_RL_VENV_DIR must be on shared /lustre storage" in result.stderr


def test_generic_profile_does_not_require_prefetched_actor_venvs(
    tmp_path: Path,
) -> None:
    _run_launcher(tmp_path, dispatcher_mode="recipe")

    metadata = (tmp_path / "run" / "submission.env").read_text()
    assert "prebuilt_actor_venvs_required=false\n" in metadata


def test_padding_logging_reaches_megatron_worker_environment(tmp_path: Path) -> None:
    driver_args = _run_launcher(
        tmp_path,
        dispatcher_mode="hybridep",
        extra_env={
            "NEMO_RL_HYBRIDEP_LOG_PACKING": "1",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS": "4096",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS": "0",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE": "1",
        },
    )

    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING='1'" in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS='4096'"
        in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS='0'"
        in driver_args
    )
    assert (
        "++policy.megatron_cfg.env_vars.NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE='1'"
        in driver_args
    )


def test_submission_metadata_records_dispatcher_and_padding_logging(
    tmp_path: Path,
) -> None:
    _run_launcher(
        tmp_path,
        dispatcher_mode="hybridep",
        extra_env={
            "NEMO_RL_HYBRIDEP_LOG_PACKING": "1",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS": "4096",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS": "0",
            "NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE": "1",
        },
    )

    metadata = (tmp_path / "run" / "submission.env").read_text()
    assert "dispatcher_mode=hybridep\n" in metadata
    assert "padding_log_enabled=1\n" in metadata
    assert "padding_log_max_calls=4096\n" in metadata
    assert "padding_log_ranks=0\n" in metadata
    assert "padding_log_reduce=1\n" in metadata


def test_hybridep_launcher_preserves_project_and_bridge_pythonpath() -> None:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    )
    source = launcher.read_text()

    project_path = 'PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"'
    bridge_path = 'PYTHONPATH="${BRIDGE_SRC}:${PYTHONPATH}"'
    overlay_path = 'PYTHONPATH="${DEEPEP_OVERLAY}:${PYTHONPATH}"'
    export_path = "export PYTHONPATH"

    assert project_path in source
    assert bridge_path in source
    assert overlay_path in source
    assert export_path in source
    assert source.index(project_path) < source.index(bridge_path)
    assert source.index(bridge_path) < source.index(overlay_path)
    assert source.index(overlay_path) < source.index(export_path)
