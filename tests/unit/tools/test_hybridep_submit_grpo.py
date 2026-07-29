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
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


ACTOR_FQNS = (
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
)
DEEPEP_COMMIT = "f725d29699f5bda9ba789456bb9579af69844685"


def _run_launcher(
    tmp_path: Path,
    *,
    dispatcher_mode: str,
    model_config_name: str = "qwen3-30ba3b-4n4g.env",
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    result, command_capture = _run_launcher_result(
        tmp_path,
        dispatcher_mode=dispatcher_mode,
        model_config_name=model_config_name,
        extra_env=extra_env,
    )
    result.check_returncode()
    return shlex.split(command_capture.read_text())


def _run_launcher_result(
    tmp_path: Path,
    *,
    dispatcher_mode: str,
    model_config_name: str = "qwen3-30ba3b-4n4g.env",
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
    model_config = launcher.parent / "models" / model_config_name
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
printf '%s\\n' "${NEMO_RL_VENV_DIR:-}" > "$VENV_DIR_CAPTURE"
printf '%s\\n' '999999'
"""
    )
    fake_sbatch.chmod(0o755)

    container = tmp_path / "nightly.sqsh"
    container.touch()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PROJECT_ROOT": str(project_root),
        "FAKE_USER": subprocess.check_output(["id", "-un"], text=True).strip(),
        "COMMAND_CAPTURE": str(command_capture),
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


def _x86_actor_venv_dir(tmp_path: Path) -> Path:
    return tmp_path / "actor-venvs"


@pytest.fixture
def lustre_tmp_path() -> Iterator[Path]:
    lustre_root = Path("/lustre")
    if not lustre_root.is_dir() or not os.access(lustre_root, os.W_OK):
        pytest.skip("requires a writable /lustre mount")
    path = Path(tempfile.mkdtemp(prefix="nemo-rl-test-", dir=lustre_root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_actor_python(actor_venv_dir: Path, actor_fqn: str) -> None:
    python_path = actor_venv_dir / actor_fqn / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)


def _write_prefetched_actor_pythons(actor_venv_dir: Path) -> None:
    for actor_fqn in ACTOR_FQNS:
        _write_actor_python(actor_venv_dir, actor_fqn)


def _synthetic_x86_shared_env() -> dict[str, str]:
    return {
        "DEEPEP_COMMIT": DEEPEP_COMMIT,
        "DEEPEP_WHEEL": "/lustre/nemo-rl-test/deep-ep.whl",
        "HF_HOME": "/lustre/nemo-rl-test/hf-home",
        "HF_DATASETS_CACHE": "/lustre/nemo-rl-test/hf-home/cache",
        "RUN_ROOT": "/lustre/nemo-rl-test/run",
    }


def _x86_shared_env(shared_root: Path) -> dict[str, str]:
    wheel = shared_root / "deep-ep.whl"
    wheel.touch()
    return {
        "DEEPEP_COMMIT": DEEPEP_COMMIT,
        "DEEPEP_WHEEL": str(wheel),
        "HF_HOME": str(shared_root / "hf-home"),
        "HF_DATASETS_CACHE": str(shared_root / "hf-home" / "cache"),
        "RUN_ROOT": str(shared_root / "run"),
    }


def test_recipe_dispatcher_preserves_recipe_default(tmp_path: Path) -> None:
    driver_args = _run_launcher(tmp_path, dispatcher_mode="recipe")

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" not in driver_args
    assert (
        "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" not in driver_args
    )
    assert "++policy.megatron_cfg.moe_hybridep_num_sms=32" not in driver_args


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


@pytest.mark.parametrize(
    "path_variable",
    (
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
    if path_variable == "DEEPEP_WHEEL":
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
