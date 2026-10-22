# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from nemo_rl.utils.venvs import (
    VENV_BUILD_ATTEMPTS,
    _run_venv_build_step,
    add_hf_modules_cache_to_pythonpath,
    create_local_venv,
    make_actor_runtime_env,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def test_create_local_venv():
    # The temporary directory is created within the project.
    # For some reason, creating a virtual environment outside of the project
    # doesn't work reliably.
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        # Mock os.environ to set NEMO_RL_VENV_DIR for this test
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_python = create_local_venv(
                py_executable="uv run --group docs", venv_name="test_venv"
            )
            assert os.path.exists(venv_python)
            assert venv_python == f"{tempdir}/test_venv/bin/python"
            # Check if sphinx package is installed in the created venv

            # Run a Python command to check if sphinx can be imported
            result = subprocess.run(
                [
                    venv_python,
                    "-c",
                    "import sphinx; print('Sphinx package is installed')",
                ],
                capture_output=True,
                text=True,
            )

            # Verify the command executed successfully (return code 0)
            assert result.returncode == 0, f"Failed to import sphinx: {result.stderr}"
            assert "Sphinx package is installed" in result.stdout


def test_venv_build_step_retries_a_transient_failure():
    failure = subprocess.CalledProcessError(1, ["uv", "run"])
    with (
        patch("nemo_rl.utils.venvs.subprocess.run", side_effect=[failure, None]) as run,
        patch("nemo_rl.utils.venvs.time.sleep") as sleep,
    ):
        _run_venv_build_step(["uv", "run"], {"A": "1"})

    assert run.call_count == 2
    run.assert_called_with(["uv", "run"], env={"A": "1"}, check=True)
    sleep.assert_called_once()


def test_venv_build_step_raises_after_last_attempt():
    failure = subprocess.CalledProcessError(1, ["uv", "run"])
    with (
        patch("nemo_rl.utils.venvs.subprocess.run", side_effect=failure) as run,
        patch("nemo_rl.utils.venvs.time.sleep") as sleep,
        pytest.raises(subprocess.CalledProcessError),
    ):
        _run_venv_build_step(["uv", "run"], {})

    assert run.call_count == VENV_BUILD_ATTEMPTS
    assert sleep.call_count == VENV_BUILD_ATTEMPTS - 1


def test_add_hf_modules_cache_to_pythonpath():
    result = add_hf_modules_cache_to_pythonpath(
        {
            "HF_MODULES_CACHE": "/hf/modules",
            "PYTHONPATH": f"/project{os.pathsep}/other",
        }
    )

    assert result["PYTHONPATH"].split(os.pathsep) == [
        "/hf/modules",
        "/project",
        "/other",
    ]


def test_add_hf_modules_cache_does_not_duplicate_pythonpath_entry():
    pythonpath = f"/project{os.pathsep}/hf/modules"

    result = add_hf_modules_cache_to_pythonpath(
        {"HF_MODULES_CACHE": "/hf/modules", "PYTHONPATH": pythonpath}
    )

    assert result["PYTHONPATH"] == pythonpath


def test_make_actor_runtime_env_builds_local_venv_for_uv_python_executable():
    """Mirrors the inline venv-creation logic that used to live in grpo.py."""
    with (
        patch(
            "nemo_rl.distributed.ray_actor_environment_registry.get_actor_python_env",
            return_value="uv run --group vllm",
        ) as mock_get_env,
        patch(
            "nemo_rl.utils.venvs.create_local_venv_on_each_node",
            return_value="/fake/venv/bin/python",
        ) as mock_create_venv,
    ):
        runtime_env = make_actor_runtime_env("some.module.SomeActor")

    mock_get_env.assert_called_once_with("some.module.SomeActor")
    mock_create_venv.assert_called_once_with(
        "uv run --group vllm", "some.module.SomeActor"
    )
    assert runtime_env["py_executable"] == "/fake/venv/bin/python"
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == "/fake/venv"
    assert runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] == "/fake/venv"
