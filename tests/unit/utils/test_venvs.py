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

from nemo_rl.utils.venvs import (
    _prepare_uv_bootstrap_packages,
    _prepare_uv_install_env,
    _prepare_uv_environment_commands,
    _py_executable_requests_extra,
    create_local_venv,
    git_root,
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


def test_prepare_uv_environment_commands_scopes_install_to_requested_extra():
    build_cmd, install_cmd, exec_cmd = _prepare_uv_environment_commands(
        "uv run --locked --extra vllm --directory /tmp/repo",
        "/tmp/venv/bin/python",
    )

    assert build_cmd == [
        "uv",
        "pip",
        "install",
        "--python",
        "/tmp/venv/bin/python",
        "--project",
        "/tmp/repo",
        "--group",
        "build",
    ]
    assert install_cmd == [
        "uv",
        "pip",
        "install",
        "--python",
        "/tmp/venv/bin/python",
        "--project",
        "/tmp/repo",
        "--editable",
        "/tmp/repo[vllm]",
    ]
    assert exec_cmd == [
        "uv",
        "run",
        "--locked",
        "--extra",
        "vllm",
        "--directory",
        "/tmp/repo",
        "--no-sync",
    ]


def test_prepare_uv_environment_commands_defaults_to_repo_root():
    build_cmd, install_cmd, exec_cmd = _prepare_uv_environment_commands(
        "uv run --group docs",
        "/tmp/venv/bin/python",
    )

    assert build_cmd == [
        "uv",
        "pip",
        "install",
        "--python",
        "/tmp/venv/bin/python",
        "--project",
        git_root,
        "--group",
        "build",
    ]
    assert install_cmd == [
        "uv",
        "pip",
        "install",
        "--python",
        "/tmp/venv/bin/python",
        "--project",
        git_root,
        "--editable",
        git_root,
        "--group",
        "docs",
    ]
    assert exec_cmd == ["uv", "run", "--group", "docs", "--no-sync"]


def test_py_executable_requests_vllm_extra():
    assert _py_executable_requests_extra(
        "uv run --locked --extra vllm --directory /tmp/repo",
        "vllm",
    )
    assert not _py_executable_requests_extra(
        "uv run --locked --extra fsdp --directory /tmp/repo",
        "vllm",
    )


def test_prepare_uv_install_env_keeps_wheel_location_but_strips_stale_vllm_overrides():
    base_env = {
        "KEEP_ME": "1",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": "https://example.invalid/vllm.whl",
        "VLLM_USE_PRECOMPILED": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.14.0",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": "deadbeef",
        "VLLM_PRECOMPILED_WHEEL_VARIANT": "cuda",
    }

    install_env = _prepare_uv_install_env(
        base_env,
        "uv run --locked --extra vllm --directory /tmp/repo",
    )

    assert install_env == {
        "KEEP_ME": "1",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": "https://example.invalid/vllm.whl",
    }


def test_prepare_uv_install_env_keeps_non_vllm_overrides():
    base_env = {
        "KEEP_ME": "1",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": "https://example.invalid/vllm.whl",
    }

    install_env = _prepare_uv_install_env(
        base_env,
        "uv run --locked --extra fsdp --directory /tmp/repo",
    )

    assert install_env == base_env


def test_prepare_uv_bootstrap_packages_adds_vllm_build_tools():
    packages = _prepare_uv_bootstrap_packages(
        "uv run --locked --extra vllm --directory /tmp/repo"
    )

    assert packages == [
        "setuptools",
        "setuptools_scm",
        "torch==2.11.0",
        "cmake>=3.26.1",
        "ninja",
    ]


def test_prepare_uv_bootstrap_packages_keeps_base_packages_for_non_vllm():
    packages = _prepare_uv_bootstrap_packages(
        "uv run --locked --extra fsdp --directory /tmp/repo"
    )

    assert packages == [
        "setuptools",
        "setuptools_scm",
        "torch==2.11.0",
    ]
