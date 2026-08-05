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
from unittest.mock import call, patch

from nemo_rl.utils.venvs import create_local_venv, git_root
from tests.unit.conftest import TEST_ASSETS_DIR


def test_create_local_venv_uses_configured_uv_binary(tmp_path) -> None:
    create_local_venv.cache_clear()
    uv_bin = "/opt/tools/uv"
    with (
        patch.dict(
            os.environ,
            {"NEMO_RL_VENV_DIR": str(tmp_path), "UV_BIN": uv_bin},
        ),
        patch("nemo_rl.utils.venvs.subprocess.run") as run,
    ):
        venv_python = create_local_venv(
            py_executable="uv run --locked --extra vllm",
            venv_name="worker",
        )
        expected_env = os.environ.copy()

    venv_path = tmp_path / "worker"
    expected_env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    assert run.call_args_list == [
        call([uv_bin, "venv", "--allow-existing", str(venv_path)], check=True),
        call(
            [uv_bin, "sync", "--directory", git_root],
            env=expected_env,
            check=True,
        ),
        call(
            [
                uv_bin,
                "run",
                "--locked",
                "--extra",
                "vllm",
                "echo",
                f"Finished creating venv {venv_path}",
            ],
            env=expected_env,
            check=True,
        ),
    ]
    assert venv_python == f"{venv_path}/bin/python"
    create_local_venv.cache_clear()


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
