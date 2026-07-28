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

from nemo_rl.utils import venvs
from nemo_rl.utils.venvs import create_local_venv
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


def test_create_local_venv_syncs_the_frozen_lock_before_running_the_extra() -> None:
    """The worker bootstrap must not resolve or rewrite the shared lockfile."""
    venv_name = "frozen_sync_contract"
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        venvs.create_local_venv.cache_clear()
        try:
            with (
                patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}),
                patch("nemo_rl.utils.venvs.subprocess.run") as run,
            ):
                create_local_venv(
                    py_executable="uv run --frozen --extra mcore",
                    venv_name=venv_name,
                )
        finally:
            venvs.create_local_venv.cache_clear()

    sync_call = next(
        call for call in run.call_args_list if call.args[0][:2] == ["uv", "sync"]
    )
    assert sync_call.args[0] == [
        "uv",
        "sync",
        "--frozen",
        "--directory",
        venvs.git_root,
    ]
    assert sync_call.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == os.path.join(
        tempdir, venv_name
    )
