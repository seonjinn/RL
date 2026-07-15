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
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nemo_rl.utils.venvs import _run_venv_post_sync_script, create_local_venv
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


def test_run_venv_post_sync_script_uses_venv_python(tmp_path: Path) -> None:
    venv_path = tmp_path / "venv"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    script_path = tmp_path / "patch.py"
    script_path.touch()
    env = {
        "NRL_VENV_POST_SYNC_SCRIPT": str(script_path),
        "NRL_VENV_POST_SYNC_TARGET": "vllm-worker",
    }

    with patch("nemo_rl.utils.venvs.subprocess.run") as run:
        _run_venv_post_sync_script(venv_path, "vllm-worker", env)

    run.assert_called_once_with(
        [str(python_path), str(script_path)], env=env, check=True
    )


def test_run_venv_post_sync_script_is_disabled_by_default(tmp_path: Path) -> None:
    with patch("nemo_rl.utils.venvs.subprocess.run") as run:
        _run_venv_post_sync_script(tmp_path / "venv", "vllm-worker", {})

    run.assert_not_called()


def test_run_venv_post_sync_script_skips_other_venvs(tmp_path: Path) -> None:
    script_path = tmp_path / "patch.py"
    script_path.touch()
    env = {
        "NRL_VENV_POST_SYNC_SCRIPT": str(script_path),
        "NRL_VENV_POST_SYNC_TARGET": "vllm-worker",
    }

    with patch("nemo_rl.utils.venvs.subprocess.run") as run:
        _run_venv_post_sync_script(tmp_path / "venv", "mcore-worker", env)

    run.assert_not_called()
