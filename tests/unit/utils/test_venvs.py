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
import multiprocessing
import os
import subprocess
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

import nemo_rl.utils.venvs as venvs
from nemo_rl.utils.venvs import create_local_venv
from tests.unit.conftest import TEST_ASSETS_DIR


def _hold_source_build_lock(
    lock_path: str,
    acquired_event: Event,
    release_event: Event,
) -> None:
    os.environ["NRL_VENV_BUILD_LOCK"] = lock_path
    os.environ["NRL_VENV_BUILD_LOCK_TIMEOUT"] = "5"
    with venvs._source_build_lock("uv run --locked --extra mcore"):
        acquired_event.set()
        release_event.wait(timeout=5)


def _raise_while_holding_source_build_lock(
    lock_path: str,
    acquired_event: Event,
    keep_alive_event: Event,
    result_connection: Connection,
) -> None:
    os.environ["NRL_VENV_BUILD_LOCK"] = lock_path
    os.environ["NRL_VENV_BUILD_LOCK_TIMEOUT"] = "5"
    try:
        with venvs._source_build_lock("uv run --locked --extra=mcore"):
            acquired_event.set()
            raise RuntimeError("test build failure")
    except RuntimeError as error:
        result_connection.send((type(error).__name__, str(error)))
        keep_alive_event.wait(timeout=10)
    finally:
        result_connection.close()


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


@pytest.mark.parametrize(
    "py_executable",
    [
        "uv run --locked --extra mcore",
        "uv run --locked --extra=mcore",
    ],
)
def test_source_build_lock_recognizes_mcore_extra_forms(tmp_path, py_executable):
    lock_path = tmp_path / "mcore-build.lock"

    with patch.dict(os.environ, {"NRL_VENV_BUILD_LOCK": str(lock_path)}):
        with venvs._source_build_lock(py_executable):
            pass

    assert lock_path.exists()


def test_source_build_lock_is_disabled_for_other_extras(tmp_path):
    lock_path = tmp_path / "mcore-build.lock"

    with (
        patch.dict(os.environ, {"NRL_VENV_BUILD_LOCK": str(lock_path)}),
        venvs._source_build_lock("uv run --locked --extra vllm"),
    ):
        pass

    assert not lock_path.exists()


def test_source_build_lock_requires_absolute_shared_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK", "mcore-build.lock")

    with pytest.raises(
        ValueError, match="absolute path.*shared filesystem.*POSIX flock"
    ):
        with venvs._source_build_lock("uv run --locked --extra mcore"):
            pass


@pytest.mark.parametrize("timeout_value", ["0", "-1", "nan", "inf", "invalid"])
def test_source_build_lock_requires_positive_finite_timeout(
    tmp_path, monkeypatch, timeout_value
):
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK", str(tmp_path / "mcore-build.lock"))
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK_TIMEOUT", timeout_value)

    with pytest.raises(
        ValueError, match="NRL_VENV_BUILD_LOCK_TIMEOUT.*positive finite number"
    ):
        with venvs._source_build_lock("uv run --locked --extra mcore"):
            pass


def test_source_build_lock_times_out_under_multiprocess_contention(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "mcore-build.lock"
    process_context = multiprocessing.get_context("fork")
    acquired_event = process_context.Event()
    release_event = process_context.Event()
    lock_holder = process_context.Process(
        target=_hold_source_build_lock,
        args=(str(lock_path), acquired_event, release_event),
    )
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK", str(lock_path))
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK_TIMEOUT", "0.2")

    try:
        lock_holder.start()
        assert acquired_event.wait(timeout=5)
        with pytest.raises(RuntimeError) as error_info:
            with venvs._source_build_lock("uv run --locked --extra=mcore"):
                pass

        error_message = str(error_info.value)
        assert str(lock_path) in error_message
        assert "0.2" in error_message
        assert "NRL_VENV_BUILD_LOCK_TIMEOUT" in error_message
    finally:
        release_event.set()
        lock_holder.join(timeout=5)
        if lock_holder.is_alive():
            lock_holder.terminate()
            lock_holder.join(timeout=3)

    assert lock_holder.exitcode == 0


def test_source_build_lock_releases_after_exception_for_another_process(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "mcore-build.lock"
    process_context = multiprocessing.get_context("fork")
    acquired_event = process_context.Event()
    keep_alive_event = process_context.Event()
    result_connection, child_result_connection = process_context.Pipe(duplex=False)
    failing_builder = process_context.Process(
        target=_raise_while_holding_source_build_lock,
        args=(
            str(lock_path),
            acquired_event,
            keep_alive_event,
            child_result_connection,
        ),
    )
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK", str(lock_path))
    monkeypatch.setenv("NRL_VENV_BUILD_LOCK_TIMEOUT", "1")

    try:
        failing_builder.start()
        child_result_connection.close()
        assert acquired_event.wait(timeout=5)
        assert result_connection.poll(timeout=5)
        assert result_connection.recv() == ("RuntimeError", "test build failure")
        assert failing_builder.is_alive()

        with venvs._source_build_lock("uv run --locked --extra mcore"):
            pass
    finally:
        keep_alive_event.set()
        failing_builder.join(timeout=5)
        if failing_builder.is_alive():
            failing_builder.terminate()
            failing_builder.join(timeout=3)
        result_connection.close()

    assert failing_builder.exitcode == 0
