import multiprocessing
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from nemo_rl.utils import venvs


def test_create_local_venv_rebuilds_a_deleted_environment(
    tmp_path: Path, monkeypatch
) -> None:
    build_paths: list[Path] = []

    def fake_run(command: list[str], **_: object) -> None:
        if command[:2] == ["uv", "venv"]:
            venv_path = Path(command[-1])
            build_paths.append(venv_path)
            python_path = venv_path / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.touch()

    cache_clear = getattr(venvs.create_local_venv, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    monkeypatch.setattr(venvs.subprocess, "run", fake_run)

    with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}):
        first_path = venvs.create_local_venv("uv run --extra vllm", "shared-vllm")
        shutil.rmtree(tmp_path / "shared-vllm")
        second_path = venvs.create_local_venv("uv run --extra vllm", "shared-vllm")

    assert first_path == second_path
    assert Path(second_path).exists()
    assert build_paths == [tmp_path / "shared-vllm", tmp_path / "shared-vllm"]


def test_actor_venv_build_is_serialized_across_processes(
    tmp_path: Path, monkeypatch
) -> None:
    context = multiprocessing.get_context("fork")
    build_count = context.Value("i", 0)

    def fake_create_local_venv(
        py_executable: str, venv_name: str, force_rebuild: bool = False
    ) -> str:
        del py_executable, force_rebuild
        with build_count.get_lock():
            build_count.value += 1
        venv_path = tmp_path / venv_name
        python_path = venv_path / "bin" / "python"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(0.2)
        python_path.touch()
        venvs._venv_ready_file(venv_path).touch()
        return str(python_path)

    monkeypatch.setattr(venvs, "create_local_venv", fake_create_local_venv)
    monkeypatch.setattr(
        venvs, "_venv_python_is_usable", lambda python_path: python_path.exists()
    )

    with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": str(tmp_path)}):
        results = context.Queue()

        def run_build() -> None:
            results.put(
                venvs._build_or_reuse_actor_venv("uv run --extra vllm", "shared-vllm")
            )

        processes = [context.Process(target=run_build) for _ in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        paths = [results.get(timeout=1) for _ in processes]

    assert build_count.value == 1
    assert paths == [
        str(tmp_path / "shared-vllm" / "bin" / "python"),
        str(tmp_path / "shared-vllm" / "bin" / "python"),
    ]
