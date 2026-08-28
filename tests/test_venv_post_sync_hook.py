from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENVS = ROOT / "nemo_rl/utils/venvs.py"


def _load_venvs(monkeypatch: pytest.MonkeyPatch):
    ray = ModuleType("ray")
    ray.remote = lambda **_kwargs: lambda function: function
    ray.util = SimpleNamespace(remove_placement_group=lambda _group: None)
    ray_util = ModuleType("ray.util")
    ray_util.placement_group = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)

    spec = importlib.util.spec_from_file_location("venvs_post_sync_test", VENVS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_sync_script_runs_with_new_venv_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venvs = _load_venvs(monkeypatch)
    venv_path = tmp_path / "venv"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    script = tmp_path / "patch.py"
    script.write_text("print('patched')", encoding="utf-8")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        venvs.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    env = {"NRL_VENV_POST_SYNC_SCRIPT": str(script)}
    venvs._run_venv_post_sync_script(venv_path, "worker", env)

    assert calls == [
        (([str(python_path), str(script)],), {"env": env, "check": True})
    ]


def test_post_sync_script_is_opt_in_and_target_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venvs = _load_venvs(monkeypatch)
    venv_path = tmp_path / "venv"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    script = tmp_path / "patch.py"
    script.write_text("print('patched')", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        venvs.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    venvs._run_venv_post_sync_script(venv_path, "worker", {})
    venvs._run_venv_post_sync_script(
        venv_path,
        "worker",
        {
            "NRL_VENV_POST_SYNC_SCRIPT": str(script),
            "NRL_VENV_POST_SYNC_TARGET": "other-worker",
        },
    )

    assert calls == []


def test_post_sync_script_fails_closed_when_script_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venvs = _load_venvs(monkeypatch)
    venv_path = tmp_path / "venv"
    (venv_path / "bin").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Venv post-sync script not found"):
        venvs._run_venv_post_sync_script(
            venv_path,
            "worker",
            {"NRL_VENV_POST_SYNC_SCRIPT": str(tmp_path / "missing.py")},
        )
