import runpy
import sys
import types
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT / "3rdparty" / "Megatron-Bridge-workspace"
BRIDGE_SOURCE_ROOT = WORKSPACE_ROOT / "Megatron-Bridge" / "src"
BRIDGE_PACKAGE_ROOT = BRIDGE_SOURCE_ROOT / "megatron" / "bridge"
PACKAGE_INCLUDE = ["megatron.bridge", "megatron.bridge.*"]


def _source_namespace_packages() -> set[str]:
    package_directories = [BRIDGE_PACKAGE_ROOT, *BRIDGE_PACKAGE_ROOT.rglob("*")]
    return {
        ".".join(path.relative_to(BRIDGE_SOURCE_ROOT).parts)
        for path in package_directories
        if path.is_dir() and path.name != "__pycache__"
    }


def test_megatron_bridge_workspace_discovers_all_source_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_packages = _source_namespace_packages()
    setup_arguments: dict[str, Any] = {}
    setuptools_stub = types.ModuleType("setuptools")

    def find_namespace_packages(*, where: str, include: list[str]) -> list[str]:
        assert where == "Megatron-Bridge/src"
        assert include == PACKAGE_INCLUDE
        return sorted(expected_packages)

    def setup(**kwargs: Any) -> None:
        setup_arguments.update(kwargs)

    setuptools_stub.find_namespace_packages = find_namespace_packages  # type: ignore[attr-defined]
    setuptools_stub.setup = setup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "setuptools", setuptools_stub)
    monkeypatch.chdir(WORKSPACE_ROOT)

    runpy.run_path(str(WORKSPACE_ROOT / "setup.py"), run_name="__main__")

    assert set(setup_arguments["packages"]) == expected_packages
    assert setup_arguments["package_dir"] == {"": "Megatron-Bridge/src"}
    assert {
        "megatron.bridge",
        "megatron.bridge.diffusion",
        "megatron.bridge.models",
        "megatron.bridge.training",
    } <= expected_packages
