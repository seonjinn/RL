"""Tests for optional vLLM source patches."""

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_patches_module() -> ModuleType:
    repository_root = Path(__file__).resolve().parents[4]
    module_path = repository_root / "nemo_rl/models/generation/vllm/patches.py"
    spec = importlib.util.spec_from_file_location(
        "_standalone_vllm_patches", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vLLM patches module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_obsolete_additional_env_vars_patch_is_noop_without_assignment(
    monkeypatch, tmp_path: Path
) -> None:
    patches = _load_patches_module()
    ray_executor = tmp_path / "ray_executor.py"
    source = (
        "class RayDistributedExecutor:\n"
        "    def initialize(self):\n"
        '        self._init_workers_ray(placement_group, runtime_env={"py_executable": "/usr/bin/python"})\n'
    )
    ray_executor.write_text(source)
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative_path: str(ray_executor)
    )

    patches._patch_vllm_init_workers_ray(
        "/usr/bin/python",
        extra_env_vars=["VLLM_MXFP8_DENSE_CONFIG_FILE"],
    )

    assert ray_executor.read_text() == source
