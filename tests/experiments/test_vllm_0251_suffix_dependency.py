from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vllm_suffix_decoding_requires_only_arctic_inference() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    optional_dependencies = project["project"]["optional-dependencies"]
    vllm_dependencies = optional_dependencies["vllm"]

    assert [
        dependency
        for dependency in vllm_dependencies
        if dependency.startswith("arctic-inference")
    ] == ["arctic-inference==0.1.1"]
    assert all(
        not any(dependency.startswith("arctic-inference") for dependency in dependencies)
        for extra, dependencies in optional_dependencies.items()
        if extra != "vllm"
    )
