import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from tomlkit import parse

from tools.configure_custom_vllm import configure_custom_vllm_pyproject


def _pyproject_with_vllm_requirements(requirements: list[str]) -> str:
    rendered_requirements = ",\n".join(f'  "{requirement}"' for requirement in requirements)
    return dedent(
        f"""\
        [project]
        name = "example"

        [project.optional-dependencies]
        vllm = [
        {rendered_requirements}
        ]

        [tool.uv.sources]
        vllm = {{ git = "https://example.invalid/old-vllm.git", rev = "main" }}

        [tool.uv]
        no-build-isolation-package = ["torch", "vllm", "vllm"]
        """
    )


def _vllm_requirements(source: str) -> list[str]:
    document = parse(source)
    requirements = document["project"]["optional-dependencies"]["vllm"]
    return [
        str(requirement)
        for requirement in requirements
        if canonicalize_name(Requirement(str(requirement)).name) == "vllm"
    ]


@pytest.mark.parametrize(
    "existing_requirement",
    [
        "vllm",
        "vLLM==0.20.0",
        "vllm>=0.19,<0.21",
        "vllm @ https://example.invalid/vllm.whl",
        "vllm==0.20.0 ; sys_platform == 'linux'",
    ],
    ids=["bare", "pinned", "specifier", "direct-url", "marker"],
)
def test_configure_replaces_every_vllm_requirement_with_one_bare_requirement(
    existing_requirement: str,
) -> None:
    source = _pyproject_with_vllm_requirements(
        ["cuda-python", existing_requirement, "num2words"]
    )

    configured = configure_custom_vllm_pyproject(source)

    assert _vllm_requirements(configured) == ["vllm"]
    document = parse(configured)
    assert list(document["project"]["optional-dependencies"]["vllm"]) == [
        "cuda-python",
        "num2words",
        "vllm",
    ]


def test_configure_removes_duplicate_vllm_requirements_only() -> None:
    source = _pyproject_with_vllm_requirements(
        [
            "vllm",
            "vLLM==0.20.0",
            "vllm @ https://example.invalid/vllm.whl",
            "vllm>=0.20 ; sys_platform == 'linux'",
            "my-vllm-helper==1.0",
            "notvllm==2.0",
        ]
    )

    configured = configure_custom_vllm_pyproject(source)

    document = parse(configured)
    assert list(document["project"]["optional-dependencies"]["vllm"]) == [
        "my-vllm-helper==1.0",
        "notvllm==2.0",
        "vllm",
    ]


def test_configure_sets_one_editable_source_and_no_build_isolation_entry() -> None:
    source = _pyproject_with_vllm_requirements(["vllm==0.20.0"])

    configured = configure_custom_vllm_pyproject(
        source, source_path="vendor/custom-vllm"
    )

    document = parse(configured)
    assert dict(document["tool"]["uv"]["sources"]["vllm"]) == {
        "path": "vendor/custom-vllm",
        "editable": True,
    }
    assert list(document["tool"]["uv"]["no-build-isolation-package"]).count("vllm") == 1


def test_configure_creates_missing_uv_source_and_no_build_isolation_tables() -> None:
    source = dedent(
        """\
        [project]
        name = "example"

        [project.optional-dependencies]
        vllm = ["vllm==0.20.0"]
        """
    )

    configured = configure_custom_vllm_pyproject(source)

    document = parse(configured)
    assert dict(document["tool"]["uv"]["sources"]["vllm"]) == {
        "path": "3rdparty/vllm",
        "editable": True,
    }
    assert list(document["tool"]["uv"]["no-build-isolation-package"]) == ["vllm"]


def test_configure_is_byte_idempotent() -> None:
    source = _pyproject_with_vllm_requirements(
        ["vllm==0.20.0", "my-vllm-helper==1.0"]
    )

    configured = configure_custom_vllm_pyproject(source)

    assert configure_custom_vllm_pyproject(configured) == configured


def test_cli_updates_pyproject_in_place(tmp_path: Path) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        _pyproject_with_vllm_requirements(["vllm==0.20.0"]), encoding="utf-8"
    )
    script_path = Path(__file__).parents[3] / "tools" / "configure_custom_vllm.py"

    subprocess.run(
        [sys.executable, str(script_path), str(pyproject_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    configured = pyproject_path.read_text(encoding="utf-8")
    assert _vllm_requirements(configured) == ["vllm"]
    document = parse(configured)
    assert dict(document["tool"]["uv"]["sources"]["vllm"]) == {
        "path": "3rdparty/vllm",
        "editable": True,
    }
