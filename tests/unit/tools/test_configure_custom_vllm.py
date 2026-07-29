import os
import shutil
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


@pytest.mark.parametrize(
    ("project_selector", "virtual_selector", "expected_root_selector"),
    [
        pytest.param(None, None, "repo-default", id="selectors-unset"),
        pytest.param("project", None, "project", id="uv-only"),
        pytest.param(None, "virtual", "virtual", id="virtual-only"),
        pytest.param("shared", "shared", "shared", id="matching-selectors"),
        pytest.param("project", "virtual", None, id="conflicting-selectors"),
    ],
)
def test_build_script_selects_root_interpreter_and_isolates_vllm_venv(
    tmp_path: Path,
    project_selector: str | None,
    virtual_selector: str | None,
    expected_root_selector: str | None,
) -> None:
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True)
    source_script = Path(__file__).parents[3] / "tools" / "build-custom-vllm.sh"
    script_path = tools_dir / source_script.name
    shutil.copy2(source_script, script_path)
    (repo_root / "3rdparty").mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """\
#!/bin/bash
set -eu
case "$1" in
  clone)
    mkdir -p "${@: -1}"
    ;;
  checkout)
    ;;
  rev-parse)
    printf '%s\\n' "$TEST_GIT_REF"
    ;;
  *)
    printf 'unexpected git command: %s\\n' "$*" >&2
    exit 1
    ;;
esac
"""
    )
    fake_git.chmod(0o755)

    uv_log = tmp_path / "uv.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """\
#!/bin/bash
set -eu
printf '%s\\t%s\\t%s\\t%s\\n' \
  "$PWD" \
  "${UV_PROJECT_ENVIRONMENT-<unset>}" \
  "${VIRTUAL_ENV-<unset>}" \
  "$*" >>"$TEST_UV_LOG"
if [[ "$1" == "venv" ]]; then
  venv_path="${2:-$PWD/.venv}"
  mkdir -p "$venv_path/bin"
  touch "$venv_path/bin/python"
fi
"""
    )
    fake_uv.chmod(0o755)
    fake_realpath = fake_bin / "realpath"
    fake_realpath.write_text(
        """\
#!/bin/bash
set -eu
target="$1"
parent="$(dirname "$target")"
name="$(basename "$target")"
printf '%s/%s\\n' "$(cd "$parent" && pwd -P)" "$name"
"""
    )
    fake_realpath.chmod(0o755)

    git_ref = "a" * 40
    root_environments = {
        "repo-default": repo_root / ".venv",
        "project": tmp_path / "root-project-environment",
        "virtual": tmp_path / "root-virtual-environment",
        "shared": tmp_path / "root-shared-environment",
    }
    for root_environment in root_environments.values():
        root_python = root_environment / "bin" / "python"
        root_python.parent.mkdir(parents=True)
        root_python.touch()
        root_python.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "TEST_GIT_REF": git_ref,
        "TEST_UV_LOG": str(uv_log),
    }
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)
    if project_selector is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(root_environments[project_selector])
    if virtual_selector is not None:
        env["VIRTUAL_ENV"] = str(root_environments[virtual_selector])

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "https://example.invalid/vllm.git",
            git_ref,
            "https://example.invalid/vllm.whl",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if expected_root_selector is None:
        assert completed.returncode == 2
        assert "select different Python environments" in completed.stderr
        assert not (repo_root / "3rdparty" / "vllm").exists()
        return

    assert completed.returncode == 0, completed.stderr

    calls = [line.split("\t", maxsplit=3) for line in uv_log.read_text().splitlines()]
    vllm_root = repo_root / "3rdparty" / "vllm"
    vllm_venv = vllm_root / ".venv"
    venv_call = next(call for call in calls if call[3].startswith("venv"))
    assert venv_call == [
        str(vllm_root),
        "<unset>",
        "<unset>",
        f"venv {vllm_venv}",
    ]

    custom_pip_calls = [
        call
        for call in calls
        if call[0] == str(vllm_root) and call[3].startswith("pip install")
    ]
    assert len(custom_pip_calls) == 3
    for _, uv_project_environment, virtual_environment, arguments in custom_pip_calls:
        assert uv_project_environment == "<unset>"
        assert virtual_environment == "<unset>"
        assert f"--python {vllm_venv / 'bin' / 'python'}" in arguments

    root_pip_call = next(
        call
        for call in calls
        if call[0] == str(repo_root) and call[3].startswith("pip install")
    )
    assert root_pip_call[1:] == [
        (
            str(root_environments[project_selector])
            if project_selector is not None
            else "<unset>"
        ),
        (
            str(root_environments[virtual_selector])
            if virtual_selector is not None
            else "<unset>"
        ),
        (
            "pip install --python "
            f"{root_environments[expected_root_selector] / 'bin' / 'python'} "
            "setuptools_scm"
        ),
    ]
