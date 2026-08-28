# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path
import subprocess
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "cw-dfw"
    / "hybridep"
    / "build_deepep_wheel_from_checkout.sbatch"
)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _source_checkout(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    (path / "README.md").write_text("fixture\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _wheel(path: Path, *, include_hybridep: bool) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("deep_ep_cpp.cpython-313-x86_64-linux-gnu.so", b"")
        if include_hybridep:
            archive.writestr("hybrid_ep_cpp.cpython-313-x86_64-linux-gnu.so", b"")


def _run_builder(tmp_path: Path, *, include_hybridep: bool) -> subprocess.CompletedProcess[str]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build_root = scratch / "build"
    source_dir = build_root / "source"
    commit = _source_checkout(source_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fixture_wheel = tmp_path / "deep_ep-1.2.1+fixture-cp313-cp313-linux_x86_64.whl"
    _wheel(fixture_wheel, include_hybridep=include_hybridep)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uv",
        """#!/bin/bash
set -euo pipefail
if [[ "$1" == "build" ]]; then
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--out-dir" ]]; then
      cp "${FAKE_WHEEL}" "$2/"
      exit 0
    fi
    shift
  done
fi
if [[ "$1" == "pip" && "$2" == "install" ]]; then
  exit 0
fi
exit 2
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/bin/bash
if [[ "$*" == *"get_device_capability"* ]]; then
  printf '9.0\n'
fi
if [[ "$*" == *"import deep_ep_cpp"* && "$*" != *"import torch;"* ]]; then
  printf 'extension probe did not preload torch\n' >&2
  exit 9
fi
exit 0
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DEEPEP_BUILD_STAGE": "container",
            "SOURCE_DIR": "/home/test/unmounted-deepep-source",
            "BUILD_SOURCE_DIR": str(source_dir),
            "SOURCE_ROOT_PREFIX": "/home",
            "OUTPUT_DIR": str(output_dir),
            "OUTPUT_ROOT_PREFIX": str(tmp_path),
            "BUILD_ROOT": str(build_root),
            "SLURM_TMPDIR": str(scratch),
            "SLURM_JOB_ID": "123",
            "DEEPEP_COMMIT": commit,
            "DEEPEP_EXPECTED_VERSION": "1.2.1+fixture",
            "GPU_ARCH": "9.0",
            "PYTHON_BIN": str(fake_bin / "python"),
            "FAKE_WHEEL": str(fixture_wheel),
        }
    )
    if not BUILDER.exists():
        return subprocess.CompletedProcess([], 127, "", "builder is missing")
    return subprocess.run(
        ["bash", str(BUILDER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_builder_does_not_publish_wheel_without_hybridep_extension(tmp_path: Path) -> None:
    result = _run_builder(tmp_path, include_hybridep=False)

    assert result.returncode != 0
    assert "missing hybrid_ep_cpp extension" in result.stderr
    assert not list((tmp_path / "output").glob("hybridep-*"))


def test_builder_publishes_only_after_both_extensions_pass_probe(tmp_path: Path) -> None:
    result = _run_builder(tmp_path, include_hybridep=True)

    assert result.returncode == 0, result.stderr
    artifacts = list((tmp_path / "output").glob("hybridep-*"))
    assert len(artifacts) == 1
    assert len(list(artifacts[0].glob("*.whl"))) == 1
    assert len(list(artifacts[0].glob("*.whl.sha256"))) == 1
    assert (artifacts[0] / "metadata.env").is_file()
