import hashlib
import os
import subprocess
import time
from pathlib import Path


EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)
SCRIPT_PATH = EXPERIMENT_DIR / "scripts" / "build_te_pr2898_backport.sub"
TE_SOURCE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/"
    "TransformerEngine-fp64-thd-cudagraph-20260730"
)
TE_COMMIT = "ba256c5b23c8f19b64a0c26499277d15c4133a1c"
IMAGE = "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/containers/nemo_rl_nightly_20260729_2472184.sqsh"
IMAGE_SHA256 = "cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91"


def _prepare_launcher_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "nemo-rl-cg"
    source = root / "src" / "TransformerEngine-fp64-thd-cudagraph-20260730"
    source.mkdir(parents=True)
    (source / "README.md").write_text("fixture\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Launcher Test",
            "-c",
            "user.email=launcher-test@example.com",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    image = root / "containers" / "nemo_rl_nightly_20260729_2472184.sqsh"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture image")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()

    launcher = tmp_path / "build_te_pr2898_backport.sub"
    script = SCRIPT_PATH.read_text()
    replacements = {
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg": str(root),
        TE_COMMIT: commit,
        IMAGE_SHA256: image_sha256,
    }
    for old, new in replacements.items():
        script = script.replace(old, new)
    launcher.write_text(script)
    launcher.chmod(0o755)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import shlex
import sys
import time

command = sys.argv[-1]
tokens = shlex.split(command)
dist_index = tokens.index("--dist-dir")
wheel_dir = Path(tokens[dist_index + 1])
staging_dir = wheel_dir.parent
mode = os.environ["FAKE_SRUN_MODE"]

if mode == "fail":
    raise SystemExit(23)

if mode == "require-isolation":
    expected_sequence = [
        "setup.py",
        "build",
        "--build-base",
        str(staging_dir / "setuptools-build"),
        "bdist_wheel",
        "--bdist-dir",
        str(staging_dir / "bdist"),
        "--dist-dir",
        str(wheel_dir),
    ]
    start = tokens.index("setup.py")
    if tokens[start : start + len(expected_sequence)] != expected_sequence:
        raise SystemExit(31)
    expected_cmake = f"export NVTE_CMAKE_BUILD_DIR={staging_dir / 'cmake'}"
    if expected_cmake not in command:
        raise SystemExit(32)
    for build_directory in ("cmake", "setuptools-build", "bdist"):
        path = staging_dir / build_directory
        path.mkdir(parents=True)
        (path / "build-intermediate").write_text("fixture")

if mode == "concurrent":
    state = Path(os.environ["FAKE_SRUN_STATE"])
    state.mkdir(exist_ok=True)
    (state / f"entered-{os.getpid()}").touch()
    while not (state / "release").exists():
        time.sleep(0.01)

wheel_dir.mkdir(parents=True, exist_ok=True)
(wheel_dir / "transformer_engine_fixture.whl").write_bytes(b"fixture wheel")
"""
    )
    fake_srun.chmod(0o755)

    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
no_target_directory = False
while args and args[0].startswith("-"):
    option = args.pop(0)
    if option == "-T":
        no_target_directory = True
    elif option != "--":
        raise SystemExit(64)
source, destination = map(Path, args)
if destination.exists():
    if no_target_directory:
        raise SystemExit(1)
    destination = destination / source.name
os.rename(source, destination)
"""
    )
    fake_mv.chmod(0o755)
    return launcher, root, fake_bin


def _launcher_environment(
    fake_bin: Path,
    *,
    mode: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_SRUN_MODE": mode,
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    return environment


def test_te_pr2898_build_launcher_pins_host_and_container_provenance() -> None:
    script = SCRIPT_PATH.read_text()

    assert f"TE_SOURCE={TE_SOURCE}" in script
    assert f"EXPECTED_TE_COMMIT={TE_COMMIT}" in script
    assert f"IMAGE={IMAGE}" in script
    assert f"IMAGE_SHA256={IMAGE_SHA256}" in script
    assert (
        'test "$(sha256sum -- "${IMAGE}" | awk \'{print $1}\')" = "${IMAGE_SHA256}"'
        in script
    )
    assert (
        'test "$(git -C "${TE_SOURCE}" rev-parse HEAD)" = "${EXPECTED_TE_COMMIT}"'
        in script
    )
    assert 'git -C "${TE_SOURCE}" diff --quiet' in script
    assert 'git -C "${TE_SOURCE}" diff --cached --quiet' in script
    assert 'git -C "${TE_SOURCE}" status --porcelain --untracked-files=all' in script
    assert 'git -C "${TE_SOURCE}" submodule status --recursive' in script
    assert script.index("sha256sum --") < script.index("srun --nodes=1 --ntasks=1")
    assert script.index("rev-parse HEAD") < script.index("srun --nodes=1 --ntasks=1")


def test_te_pr2898_build_launcher_is_offline_bounded_and_atomically_publishes_one_wheel() -> (
    None
):
    script = SCRIPT_PATH.read_text()

    assert "NVTE_FRAMEWORK=pytorch" in script
    assert "NVTE_CUDA_ARCHS=100" in script
    assert "TE_BUILD_JOBS=${TE_BUILD_JOBS:-8}" in script
    assert "TE_BUILD_JOBS must be an integer from 1 through 16" in script
    assert "MAX_JOBS=${TE_BUILD_JOBS}" in script
    assert "PIP_NO_INDEX=1" in script
    assert "PIP_NO_BUILD_ISOLATION=1" in script
    assert "/opt/nemo_rl_venv/bin/python setup.py \\" in script
    assert "pip install" not in script
    assert "uv sync" not in script
    assert "--checkpoint" not in script
    assert "Expected exactly one wheel" in script
    assert "wheel_sha256" in script
    assert "provenance.json" in script
    assert 'mv -T -- "${PUBLISH_STAGING}" "${OUTPUT_DIR}"' in script
    assert "transformer-engine-pr2898-${EXPECTED_TE_COMMIT}" in script


def test_te_pr2898_build_stays_out_of_performance_launchers_and_is_documented() -> None:
    runner = (EXPERIMENT_DIR / "run_scope.sh").read_text()
    performance_submitter = (EXPERIMENT_DIR / "submit_performance.sh").read_text()
    readme = (EXPERIMENT_DIR / "README.md").read_text()

    for text in (runner, performance_submitter):
        assert "setup.py bdist_wheel" not in text
        assert "build_te_pr2898_backport" not in text
    assert "build_te_pr2898_backport.sub" in readme
    assert TE_COMMIT in readme


def test_te_pr2898_build_concurrent_launchers_publish_exactly_once(
    tmp_path: Path,
) -> None:
    launcher, root, fake_bin = _prepare_launcher_fixture(tmp_path)
    state = tmp_path / "srun-state"
    environment = _launcher_environment(fake_bin, mode="concurrent")
    environment["FAKE_SRUN_STATE"] = str(state)

    first = subprocess.Popen(
        ["/bin/bash", str(launcher)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not list(state.glob("entered-*")):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    second = subprocess.Popen(
        ["/bin/bash", str(launcher)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.1)
    (state / "release").touch()
    first_stdout, first_stderr = first.communicate(timeout=5)
    second_stdout, second_stderr = second.communicate(timeout=5)

    assert sorted((first.returncode, second.returncode)) == [0, 2], (
        first_stdout,
        first_stderr,
        second_stdout,
        second_stderr,
    )
    output_root = root / "artifacts" / "transformer-engine"
    outputs = [
        path
        for path in output_root.iterdir()
        if path.name.startswith("transformer-engine-pr2898-")
    ]
    assert len(outputs) == 1
    assert list(outputs[0].glob("wheel/*.whl"))
    assert not list(outputs[0].glob(".transformer-engine-pr2898-*"))
    assert not list(output_root.glob("*.lock"))


def test_te_pr2898_build_isolates_native_and_setuptools_build_dirs(
    tmp_path: Path,
) -> None:
    launcher, root, fake_bin = _prepare_launcher_fixture(tmp_path)
    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=_launcher_environment(fake_bin, mode="require-isolation"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_root = root / "artifacts" / "transformer-engine"
    output = next(
        path for path in output_root.iterdir() if not path.name.endswith(".lock")
    )
    assert (output / "provenance.json").is_file()
    assert list(output.glob("wheel/*.whl"))


def test_te_pr2898_publish_excludes_build_intermediates(tmp_path: Path) -> None:
    launcher, root, fake_bin = _prepare_launcher_fixture(tmp_path)
    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=_launcher_environment(fake_bin, mode="require-isolation"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_root = root / "artifacts" / "transformer-engine"
    output = next(
        path for path in output_root.iterdir() if not path.name.endswith(".lock")
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "provenance.json",
        "wheel",
    ]
    wheel_artifacts = sorted(path.name for path in (output / "wheel").iterdir())
    assert wheel_artifacts == [
        "transformer_engine_fixture.whl",
        "transformer_engine_fixture.whl.sha256",
    ]


def test_te_pr2898_build_failure_cleans_staging_and_releases_lock(
    tmp_path: Path,
) -> None:
    launcher, root, fake_bin = _prepare_launcher_fixture(tmp_path)
    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=_launcher_environment(fake_bin, mode="fail"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    output_root = root / "artifacts" / "transformer-engine"
    assert list(output_root.iterdir()) == []
