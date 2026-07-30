import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)
SCRIPT_PATH = EXPERIMENT_DIR / "scripts" / "validate_te_pr2898_wheel.sub"
ROOT = "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg"
TE_SOURCE = f"{ROOT}/src/TransformerEngine-fp64-thd-cudagraph-20260730"
TE_COMMIT = "c16cb9a1d850f8b8228959145c98541958903b8f"
IMAGE = f"{ROOT}/containers/nemo_rl_nightly_20260729_2472184.sqsh"
IMAGE_SHA256 = "cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91"


def _git_commit(source: Path) -> str:
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
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
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_launcher_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "nemo-rl-cg"
    source = root / "src" / "TransformerEngine-fp64-thd-cudagraph-20260730"
    tests = source / "tests" / "pytorch"
    tests.mkdir(parents=True)
    (source / "transformer_engine").mkdir()
    (source / "transformer_engine" / "__init__.py").write_text("")
    (tests / "test_pr2898_backport_compat.py").write_text(
        "def test_fixture_compat():\n    assert True\n"
    )
    (tests / "test_fused_router.py").write_text(
        "def test_fused_moe_aux_loss_cuda_graph_capture():\n    assert True\n"
    )
    commit = _git_commit(source)

    image = root / "containers" / "nemo_rl_nightly_20260729_2472184.sqsh"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture image")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()

    artifact = (
        root
        / "artifacts"
        / "transformer-engine"
        / f"transformer-engine-pr2898-{commit}"
    )
    wheel_dir = artifact / "wheel"
    wheel_dir.mkdir(parents=True)
    wheel = wheel_dir / "transformer_engine_fixture.whl"
    wheel.write_bytes(b"fixture wheel")
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (wheel_dir / f"{wheel.name}.sha256").write_text(f"{wheel_sha256}  {wheel.name}\n")
    (artifact / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "commit": commit,
                "image": str(image),
                "image_sha256": image_sha256,
                "wheel": wheel.name,
                "wheel_sha256": wheel_sha256,
                "nvte_framework": "pytorch",
                "nvte_cuda_archs": "100",
                "max_jobs": 8,
            },
            indent=2,
        )
        + "\n"
    )

    launcher = tmp_path / "validate_te_pr2898_wheel.sub"
    script = SCRIPT_PATH.read_text()
    replacements = {
        ROOT: str(root),
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
capture = os.environ.get("FAKE_SRUN_CAPTURE")
if capture:
    Path(capture).write_text(command)
tokens = shlex.split(command)
target = Path(tokens[tokens.index("--target") + 1])
target.mkdir(parents=True, exist_ok=True)
(target / "transformer_engine").mkdir()
(target / "transformer_engine" / "__init__.py").write_text("")
(target / "transformer_engine" / "wheel_lib").mkdir()
(target / "transformer_engine" / "wheel_lib" / "libtransformer_engine.so").write_bytes(b"core")
(target / "transformer_engine_torch.so").write_bytes(b"torch")
(target / "transformer_engine_fixture.dist-info").mkdir()

mode = os.environ["FAKE_SRUN_MODE"]
if mode == "fail":
    raise SystemExit(23)
if mode == "concurrent":
    state = Path(os.environ["FAKE_SRUN_STATE"])
    state.mkdir(exist_ok=True)
    (state / f"entered-{os.getpid()}").touch()
    while not (state / "release").exists():
        time.sleep(0.01)
"""
    )
    fake_srun.chmod(0o755)

    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
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
    return launcher, root, artifact, fake_bin


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


def _run_launcher(
    launcher: Path,
    fake_bin: Path,
    *,
    mode: str = "pass",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(launcher)],
        env=_launcher_environment(fake_bin, mode=mode),
        check=False,
        capture_output=True,
        text=True,
    )


def test_te_pr2898_wheel_gate_pins_gpu_source_artifact_and_image() -> None:
    script = SCRIPT_PATH.read_text()

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert f"TE_SOURCE={TE_SOURCE}" in script
    assert f"EXPECTED_TE_COMMIT={TE_COMMIT}" in script
    assert f"IMAGE={IMAGE}" in script
    assert f"IMAGE_SHA256={IMAGE_SHA256}" in script
    assert (
        "ARTIFACT_DIR=${ROOT}/artifacts/transformer-engine/"
        "transformer-engine-pr2898-${EXPECTED_TE_COMMIT}"
    ) in script
    assert (
        'test "$(sha256sum -- "${IMAGE}" | awk \'{print $1}\')" = "${IMAGE_SHA256}"'
        in script
    )
    assert (
        'test "$(git -C "${TE_SOURCE}" rev-parse HEAD)" = "${EXPECTED_TE_COMMIT}"'
        in script
    )
    assert 'git -C "${TE_SOURCE}" status --porcelain --untracked-files=all' in script
    assert 'git -C "${TE_SOURCE}" submodule status --recursive' in script


def test_te_pr2898_wheel_gate_is_offline_prefix_first_and_runs_focused_tests() -> None:
    script = SCRIPT_PATH.read_text()

    assert "uv pip install" in script
    assert "--offline" in script
    assert "--no-index" in script
    assert "--no-deps" in script
    assert "--target" in script
    assert "pip install" not in script.replace("uv pip install", "")
    assert "PYTHONNOUSERSITE=1" in script
    assert "PYTHONPATH=${INSTALL_SITE_PACKAGES}" in script
    assert "importlib.import_module('transformer_engine_torch')" in script
    assert "_get_shared_object_file('core')" in script
    assert "Path(transformer_engine.__file__).resolve()" in script
    assert "Path(transformer_engine_torch.__file__).resolve()" in script
    assert "test_pr2898_backport_compat.py" in script
    assert "test_fused_router.py::test_fused_moe_aux_loss_cuda_graph_capture" in script
    assert "--import-mode=importlib" in script
    assert "NRL_FORCE_REBUILD_VENVS" not in script
    assert "run_grpo.py" not in script


def test_te_pr2898_wheel_gate_publishes_commit_and_wheel_hash_prefix(
    tmp_path: Path,
) -> None:
    launcher, root, _, fake_bin = _prepare_launcher_fixture(tmp_path)
    result = _run_launcher(launcher, fake_bin)

    assert result.returncode == 0, result.stderr
    install_root = root / "runtimes" / "transformer-engine"
    outputs = [
        path for path in install_root.iterdir() if not path.name.endswith(".lock")
    ]
    assert len(outputs) == 1
    output = outputs[0]
    assert output.name.startswith("transformer-engine-pr2898-")
    assert "-wheel-" in output.name
    assert sorted(path.name for path in output.iterdir()) == [
        "provenance.json",
        "site-packages",
    ]
    assert (output / "site-packages" / "transformer_engine").is_dir()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["install_prefix"] == str(output)
    assert provenance["wheel_sha256"] in output.name
    assert not list(install_root.glob("*.lock"))
    assert not list(install_root.glob(".transformer-engine-pr2898-*"))


def test_te_pr2898_wheel_gate_rejects_provenance_mismatch_before_install(
    tmp_path: Path,
) -> None:
    launcher, root, artifact, fake_bin = _prepare_launcher_fixture(tmp_path)
    provenance_path = artifact / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["commit"] = "0" * 40
    provenance_path.write_text(json.dumps(provenance) + "\n")
    capture = tmp_path / "srun-command"
    environment = _launcher_environment(fake_bin, mode="pass")
    environment["FAKE_SRUN_CAPTURE"] = str(capture)

    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "provenance" in result.stderr.lower()
    assert not capture.exists()
    assert not (root / "runtimes" / "transformer-engine").exists()


def test_te_pr2898_wheel_gate_fails_closed_when_git_inspection_fails(
    tmp_path: Path,
) -> None:
    real_git = shutil.which("git")
    assert real_git is not None

    for failure_mode in ("status", "submodule"):
        launcher, root, _, fake_bin = _prepare_launcher_fixture(tmp_path / failure_mode)
        fake_git = fake_bin / "git"
        fake_git.write_text(
            f"""#!/usr/bin/env python3
import os
import sys

arguments = sys.argv[1:]
failure_mode = os.environ["FAKE_GIT_FAILURE_MODE"]
if failure_mode == "status" and "status" in arguments:
    raise SystemExit(91)
if failure_mode == "submodule" and "submodule" in arguments:
    raise SystemExit(92)
os.execv({real_git!r}, [{real_git!r}, *arguments])
"""
        )
        fake_git.chmod(0o755)
        capture = tmp_path / f"srun-command-{failure_mode}"
        environment = _launcher_environment(fake_bin, mode="pass")
        environment.update(
            {
                "FAKE_GIT_FAILURE_MODE": failure_mode,
                "FAKE_SRUN_CAPTURE": str(capture),
            }
        )

        result = subprocess.run(
            ["/bin/bash", str(launcher)],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "failed to inspect transformer engine" in result.stderr.lower()
        assert not capture.exists()
        assert not (root / "runtimes" / "transformer-engine").exists()


def test_te_pr2898_wheel_gate_rejects_hash_mismatch_and_extra_artifact(
    tmp_path: Path,
) -> None:
    launcher, root, artifact, fake_bin = _prepare_launcher_fixture(tmp_path)
    wheel = next((artifact / "wheel").glob("*.whl"))
    wheel.write_bytes(b"mutated")

    hash_result = _run_launcher(launcher, fake_bin)

    assert hash_result.returncode == 2
    assert "sha256" in hash_result.stderr.lower()
    assert not (root / "runtimes" / "transformer-engine").exists()

    wheel.write_bytes(b"fixture wheel")
    (artifact / "unexpected.txt").write_text("not allowed\n")
    whitelist_result = _run_launcher(launcher, fake_bin)

    assert whitelist_result.returncode == 2
    assert "whitelist" in whitelist_result.stderr.lower()
    assert not (root / "runtimes" / "transformer-engine").exists()


def test_te_pr2898_wheel_gate_failure_cleans_staging_and_lock(
    tmp_path: Path,
) -> None:
    launcher, root, _, fake_bin = _prepare_launcher_fixture(tmp_path)
    result = _run_launcher(launcher, fake_bin, mode="fail")

    assert result.returncode == 23
    install_root = root / "runtimes" / "transformer-engine"
    assert list(install_root.iterdir()) == []


def test_te_pr2898_wheel_gate_concurrent_install_publishes_exactly_once(
    tmp_path: Path,
) -> None:
    launcher, root, _, fake_bin = _prepare_launcher_fixture(tmp_path)
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
    install_root = root / "runtimes" / "transformer-engine"
    outputs = [
        path for path in install_root.iterdir() if not path.name.endswith(".lock")
    ]
    assert len(outputs) == 1
    assert not list(install_root.glob("*.lock"))
    assert not list(install_root.glob(".transformer-engine-pr2898-*"))


def test_te_pr2898_wheel_gate_refuses_to_replace_immutable_prefix(
    tmp_path: Path,
) -> None:
    launcher, root, _, fake_bin = _prepare_launcher_fixture(tmp_path)
    first = _run_launcher(launcher, fake_bin)
    assert first.returncode == 0, first.stderr
    output = next((root / "runtimes" / "transformer-engine").iterdir())
    sentinel = output / "sentinel"
    sentinel.write_text("keep\n")

    second = _run_launcher(launcher, fake_bin)

    assert second.returncode == 2
    assert "immutable" in second.stderr.lower()
    assert sentinel.read_text() == "keep\n"
