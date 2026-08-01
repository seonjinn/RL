from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
BRIDGE_SHA = "8e8156896abf194b99b0ac5a90bf449bd75c07eb"
MCORE_SHA = "2d19c0e07d2e8d6f061e05d55af1445bcef120a9"
NEMORL_SHA = "0" * 40
TE_SHA = "e" * 40
CONTAINER_SHA256 = "32f07be22293d9a3979e8ba04772ad48a8157dad04fd92577063ed4e07ab1493"
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)
VALID_TE_SCOPES = {
    tuple(
        module
        for enabled, module in zip(enabled_dense, DENSE_AXES, strict=True)
        if enabled
    )
    + moe_scope
    for enabled_dense in itertools.product((False, True), repeat=3)
    for moe_scope in MOE_AXES
}


def _run_script(
    relative_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_clean_git_repository(tmp_path: Path, name: str) -> tuple[Path, str]:
    repository = tmp_path / name
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "tracked.txt").write_text(f"{name}\n")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-qm", f"create {name}")
    return repository, _git(repository, "rev-parse", "HEAD")


def _load_experiment_module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nemotron_experiment_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _create_bridge_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    mcore = tmp_path / "mcore-source"
    mcore.mkdir()
    _git(mcore, "init", "-q")
    _git(mcore, "config", "user.email", "test@example.com")
    _git(mcore, "config", "user.name", "Test")
    (mcore / "README.md").write_text("fixture\n")
    _git(mcore, "add", "README.md")
    _git(mcore, "commit", "-qm", "fixture mcore")
    mcore_sha = _git(mcore, "rev-parse", "HEAD")

    bridge = tmp_path / "bridge-source"
    bridge.mkdir()
    _git(bridge, "init", "-q")
    _git(bridge, "config", "user.email", "test@example.com")
    _git(bridge, "config", "user.name", "Test")
    (bridge / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    (bridge / "uv.lock").write_text("committed-lock\n")
    for name in ("nano", "super", "ultra"):
        recipe_test = (
            bridge
            / "tests"
            / "unit_tests"
            / "recipes"
            / "nemotronh"
            / f"test_nemotron_3_{name}.py"
        )
        recipe_test.parent.mkdir(parents=True, exist_ok=True)
        recipe_test.write_text(f"def test_{name}():\n    assert True\n")
    _git(bridge, "add", ".")
    _git(bridge, "commit", "-qm", "fixture bridge")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(bridge),
            "submodule",
            "add",
            "-q",
            str(mcore),
            "3rdparty/Megatron-LM",
        ],
        check=True,
    )
    _git(bridge, "commit", "-qam", "pin fixture mcore")
    return bridge, _git(bridge, "rev-parse", "HEAD"), mcore_sha


def test_oci_bridge_bootstrap_test_only_renders_reproducible_batch_submission(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_bridge_bootstrap.sub",
        TEST_ONLY="1",
        BRIDGE_REPOSITORY="git@github.com:seonjinn/Megatron-Bridge.git",
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--account=coreai_dlalgo_nemorl" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert f"EXPECTED_BRIDGE_SHA={BRIDGE_SHA}" in result.stdout
    assert f"EXPECTED_MCORE_SHA={MCORE_SHA}" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_oci_bridge_bootstrap_rejects_credential_bearing_or_ambiguous_remote(
    tmp_path: Path,
) -> None:
    invalid_repositories = (
        "https://user:placeholder@github.com/org/Megatron-Bridge.git",
        "https://github.com/org/Megatron-Bridge.git?token=placeholder",
        "https://github.com/org/Megatron-Bridge.git#fragment",
        "http://github.com/org/Megatron-Bridge.git",
        "ssh://git@github.com/org/Megatron-Bridge.git",
        "git@github.com:org/Megatron Bridge.git",
        "git@github.com:org/Megatron-Bridge.git\nsecond-line",
    )
    for repository in invalid_repositories:
        result = _run_script(
            "scripts/validate_oci_bridge_bootstrap.sub",
            TEST_ONLY="1",
            BRIDGE_REPOSITORY=repository,
            EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
            EXPECTED_MCORE_SHA=MCORE_SHA,
            ARTIFACT_DIR=str(tmp_path / "artifacts"),
            CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
            CONTAINER_SHA256=CONTAINER_SHA256,
        )

        assert result.returncode == 2
        assert "credential-free public HTTPS or git@host:path remote" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert repository not in result.stdout
        assert repository not in result.stderr


def test_oci_bridge_bootstrap_has_no_singleton_dependency() -> None:
    result = _run_script(
        "scripts/validate_oci_bridge_bootstrap.sub",
        TEST_ONLY="1",
        BRIDGE_REPOSITORY="git@github.com:seonjinn/Megatron-Bridge.git",
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR="/lustre/example/bridge-bootstrap",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
    )

    assert result.returncode == 0, result.stderr
    assert "dependency" not in result.stdout.lower()


def test_oci_bridge_bootstrap_uses_persistent_payload_when_wrapper_is_spooled(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_bridge_bootstrap.sub"
    persistent_payload = (
        EXPERIMENT_DIR / "scripts" / "bridge_bootstrap_payload.sh"
    ).resolve()
    spool_dir = tmp_path / "slurm-spool" / "job314"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    digest = hashlib.sha256(container.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.txt"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf \'%s\\n\' "$*" >"${SRUN_LOG}"\n')
    fake_srun.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "314",
            "BRIDGE_BOOTSTRAP_PAYLOAD": str(persistent_payload),
            "BRIDGE_REPOSITORY": "git@github.com:seonjinn/Megatron-Bridge.git",
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "CONTAINER": str(container),
            "CONTAINER_SHA256": digest,
        }
    )

    result = subprocess.run(
        ["bash", str(spooled_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(persistent_payload) in srun_log.read_text()
    assert str(spool_dir / "bridge_bootstrap_payload.sh") not in srun_log.read_text()


def test_bridge_bootstrap_payload_relocks_and_runs_three_recipe_files(
    tmp_path: Path,
) -> None:
    bridge, bridge_sha, mcore_sha = _create_bridge_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_call_log = tmp_path / "uv-calls.txt"
    python_call_log = tmp_path / "python-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s|%s\\n' "${FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD:-}" "$*" >>"${UV_CALL_LOG}"
if [[ "$1" == "lock" ]]; then
  printf 'resolved-lock\\n' >uv.lock
  exit 0
fi
exit 99
"""
    )
    fake_uv.chmod(0o755)
    fake_python = fake_bin / "container-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s|%s\\n' "${PYTHONPATH:-}" "$*" >>"${PYTHON_CALL_LOG}"
if [[ "$1" == "--version" ]]; then
  printf 'Python 3.12.9\\n'
  exit 0
fi
if [[ "$1" == "-c" ]]; then
  exit 0
fi
for argument in "$@"; do
  case "${argument}" in
    --junitxml=*)
      junit=${argument#--junitxml=}
      mkdir -p "$(dirname "${junit}")"
      printf '<testsuite tests="3" failures="0"/>\\n' >"${junit}"
      ;;
  esac
done
printf '3 passed\\n'
"""
    )
    fake_python.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    work_parent = tmp_path / "work"
    work_parent.mkdir()
    sentinel = work_parent / "caller-owned.txt"
    sentinel.write_text("preserve\n")

    result = _run_script(
        "scripts/bridge_bootstrap_payload.sh",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        GIT_ALLOW_PROTOCOL="file",
        UV_CALL_LOG=str(uv_call_log),
        PYTHON_CALL_LOG=str(python_call_log),
        BRIDGE_REPOSITORY=str(bridge),
        EXPECTED_BRIDGE_SHA=bridge_sha,
        EXPECTED_MCORE_SHA=mcore_sha,
        ARTIFACT_DIR=str(artifacts),
        CONTAINER="/lustre/example/nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        SLURM_JOB_ID="314",
        WORK_ROOT=str(work_parent),
        LOCK_PYTHON="/usr/bin/python3.12",
        CONTAINER_PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text() == "preserve\n"
    assert list(work_parent.glob("bridge-bootstrap-314.*")) == []
    result_dir = artifacts / f"bridge-{bridge_sha}-314"
    assert (result_dir / "uv.lock").read_text() == "resolved-lock\n"
    assert (result_dir / "status.txt").read_text() == "passed\n"
    assert (result_dir / "recipe-tests.junit.xml").is_file()
    calls = uv_call_log.read_text().splitlines()
    assert calls == [
        "TRUE|lock --no-build-isolation-package fast-hadamard-transform "
        "--python /usr/bin/python3.12 --no-python-downloads"
    ]
    python_calls = python_call_log.read_text().splitlines()
    assert python_calls[0].endswith("|--version")
    assert "|-c import megatron.bridge" in python_calls[1]
    assert "transformer_engine" in python_calls[1]
    assert "|-m pytest -q" in python_calls[2]
    assert "test_nemotron_3_nano.py" in python_calls[2]
    assert "test_nemotron_3_super.py" in python_calls[2]
    assert "test_nemotron_3_ultra.py" in python_calls[2]
    assert "/Megatron-Bridge/src:" in python_calls[1]
    assert "/Megatron-Bridge/3rdparty/Megatron-LM" in python_calls[1]
    provenance = (result_dir / "provenance.env").read_text()
    assert f"bridge_sha={bridge_sha}" in provenance
    assert f"mcore_sha={mcore_sha}" in provenance
    assert "uv_lock_sha256=" in provenance
    assert "lock_python=/usr/bin/python3.12" in provenance
    assert f"container_python={fake_python}" in provenance
    assert "container_python_version=Python 3.12.9" in provenance


def test_bridge_bootstrap_payload_rejects_credential_bearing_remote(
    tmp_path: Path,
) -> None:
    repository = "https://user:placeholder@github.com/org/Megatron-Bridge.git"
    result = _run_script(
        "scripts/bridge_bootstrap_payload.sh",
        BRIDGE_REPOSITORY=repository,
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        CONTAINER="/lustre/example/nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        WORK_ROOT=str(tmp_path / "work"),
    )

    assert result.returncode == 2
    assert (
        "BRIDGE_REPOSITORY is not an approved credential-free source" in result.stderr
    )
    assert repository not in result.stderr
    assert not (tmp_path / "artifacts").exists()


def test_stage_enroot_image_test_only_renders_immutable_batch_submission(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidia/nemo:nightly",
        SOURCE_DIGEST=digest,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_20260731",
        CONTAINER_DIR=str(tmp_path / "containers"),
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert f"SOURCE_DIGEST={digest}" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "containers").exists()


def test_stage_enroot_image_rejects_unpinned_source_before_submission(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidia/nemo:nightly",
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_20260731",
        CONTAINER_DIR=str(tmp_path / "containers"),
    )

    assert result.returncode == 2
    assert "SOURCE_DIGEST" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_stage_enroot_image_rejects_credential_bearing_or_ambiguous_references(
    tmp_path: Path,
) -> None:
    invalid_images = (
        "user:placeholder@nvcr.io/nvidia/nemo-rl:nightly",
        "nvcr.io/nvidia/nemo-rl:nightly?token=placeholder",
        "https://nvcr.io/nvidia/nemo-rl:nightly",
        "nvcr.io/nvidia/nemo rl:nightly",
        "nvcr.io/nvidia/nemo-rl:nightly\nsecond-line",
    )
    for source_image in invalid_images:
        result = _run_script(
            "scripts/stage_enroot_image.sbatch",
            TEST_ONLY="1",
            SOURCE_IMAGE=source_image,
            SOURCE_DIGEST="sha256:" + "a" * 64,
            SOURCE_COMMIT="b" * 40,
            OUTPUT_PREFIX="nemo_rl_nightly_20260731",
            CONTAINER_DIR=str(tmp_path / "containers"),
        )

        assert result.returncode == 2
        assert "credential-free registry/repository reference" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert source_image not in result.stdout
        assert source_image not in result.stderr


def test_stage_enroot_image_requires_full_source_commit_before_submission(
    tmp_path: Path,
) -> None:
    for source_commit in ("", "abc123"):
        result = _run_script(
            "scripts/stage_enroot_image.sbatch",
            TEST_ONLY="1",
            SOURCE_IMAGE="nvcr.io/nvidia/nemo-rl:nightly",
            SOURCE_DIGEST="sha256:" + "a" * 64,
            SOURCE_COMMIT=source_commit,
            OUTPUT_PREFIX="nemo_rl_nightly_20260731",
            CONTAINER_DIR=str(tmp_path / "containers"),
        )

        assert result.returncode == 2
        assert "SOURCE_COMMIT must be a full 40-character commit" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert "SOURCE_IMAGE=" not in result.stdout


def test_scope_matrix_contains_all_32_te_rows_and_baseline() -> None:
    module = _load_experiment_module("scope_matrix")

    rows = module.load_scope_matrix()

    assert len(rows) == 33
    assert rows[0].scope == ()
    assert rows[0].cuda_graph_enabled is False
    assert {row.scope for row in rows[1:]} == VALID_TE_SCOPES
    assert rows[-1].scope == (
        "attn",
        "mlp",
        "mamba",
        "moe_router",
        "moe_preprocess",
    )


def test_scope_matrix_list_command_prints_every_persistent_row() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "scope_matrix.py"),
            "list",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 33
    assert lines[0].startswith("00\tbaseline_no_cg\tbaseline\t")
    assert lines[-1].startswith(
        "32\tattn_mlp_mamba_moe_router_preprocess\t"
        "attn,mlp,mamba,moe_router,moe_preprocess\t"
    )


def test_scope_launcher_does_not_require_host_uv(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 91\n")
    fake_uv.chmod(0o755)

    result = _run_script(
        "scopes/17_attn.sh",
        TEST_ONLY="1",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert "COMMAND:" in result.stdout


def test_scope_classifier_reports_pre_submission_outcomes() -> None:
    module = _load_experiment_module("scope_matrix")
    rows = module.load_scope_matrix()
    by_name = {row.name: row for row in rows}

    assert module.classify_scope(by_name["attn"], model="nano").status == "runnable"
    assert (
        module.classify_scope(by_name["mamba"], model="qwen3_30ba3b").status
        == "model-incompatible"
    )
    assert (
        module.classify_scope(by_name["mlp"], model="qwen3_30ba3b").status
        == "model-incompatible"
    )
    assert (
        module.classify_scope(by_name["moe"], model="nano").status == "capacity-blocked"
    )
    nano_preprocess = module.classify_scope(
        by_name["moe_router_preprocess"], model="nano"
    )
    assert nano_preprocess.status == "capacity-blocked"
    assert "HybridEP moe_preprocess" in nano_preprocess.reason
    assert (
        module.classify_scope(by_name["attn"], model="ultra").status
        == "dependency-blocked"
    )
    ultra_with_external_paths = module.classify_scope(
        by_name["attn"],
        model="ultra",
        external_dependencies_ready=True,
    )
    assert ultra_with_external_paths.status == "dependency-blocked"
    assert "validated launcher adapter" in ultra_with_external_paths.reason
    assert (
        module.classify_scope(by_name["attn"], model="nano", mode="mcore").status
        == "dependency-blocked"
    )
    assert (
        module.classify_scope(
            by_name["attn"],
            model="nano",
            submitted_job_id="12345",
        ).status
        == "submitted"
    )


def test_rendered_nemorl_command_uses_only_current_graph_fields() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn",),
        steps=20,
        run_name="nano-attn-test",
    )

    assert "checkpointing.enabled=false" in command
    assert "policy.megatron_cfg.cuda_graph_warmup_steps=3" in command
    assert "policy.megatron_cfg.cuda_graph_modules=[attn]" in command
    assert "policy.megatron_cfg.thd_max_packed_sequences=16" in command
    assert "logger.wandb.project=sna-cg-study" in shlex.split(command)
    assert "NRL_FORCE_REBUILD_VENVS=true" in command
    assert "cuda_graph_scope" not in command
    assert "cuda_graph_max_packed_seqs" not in command
    assert "cuda_graph_max_cached_schedules" not in command


def test_rendered_nemorl_command_adds_experimental_struct_keys_with_hydra_plus_plus() -> (
    None
):
    module = _load_experiment_module("scope_matrix")

    baseline_arguments = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=(),
            steps=20,
            run_name="nano-baseline-struct-test",
            cuda_graph_enabled=False,
        )
    )
    graph_arguments = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=("attn", "moe_router"),
            steps=20,
            run_name="nano-graph-struct-test",
        )
    )

    assert "++policy.megatron_cfg.cuda_graph_impl=none" in baseline_arguments
    assert not any(
        argument.startswith("++policy.megatron_cfg.cuda_graph_modules=")
        for argument in baseline_arguments
    )
    assert (
        "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" in graph_arguments
    )
    assert "++policy.megatron_cfg.cuda_graph_impl=transformer_engine" in graph_arguments
    assert (
        "++policy.megatron_cfg.cuda_graph_modules=[attn,moe_router]" in graph_arguments
    )
    assert "++policy.megatron_cfg.cuda_graph_warmup_steps=3" in graph_arguments
    assert "++policy.megatron_cfg.thd_max_packed_sequences=16" in graph_arguments


def test_rendered_nano_command_pins_the_claimed_hybridep_dispatcher() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn", "moe_router"),
        steps=20,
        run_name="nano-hybridep-test",
    )

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" in command
    assert "policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" in command
    assert "policy.megatron_cfg.moe_token_dispatcher_type=hybridep" not in command


def test_rendered_command_shell_quotes_log_paths_without_changing_arguments() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn",),
        steps=20,
        run_name="nano-safe-path",
        log_dir="/lustre/experiment path/attn;literal",
    )

    arguments = shlex.split(command)
    assert "logger.log_dir=/lustre/experiment path/attn;literal" in arguments
    assert arguments.count("uv") == 1


def test_ultra_command_is_fail_closed_until_launcher_adapter_is_validated() -> None:
    module = _load_experiment_module("scope_matrix")

    with pytest.raises(ValueError, match="validated launcher adapter"):
        module.render_scope_command(
            model="ultra",
            scope=("attn",),
            steps=20,
            run_name="ultra-attn-test",
        )


def test_scope_and_variant_leaves_are_persistent_and_exact() -> None:
    module = _load_experiment_module("scope_matrix")
    rows = module.load_scope_matrix()
    scopes = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    variants = sorted((EXPERIMENT_DIR / "variants").glob("*.sh"))

    assert [path.name for path in scopes] == [
        f"{row.index:02d}_{row.name}.sh" for row in rows
    ]
    assert len(variants) == 9
    for launcher in [*scopes, *variants]:
        text = launcher.read_text()
        assert "WARMUP_STEPS=3" in text
        assert "THD_MAX_PACKED_SEQUENCES=16" in text
        assert "CHECKPOINTING_ENABLED=false" in text
        assert "WANDB_PROJECT=sna-cg-study" in text
        assert 'bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"' in text


def test_model_selectors_cover_nemotron_and_qwen_recipes() -> None:
    expected = {
        "nano.env": "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml",
        "super.env": "examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-8n4g-megatron.yaml",
        "ultra.env": "examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml",
        "qwen3_30ba3b.env": "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
    }

    assert {
        path.name: next(
            line.removeprefix("NEMORL_RECIPE=")
            for line in path.read_text().splitlines()
            if line.startswith("NEMORL_RECIPE=")
        )
        for path in sorted((EXPERIMENT_DIR / "models").glob("*.env"))
    } == expected


def test_nano_test_only_launcher_renders_batch_job_without_singleton() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_GROUP="unit-performance-group",
        REPEAT_INDEX="2",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: runnable" in result.stdout
    assert "verify_runtime_attestation.py" in result.stdout
    assert "validate_te_runtime.py" not in result.stdout
    assert "policy.megatron_cfg.cuda_graph_modules=\\[attn\\]" in result.stdout
    assert "policy.megatron_cfg.thd_max_packed_sequences=16" in result.stdout
    assert "policy.megatron_cfg.cuda_graph_warmup_steps=3" in result.stdout
    assert "checkpointing.enabled=false" in result.stdout
    assert "logger.wandb.project=sna-cg-study" in result.stdout
    assert "--partition=batch" in result.stdout
    assert f"--chdir={REPO_ROOT}" in result.stdout
    assert "RUN_GROUP: unit-performance-group" in result.stdout
    assert "REPEAT_INDEX: 2" in result.stdout
    assert "run_nemorl_scope.sub" in result.stdout
    assert "dependency" not in result.stdout.lower()
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_leaf_job_depends_on_one_exact_runtime_preflight_artifact(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "oci-runtime-attested.env"
    attestation = "/lustre/example/runtime/oci-container-runtime-733.json"
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=oci-hsg-runtime-attested",
                "ACCOUNT=coreai_dlalgo_nemorl",
                "PARTITION=batch",
                "CONTAINER=/lustre/example/nemo_rl_immutable.sqsh",
                f"CONTAINER_SHA256={CONTAINER_SHA256}",
                "MOUNTS=/lustre:/lustre",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=gpu:4",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=04:00:00",
                f"RUNTIME_ATTESTATION={attestation}",
                "RUNTIME_PREFLIGHT_JOB_ID=733",
                f"EXPECTED_TE_SHA={TE_SHA}",
                f"EXPECTED_NEMORL_SHA={NEMORL_SHA}",
                f"EXPECTED_BRIDGE_SHA={BRIDGE_SHA}",
                f"EXPECTED_MCORE_SHA={MCORE_SHA}",
                "",
            )
        )
    )

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        PROFILE_FILE=str(profile),
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "--dependency=afterok:733" in result.stdout
    assert "verify_runtime_attestation.py" in result.stdout
    assert attestation in result.stdout
    assert "validate_te_runtime.py" not in result.stdout


def test_source_provenance_verifier_rejects_queued_worktree_mutation(
    tmp_path: Path,
) -> None:
    repositories_and_commits = [
        _create_clean_git_repository(tmp_path, name)
        for name in ("nemo-rl", "bridge", "mcore")
    ]
    verifier = (EXPERIMENT_DIR / "scripts" / "verify_source_provenance.sh").resolve()
    arguments = [str(verifier)]
    for repository, commit in repositories_and_commits:
        arguments.extend((str(repository.resolve()), commit))

    clean = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr
    assert "SOURCE_PROVENANCE_VERIFIED=true" in clean.stdout

    (repositories_and_commits[1][0] / "tracked.txt").write_text("mutated\n")
    dirty = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dirty.returncode != 0
    assert "Megatron-Bridge source worktree has unstaged changes" in dirty.stderr


def test_scope_job_wrappers_revalidate_source_before_container_execution() -> None:
    for name in ("run_nemorl_scope.sub", "run_mcore_scope.sub"):
        source = (EXPERIMENT_DIR / "scripts" / name).read_text()
        verification_offset = source.index('"${SOURCE_PROVENANCE_VERIFIER}"')
        attestation_offset = source.index('"${RUNTIME_ATTESTATION_COMMAND}"')

        assert verification_offset < attestation_offset
        assert "EXPECTED_NEMORL_SHA" in source
        assert "EXPECTED_BRIDGE_SHA" in source
        assert "EXPECTED_MCORE_SHA" in source
        assert 'sha256sum "${CONTAINER}"' not in source


def test_source_snapshot_copies_exact_recursive_gitlinks_and_writes_manifest(
    tmp_path: Path,
) -> None:
    mcore, mcore_sha = _create_clean_git_repository(tmp_path, "mcore-source")
    bridge, _ = _create_clean_git_repository(tmp_path, "bridge-source")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(bridge),
            "submodule",
            "add",
            "-q",
            str(mcore),
            "3rdparty/Megatron-LM",
        ],
        check=True,
    )
    _git(bridge, "commit", "-qm", "pin mcore")
    bridge_sha = _git(bridge, "rev-parse", "HEAD")
    nested_mcore = bridge / "3rdparty" / "Megatron-LM"

    outer, _ = _create_clean_git_repository(tmp_path, "nemo-rl-source")
    experiment_scripts = (
        outer
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "scripts"
    )
    experiment_scripts.mkdir(parents=True)
    for name in ("create_source_snapshot.sh", "verify_source_provenance.sh"):
        shutil.copy2(EXPERIMENT_DIR / "scripts" / name, experiment_scripts / name)
    (outer / "uv.lock").write_text("fixture-lock\n")
    _git(outer, "add", "experiments", "uv.lock")
    _git(outer, "commit", "-qm", "add snapshot tools")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(outer),
            "submodule",
            "add",
            "-q",
            str(bridge),
            "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge",
        ],
        check=True,
    )
    _git(outer, "commit", "-qm", "pin bridge")
    outer_sha = _git(outer, "rev-parse", "HEAD")
    nested_bridge = outer / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge"
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(outer),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )

    snapshot_store = tmp_path / "snapshots"
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "scripts" / "create_source_snapshot.sh")],
        env=os.environ
        | {
            "SOURCE_ROOT": str(outer.resolve()),
            "SNAPSHOT_STORE": str(snapshot_store.resolve()),
            "EXPECTED_NEMORL_SHA": outer_sha,
            "EXPECTED_BRIDGE_SHA": bridge_sha,
            "EXPECTED_MCORE_SHA": mcore_sha,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    snapshot = snapshot_store / f"{outer_sha[:12]}-{bridge_sha[:12]}-{mcore_sha[:12]}"
    manifest = snapshot / ".source-manifest.env"
    assert _git(snapshot, "rev-parse", "HEAD") == outer_sha
    assert (
        _git(
            snapshot / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge",
            "rev-parse",
            "HEAD",
        )
        == bridge_sha
    )
    assert (
        _git(
            snapshot
            / "3rdparty"
            / "Megatron-Bridge-workspace"
            / "Megatron-Bridge"
            / "3rdparty"
            / "Megatron-LM",
            "rev-parse",
            "HEAD",
        )
        == mcore_sha
    )
    submodule_status = subprocess.run(
        ["git", "-C", str(snapshot), "submodule", "status", "--recursive"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(submodule_status) == 2
    assert all(line.startswith(" ") for line in submodule_status)
    manifest_text = manifest.read_text()
    assert f"nemo_rl_commit={outer_sha}" in manifest_text
    assert f"bridge_commit={bridge_sha}" in manifest_text
    assert f"mcore_commit={mcore_sha}" in manifest_text
    assert "uv_lock_sha256=" in manifest_text


def test_ray_submission_has_no_global_singleton_dependency() -> None:
    ray_submission = (REPO_ROOT / "ray.sub").read_text()

    assert "#SBATCH --dependency=singleton" not in ray_submission


def test_cluster_profiles_render_cluster_specific_gres_and_segment_contracts() -> None:
    ptyche = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )
    lyris = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="lyris",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert ptyche.returncode == 0, ptyche.stderr
    assert "--gres=gpu:4" in ptyche.stdout
    assert "--segment=16" in ptyche.stdout
    assert lyris.returncode == 0, lyris.stderr
    assert "--gres=" not in lyris.stdout
    assert "--segment=" not in lyris.stdout


def test_mcore_launcher_is_dependency_blocked_without_standalone_driver() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="mcore",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: dependency-blocked" in result.stdout
    assert "MCORE_DRIVER" in result.stdout
    assert "SBATCH:" not in result.stdout


def test_mcore_launcher_rejects_successful_noop_as_a_driver() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="mcore",
        MCORE_DRIVER="true",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: dependency-blocked" in result.stdout
    assert "committed standalone driver" in result.stdout
    assert "COMMAND:" not in result.stdout
    assert "SBATCH:" not in result.stdout


def test_submitters_pin_smoke_performance_and_accuracy_steps() -> None:
    cases = (
        ("submit_smoke_matrix.sh", "5"),
        ("submit_performance_matrix.sh", "20"),
        ("submit_accuracy_soak.sh", "100"),
    )
    for relative_path, steps in cases:
        result = _run_script(
            relative_path,
            CLUSTER="oci-hsg",
            MODEL="qwen3_30ba3b",
            MODE="nemorl",
            TEST_ONLY="1",
            RUN_TAG="unit",
        )
        assert result.returncode == 0, result.stderr
        assert f"STEPS: {steps}" in result.stdout


def test_oci_container_runtime_smoke_renders_four_gpu_batch_job(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--account=coreai_dlalgo_nemorl" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert "singleton" not in result.stdout.lower()
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_oci_container_runtime_smoke_uses_persistent_probe_when_spooled(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    persistent_probe = (EXPERIMENT_DIR / "validate_container_runtime.py").resolve()
    spool_dir = tmp_path / "slurm-spool" / "job315"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    digest = hashlib.sha256(container.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.txt"
    provenance_verifier = tmp_path / "verify_source_provenance.sh"
    provenance_verifier.write_text("#!/bin/bash\nset -euo pipefail\n")
    provenance_verifier.chmod(0o755)
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >"${SRUN_LOG}"
output=
while (($#)); do
  if [[ "$1" == "--output" ]]; then
    shift
    output=$1
  fi
  shift
done
printf '{"status":"passed"}\n' >"${output}"
"""
    )
    fake_srun.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "315",
            "CONTAINER_RUNTIME_VALIDATOR": str(persistent_probe),
            "CONTAINER": str(container),
            "CONTAINER_SHA256": digest,
            "ARTIFACT_DIR": str(artifacts),
            "CONTAINER_PYTHON": "/fixture/python",
            "EXPECTED_NEMORL_SHA": NEMORL_SHA,
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "EXPECTED_TE_SHA": TE_SHA,
            "SOURCE_PROVENANCE_VERIFIER": str(provenance_verifier),
        }
    )

    result = subprocess.run(
        ["bash", str(spooled_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(persistent_probe) in srun_log.read_text()
    assert (
        str(spool_dir / "../validate_container_runtime.py") not in srun_log.read_text()
    )
    assert (artifacts / "oci-container-runtime-315.json").is_file()


def test_container_runtime_probe_requires_four_visible_gpus_and_packages(
    tmp_path: Path,
) -> None:
    module = _load_experiment_module("validate_container_runtime")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"GPU-{index}"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            del index
            return 10, 0

    modules = {
        name: SimpleNamespace(__file__=str(tmp_path / f"{name}.py"))
        for name in module.REQUIRED_MODULE_DISTRIBUTIONS
    }
    modules["torch"] = SimpleNamespace(
        __file__=str(tmp_path / "torch.py"),
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
    )

    result = module.probe_runtime(
        expected_device_count=4,
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
    )

    assert result["cuda_available"] is True
    assert result["device_count"] == 4
    assert [device["name"] for device in result["devices"]] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    assert set(result["packages"]) == {
        "torch",
        "transformer_engine.pytorch",
        "megatron.core",
        "megatron.bridge",
        "mamba_ssm",
        "causal_conv1d",
        "cupy",
        "grouped_gemm",
    }


def test_container_runtime_probe_rejects_wrong_gpu_count(tmp_path: Path) -> None:
    module = _load_experiment_module("validate_container_runtime")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 3

    torch_module = SimpleNamespace(
        __file__=str(tmp_path / "torch.py"),
        cuda=FakeCuda(),
    )

    try:
        module.probe_runtime(
            expected_device_count=4,
            importer=lambda name: torch_module if name == "torch" else None,
            version_getter=lambda distribution: distribution,
        )
    except RuntimeError as error:
        assert "expected exactly 4 visible CUDA devices, got 3" in str(error)
    else:
        raise AssertionError("three visible GPUs must fail the OCI runtime smoke")


def test_readme_documents_container_runtime_gate_and_artifact() -> None:
    readme = (EXPERIMENT_DIR / "README.md").read_text()
    normalized_readme = " ".join(readme.split())

    assert "scripts/validate_oci_container_runtime.sub" in readme
    assert "exactly four visible devices" in readme
    assert "machine-readable success or failure artifact" in readme
    for package in (
        "PyTorch",
        "Transformer Engine",
        "Megatron Core",
        "Megatron Bridge",
        "Mamba SSM",
        "causal-conv1d",
        "CuPy",
        "grouped GEMM",
    ):
        assert package in normalized_readme


def _complete_result_record(*, model: str, step: int) -> dict[str, object]:
    return {
        "model": model,
        "dispatcher": "alltoall",
        "scope": "attn,moe_router",
        "status": "passed",
        "mode": "nemorl",
        "cluster": "oci-hsg",
        "profile": "oci-hsg-gb200",
        "phase": "performance",
        "steps": 20,
        "step": step,
        "job_id": f"job-{model}",
        "nemo_rl_commit": "1" * 40,
        "bridge_commit": "2" * 40,
        "mcore_commit": "3" * 40,
        "te_commit": "4" * 40,
        "te_version": "2.16.0.dev0",
        "container_sha256": "5" * 64,
        "metrics": {
            "timing/train/total_step_time": float(step),
            "timing/train/generation": float(step + 1),
            "timing/train/policy_training": float(step + 2),
            "timing/train/policy_and_reference_logprobs": float(step + 3),
            "performance/tokens_per_sec_per_gpu": float(1000 + step),
            "performance/generation_tokens_per_sec_per_gpu": float(2000 + step),
            "performance/policy_training_tokens_per_sec_per_gpu": float(3000 + step),
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": float(
                4000 + step
            ),
            "cuda_graph/graph_calls": 75,
            "cuda_graph/eligible_calls": 100,
            "cuda_graph/coverage": 0.75,
            "cuda_graph/logical_tokens": 80,
            "cuda_graph/padded_tokens": 96,
            "cuda_graph/capacity_tokens": 128,
            "cuda_graph/capacity_utilization": 0.625,
            "cuda_graph/padding_utilization": 0.833333,
            "train/reward": 0.8,
            "train/loss": 0.2,
            "train/gen_kl_error": 0.01,
            "train/token_mult_prob_error": 0.02,
            "train/grad_norm": 1.5,
            "correctness/router_topk_parity": True,
            "correctness/expert_count_parity": True,
            "correctness/nan_inf_status": "clear",
        },
    }


def test_collector_normalizes_full_current_result_schema(tmp_path: Path) -> None:
    collector = _load_experiment_module("collect_results")
    row = collector.normalize_record(_complete_result_record(model="nano", step=6))

    assert set(collector.REQUIRED_REPORT_FIELDS) <= set(collector.CSV_FIELDS)
    assert row["model"] == "nano"
    assert row["profile"] == "oci-hsg-gb200"
    assert row["e2e_step_time"] == 6.0
    assert row["logprob_tokens_per_sec_per_gpu"] == 4006.0
    assert row["graph_calls"] == 75
    assert row["eligible_calls"] == 100
    assert row["graph_coverage"] == 0.75
    assert row["logical_tokens"] == 80
    assert row["capacity_utilization"] == 0.625
    assert row["reward"] == 0.8
    assert row["policy_loss"] == 0.2
    assert row["gen_kl_error"] == 0.01
    assert row["token_mult_prob_error"] == 0.02
    assert row["router_topk_parity"] is True
    assert row["expert_count_parity"] is True
    assert row["nan_inf_status"] == "clear"
    assert row["te_version"] == "2.16.0.dev0"

    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"
    collector.write_results([row], output_json=output_json, output_csv=output_csv)
    payload = json.loads(output_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["fields"] == list(collector.CSV_FIELDS)
    assert payload["rows"] == [row]
    assert output_csv.read_text().splitlines()[0].split(",") == list(
        collector.CSV_FIELDS
    )


def test_collector_preserves_failures_without_metric_rows() -> None:
    collector = _load_experiment_module("collect_results")

    row = collector.normalize_record(
        {
            "model": "ultra",
            "dispatcher": "flex",
            "scope": "attn",
            "status": "failed",
            "failure": "CUDA out of memory",
            "exit_code": 1,
            "mode": "nemorl",
            "cluster": "oci-hsg",
            "profile": "oci-hsg-gb200",
            "phase": "smoke",
            "steps": 5,
            "job_id": "failed-job",
        }
    )

    assert row["status"] == "failed"
    assert row["failure"] == "CUDA out of memory"
    assert row["exit_code"] == 1
    assert row["e2e_step_time"] == ""


def test_steady_state_rows_exclude_three_warmups_and_capture_window() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        collector.normalize_record(_complete_result_record(model="nano", step=step))
        for step in range(1, 21)
    ]

    steady = collector.steady_state_rows(rows)

    assert [row["step"] for row in steady] == list(range(6, 21))


def test_report_is_multi_model_escaped_and_separates_coverage_definitions() -> None:
    collector = _load_experiment_module("collect_results")
    renderer = _load_experiment_module("render_report")
    rows = [
        collector.normalize_record(_complete_result_record(model="nano", step=6)),
        collector.normalize_record(_complete_result_record(model="super", step=6)),
        collector.normalize_record(
            {
                "model": "<script>alert(1)</script>",
                "dispatcher": "flex",
                "scope": "attn",
                "status": "failed",
                "failure": "OOM <node>",
                "mode": "nemorl",
                "cluster": "oci-hsg",
                "profile": "oci-hsg-gb200",
                "phase": "smoke",
                "steps": 5,
                "job_id": "failed-job",
            }
        ),
    ]
    nsys = {
        "nano-attn": {
            "nsys_profile_count": 4,
            "nsys_profiles_with_cuda_graph_launches": 4,
            "nsys_cuda_graph_launch_share_of_cuda_api_calls_pct": 12.5,
        }
    }

    report = renderer.render_html(rows, nsys_coverage=nsys)

    assert "nano" in report
    assert "super" in report
    assert "Runtime graph coverage (graph_calls / eligible_calls)" in report
    assert "75.00%" in report
    assert "Nsight CUDA API launch share" in report
    assert "12.50%" in report
    assert "Raw failures" in report
    assert "OOM &lt;node&gt;" in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<script>alert(1)</script>" not in report
    assert "https://" not in report
