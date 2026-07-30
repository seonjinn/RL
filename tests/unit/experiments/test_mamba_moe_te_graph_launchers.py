import hashlib
import importlib.util
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENT_DIR = (
    Path(__file__).parents[3]
    / "experiments"
    / "cuda_graph"
    / "mamba_moe_te_graph_20260729"
)
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)
VALID_GRAPH_SCOPES = {
    tuple(
        name for enabled, name in zip(enabled_dense, DENSE_AXES, strict=True) if enabled
    )
    + moe_scope
    for enabled_dense in itertools.product((False, True), repeat=3)
    for moe_scope in MOE_AXES
}
TE_FP64_WEAKREF_COMMIT = "e707aa46869dc2aec08dfea25402e97a61d49fef"
TE_FP64_WEAKREF_SHA256 = (
    "39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b"
)
TE_FP64_WEAKREF_SOURCE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/src/"
    "TransformerEngine-fp64-weakref-20260729/.overlay/"
    f"{TE_FP64_WEAKREF_COMMIT}/utils.py"
)
TE_FP64_WEAKREF_TARGET = (
    "/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e/transformer_engine/pytorch/utils.py"
)
TE_EXPECTED_VERSION = "2.15.0+42b84005"
MCORE_LOCK_BLOB = "96543608420ac6746cfd18d1fcd8ee1bd3c91caf"
MCORE_DRIVER_PYTHON = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/venvs/"
    "nemorl-integration-mcore-py313-aarch64-lock-"
    f"{MCORE_LOCK_BLOB}-image-cb8ae0ade02b/bin/python"
)
IMMUTABLE_RUNTIME_ARCHIVES = (
    (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/runtimes/"
        "transformer-engine/transformer-engine-pr2898-"
        "4a18653fc7274b10e33cd786b91be6261c523dc0-wheel-"
        "029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e/"
        "site-packages"
    ),
    "/root/.cache/uv/archive-v0/26H_iFoUOK00pyG5",
    "/root/.cache/uv/archive-v0/ymbKBYrUysuiERDQ",
    "/root/.cache/uv/archive-v0/Lp_mVBWGrC-sLPL6",
    "/root/.cache/uv/archive-v0/kIpfdwf26Al4-BTb",
    "/root/.cache/uv/archive-v0/i7-d_jifMXRoKKrY",
)
TE_NATIVE_COMMIT = "4a18653fc7274b10e33cd786b91be6261c523dc0"
TE_NATIVE_WHEEL_SHA256 = (
    "029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e"
)
TE_NATIVE_RUNTIME = str(Path(IMMUTABLE_RUNTIME_ARCHIVES[0]).parent)
CONTAINER_SHA256 = (
    "cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91"
)


def _assignment(script: Path, name: str) -> str:
    match = re.search(
        rf"(?:^|[ \n]){name}=(?:'([^']*)'|([^ \n]+))",
        script.read_text(),
        re.MULTILINE,
    )
    assert match is not None, f"{script.name} does not set {name}"
    return match.group(1) or match.group(2)


def _scope(script: Path) -> tuple[str, ...]:
    value = _assignment(script, "SCOPE")
    assert value.startswith("[") and value.endswith("]")
    return tuple(part for part in value[1:-1].split(",") if part)


def _run_script(
    relative_path: str,
    **environment: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / relative_path)],
        cwd=EXPERIMENT_DIR.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _create_test_native_runtime(site_packages: Path, container: Path) -> Path:
    package = site_packages / "transformer_engine"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "libtransformer_engine.so").touch()
    provenance = site_packages.parent / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "commit": TE_NATIVE_COMMIT,
                "wheel_sha256": TE_NATIVE_WHEEL_SHA256,
                "image": str(container),
                "image_sha256": CONTAINER_SHA256,
                "install_prefix": str(site_packages.parent),
            }
        )
    )
    return provenance


def _run_integration_repo_preflight(
    root: Path,
    repo_override: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute only the runner's repository selection before any scheduler command."""
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    preflight, marker, _ = script.partition("IMAGE=")
    assert marker
    preflight = re.sub(
        r"(?m)^ROOT=.*$",
        f"ROOT={shlex.quote(str(root))}",
        preflight,
    )
    environment = os.environ.copy()
    if repo_override is None:
        environment.pop("REPO_OVERRIDE", None)
    else:
        environment["REPO_OVERRIDE"] = repo_override
    return subprocess.run(
        ["bash", "-c", preflight + '\nprintf "REPO=%s\\n" "${REPO}"\n'],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _run_untracked_path_guard(
    path_listing_command: str,
) -> subprocess.CompletedProcess[str]:
    guard = f"""\
set -euo pipefail
{path_listing_command} | while IFS= read -r untracked_path; do
  case "${{untracked_path}}" in
    tests/unit/unit_results.json) ;;
    *)
      if [[ ! "${{untracked_path}}" =~ ^tests/unit/unit_results/[^/]+\\.json$ ]]; then
        echo "Unexpected untracked path: ${{untracked_path}}" >&2
        exit 1
      fi
      ;;
  esac
done
"""
    return subprocess.run(
        ["bash", "-c", guard],
        check=False,
        capture_output=True,
        text=True,
    )


def _load_experiment_module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_scope_matrix_is_complete_and_exact() -> None:
    assert len(VALID_GRAPH_SCOPES) == 32
    assert EXPERIMENT_DIR.is_dir()

    scripts = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    baseline = [script for script in scripts if script.name == "00_baseline_no_cg.sh"]
    te_scripts = [script for script in scripts if script not in baseline]

    assert len(baseline) == 1
    assert len(te_scripts) == 32
    assert {_scope(script) for script in te_scripts} == VALID_GRAPH_SCOPES
    assert _assignment(baseline[0], "CUDA_GRAPH_IMPL") == "none"
    assert _assignment(baseline[0], "SCOPE") == "[no_cg]"


def test_scope_scripts_pin_graph_and_run_contracts() -> None:
    scripts = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    run_names = [_assignment(script, "RUN_NAME") for script in scripts]

    assert len(run_names) == len(set(run_names)) == 33
    for script in scripts:
        assert _assignment(script, "WARMUP_STEPS") == "3"
        assert _assignment(script, "CACHE_CAPACITY") == "2"
        assert _assignment(script, "MAX_PACKED_SEQS") == "16"
        assert _assignment(script, "CHECKPOINTING_ENABLED") == "false"
        assert _assignment(script, "WANDB_PROJECT") == "sna-cg-study"
        assert (
            'bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"'
            in script.read_text()
        )

    for script in scripts[1:]:
        assert _assignment(script, "CUDA_GRAPH_IMPL") == "transformer_engine"


def test_moe_configuration_variants_are_persistent_and_not_graph_scopes() -> None:
    variants = sorted(
        script
        for script in (EXPERIMENT_DIR / "variants").glob("*.sh")
        if script.name != "attn_mamba_router_preprocess_overlap_false.sh"
    )
    expected = {
        (scope, overlap, moe_act)
        for scope in (
            ("moe",),
            ("moe_router", "moe_preprocess"),
        )
        for overlap in ("false", "true")
        for moe_act in ("false", "true")
    }
    actual = {
        (
            _scope(script),
            _assignment(script, "MOE_SHARED_EXPERT_OVERLAP"),
            _assignment(script, "MOE_ACT_RECOMPUTE"),
        )
        for script in variants
    }

    assert len(variants) == 8
    assert actual == expected
    assert len({_assignment(script, "RUN_NAME") for script in variants}) == 8
    for script in variants:
        assert "moe_act" not in _scope(script)
        assert "shared_expert" not in _scope(script)
        assert _assignment(script, "CHECKPOINTING_ENABLED") == "false"


@pytest.mark.parametrize(
    "launcher",
    [
        "pairs/00_drop_pad_baseline_no_cg.sh",
        "pairs/01_drop_pad_moe.sh",
    ],
)
def test_drop_pad_moe_pair_fails_before_scheduler(launcher: str) -> None:
    """Packed drop-and-pad experiments must not consume nodes and then deadlock."""
    result = _run_script(launcher, CLUSTER="ptyche", TEST_ONLY="1")

    assert result.returncode == 2
    assert "sequence packing with drop-and-pad MoE is unsupported" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_dropless_moe_launchers_do_not_enable_drop_and_pad() -> None:
    """Recommended partial scopes must retain the ordinary dropless routing path."""
    for launcher in (
        "scopes/00_baseline_no_cg.sh",
        "scopes/02_moe.sh",
    ):
        result = _run_script(launcher, CLUSTER="ptyche", TEST_ONLY="1")

        assert result.returncode == 0, result.stderr
        assert "moe_expert_capacity_factor" not in result.stdout
        assert "moe_pad_expert_input_to_capacity" not in result.stdout


def test_drop_pad_moe_pair_rejects_non_nano_recipes_before_scheduler() -> None:
    result = _run_script(
        "pairs/01_drop_pad_moe.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-235b-a22b",
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "Drop-pad MoE pair requires MODEL=nano-hybrid" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_test_only_reports_resolved_ptyche_nano_provenance_and_never_submits() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "UNRESOLVED: none" in result.stdout
    assert "COMMAND:" in result.stdout
    assert "SBATCH:" in result.stdout
    assert (
        "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
        in result.stdout
    )
    assert "+policy.megatron_cfg.cuda_graph_modules=\\[attn\\]" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_warmup_steps=3" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_max_cached_schedules=2" in result.stdout
    assert "+policy.megatron_cfg.cuda_graph_max_packed_seqs=16" in result.stdout
    assert "checkpointing.enabled=false" in result.stdout
    assert (
        "+checkpointing.pretrained_checkpoint.path="
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-cg/checkpoints/"
        "nanov3-30b-a3b-pr5672-20260720/nvidia/"
        "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in result.stdout
    )
    assert "+checkpointing.pretrained_checkpoint.format=megatron_lm" in result.stdout
    assert "logger.wandb.project=sna-cg-study" in result.stdout
    assert f"MCORE_DRIVER_PYTHON: {MCORE_DRIVER_PYTHON}" in result.stdout
    assert f"MCORE_LOCK_BLOB: {MCORE_LOCK_BLOB}" in result.stdout
    assert (
        "IMMUTABLE_RUNTIME_PYTHONPATH: " + ":".join(IMMUTABLE_RUNTIME_ARCHIVES)
        in result.stdout
    )
    assert "NEMO_RL_REQUIRE_SYSTEM_MCORE=1" in result.stdout
    assert "NRL_FORCE_REBUILD_VENVS=true" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_ptyche_performance_launcher_uses_task6_immutable_mcore_runtime() -> None:
    profile = (EXPERIMENT_DIR / "profiles" / "ptyche.env").read_text()
    launcher = (EXPERIMENT_DIR / "run_scope.sh").read_text()
    registry = (
        EXPERIMENT_DIR.parents[2]
        / "nemo_rl"
        / "distributed"
        / "ray_actor_environment_registry.py"
    ).read_text()

    assert f"MCORE_LOCK_BLOB={MCORE_LOCK_BLOB}" in profile
    assert MCORE_DRIVER_PYTHON in profile
    assert "RUNTIME_ARCHIVE_PREFIX=" + ":".join(IMMUTABLE_RUNTIME_ARCHIVES) in profile
    assert "uv pip install" not in profile
    assert "pip install" not in profile
    assert "/opt/nemo_rl_venv/bin/python" not in profile
    assert "transformer_engine_torch*" in profile
    assert r"for archive in \${RUNTIME_ARCHIVE_PREFIX//:/ }" in profile
    assert 'git hash-object uv.lock)\\" = \\"${MCORE_LOCK_BLOB}' in profile

    archive_prefix = ":".join(IMMUTABLE_RUNTIME_ARCHIVES)
    assert archive_prefix in profile
    assert "${RUNTIME_ARCHIVE_PREFIX}:${REPO_ROOT}:" in launcher
    assert 'RUNTIME_ARCHIVE_PREFIX="${RUNTIME_ARCHIVE_PREFIX}"' in launcher
    assert "NEMO_RL_REQUIRE_SYSTEM_MCORE=1" in launcher
    assert "NEMO_RL_MCORE_SYSTEM_PYTHON=${MCORE_DRIVER_PYTHON}" in launcher
    assert "NRL_FORCE_REBUILD_VENVS=true" in launcher
    assert "/opt/nemo_rl_venv/bin/python" not in launcher

    assert "NEMO_RL_REQUIRE_SYSTEM_MCORE" in registry
    assert "NEMO_RL_MCORE_SYSTEM_PYTHON" in registry
    assert "MCORE_EXECUTABLE" in registry
    assert "PY_EXECUTABLES.SYSTEM" in registry


def test_ptyche_performance_launcher_uses_validated_native_te_runtime() -> None:
    profile = (EXPERIMENT_DIR / "profiles" / "ptyche.env").read_text()
    launcher = (EXPERIMENT_DIR / "run_scope.sh").read_text()

    assert f"TE_NATIVE_COMMIT={TE_NATIVE_COMMIT}" in profile
    assert f"TE_NATIVE_WHEEL_SHA256={TE_NATIVE_WHEEL_SHA256}" in profile
    assert f"TE_NATIVE_RUNTIME={TE_NATIVE_RUNTIME}" in profile
    assert f"RUNTIME_ARCHIVE_PREFIX={':'.join(IMMUTABLE_RUNTIME_ARCHIVES)}" in profile
    assert "TE_FP64_WEAKREF_SOURCE=" not in profile
    assert "TE_FP64_WEAKREF_TARGET=" not in profile
    assert "MOUNTS=/lustre:/lustre" in profile
    assert "validate_te_native_runtime.py" in profile

    assert "TE_NATIVE_COMMIT" in launcher
    assert "TE_NATIVE_WHEEL_SHA256" in launcher
    assert "TE_NATIVE_PROVENANCE" in launcher
    assert "Transformer Engine native runtime provenance mismatch" in launcher
    assert "TE_FP64_WEAKREF_SOURCE" not in launcher
    assert "TE_FP64_WEAKREF_TARGET" not in launcher


def test_combined_correctness_first_performance_variant_is_persistent() -> None:
    launcher = (
        EXPERIMENT_DIR
        / "variants"
        / "attn_mamba_router_preprocess_overlap_false.sh"
    )

    assert launcher.is_file()
    assert _scope(launcher) == (
        "attn",
        "mamba",
        "moe_router",
        "moe_preprocess",
    )
    assert _assignment(launcher, "MOE_SHARED_EXPERT_OVERLAP") == "false"
    assert _assignment(launcher, "MOE_ACT_RECOMPUTE") == "false"


def test_real_submission_fails_before_sbatch_when_runtime_lock_mismatches(
    tmp_path: Path,
) -> None:
    archives = tuple(tmp_path / f"archive-{index}" for index in range(6))
    for archive in archives:
        archive.mkdir()
    driver = tmp_path / "locked-mcore" / "bin" / "python"
    driver.parent.mkdir(parents=True)
    driver.symlink_to(
        "/root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13"
    )
    container = tmp_path / "nightly.sqsh"
    container.touch()
    provenance = _create_test_native_runtime(archives[0], container)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    record = tmp_path / "sbatch-invoked"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf invoked > "${SBATCH_RECORD:?}"\n'
    )
    fake_sbatch.chmod(0o755)

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        SBATCH_TEST_ONLY="1",
        SBATCH_RECORD=str(record),
        LOG_ROOT_OVERRIDE=str(tmp_path / "logs"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        LAUNCHER_TEST_CONTRACT_OVERRIDE="1",
        MCORE_DRIVER_PYTHON_OVERRIDE=str(driver),
        MCORE_LOCK_BLOB_OVERRIDE="0" * 40,
        RUNTIME_ARCHIVE_PREFIX_OVERRIDE=":".join(map(str, archives)),
        TE_NATIVE_PROVENANCE_OVERRIDE=str(provenance),
        CONTAINER_OVERRIDE=str(container),
    )

    assert result.returncode == 2
    assert "uv.lock hash mismatch" in result.stderr
    assert not record.exists()


def test_runtime_contract_override_is_rejected_outside_scheduler_test_mode(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    record = tmp_path / "sbatch-invoked"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf invoked > "${SBATCH_RECORD:?}"\n'
    )
    fake_sbatch.chmod(0o755)

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        LAUNCHER_TEST_CONTRACT_OVERRIDE="1",
        SBATCH_RECORD=str(record),
        LOG_ROOT_OVERRIDE=str(tmp_path / "logs"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode == 2
    assert "only allowed with SBATCH_TEST_ONLY=1" in result.stderr
    assert not record.exists()


def test_pre_sbatch_runtime_contract_accepts_verified_test_artifacts(
    tmp_path: Path,
) -> None:
    archives = tuple(tmp_path / f"archive-{index}" for index in range(6))
    for archive in archives:
        archive.mkdir()
    driver = tmp_path / "locked-mcore" / "bin" / "python"
    driver.parent.mkdir(parents=True)
    driver.symlink_to(
        "/root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13"
    )
    container = tmp_path / "nightly.sqsh"
    container.touch()
    provenance = _create_test_native_runtime(archives[0], container)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    record = tmp_path / "sbatch-argv"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${SBATCH_RECORD:?}"\n'
        "printf '2476001;ptyche\\n'\n"
    )
    fake_sbatch.chmod(0o755)
    lock_blob = subprocess.run(
        ["git", "hash-object", "uv.lock"],
        check=True,
        cwd=EXPERIMENT_DIR.parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        SBATCH_TEST_ONLY="1",
        SBATCH_RECORD=str(record),
        LOG_ROOT_OVERRIDE=str(tmp_path / "logs"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        LAUNCHER_TEST_CONTRACT_OVERRIDE="1",
        MCORE_DRIVER_PYTHON_OVERRIDE=str(driver),
        MCORE_LOCK_BLOB_OVERRIDE=lock_blob,
        RUNTIME_ARCHIVE_PREFIX_OVERRIDE=":".join(map(str, archives)),
        TE_NATIVE_PROVENANCE_OVERRIDE=str(provenance),
        CONTAINER_OVERRIDE=str(container),
    )

    assert result.returncode == 0, result.stderr
    assert "RUNTIME_PREFLIGHT: passed" in result.stdout
    assert record.exists()


def test_pre_sbatch_runtime_contract_rejects_arbitrary_interpreter_symlink(
    tmp_path: Path,
) -> None:
    archives = tuple(tmp_path / f"archive-{index}" for index in range(6))
    for archive in archives:
        archive.mkdir()
    driver = tmp_path / "locked-mcore" / "bin" / "python"
    driver.parent.mkdir(parents=True)
    driver.symlink_to("/usr/bin/python3")
    container = tmp_path / "nightly.sqsh"
    container.touch()
    provenance = _create_test_native_runtime(archives[0], container)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    record = tmp_path / "sbatch-invoked"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf invoked > "${SBATCH_RECORD:?}"\n'
    )
    fake_sbatch.chmod(0o755)
    lock_blob = subprocess.run(
        ["git", "hash-object", "uv.lock"],
        check=True,
        cwd=EXPERIMENT_DIR.parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        SBATCH_TEST_ONLY="1",
        SBATCH_RECORD=str(record),
        LOG_ROOT_OVERRIDE=str(tmp_path / "logs"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        LAUNCHER_TEST_CONTRACT_OVERRIDE="1",
        MCORE_DRIVER_PYTHON_OVERRIDE=str(driver),
        MCORE_LOCK_BLOB_OVERRIDE=lock_blob,
        RUNTIME_ARCHIVE_PREFIX_OVERRIDE=":".join(map(str, archives)),
        TE_NATIVE_PROVENANCE_OVERRIDE=str(provenance),
        CONTAINER_OVERRIDE=str(container),
    )

    assert result.returncode == 2
    assert "unusable symlink target: /usr/bin/python3" in result.stderr
    assert not record.exists()


def test_real_submission_uses_parsable_sbatch_job_ids_without_dependencies() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    sbatch_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SBATCH:")
    )
    assert "--parsable" in sbatch_line
    assert "--dependency" not in sbatch_line


def test_sbatch_test_only_executes_the_exact_parsable_scheduler_command(
    tmp_path: Path,
) -> None:
    archives = tuple(tmp_path / f"archive-{index}" for index in range(6))
    for archive in archives:
        archive.mkdir()
    driver = tmp_path / "locked-mcore" / "bin" / "python"
    driver.parent.mkdir(parents=True)
    driver.symlink_to(
        "/root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin/python3.13"
    )
    container = tmp_path / "nightly.sqsh"
    container.touch()
    provenance = _create_test_native_runtime(archives[0], container)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    sbatch_record = tmp_path / "sbatch-argv"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${SBATCH_RECORD:?}"\n'
        "printf '2476000;ptyche\\n'\n"
    )
    fake_sbatch.chmod(0o755)
    lock_blob = subprocess.run(
        ["git", "hash-object", "uv.lock"],
        check=True,
        cwd=EXPERIMENT_DIR.parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        SBATCH_TEST_ONLY="1",
        SBATCH_RECORD=str(sbatch_record),
        LOG_ROOT_OVERRIDE=str(tmp_path / "logs"),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        LAUNCHER_TEST_CONTRACT_OVERRIDE="1",
        MCORE_DRIVER_PYTHON_OVERRIDE=str(driver),
        MCORE_LOCK_BLOB_OVERRIDE=lock_blob,
        RUNTIME_ARCHIVE_PREFIX_OVERRIDE=":".join(map(str, archives)),
        TE_NATIVE_PROVENANCE_OVERRIDE=str(provenance),
        CONTAINER_OVERRIDE=str(container),
    )

    assert result.returncode == 0, result.stderr
    assert "--parsable" in result.stdout
    assert "--test-only" in result.stdout
    assert "SLURM_JOB_ID: 2476000;ptyche" in result.stdout
    submitted_argv = sbatch_record.read_text().splitlines()
    assert "--parsable" in submitted_argv
    assert "--test-only" in submitted_argv
    assert "--dependency" not in submitted_argv
    assert "--nodes=8" in submitted_argv
    assert submitted_argv[-1] == "ray.sub"


@pytest.mark.parametrize(
    "launcher",
    ["scopes/00_baseline_no_cg.sh", "scopes/17_attn.sh"],
)
def test_native_te_provenance_is_identical_for_baseline_and_te_scopes(
    launcher: str,
) -> None:
    result = _run_script(launcher, CLUSTER="ptyche", TEST_ONLY="1")

    assert result.returncode == 0, result.stderr
    for field, value in (
        ("TE_NATIVE_COMMIT", TE_NATIVE_COMMIT),
        ("TE_NATIVE_WHEEL_SHA256", TE_NATIVE_WHEEL_SHA256),
        ("TE_NATIVE_RUNTIME", TE_NATIVE_RUNTIME),
        ("TE_NATIVE_PROVENANCE", f"{TE_NATIVE_RUNTIME}/provenance.json"),
        ("TE_NATIVE_SITE_PACKAGES", IMMUTABLE_RUNTIME_ARCHIVES[0]),
        ("TE_EXPECTED_VERSION", "2.15.0+4a18653f"),
    ):
        assert f"{field}: {value}" in result.stdout
    assert "MOUNTS: /lustre:/lustre" in result.stdout
    assert "validate_te_native_runtime.py" in result.stdout
    assert "--expected-version 2.15.0+4a18653f" in result.stdout
    assert f"--expected-wheel-sha256 {TE_NATIVE_WHEEL_SHA256}" in result.stdout


def test_fp64_overlay_rejects_wrong_sha_before_cuda_preflight(
    tmp_path: Path,
) -> None:
    validator_path = EXPERIMENT_DIR / "validate_te_fp64_overlay.py"
    assert validator_path.is_file(), "FP64 overlay validator must be committed"

    package_root = tmp_path / "packages"
    transformer_engine_root = package_root / "transformer_engine"
    pytorch_root = transformer_engine_root / "pytorch"
    pytorch_root.mkdir(parents=True)
    (package_root / "torch.py").write_text(
        "float64 = object()\n"
        "def arange(*args, **kwargs):\n"
        "    raise RuntimeError('CUDA preflight ran')\n"
    )
    (transformer_engine_root / "__init__.py").write_text(
        f"__version__ = {TE_EXPECTED_VERSION!r}\n"
    )
    (pytorch_root / "__init__.py").touch()
    (pytorch_root / "utils.py").write_text(
        "import torch\n"
        "_torch_dtype_to_np_typestr_dict = {torch.float64: '<f8'}\n"
        "def make_weak_ref(tensor):\n"
        "    return tensor\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [
            sys.executable,
            str(validator_path),
            "--expected-version",
            TE_EXPECTED_VERSION,
            "--expected-sha256",
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert "CUDA preflight ran" not in result.stderr


def test_fp64_overlay_rejects_writable_source_before_cuda_preflight(
    tmp_path: Path,
) -> None:
    validator_path = EXPERIMENT_DIR / "validate_te_fp64_overlay.py"
    package_root = tmp_path / "packages"
    transformer_engine_root = package_root / "transformer_engine"
    pytorch_root = transformer_engine_root / "pytorch"
    pytorch_root.mkdir(parents=True)
    (package_root / "torch.py").write_text(
        "float64 = object()\n"
        "def arange(*args, **kwargs):\n"
        "    raise RuntimeError('CUDA preflight ran')\n"
    )
    (transformer_engine_root / "__init__.py").write_text(
        f"__version__ = {TE_EXPECTED_VERSION!r}\n"
    )
    (pytorch_root / "__init__.py").touch()
    utils_path = pytorch_root / "utils.py"
    utils_path.write_text(
        "import torch\n"
        "_torch_dtype_to_np_typestr_dict = {torch.float64: '<f8'}\n"
        "def make_weak_ref(tensor):\n"
        "    return tensor\n"
    )
    utils_path.chmod(0o644)
    expected_sha256 = hashlib.sha256(utils_path.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [
            sys.executable,
            str(validator_path),
            "--expected-version",
            TE_EXPECTED_VERSION,
            "--expected-sha256",
            expected_sha256,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "mode mismatch" in result.stderr
    assert "CUDA preflight ran" not in result.stderr


@pytest.mark.parametrize("phase", ["profile", "benchmark"])
def test_runner_rejects_invalid_phases(phase: str) -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        PHASE=phase,
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "PHASE must be smoke, performance, or accuracy" in result.stderr


def test_runner_rejects_unknown_cluster_profiles() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="unknown",
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "CLUSTER must be ptyche or oci-hsg" in result.stderr


def test_runner_rejects_qwen_mamba_scopes_before_submission() -> None:
    result = _run_script(
        "scopes/05_mamba.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-30b-a3b",
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert "has no Mamba layers" in result.stderr
    assert "SBATCH:" not in result.stdout


@pytest.mark.parametrize(
    ("model", "recipe"),
    [
        (
            "qwen3-30b-a3b",
            "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        ),
        (
            "qwen3-235b-a22b",
            "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml",
        ),
    ],
)
def test_qwen_profiles_accept_non_mamba_scopes(model: str, recipe: str) -> None:
    result = _run_script(
        "scopes/02_moe.sh",
        CLUSTER="oci-hsg",
        MODEL=model,
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert recipe in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_qwen235_profile_leaves_unverified_noncolocated_geometry_unresolved() -> None:
    result = _run_script(
        "scopes/02_moe.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-235b-a22b",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    unresolved = result.stdout.splitlines()[0]
    assert "TOTAL_NODES" in unresolved
    assert "INFERENCE_NODES" in unresolved


def test_real_mode_fails_on_unresolved_provenance_before_sbatch() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        MODEL="qwen3-235b-a22b",
        TEST_ONLY="0",
    )

    assert result.returncode == 2
    assert "SBATCH:" in result.stdout
    assert "Refusing submission with unresolved fields" in result.stderr


def test_variant_command_uses_configuration_not_graph_scope() -> None:
    result = _run_script(
        "variants/router_preprocess_overlap_true_moe_act_true.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "+policy.megatron_cfg.cuda_graph_modules=\\[moe_router\\,moe_preprocess\\]"
        in result.stdout
    )
    assert "+policy.megatron_cfg.moe_shared_expert_overlap=true" in result.stdout
    assert "+policy.megatron_cfg.activation_checkpointing=true" in result.stdout
    assert "+policy.megatron_cfg.recompute_granularity=selective" in result.stdout
    assert "+policy.megatron_cfg.recompute_modules=\\[moe_act\\]" in result.stdout


def test_submit_all_smokes_reuses_every_persistent_launcher() -> None:
    result = _run_script(
        "submit_all_smokes.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="unit-test",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TEST_ONLY: no submission performed") == 42
    assert result.stdout.count("Submitting smoke launcher:") == 42


def test_submit_performance_accepts_explicit_reusable_selection() -> None:
    result = _run_script(
        "submit_performance.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="unit-test",
        PERFORMANCE_SCRIPTS="scopes/00_baseline_no_cg.sh scopes/01_whole_layer.sh",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TEST_ONLY: no submission performed") == 2
    assert result.stdout.count("Submitting performance launcher:") == 2
    assert "baseline-no-cg" in result.stdout
    assert "whole-layer" in result.stdout


def test_submit_performance_rejects_drop_pad_pair_before_scheduler() -> None:
    result = _run_script(
        "submit_performance.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="unit-test",
        PERFORMANCE_SCRIPTS=(
            "pairs/00_drop_pad_baseline_no_cg.sh pairs/01_drop_pad_moe.sh"
        ),
    )

    assert result.returncode == 2
    assert "sequence packing with drop-and-pad MoE is unsupported" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_native_te_20step_comparison_reuses_exact_matched_launchers() -> None:
    result = _run_script(
        "submit_20step_native_te_comparison.sh",
        CLUSTER="ptyche",
        TEST_ONLY="1",
        RUN_TAG="native-te-unit-test",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("TEST_ONLY: no submission performed") == 6
    assert result.stdout.count("Submitting performance launcher:") == 6
    assert result.stdout.count("grpo.max_num_steps=20") == 6
    assert result.stdout.count("checkpointing.enabled=false") >= 6
    assert result.stdout.count("native-te-unit-test") > 6
    for launcher in (
        "scopes/00_baseline_no_cg.sh",
        "scopes/17_attn.sh",
        "scopes/05_mamba.sh",
        "scopes/03_moe_router.sh",
        "variants/router_preprocess_overlap_false_moe_act_false.sh",
        "variants/attn_mamba_router_preprocess_overlap_false.sh",
    ):
        assert f"Submitting performance launcher: {launcher}" in result.stdout
    assert "drop-pad-baseline-no-cg" not in result.stdout
    assert "drop-pad-moe" not in result.stdout


def test_nemorl_integration_gate_uses_bridge_src_layout() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src" in script


def test_nemorl_integration_gate_uses_batch_partition() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --partition=backfill" not in script


def test_nemorl_integration_gate_uses_official_mcore_environment() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert "NRL_FORCE_REBUILD_VENVS" not in script
    assert "uv run" not in script
    assert (
        "MCORE_VENV_KEY=py313-aarch64-lock-${EXPECTED_LOCK_BLOB}-"
        "image-cb8ae0ade02b" in script
    )
    assert (
        "MCORE_VENV=${ROOT}/venvs/nemorl-integration-mcore-${MCORE_VENV_KEY}" in script
    )
    assert "export UV_PROJECT_ENVIRONMENT=${MCORE_VENV}" in script
    sync_index = script.index("uv sync --frozen --extra mcore")
    wrapper_command = (
        "\\${UV_PROJECT_ENVIRONMENT}/bin/python "
        "experiments/cuda_graph/mamba_moe_te_graph_20260729/"
        "run_pytest_with_te_overlay.py"
    )
    first_wrapper_index = script.index(wrapper_command)
    sync_block = script[sync_index:first_wrapper_index]
    for sync_flag in (
        "--group test",
        "--no-install-project",
        "--no-install-local",
        "--python /opt/nemo_rl_venv/bin/python",
        "--no-python-downloads",
    ):
        assert sync_flag in sync_block
    assert re.search(r"(?m)^\s*--no-build(?:\s|\\|$)", sync_block) is None
    assert sync_block.count("--no-install-package transformer-engine") == 1
    assert "--no-install-package transformer-engine-torch" not in sync_block
    assert "--no-build-package" not in sync_block
    assert sync_block.count("--group test") == 1
    assert "uv tool install" not in sync_block
    assert "pip install" not in sync_block
    post_sync_te_check = script.index("Unexpected Transformer Engine artifact in venv")
    assert sync_index < post_sync_te_check < first_wrapper_index
    assert "sysconfig.get_path('purelib')" in script
    assert "transformer_engine*.dist-info" in script
    last_wrapper_index = script.rindex(wrapper_command)
    flock_index = script.index("flock -x 9")
    venv_parent_creation = 'mkdir -p "$(dirname "${MCORE_VENV}")"'
    assert venv_parent_creation in script
    assert (
        script.index(venv_parent_creation)
        < script.index("srun --nodes=1 --ntasks=1")
        < script.index('exec 9>\\"\\${UV_PROJECT_ENVIRONMENT}.lock\\"')
    )
    assert flock_index < sync_index < first_wrapper_index < last_wrapper_index
    assert 'exec 9>\\"\\${UV_PROJECT_ENVIRONMENT}.lock\\"' in script
    assert script.count(wrapper_command) == 2
    assert script.count("/opt/nemo_rl_venv/bin/python") == 1
    assert "export NVTE_CUDA_ARCHS=100" in script
    assert "#SBATCH --time=01:00:00" in script


def test_nemorl_integration_gate_uses_the_validated_immutable_runtime_archives() -> (
    None
):
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    archives = (
        ("TE_ARCHIVE", "/root/.cache/uv/archive-v0/AdbVCNRp6JVFPo0e"),
        ("FLASH_ATTN_ARCHIVE", "/root/.cache/uv/archive-v0/26H_iFoUOK00pyG5"),
        ("ML_DTYPES_ARCHIVE", "/root/.cache/uv/archive-v0/ymbKBYrUysuiERDQ"),
        ("ONNX_ARCHIVE", "/root/.cache/uv/archive-v0/Lp_mVBWGrC-sLPL6"),
        ("ONNX_IR_ARCHIVE", "/root/.cache/uv/archive-v0/kIpfdwf26Al4-BTb"),
        ("ONNXSCRIPT_ARCHIVE", "/root/.cache/uv/archive-v0/i7-d_jifMXRoKKrY"),
    )
    for name, path in archives:
        assert f"{name}={path}" in script

    assert (
        "IMMUTABLE_RUNTIME_PYTHONPATH=${TE_ARCHIVE}:${FLASH_ATTN_ARCHIVE}:"
        "${ML_DTYPES_ARCHIVE}:${ONNX_ARCHIVE}:${ONNX_IR_ARCHIVE}:"
        "${ONNXSCRIPT_ARCHIVE}" in script
    )
    assert (
        "export PYTHONPATH=${IMMUTABLE_RUNTIME_PYTHONPATH}:\\${PWD}:"
        "\\${PWD}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:"
        "\\${PWD}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/"
        "3rdparty/Megatron-LM" in script
    )
    for native_build_command in (
        "uv run",
        "NRL_FORCE_REBUILD_VENVS",
        "pip install",
        "setup.py build",
    ):
        assert native_build_command not in script
    assert script.count("--no-install-package transformer-engine") == 1
    assert "--no-install-package transformer-engine-torch" not in script
    assert "--no-build-package" not in script


def test_nemorl_integration_gate_renders_the_post_sync_venv_check_safely() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    marker = "\\${UV_PROJECT_ENVIRONMENT}/bin/python - <<'PY'\n"
    _, found_marker, remainder = script.partition(marker)
    assert found_marker
    python_source, found_terminator, _ = remainder.partition("\nPY\n")
    assert found_terminator
    assert '"' not in python_source

    rendered_command = "bash -lc \"\npython3 - <<'PY'\n" + python_source + '\nPY\n"'
    result = subprocess.run(
        ["bash", "-c", rendered_command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_nemorl_integration_gate_pins_clean_runner_lock_and_image_provenance() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    lock_blob = "96543608420ac6746cfd18d1fcd8ee1bd3c91caf"
    image_sha256 = "cb8ae0ade02b876f1b3380c8375eb92f95033dece6b2bfdc678b47f2da1aea91"

    assert "REPO=${ROOT}/src/RL-pr5672-mamba-moe-graph-cache-runner-20260730" in script
    assert "RL-pr5672-mamba-moe-graph-cache-20260729-d7f1d496f" not in script
    assert "IMAGE=${ROOT}/containers/nemo_rl_nightly_20260729_2472184.sqsh" in script
    assert "CONTAINER" not in script
    assert f"IMAGE_SHA256={image_sha256}" in script
    image_check = (
        'test "$(sha256sum -- "${IMAGE}" | awk \'{print $1}\')" = "${IMAGE_SHA256}"'
    )
    assert image_check in script
    assert script.index(image_check) < script.index("srun --nodes=1 --ntasks=1")
    assert f"EXPECTED_LOCK_BLOB={lock_blob}" in script
    assert (
        r"test \"\$(git rev-parse ${EXPECTED_SHA}:uv.lock)\" = ${EXPECTED_LOCK_BLOB}"
        in script
    )
    assert r"test \"\$(git hash-object uv.lock)\" = ${EXPECTED_LOCK_BLOB}" in script
    assert script.count("EXPECTED_LOCK_BLOB") >= 3
    assert (
        "git diff --quiet --ignore-submodules=dirty -- . ':(exclude)uv.lock'" in script
    )
    assert "git diff --cached --quiet" in script


def test_nemorl_integration_repo_override_is_real_and_confined_before_srun(
    tmp_path: Path,
) -> None:
    src_root = tmp_path / "src"
    clean_runner = src_root / "RL-pr5672-clean-runner"
    clean_runner.mkdir(parents=True)
    outside = tmp_path / "outside-runner"
    outside.mkdir()

    default = _run_integration_repo_preflight(tmp_path)
    assert default.returncode == 0, default.stderr
    assert (
        "REPO=" + str(src_root / "RL-pr5672-mamba-moe-graph-cache-runner-20260730")
        in default.stdout
    )

    valid = _run_integration_repo_preflight(tmp_path, str(clean_runner))
    assert valid.returncode == 0, valid.stderr
    assert f"REPO={clean_runner.resolve()}" in valid.stdout

    relative = _run_integration_repo_preflight(tmp_path, "src/RL-pr5672-clean-runner")
    assert relative.returncode == 2
    assert "REPO_OVERRIDE must be an absolute path" in relative.stderr

    for invalid_path in (
        str(clean_runner / ".." / ".." / outside.name),
        str(outside),
    ):
        invalid = _run_integration_repo_preflight(tmp_path, invalid_path)
        assert invalid.returncode == 2
        assert "REPO_OVERRIDE must resolve below" in invalid.stderr
        assert "srun" not in invalid.stdout

    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    assert script.index('if [[ -n "${REPO_OVERRIDE:-}" ]]') < script.index(
        "srun --nodes=1 --ntasks=1"
    )


def test_nemorl_integration_gate_allows_only_generated_unit_result_json_files() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    generated_result = re.compile(r"^tests/unit/unit_results/[^/]+\.json$")

    assert generated_result.fullmatch("tests/unit/unit_results/20260729_214835.json")
    assert generated_result.fullmatch("tests/unit/unit_results/summary.json")
    for path in (
        "tests/unit/unit_results.py",
        "tests/unit/unit_results/summary.py",
        "tests/unit/unit_results/nested/summary.json",
        "experiments/untracked.py",
    ):
        assert generated_result.fullmatch(path) is None

    assert "while IFS= read -r untracked_path; do" in script
    assert (
        "git ls-files --others --exclude-standard | while IFS= read -r untracked_path; do"
        in script
    )
    assert "done < <(git ls-files --others --exclude-standard)" not in script
    assert "tests/unit/unit_results.json" in script
    assert "^tests/unit/unit_results/[^/]+\\.json$" in script
    assert "Unexpected untracked path:" in script
    assert 'test -z "\\$(git ls-files --others --exclude-standard)"' not in script

    producer_failure = _run_untracked_path_guard("false")
    assert producer_failure.returncode != 0

    allowed_paths = _run_untracked_path_guard(
        "printf '%s\\n' tests/unit/unit_results.json "
        "tests/unit/unit_results/20260729_214835.json"
    )
    assert allowed_paths.returncode == 0, allowed_paths.stderr

    unexpected_path = _run_untracked_path_guard(
        "printf '%s\\n' experiments/untracked.py"
    )
    assert unexpected_path.returncode != 0
    assert (
        "Unexpected untracked path: experiments/untracked.py" in unexpected_path.stderr
    )


def test_nemorl_integration_gate_validates_fp64_overlay_and_mcore_graph_suite() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()

    assert (
        "TE_FP64_WEAKREF_SOURCE=${ROOT}/src/TransformerEngine-fp64-weakref-20260729/"
        f".overlay/{TE_FP64_WEAKREF_COMMIT}/utils.py" in script
    )
    assert f"TE_FP64_WEAKREF_TARGET={TE_FP64_WEAKREF_TARGET}" in script
    assert (
        "--container-mounts=/lustre:/lustre,${TE_FP64_WEAKREF_SOURCE}:"
        "${TE_FP64_WEAKREF_TARGET}:ro" in script
    )
    assert "run_pytest_with_te_overlay.py" in script
    assert "tests/unit/distributed/test_mcore_system_executable_contract.py" in script
    assert f"TE_EXPECTED_VERSION={TE_EXPECTED_VERSION}" in script
    assert "--expected-version ${TE_EXPECTED_VERSION}" in script
    assert f"TE_FP64_WEAKREF_SHA256={TE_FP64_WEAKREF_SHA256}" in script
    assert "--expected-sha256 ${TE_FP64_WEAKREF_SHA256}" in script
    assert (
        "MCORE_ROOT=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
        in script
    )
    for test_path in (
        "tests/unit_tests/transformer/test_packed_seq_params_cuda_graph.py",
        "tests/unit_tests/transformer/test_te_cuda_graph_bank.py",
        "tests/unit_tests/transformer/test_cuda_graphs.py::test_moe_router_fp64_output_is_preserved_at_te_graph_boundary",
        "tests/unit_tests/transformer/test_cuda_graphs.py::test_packed_mamba_te_cuda_graph_parity",
        "tests/unit_tests/transformer/test_cuda_graphs.py::test_te_graph_bank_schedule_switch_5_3_5",
    ):
        assert f"${{MCORE_ROOT}}/{test_path}" in script
    assert "tests/unit/models/megatron/test_cuda_graph_lifecycle.py" in script


def test_nemorl_integration_gate_runs_required_mcore_targets_distributed() -> None:
    script = (
        EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    ).read_text()
    distributed_command = (
        "\\${UV_PROJECT_ENVIRONMENT}/bin/python -m torch.distributed.run"
    )
    distributed_index = script.index(distributed_command)
    mcore_root_index = script.index("MCORE_ROOT=")
    single_rank_mcore_block = script[mcore_root_index:distributed_index]
    distributed_block = script[distributed_index:]
    distributed_targets = (
        "${MCORE_ROOT}/tests/unit_tests/transformer/"
        "test_cuda_graphs.py::test_packed_mamba_te_cuda_graph_parity",
        "${MCORE_ROOT}/tests/unit_tests/transformer/"
        "test_cuda_graphs.py::test_te_graph_bank_schedule_switch_5_3_5",
    )

    assert "--nproc_per_node 2" in distributed_block
    assert "#SBATCH --segment=1" in script
    assert "#SBATCH --gpus" not in script
    assert "#SBATCH --gres" not in script
    assert "--nnodes 1" in distributed_block
    assert "--master_addr localhost" in distributed_block
    assert "--node_rank 0" in distributed_block
    assert r"--log-dir \"${MCORE_DISTRIBUTED_LOG_DIR}\"" in distributed_block
    assert r"--tee \"0:3\"" in distributed_block
    assert r"--redirects \"3\"" in distributed_block
    assert "run_pytest_with_te_overlay.py" in distributed_block
    assert " -m pytest" not in distributed_block
    assert "-rs" in distributed_block
    assert (
        r"rank_stdout=\$(find \"${MCORE_DISTRIBUTED_LOG_DIR}\" -type f "
        r"-path \"*/\${rank}/stdout.log\" -print -quit)"
    ) in distributed_block
    assert "for rank in 0 1; do" in distributed_block
    assert "Missing distributed MCore stdout for rank" in distributed_block
    assert r"grep -Eq \"^2 passed([,[:space:]]|$)\"" in distributed_block
    assert r"grep -Eiq \"SKIPPED|[0-9]+ skipped\"" in distributed_block
    assert "Distributed MCore focused target skipped on rank" in distributed_block
    for target in distributed_targets:
        assert target not in single_rank_mcore_block
        assert distributed_block.count(target) == 1
    assert (
        "MCORE_DISTRIBUTED_LOG_DIR=${ROOT}/logs/"
        "nemorl-integration-distributed-${SLURM_JOB_ID:?}" in script
    )
    assert 'mkdir -p "${MCORE_DISTRIBUTED_LOG_DIR}"' in script
    assert "mktemp" not in script
    assert "rm -rf" not in script


@pytest.mark.parametrize(
    ("rank_stdout", "expected"),
    [
        ("2 passed, 33 warnings in 73.90s\n", True),
        ("2 passed\n", True),
        ("12 passed in 1.00s\n", False),
        ("20 passed in 1.00s\n", False),
        ("2x passed in 1.00s\n", False),
        ("no tests ran in 0.01s\n", False),
        ("2 passed, 1 skipped in 1.00s\n", False),
    ],
)
def test_nemorl_integration_rank_summary_requires_exact_unskipped_pair(
    rank_stdout: str, expected: bool
) -> None:
    pass_result = subprocess.run(
        ["grep", "-Eq", r"^2 passed([,[:space:]]|$)"],
        input=rank_stdout,
        check=False,
        capture_output=True,
        text=True,
    )
    skip_result = subprocess.run(
        ["grep", "-Eiq", r"SKIPPED|[0-9]+ skipped"],
        input=rank_stdout,
        check=False,
        capture_output=True,
        text=True,
    )

    assert (pass_result.returncode == 0 and skip_result.returncode != 0) is expected


def test_nemorl_integration_gate_uses_the_same_immutable_python_for_both_suites() -> (
    None
):
    script_path = EXPERIMENT_DIR / "scripts" / "validate_nemorl_integration.sub"
    helper_path = EXPERIMENT_DIR / "run_pytest_with_te_overlay.py"
    script = script_path.read_text()

    assert helper_path.is_file()
    assert "uv run" not in script
    assert "NRL_FORCE_REBUILD_VENVS" not in script
    assert script.count("run_pytest_with_te_overlay.py") == 3
    assert "tests/unit/models/megatron/test_cuda_graph_lifecycle.py" in script
    assert (
        "${MCORE_ROOT}/tests/unit_tests/transformer/test_te_cuda_graph_bank.py"
        in script
    )


def test_overlay_pytest_wrapper_validates_before_running_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    validator = ModuleType("validate_te_fp64_overlay")

    def validate_overlay(
        *, expected_version: str, expected_sha256: str
    ) -> dict[str, str]:
        calls.append(("validate", (expected_version, expected_sha256)))
        return {}

    validator.validate_overlay = validate_overlay
    monkeypatch.setitem(sys.modules, "validate_te_fp64_overlay", validator)
    module = _load_experiment_module("run_pytest_with_te_overlay")

    def pytest_main(args: list[str]) -> int:
        calls.append(("pytest", args))
        return 7

    monkeypatch.setattr(module.pytest, "main", pytest_main)

    assert (
        module.run_pytest(
            expected_version=TE_EXPECTED_VERSION,
            expected_sha256=TE_FP64_WEAKREF_SHA256,
            pytest_args=["-q", "test_path.py"],
        )
        == 7
    )
    assert calls == [
        ("validate", (TE_EXPECTED_VERSION, TE_FP64_WEAKREF_SHA256)),
        ("pytest", ["-q", "test_path.py"]),
    ]


def test_overlay_pytest_wrapper_strips_the_launcher_delimiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = ModuleType("validate_te_fp64_overlay")
    validator.validate_overlay = lambda **_: {}
    monkeypatch.setitem(sys.modules, "validate_te_fp64_overlay", validator)
    module = _load_experiment_module("run_pytest_with_te_overlay")

    args = module.parse_args(
        [
            "--expected-version",
            TE_EXPECTED_VERSION,
            "--expected-sha256",
            TE_FP64_WEAKREF_SHA256,
            "--",
            "-q",
            "tests/unit/test_cuda_graph.py",
        ]
    )

    assert args.expected_version == TE_EXPECTED_VERSION
    assert args.expected_sha256 == TE_FP64_WEAKREF_SHA256
    assert args.pytest_args == ["-q", "tests/unit/test_cuda_graph.py"]

    forwarded_args: list[list[str]] = []
    monkeypatch.setattr(
        module.pytest,
        "main",
        lambda pytest_args: forwarded_args.append(pytest_args) or 0,
    )
    assert (
        module.run_pytest(
            expected_version=args.expected_version,
            expected_sha256=args.expected_sha256,
            pytest_args=args.pytest_args,
        )
        == 0
    )
    assert forwarded_args == [["-q", "tests/unit/test_cuda_graph.py"]]


def test_gb200_profiles_limit_transformer_engine_build_to_sm100() -> None:
    runner = (EXPERIMENT_DIR / "run_scope.sh").read_text()

    assert "/opt/nemo_rl_venv/bin/python" not in runner
    assert "uv\n  run\n" not in runner
    assert "MCORE_PYTHONPATH=" not in runner
    assert "IMMUTABLE_RUNTIME_PYTHONPATH=" in runner
    assert 'PYTHONPATH="${IMMUTABLE_RUNTIME_PYTHONPATH}" \\' in runner
    assert "NEMO_RL_MCORE_SYSTEM_PYTHON=${MCORE_DRIVER_PYTHON}" in runner
    assert 'RUNTIME_ARCHIVE_PREFIX="${RUNTIME_ARCHIVE_PREFIX}" \\' in runner
    assert "Megatron-Bridge/3rdparty/Megatron-LM" in runner
    assert "Megatron-Bridge/src" in runner
    assert 'NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" \\' in runner
    assert 'UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE}" \\' in runner
    assert 'SETUP_COMMAND="${SETUP_COMMAND}" \\' in runner
    assert 'SETUP_COMMAND_ON_WORKERS="${SETUP_COMMAND_ON_WORKERS}" \\' in runner
    assert 'RAY_CLIENT_SERVER_ENABLED="${RAY_CLIENT_SERVER_ENABLED}" \\' in runner
    assert 'RAY_DASHBOARD_ENABLED="${RAY_DASHBOARD_ENABLED}" \\' in runner
    ray_submit = (EXPERIMENT_DIR.parents[2] / "ray.sub").read_text()
    assert "__CONTAINER_LOCAL__" in ray_submit
    assert 'export UV_CACHE_DIR="/root/.cache/uv"' in ray_submit
    assert 'export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"' in ray_submit
    assert "SETUP_COMMAND_ON_WORKERS" in ray_submit
    worker_setup = ray_submit.split("# Workers retry more often", maxsplit=1)[1]
    assert (
        '[[ "$SETUP_COMMAND_ON_WORKERS" == 1 ]]'
        ' && [[ -n "$SETUP_COMMAND_FILE" ]]' in worker_setup
    )
    for cluster in ("ptyche", "oci-hsg"):
        profile = (EXPERIMENT_DIR / "profiles" / f"{cluster}.env").read_text()
        assert "NVTE_CUDA_ARCHS=100" in profile
        assert "UV_CACHE_DIR_OVERRIDE=__CONTAINER_LOCAL__" in profile
        assert "SETUP_COMMAND=" in profile
        assert "SETUP_COMMAND_ON_WORKERS=0" in profile
        assert "uv run --frozen --extra mcore" not in profile
        assert "--editable" not in profile
        assert "RAY_CLIENT_SERVER_ENABLED=0" in profile
        assert "RAY_DASHBOARD_ENABLED=0" in profile
        if cluster == "ptyche":
            assert MCORE_DRIVER_PYTHON in profile
            assert (
                "RUNTIME_ARCHIVE_PREFIX=" + ":".join(IMMUTABLE_RUNTIME_ARCHIVES)
                in profile
            )
            assert "--reinstall --no-cache 'ray[default]==2.56.1'" not in profile
            assert "/opt/nemo_rl_venv/bin/python" not in profile
        else:
            assert "--reinstall --no-cache 'ray[default]==2.56.1'" in profile
            assert "--reinstall --no-cache 'dill==0.4.1'" in profile
            assert "--reinstall --no-cache 'numpy==2.5.1'" in profile
            assert "from nemo_rl.algorithms.grpo import MasterConfig" in profile


def test_container_smoke_reuses_official_mcore_environment() -> None:
    script = (EXPERIMENT_DIR / "scripts" / "smoke_nemo_container.sub").read_text()

    assert "UV_CACHE=/tmp/nemo-rl-uv-cache-" not in script
    assert "export UV_CACHE_DIR=" not in script
    assert "export NVTE_CUDA_ARCHS=100" in script
    assert "export PYTHONPATH=" in script
    assert "Megatron-Bridge/3rdparty/Megatron-LM" in script
    assert "Megatron-Bridge/src" in script
    assert "uv run --frozen --extra mcore" not in script
    assert "--editable" not in script
    assert "--reinstall --no-cache 'ray[default]==2.56.1'" in script
    assert "--reinstall --no-cache 'dill==0.4.1'" in script
    assert "--reinstall --no-cache 'numpy==2.5.1'" in script
    assert "from nemo_rl.algorithms.grpo import MasterConfig" in script
    assert "ray --version" in script
    assert "ray start --head" in script
    assert "ray stop" in script
    assert '"ray"' in script
    assert '"grouped_gemm"' not in script
    assert "src/RL-pr5672-mamba-moe-graph-cache-20260729-d7f1d496f" in script


def test_ray_submission_has_no_singleton_dependency() -> None:
    script = (EXPERIMENT_DIR.parents[2] / "ray.sub").read_text()

    assert "--dependency=singleton" not in script
    assert "RAY_CLIENT_SERVER_ENABLED" in script
    assert "${RAY_CLIENT_SERVER_ARG}" in script
    assert "RAY_DASHBOARD_ENABLED" in script
    assert "${RAY_DASHBOARD_ARGS}" in script


def test_collector_schema_and_wandb_metric_mapping_are_exact() -> None:
    collector = _load_experiment_module("collect_results")
    assert collector.CSV_FIELDS == (
        "scope",
        "job_id",
        "status",
        "failure",
        "exit_code",
        "elapsed",
        "completed_steps",
        "step",
        "geometry_key",
        "capture_count",
        "replay_count",
        "cache_hit_count",
        "eviction_count",
        "fallback_count",
        "e2e_step_time",
        "e2e_tokens_per_sec_per_gpu",
        "generation_time",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_time",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_time",
        "logprob_tokens_per_sec_per_gpu",
        "reward_mean",
        "generation_kl_error",
        "token_mult_prob_error",
        "policy_kl_error",
        "js_divergence_error",
        "sampling_importance_ratio",
        "num_masked_seqs_by_logprob_error",
        "policy_loss",
        "grad_norm",
        "peak_allocated_gib",
        "peak_reserved_gib",
    )
    assert collector.WANDB_METRIC_MAP == {
        "e2e_tokens_per_sec_per_gpu": "performance/tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu": (
            "performance/generation_tokens_per_sec_per_gpu"
        ),
        "policy_training_tokens_per_sec_per_gpu": (
            "performance/policy_training_tokens_per_sec_per_gpu"
        ),
        "logprob_tokens_per_sec_per_gpu": (
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
        ),
        "e2e_step_time": "timing/train/total_step_time",
        "generation_time": "timing/train/generation",
        "policy_training_time": "timing/train/policy_training",
        "logprob_time": "timing/train/policy_and_reference_logprobs",
    }
    assert collector.CORRECTNESS_METRIC_MAP == {
        "reward_mean": "train/reward",
        "generation_kl_error": "train/gen_kl_error",
        "token_mult_prob_error": "train/token_mult_prob_error",
        "policy_kl_error": "train/policy_kl_error",
        "js_divergence_error": "train/js_divergence_error",
        "sampling_importance_ratio": "train/sampling_importance_ratio",
        "num_masked_seqs_by_logprob_error": ("train/num_masked_seqs_by_logprob_error"),
        "policy_loss": "train/loss",
        "grad_norm": "train/grad_norm",
    }
    assert collector.QUALITY_METRICS == (
        "train/reward",
        "train/accuracy",
        "train/gen_kl_error",
        "train/token_mult_prob_error",
        "train/policy_kl_error",
        "train/js_divergence_error",
        "train/sampling_importance_ratio",
        "train/num_masked_seqs_by_logprob_error",
        "train/loss",
        "train/grad_norm",
    )


def test_collector_normalizes_nested_local_export_without_network() -> None:
    collector = _load_experiment_module("collect_results")
    row = collector.normalize_record(
        {
            "scope": "attn-moe",
            "job_id": "123",
            "status": "performance:passed",
            "metrics": {
                "_step": 7,
                "cuda_graph/geometry_key": "pp:1",
                "cuda_graph/capture_count": 1,
                "performance/tokens_per_sec_per_gpu": 42.5,
                "performance/generation_tokens_per_sec_per_gpu": 40.0,
                "performance/policy_training_tokens_per_sec_per_gpu": 39.0,
                "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
                    38.0
                ),
                "timing/train/total_step_time": 3.5,
                "timing/train/generation": 1.0,
                "timing/train/policy_training": 1.5,
                "timing/train/policy_and_reference_logprobs": 1.0,
                "train/reward": 0.75,
                "train/gen_kl_error": 0.01,
                "train/token_mult_prob_error": 1.01,
                "train/policy_kl_error": 0.02,
                "train/js_divergence_error": 0.03,
                "train/sampling_importance_ratio": 1.04,
                "train/num_masked_seqs_by_logprob_error": 5.0,
                "train/loss": 0.1,
                "train/grad_norm": 0.4,
            },
        }
    )

    assert tuple(row) == collector.CSV_FIELDS
    assert row["step"] == 7
    assert row["geometry_key"] == "pp:1"
    assert row["capture_count"] == 1
    assert row["e2e_tokens_per_sec_per_gpu"] == 42.5
    assert row["logprob_tokens_per_sec_per_gpu"] == 38.0
    assert row["reward_mean"] == 0.75
    assert row["generation_kl_error"] == 0.01
    assert row["token_mult_prob_error"] == 1.01
    assert row["policy_kl_error"] == 0.02
    assert row["js_divergence_error"] == 0.03
    assert row["sampling_importance_ratio"] == 1.04
    assert row["num_masked_seqs_by_logprob_error"] == 5.0
    assert row["policy_loss"] == 0.1


def test_collector_preserves_submission_failure_detail_without_inventing_telemetry() -> (
    None
):
    collector = _load_experiment_module("collect_results")
    row = collector.normalize_record(
        {
            "scope": "attn-mamba",
            "job_id": "2475436",
            "status": "performance:failed-capture",
            "failure": "cudaErrorStreamCaptureUnsupported",
            "exit_code": "1:0",
            "elapsed": "00:12:50",
            "completed_steps": 3,
        }
    )
    performance_row = collector.normalize_record(
        {
            "scope": "baseline-no-cg",
            "job_id": "2475435",
            "status": "performance:completed",
            "step": 1,
            "metrics": {"performance/tokens_per_sec_per_gpu": 8.0},
        }
    )

    assert row["failure"] == "cudaErrorStreamCaptureUnsupported"
    assert row["exit_code"] == "1:0"
    assert row["elapsed"] == "00:12:50"
    assert row["completed_steps"] == 3
    assert performance_row["failure"] == ""
    assert performance_row["exit_code"] == ""
    assert performance_row["elapsed"] == ""
    assert performance_row["completed_steps"] == ""
    assert performance_row["eviction_count"] == ""
    assert performance_row["fallback_count"] == ""

    renderer = _load_experiment_module("render_report")
    report = renderer.render_html(
        [row],
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256=TE_FP64_WEAKREF_SHA256,
    )

    for value in (
        "Failure detail",
        "Exit code",
        "Elapsed",
        "Completed steps",
        "cudaErrorStreamCaptureUnsupported",
        "1:0",
        "00:12:50",
        "3",
    ):
        assert value in report


def test_submission_failures_survive_csv_and_render_for_all_terminal_signals(
    tmp_path: Path,
) -> None:
    collector = _load_experiment_module("collect_results")
    renderer = _load_experiment_module("render_report")
    csv_path = tmp_path / "results.csv"
    records = [
        {
            "scope": "terminal-status",
            "job_id": "timeout",
            "status": "performance:timeout",
        },
        {
            "scope": "terminal-status",
            "job_id": "oom",
            "status": "OOM",
        },
        {
            "scope": "terminal-status",
            "job_id": "preempted",
            "status": "PREEMPTED",
        },
        {
            "scope": "terminal-status",
            "job_id": "node-fail",
            "status": "NODE_FAIL",
        },
        {
            "scope": "terminal-status",
            "job_id": "boot-fail",
            "status": "BOOT_FAIL",
        },
        {
            "scope": "terminal-status",
            "job_id": "deadline",
            "status": "DEADLINE",
        },
        {
            "scope": "terminal-status",
            "job_id": "revoked",
            "status": "REVOKED",
        },
        {
            "scope": "failure-detail",
            "job_id": "detail-only",
            "status": "performance:completed",
            "failure": "post-run validation failed",
        },
        {
            "scope": "nonzero-exit",
            "job_id": "exit-only",
            "status": "performance:completed",
            "exit_code": "1:0",
        },
        {
            "scope": "successful",
            "job_id": "completed-zero",
            "status": "performance:completed",
            "exit_code": "0:0",
        },
    ]

    collector.write_csv(records, csv_path)
    report = renderer.render_html(
        renderer.read_rows(csv_path),
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256=TE_FP64_WEAKREF_SHA256,
    )
    failures = report.split('<section id="failures">', maxsplit=1)[1].split(
        "</section>", maxsplit=1
    )[0]

    for job_id in (
        "timeout",
        "oom",
        "preempted",
        "node-fail",
        "boot-fail",
        "deadline",
        "revoked",
        "detail-only",
        "exit-only",
    ):
        assert f"<td>{job_id}</td>" in failures
    assert "<td>completed-zero</td>" not in failures
    assert "post-run validation failed" in failures
    assert "<td>1:0</td>" in failures


def test_collector_writes_repository_safe_lf_csv(tmp_path: Path) -> None:
    collector = _load_experiment_module("collect_results")
    output = tmp_path / "results.csv"

    collector.write_csv(
        [{"scope": "attn", "job_id": "123", "status": "smoke:submitted"}],
        output,
    )

    assert b"\r\n" not in output.read_bytes()


def _performance_row(
    *,
    scope: str,
    job_id: str,
    step: int,
    multiplier: float,
    correctness_offset: float = 0.0,
    eviction_count: int = 0,
    fallback_count: int = 0,
) -> dict[str, str]:
    """Build one normalized result row with independently checkable metrics."""
    value = float(step)
    return {
        "scope": scope,
        "job_id": job_id,
        "status": "performance:passed",
        "step": str(step),
        "eviction_count": str(eviction_count),
        "fallback_count": str(fallback_count),
        "e2e_step_time": str(value),
        "e2e_tokens_per_sec_per_gpu": str(value * multiplier),
        "generation_time": str(value + 1),
        "generation_tokens_per_sec_per_gpu": str((value + 1) * multiplier),
        "policy_training_time": str(value + 2),
        "policy_training_tokens_per_sec_per_gpu": str((value + 2) * multiplier),
        "logprob_time": str(value + 3),
        "logprob_tokens_per_sec_per_gpu": str((value + 3) * multiplier),
        "reward_mean": str(value / 100 + correctness_offset),
        "generation_kl_error": str(value / 1000 + correctness_offset),
        "token_mult_prob_error": str(value / 900 + correctness_offset),
        "policy_kl_error": str(value / 800 + correctness_offset),
        "js_divergence_error": str(value / 700 + correctness_offset),
        "sampling_importance_ratio": str(value / 600 + correctness_offset),
        "num_masked_seqs_by_logprob_error": str(value / 500 + correctness_offset),
        "policy_loss": str(value / 10000 + correctness_offset),
        "grad_norm": str(value / 10 + correctness_offset),
    }


def test_steady_state_rows_excludes_capture_window() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(4, 21)
    ]

    steady_rows = collector.steady_state_rows(rows)

    assert [row["step"] for row in steady_rows] == [str(step) for step in range(6, 21)]


def test_steady_state_aggregate_groups_runs_and_compares_to_baseline() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(4, 21)
    ]
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
            correctness_offset=0.1,
        )
        for step in range(4, 21)
    )
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-2",
            step=step,
            multiplier=1.5,
        )
        for step in range(4, 21)
    )

    aggregated = collector.aggregate_performance(collector.steady_state_rows(rows))

    assert [(row["scope"], row["job_id"]) for row in aggregated] == [
        ("attn", "cg-1"),
        ("attn", "cg-2"),
        ("baseline-no-cg", "baseline-1"),
    ]
    cg_row = aggregated[0]
    for field, median, p95 in (
        ("e2e_step_time", "13", "20"),
        ("e2e_tokens_per_sec_per_gpu", "26", "40"),
        ("generation_time", "14", "21"),
        ("generation_tokens_per_sec_per_gpu", "28", "42"),
        ("policy_training_time", "15", "22"),
        ("policy_training_tokens_per_sec_per_gpu", "30", "44"),
        ("logprob_time", "16", "23"),
        ("logprob_tokens_per_sec_per_gpu", "32", "46"),
    ):
        assert cg_row[f"{field}_median"] == median
        assert cg_row[f"{field}_p95"] == p95
    for field in (
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_tokens_per_sec_per_gpu",
    ):
        assert cg_row[f"{field}_ratio_to_baseline"] == "2"
    for field in collector.CORRECTNESS_FIELDS:
        assert cg_row[f"{field}_delta"] == "0.1"
    assert cg_row["valid"] == "true"
    assert cg_row["invalid_reason"] == ""


def test_steady_state_aggregate_invalidates_evictions_and_fallbacks() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-eviction",
            step=step,
            multiplier=2.0,
            eviction_count=1 if step >= 8 else 0,
        )
        for step in range(6, 21)
    )
    rows.extend(
        _performance_row(
            scope="attn",
            job_id="cg-fallback",
            step=step,
            multiplier=2.0,
            fallback_count=2 if step >= 12 else 0,
        )
        for step in range(6, 21)
    )

    aggregated = collector.aggregate_performance(rows)
    invalid_rows = {row["job_id"]: row for row in aggregated}

    assert invalid_rows["cg-eviction"]["valid"] == "false"
    assert invalid_rows["cg-eviction"]["invalid_reason"] == "eviction_count=1"
    assert invalid_rows["cg-fallback"]["valid"] == "false"
    assert invalid_rows["cg-fallback"]["invalid_reason"] == "fallback_count=2"
    for invalid_job_id in ("cg-eviction", "cg-fallback"):
        invalid_aggregate = invalid_rows[invalid_job_id]
        for field in (
            "e2e_step_time",
            "e2e_tokens_per_sec_per_gpu",
            "generation_time",
            "generation_tokens_per_sec_per_gpu",
            "policy_training_time",
            "policy_training_tokens_per_sec_per_gpu",
            "logprob_time",
            "logprob_tokens_per_sec_per_gpu",
        ):
            assert invalid_aggregate[f"{field}_median"] == ""
            assert invalid_aggregate[f"{field}_p95"] == ""
        for field in (*collector.THROUGHPUT_FIELDS, *collector.CORRECTNESS_FIELDS):
            suffix = "ratio_to_baseline" if "tokens" in field else "delta"
            assert invalid_aggregate[f"{field}_{suffix}"] == ""


def test_steady_state_aggregate_rejects_missing_baseline() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]

    aggregate = collector.aggregate_performance(rows)[0]

    assert aggregate["valid"] == "false"
    assert aggregate["invalid_reason"] == "baseline_missing"
    for field in (*collector.THROUGHPUT_FIELDS, *collector.CORRECTNESS_FIELDS):
        suffix = "ratio_to_baseline" if "tokens" in field else "delta"
        assert aggregate[f"{field}_{suffix}"] == ""


@pytest.mark.parametrize(
    ("field", "value", "baseline_reason"),
    [
        ("eviction_count", "1", "eviction_count=1"),
        ("fallback_count", "1", "fallback_count=1"),
        ("e2e_step_time", "", "e2e_step_time_missing"),
    ],
)
def test_steady_state_aggregate_rejects_invalid_baseline(
    field: str,
    value: str,
    baseline_reason: str,
) -> None:
    collector = _load_experiment_module("collect_results")
    baseline_rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    baseline_rows[2][field] = value
    cg_rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]

    aggregates = collector.aggregate_performance([*baseline_rows, *cg_rows])
    by_job_id = {aggregate["job_id"]: aggregate for aggregate in aggregates}

    assert by_job_id["baseline-1"]["valid"] == "false"
    assert by_job_id["baseline-1"]["invalid_reason"] == baseline_reason
    assert by_job_id["cg-1"]["valid"] == "false"
    assert (
        by_job_id["cg-1"]["invalid_reason"]
        == "baseline_invalid=baseline-no-cg/baseline-1"
    )
    assert by_job_id["cg-1"]["e2e_tokens_per_sec_per_gpu_ratio_to_baseline"] == ""
    assert by_job_id["cg-1"]["reward_mean_delta"] == ""


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("e2e_step_time", "", "e2e_step_time_missing"),
        ("generation_time", "nan", "generation_time_nonfinite"),
        ("policy_training_time", "inf", "policy_training_time_nonfinite"),
        ("eviction_count", None, "eviction_count_missing"),
        ("fallback_count", None, "fallback_count_missing"),
        ("reward_mean", "", "reward_mean_missing"),
        ("policy_kl_error", "nan", "policy_kl_error_nonfinite"),
    ],
)
def test_steady_state_aggregate_rejects_incomplete_or_nonfinite_samples(
    field: str,
    value: str | None,
    reason: str,
) -> None:
    collector = _load_experiment_module("collect_results")
    baseline_rows = [
        _performance_row(
            scope="baseline-no-cg",
            job_id="baseline-1",
            step=step,
            multiplier=1.0,
        )
        for step in range(6, 21)
    ]
    cg_rows = [
        _performance_row(
            scope="attn",
            job_id="cg-1",
            step=step,
            multiplier=2.0,
        )
        for step in range(6, 21)
    ]
    if value is None:
        del cg_rows[4][field]
    else:
        cg_rows[4][field] = value

    aggregates = collector.aggregate_performance([*baseline_rows, *cg_rows])
    cg_aggregate = next(
        aggregate for aggregate in aggregates if aggregate["job_id"] == "cg-1"
    )

    assert cg_aggregate["valid"] == "false"
    assert cg_aggregate["invalid_reason"] == reason
    for throughput_field in (
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "logprob_tokens_per_sec_per_gpu",
    ):
        assert cg_aggregate[f"{throughput_field}_ratio_to_baseline"] == ""
    for correctness_field in collector.CORRECTNESS_FIELDS:
        assert cg_aggregate[f"{correctness_field}_delta"] == ""


def test_report_renders_overlay_provenance_steady_state_and_raw_failures() -> None:
    renderer = _load_experiment_module("render_report")
    report = renderer.render_html(
        [
            {
                "scope": "attn",
                "job_id": "raw-failure",
                "status": "performance:invalid",
                "step": "8",
                "fallback_count": "1",
                "generation_kl_error": "0.01",
                "token_mult_prob_error": "1.01",
                "policy_kl_error": "0.02",
                "js_divergence_error": "0.03",
                "sampling_importance_ratio": "1.04",
                "num_masked_seqs_by_logprob_error": "5",
            }
        ],
        nemo_rl_sha="nemo-sha",
        bridge_sha="bridge-sha",
        container_sha256="container-sha",
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256="39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b",
    )

    for value in (
        "2.15.0+42b84005",
        "e707aa46869dc2aec08dfea25402e97a61d49fef",
        "39f7b26b8cf127e3ca104c0375c97ce4e6d047178f9d00836b92469b1c2e544b",
        "container-sha",
        "6–20",
        "raw-failure",
    ):
        assert value in report
    assert '<section id="steady-state-performance">' in report
    assert '<section id="correctness-deltas">' in report
    assert '<section id="failures">' in report
    assert "Generation KL error" in report
    assert "Token multiplication probability error" in report
    assert "Policy KL error" in report
    assert "JS divergence error" in report
    assert "Sampling importance ratio" in report
    assert "Masked sequences by logprob error" in report


def test_report_has_required_sections_scope_labels_and_verified_status() -> None:
    renderer = _load_experiment_module("render_report")
    assert renderer.DEFAULT_MCORE_SHA == "100047b517ea91526dc465448fcb3b37b2598388"
    report = renderer.render_html(
        [
            {
                "scope": "baseline-no-cg",
                "job_id": "1",
                "status": "smoke:passed",
            },
            {
                "scope": "whole-layer",
                "job_id": "2",
                "status": "performance:passed",
                "e2e_step_time": "1.5",
            },
            {
                "scope": "moe-overlap-true-moe-act-false",
                "job_id": "3",
                "status": "accuracy:failed",
                "reward_mean": "0.25",
            },
        ],
        te_version="2.15.0+42b84005",
        te_source_commit="e707aa46869dc2aec08dfea25402e97a61d49fef",
        te_overlay_sha256=TE_FP64_WEAKREF_SHA256,
    )

    for section in (
        "Correctness",
        "Smoke",
        "Performance",
        "Accuracy",
        "Failures",
        "Provenance",
    ):
        assert f"<h2>{section}</h2>" in report
    assert "No-CG baseline (CUDA graphs disabled)" in report
    assert "TE whole-layer capture (empty module list)" in report
    assert "configuration variant; graph scope unchanged" in report
    assert "2471224" in report and "66 passed" in report
    assert "2471343" in report and "29 + 3 passed" in report
    assert "2471570" in report and "38 + 3 passed" in report
    assert "2471681" in report and "43 + 23 passed" in report
    assert "2471988" in report and "packed Mamba parity passed" in report
    assert "MoE 5→3→5 passed" in report
    assert "74.33s" in report and "6.96s" in report and "82.78s" in report
    assert "100047b517ea91526dc465448fcb3b37b2598388" in report
    assert "37 host tests + Pyrefly passed" in report
    assert "Task 7" in report and "Slurm 2472646" in report
    assert "138 passed" in report
    assert "Task 9" in report and "2475736, 2475881" in report
    assert "graph-safe MoE aux-loss CUDA Graph capture/replay" in report
    assert "029fdbcb3fc0aa17b1a4f7398f56040204307d4bc839d318feda1677c98fff5e" in report
    assert renderer.DEFAULT_MODEL_SNAPSHOT in report
    assert renderer.DEFAULT_TOKENIZER_SNAPSHOT in report
    assert "__REQUIRED_*_MODEL_SNAPSHOT__" not in report


def test_checked_in_report_is_static_and_has_all_sections() -> None:
    renderer = _load_experiment_module("render_report")
    report_path = (
        EXPERIMENT_DIR.parents[0]
        / "results"
        / "mamba_moe_te_graph_20260729_report.html"
    )
    report = report_path.read_text()

    assert report.startswith("<!doctype html>")
    for section_id in (
        "correctness",
        "smoke",
        "performance",
        "accuracy",
        "failures",
        "provenance",
    ):
        assert f'<section id="{section_id}">' in report
    assert "2471988" in report
    assert renderer.DEFAULT_MCORE_SHA in report
    assert "Slurm 2472646" in report
    assert "138 passed" in report
    assert "2475736, 2475881" in report
    assert "6 passed in 3.58s" in report
    assert renderer.DEFAULT_MODEL_SNAPSHOT in report
    assert renderer.DEFAULT_TOKENIZER_SNAPSHOT in report
