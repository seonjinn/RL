from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
DRIVER_PATH = EXPERIMENT_DIR / "scripts" / "run_mcore_training.py"
MATRIX_PATH = EXPERIMENT_DIR / "mcore_test_matrix.json"
PRIMARY_NODE = (
    "tests/unit_tests/transformer/test_cuda_graphs.py::"
    "test_te_make_graphed_callables_supports_eval_no_grad"
)
REUSE_NODE = (
    "tests/unit_tests/transformer/test_cuda_graphs.py::"
    "test_te_eval_graph_input_output_buffer_reuse_capability"
)
PARTIAL_MOE_TEST = (
    "tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py::"
    "test_dropless_partial_moe_cuda_graph_distributed"
)
FIXED_THD_PARITY_TEST = (
    "tests/unit_tests/transformer/test_fixed_capacity_thd_parity.py::"
    "test_fixed_capacity_thd_matches_compact_thd"
)
R3_ROUTER_GRAPH_ROW = "dropless_hybridep_nano16_r3_router_graph"
R3_ROUTER_GRAPH_FEATURE_SET = "dropless_hybridep_nano16_r3_router_graph_v1"
R3_ROUTER_GRAPH_TESTS = (
    "tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py::"
    "test_dropless_hybridep_nano16_r3_router_graph",
    "tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py::"
    "test_dropless_hybridep_nano16_r3_router_graph_rejects_invalid_route",
)
PARTIAL_MOE_ROWS = {
    "dropless_hybridep_nano16": (16, 4, 4),
    "dropless_alltoall_qwen30_16": (16, 4, 4),
    "dropless_alltoall_super32": (32, 8, 4),
    "dropless_hybridep_qwen235_64": (64, 16, 4),
}
CURRENT_ROOT_BRANCH = "sj/r3-cg-router-input"
CURRENT_MCORE_BRANCH = "sj/r3-cg-router-input-mcore"
STALE_ROOT_BRANCH = "experiment/thd-cg-hybrid-nemotron-main-20260806"
STALE_MCORE_BRANCH = "sj/thd-cg-hybrid-nemotron-main-20260806"


@dataclass(frozen=True)
class MCoreSubmitterHarness:
    root: Path
    submitter: Path
    environment: dict[str, str]
    scheduler_argv: Path
    scheduler_contact: Path
    run_log_root: Path
    cleanup: Callable[[], None]


def _load_driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_mcore_training", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _device_bindings(
    *, num_nodes: int = 2, gpus_per_node: int = 4
) -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "global_rank": node_rank * gpus_per_node + local_rank,
            "node_rank": node_rank,
            "local_rank": local_rank,
            "cuda_device_index": local_rank,
        }
        for node_rank in range(num_nodes)
        for local_rank in range(gpus_per_node)
    )


def _rank_payloads(
    *,
    run_identity: str,
    num_nodes: int = 2,
    gpus_per_node: int = 4,
) -> tuple[dict[str, object], ...]:
    world_size = num_nodes * gpus_per_node
    return tuple(
        {
            "run_identity": run_identity,
            "rank": rank,
            "world_size": world_size,
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "candidate_kind": "mcore",
            "candidate_sha": "a" * 40,
            "test_row_id": "te_eval_capability_8",
            "node_results": [
                {
                    "node": "tests/test_graphs.py::test_one",
                    "status": "passed",
                    "exit_code": 0,
                }
            ],
            "capability": {
                "global_rank": rank,
                "node_rank": rank // gpus_per_node,
                "local_rank": rank % gpus_per_node,
                "cuda_device_index": rank % gpus_per_node,
                "all_eval_callables_supported": True,
                "backward_executed": False,
                "no_parameter_grads": True,
                "outputs_changed": True,
                "raw_te_eval_reuse_graph_io": False,
            },
        }
        for rank in range(world_size)
    )


def _capability_markers(
    *, raw_reuse: bool = True
) -> dict[str, tuple[dict[str, object], ...]]:
    device = {"node_rank": 0, "local_rank": 0, "cuda_device_index": 0}
    return {
        PRIMARY_NODE: (
            {
                **device,
                "all_eval_callables_supported": True,
                "backward_executed": False,
                "fallback_forward_counter_increment": 1,
                "forward_invocations_after_capture": 3,
                "no_parameter_grads": True,
                "outputs_changed": True,
                "replay_forward_counter_increment": 0,
            },
        ),
        REUSE_NODE: (
            {
                **device,
                "mcore_eval_reuse_graph_io": "not_implemented",
                "raw_te_eval_reuse_graph_io": raw_reuse,
                "raw_te_eval_reuse_rejection": (
                    None if raw_reuse else "only available in training mode"
                ),
                "raw_te_eval_reuse_eager_parity": True if raw_reuse else None,
                "raw_te_eval_reuse_fallback_forward_counter_increment": (
                    1 if raw_reuse else None
                ),
                "raw_te_eval_reuse_no_parameter_grads": True,
                "raw_te_eval_reuse_outputs_changed": True if raw_reuse else None,
                "raw_te_eval_reuse_replay_forward_counter_increment": (
                    0 if raw_reuse else None
                ),
            },
        ),
    }


def _git_repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", path, "config", "user.email", "fixture@example.com"],
        check=True,
    )
    (path / "payload.py").write_text("VALUE = 1\n")
    (path / "alias.py").symlink_to("payload.py")
    subprocess.run(["git", "-C", path, "add", "payload.py", "alias.py"], check=True)
    subprocess.run(["git", "-C", path, "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, commit


def _init_bare_repository(path: Path) -> None:
    subprocess.run(["git", "init", "--bare", "-q", path], check=True)


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", path, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mcore_submitter_harness(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    *,
    root_branch: str = CURRENT_ROOT_BRANCH,
    mcore_branch: str = CURRENT_MCORE_BRANCH,
    root_remote_matches: bool = True,
    mcore_remote_matches: bool = True,
) -> MCoreSubmitterHarness:
    def cleanup() -> None:
        _restore_owner_write(tmp_path)

    request.addfinalizer(cleanup)
    root = tmp_path / "root"
    experiment = (
        root / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
    )
    scripts = experiment / "scripts"
    profiles = experiment / "profiles"
    scripts.mkdir(parents=True)
    profiles.mkdir()
    for name in (
        "mcore_test_matrix.json",
        "profile_snapshot.py",
        "slurm_segment.py",
        "submit_mcore_matrix.sh",
        "verify_runtime_attestation.py",
    ):
        shutil.copy2(EXPERIMENT_DIR / name, experiment / name)
    for name in (
        "run_mcore_scope.sub",
        "run_mcore_training.py",
        "verify_source_provenance.sh",
    ):
        shutil.copy2(EXPERIMENT_DIR / "scripts" / name, scripts / name)
    (root / ".gitignore").write_text("__pycache__/\n")

    mcore_root = (
        root
        / "3rdparty"
        / "Megatron-Bridge-workspace"
        / "Megatron-Bridge"
        / "3rdparty"
        / "Megatron-LM"
    )
    mcore_root.mkdir(parents=True)
    _git(mcore_root, "init", "-q")
    _git(mcore_root, "config", "user.name", "Fixture")
    _git(mcore_root, "config", "user.email", "fixture@example.com")
    (mcore_root / "candidate.py").write_text("VALUE = 1\n")
    _git(mcore_root, "add", "candidate.py")
    _git(mcore_root, "commit", "-qm", "candidate base")
    _git(mcore_root, "branch", "-M", mcore_branch)
    mcore_remote = tmp_path / "mcore-remote.git"
    _init_bare_repository(mcore_remote)
    _git(mcore_root, "remote", "add", "origin", str(mcore_remote))
    _git(mcore_root, "push", "-q", "origin", f"HEAD:refs/heads/{mcore_branch}")
    if not mcore_remote_matches:
        (mcore_root / "candidate.py").write_text("VALUE = 2\n")
        _git(mcore_root, "add", "candidate.py")
        _git(mcore_root, "commit", "-qm", "unpushed candidate")
    candidate_sha = _git(mcore_root, "rev-parse", "HEAD")

    runtime_attestation = tmp_path / "runtime-attestation.json"
    runtime_attestation.write_text(
        json.dumps(
            {
                "runtime_feature_set": (
                    "dropless_hybridep_nano16_r3_router_graph_v1"
                ),
                "excluded_packages": ["fast-hadamard-transform"],
                "torch_cuda_arch_list": "10.0a",
                "nvte_cuda_archs": "100a",
            }
        )
    )
    run_log_root = tmp_path / "run-logs"
    profile = profiles / "oci-hsg.env"
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=oci-hsg",
                "ACCOUNT=fixture",
                "PARTITION=batch",
                f"CONTAINER={tmp_path / 'runtime.sqsh'}",
                f"CONTAINER_SHA256={'a' * 64}",
                f"MOUNTS={tmp_path}:{tmp_path}",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=none",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=00:10:00",
                f"RUNTIME_ATTESTATION={runtime_attestation}",
                "RUNTIME_PREFLIGHT_JOB_ID=733",
                f"UV_EXECUTABLE={tmp_path / 'uv'}",
                f"EXPECTED_TE_SHA={'b' * 40}",
                f"EXPECTED_TE_VERSION_BASE_SHA={'c' * 40}",
                f"EXPECTED_NEMORL_SHA={'d' * 40}",
                f"EXPECTED_BRIDGE_SHA={'e' * 40}",
                f"EXPECTED_MCORE_SHA={candidate_sha}",
                f"RUN_LOG_ROOT={run_log_root}",
                "",
            )
        )
    )

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "root base")
    _git(root, "branch", "-M", root_branch)
    root_remote = tmp_path / "root-remote.git"
    _init_bare_repository(root_remote)
    _git(root, "remote", "add", "seonjinn", str(root_remote))
    _git(root, "push", "-q", "seonjinn", f"HEAD:refs/heads/{root_branch}")
    if not root_remote_matches:
        (root / "root-revision.txt").write_text("unpushed\n")
        _git(root, "add", "root-revision.txt")
        _git(root, "commit", "-qm", "unpushed root")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scheduler_argv = tmp_path / "scheduler-argv.json"
    scheduler_contact = tmp_path / "scheduler-contact"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['SCHEDULER_CONTACT']).write_text('contacted')\n"
        "pathlib.Path(os.environ['SCHEDULER_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('733')\n"
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "CLUSTER": "oci-hsg",
        "PROFILE_FILE": str(profile),
        "MCORE_CANDIDATE_SHA": candidate_sha,
        "MCORE_TEST_ROWS": R3_ROUTER_GRAPH_ROW,
        "MCORE_TEST_VARIANT": "fixture",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SCHEDULER_ARGV": str(scheduler_argv),
        "SCHEDULER_CONTACT": str(scheduler_contact),
        "TEST_ONLY": "0",
        "SBATCH_TEST_ONLY": "1",
    }
    environment.pop("COMMAND", None)
    environment.pop("MCORE_COMMAND", None)
    return MCoreSubmitterHarness(
        root=root,
        submitter=experiment / "submit_mcore_matrix.sh",
        environment=environment,
        scheduler_argv=scheduler_argv,
        scheduler_contact=scheduler_contact,
        run_log_root=run_log_root,
        cleanup=cleanup,
    )


def _run_mcore_submitter(
    harness: MCoreSubmitterHarness, **environment: str
) -> subprocess.CompletedProcess[str]:
    merged_environment = {**harness.environment, **environment}
    return subprocess.run(
        ["bash", str(harness.submitter)],
        cwd=harness.root,
        env=merged_environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _normalize_mcore_sbatch_command(arguments: list[str]) -> tuple[str, ...]:
    dynamic_exports = {
        "CANDIDATE_SOURCE_ROOT",
        "SUBMISSION_INTENT",
        "SUBMISSION_INTENT_SHA256",
    }
    normalized: list[str] = []
    for argument in arguments:
        if not argument.startswith("--export="):
            normalized.append(argument)
            continue
        exports = argument.removeprefix("--export=").split(",")
        normalized_exports = []
        for export in exports:
            name, separator, value = export.partition("=")
            normalized_exports.append(
                f"{name}=<dynamic>" if separator and name in dynamic_exports else export
            )
        normalized.append("--export=" + ",".join(normalized_exports))
    return tuple(normalized)


def _restore_owner_write(path: Path) -> None:
    for descendant in path.rglob("*"):
        if not descendant.is_symlink():
            descendant.chmod(descendant.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_manifest_selects_exact_te_capability_nodes() -> None:
    module = _load_driver()

    rows = module.load_matrix(MATRIX_PATH, candidate_kind="mcore")

    assert tuple(rows) == (
        "te_eval_capability_8",
        "dropless_hybridep_nano16",
        R3_ROUTER_GRAPH_ROW,
        "dropless_alltoall_qwen30_16",
        "dropless_alltoall_super32",
        "dropless_hybridep_qwen235_64",
    )
    assert not any("router_replay" in row_id for row_id in rows)
    assert rows["te_eval_capability_8"].pytest_nodes == (
        "tests/unit_tests/transformer/test_cuda_graphs.py::"
        "test_te_make_graphed_callables_supports_eval_no_grad",
        "tests/unit_tests/transformer/test_cuda_graphs.py::"
        "test_te_eval_graph_input_output_buffer_reuse_capability",
    )
    for row_id, (world_size, num_nodes, gpus_per_node) in PARTIAL_MOE_ROWS.items():
        row = rows[row_id]
        assert row.world_size == world_size
        assert row.allocations == ((num_nodes, gpus_per_node),)
        assert row.pytest_filters == ()
        expected_nodes = (f"{PARTIAL_MOE_TEST}[{row_id}]",)
        if row_id == "dropless_hybridep_nano16":
            expected_nodes += (FIXED_THD_PARITY_TEST,)
        assert row.pytest_nodes == expected_nodes
    r3_row = rows[R3_ROUTER_GRAPH_ROW]
    assert r3_row.world_size == 16
    assert r3_row.allocations == ((4, 4),)
    assert r3_row.pytest_filters == ()
    assert r3_row.pytest_nodes == R3_ROUTER_GRAPH_TESTS


def test_submission_preparation_creates_fresh_verified_immutable_snapshots(
    tmp_path: Path,
) -> None:
    module = _load_driver()
    repository, commit = _git_repository(tmp_path / "candidate")
    payload = {
        "schema_version": 1,
        "candidate_kind": "mcore",
        "candidate_sha": commit,
        "integration_sha": "b" * 40,
        "profile_sha256": "c" * 64,
        "rows": ["te_eval_capability_8"],
        "runtime_feature_set": "te_eval_capability_8",
        "excluded_packages": ["mamba-ssm"],
        "torch_cuda_arch_list": "10.0a",
        "nvte_cuda_archs": "100a",
    }

    first = module.prepare_candidate_submission(
        archive_sources=((repository, commit, Path(".")),),
        run_log_root=tmp_path / "logs",
        candidate_kind="mcore",
        candidate_sha=commit,
        intent_payload=payload,
    )
    second = module.prepare_candidate_submission(
        archive_sources=((repository, commit, Path(".")),),
        run_log_root=tmp_path / "logs",
        candidate_kind="mcore",
        candidate_sha=commit,
        intent_payload=payload,
    )

    assert first.snapshot_root != second.snapshot_root
    assert first.intent_path != second.intent_path
    for artifact in (first, second):
        module.verify_source_snapshot(
            source_root=artifact.snapshot_root,
            candidate_sha=commit,
            expected_sha256=artifact.snapshot_sha256,
        )
        loaded = module.load_submission_intent(
            artifact.intent_path,
            expected_sha256=artifact.intent_sha256,
        )
        assert loaded["snapshot_path"] == str(artifact.snapshot_root)
        assert loaded["snapshot_sha256"] == artifact.snapshot_sha256
        assert artifact.intent_path.stat().st_mode & stat.S_IWUSR == 0
    for artifact in (first, second):
        _restore_owner_write(artifact.snapshot_root)
        artifact.intent_path.chmod(0o644)


def test_snapshot_and_intent_verification_rejects_tampering_or_writable_state(
    tmp_path: Path,
) -> None:
    module = _load_driver()
    repository, commit = _git_repository(tmp_path / "candidate")
    artifact = module.prepare_candidate_submission(
        archive_sources=((repository, commit, Path(".")),),
        run_log_root=tmp_path / "logs",
        candidate_kind="mcore",
        candidate_sha=commit,
        intent_payload={
            "schema_version": 1,
            "candidate_kind": "mcore",
            "candidate_sha": commit,
        },
    )
    payload_file = artifact.snapshot_root / "payload.py"
    payload_file.chmod(0o644)
    with pytest.raises(ValueError, match="writable path"):
        module.verify_source_snapshot(
            source_root=artifact.snapshot_root,
            candidate_sha=commit,
            expected_sha256=artifact.snapshot_sha256,
        )
    payload_file.write_text("VALUE = 2\n")
    payload_file.chmod(0o444)
    with pytest.raises(ValueError, match="snapshot SHA256"):
        module.verify_source_snapshot(
            source_root=artifact.snapshot_root,
            candidate_sha=commit,
            expected_sha256=artifact.snapshot_sha256,
        )

    artifact.intent_path.chmod(0o644)
    artifact.intent_path.write_text("{}\n")
    artifact.intent_path.chmod(0o444)
    with pytest.raises(ValueError, match="submission intent SHA256"):
        module.load_submission_intent(
            artifact.intent_path,
            expected_sha256=artifact.intent_sha256,
        )
    _restore_owner_write(artifact.snapshot_root)
    artifact.intent_path.chmod(0o644)


def test_te_capability_row_requires_one_exact_marker_from_each_node() -> None:
    module = _load_driver()
    markers = _capability_markers()
    expected_device_binding = {
        "node_rank": 0,
        "local_rank": 0,
        "cuda_device_index": 0,
    }

    evidence = module.validate_row_capability(
        row_id="te_eval_capability_8",
        node_capabilities=markers,
        expected_device_binding=expected_device_binding,
    )

    assert evidence["all_eval_callables_supported"] is True
    assert evidence["raw_te_eval_reuse_graph_io"] is True
    for invalid in (
        {**markers, PRIMARY_NODE: ()},
        {**markers, PRIMARY_NODE: markers[PRIMARY_NODE] * 2},
        {
            **markers,
            PRIMARY_NODE: (
                {**markers[PRIMARY_NODE][0], "all_eval_callables_supported": False},
            ),
        },
        {
            **markers,
            PRIMARY_NODE: (
                {**markers[PRIMARY_NODE][0], "replay_forward_counter_increment": 1},
            ),
        },
        {
            **markers,
            REUSE_NODE: (
                {**markers[REUSE_NODE][0], "raw_te_eval_reuse_eager_parity": False},
            ),
        },
        {
            PRIMARY_NODE: (
                {
                    **markers[PRIMARY_NODE][0],
                    "node_rank": 1,
                    "local_rank": 3,
                    "cuda_device_index": 3,
                },
            ),
            REUSE_NODE: (
                {
                    **markers[REUSE_NODE][0],
                    "node_rank": 1,
                    "local_rank": 3,
                    "cuda_device_index": 3,
                },
            ),
        },
    ):
        with pytest.raises(ValueError, match="capability"):
            module.validate_row_capability(
                row_id="te_eval_capability_8",
                node_capabilities=invalid,
                expected_device_binding=expected_device_binding,
            )


def test_rank_aggregation_rejects_result_from_an_earlier_scheduler_run(
    tmp_path: Path,
) -> None:
    module = _load_driver()
    intent_sha256 = "f" * 64
    previous_run = module.derive_run_identity(
        scheduler_job_id="41",
        scheduler_restart_count=0,
        submission_intent_sha256=intent_sha256,
    )
    current_run = module.derive_run_identity(
        scheduler_job_id="42",
        scheduler_restart_count=0,
        submission_intent_sha256=intent_sha256,
    )
    assert module.rank_result_dir(
        run_log_root=tmp_path,
        candidate_kind="mcore",
        candidate_sha="a" * 40,
        row_id="te_eval_capability_8",
        run_identity=previous_run,
    ) != module.rank_result_dir(
        run_log_root=tmp_path,
        candidate_kind="mcore",
        candidate_sha="a" * 40,
        row_id="te_eval_capability_8",
        run_identity=current_run,
    )
    payloads = list(_rank_payloads(run_identity=current_run))
    payloads[3] = {**payloads[3], "run_identity": previous_run}

    with pytest.raises(ValueError, match="run identity"):
        module.validate_rank_payloads(
            tuple(payloads),
            run_identity=current_run,
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            pytest_nodes=("tests/test_graphs.py::test_one",),
        )


def test_rank_aggregation_requires_semantic_capability_consensus() -> None:
    module = _load_driver()
    run_identity = module.derive_run_identity(
        scheduler_job_id="42",
        scheduler_restart_count=0,
        submission_intent_sha256="f" * 64,
    )
    payloads = list(_rank_payloads(run_identity=run_identity))

    module.validate_rank_payloads(
        tuple(payloads),
        run_identity=run_identity,
        candidate_kind="mcore",
        candidate_sha="a" * 40,
        row_id="te_eval_capability_8",
        world_size=8,
        num_nodes=2,
        gpus_per_node=4,
        pytest_nodes=("tests/test_graphs.py::test_one",),
    )
    payloads[5] = {
        **payloads[5],
        "capability": {**payloads[5]["capability"], "outputs_changed": False},
    }

    with pytest.raises(ValueError, match="semantic capability"):
        module.validate_rank_payloads(
            tuple(payloads),
            run_identity=run_identity,
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            pytest_nodes=("tests/test_graphs.py::test_one",),
        )


def test_node_result_exchange_waits_for_every_rank_before_next_test(
    tmp_path: Path,
) -> None:
    module = _load_driver()
    phase_dir = tmp_path / "node-0"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            module.synchronize_node_results,
            phase_dir=phase_dir,
            rank=0,
            world_size=2,
            node="tests/test_graphs.py::test_one",
            exit_code=0,
            timeout_seconds=2.0,
            poll_interval_seconds=0.01,
        )
        deadline = time.monotonic() + 1.0
        while not (phase_dir / "rank-0.json").is_file():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not first.done()

        second = executor.submit(
            module.synchronize_node_results,
            phase_dir=phase_dir,
            rank=1,
            world_size=2,
            node="tests/test_graphs.py::test_one",
            exit_code=0,
            timeout_seconds=2.0,
            poll_interval_seconds=0.01,
        )

        expected = (
            {
                "rank": 0,
                "node": "tests/test_graphs.py::test_one",
                "status": "passed",
                "exit_code": 0,
            },
            {
                "rank": 1,
                "node": "tests/test_graphs.py::test_one",
                "status": "passed",
                "exit_code": 0,
            },
        )
        assert first.result(timeout=2.0) == expected
        assert second.result(timeout=2.0) == expected


def test_pytest_timeout_is_reported_as_a_failed_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_driver()

    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=("python", "-m", "pytest"),
            timeout=12.0,
            output="partial stdout\n",
            stderr="partial stderr\n",
        )

    monkeypatch.setattr(module.subprocess, "run", raise_timeout)
    exit_code, output = module.run_pytest_command(
        command=("python", "-m", "pytest", "tests/test_graphs.py::test_one"),
        source_root=tmp_path,
        timeout_seconds=12.0,
    )

    assert exit_code == 124
    assert "partial stdout" in output
    assert "partial stderr" in output
    assert "PYTEST_TIMEOUT" in output


def test_pytest_command_uses_the_explicit_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_driver()
    captured_environment: dict[str, str] | None = None

    def run_command(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal captured_environment
        captured_environment = kwargs["env"]
        return subprocess.CompletedProcess(
            args[0], returncode=0, stdout="ok", stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", run_command)
    environment = {"MASTER_PORT": "31000", "MCORE_PYTEST_NODE_INDEX": "1"}

    exit_code, output = module.run_pytest_command(
        command=("python", "-m", "pytest", "tests/test_graphs.py::test_one"),
        source_root=tmp_path,
        timeout_seconds=12.0,
        environment=environment,
    )

    assert exit_code == 0
    assert output == "ok"
    assert captured_environment == environment
    assert captured_environment is not environment


def test_distributed_pytest_nodes_use_distinct_rendezvous_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_driver()
    monkeypatch.setenv("MASTER_PORT", "29999")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")

    first = module.pytest_node_environment(node_index=0)
    second = module.pytest_node_environment(node_index=1)

    assert first["MASTER_PORT"] == "30999"
    assert second["MASTER_PORT"] == "31000"
    assert first["MCORE_PYTEST_NODE_INDEX"] == "0"
    assert second["MCORE_PYTEST_NODE_INDEX"] == "1"
    assert "TORCHELASTIC_USE_AGENT_STORE" not in first
    assert "TORCHELASTIC_USE_AGENT_STORE" not in second
    assert os.environ["MASTER_PORT"] == "29999"
    assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "True"


@pytest.mark.parametrize("master_port", (None, "not-a-port", "65000"))
def test_distributed_pytest_node_rejects_invalid_rendezvous_port(
    monkeypatch: pytest.MonkeyPatch, master_port: str | None
) -> None:
    module = _load_driver()
    if master_port is None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
    else:
        monkeypatch.setenv("MASTER_PORT", master_port)

    with pytest.raises(ValueError, match="MASTER_PORT"):
        module.pytest_node_environment(node_index=0)


def test_rank_aggregation_rejects_measured_binding_for_another_rank() -> None:
    module = _load_driver()
    run_identity = module.derive_run_identity(
        scheduler_job_id="42",
        scheduler_restart_count=0,
        submission_intent_sha256="f" * 64,
    )
    payloads = list(_rank_payloads(run_identity=run_identity))
    payloads[0] = {
        **payloads[0],
        "capability": {
            **payloads[0]["capability"],
            "node_rank": 1,
            "local_rank": 3,
            "cuda_device_index": 3,
        },
    }

    with pytest.raises(ValueError, match="rank payload device binding"):
        module.validate_rank_payloads(
            tuple(payloads),
            run_identity=run_identity,
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            pytest_nodes=("tests/test_graphs.py::test_one",),
        )


@pytest.mark.parametrize(
    ("num_nodes", "gpus_per_node", "world_size"),
    ((1, 8, 8), (2, 4, 8), (4, 4, 16), (8, 4, 32), (16, 4, 64)),
)
def test_allocation_validator_accepts_typed_layouts(
    num_nodes: int, gpus_per_node: int, world_size: int
) -> None:
    module = _load_driver()

    assert (
        module.validate_allocation(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            world_size=world_size,
        )
        == world_size
    )


@pytest.mark.parametrize(
    ("num_nodes", "gpus_per_node", "world_size"),
    (
        (1, 4, 8),
        (2, 8, 16),
        (3, 4, 12),
        (8, 4, 16),
        (8, 8, 64),
        (32, 2, 64),
    ),
)
def test_allocation_validator_rejects_mismatch_or_unknown_layout(
    num_nodes: int, gpus_per_node: int, world_size: int
) -> None:
    module = _load_driver()

    with pytest.raises(ValueError, match="allocation|world size|layout"):
        module.validate_allocation(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            world_size=world_size,
        )


def test_result_path_is_candidate_bound_and_rejects_escape(tmp_path: Path) -> None:
    module = _load_driver()
    candidate_sha = "a" * 40
    root = tmp_path / "logs"

    result = module.result_path(
        run_log_root=root,
        candidate_kind="mcore",
        candidate_sha=candidate_sha,
        row_id="te_eval_capability_8",
    )

    assert result == (
        root / "attestations" / "mcore" / candidate_sha / "te_eval_capability_8.json"
    )
    with pytest.raises(ValueError, match="row ID"):
        module.result_path(
            run_log_root=root,
            candidate_kind="mcore",
            candidate_sha=candidate_sha,
            row_id="../escape",
        )


def test_atomic_result_records_each_node_and_all_joined_ranks(tmp_path: Path) -> None:
    module = _load_driver()
    output = tmp_path / "result.json"
    node_results = (
        {"node": "tests/test_graphs.py::test_one", "status": "passed", "exit_code": 0},
        {"node": "tests/test_graphs.py::test_two", "status": "passed", "exit_code": 0},
    )

    payload = module.build_result(
        run_identity=f"slurm-42-0-{'f' * 64}",
        candidate_kind="mcore",
        candidate_sha="a" * 40,
        integration_sha="b" * 40,
        row_id="te_eval_capability_8",
        world_size=8,
        num_nodes=2,
        gpus_per_node=4,
        joined_ranks=tuple(range(8)),
        device_bindings=_device_bindings(),
        node_results=node_results,
        container_sha256="c" * 64,
        transformer_engine_version="2.19.0.dev0",
        transformer_engine_source_commit="d" * 40,
        transformer_engine_version_base_commit="e" * 40,
        all_eval_callables_supported=True,
        mcore_eval_reuse_graph_io="not_implemented",
        raw_te_eval_reuse_graph_io=True,
    )
    module.write_json_atomic(payload, output)

    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "passed"
    assert payload["run_identity"] == f"slurm-42-0-{'f' * 64}"
    assert payload["topology"] == {
        "world_size": 8,
        "num_nodes": 2,
        "gpus_per_node": 4,
        "joined_ranks": list(range(8)),
        "device_bindings": list(_device_bindings()),
    }
    assert [item["node"] for item in payload["node_results"]] == [
        "tests/test_graphs.py::test_one",
        "tests/test_graphs.py::test_two",
    ]


def test_result_fails_when_one_node_or_rank_is_missing() -> None:
    module = _load_driver()

    with pytest.raises(ValueError, match="joined ranks"):
        module.build_result(
            run_identity=f"slurm-42-0-{'f' * 64}",
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            integration_sha="b" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            joined_ranks=tuple(range(7)),
            device_bindings=_device_bindings(),
            node_results=(
                {"node": "tests/test.py::test_one", "status": "passed", "exit_code": 0},
            ),
            container_sha256="c" * 64,
            transformer_engine_version="2.19.0.dev0",
            transformer_engine_source_commit="d" * 40,
            transformer_engine_version_base_commit="e" * 40,
            all_eval_callables_supported=True,
            mcore_eval_reuse_graph_io="not_implemented",
            raw_te_eval_reuse_graph_io=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("local_rank", 4, "local rank"),
        ("cuda_device_index", 1, "CUDA device"),
    ),
)
def test_result_rejects_out_of_range_or_mismatched_rank_device_binding(
    field: str, value: int, message: str
) -> None:
    module = _load_driver()
    bindings = list(_device_bindings())
    bindings[0] = {**bindings[0], field: value}

    with pytest.raises(ValueError, match=message):
        module.build_result(
            run_identity=f"slurm-42-0-{'f' * 64}",
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            integration_sha="b" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            joined_ranks=tuple(range(8)),
            device_bindings=tuple(bindings),
            node_results=(
                {"node": "tests/test.py::test_one", "status": "passed", "exit_code": 0},
            ),
            container_sha256="c" * 64,
            transformer_engine_version="2.19.0.dev0",
            transformer_engine_source_commit="d" * 40,
            transformer_engine_version_base_commit="e" * 40,
            all_eval_callables_supported=True,
            mcore_eval_reuse_graph_io="not_implemented",
            raw_te_eval_reuse_graph_io=False,
        )


def test_result_rejects_duplicate_or_missing_per_node_device_slots() -> None:
    module = _load_driver()
    bindings = list(_device_bindings())
    bindings[1] = {
        **bindings[1],
        "local_rank": 0,
        "cuda_device_index": 0,
    }

    with pytest.raises(ValueError, match="duplicate or missing"):
        module.build_result(
            run_identity=f"slurm-42-0-{'f' * 64}",
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            integration_sha="b" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            joined_ranks=tuple(range(8)),
            device_bindings=tuple(bindings),
            node_results=(
                {"node": "tests/test.py::test_one", "status": "passed", "exit_code": 0},
            ),
            container_sha256="c" * 64,
            transformer_engine_version="2.19.0.dev0",
            transformer_engine_source_commit="d" * 40,
            transformer_engine_version_base_commit="e" * 40,
            all_eval_callables_supported=True,
            mcore_eval_reuse_graph_io="not_implemented",
            raw_te_eval_reuse_graph_io=False,
        )


def test_result_rejects_global_rank_bound_to_the_wrong_node_device_slot() -> None:
    module = _load_driver()
    bindings = list(_device_bindings())
    bindings[0] = {**bindings[0], "global_rank": 4}
    bindings[4] = {**bindings[4], "global_rank": 0}

    with pytest.raises(ValueError, match="global rank.*device slot"):
        module.build_result(
            run_identity=f"slurm-42-0-{'f' * 64}",
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            integration_sha="b" * 40,
            row_id="te_eval_capability_8",
            world_size=8,
            num_nodes=2,
            gpus_per_node=4,
            joined_ranks=tuple(range(8)),
            device_bindings=tuple(bindings),
            node_results=(
                {"node": "tests/test.py::test_one", "status": "passed", "exit_code": 0},
            ),
            container_sha256="c" * 64,
            transformer_engine_version="2.19.0.dev0",
            transformer_engine_source_commit="d" * 40,
            transformer_engine_version_base_commit="e" * 40,
            all_eval_callables_supported=True,
            mcore_eval_reuse_graph_io="not_implemented",
            raw_te_eval_reuse_graph_io=False,
        )


@pytest.mark.parametrize(
    "submitter", ("submit_mcore_matrix.sh", "submit_bridge_matrix.sh")
)
def test_submitters_reject_raw_command_before_remote_or_scheduler(
    submitter: str,
) -> None:
    environment = os.environ.copy()
    environment["COMMAND"] = "touch /tmp/command-injection"

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / submitter)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "raw command" in result.stderr.lower()


@pytest.mark.parametrize(
    "submitter", ("submit_mcore_matrix.sh", "submit_bridge_matrix.sh")
)
def test_submitters_render_runtime_and_literal_digest_exports(submitter: str) -> None:
    source = (EXPERIMENT_DIR / submitter).read_text()
    assignment = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("exports=")
    )
    variable_names = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", assignment))
    environment = {**os.environ, **{name: f"fixture-{name}" for name in variable_names}}

    rendered = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            f"{assignment}\nprintf '%s' \"$exports\"",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    expected_variables = {
        "RUNTIME_FEATURE_SET": "RUNTIME_FEATURE_SET",
        "RUNTIME_EXCLUDED_PACKAGES": "RUNTIME_EXCLUDED_PACKAGES",
        "TORCH_CUDA_ARCH_LIST": "TORCH_CUDA_ARCH_LIST",
        "NVTE_CUDA_ARCHS": "NVTE_CUDA_ARCHS",
        "SUBMISSION_INTENT_SHA256": "intent_sha256",
        "CANDIDATE_SNAPSHOT_SHA256": "snapshot_sha256",
    }
    for field, variable in expected_variables.items():
        assert f"{field}=fixture-{variable}" in rendered


@pytest.mark.parametrize(
    "submitter", ("submit_mcore_matrix.sh", "submit_bridge_matrix.sh")
)
def test_submitters_omit_gpu_request_when_profile_disables_gres(
    submitter: str,
) -> None:
    source = (EXPERIMENT_DIR / submitter).read_text()

    assert 'if [[ "${SBATCH_GRES}" != none ]]' in source
    assert '"--gres=${SBATCH_GRES}"' in source
    assert '"--gpus-per-node=${SBATCH_GPUS_PER_NODE}"' not in source


def test_mcore_submitter_rejects_unknown_row_before_remote_lookup() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CLUSTER": "ptyche",
            "PROFILE_FILE": str(EXPERIMENT_DIR / "profiles" / "ptyche.env.example"),
            "MCORE_CANDIDATE_SHA": "a" * 40,
            "MCORE_TEST_ROWS": "unknown_row",
            "SBATCH_TEST_ONLY": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_mcore_matrix.sh")],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown test row" in result.stderr.lower()


def test_mcore_submitter_accepts_only_current_campaign_refs(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    harness = _mcore_submitter_harness(tmp_path, request)

    result = _run_mcore_submitter(harness)

    assert result.returncode == 0, result.stderr
    assert harness.scheduler_contact.read_text() == "contacted"
    scheduler_arguments = json.loads(harness.scheduler_argv.read_text())
    assert "--test-only" in scheduler_arguments
    exports = dict(
        field.partition("=")[::2]
        for field in next(
            argument.removeprefix("--export=").split(",")
            for argument in scheduler_arguments
            if argument.startswith("--export=")
        )
        if "=" in field
    )
    intent = json.loads(Path(exports["SUBMISSION_INTENT"]).read_text())
    assert exports["EXPECTED_PROFILE_SHA256"] == intent["profile_sha256"]


def test_mcore_submitter_rejects_stale_campaign_refs(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    harness = _mcore_submitter_harness(
        tmp_path,
        request,
        root_branch=STALE_ROOT_BRANCH,
        mcore_branch=STALE_MCORE_BRANCH,
    )

    result = _run_mcore_submitter(
        harness,
        TEST_ONLY="1",
        SBATCH_TEST_ONLY="0",
    )

    assert result.returncode == 2
    assert "current R3 campaign" in result.stderr
    assert "SBATCH:" not in result.stdout
    assert not harness.scheduler_contact.exists()


@pytest.mark.parametrize(
    ("root_remote_matches", "mcore_remote_matches", "expected_error"),
    (
        (False, True, "local HEAD"),
        (True, False, "candidate SHA"),
    ),
)
def test_mcore_submitter_rejects_remote_ref_mismatch(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    root_remote_matches: bool,
    mcore_remote_matches: bool,
    expected_error: str,
) -> None:
    harness = _mcore_submitter_harness(
        tmp_path,
        request,
        root_remote_matches=root_remote_matches,
        mcore_remote_matches=mcore_remote_matches,
    )

    result = _run_mcore_submitter(
        harness,
        TEST_ONLY="1",
        SBATCH_TEST_ONLY="0",
    )

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "SBATCH:" not in result.stdout
    assert not harness.scheduler_contact.exists()


@pytest.mark.parametrize(
    ("mode_environment", "expected_error"),
    (
        ({"TEST_ONLY": "2", "SBATCH_TEST_ONLY": "0"}, "TEST_ONLY must be 0 or 1"),
        (
            {"TEST_ONLY": "0", "SBATCH_TEST_ONLY": "yes"},
            "SBATCH_TEST_ONLY must be 0 or 1",
        ),
        (
            {"TEST_ONLY": "1", "SBATCH_TEST_ONLY": "1"},
            "mutually exclusive",
        ),
    ),
)
def test_mcore_submitter_rejects_invalid_test_modes_before_external_work(
    mode_environment: dict[str, str],
    expected_error: str,
) -> None:
    environment = {
        **os.environ,
        "MCORE_TEST_ROWS": R3_ROUTER_GRAPH_ROW,
        **mode_environment,
    }
    environment.pop("COMMAND", None)
    environment.pop("MCORE_COMMAND", None)

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_mcore_matrix.sh")],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_error in result.stderr


@pytest.mark.parametrize("empty_mode", ("TEST_ONLY", "SBATCH_TEST_ONLY"))
def test_mcore_submitter_rejects_explicit_empty_test_mode_before_external_work(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    empty_mode: str,
) -> None:
    harness = _mcore_submitter_harness(tmp_path, request)
    mode_environment = {"TEST_ONLY": "0", "SBATCH_TEST_ONLY": "0"}
    mode_environment[empty_mode] = ""

    result = _run_mcore_submitter(harness, **mode_environment)

    assert result.returncode == 2
    assert f"{empty_mode} must be 0 or 1" in result.stderr
    assert not harness.scheduler_contact.exists()
    assert not harness.run_log_root.exists()


def test_mcore_submitter_test_only_renders_without_scheduler_or_artifacts(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    harness = _mcore_submitter_harness(tmp_path, request)

    result = _run_mcore_submitter(
        harness,
        TEST_ONLY="1",
        SBATCH_TEST_ONLY="0",
    )

    assert result.returncode == 0, result.stderr
    assert "ROW: dropless_hybridep_nano16_r3_router_graph" in result.stdout
    assert "SBATCH: sbatch " in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not harness.scheduler_contact.exists()
    assert not harness.run_log_root.exists()


def test_mcore_submitter_test_only_uses_real_command_builder(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    harness = _mcore_submitter_harness(tmp_path, request)

    rendered = _run_mcore_submitter(
        harness,
        TEST_ONLY="1",
        SBATCH_TEST_ONLY="0",
    )
    rendered_line = next(
        line.removeprefix("SBATCH: ")
        for line in rendered.stdout.splitlines()
        if line.startswith("SBATCH: ")
    )
    rendered_arguments = shlex.split(rendered_line)
    assert not harness.scheduler_contact.exists()

    submitted = _run_mcore_submitter(
        harness,
        TEST_ONLY="0",
        SBATCH_TEST_ONLY="0",
    )
    submitted_arguments = ["sbatch", *json.loads(harness.scheduler_argv.read_text())]

    assert rendered.returncode == 0, rendered.stderr
    assert submitted.returncode == 0, submitted.stderr
    assert _normalize_mcore_sbatch_command(rendered_arguments) == (
        _normalize_mcore_sbatch_command(submitted_arguments)
    )
    assert "--test-only" not in submitted_arguments


def test_mcore_submitter_harness_cleanup_restores_readonly_tree(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    harness = _mcore_submitter_harness(tmp_path, request)

    try:
        result = _run_mcore_submitter(harness)
        assert result.returncode == 0, result.stderr
        immutable_snapshot = next(
            (harness.run_log_root / "source-snapshots" / "mcore").glob("*/*")
        )
        assert immutable_snapshot.stat().st_mode & stat.S_IWUSR == 0

        cleanup = getattr(harness, "cleanup", None)
        assert callable(cleanup)
        cleanup()
        assert immutable_snapshot.stat().st_mode & stat.S_IWUSR != 0
        shutil.rmtree(harness.run_log_root)
        assert not harness.run_log_root.exists()
    finally:
        _restore_owner_write(tmp_path)


@pytest.mark.parametrize("cluster", ("ptyche", "oci-hsg", "lyris"))
def test_profile_templates_require_absolute_run_log_root(cluster: str) -> None:
    profile_path = EXPERIMENT_DIR / "profiles" / f"{cluster}.env.example"
    values = dict(
        line.split("=", 1)
        for line in profile_path.read_text().splitlines()
        if line and not line.startswith("#")
    )

    assert Path(values["RUN_LOG_ROOT"]).is_absolute()


def test_pytest_commands_preserve_fully_qualified_nodes_as_arguments() -> None:
    module = _load_driver()
    row = module.load_matrix(MATRIX_PATH, candidate_kind="mcore")[
        "te_eval_capability_8"
    ]

    commands = module.pytest_commands(row, python_executable=Path("/runtime/python"))

    assert commands == (
        (
            "/runtime/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit_tests/transformer/test_cuda_graphs.py::"
            "test_te_make_graphed_callables_supports_eval_no_grad",
        ),
        (
            "/runtime/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit_tests/transformer/test_cuda_graphs.py::"
            "test_te_eval_graph_input_output_buffer_reuse_capability",
        ),
    )


@pytest.mark.parametrize(
    "wrapper", ("scripts/run_mcore_scope.sub", "scripts/run_bridge_scope.sub")
)
def test_distributed_wrappers_reject_raw_command_payload(wrapper: str) -> None:
    environment = os.environ.copy()
    environment["COMMAND"] = "touch /tmp/command-injection"

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / wrapper)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "raw command" in result.stderr.lower()


@pytest.mark.parametrize(
    "wrapper", ("scripts/run_mcore_scope.sub", "scripts/run_bridge_scope.sub")
)
def test_distributed_wrappers_are_bash_syntax_valid(wrapper: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(EXPERIMENT_DIR / wrapper)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_worker_rehashes_exact_intent_bytes_used_by_runtime_contract() -> None:
    source = (EXPERIMENT_DIR / "scripts" / "run_mcore_scope.sub").read_text()

    assert "\"${SUBMISSION_INTENT_SHA256}\" <<'PY'" in source
    assert "serialized_intent = intent_path.read_bytes()" in source
    assert "hashlib.sha256(serialized_intent).hexdigest() != sys.argv[11]" in source
    assert "intent = json.loads(serialized_intent)" in source


def _run_leaf_runtime_contract(
    tmp_path: Path,
    *,
    row_id: str,
    runtime_feature_set: str,
    intent_rows: tuple[str, ...] | None = None,
    excluded_packages: tuple[str, ...] = ("fast-hadamard-transform",),
    intent_profile_sha256: str = "d" * 64,
    expected_profile_sha256: str = "d" * 64,
) -> subprocess.CompletedProcess[str]:
    source = (EXPERIMENT_DIR / "scripts" / "run_mcore_scope.sub").read_text()
    start_marker = "  \"${SUBMISSION_INTENT_SHA256}\" <<'PY'\n"
    start = source.index(start_marker) + len(start_marker)
    program = source[start : source.index("\nPY\n)", start)]

    environment_root = tmp_path / "runtime"
    python_executable = environment_root / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n")
    python_executable.chmod(0o555)
    intent_path = tmp_path / "submission-intent.json"
    intent_path.write_text(
        json.dumps(
            {
                "runtime_feature_set": runtime_feature_set,
                "excluded_packages": list(excluded_packages),
                "torch_cuda_arch_list": "10.0a",
                "nvte_cuda_archs": "100a",
                "rows": list(intent_rows if intent_rows is not None else (row_id,)),
                "profile_sha256": intent_profile_sha256,
            }
        )
    )
    intent_path.chmod(0o444)
    intent_sha256 = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    attestation = tmp_path / "runtime-attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "status": "passed",
                "container_sha256": "a" * 64,
                "transformer_engine_source_commit": "b" * 40,
                "transformer_engine_version_base_commit": "c" * 40,
                "runtime_feature_set": runtime_feature_set,
                "excluded_packages": list(excluded_packages),
                "torch_cuda_arch_list": "10.0a",
                "nvte_cuda_archs": "100a",
                "packages": {"transformer_engine.pytorch": {"version": "2.19.0.dev0"}},
                "expected_python_version": "3.13.14",
                "uv_python_install_dir": str(tmp_path / "uv-python-installations"),
                "expected_uv_version": "0.11.28",
                "uv_executable": str(tmp_path / "uv"),
                "expected_nvte_with_nccl_ep": "0",
                "expected_environment_root": str(environment_root),
                "python_executable": str(python_executable),
                "runtime_prefix": str(environment_root),
            }
        )
    )

    return subprocess.run(
        [
            sys.executable,
            "-",
            str(attestation),
            str(intent_path),
            "a" * 64,
            "b" * 40,
            "c" * 40,
            row_id,
            runtime_feature_set,
            ",".join(excluded_packages),
            "10.0a",
            "100a",
            intent_sha256,
        ],
        input=program,
        env={**os.environ, "EXPECTED_PROFILE_SHA256": expected_profile_sha256},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("row_id", "runtime_feature_set", "excluded_packages"),
    (
        (
            R3_ROUTER_GRAPH_ROW,
            R3_ROUTER_GRAPH_FEATURE_SET,
            ("fast-hadamard-transform",),
        ),
        (
            "dropless_hybridep_nano16",
            "dropless_hybridep_nano16",
            ("fast-hadamard-transform",),
        ),
        (
            "te_eval_capability_8",
            "te_eval_capability_8",
            ("causal-conv1d", "deep-ep", "fast-hadamard-transform", "mamba-ssm"),
        ),
        (
            "dropless_alltoall_qwen30_16",
            "dropless_alltoall_qwen30_16",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        (
            "dropless_alltoall_super32",
            "dropless_alltoall_super32",
            ("deep-ep", "fast-hadamard-transform"),
        ),
        (
            "dropless_hybridep_qwen235_64",
            "dropless_hybridep_qwen235_64",
            ("fast-hadamard-transform",),
        ),
        (
            "bridge_forward_only_eval_8",
            "bridge_forward_only_eval_8",
            ("causal-conv1d", "deep-ep", "fast-hadamard-transform", "mamba-ssm"),
        ),
    ),
)
def test_leaf_runtime_contract_accepts_only_supported_row_feature_pairs(
    tmp_path: Path,
    row_id: str,
    runtime_feature_set: str,
    excluded_packages: tuple[str, ...],
) -> None:
    result = _run_leaf_runtime_contract(
        tmp_path,
        row_id=row_id,
        runtime_feature_set=runtime_feature_set,
        excluded_packages=excluded_packages,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 12


@pytest.mark.parametrize(
    ("row_id", "runtime_feature_set", "intent_rows"),
    (
        (R3_ROUTER_GRAPH_ROW, R3_ROUTER_GRAPH_ROW, None),
        (R3_ROUTER_GRAPH_ROW, "dropless_hybridep_nano16_r3_router_graph_v0", None),
        (R3_ROUTER_GRAPH_ROW, "dropless_hybridep_nano16", None),
        ("dropless_hybridep_nano16", R3_ROUTER_GRAPH_FEATURE_SET, None),
        ("unknown_row", "unknown_row", None),
        (
            R3_ROUTER_GRAPH_ROW,
            R3_ROUTER_GRAPH_FEATURE_SET,
            (R3_ROUTER_GRAPH_ROW, "dropless_hybridep_nano16"),
        ),
    ),
    ids=(
        "unversioned-r3-feature",
        "wrong-r3-version",
        "legacy-feature-for-r3-row",
        "r3-feature-for-other-row",
        "unknown-row",
        "multiple-rows",
    ),
)
def test_leaf_runtime_contract_rejects_wrong_or_ambiguous_pairs_before_runner(
    tmp_path: Path,
    row_id: str,
    runtime_feature_set: str,
    intent_rows: tuple[str, ...] | None,
) -> None:
    result = _run_leaf_runtime_contract(
        tmp_path,
        row_id=row_id,
        runtime_feature_set=runtime_feature_set,
        intent_rows=intent_rows,
    )

    assert result.returncode != 0
    assert "narrow runtime" in result.stderr


def test_leaf_runtime_contract_rejects_profile_digest_mismatch(
    tmp_path: Path,
) -> None:
    result = _run_leaf_runtime_contract(
        tmp_path,
        row_id=R3_ROUTER_GRAPH_ROW,
        runtime_feature_set=R3_ROUTER_GRAPH_FEATURE_SET,
        intent_profile_sha256="d" * 64,
        expected_profile_sha256="e" * 64,
    )

    assert result.returncode != 0
    assert "profile SHA256 mismatch" in result.stderr


def test_worker_accepts_r3_runtime_contract_without_dependency_rebuild(
    tmp_path: Path,
) -> None:
    """The GPU worker must launch the staged Python without a networked uv sync."""
    module = _load_driver()
    repository, candidate_sha = _git_repository(tmp_path / "candidate")
    run_log_root = tmp_path / "logs"
    artifacts = module.prepare_candidate_submission(
        archive_sources=((repository, candidate_sha, Path(".")),),
        run_log_root=run_log_root,
        candidate_kind="mcore",
        candidate_sha=candidate_sha,
        intent_payload={
            "schema_version": 1,
            "candidate_kind": "mcore",
            "candidate_sha": candidate_sha,
            "integration_sha": candidate_sha,
            "profile_sha256": "a" * 64,
            "runtime_feature_set": R3_ROUTER_GRAPH_FEATURE_SET,
            "excluded_packages": ["fast-hadamard-transform"],
            "torch_cuda_arch_list": "10.0a",
            "nvte_cuda_archs": "100a",
            "rows": [R3_ROUTER_GRAPH_ROW],
        },
    )
    runtime_root = tmp_path / "staged-runtime"
    environment_root = runtime_root / "environment"
    python_executable = environment_root / "bin" / "python"
    uv_executable = runtime_root / "uv" / "uv"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n")
    python_executable.chmod(0o555)
    python_install_dir = tmp_path / "uv-python-installations"
    attestation = tmp_path / "runtime-attestation.json"
    container_sha256 = "b" * 64
    te_sha = "c" * 40
    attestation.write_text(
        json.dumps(
            {
                "status": "passed",
                "container_sha256": container_sha256,
                "transformer_engine_source_commit": te_sha,
                "transformer_engine_version_base_commit": te_sha,
                "runtime_feature_set": R3_ROUTER_GRAPH_FEATURE_SET,
                "excluded_packages": ["fast-hadamard-transform"],
                "torch_cuda_arch_list": "10.0a",
                "nvte_cuda_archs": "100a",
                "packages": {"transformer_engine.pytorch": {"version": "2.19.0.dev0"}},
                "expected_python_version": "3.13.14",
                "uv_python_install_dir": str(python_install_dir),
                "expected_uv_version": "0.11.28",
                "uv_executable": str(uv_executable),
                "expected_nvte_with_nccl_ep": "0",
                "expected_environment_root": str(environment_root),
                "python_executable": str(python_executable),
                "runtime_prefix": str(environment_root),
            }
        )
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.jsonl"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "payload = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'environment': {\n"
        "        key: os.environ.get(key)\n"
        "        for key in ('NRL_FORCE_REBUILD_VENVS', 'UV_PROJECT_ENVIRONMENT')\n"
        "    },\n"
        "}\n"
        "with open(os.environ['FAKE_SRUN_LOG'], 'a') as output:\n"
        "    output.write(json.dumps(payload) + '\\n')\n"
    )
    fake_srun.chmod(0o755)
    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text("#!/bin/sh\nprintf 'node0\\n'\n")
    fake_scontrol.chmod(0o755)
    bash_with_mapfile = fake_bin / "bash-with-mapfile"
    bash_with_mapfile.write_text(
        "#!/bin/bash\n"
        "mapfile() {\n"
        "    runtime_fields=()\n"
        '    while IFS= read -r line; do runtime_fields+=("${line}"); done\n'
        "    return 0\n"
        "}\n"
        'source "$1"\n'
    )
    bash_with_mapfile.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("COMMAND", None)
    environment.pop("NRL_FORCE_REBUILD_VENVS", None)
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "FAKE_SRUN_LOG": str(srun_log),
            "TEST_ROW_ID": R3_ROUTER_GRAPH_ROW,
            "TEST_WORLD_SIZE": "16",
            "TEST_NUM_NODES": "4",
            "TEST_GPUS_PER_NODE": "4",
            "CANDIDATE_KIND": "mcore",
            "CANDIDATE_SHA": candidate_sha,
            "INTEGRATION_SHA": candidate_sha,
            "CANDIDATE_SOURCE_ROOT": str(artifacts.snapshot_root),
            "CANDIDATE_SNAPSHOT_SHA256": artifacts.snapshot_sha256,
            "SUBMISSION_INTENT": str(artifacts.intent_path),
            "SUBMISSION_INTENT_SHA256": artifacts.intent_sha256,
            "EXPECTED_PROFILE_SHA256": "a" * 64,
            "RUN_LOG_ROOT": str(run_log_root),
            "TEST_MATRIX": str(MATRIX_PATH),
            "RUNNER_PATH": str(DRIVER_PATH),
            "CONTAINER": str(tmp_path / "runtime.sqsh"),
            "CONTAINER_SHA256": container_sha256,
            "MOUNTS": f"{tmp_path}:{tmp_path}",
            "EXPECTED_TE_SHA": te_sha,
            "EXPECTED_TE_VERSION_BASE_SHA": te_sha,
            "RUNTIME_ATTESTATION": str(attestation),
            "RUNTIME_PREFLIGHT_JOB_ID": "734",
            "EXPECTED_UV_EXECUTABLE": str(uv_executable),
            "EXPECTED_NEMORL_SHA": "d" * 40,
            "EXPECTED_BRIDGE_SHA": "e" * 40,
            "EXPECTED_MCORE_SHA": candidate_sha,
            "SOURCE_PROVENANCE_VERIFIER": "/usr/bin/true",
            "RUNTIME_ATTESTATION_COMMAND": str(
                EXPERIMENT_DIR / "verify_runtime_attestation.py"
            ),
            "RUNTIME_FEATURE_SET": R3_ROUTER_GRAPH_FEATURE_SET,
            "RUNTIME_EXCLUDED_PACKAGES": "fast-hadamard-transform",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "NVTE_CUDA_ARCHS": "100a",
            "REPO_ROOT": str(REPO_ROOT),
            "SLURM_JOB_NUM_NODES": "4",
            "SLURM_JOB_NODELIST": "node[0-3]",
            "SLURM_JOB_ID": "1234",
            "SLURM_RESTART_COUNT": "0",
        }
    )
    try:
        result = subprocess.run(
            [
                str(bash_with_mapfile),
                str(EXPERIMENT_DIR / "scripts" / "run_mcore_scope.sub"),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        _restore_owner_write(artifacts.snapshot_root)

    assert result.returncode == 0, result.stderr
    calls = tuple(json.loads(line) for line in srun_log.read_text().splitlines())
    assert len(calls) == 2
    verifier_call = calls[0]["argv"]
    assert str(EXPERIMENT_DIR / "verify_runtime_attestation.py") in verifier_call
    feature_index = verifier_call.index("--runtime-feature-set")
    exclusions_index = verifier_call.index("--excluded-packages")
    assert verifier_call[feature_index + 1] == R3_ROUTER_GRAPH_FEATURE_SET
    assert verifier_call[exclusions_index + 1] == "fast-hadamard-transform"
    worker_call = calls[1]
    assert str(python_executable) in worker_call["argv"]
    assert not any("sync --python" in argument for argument in worker_call["argv"])
    assert worker_call["environment"] == {
        "NRL_FORCE_REBUILD_VENVS": None,
        "UV_PROJECT_ENVIRONMENT": str(environment_root),
    }


def test_scope_classifier_accepts_only_the_committed_mcore_driver() -> None:
    scope_path = EXPERIMENT_DIR / "scope_matrix.py"
    spec = importlib.util.spec_from_file_location("scope_matrix", scope_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    row = module.find_scope_row("baseline")

    accepted = module.classify_scope(
        row,
        model="nano",
        mode="mcore",
        mcore_driver=str(DRIVER_PATH),
    )
    rejected = module.classify_scope(
        row,
        model="nano",
        mode="mcore",
        mcore_driver="/bin/true",
    )

    assert accepted.status == "runnable"
    assert rejected.status == "dependency-blocked"
