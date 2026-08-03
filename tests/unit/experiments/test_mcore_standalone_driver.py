from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
DRIVER_PATH = EXPERIMENT_DIR / "scripts" / "run_mcore_training.py"
MATRIX_PATH = EXPERIMENT_DIR / "mcore_test_matrix.json"


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


def test_manifest_selects_exact_te_capability_nodes() -> None:
    module = _load_driver()

    rows = module.load_matrix(MATRIX_PATH, candidate_kind="mcore")

    assert tuple(rows) == (
        "te_eval_capability_8",
        "execution_kind_bank_8",
        "forward_only_schedule_8",
        "packed_eval_8",
        "packed_tp2_cp2_pp2_8",
        "hybrid_ep16",
        "hybrid_ep32",
        "router_replay_8",
        "router_replay_1f1b_8",
    )
    assert rows["te_eval_capability_8"].pytest_nodes == (
        "tests/unit_tests/transformer/test_cuda_graphs.py::"
        "test_te_make_graphed_callables_supports_eval_no_grad",
        "tests/unit_tests/transformer/test_cuda_graphs.py::"
        "test_te_eval_graph_input_output_buffer_reuse_capability",
    )


@pytest.mark.parametrize(
    ("num_nodes", "gpus_per_node", "world_size"),
    ((1, 8, 8), (2, 4, 8), (4, 4, 16), (8, 4, 32)),
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
    ((1, 4, 8), (2, 8, 16), (3, 4, 12), (8, 4, 16)),
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
        root
        / "attestations"
        / "mcore"
        / candidate_sha
        / "te_eval_capability_8.json"
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


@pytest.mark.parametrize("submitter", ("submit_mcore_matrix.sh", "submit_bridge_matrix.sh"))
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
