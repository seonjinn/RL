from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
MCORE_DRIVER_PATH = EXPERIMENT_DIR / "scripts" / "run_mcore_training.py"
BASELINE = "scopes/00_baseline_no_cg.sh"
NANO_PERFORMANCE_SCOPES = (
    BASELINE,
    "scopes/17_attn.sh",
    "scopes/09_mlp.sh",
    "scopes/05_mamba.sh",
    "scopes/03_moe_router.sh",
    "scopes/31_attn_mlp_mamba_moe_router.sh",
)
SUPER_PERFORMANCE_SCOPES = (
    BASELINE,
    "scopes/17_attn.sh",
    "scopes/09_mlp.sh",
    "scopes/05_mamba.sh",
    "scopes/03_moe_router.sh",
    "scopes/04_moe_router_preprocess.sh",
    "scopes/32_attn_mlp_mamba_moe_router_preprocess.sh",
)
QWEN_ROUTER_CONDITIONS = (
    "conditions/qwen_A_baseline_r3off.sh",
    "conditions/qwen_B_moe_router_r3off.sh",
    "conditions/qwen_C_baseline_r3on.sh",
    "conditions/qwen_E_attn_r3on.sh",
)
PROVENANCE = {
    "nemo_rl_commit": "1" * 40,
    "bridge_commit": "2" * 40,
    "mcore_commit": "3" * 40,
    "container_sha256": "4" * 64,
}


def _write_gate(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gate_profile(tmp_path: Path) -> tuple[Path, str]:
    runtime_attestation = tmp_path / "runtime-attestation.json"
    runtime_attestation.write_text('{"attestation":"unit"}\n')
    profile = tmp_path / "experiment" / "profiles" / "oci-hsg.env"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        "EXPECTED_NEMORL_SHA=" + PROVENANCE["nemo_rl_commit"] + "\n"
        "EXPECTED_BRIDGE_SHA=" + PROVENANCE["bridge_commit"] + "\n"
        "EXPECTED_MCORE_SHA=" + PROVENANCE["mcore_commit"] + "\n"
        "CONTAINER_SHA256=" + PROVENANCE["container_sha256"] + "\n"
        f"RUNTIME_ATTESTATION={runtime_attestation}\n"
    )
    return profile, hashlib.sha256(runtime_attestation.read_bytes()).hexdigest()


def _r3_gate(runtime_digest: str) -> dict[str, object]:
    return {
        "gate_type": "qwen235_r3_routes",
        "status": "passed",
        "model": "qwen3_235b",
        "slurm_job_id": 123,
        "provenance": {**PROVENANCE, "runtime_attestation_sha256": runtime_digest},
        "diagnostic": {
            "model": "Qwen/Qwen3-235B-A22B",
            "num_prompts": 128,
            "max_tokens": 256,
            "max_model_len": 8192,
            "prompt_repeat": 128,
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enforce_eager": False,
            "moe_backend": "triton",
            "num_outputs": 128,
            "num_failures": 0,
        },
    }


def _promotion_gate(
    model: str, runtime_digest: str, arms: tuple[str, ...]
) -> dict[str, object]:
    arm_payload: dict[str, object] = {}
    for arm in arms:
        r3_on = arm in {"C", "E"}
        arm_payload[arm] = {
            "job_id": 456,
            "status": "passed",
            "completed_steps": 5,
            "metrics_finite": True,
            "correctness_passed": True,
            "undeclared_fallbacks": 0,
            "router_replay": "on" if r3_on else "off",
            "graph_coverage_status": "passed"
            if arm in {"B", "E"}
            else "not_applicable",
            "r3_trace_status": "passed" if r3_on else "not_applicable",
        }
    return {
        "gate_type": "smoke_promotion",
        "status": "passed",
        "model": model,
        "phase": "smoke",
        "steps": 5,
        "provenance": {**PROVENANCE, "runtime_attestation_sha256": runtime_digest},
        "arms": arm_payload,
    }


def _write_launcher(path: Path, relative_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "
        f'\'{relative_path}\' "${{STEPS}}" "${{RUN_GROUP:-}}" '
        '"${REPEAT_INDEX:-}" "${RUN_TAG}" >>"${CAPTURE_FILE}"\n'
    )
    path.chmod(0o755)


def _make_harness(
    tmp_path: Path, submitter: str, launchers: tuple[str, ...]
) -> tuple[Path, Path]:
    harness = tmp_path / "experiment"
    harness.mkdir()
    shutil.copy2(EXPERIMENT_DIR / submitter, harness / submitter)
    shutil.copy2(
        EXPERIMENT_DIR / "validate_campaign_gate.py",
        harness / "validate_campaign_gate.py",
    )
    shutil.copy2(
        EXPERIMENT_DIR / "profile_snapshot.py", harness / "profile_snapshot.py"
    )
    capture_file = tmp_path / "captured.tsv"
    for relative_path in launchers:
        _write_launcher(harness / relative_path, relative_path)
    return harness, capture_file


def _load_gate_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_campaign_gate",
        EXPERIMENT_DIR / "validate_campaign_gate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_mcore_driver():
    spec = importlib.util.spec_from_file_location(
        "run_mcore_training_for_matrix_test", MCORE_DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_shell_profile_consumers_accept_every_snapshot_field() -> None:
    spec = importlib.util.spec_from_file_location(
        "profile_snapshot_for_consumer_test",
        EXPERIMENT_DIR / "profile_snapshot.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    for relative_path in (
        "run_scope.sh",
        "submit_mcore_matrix.sh",
        "submit_bridge_matrix.sh",
    ):
        source = (EXPERIMENT_DIR / relative_path).read_text()
        allowlist = next(
            line.strip().removesuffix(")").split("|")
            for line in source.splitlines()
            if line.strip().startswith(("PROFILE_SHA256|", "PROFILE_ID|"))
        )
        assert set(module.PROFILE_FIELDS) <= set(allowlist), relative_path


def test_direct_sbatch_submitters_scrub_reserved_environment() -> None:
    expected_invocations = {
        "run_scope.sh": (
            'scheduler_test_output=$(run_sbatch_without_reserved_environment "${sbatch_command[@]}")',
            'job_id=$(run_sbatch_without_reserved_environment "${sbatch_command[@]}")',
        ),
        "submit_mcore_matrix.sh": (
            'output=$(run_sbatch_without_reserved_environment "${command[@]}")',
        ),
        "submit_bridge_matrix.sh": (
            'output=$(run_sbatch_without_reserved_environment "${command[@]}")',
        ),
    }

    for relative_path, invocations in expected_invocations.items():
        source = (EXPERIMENT_DIR / relative_path).read_text()
        assert '[[ "${exported_name}" == SBATCH_* ]]' in source
        assert 'clean_environment+=(-u "${exported_name}")' in source
        assert "done < <(compgen -e)" in source
        for invocation in invocations:
            assert invocation in source


def test_typed_matrix_workers_bind_runtime_attestation_to_producer_job() -> None:
    worker = (EXPERIMENT_DIR / "scripts" / "run_mcore_scope.sub").read_text()

    assert "RUNTIME_PREFLIGHT_JOB_ID" in worker.split("; do", 1)[0]
    assert (
        '--expected-runtime-attestation-job-id "${RUNTIME_PREFLIGHT_JOB_ID}"' in worker
    )
    assert '--expected-uv-executable "${EXPECTED_UV_EXECUTABLE}"' in worker
    for relative_path in ("submit_mcore_matrix.sh", "submit_bridge_matrix.sh"):
        submitter = (EXPERIMENT_DIR / relative_path).read_text()
        assert "RUNTIME_PREFLIGHT_JOB_ID=${RUNTIME_PREFLIGHT_JOB_ID}" in submitter, (
            relative_path
        )
        assert "EXPECTED_UV_EXECUTABLE=${UV_EXECUTABLE}" in submitter, relative_path


def test_mcore_candidate_archive_collection_resolves_every_literal_manifest_node(
    tmp_path: Path,
) -> None:
    module = _load_mcore_driver()
    source_root = tmp_path / "candidate"
    test_path = source_root / "tests" / "test_candidate.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('case', ('alpha', 'beta'), ids=('alpha', 'beta'))\n"
        "def test_replay(case):\n"
        "    pass\n"
    )
    row = module.MatrixRow(
        row_id="candidate_row",
        world_size=8,
        allocations=((2, 4),),
        pytest_nodes=("tests/test_candidate.py::test_replay[beta]",),
        pytest_filters=(),
    )

    assert module.validate_pytest_node_collection(
        source_root=source_root,
        rows={row.row_id: row},
        python_executable=Path(sys.executable),
    ) == ("tests/test_candidate.py::test_replay[beta]",)

    parametrized_base = module.MatrixRow(
        row_id="parametrized_base_row",
        world_size=8,
        allocations=((2, 4),),
        pytest_nodes=("tests/test_candidate.py::test_replay",),
        pytest_filters=(),
    )
    assert module.validate_pytest_node_collection(
        source_root=source_root,
        rows={parametrized_base.row_id: parametrized_base},
        python_executable=Path(sys.executable),
    ) == ("tests/test_candidate.py::test_replay",)

    missing = module.MatrixRow(
        row_id="missing_row",
        world_size=8,
        allocations=((2, 4),),
        pytest_nodes=("tests/test_candidate.py::test_replay[absent]",),
        pytest_filters=(),
    )
    with pytest.raises(ValueError, match="missing_row.*absent"):
        module.validate_pytest_node_collection(
            source_root=source_root,
            rows={row.row_id: row, missing.row_id: missing},
            python_executable=Path(sys.executable),
        )


@pytest.mark.parametrize(
    ("collected_output", "accepted"),
    (
        ("tests/test_candidate.py::test_replay\n", True),
        ("tests/test_candidate.py::test_replay[alpha]\n", True),
        (
            "tests/test_candidate.py::test_replay[alpha]\n"
            "tests/test_candidate.py::test_replay[beta]\n",
            True,
        ),
        ("tests/test_candidate.py::test_replay[alpha-beta_1.2 space]\n", True),
        ("tests/test_candidate.py::test_replay_extra[alpha]\n", False),
        ("tests/test_candidate.py::test_sibling[alpha]\n", False),
        ("tests/test_other.py::test_replay[alpha]\n", False),
        ("tests/test_candidate.py::test_replay[alpha][beta]\n", False),
        ("tests/test_candidate.py::test_replay[[alpha]]\n", False),
        ("", False),
        (
            "tests/test_candidate.py::test_replay[alpha]\n"
            "tests/test_candidate.py::test_replay[alpha]\n",
            False,
        ),
        (
            "tests/test_candidate.py::test_replay_extra[alpha]\n"
            "tests/test_candidate.py::test_sibling[beta]\n",
            False,
        ),
    ),
    ids=(
        "exact",
        "parameter-suffix",
        "multiple-parameter-suffixes",
        "common-parameter-id-characters",
        "lookalike-prefix",
        "sibling",
        "file-mismatch",
        "multiple-bracket-suffixes",
        "nested-brackets",
        "empty",
        "duplicate-collected-id",
        "multiple-unsafe",
    ),
)
def test_mcore_collection_base_selector_matches_only_exact_parameter_expansions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collected_output: str,
    accepted: bool,
) -> None:
    module = _load_mcore_driver()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nprintf '%s' \"${FAKE_COLLECTION_OUTPUT}\"\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv("FAKE_COLLECTION_OUTPUT", collected_output)
    row = module.MatrixRow(
        row_id="candidate_row",
        world_size=8,
        allocations=((2, 4),),
        pytest_nodes=("tests/test_candidate.py::test_replay",),
        pytest_filters=(),
    )

    if accepted:
        assert module.validate_pytest_node_collection(
            source_root=tmp_path,
            rows={row.row_id: row},
            python_executable=fake_python,
        ) == ("tests/test_candidate.py::test_replay",)
    else:
        with pytest.raises(ValueError, match="test_replay"):
            module.validate_pytest_node_collection(
                source_root=tmp_path,
                rows={row.row_id: row},
                python_executable=fake_python,
            )


def test_mcore_collection_rejects_overlapping_base_and_explicit_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_mcore_driver()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nprintf '%s' \"${FAKE_COLLECTION_OUTPUT}\"\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv(
        "FAKE_COLLECTION_OUTPUT", "tests/test_candidate.py::test_replay[alpha]\n"
    )
    row = module.MatrixRow(
        row_id="ambiguous_row",
        world_size=8,
        allocations=((2, 4),),
        pytest_nodes=(
            "tests/test_candidate.py::test_replay",
            "tests/test_candidate.py::test_replay[alpha]",
        ),
        pytest_filters=(),
    )

    with pytest.raises(ValueError, match=r"ambiguous.*test_replay\[alpha\]"):
        module.validate_pytest_node_collection(
            source_root=tmp_path,
            rows={row.row_id: row},
            python_executable=fake_python,
        )


def test_mcore_worker_validates_entire_candidate_matrix_before_execution() -> None:
    source = MCORE_DRIVER_PATH.read_text()
    module = ast.parse(source)
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    validation = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "validate_pytest_node_collection"
    )
    execution = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "run_pytest_command"
    )
    rows = next(
        keyword.value for keyword in validation.keywords if keyword.arg == "rows"
    )

    assert validation.lineno < execution.lineno
    assert isinstance(rows, ast.Name)
    assert rows.id == "rows"


def _run_submitter(
    harness: Path,
    submitter: str,
    capture_file: Path,
    *,
    model: str = "nano",
    arguments: tuple[str, ...] = (),
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "ACCURACY_SCRIPT",
        "PHASE",
        "PERFORMANCE_SCRIPTS",
        "REPEATS",
        "RUN_GROUP",
        "REPEAT_INDEX",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "CAPTURE_FILE": str(capture_file),
            "CLUSTER": "oci-hsg",
            "MODEL": model,
            "MODE": "nemorl",
            "RUN_TAG": "unit",
            "TEST_ONLY": "1",
        }
    )
    environment.update(extra_environment or {})
    return subprocess.run(
        ["/bin/bash", str(harness / submitter), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _captured_rows(capture_file: Path) -> list[tuple[str, str, str, str, str]]:
    if not capture_file.exists():
        return []
    rows = []
    for line in capture_file.read_text().splitlines():
        fields = line.split("\t")
        assert len(fields) == 5
        rows.append((fields[0], fields[1], fields[2], fields[3], fields[4]))
    return rows


def test_qwen_router_validation_smoke_defaults_to_ordered_paired_arms(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_30ba3b",
        extra_environment={"PHASE": "smoke"},
    )

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert tuple(row[0] for row in rows) == QWEN_ROUTER_CONDITIONS
    assert {row[1] for row in rows} == {"5"}
    assert {row[3] for row in rows} == {"1"}
    assert len({row[2] for row in rows[:2]}) == 1
    assert len({row[2] for row in rows[2:]}) == 1
    assert rows[0][2] != rows[2][2]


def test_qwen235_smoke_defaults_to_a_and_b_without_a_route_gate(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        extra_environment={"PHASE": "smoke"},
    )

    assert result.returncode == 0, result.stderr
    assert tuple(row[0] for row in _captured_rows(capture_file)) == (
        "conditions/qwen_A_baseline_r3off.sh",
        "conditions/qwen_B_moe_router_r3off.sh",
    )


def test_qwen30_performance_defaults_to_a_and_b(tmp_path: Path) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "promotion-gate.json"
    gate_digest = _write_gate(
        gate, _promotion_gate("qwen3_30ba3b", runtime_digest, ("A", "B"))
    )

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_30ba3b",
        extra_environment={
            "PHASE": "performance",
            "PROFILE_FILE": str(profile),
            "SMOKE_PROMOTION_FILE": str(gate),
            "SMOKE_PROMOTION_SHA256": gate_digest,
        },
    )

    assert result.returncode == 0, result.stderr
    assert tuple(row[0] for row in _captured_rows(capture_file)) == (
        "conditions/qwen_A_baseline_r3off.sh",
        "conditions/qwen_B_moe_router_r3off.sh",
    )


def test_qwen235_r3_smoke_rejects_self_attested_route_gate_before_leaf(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "r3-gate.json"
    gate_digest = _write_gate(gate, _r3_gate(runtime_digest))

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("C", "E"),
        extra_environment={
            "PHASE": "smoke",
            "PROFILE_FILE": str(profile),
            "R3_PREFLIGHT_FILE": str(gate),
            "R3_PREFLIGHT_SHA256": gate_digest,
        },
    )

    assert result.returncode == 2
    assert "content-bound Slurm diagnostic producer" in result.stderr
    assert _captured_rows(capture_file) == []


def test_qwen_performance_requires_valid_promotion_gate_before_any_leaf(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "promotion-gate.json"
    gate_digest = _write_gate(
        gate, _promotion_gate("qwen3_235b", runtime_digest, ("A", "B"))
    )

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("A", "B"),
        extra_environment={
            "PHASE": "performance",
            "PROFILE_FILE": str(profile),
            "SMOKE_PROMOTION_FILE": str(gate),
            "SMOKE_PROMOTION_SHA256": gate_digest,
        },
    )

    assert result.returncode == 0, result.stderr
    assert tuple(row[0] for row in _captured_rows(capture_file)) == (
        "conditions/qwen_A_baseline_r3off.sh",
        "conditions/qwen_B_moe_router_r3off.sh",
    )


@pytest.mark.parametrize(
    "extra_environment",
    (
        {"TEST_ONLY": "2"},
        {"TEST_ONLY": "1", "SBATCH_TEST_ONLY": "1"},
        {"PHASE": "performance"},
    ),
)
def test_qwen_router_validation_rejects_invalid_gate_controls_before_leaf_invocation(
    tmp_path: Path, extra_environment: dict[str, str]
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_30ba3b",
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    "case",
    (
        "missing_digest",
        "symlink",
        "digest_mismatch",
        "failed",
        "wrong_model",
        "unknown_field",
        "provenance_mismatch",
    ),
)
def test_qwen235_route_gate_rejects_untrusted_or_invalid_evidence_before_leaf(
    tmp_path: Path, case: str
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    payload = _r3_gate(runtime_digest)
    gate = tmp_path / "r3-gate.json"
    if case == "failed":
        payload["status"] = "failed"
    elif case == "wrong_model":
        payload["model"] = "qwen3_30ba3b"
    elif case == "unknown_field":
        payload["unexpected"] = True
    elif case == "provenance_mismatch":
        provenance = payload["provenance"]
        assert isinstance(provenance, dict)
        provenance["container_sha256"] = "0" * 64
    gate_digest = _write_gate(gate, payload)
    if case == "symlink":
        linked_gate = tmp_path / "linked-r3-gate.json"
        linked_gate.symlink_to(gate)
        gate = linked_gate
    if case == "digest_mismatch":
        gate_digest = "0" * 64
    extra_environment = {
        "PHASE": "smoke",
        "PROFILE_FILE": str(profile),
        "R3_PREFLIGHT_FILE": str(gate),
        "R3_PREFLIGHT_SHA256": gate_digest,
    }
    if case == "missing_digest":
        extra_environment.pop("R3_PREFLIGHT_SHA256")

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("C",),
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize("case", ("outside", "symlink", "malicious"))
def test_qwen_router_validation_rejects_untrusted_profiles_without_execution(
    tmp_path: Path, case: str
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "r3-gate.json"
    gate_digest = _write_gate(gate, _r3_gate(runtime_digest))
    marker = tmp_path / "scheduler-contacted"
    if case == "outside":
        profile = tmp_path / "outside-profile.env"
        profile.write_text(
            "EXPECTED_NEMORL_SHA=" + PROVENANCE["nemo_rl_commit"] + "\n"
            "EXPECTED_BRIDGE_SHA=" + PROVENANCE["bridge_commit"] + "\n"
            "EXPECTED_MCORE_SHA=" + PROVENANCE["mcore_commit"] + "\n"
            "CONTAINER_SHA256=" + PROVENANCE["container_sha256"] + "\n"
            f"RUNTIME_ATTESTATION={tmp_path / 'runtime-attestation.json'}\n"
        )
    elif case == "symlink":
        linked_profile = profile.with_name("linked-profile.env")
        linked_profile.symlink_to(profile)
        profile = linked_profile
    else:
        profile.write_text(f"$(touch {marker})\n")

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("C",),
        extra_environment={
            "PHASE": "smoke",
            "PROFILE_FILE": str(profile),
            "R3_PREFLIGHT_FILE": str(gate),
            "R3_PREFLIGHT_SHA256": gate_digest,
        },
    )

    assert result.returncode == 2
    assert not marker.exists()
    assert _captured_rows(capture_file) == []


def test_qwen_router_validation_rejects_cluster_before_profile_or_leaf_resolution(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("C",),
        extra_environment={"CLUSTER": "../../outside", "PHASE": "smoke"},
    )

    assert result.returncode == 2
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize("job_field", ("slurm_job_id", "job_id"))
def test_campaign_gate_rejects_fractional_slurm_job_ids(
    tmp_path: Path, job_field: str
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "gate.json"
    if job_field == "slurm_job_id":
        payload = _r3_gate(runtime_digest)
        payload["slurm_job_id"] = 1.5
        environment = {
            "PHASE": "smoke",
            "R3_PREFLIGHT_FILE": str(gate),
            "R3_PREFLIGHT_SHA256": "",
        }
        arguments = ("C",)
    else:
        payload = _promotion_gate("qwen3_235b", runtime_digest, ("A",))
        arms = payload["arms"]
        assert isinstance(arms, dict)
        arm = arms["A"]
        assert isinstance(arm, dict)
        arm["job_id"] = 1.5
        environment = {
            "PHASE": "performance",
            "SMOKE_PROMOTION_FILE": str(gate),
            "SMOKE_PROMOTION_SHA256": "",
        }
        arguments = ("A",)
    digest = _write_gate(gate, payload)
    for key in tuple(environment):
        if key.endswith("SHA256"):
            environment[key] = digest
    environment["PROFILE_FILE"] = str(profile)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=arguments,
        extra_environment=environment,
    )

    assert result.returncode == 2
    assert _captured_rows(capture_file) == []


def test_campaign_gate_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    validator = _load_gate_validator()
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"failed","status":"passed"}')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validator._parse_json(gate.read_bytes())


def test_campaign_gate_reads_opened_file_even_if_path_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_gate_validator()
    gate = tmp_path / "gate.json"
    replacement = tmp_path / "replacement.json"
    gate.write_bytes(b'{"old":true}')
    replacement.write_bytes(b'{"new":true}')
    original_open = validator.os.open

    def open_then_swap(path: str, flags: int) -> int:
        descriptor = original_open(path, flags)
        replacement.replace(gate)
        return descriptor

    monkeypatch.setattr(validator.os, "open", open_then_swap)

    assert validator._read_regular_file(gate, "gate file") == b'{"old":true}'


def test_r3_validator_rejects_non_qwen235_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_gate_validator()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_campaign_gate.py",
            "r3",
            "--gate-file",
            "/tmp/gate.json",
            "--gate-sha256",
            "0" * 64,
            "--model",
            "qwen3_30ba3b",
            "--profile-file",
            "/tmp/profile.env",
            "--profile-dir",
            "/tmp",
            "--cluster",
            "oci-hsg",
        ],
    )

    with pytest.raises(SystemExit) as error:
        validator._parse_args()
    assert error.value.code == 2


def test_qwen_router_validation_performance_selected_pair_uses_twenty_steps(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)
    profile, runtime_digest = _write_gate_profile(tmp_path)
    gate = tmp_path / "promotion-gate.json"
    gate_digest = _write_gate(
        gate, _promotion_gate("qwen3_235b", runtime_digest, ("A", "B"))
    )

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_235b",
        arguments=("A", "B"),
        extra_environment={
            "PHASE": "performance",
            "PROFILE_FILE": str(profile),
            "SMOKE_PROMOTION_FILE": str(gate),
            "SMOKE_PROMOTION_SHA256": gate_digest,
        },
    )

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert tuple(row[0] for row in rows) == QWEN_ROUTER_CONDITIONS[:2]
    assert {row[1] for row in rows} == {"20"}
    assert {row[3] for row in rows} == {"1"}
    assert len({row[2] for row in rows}) == 1


def test_qwen_router_validation_assigns_distinct_repeat_indices(
    tmp_path: Path,
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model="qwen3_30ba3b",
        arguments=("A", "B"),
        extra_environment={"REPEATS": "3"},
    )

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert [row[3] for row in rows] == ["1", "1", "2", "2", "3", "3"]
    assert [row[4] for row in rows] == [
        "unit-r1",
        "unit-r1",
        "unit-r2",
        "unit-r2",
        "unit-r3",
        "unit-r3",
    ]


@pytest.mark.parametrize(
    ("model", "phase", "arguments"),
    (
        ("nano", "smoke", ()),
        ("qwen3_30ba3b", "accuracy", ()),
        ("qwen3_30ba3b", "smoke", ("D",)),
        ("qwen3_30ba3b", "smoke", ("../A",)),
    ),
)
def test_qwen_router_validation_rejects_invalid_inputs_before_leaf_invocation(
    tmp_path: Path, model: str, phase: str, arguments: tuple[str, ...]
) -> None:
    submitter = "submit_qwen_router_validation.sh"
    harness, capture_file = _make_harness(tmp_path, submitter, QWEN_ROUTER_CONDITIONS)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        model=model,
        arguments=arguments,
        extra_environment={"PHASE": phase},
    )

    assert result.returncode == 2
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    ("submitter", "selector_name"),
    (
        ("submit_performance_matrix.sh", "argument"),
        ("submit_accuracy_soak.sh", "environment"),
    ),
)
@pytest.mark.parametrize(
    "unsafe_path",
    (
        "scopes/../outside.sh",
        "scopes/nested/leaf.sh",
        "scopes//leaf.sh",
    ),
)
def test_selected_launcher_rejects_traversal_and_embedded_path_segments(
    tmp_path: Path, submitter: str, selector_name: str, unsafe_path: str
) -> None:
    harness, capture_file = _make_harness(tmp_path, submitter, (BASELINE,))
    _write_launcher(harness / unsafe_path, unsafe_path)
    if selector_name == "argument":
        arguments = (unsafe_path,)
        extra_environment = None
    else:
        arguments = ()
        extra_environment = {"ACCURACY_SCRIPT": unsafe_path}

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        arguments=arguments,
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert "single persistent" in result.stderr
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    ("submitter", "arguments", "extra_environment"),
    (
        ("submit_smoke_matrix.sh", (), None),
        (
            "submit_performance_matrix.sh",
            ("scopes/17_attn.sh",),
            None,
        ),
        (
            "submit_accuracy_soak.sh",
            (),
            {"ACCURACY_SCRIPT": "scopes/17_attn.sh"},
        ),
    ),
)
def test_submitter_rejects_launcher_symlink_escape(
    tmp_path: Path,
    submitter: str,
    arguments: tuple[str, ...],
    extra_environment: dict[str, str] | None,
) -> None:
    harness, capture_file = _make_harness(tmp_path, submitter, (BASELINE,))
    outside = tmp_path / "outside.sh"
    _write_launcher(outside, "outside.sh")
    escaped_launcher = harness / "scopes" / "17_attn.sh"
    escaped_launcher.parent.mkdir(exist_ok=True)
    escaped_launcher.symlink_to(outside)

    result = _run_submitter(
        harness,
        submitter,
        capture_file,
        arguments=arguments,
        extra_environment=extra_environment,
    )

    assert result.returncode == 2
    assert "escapes" in result.stderr
    assert _captured_rows(capture_file) == []


@pytest.mark.parametrize(
    ("model", "expected_scopes"),
    (
        ("nano", NANO_PERFORMANCE_SCOPES),
        ("super", SUPER_PERFORMANCE_SCOPES),
    ),
)
def test_performance_defaults_submit_three_matched_model_compatible_repeats(
    tmp_path: Path, model: str, expected_scopes: tuple[str, ...]
) -> None:
    submitter = "submit_performance_matrix.sh"
    all_launchers = tuple(
        dict.fromkeys(
            (*NANO_PERFORMANCE_SCOPES, *SUPER_PERFORMANCE_SCOPES, *expected_scopes)
        )
    )
    harness, capture_file = _make_harness(tmp_path, submitter, all_launchers)

    result = _run_submitter(harness, submitter, capture_file, model=model)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert len(rows) == 3 * len(expected_scopes)
    assert {row[2] for row in rows} == {f"performance-{model}-nemorl-oci-hsg-unit"}
    for repeat_index in (1, 2, 3):
        repeat_rows = [row for row in rows if row[3] == str(repeat_index)]
        assert tuple(row[0] for row in repeat_rows) == expected_scopes
        assert {row[1] for row in repeat_rows} == {"20"}
        assert {row[4] for row in repeat_rows} == {f"unit-r{repeat_index}"}
        assert sum(row[0] == BASELINE for row in repeat_rows) == 1


def test_performance_custom_selection_deduplicates_baseline_per_repeat(
    tmp_path: Path,
) -> None:
    submitter = "submit_performance_matrix.sh"
    selected = (
        BASELINE,
        BASELINE,
        "scopes/17_attn.sh",
        "scopes/17_attn.sh",
    )
    harness, capture_file = _make_harness(
        tmp_path, submitter, (BASELINE, "scopes/17_attn.sh")
    )

    result = _run_submitter(harness, submitter, capture_file, arguments=selected)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    for repeat_index in (1, 2, 3):
        repeat_scopes = tuple(row[0] for row in rows if row[3] == str(repeat_index))
        assert repeat_scopes == (BASELINE, "scopes/17_attn.sh")


@pytest.mark.parametrize(
    ("model", "best_combined"),
    (
        ("nano", "scopes/31_attn_mlp_mamba_moe_router.sh"),
        ("super", "scopes/32_attn_mlp_mamba_moe_router_preprocess.sh"),
    ),
)
def test_accuracy_defaults_pair_baseline_and_best_combined_for_three_repeats(
    tmp_path: Path, model: str, best_combined: str
) -> None:
    submitter = "submit_accuracy_soak.sh"
    harness, capture_file = _make_harness(
        tmp_path, submitter, (BASELINE, best_combined)
    )

    result = _run_submitter(harness, submitter, capture_file, model=model)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert len(rows) == 6
    assert {row[2] for row in rows} == {f"accuracy-{model}-nemorl-oci-hsg-unit"}
    for repeat_index in (1, 2, 3):
        repeat_rows = [row for row in rows if row[3] == str(repeat_index)]
        assert tuple(row[0] for row in repeat_rows) == (BASELINE, best_combined)
        assert {row[1] for row in repeat_rows} == {"100"}
        assert {row[4] for row in repeat_rows} == {f"unit-r{repeat_index}"}


def test_smoke_keeps_five_step_batch_rows_without_repeats(tmp_path: Path) -> None:
    submitter = "submit_smoke_matrix.sh"
    launchers = (BASELINE, "scopes/17_attn.sh")
    harness, capture_file = _make_harness(tmp_path, submitter, launchers)

    result = _run_submitter(harness, submitter, capture_file)

    assert result.returncode == 0, result.stderr
    rows = _captured_rows(capture_file)
    assert tuple(row[0] for row in rows) == launchers
    assert {row[1] for row in rows} == {"5"}
