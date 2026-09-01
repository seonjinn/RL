from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Sequence

import pytest

from experiments.vllm_028_q30_sync_dynamicsd.contract import (
    ExperimentContract,
    MethodPlan,
    build_barrier_rows,
    build_calibration_rows,
)
from experiments.vllm_028_q30_sync_dynamicsd.benchmark import (
    EngineCompletion,
    EngineRun,
    GenerationRequest,
    PromptManifest,
    build_worker_requests,
    run_one_engine,
    seal_prompt_manifest,
)
from experiments.vllm_028_q30_sync_dynamicsd.results import (
    CudaGraphEvidence,
    RuntimeProvenance,
    WorkerResult,
    publish_worker_result,
    validate_result_payload,
    validate_worker_result,
)


ASSET_ROOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec/assets/"
    "q30-base-opb-drafters-s4166-eval-v1"
)


class _FakeLocalEngine:
    def __init__(
        self,
        completions: tuple[EngineCompletion, ...],
        *,
        metric_evidence: dict[str, object] | None = None,
    ) -> None:
        self.completions = completions
        self.requests: tuple[GenerationRequest, ...] = ()
        self.metric_evidence = metric_evidence or {
            "proposed_tokens": 16,
            "accepted_tokens": 9,
            "draft_iterations": 4,
            "selected_k_histogram": {"2": 4},
            "selected_verifier_k": 2,
            "configured_draft_width": 8,
            "physical_draft_width": 8,
            "observed_drafter_execution": True,
            "drafter_execution_evidence_source": "profiler_trace",
            "drafter_execution_count": 4,
        }

    def generate(self, requests: Sequence[GenerationRequest]) -> EngineRun:
        self.requests = tuple(requests)
        return EngineRun(
            completions=self.completions,
            metric_evidence=self.metric_evidence,
        )


def _runtime_provenance(
    contract: ExperimentContract,
    *,
    worker_index: int = 0,
    prompt_manifest_sha256: str,
) -> RuntimeProvenance:
    return RuntimeProvenance(
        container_path=contract.container_path,
        container_digest="sha256:" + "1" * 64,
        vllm_version=contract.vllm_version,
        vllm_commit=contract.vllm_commit,
        target_path=contract.target_path,
        target_config_sha256="2" * 64,
        drafter_path=contract.drafter_paths["dflash"],
        drafter_config_sha256="3" * 64,
        prompt_manifest_sha256=prompt_manifest_sha256,
        tensor_parallel_size=1,
        data_parallel_size=1,
        external_engine_count=16,
        worker_index=worker_index,
        slurm_job_id="local-test",
        cuda_graph_evidence=CudaGraphEvidence(
            mode="FULL_AND_PIECEWISE",
            target_full=True,
            target_piecewise=True,
            drafter_full=True,
            drafter_piecewise=True,
            drafter_decode_full=True,
        ),
    )


def _calibration_plan(verifier_k: int = 2) -> MethodPlan:
    return next(
        row
        for row in build_calibration_rows()
        if row.drafter == "dflash"
        and row.batch_size == 2
        and row.verifier_k == verifier_k
    )


def _complete_worker_result(
    *,
    verifier_k: int = 2,
    metric_evidence: dict[str, object] | None = None,
) -> tuple[ExperimentContract, MethodPlan, PromptManifest, WorkerResult]:
    contract = ExperimentContract()
    plan = _calibration_plan(verifier_k)
    manifest = seal_prompt_manifest(tuple(f"prompt-{index}" for index in range(64)))
    engine = _FakeLocalEngine(
        (
            EngineCompletion(
                request_id="request-0000",
                text="alpha beta gamma",
                token_ids=(11, 12, 13),
                finish_reason="eos",
                finished_at_seconds=101.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="delta epsilon",
                token_ids=(21, 22),
                finish_reason="eos",
                finished_at_seconds=102.5,
            ),
        ),
        metric_evidence=metric_evidence,
    )
    clock_values = iter((100.0, 102.5))
    result = run_one_engine(
        contract=contract,
        plan=plan,
        prompt_manifest=manifest,
        worker_index=0,
        engine=engine,
        runtime_provenance=_runtime_provenance(
            contract,
            prompt_manifest_sha256=manifest.sha256,
        ),
        clock=lambda: next(clock_values),
    )
    return contract, plan, manifest, result


def test_experiment_contract_pins_the_lyris_q30_workload_and_runtime() -> None:
    contract = ExperimentContract()

    assert contract.target_path == f"{ASSET_ROOT}/q30-base"
    assert contract.drafter_paths == {
        "dflash": f"{ASSET_ROOT}/dflash-s4166",
        "dspark": f"{ASSET_ROOT}/dspark-s4166",
    }
    assert contract.vllm_version == "0.28.0"
    assert contract.vllm_commit == "2cf0a6915ce544dc493a0990f2ea38d81601128a"
    assert contract.container_path == (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
        "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
    )
    assert contract.max_tokens == 1024
    assert contract.prompt_count == 64
    assert contract.generations_per_prompt == 32
    assert contract.global_request_count == 2_048
    assert contract.engine_count == 16
    assert contract.requests_per_engine == 128
    assert contract.tensor_parallel_size == 1
    assert contract.data_parallel_size == 1
    assert contract.engine_coordination == "external"
    assert contract.cuda_graph_mode == "FULL_AND_PIECEWISE"
    assert contract.temperature == 1.0
    assert contract.top_p == 1.0
    assert contract.drafter_block_sizes == {"dflash": 8, "dspark": 8}


def test_experiment_contract_is_frozen_and_rejects_contract_drift() -> None:
    contract = ExperimentContract()

    with pytest.raises(FrozenInstanceError):
        contract.max_tokens = 2_048  # type: ignore[misc]

    for field_name, invalid_value in (
        ("max_tokens", 2_048),
        ("engine_count", 15),
        ("requests_per_engine", 127),
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 2),
    ):
        with pytest.raises(ValueError, match=field_name):
            replace(contract, **{field_name: invalid_value})


def test_experiment_contract_defensively_freezes_caller_owned_mappings() -> None:
    drafter_paths = {
        "dflash": f"{ASSET_ROOT}/dflash-s4166",
        "dspark": f"{ASSET_ROOT}/dspark-s4166",
    }
    block_sizes = {"dflash": 8, "dspark": 8}
    contract = ExperimentContract(
        drafter_paths=drafter_paths,
        drafter_block_sizes=block_sizes,
    )

    drafter_paths["dflash"] = "/tmp/mutated-dflash"
    block_sizes["dspark"] = 1

    assert contract.drafter_paths["dflash"] == f"{ASSET_ROOT}/dflash-s4166"
    assert contract.drafter_block_sizes["dspark"] == 8
    with pytest.raises(TypeError):
        contract.drafter_paths["dflash"] = "/tmp/direct-mutation"  # type: ignore[index]


def test_experiment_contract_pins_unique_per_request_seeds_and_natural_eos() -> None:
    contract = ExperimentContract()
    seeds = tuple(
        contract.seed_for_request(global_request_index)
        for global_request_index in range(contract.global_request_count)
    )

    assert contract.seed_policy == "base_seed_plus_global_request_index"
    assert contract.base_seed == 20_260_901
    assert contract.ignore_eos is False
    assert seeds[0] == 20_260_901
    assert seeds[-1] == 20_262_948
    assert len(seeds) == len(set(seeds)) == 64 * 32

    for invalid_index in (-1, contract.global_request_count, True):
        with pytest.raises(ValueError, match="global_request_index"):
            contract.seed_for_request(invalid_index)

    with pytest.raises(ValueError, match="ignore_eos"):
        replace(contract, ignore_eos=True)


def test_calibration_rows_cover_every_batch_k_and_drafter_cell() -> None:
    contract = ExperimentContract()
    rows = build_calibration_rows(contract)

    assert contract.calibration_batch_sizes == (1, 2, 4, 8, 16, 32, 64, 96, 128)
    assert contract.calibration_k_values == (0, 1, 2, 3, 5, 7)
    assert len(rows) == 2 * 9 * 6
    assert {
        (row.drafter, row.batch_size, row.verifier_k)
        for row in rows
    } == {
        (drafter, batch_size, verifier_k)
        for drafter in ("dflash", "dspark")
        for batch_size in (1, 2, 4, 8, 16, 32, 64, 96, 128)
        for verifier_k in (0, 1, 2, 3, 5, 7)
    }
    assert all(row.stage == "calibration" for row in rows)
    assert all(row.method == "fixed" for row in rows)
    assert all(row.physical_block_size == 8 for row in rows)


def test_k0_calibration_cells_remain_drafter_diagnostic_rows() -> None:
    rows = build_calibration_rows()
    k0_rows = [row for row in rows if row.verifier_k == 0]

    assert len(k0_rows) == 18
    assert {row.drafter for row in k0_rows} == {"dflash", "dspark"}
    assert {row.controller for row in k0_rows} == {"k0_diagnostic"}
    assert {row.physical_block_size for row in k0_rows} == {8}
    assert not any(row.method == "baseline" for row in k0_rows)


def test_barrier_rows_keep_each_method_and_controller_arm_distinct() -> None:
    rows = build_barrier_rows()

    assert [row.key for row in rows] == [
        "target_only",
        "dflash_fixed_best",
        "dflash_dynamicsd",
        "dspark_fixed_best",
        "dspark_dynamicsd",
    ]
    assert [(row.drafter, row.method, row.controller) for row in rows] == [
        (None, "baseline", "none"),
        ("dflash", "fixed", "fixed_k"),
        ("dflash", "dynamic", "dynamicsd"),
        ("dspark", "fixed", "fixed_k"),
        ("dspark", "dynamic", "dynamicsd"),
    ]
    assert rows[0].physical_block_size is None
    assert {row.physical_block_size for row in rows[1:]} == {8}
    assert len(rows) == len(set(rows))


def test_dspark_adaptive_barrier_arm_requires_explicit_compatibility_opt_in() -> None:
    default_keys = {row.key for row in build_barrier_rows()}
    opted_in_rows = build_barrier_rows(include_dspark_adaptive=True)

    assert "dspark_adaptive_verification" not in default_keys
    assert opted_in_rows[-1] == MethodPlan(
        key="dspark_adaptive_verification",
        stage="barrier",
        drafter="dspark",
        method="adaptive",
        controller="dspark_adaptive_verification",
        batch_size=None,
        verifier_k=None,
        physical_block_size=8,
    )

    with pytest.raises(ValueError, match="include_dspark_adaptive"):
        build_barrier_rows(include_dspark_adaptive=1)  # type: ignore[arg-type]


def test_barrier_rows_materialize_only_fixed_arms_with_calibrated_k() -> None:
    rows = build_barrier_rows(best_fixed_k={"dflash": 5, "dspark": 0})
    by_key = {row.key: row for row in rows}

    assert by_key["dflash_fixed_best"].verifier_k == 5
    assert by_key["dflash_fixed_best"].controller == "fixed_k"
    assert by_key["dspark_fixed_best"].verifier_k == 0
    assert by_key["dspark_fixed_best"].controller == "k0_diagnostic"
    assert {
        row.key: row.verifier_k
        for row in rows
        if row.method != "fixed"
    } == {
        "target_only": None,
        "dflash_dynamicsd": None,
        "dspark_dynamicsd": None,
    }


@pytest.mark.parametrize(
    "best_fixed_k",
    [
        {"dflash": 5},
        {"dflash": 4, "dspark": 5},
        {"dflash": True, "dspark": 5},
    ],
)
def test_barrier_rows_reject_incomplete_or_non_grid_fixed_k(
    best_fixed_k: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="best_fixed_k"):
        build_barrier_rows(best_fixed_k=best_fixed_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"drafter": None},
        {"physical_block_size": None},
        {"controller": "none"},
        {"verifier_k": 4},
    ],
)
def test_method_plan_rejects_invalid_calibration_identity(
    changes: dict[str, object],
) -> None:
    row = build_calibration_rows()[0]

    with pytest.raises(ValueError):
        replace(row, **changes)


def test_method_plan_is_frozen() -> None:
    row: MethodPlan = build_barrier_rows()[1]

    with pytest.raises(FrozenInstanceError):
        row.method = "baseline"  # type: ignore[misc]


def test_one_engine_runner_partitions_seeded_requests_and_reports_literal_summary() -> None:
    contract = ExperimentContract()
    plan = next(
        row
        for row in build_calibration_rows(contract)
        if row.drafter == "dflash" and row.batch_size == 2 and row.verifier_k == 2
    )
    manifest = seal_prompt_manifest(tuple(f"prompt-{index}" for index in range(64)))
    engine = _FakeLocalEngine(
        (
            EngineCompletion(
                request_id="request-0000",
                text="alpha beta gamma",
                token_ids=(11, 12, 13),
                finish_reason="eos",
                finished_at_seconds=101.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="delta epsilon",
                token_ids=(21, 22),
                finish_reason="eos",
                finished_at_seconds=102.5,
            ),
        )
    )
    clock_values = iter((100.0, 102.5))

    result = run_one_engine(
        contract=contract,
        plan=plan,
        prompt_manifest=manifest,
        worker_index=0,
        engine=engine,
        runtime_provenance=_runtime_provenance(
            contract,
            prompt_manifest_sha256=manifest.sha256,
        ),
        clock=lambda: next(clock_values),
    )

    assert [request.prompt for request in engine.requests] == ["prompt-0", "prompt-0"]
    assert [request.seed for request in engine.requests] == [20_260_901, 20_260_902]
    assert [request.ignore_eos for request in engine.requests] == [False, False]
    assert result.to_payload()["summary"] == {
        "elapsed_seconds": 2.5,
        "barrier_seconds": 2.5,
        "request_count": 2,
        "output_tokens": 5,
        "output_tokens_per_second": 2.0,
    }
    assert [row.finish_seconds for row in result.rows] == [1.0, 2.5]
    assert [row.text for row in result.rows] == [
        "alpha beta gamma",
        "delta epsilon",
    ]
    assert [row.seed for row in result.rows] == [20_260_901, 20_260_902]
    assert [row.ignore_eos for row in result.rows] == [False, False]


def test_worker_result_validation_rejects_osl_above_1024() -> None:
    contract, plan, manifest, result = _complete_worker_result()

    with pytest.raises(ValueError, match="max_tokens"):
        validate_worker_result(
            replace(result, max_tokens=1_025),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_worker_result_validation_rejects_internal_dp_above_one() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    invalid_provenance = replace(result.runtime_provenance, data_parallel_size=2)

    with pytest.raises(ValueError, match="data_parallel_size"):
        validate_worker_result(
            replace(result, runtime_provenance=invalid_provenance),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_worker_result_validation_rejects_missing_full_and_piecewise_graph_evidence() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    invalid_graph = replace(
        result.runtime_provenance.cuda_graph_evidence,
        target_piecewise=False,
    )
    invalid_provenance = replace(
        result.runtime_provenance,
        cuda_graph_evidence=invalid_graph,
    )

    with pytest.raises(ValueError, match="FULL_AND_PIECEWISE"):
        validate_worker_result(
            replace(result, runtime_provenance=invalid_provenance),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


@pytest.mark.parametrize(
    ("provenance_change", "error_match"),
    [
        ({"vllm_version": "0.28.1"}, "vllm_version"),
        ({"vllm_commit": "wrong"}, "vllm_commit"),
        ({"target_path": "/wrong-target"}, "target_path"),
        ({"drafter_path": "/wrong-drafter"}, "drafter_path"),
    ],
)
def test_worker_result_validation_rejects_wrong_runtime_identity(
    provenance_change: dict[str, object],
    error_match: str,
) -> None:
    contract, plan, manifest, result = _complete_worker_result()
    invalid_provenance = replace(
        result.runtime_provenance,
        **provenance_change,
    )

    with pytest.raises(ValueError, match=error_match):
        validate_worker_result(
            replace(result, runtime_provenance=invalid_provenance),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_worker_result_validation_rejects_incomplete_rows_and_exact_work_mismatch() -> None:
    contract, plan, manifest, result = _complete_worker_result()

    with pytest.raises(ValueError, match="exact request work"):
        validate_worker_result(
            replace(result, rows=result.rows[:1]),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )
    with pytest.raises(ValueError, match="output_tokens"):
        validate_worker_result(
            replace(
                result,
                summary=replace(result.summary, output_tokens=6),
            ),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_fixed_result_validation_rejects_wrong_selected_k_histogram() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    wrong_k_metrics = replace(result.spec_decode, selected_k_histogram={0: 4})

    with pytest.raises(ValueError, match="selected-K"):
        validate_worker_result(
            replace(result, spec_decode=wrong_k_metrics),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_k0_diagnostic_keeps_verifier_width_and_execution_evidence_independent() -> None:
    metric_evidence: dict[str, object] = {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "draft_iterations": 0,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_width": 8,
        "physical_draft_width": 8,
        "observed_drafter_execution": True,
        "drafter_execution_evidence_source": "profiler_trace",
        "drafter_execution_count": 2,
    }
    contract, plan, manifest, result = _complete_worker_result(
        verifier_k=0,
        metric_evidence=metric_evidence,
    )

    validated = validate_worker_result(
        result,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=manifest.sha256,
    )

    assert validated.spec_decode.selected_verifier_k == 0
    assert validated.spec_decode.configured_draft_width == 8
    assert validated.spec_decode.physical_draft_width == 8
    assert validated.spec_decode.observed_drafter_execution is True
    assert validated.spec_decode.drafter_execution_count == 2


@pytest.mark.parametrize(
    "metric_change",
    [
        {"selected_verifier_k": None},
        {"configured_draft_width": None},
        {"physical_draft_width": None},
        {"observed_drafter_execution": None},
        {"drafter_execution_evidence_source": None},
        {"drafter_execution_count": None},
    ],
)
def test_k0_diagnostic_rejects_missing_independent_evidence(
    metric_change: dict[str, object],
) -> None:
    metric_evidence: dict[str, object] = {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "draft_iterations": 0,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_width": 8,
        "physical_draft_width": 8,
        "observed_drafter_execution": True,
        "drafter_execution_evidence_source": "profiler_trace",
        "drafter_execution_count": 2,
    }
    contract, plan, manifest, result = _complete_worker_result(
        verifier_k=0,
        metric_evidence=metric_evidence,
    )
    invalid_metrics = replace(result.spec_decode, **metric_change)

    with pytest.raises(ValueError, match="K0 diagnostic"):
        validate_worker_result(
            replace(result, spec_decode=invalid_metrics),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_k0_diagnostic_cannot_claim_kernel_absence_from_counters() -> None:
    metric_evidence: dict[str, object] = {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "draft_iterations": 0,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_width": 8,
        "physical_draft_width": 8,
        "observed_drafter_execution": True,
        "drafter_execution_evidence_source": "profiler_trace",
        "drafter_execution_count": 2,
    }
    contract, plan, manifest, result = _complete_worker_result(
        verifier_k=0,
        metric_evidence=metric_evidence,
    )
    counters_only = replace(
        result.spec_decode,
        observed_drafter_execution=False,
        drafter_execution_evidence_source="spec_decode_counters",
        drafter_execution_count=0,
    )

    with pytest.raises(ValueError, match="trace evidence"):
        validate_worker_result(
            replace(result, spec_decode=counters_only),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_runtime_provenance_is_immutable() -> None:
    _, _, _, result = _complete_worker_result()

    with pytest.raises(FrozenInstanceError):
        result.runtime_provenance.vllm_version = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.runtime_provenance.cuda_graph_evidence.mode = "EAGER"  # type: ignore[misc]


def test_worker_result_publication_is_atomic_and_never_clobbers(tmp_path: Path) -> None:
    contract, plan, manifest, result = _complete_worker_result()
    output = tmp_path / "worker-0.json"

    publish_worker_result(
        output,
        result,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=manifest.sha256,
    )
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        publish_worker_result(
            output,
            result,
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )

    assert output.read_bytes() == original
    assert list(tmp_path.glob("*.partial.*")) == []


def test_json_result_boundary_reconstructs_and_strictly_validates_payload() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    payload = json.loads(json.dumps(result.to_payload()))

    validated = validate_result_payload(
        payload,
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=manifest.sha256,
    )

    assert validated == result
    rows = payload["rows"]
    assert isinstance(rows, list)
    rows.pop()
    with pytest.raises(ValueError, match="exact request work"):
        validate_result_payload(
            payload,
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_prompt_manifest_must_be_sealed_complete_and_unmodified() -> None:
    contract = ExperimentContract()
    plan = _calibration_plan()
    complete = seal_prompt_manifest(
        tuple(f"prompt-{index}" for index in range(contract.prompt_count))
    )

    requests = build_worker_requests(contract, plan, complete, worker_index=0)

    assert len(requests) == 2
    for invalid_manifest in (
        PromptManifest(complete.prompts, "0" * 64),
        seal_prompt_manifest(complete.prompts[:-1]),
    ):
        with pytest.raises(ValueError, match="prompt manifest"):
            build_worker_requests(
                contract,
                plan,
                invalid_manifest,
                worker_index=0,
            )


def test_barrier_prompt_partition_is_disjoint_and_uses_global_seed_indices() -> None:
    contract = ExperimentContract()
    plan = build_barrier_rows(best_fixed_k={"dflash": 2, "dspark": 2})[1]
    manifest = seal_prompt_manifest(
        tuple(f"prompt-{index}" for index in range(contract.prompt_count))
    )

    worker_zero = build_worker_requests(contract, plan, manifest, worker_index=0)
    worker_fifteen = build_worker_requests(contract, plan, manifest, worker_index=15)

    assert (worker_zero[0].global_request_index, worker_zero[-1].global_request_index) == (
        0,
        127,
    )
    assert (
        worker_fifteen[0].global_request_index,
        worker_fifteen[-1].global_request_index,
    ) == (1_920, 2_047)
    assert worker_fifteen[0].prompt == "prompt-60"
    assert worker_fifteen[-1].prompt == "prompt-63"
    assert worker_fifteen[0].seed == 20_262_821
    assert worker_fifteen[-1].seed == 20_262_948


def test_runner_rejects_incomplete_engine_completion_rows() -> None:
    contract = ExperimentContract()
    plan = _calibration_plan()
    manifest = seal_prompt_manifest(tuple(f"prompt-{index}" for index in range(64)))
    engine = _FakeLocalEngine(
        (
            EngineCompletion(
                request_id="request-0000",
                text="only one completion",
                token_ids=(1, 2, 3),
                finish_reason="eos",
                finished_at_seconds=101.0,
            ),
        )
    )
    clock_values = iter((100.0, 102.5))

    with pytest.raises(ValueError, match="exact request work"):
        run_one_engine(
            contract=contract,
            plan=plan,
            prompt_manifest=manifest,
            worker_index=0,
            engine=engine,
            runtime_provenance=_runtime_provenance(
                contract,
                prompt_manifest_sha256=manifest.sha256,
            ),
            clock=lambda: next(clock_values),
        )


def test_runner_rejects_unknown_engine_completion_identity_cleanly() -> None:
    contract = ExperimentContract()
    plan = _calibration_plan()
    manifest = seal_prompt_manifest(tuple(f"prompt-{index}" for index in range(64)))
    engine = _FakeLocalEngine(
        (
            EngineCompletion(
                request_id="unknown-request",
                text="wrong request",
                token_ids=(1,),
                finish_reason="eos",
                finished_at_seconds=101.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="second completion",
                token_ids=(2,),
                finish_reason="eos",
                finished_at_seconds=102.0,
            ),
        )
    )
    clock_values = iter((100.0, 102.5))

    with pytest.raises(ValueError, match="exact request work"):
        run_one_engine(
            contract=contract,
            plan=plan,
            prompt_manifest=manifest,
            worker_index=0,
            engine=engine,
            runtime_provenance=_runtime_provenance(
                contract,
                prompt_manifest_sha256=manifest.sha256,
            ),
            clock=lambda: next(clock_values),
        )
