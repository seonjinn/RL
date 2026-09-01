from __future__ import annotations

import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

import experiments.vllm_028_q30_sync_dynamicsd.submit as submit_module

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
from experiments.vllm_028_q30_sync_dynamicsd.calibrate import (
    CalibrationResultRow,
    calibrate_drafter,
)
from experiments.vllm_028_q30_sync_dynamicsd.results import (
    CudaGraphEvidence,
    DrafterTraceEvidence,
    RuntimeProvenance,
    SpecDecodeMetrics,
    WorkerResult,
    publish_worker_result,
    validate_result_payload,
    validate_spec_decode_metrics,
    validate_worker_result,
)
from experiments.vllm_028_q30_sync_dynamicsd.live_runner import (
    EvidenceUnavailableError,
    VllmOfflineEngine,
    build_speculative_config,
    load_real_prompt_manifest,
)
from experiments.vllm_028_q30_sync_dynamicsd.submit import (
    JobSpec,
    dispatch_scripts,
    load_cluster_config,
    render_adaptive_overlay,
    render_job_sbatch,
    render_stage,
)


ASSET_ROOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec/assets/"
    "q30-base-opb-drafters-s4166-eval-v1"
)
DEFAULT_RUN_ID = "local-test:worker-0:calibration_dflash_bs2_k2:attempt-0"


def _trace_payload(
    *,
    drafter: str = "dflash",
    configured_k: int = 2,
    kernel_count: int = 4,
    kernel_time_seconds: float = 0.4,
    run_id: str = DEFAULT_RUN_ID,
) -> dict[str, object]:
    executed = kernel_count > 0
    query_width = configured_k + 1 if drafter == "dflash" else configured_k
    output_width = configured_k
    return {
        "run_id": run_id,
        "source_kind": "nsys",
        "clock_domain": "monotonic",
        "artifact_uri": f"file:///traces/{drafter}-k{configured_k}.nsys-rep",
        "artifact_sha256": "4" * 64,
        "artifact_size_bytes": 4_096,
        "capture_start_offset_seconds": 0.0,
        "capture_end_offset_seconds": 2.5,
        "capture_duration_seconds": 2.5,
        "draft_kernel_count": kernel_count,
        "draft_kernel_time_seconds": kernel_time_seconds,
        "observed_query_width": query_width if executed else 0,
        "observed_output_width": output_width if executed else 0,
    }


def _trace_record(
    *,
    drafter: str = "dflash",
    configured_k: int = 2,
    kernel_count: int = 4,
    kernel_time_seconds: float = 0.4,
    run_id: str = DEFAULT_RUN_ID,
) -> DrafterTraceEvidence:
    executed = kernel_count > 0
    query_width = configured_k + 1 if drafter == "dflash" else configured_k
    return DrafterTraceEvidence(
        run_id=run_id,
        source_kind="nsys",
        clock_domain="monotonic",
        artifact_uri=f"file:///traces/{drafter}-k{configured_k}.nsys-rep",
        artifact_sha256="4" * 64,
        artifact_size_bytes=4_096,
        capture_start_offset_seconds=0.0,
        capture_end_offset_seconds=2.5,
        capture_duration_seconds=2.5,
        draft_kernel_count=kernel_count,
        draft_kernel_time_seconds=kernel_time_seconds,
        observed_query_width=query_width if executed else 0,
        observed_output_width=configured_k if executed else 0,
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
            "proposed_tokens": 8,
            "accepted_tokens": 4,
            "draft_iterations": 4,
            "selected_k_histogram": {"2": 4},
            "selected_verifier_k": 2,
            "configured_draft_k": 2,
            "drafter_trace": _trace_payload(),
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
                finish_seconds=1.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="delta epsilon",
                token_ids=(21, 22),
                finish_reason="eos",
                finish_seconds=2.5,
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
                finish_seconds=1.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="delta epsilon",
                token_ids=(21, 22),
                finish_reason="eos",
                finish_seconds=2.5,
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
        "draft_iterations": 2,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_k": 7,
        "drafter_trace": _trace_payload(
            configured_k=7,
            kernel_count=2,
            run_id="local-test:worker-0:calibration_dflash_bs2_k0:attempt-0",
        ),
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
    assert validated.spec_decode.proposed_tokens == 0
    assert validated.spec_decode.draft_iterations == 2
    assert validated.method_plan.physical_block_size == 8
    assert validated.spec_decode.configured_draft_k == 7
    assert validated.spec_decode.drafter_trace is not None
    assert validated.spec_decode.drafter_trace.observed_query_width == 8
    assert validated.spec_decode.drafter_trace.observed_output_width == 7
    assert validated.spec_decode.observed_drafter_execution is True
    assert validated.spec_decode.drafter_trace.draft_kernel_count == 2


@pytest.mark.parametrize(
    "metric_change",
    [
        {"selected_verifier_k": None},
        {"configured_draft_k": None},
        {"drafter_trace": None},
    ],
)
def test_k0_diagnostic_rejects_missing_independent_evidence(
    metric_change: dict[str, object],
) -> None:
    metric_evidence: dict[str, object] = {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "draft_iterations": 2,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_k": 7,
        "drafter_trace": _trace_payload(
            configured_k=7,
            kernel_count=2,
            run_id="local-test:worker-0:calibration_dflash_bs2_k0:attempt-0",
        ),
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
        "draft_iterations": 2,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_k": 7,
        "drafter_trace": _trace_payload(
            configured_k=7,
            kernel_count=2,
            run_id="local-test:worker-0:calibration_dflash_bs2_k0:attempt-0",
        ),
    }
    contract, plan, manifest, result = _complete_worker_result(
        verifier_k=0,
        metric_evidence=metric_evidence,
    )
    assert result.spec_decode.drafter_trace is not None
    counters_only_trace = replace(
        result.spec_decode.drafter_trace,
        source_kind="spec_decode_counters",
        draft_kernel_count=0,
        draft_kernel_time_seconds=0.0,
        observed_query_width=0,
        observed_output_width=0,
    )
    counters_only = replace(result.spec_decode, drafter_trace=counters_only_trace)

    with pytest.raises(ValueError, match="source_kind"):
        validate_worker_result(
            replace(result, spec_decode=counters_only),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_s4166_drafter_capabilities_keep_checkpoint_k_and_widths_distinct() -> None:
    plans = {row.key: row for row in build_barrier_rows()}
    dflash = SpecDecodeMetrics(
        proposed_tokens=21,
        accepted_tokens=12,
        draft_iterations=4,
        selected_k_histogram={0: 1, 7: 3},
        selected_verifier_k=None,
        configured_draft_k=7,
        drafter_trace=_trace_record(drafter="dflash", configured_k=7),
    )
    dspark = SpecDecodeMetrics(
        proposed_tokens=24,
        accepted_tokens=13,
        draft_iterations=4,
        selected_k_histogram={0: 1, 8: 3},
        selected_verifier_k=None,
        configured_draft_k=8,
        drafter_trace=_trace_record(drafter="dspark", configured_k=8),
    )

    validate_spec_decode_metrics(dflash, plans["dflash_dynamicsd"])
    validate_spec_decode_metrics(dspark, plans["dspark_dynamicsd"])

    assert plans["dflash_dynamicsd"].physical_block_size == 8
    assert dflash.configured_draft_k == 7
    assert dflash.drafter_trace is not None
    assert dflash.drafter_trace.observed_query_width == 8
    assert dflash.drafter_trace.observed_output_width == 7
    assert plans["dspark_dynamicsd"].physical_block_size == 8
    assert dspark.configured_draft_k == 8
    assert dspark.drafter_trace is not None
    assert dspark.drafter_trace.observed_query_width == 8
    assert dspark.drafter_trace.observed_output_width == 8


def test_s4166_dflash_rejects_configured_k8_query_width9() -> None:
    plan = {row.key: row for row in build_barrier_rows()}["dflash_dynamicsd"]
    metrics = SpecDecodeMetrics(
        proposed_tokens=24,
        accepted_tokens=12,
        draft_iterations=4,
        selected_k_histogram={8: 3},
        selected_verifier_k=None,
        configured_draft_k=8,
        drafter_trace=_trace_record(drafter="dflash", configured_k=8),
    )

    with pytest.raises(ValueError, match="DFlash.*configured K"):
        validate_spec_decode_metrics(metrics, plan)


def test_baseline_requires_zero_spec_counters_empty_histogram_and_no_trace() -> None:
    plan = build_barrier_rows()[0]
    baseline = SpecDecodeMetrics(
        proposed_tokens=0,
        accepted_tokens=0,
        draft_iterations=0,
        selected_k_histogram={},
        selected_verifier_k=None,
        configured_draft_k=None,
        drafter_trace=None,
    )

    validate_spec_decode_metrics(baseline, plan)

    invalid_rows = (
        replace(baseline, proposed_tokens=1),
        replace(baseline, accepted_tokens=1, proposed_tokens=1),
        replace(baseline, draft_iterations=1),
        replace(baseline, selected_k_histogram={0: 1}),
        replace(baseline, selected_verifier_k=0),
        replace(baseline, configured_draft_k=1),
        replace(baseline, drafter_trace=_trace_record()),
    )
    for invalid in invalid_rows:
        with pytest.raises(ValueError, match="baseline"):
            validate_spec_decode_metrics(invalid, plan)


@pytest.mark.parametrize(
    "histogram",
    [{}, {2: 0}, {3: 4}],
)
def test_fixed_selected_k_histogram_requires_positive_exact_observations(
    histogram: dict[int, int],
) -> None:
    plan = _calibration_plan(2)
    metrics = SpecDecodeMetrics(
        proposed_tokens=8,
        accepted_tokens=4,
        draft_iterations=4,
        selected_k_histogram=histogram,
        selected_verifier_k=2,
        configured_draft_k=2,
        drafter_trace=_trace_record(configured_k=2),
    )

    with pytest.raises(ValueError, match="selected-K"):
        validate_spec_decode_metrics(metrics, plan)


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_k_histogram": {}},
        {"selected_k_histogram": {8: 0}},
        {"selected_k_histogram": {9: 1}, "draft_iterations": 1},
        {"selected_k_histogram": {0: 1, 8: 2}, "draft_iterations": 3},
    ],
)
def test_dynamic_selected_k_histogram_is_nonempty_positive_bounded_and_consistent(
    changes: dict[str, object],
) -> None:
    plan = {row.key: row for row in build_barrier_rows()}["dspark_dynamicsd"]
    valid = SpecDecodeMetrics(
        proposed_tokens=24,
        accepted_tokens=13,
        draft_iterations=4,
        selected_k_histogram={0: 1, 8: 3},
        selected_verifier_k=None,
        configured_draft_k=8,
        drafter_trace=_trace_record(drafter="dspark", configured_k=8),
    )

    with pytest.raises(ValueError, match="selected-K"):
        validate_spec_decode_metrics(replace(valid, **changes), plan)


@pytest.mark.parametrize(
    "trace_change",
    [
        {"run_id": ""},
        {"source_kind": "spec_decode_counters"},
        {"clock_domain": "wall"},
        {"artifact_uri": ""},
        {"artifact_sha256": "A" * 64},
        {"artifact_size_bytes": 0},
        {"capture_duration_seconds": 2.0},
        {"draft_kernel_count": 0},
        {"draft_kernel_time_seconds": 0.0},
        {"draft_kernel_time_seconds": True},
    ],
)
def test_trace_evidence_requires_reproducible_consistent_artifact(
    trace_change: dict[str, object],
) -> None:
    plan = _calibration_plan(2)
    trace = replace(_trace_record(), **trace_change)
    metrics = SpecDecodeMetrics(8, 4, 4, {2: 4}, 2, 2, trace)

    with pytest.raises(ValueError, match="trace"):
        validate_spec_decode_metrics(metrics, plan)


def test_k0_absence_is_valid_only_with_zero_width_direct_trace() -> None:
    plan = _calibration_plan(0)
    trace = _trace_record(
        configured_k=7,
        kernel_count=0,
        kernel_time_seconds=0.0,
    )
    metrics = SpecDecodeMetrics(0, 0, 2, {0: 2}, 0, 7, trace)

    validate_spec_decode_metrics(metrics, plan)

    assert metrics.observed_drafter_execution is False
    assert trace.observed_query_width == trace.observed_output_width == 0


def test_runtime_provenance_is_immutable() -> None:
    _, _, _, result = _complete_worker_result()

    with pytest.raises(FrozenInstanceError):
        result.runtime_provenance.vllm_version = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.runtime_provenance.cuda_graph_evidence.mode = "EAGER"  # type: ignore[misc]
    assert result.spec_decode.drafter_trace is not None
    with pytest.raises(FrozenInstanceError):
        result.spec_decode.drafter_trace.artifact_uri = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.run_id = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.spec_decode.drafter_trace.run_id = "mutated"  # type: ignore[misc]


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


def test_json_trace_round_trip_preserves_artifact_and_rejects_tampering() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    payload = json.loads(json.dumps(result.to_payload()))
    spec_decode = payload["spec_decode"]
    assert isinstance(spec_decode, dict)
    trace = spec_decode["drafter_trace"]
    assert isinstance(trace, dict)

    assert trace == _trace_payload()
    assert spec_decode["observed_drafter_execution"] is True

    trace["artifact_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="trace artifact_sha256"):
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
                finish_seconds=1.0,
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
                finish_seconds=1.0,
            ),
            EngineCompletion(
                request_id="request-0001",
                text="second completion",
                token_ids=(2,),
                finish_reason="eos",
                finish_seconds=2.0,
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


@pytest.mark.parametrize(
    "finish_values",
    [(1.0, float("inf")), (-0.1, 1.0), (1.0, 2.6)],
)
def test_completion_finish_durations_are_finite_and_bounded(
    finish_values: tuple[float, float],
) -> None:
    contract, plan, manifest, result = _complete_worker_result()
    invalid_rows = (
        replace(result.rows[0], finish_seconds=finish_values[0]),
        replace(result.rows[1], finish_seconds=finish_values[1]),
    )

    with pytest.raises(ValueError, match="finish_seconds"):
        validate_worker_result(
            replace(result, rows=invalid_rows),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_completion_finish_durations_allow_independent_out_of_order_requests() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    out_of_order_rows = (
        replace(result.rows[0], finish_seconds=2.0),
        replace(result.rows[1], finish_seconds=1.0),
    )

    validated = validate_worker_result(
        replace(result, rows=out_of_order_rows),
        contract=contract,
        plan=plan,
        prompt_manifest_sha256=manifest.sha256,
    )

    assert [row.global_request_index for row in validated.rows] == [0, 1]
    assert [row.finish_seconds for row in validated.rows] == [2.0, 1.0]


@pytest.mark.parametrize(
    "trace_span",
    [
        {
            "capture_start_offset_seconds": 0.0,
            "capture_end_offset_seconds": 0.1,
            "capture_duration_seconds": 0.1,
        },
        {
            "capture_start_offset_seconds": 0.1,
            "capture_end_offset_seconds": 2.5,
            "capture_duration_seconds": 2.4,
        },
    ],
)
def test_trace_must_cover_full_run_before_proving_drafter_absence(
    trace_span: dict[str, float],
) -> None:
    metric_evidence: dict[str, object] = {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "draft_iterations": 2,
        "selected_k_histogram": {"0": 2},
        "selected_verifier_k": 0,
        "configured_draft_k": 7,
        "drafter_trace": _trace_payload(
            configured_k=7,
            kernel_count=0,
            kernel_time_seconds=0.0,
            run_id="local-test:worker-0:calibration_dflash_bs2_k0:attempt-0",
        ),
    }
    contract, plan, manifest, result = _complete_worker_result(
        verifier_k=0,
        metric_evidence=metric_evidence,
    )
    assert result.spec_decode.drafter_trace is not None
    short_trace = replace(result.spec_decode.drafter_trace, **trace_span)

    with pytest.raises(ValueError, match="full generation run"):
        validate_worker_result(
            replace(
                result,
                spec_decode=replace(result.spec_decode, drafter_trace=short_trace),
            ),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


def test_result_payload_serializes_and_validates_matching_trace_run_id() -> None:
    contract, plan, manifest, result = _complete_worker_result()
    with pytest.raises(ValueError, match="run_id"):
        validate_worker_result(
            replace(result, run_id="wrong-run"),
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )
    payload = json.loads(json.dumps(result.to_payload()))
    expected_run_id = DEFAULT_RUN_ID
    spec_decode = payload["spec_decode"]
    assert isinstance(spec_decode, dict)
    trace = spec_decode["drafter_trace"]
    assert isinstance(trace, dict)

    assert payload.get("run_id") == expected_run_id
    assert trace.get("run_id") == expected_run_id

    trace["run_id"] = "wrong-run"
    with pytest.raises(ValueError, match="run_id"):
        validate_result_payload(
            payload,
            contract=contract,
            plan=plan,
            prompt_manifest_sha256=manifest.sha256,
        )


@pytest.mark.parametrize("method_key", ["fixed", "dynamic"])
def test_positive_spec_arm_requires_trace_observed_drafter_execution(
    method_key: str,
) -> None:
    plans = {row.key: row for row in build_barrier_rows()}
    if method_key == "fixed":
        plan = _calibration_plan(2)
        metrics = SpecDecodeMetrics(
            8,
            4,
            4,
            {2: 4},
            2,
            2,
            _trace_record(
                configured_k=2,
                kernel_count=0,
                kernel_time_seconds=0.0,
            ),
        )
    else:
        plan = plans["dflash_dynamicsd"]
        metrics = SpecDecodeMetrics(
            8,
            4,
            5,
            {0: 1, 2: 4},
            None,
            2,
            _trace_record(
                configured_k=2,
                kernel_count=0,
                kernel_time_seconds=0.0,
            ),
        )

    with pytest.raises(ValueError, match="drafter execution"):
        validate_spec_decode_metrics(metrics, plan)


def test_selected_k_proposal_accounting_rejects_mismatched_total() -> None:
    metrics = SpecDecodeMetrics(
        proposed_tokens=7,
        accepted_tokens=4,
        draft_iterations=4,
        selected_k_histogram={2: 4},
        selected_verifier_k=2,
        configured_draft_k=2,
        drafter_trace=_trace_record(configured_k=2),
    )

    with pytest.raises(ValueError, match="proposed_tokens"):
        validate_spec_decode_metrics(metrics, _calibration_plan(2))


def test_selected_k_decision_count_rejects_mismatched_draft_iterations() -> None:
    metrics = SpecDecodeMetrics(
        proposed_tokens=8,
        accepted_tokens=4,
        draft_iterations=3,
        selected_k_histogram={2: 4},
        selected_verifier_k=2,
        configured_draft_k=2,
        drafter_trace=_trace_record(configured_k=2),
    )

    with pytest.raises(ValueError, match="draft_iterations"):
        validate_spec_decode_metrics(metrics, _calibration_plan(2))


def test_calibration_selects_hand_calculated_monotone_schedule_and_best_fixed_k(
) -> None:
    k_values = (0, 1, 2, 3, 5, 7)
    throughput_by_batch_size = {
        1: (40.0, 50.0, 60.0, 80.0, 100.0, 90.0),
        2: (40.0, 50.0, 60.0, 75.0, 95.0, 100.0),
        4: (40.0, 50.0, 60.0, 80.0, 100.0, 90.0),
        8: (40.0, 50.0, 60.0, 80.0, 100.0, 90.0),
        16: (60.0, 70.0, 80.0, 100.0, 90.0, 80.0),
        32: (60.0, 70.0, 80.0, 100.0, 90.0, 80.0),
        64: (80.0, 90.0, 100.0, 100.0, 90.0, 80.0),
        96: (100.0, 95.0, 90.0, 85.0, 80.0, 75.0),
        128: (100.0, 95.0, 90.0, 85.0, 80.0, 75.0),
    }
    rows = tuple(
        CalibrationResultRow(
            drafter="dflash",
            batch_size=batch_size,
            verifier_k=verifier_k,
            repetition=1,
            output_tokens=1_000,
            elapsed_seconds=1_000 / throughput,
            validated=True,
        )
        for batch_size, throughputs in throughput_by_batch_size.items()
        for verifier_k, throughput in zip(k_values, throughputs, strict=True)
    )

    selection = calibrate_drafter(rows, drafter="dflash")

    assert selection.schedule == [
        [1, 8, 5],
        [9, 32, 3],
        [33, 64, 2],
        [65, 128, 0],
    ]
    assert selection.best_fixed_k == 5
    assert selection.throughput_objective == (
        "median per-result output_tokens / elapsed_seconds for each batch/K cell; "
        "equal-weight mean of cell medians across batch sizes for fixed K"
    )


def _uniform_calibration_rows(
    *,
    drafter: str = "dflash",
    repetitions: tuple[int, ...] = (1,),
) -> tuple[CalibrationResultRow, ...]:
    contract = ExperimentContract()
    return tuple(
        CalibrationResultRow(
            drafter=drafter,  # type: ignore[arg-type]
            batch_size=batch_size,
            verifier_k=verifier_k,
            repetition=repetition,
            output_tokens=1_000 * repetition,
            elapsed_seconds=10.0 * repetition,
            validated=True,
        )
        for repetition in repetitions
        for batch_size in contract.calibration_batch_sizes
        for verifier_k in contract.calibration_k_values
    )


def test_calibration_ties_choose_smaller_k_for_schedule_and_best_fixed() -> None:
    selection = calibrate_drafter(
        _uniform_calibration_rows(repetitions=(1, 2)),
        drafter="dflash",
    )

    assert selection.schedule == [[1, 128, 0]]
    assert selection.best_fixed_k == 0


def test_calibration_finds_global_monotone_throughput_optimum() -> None:
    def throughput_for(batch_size: int, verifier_k: int) -> int:
        if verifier_k == 0:
            return 10
        if verifier_k == 7:
            return 9 if batch_size == 1 else 100
        return 1

    contract = ExperimentContract()
    rows = tuple(
        CalibrationResultRow(
            drafter="dflash",
            batch_size=batch_size,
            verifier_k=verifier_k,
            repetition=1,
            output_tokens=throughput_for(batch_size, verifier_k),
            elapsed_seconds=1.0,
            validated=True,
        )
        for batch_size in contract.calibration_batch_sizes
        for verifier_k in contract.calibration_k_values
    )

    selection = calibrate_drafter(rows, drafter="dflash")

    assert selection.schedule == [[1, 128, 7]]
    assert selection.best_fixed_k == 7


def test_calibration_rejects_duplicate_or_missing_grid_cells() -> None:
    rows = _uniform_calibration_rows()

    with pytest.raises(ValueError, match="duplicate"):
        calibrate_drafter(rows + (rows[0],), drafter="dflash")
    with pytest.raises(ValueError, match="exact required BS/K grid"):
        calibrate_drafter(rows[:-1], drafter="dflash")


def test_calibration_rejects_incomplete_repetition_grid() -> None:
    rows = _uniform_calibration_rows()
    second_repetition = replace(rows[0], repetition=2)

    with pytest.raises(ValueError, match="repetition 2.*exact required BS/K grid"):
        calibrate_drafter(rows + (second_repetition,), drafter="dflash")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"validated": False}, "unvalidated"),
        ({"validated": 1}, "validated"),
        ({"batch_size": True}, "batch_size"),
        ({"verifier_k": True}, "verifier_k"),
        ({"repetition": 0}, "repetition"),
        ({"repetition": True}, "repetition"),
        ({"output_tokens": 0}, "output_tokens"),
        ({"output_tokens": True}, "output_tokens"),
        ({"elapsed_seconds": 0.0}, "elapsed_seconds"),
        ({"elapsed_seconds": float("inf")}, "elapsed_seconds"),
        ({"elapsed_seconds": True}, "elapsed_seconds"),
    ],
)
def test_calibration_rejects_invalid_or_unvalidated_results(
    changes: dict[str, object],
    message: str,
) -> None:
    rows = _uniform_calibration_rows()

    with pytest.raises(ValueError, match=message):
        calibrate_drafter(
            (replace(rows[0], **changes),) + rows[1:],
            drafter="dflash",
        )


def test_calibration_rejects_objects_not_marked_as_validated_rows() -> None:
    with pytest.raises(ValueError, match="validated CalibrationResultRow"):
        calibrate_drafter([{"validated": True}], drafter="dflash")  # type: ignore[list-item]


def test_calibration_rejects_mixed_or_unknown_drafters() -> None:
    rows = _uniform_calibration_rows()

    with pytest.raises(ValueError, match="one requested drafter"):
        calibrate_drafter(
            rows + (replace(rows[0], drafter="dspark"),),
            drafter="dflash",
        )
    with pytest.raises(ValueError, match="unsupported drafter"):
        calibrate_drafter(rows, drafter="other")  # type: ignore[arg-type]


def test_calibration_applies_method_capability_before_current_grid() -> None:
    dflash_rows = _uniform_calibration_rows()
    dspark_rows = _uniform_calibration_rows(drafter="dspark")

    with pytest.raises(ValueError, match="DFlash supports at most K7"):
        calibrate_drafter(
            (replace(dflash_rows[0], verifier_k=8),) + dflash_rows[1:],
            drafter="dflash",
        )
    with pytest.raises(ValueError, match="DSpark supports K8.*grid through K7"):
        calibrate_drafter(
            (replace(dspark_rows[0], verifier_k=8),) + dspark_rows[1:],
            drafter="dspark",
        )


def test_lyris_renderer_pins_safe_runtime_and_provenance_contract() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    contract = ExperimentContract()
    script = render_job_sbatch(
        JobSpec(
            key="canary_target_only",
            plan=build_barrier_rows()[0],
            nodes=1,
            gpus_per_node=1,
            worker_count=1,
            result_subdir="canary/target_only",
        ),
        cluster=cluster,
        source_commit="a" * 40,
    )

    assert cluster.account == "coreai_dlalgo_llm"
    assert cluster.partition == "gb200"
    assert cluster.remote_cwd.startswith("/home/")
    assert cluster.result_root.startswith("/lustre/")
    assert (
        "#SBATCH --job-name=coreai_dlalgo_llm-q30dyn.canary_target_only"
        in script
    )
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node" not in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --segment=1" in script
    assert "/raid/scratch/${USER}/q30-vllm028-${SLURM_JOB_ID}" in script
    assert contract.container_path in script
    assert contract.vllm_commit in script
    assert "patchset_manifest_sha256" in script
    assert "artifact_sha256" in script
    assert 'sha256sum "${TARGET_PATH}/config.json"' in script
    assert 'sha256sum "${PROMPT_JSONL}"' in script
    assert "--max-tokens 1024" in script
    assert "--temperature 1.0" in script
    assert "--top-p 1.0" in script
    assert "FULL_AND_PIECEWISE" in script
    assert "--data-parallel-size 1" in script
    assert "flashinfer_trtllm" in script
    assert "job-${SLURM_JOB_ID}" in script
    assert "Refusing to overwrite" in script
    assert "EXPECTED_SOURCE_COMMIT=" + "a" * 40 in script
    assert "runtime-provenance.json" in script
    assert "target_config_sha256" in script
    assert "prompt_source_sha256" in script


def test_renderer_emits_independent_calibration_and_external_barrier_jobs(
    tmp_path: Path,
) -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    calibration = render_stage(
        stage="calibration",
        output_dir=tmp_path / "calibration",
        cluster=cluster,
        source_commit="b" * 40,
    )
    assert len(calibration) == 108
    assert all("--dependency" not in path.read_text() for path in calibration)

    schedules = {
        "dflash": [[1, 8, 5], [9, 128, 0]],
        "dspark": [[1, 32, 3], [33, 128, 0]],
    }
    barrier = render_stage(
        stage="barrier",
        output_dir=tmp_path / "barrier",
        cluster=cluster,
        source_commit="b" * 40,
        schedules=schedules,
        best_fixed_k={"dflash": 5, "dspark": 3},
    )
    assert len(barrier) == 5
    dynamic_script = next(path for path in barrier if "dflash_dynamicsd" in path.name)
    text = dynamic_script.read_text()
    assert "#SBATCH --nodes=4" in text
    assert "#SBATCH --gpus-per-node" not in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --segment=4" in text
    assert "--ntasks=16" in text
    assert "--ntasks-per-node=4" in text
    assert "--gpus-per-task" not in text
    assert 'export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"' in text
    assert '--worker-index "${SLURM_PROCID}"' in text
    assert "--external-engine-count 16" in text
    assert "--requests-per-engine 128" in text
    assert '"num_speculative_tokens_per_batch_size":[[1,8,5],[9,128,0]]' in text

    with pytest.raises(FileExistsError, match="refusing to render"):
        render_stage(
            stage="barrier",
            output_dir=tmp_path / "barrier",
            cluster=cluster,
            source_commit="b" * 40,
            schedules=schedules,
            best_fixed_k={"dflash": 5, "dspark": 3},
        )


def test_renderer_emits_separate_one_gpu_canaries(tmp_path: Path) -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    scripts = render_stage(
        stage="canary",
        output_dir=tmp_path / "canary",
        cluster=cluster,
        source_commit="d" * 40,
        schedules={
            "dflash": [[1, 128, 5]],
            "dspark": [[1, 128, 3]],
        },
    )

    assert len(scripts) == 6
    assert all("#SBATCH --nodes=1" in path.read_text() for path in scripts)
    assert all("#SBATCH --gpus-per-node" not in path.read_text() for path in scripts)
    assert all("#SBATCH --exclusive" in path.read_text() for path in scripts)
    assert all("#SBATCH --segment=1" in path.read_text() for path in scripts)
    assert all("--gpus-per-task" not in path.read_text() for path in scripts)
    assert all(
        'export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"' in path.read_text()
        for path in scripts
    )
    names = {path.name for path in scripts}
    assert "canary_calibration_dflash_bs1_k0.sbatch" in names
    assert "canary_calibration_dspark_bs1_k0.sbatch" in names
    assert "canary_dspark_adaptive_verification.sbatch" in names


def test_renderer_keeps_k0_and_dspark_adaptive_separate_and_fail_closed() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    k0_plan = next(
        row
        for row in build_calibration_rows()
        if row.drafter == "dflash" and row.batch_size == 1 and row.verifier_k == 0
    )
    k0 = render_job_sbatch(
        JobSpec(
            key="trace_dflash_k0",
            plan=k0_plan,
            nodes=1,
            gpus_per_node=1,
            worker_count=1,
            result_subdir="trace/dflash-k0",
        ),
        cluster=cluster,
        source_commit="c" * 40,
    )
    assert "unsupported-receipt.json" in k0
    assert "exact selected-K and physical-width profiler integration unavailable" in k0
    assert "exit 2" in k0
    assert "enable_adaptive_verification" not in k0

    adaptive_plan = build_barrier_rows(include_dspark_adaptive=True)[-1]
    adaptive = render_job_sbatch(
        JobSpec(
            key="canary_dspark_adaptive",
            plan=adaptive_plan,
            nodes=1,
            gpus_per_node=1,
            worker_count=1,
            result_subdir="canary/adaptive",
        ),
        cluster=cluster,
        source_commit="c" * 40,
    )
    assert "enable_adaptive_verification" in adaptive
    assert "num_speculative_tokens_per_batch_size" not in adaptive
    assert "DSpark adaptive overlay" in adaptive


def test_dspark_adaptive_overlay_copies_checkpoint_and_changes_only_overlay_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_config = {"model_type": "dspark", "block_size": 8}
    (source / "config.json").write_text(json.dumps(source_config))
    overlay = tmp_path / "overlay"

    render_adaptive_overlay(source, overlay)

    assert json.loads((source / "config.json").read_text()) == source_config
    overlay_config = json.loads((overlay / "config.json").read_text())
    assert overlay_config["enable_confidence_head"] is True
    assert overlay_config["confidence_head_with_markov"] is True
    with pytest.raises(FileExistsError):
        render_adaptive_overlay(source, overlay)


def test_submit_modes_are_explicit_and_mockable(tmp_path: Path) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text("#!/usr/bin/env bash\ntrue\n")
    calls: list[list[str]] = []

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "--test-only" in argv:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr="sbatch: Job 98765 to start later\n",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="12345\n", stderr="")

    assert dispatch_scripts([script], mode="render", runner=runner) == []
    assert calls == []
    assert dispatch_scripts([script], mode="test-only", runner=runner) == []
    assert calls[-1] == ["sbatch", "--test-only", "--parsable", str(script)]
    assert dispatch_scripts([script], mode="submit", runner=runner) == ["12345"]
    assert calls[-1] == ["sbatch", "--parsable", str(script)]


def test_live_adapter_builds_method_aware_configs_and_explicit_g_copies() -> None:
    contract = ExperimentContract()
    dflash = next(
        row
        for row in build_calibration_rows()
        if row.drafter == "dflash" and row.batch_size == 2 and row.verifier_k == 5
    )
    assert build_speculative_config(dflash) == {
        "method": "dflash",
        "model": contract.drafter_paths["dflash"],
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
        "max_model_len": 4096,
    }
    dynamic = next(row for row in build_barrier_rows() if row.key == "dspark_dynamicsd")
    assert build_speculative_config(dynamic, [[1, 32, 3], [33, 128, 0]]) == {
        "method": "dspark",
        "model": contract.drafter_paths["dspark"],
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": [[1, 32, 3], [33, 128, 0]],
        "draft_tensor_parallel_size": 1,
        "attention_backend": "FLASH_ATTN",
        "max_model_len": 4096,
    }

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.sampling: list[FakeSamplingParams] = []

        def get_metrics(self) -> list[object]:
            return []

        def generate(
            self,
            prompts: list[str],
            sampling: list[FakeSamplingParams],
            *,
            use_tqdm: bool,
        ) -> list[object]:
            assert use_tqdm is False
            self.prompts = prompts
            self.sampling = sampling
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=f"o{index}",
                            token_ids=[index],
                            finish_reason="eos",
                        )
                    ],
                    metrics=SimpleNamespace(last_token_ts=101.0 + index),
                )
                for index in range(len(prompts))
            ]

    requests = (
        GenerationRequest("r0", 0, 0, 0, "same", 11, 1024, 1.0, 1.0, False),
        GenerationRequest("r1", 1, 0, 1, "same", 12, 1024, 1.0, 1.0, False),
    )
    fake = FakeLLM()
    engine = VllmOfflineEngine(
        fake,
        sampling_params_factory=FakeSamplingParams,
        monotonic=lambda: 100.0,
        plan=build_barrier_rows()[0],
    )
    run = engine.generate(requests)
    assert fake.prompts == ["same", "same"]
    assert [row.kwargs["seed"] for row in fake.sampling] == [11, 12]
    assert all(row.kwargs["n"] == 1 for row in fake.sampling)
    assert [row.request_id for row in run.completions] == ["r0", "r1"]
    assert [row.finish_seconds for row in run.completions] == [1.0, 2.0]

    raw_engine = VllmOfflineEngine(
        FakeLLM(),
        sampling_params_factory=FakeSamplingParams,
        monotonic=lambda: 100.0,
        plan=dynamic,
    )
    raw_completions, before, after = raw_engine.generate_raw(requests)
    assert [row.request_id for row in raw_completions] == ["r0", "r1"]
    assert before == after == []
    with pytest.raises(EvidenceUnavailableError, match="selected-K histogram"):
        raw_engine.normalize_metric_evidence(before=before, after=after)


def test_live_adapter_fails_closed_without_exact_dynamic_selected_k_evidence() -> None:
    dynamic = next(row for row in build_barrier_rows() if row.key == "dflash_dynamicsd")
    engine = VllmOfflineEngine(
        SimpleNamespace(get_metrics=lambda: []),
        sampling_params_factory=lambda **_: object(),
        monotonic=lambda: 0.0,
        plan=dynamic,
    )
    with pytest.raises(EvidenceUnavailableError, match="selected-K histogram"):
        engine.normalize_metric_evidence(before=[], after=[])


def test_real_prompt_loader_seals_exact_first_64_prompts(tmp_path: Path) -> None:
    source = tmp_path / "prompts.jsonl"
    source.write_text(
        "".join(json.dumps({"prompt": f"p{index}"}) + "\n" for index in range(65))
    )
    manifest, source_sha = load_real_prompt_manifest(source)
    assert manifest.prompts == tuple(f"p{index}" for index in range(64))
    assert len(source_sha) == 64


def test_evidence_disposition_allows_baseline_and_positive_fixed_raw_completion() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    baseline = render_job_sbatch(
        JobSpec("baseline", build_barrier_rows()[0], 1, 1, 1, "canary/baseline"),
        cluster=cluster,
        source_commit="a" * 40,
    )
    positive_fixed = next(
        row
        for row in build_calibration_rows()
        if row.drafter == "dflash" and row.batch_size == 2 and row.verifier_k == 5
    )
    fixed = render_job_sbatch(
        JobSpec("fixed", positive_fixed, 1, 1, 1, "calibration/fixed"),
        cluster=cluster,
        source_commit="a" * 40,
    )
    unresolved_plans = (
        next(
            row
            for row in build_calibration_rows()
            if row.drafter == "dflash" and row.batch_size == 1 and row.verifier_k == 0
        ),
        next(row for row in build_barrier_rows() if row.method == "dynamic"),
        build_barrier_rows(include_dspark_adaptive=True)[-1],
    )

    assert "exit 2" not in baseline
    assert "--evidence-status baseline_no_speculation" in baseline
    assert "exit 2" not in fixed
    assert "--evidence-status aggregate_fixed_k_counters_only" in fixed
    for index, plan in enumerate(unresolved_plans):
        script = render_job_sbatch(
            JobSpec(
                f"unresolved_{index}",
                plan,
                1,
                1,
                1,
                f"canary/u{index}",
                [[1, 128, 3]] if plan.method == "dynamic" else None,
            ),
            cluster=cluster,
            source_commit="a" * 40,
        )
        assert "exit 2" in script
        assert "selected_k_and_physical_trace_unavailable" in script


@pytest.mark.parametrize(
    ("drafter", "physical_k"),
    (("dflash", 7), ("dspark", 8)),
)
def test_k0_configures_physical_max_with_zero_verifier_schedule(
    drafter: str,
    physical_k: int,
) -> None:
    plan = next(
        row
        for row in build_calibration_rows()
        if row.drafter == drafter and row.batch_size == 1 and row.verifier_k == 0
    )
    config = build_speculative_config(plan)

    assert config is not None
    assert config["num_speculative_tokens"] == physical_k
    assert config["num_speculative_tokens_per_batch_size"] == [[1, 128, 0]]


def test_renderer_requires_exact_clean_source_and_small_container_receipt() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    script = render_job_sbatch(
        JobSpec("baseline", build_barrier_rows()[0], 1, 1, 1, "canary/baseline"),
        cluster=cluster,
        source_commit="b" * 40,
    )

    assert 'git -C "${REPO_ROOT}" status --porcelain --untracked-files=all' in script
    assert "source worktree is not clean" in script
    assert cluster.container_verification_receipt in script
    assert "container_size_bytes" in script
    assert "container_mtime_ns" in script
    assert "Path(sys.argv[2]).read_bytes()" not in script
    assert f"readonly REPO_ROOT='{cluster.remote_cwd}'" in script
    assert f"readonly STABLE_CONTAINER_IMAGE='{cluster.container_image}'" in script
    assert f"readonly PROMPT_JSONL='{cluster.prompt_jsonl}'" in script


def test_adaptive_worker_records_the_config_it_actually_loads() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    plan = build_barrier_rows(include_dspark_adaptive=True)[-1]
    script = render_job_sbatch(
        JobSpec("adaptive", plan, 1, 1, 1, "canary/adaptive"),
        cluster=cluster,
        source_commit="c" * 40,
    )

    assert "--export=ALL,NODE_LOCAL_ROOT" in script
    assert '--runtime-drafter-path "${RUNTIME_DRAFTER_PATH}"' in script
    assert "--runtime-drafter-config-sha256" in script
    assert "--source-drafter-config-sha256" in script
    assert 'sha256sum "${RUNTIME_DRAFTER_PATH}/config.json"' in script


@pytest.mark.parametrize(
    ("key", "result_subdir"),
    (
        ("bad;touch", "canary/good"),
        ("good", "../escape"),
        ("good", "canary/$(touch pwned)"),
        ("good", "/absolute"),
    ),
)
def test_renderer_rejects_job_identity_and_result_path_injection(
    key: str,
    result_subdir: str,
) -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    with pytest.raises(ValueError, match="key|result_subdir"):
        render_job_sbatch(
            JobSpec(key, build_barrier_rows()[0], 1, 1, 1, result_subdir),
            cluster=cluster,
            source_commit="d" * 40,
        )


def test_calibration_provenance_records_actual_request_count() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    plan = next(
        row
        for row in build_calibration_rows()
        if row.drafter == "dflash" and row.batch_size == 2 and row.verifier_k == 5
    )
    script = render_job_sbatch(
        JobSpec("calibration", plan, 1, 1, 1, "calibration/bs2"),
        cluster=cluster,
        source_commit="e" * 40,
    )

    assert "--requests-per-engine 2" in script
    assert '"requests_per_engine": 2' in script
    assert '"actual_request_count": 2' in script


def test_submit_failure_keeps_durable_prior_job_ids(tmp_path: Path) -> None:
    scripts = (tmp_path / "one.sbatch", tmp_path / "two.sbatch")
    for script in scripts:
        script.write_text("#!/usr/bin/env bash\ntrue\n")
    receipt = tmp_path / "submission.jsonl"
    callbacks: list[tuple[str, Path]] = []
    calls = 0

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, argv, stderr="rejected")
        return subprocess.CompletedProcess(argv, 0, stdout="12345\n", stderr="")

    with pytest.raises(submit_module.SubmissionDispatchError) as captured:
        dispatch_scripts(
            scripts,
            mode="submit",
            runner=runner,
            receipt_path=receipt,
            on_accepted=lambda job_id, script: callbacks.append((job_id, script)),
        )

    assert captured.value.accepted_job_ids == ("12345",)
    assert [json.loads(line) for line in receipt.read_text().splitlines()] == [
        {"job_id": "12345", "script": str(scripts[0])}
    ]
    assert callbacks == [("12345", scripts[0])]


def test_live_runtime_knobs_are_pinned_in_command_and_provenance() -> None:
    cluster = load_cluster_config(
        Path("experiments/vllm_028_q30_sync_dynamicsd/cluster-lyris.yaml")
    )
    script = render_job_sbatch(
        JobSpec("baseline", build_barrier_rows()[0], 1, 1, 1, "canary/baseline"),
        cluster=cluster,
        source_commit="f" * 40,
    )

    for expected in (
        "--dtype bfloat16",
        "--gpu-memory-utilization 0.9",
        "--max-num-batched-tokens 32768",
        "--disable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-model-len 4096",
        "#SBATCH --cpus-per-task=16",
    ):
        assert expected in script
