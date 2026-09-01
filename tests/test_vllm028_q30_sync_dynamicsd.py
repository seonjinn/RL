from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from experiments.vllm_028_q30_sync_dynamicsd.contract import (
    ExperimentContract,
    MethodPlan,
    build_barrier_rows,
    build_calibration_rows,
)


ASSET_ROOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec/assets/"
    "q30-base-opb-drafters-s4166-eval-v1"
)


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
        "dspark_adaptive_verification",
    ]
    assert [(row.drafter, row.method, row.controller) for row in rows] == [
        (None, "baseline", "none"),
        ("dflash", "fixed", "fixed_k"),
        ("dflash", "dynamic", "dynamicsd"),
        ("dspark", "fixed", "fixed_k"),
        ("dspark", "dynamic", "dynamicsd"),
        ("dspark", "adaptive", "dspark_adaptive_verification"),
    ]
    assert rows[0].physical_block_size is None
    assert {row.physical_block_size for row in rows[1:]} == {8}
    assert len(rows) == len(set(rows))


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
