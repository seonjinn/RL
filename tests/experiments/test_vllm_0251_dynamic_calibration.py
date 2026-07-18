import hashlib
import json
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.calibrate_dynamic_sd import (
    derive_schedule,
    load_profile,
    write_schedule,
)


G_TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
G_DRAFTER_REVISION = "a1403e07b73a66fc9ef561463631c31864616933"


def _profile_payload() -> dict[str, object]:
    batch_sizes = [1, 4]
    k_values = [0, 1, 2, 3, 4, 5]
    rows = []
    for batch_size in batch_sizes:
        for k in k_values:
            baseline_itl = 10.0 if batch_size == 1 else 20.0
            rows.append(
                {
                    "batch_size": batch_size,
                    "k": k,
                    "median_itl_ms": baseline_itl + (k * 2.0),
                    "completed_batches": 20,
                }
            )
    return {
        "schema_version": 1,
        "calibration_status": "complete",
        "model_key": "qwen32",
        "target_revision": G_TARGET_REVISION,
        "drafter_revision": G_DRAFTER_REVISION,
        "runtime_vllm": "0.25.1",
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_name": "OpenMathInstruct-2",
        "dataset_revision": "469216e3f46f4dacf476b382e192485ea51a143e",
        "prompt_template_sha256": "1" * 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_model_len": 4096,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 4,
        "target_tensor_parallel_size": 2,
        "draft_tensor_parallel_size": 1,
        "num_batches_per_point": 20,
        "batch_sizes": batch_sizes,
        "k_values": k_values,
        "acceptance_rate_per_pos": [0.9, 0.8, 0.7, 0.6, 0.5],
        "rows": rows,
    }


def _write_profile(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(payload or _profile_payload(), sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_derivation_matches_accepted_length_over_interpolated_itl(
    tmp_path: Path,
) -> None:
    payload = _profile_payload()
    rows = payload["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        batch_size = row["batch_size"]
        k = row["k"]
        if batch_size == 1:
            row["median_itl_ms"] = [10.0, 11.0, 13.0, 16.0, 19.0, 18.0][k]
        else:
            row["median_itl_ms"] = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0][k]

    profile = load_profile(_write_profile(tmp_path, payload))
    schedule = derive_schedule(profile)

    # At BS=1, K5 has the highest AL/ITL. At BS=4, K0 wins. The
    # interpolated BS=2/3 crossover is computed from the same upstream rule.
    assert schedule.max_num_speculative_tokens == 5
    assert schedule.selected_k_by_batch[1] == 5
    assert schedule.selected_k_by_batch[4] == 0
    assert schedule.ranges[0].start_batch == 1
    assert schedule.ranges[-1].end_batch == 4


def test_exact_goodput_tie_prefers_lower_k(tmp_path: Path) -> None:
    payload = _profile_payload()
    payload["acceptance_rate_per_pos"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    rows = payload["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row["median_itl_ms"] = 10.0 if row["k"] == 0 else 20.0

    schedule = derive_schedule(load_profile(_write_profile(tmp_path, payload)))

    assert set(schedule.selected_k_by_batch.values()) == {0}


def test_minimum_gain_keeps_k0_when_speculation_gain_is_too_small(
    tmp_path: Path,
) -> None:
    payload = _profile_payload()
    payload["acceptance_rate_per_pos"] = [0.1, 0.0, 0.0, 0.0, 0.0]
    rows = payload["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row["median_itl_ms"] = 10.0 if row["k"] == 0 else 10.5

    profile = load_profile(_write_profile(tmp_path, payload))
    schedule = derive_schedule(profile, minimum_goodput_gain=0.10)

    assert set(schedule.selected_k_by_batch.values()) == {0}


def test_write_schedule_records_profile_hash_and_k5_contract(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    profile = load_profile(profile_path)
    schedule = derive_schedule(profile)
    output = tmp_path / "schedule.json"

    write_schedule(profile, schedule, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["max_num_speculative_tokens"] == 5
    assert payload["selection_metric"] == "accepted_length_over_median_itl"
    assert (
        payload["profile_sha256"]
        == hashlib.sha256(profile_path.read_bytes()).hexdigest()
    )
    assert payload["ranges"] == [
        [item.start_batch, item.end_batch, item.k] for item in schedule.ranges
    ]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload.update(calibration_status="partial"), "complete"),
        (
            lambda payload: payload.update(k_values=[0, 1, 2, 3, 5]),
            "K0 through max K",
        ),
        (lambda payload: payload.update(batch_sizes=[2, 4]), "batch size 1"),
        (lambda payload: payload.update(max_num_seqs=8), "max_num_seqs"),
        (
            lambda payload: payload.update(acceptance_rate_per_pos=[0.9, 0.8]),
            "acceptance",
        ),
        (lambda payload: payload["rows"].pop(), "complete grid"),
    ],
)
def test_incomplete_or_mismatched_profiles_fail_closed(
    tmp_path: Path,
    mutate: object,
    error: str,
) -> None:
    payload = _profile_payload()
    assert callable(mutate)
    mutate(payload)

    with pytest.raises(ValueError, match=error):
        load_profile(_write_profile(tmp_path, payload))


def test_duplicate_grid_cell_fails_closed(tmp_path: Path) -> None:
    payload = _profile_payload()
    rows = payload["rows"]
    assert isinstance(rows, list)
    rows[-1] = dict(rows[0])

    with pytest.raises(ValueError, match="duplicate"):
        load_profile(_write_profile(tmp_path, payload))
