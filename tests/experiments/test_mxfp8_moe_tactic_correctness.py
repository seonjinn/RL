from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.mxfp8_moe_tactic_audit.compare_gsm8k import (
    DATASET_SHA256,
    compare_gsm8k,
    exact_mcnemar_p_value,
    main as compare_gsm8k_main,
    paired_bootstrap_ci,
)
from experiments.mxfp8_moe_tactic_audit.schema import TacticMeasurement, TacticPair
from experiments.mxfp8_moe_tactic_audit.validate_correctness import (
    Bf16PythonReferenceEvidence,
    MicroCorrectnessEvidence,
    MicroMeasurementEvidence,
    compare_generations,
    main as validate_correctness_main,
    validate_micro,
)


def _measurement(
    *,
    signature_key: str = "balanced",
    tactic: TacticPair = TacticPair(17, 23),
    finite: bool = True,
    deterministic: bool = True,
    max_abs_error: float = 0.01,
    cosine_similarity: float = 0.9999,
    failure: str | None = None,
) -> TacticMeasurement:
    return TacticMeasurement(
        signature_key=signature_key,
        tactic=tactic,
        median_us=100.0,
        p95_us=105.0,
        cv=0.01,
        warmups=3,
        repetitions=10,
        finite=finite,
        deterministic=deterministic,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
        failure=failure,
    )


def _complete_micro_evidence(
    measurements: list[TacticMeasurement],
) -> MicroCorrectnessEvidence:
    skew_classes = {
        "balanced": "balanced",
        "high-skew": "high-skew",
    }
    measurement_evidence = tuple(
        MicroMeasurementEvidence(
            signature_key=measurement.signature_key,
            tactic=measurement.tactic,
            skew_class=skew_classes[measurement.signature_key],  # type: ignore[arg-type]
            routing_counts_match=True,
            fc1_stock_compared=True,
            fc2_stock_compared=True,
            within_upstream_mxfp8_bounds=True,
        )
        for measurement in measurements
    )
    references = tuple(
        Bf16PythonReferenceEvidence(
            signature_key=row.signature_key,
            tactic=row.tactic,
            skew_class=row.skew_class,
            comparison_target="fc2_final",
            finite=True,
            max_abs_error=0.01,
            cosine_similarity=0.9999,
            within_upstream_mxfp8_bounds=True,
        )
        for row in measurement_evidence
    )
    return MicroCorrectnessEvidence(
        measurement_evidence=measurement_evidence,
        bf16_python_references=references,
    )


@pytest.mark.parametrize(
    ("measurement", "reason"),
    [
        (_measurement(finite=False), "nonfinite"),
        (_measurement(deterministic=False), "nondeterministic"),
        (_measurement(cosine_similarity=0.9989), "cosine"),
        (_measurement(max_abs_error=0.1001), "max_abs_error"),
        (_measurement(failure="routing count mismatch"), "routing count mismatch"),
        (
            _measurement(failure="flashinfer_intermediate_api_unavailable"),
            "stock comparison",
        ),
    ],
)
def test_micro_gate_rejects_promotion_blocking_rows(
    measurement: TacticMeasurement, reason: str
) -> None:
    summary = validate_micro([measurement])

    assert not summary.passed
    assert summary.checked_tactics == 1
    assert any(reason in failure for failure in summary.failures)


def test_micro_gate_rejects_nan_even_if_schema_is_bypassed() -> None:
    measurement = _measurement()
    object.__setattr__(measurement, "max_abs_error", float("nan"))

    summary = validate_micro([measurement])

    assert not summary.passed
    assert any("nonfinite" in failure for failure in summary.failures)


def test_micro_gate_one_argument_interface_fails_closed() -> None:
    summary = validate_micro([_measurement()])

    assert not summary.passed
    assert any("evidence" in failure for failure in summary.failures)


def test_micro_gate_accepts_complete_measurements_and_evidence() -> None:
    measurements = [
        _measurement(signature_key="balanced"),
        _measurement(signature_key="high-skew", tactic=TacticPair(31, 47)),
    ]

    summary = validate_micro(measurements, _complete_micro_evidence(measurements))

    assert summary.passed
    assert summary.checked_tactics == 2
    assert summary.failures == ()


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("routing_counts_match", "routing count mismatch"),
        ("fc1_stock_compared", "FC1 stock comparison"),
        ("fc2_stock_compared", "FC2 stock comparison"),
    ],
)
def test_programmatic_micro_gate_rejects_incomplete_measurement_evidence(
    field_name: str, reason: str
) -> None:
    measurements = [
        _measurement(signature_key="balanced"),
        _measurement(signature_key="high-skew", tactic=TacticPair(31, 47)),
    ]
    evidence = _complete_micro_evidence(measurements)
    first = replace(evidence.measurement_evidence[0], **{field_name: False})
    incomplete = replace(
        evidence,
        measurement_evidence=(first, *evidence.measurement_evidence[1:]),
    )

    summary = validate_micro(measurements, incomplete)

    assert not summary.passed
    assert any(reason in failure for failure in summary.failures)


@pytest.mark.parametrize(
    "reference_update",
    [
        {"signature_key": "unrelated"},
        {"tactic": TacticPair(99, 100)},
        {"skew_class": "high-skew"},
        {"comparison_target": "fc1_activated_intermediate"},
    ],
)
def test_micro_gate_rejects_unbound_or_non_fc2_representative_references(
    reference_update: dict[str, object],
) -> None:
    measurements = [
        _measurement(signature_key="balanced"),
        _measurement(signature_key="high-skew", tactic=TacticPair(31, 47)),
    ]
    evidence = _complete_micro_evidence(measurements)
    bad_reference = replace(evidence.bf16_python_references[0], **reference_update)
    incomplete = replace(
        evidence,
        bf16_python_references=(
            *evidence.bf16_python_references,
            bad_reference,
        ),
    )

    summary = validate_micro(measurements, incomplete)

    assert not summary.passed
    assert any("BF16/Python reference" in failure for failure in summary.failures)


def test_micro_gate_rejects_reference_tied_to_failed_measurement() -> None:
    measurements = [
        _measurement(signature_key="balanced", failure="kernel failed"),
        _measurement(signature_key="high-skew", tactic=TacticPair(31, 47)),
    ]

    summary = validate_micro(measurements, _complete_micro_evidence(measurements))

    assert not summary.passed
    assert any("balanced BF16/Python" in failure for failure in summary.failures)


def _generation_provenance() -> dict[str, object]:
    return {
        "model_revision": "qwen3-30ba3b-revision",
        "tokenizer_revision": "qwen3-tokenizer-revision",
        "runtime_fingerprint": "runtime-sha256",
        "decoding": {
            "mode": "greedy",
            "temperature": 0,
            "top_p": 1,
            "seed": 17,
            "max_tokens": 64,
        },
    }


def _write_generation(
    path: Path,
    rows: list[tuple[str, str, list[int]]],
    *,
    provenance: dict[str, object] | None = None,
) -> None:
    run_provenance = provenance or _generation_provenance()
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": identifier,
                    "prompt_sha256": prompt_sha256,
                    "token_ids": token_ids,
                    "provenance": run_provenance,
                },
                sort_keys=True,
            )
            + "\n"
            for identifier, prompt_sha256, token_ids in rows
        ),
        encoding="ascii",
    )


def test_generation_comparison_reports_exact_mismatch_ids(tmp_path: Path) -> None:
    stock = tmp_path / "stock.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_generation(
        stock,
        [("example-b", "b" * 64, [1, 2]), ("example-a", "a" * 64, [3])],
    )
    _write_generation(
        candidate,
        [("example-b", "b" * 64, [1, 9]), ("example-a", "a" * 64, [3])],
    )

    comparison = compare_generations(stock, candidate)

    assert not comparison.passed
    assert comparison.compared_examples == 2
    assert comparison.mismatched_ids == ("example-b",)


def test_generation_comparison_requires_identical_provenance(tmp_path: Path) -> None:
    stock = tmp_path / "stock.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_generation(stock, [("example", "a" * 64, [1, 2])])
    candidate_provenance = _generation_provenance()
    decoding = dict(candidate_provenance["decoding"])  # type: ignore[arg-type]
    decoding["max_tokens"] = 65
    candidate_provenance["decoding"] = decoding
    _write_generation(
        candidate,
        [("example", "a" * 64, [1, 2])],
        provenance=candidate_provenance,
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        compare_generations(stock, candidate)

    assert (
        validate_correctness_main(
            ["generation", "--stock", str(stock), "--candidate", str(candidate)]
        )
        != 0
    )


def test_generation_comparison_rejects_non_greedy_decoding(tmp_path: Path) -> None:
    stock = tmp_path / "stock.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    provenance = _generation_provenance()
    decoding = dict(provenance["decoding"])  # type: ignore[arg-type]
    decoding["temperature"] = 0.5
    provenance["decoding"] = decoding
    _write_generation(stock, [("example", "a" * 64, [1])], provenance=provenance)
    _write_generation(candidate, [("example", "a" * 64, [1])], provenance=provenance)

    with pytest.raises(ValueError, match="temperature"):
        compare_generations(stock, candidate)


@pytest.mark.parametrize(
    ("field_name", "stock_value", "candidate_value"),
    [
        ("temperature", 0, False),
        ("top_p", 1, True),
    ],
)
def test_generation_rejects_boolean_numeric_decoding_fields(
    tmp_path: Path,
    field_name: str,
    stock_value: int,
    candidate_value: bool,
) -> None:
    stock = tmp_path / "stock.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    stock_provenance = _generation_provenance()
    stock_decoding = dict(stock_provenance["decoding"])  # type: ignore[arg-type]
    stock_decoding[field_name] = stock_value
    stock_provenance["decoding"] = stock_decoding
    candidate_provenance = _generation_provenance()
    candidate_decoding = dict(candidate_provenance["decoding"])  # type: ignore[arg-type]
    candidate_decoding[field_name] = candidate_value
    candidate_provenance["decoding"] = candidate_decoding
    _write_generation(stock, [("example", "a" * 64, [1])], provenance=stock_provenance)
    _write_generation(
        candidate,
        [("example", "a" * 64, [1])],
        provenance=candidate_provenance,
    )

    with pytest.raises(ValueError, match=field_name):
        compare_generations(stock, candidate)


def test_generation_compares_complete_config_with_json_type_sensitivity(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    stock_provenance = _generation_provenance()
    stock_decoding = dict(stock_provenance["decoding"])  # type: ignore[arg-type]
    stock_decoding["min_tokens"] = 0
    stock_provenance["decoding"] = stock_decoding
    candidate_provenance = _generation_provenance()
    candidate_decoding = dict(candidate_provenance["decoding"])  # type: ignore[arg-type]
    candidate_decoding["min_tokens"] = False
    candidate_provenance["decoding"] = candidate_decoding
    _write_generation(stock, [("example", "a" * 64, [1])], provenance=stock_provenance)
    _write_generation(
        candidate,
        [("example", "a" * 64, [1])],
        provenance=candidate_provenance,
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        compare_generations(stock, candidate)


def test_exact_mcnemar_handles_zero_and_one_sided_disagreements() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(1, 0) == 1.0
    assert exact_mcnemar_p_value(10, 0) == pytest.approx(2 / 2**10)
    assert exact_mcnemar_p_value(5, 5) == 1.0


def test_exact_mcnemar_is_symmetric_for_large_counts() -> None:
    forward = exact_mcnemar_p_value(47, 19)
    reverse = exact_mcnemar_p_value(19, 47)

    assert forward == reverse
    assert 0.0 < forward < 0.05


def test_paired_bootstrap_is_fixed_seed_and_paired() -> None:
    stock = [True] * 30 + [False] * 20
    candidate = [True] * 20 + [False] * 10 + [True] * 10 + [False] * 10

    first = paired_bootstrap_ci(stock, candidate)
    second = paired_bootstrap_ci(stock, candidate)

    assert first == second
    assert first[0] < 0 < first[1]


def _gsm8k_provenance() -> dict[str, object]:
    return {
        "evaluation_contract": {
            "dataset": {
                "name": "GSM8K",
                "revision": "openai_1319_immutable",
                "sha256": DATASET_SHA256,
                "total": 1319,
            },
            "model_revision": "qwen3-30ba3b-revision",
            "tokenizer_revision": "qwen3-tokenizer-revision",
            "generation_args": {
                "mode": "greedy",
                "temperature": 0,
                "top_p": 1,
                "seed": 17,
                "max_tokens": 256,
            },
            "runtime_fingerprint": {
                "vllm_commit": "a" * 40,
                "flashinfer_version": "0.6.13",
                "cuda": "13.0",
                "gpu": "GB200",
            },
        },
        "evaluator": {
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "model": "qwen3-30ba3b",
            "dataset_path": "/immutable/gsm8k.jsonl",
            "dataset_sha256": DATASET_SHA256,
            "limit": 1319,
            "seed": 17,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 256,
            "concurrency": 1,
            "timeout_seconds": 600.0,
            "system_prompt": "gsm8k-system-prompt",
            "scoring": "normalized_numeric_exact_match_v1",
        },
    }


def _canonical_sha256(value: object) -> str:
    import hashlib

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_gsm8k_result(
    root: Path,
    correct_ids: set[str],
    *,
    provenance: dict[str, object] | None = None,
) -> None:
    root.mkdir()
    run_provenance = provenance or _gsm8k_provenance()
    provenance_sha256 = _canonical_sha256(run_provenance)
    rows = [
        {
            "id": f"gsm8k-{index:04d}",
            "question_sha256": f"{index:064x}",
            "target": str(index),
            "model_output": str(index),
            "prediction": str(index),
            "correct": f"gsm8k-{index:04d}" in correct_ids,
            "invalid": False,
            "empty": False,
            "provenance_sha256": provenance_sha256,
        }
        for index in range(1319)
    ]
    correct = sum(row["correct"] is True for row in rows)
    (root / "per_example.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    (root / "results.json").write_text(
        json.dumps(
            {
                "exact_match": correct / 1319,
                "correct": correct,
                "total": 1319,
                "invalid_predictions": 0,
                "empty_predictions": 0,
                "elapsed_seconds": 1.0,
                "provenance_sha256": provenance_sha256,
                "provenance": run_provenance,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )


def test_compare_gsm8k_reports_exact_paired_counts(tmp_path: Path) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    both_correct = {f"gsm8k-{index:04d}" for index in range(1000)}
    stock_only = {f"gsm8k-{index:04d}" for index in range(1000, 1010)}
    candidate_only = {f"gsm8k-{index:04d}" for index in range(1010, 1020)}
    _write_gsm8k_result(stock, both_correct | stock_only)
    _write_gsm8k_result(candidate, both_correct | candidate_only)

    comparison = compare_gsm8k(stock, candidate)

    assert comparison.stock_accuracy == 1010 / 1319
    assert comparison.candidate_accuracy == 1010 / 1319
    assert comparison.both_correct == 1000
    assert comparison.candidate_only_wins == 10
    assert comparison.stock_only_wins == 10
    assert comparison.both_wrong == 299
    assert comparison.accuracy_delta == 0.0
    assert comparison.mcnemar_p_value == 1.0
    assert comparison.delta_ci95[0] < 0 < comparison.delta_ci95[1]
    assert comparison.provenance_matched
    assert comparison.passed


def test_compare_gsm8k_requires_immutable_dataset_sha(tmp_path: Path) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    _write_gsm8k_result(stock, set())
    bad_provenance = _gsm8k_provenance()
    contract = dict(bad_provenance["evaluation_contract"])  # type: ignore[arg-type]
    dataset = dict(contract["dataset"])  # type: ignore[arg-type]
    dataset["sha256"] = "0" * 64
    contract["dataset"] = dataset
    bad_provenance["evaluation_contract"] = contract
    _write_gsm8k_result(candidate, set(), provenance=bad_provenance)

    with pytest.raises(ValueError, match="immutable dataset SHA256"):
        compare_gsm8k(stock, candidate)

    assert (
        compare_gsm8k_main(["--stock", str(stock), "--candidate", str(candidate)]) != 0
    )


def test_compare_gsm8k_requires_exactly_1319_matched_ids(tmp_path: Path) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    _write_gsm8k_result(stock, set())
    _write_gsm8k_result(candidate, set())
    rows_path = candidate / "per_example.jsonl"
    rows = rows_path.read_text(encoding="ascii").splitlines()
    rows_path.write_text("\n".join(rows[:-1]) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="exactly 1319"):
        compare_gsm8k(stock, candidate)


def test_compare_gsm8k_rejects_runtime_or_generation_mismatch(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    _write_gsm8k_result(stock, set())
    candidate_provenance = _gsm8k_provenance()
    contract = dict(candidate_provenance["evaluation_contract"])  # type: ignore[arg-type]
    generation_args = dict(contract["generation_args"])  # type: ignore[arg-type]
    generation_args["max_tokens"] = 128
    contract["generation_args"] = generation_args
    candidate_provenance["evaluation_contract"] = contract
    evaluator = dict(candidate_provenance["evaluator"])  # type: ignore[arg-type]
    evaluator["max_tokens"] = 128
    candidate_provenance["evaluator"] = evaluator
    _write_gsm8k_result(candidate, set(), provenance=candidate_provenance)

    with pytest.raises(ValueError, match="generation arguments mismatch"):
        compare_gsm8k(stock, candidate)


@pytest.mark.parametrize(
    ("field_name", "stock_value", "candidate_value"),
    [
        ("temperature", 0, False),
        ("top_p", 1, True),
    ],
)
def test_compare_gsm8k_rejects_boolean_numeric_generation_fields(
    tmp_path: Path,
    field_name: str,
    stock_value: int,
    candidate_value: bool,
) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    stock_provenance = _gsm8k_provenance()
    stock_contract = dict(stock_provenance["evaluation_contract"])  # type: ignore[arg-type]
    stock_args = dict(stock_contract["generation_args"])  # type: ignore[arg-type]
    stock_args[field_name] = stock_value
    stock_contract["generation_args"] = stock_args
    stock_provenance["evaluation_contract"] = stock_contract
    stock_evaluator = dict(stock_provenance["evaluator"])  # type: ignore[arg-type]
    stock_evaluator[field_name] = stock_value
    stock_provenance["evaluator"] = stock_evaluator
    candidate_provenance = _gsm8k_provenance()
    candidate_contract = dict(candidate_provenance["evaluation_contract"])  # type: ignore[arg-type]
    candidate_args = dict(candidate_contract["generation_args"])  # type: ignore[arg-type]
    candidate_args[field_name] = candidate_value
    candidate_contract["generation_args"] = candidate_args
    candidate_provenance["evaluation_contract"] = candidate_contract
    candidate_evaluator = dict(candidate_provenance["evaluator"])  # type: ignore[arg-type]
    candidate_evaluator[field_name] = candidate_value
    candidate_provenance["evaluator"] = candidate_evaluator
    _write_gsm8k_result(stock, set(), provenance=stock_provenance)
    _write_gsm8k_result(candidate, set(), provenance=candidate_provenance)

    with pytest.raises(ValueError, match=field_name):
        compare_gsm8k(stock, candidate)


def test_compare_gsm8k_uses_type_sensitive_complete_generation_config(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    stock_provenance = _gsm8k_provenance()
    stock_contract = dict(stock_provenance["evaluation_contract"])  # type: ignore[arg-type]
    stock_args = dict(stock_contract["generation_args"])  # type: ignore[arg-type]
    stock_args["min_tokens"] = 0
    stock_contract["generation_args"] = stock_args
    stock_provenance["evaluation_contract"] = stock_contract
    candidate_provenance = _gsm8k_provenance()
    candidate_contract = dict(candidate_provenance["evaluation_contract"])  # type: ignore[arg-type]
    candidate_args = dict(candidate_contract["generation_args"])  # type: ignore[arg-type]
    candidate_args["min_tokens"] = False
    candidate_contract["generation_args"] = candidate_args
    candidate_provenance["evaluation_contract"] = candidate_contract
    _write_gsm8k_result(stock, set(), provenance=stock_provenance)
    _write_gsm8k_result(candidate, set(), provenance=candidate_provenance)

    with pytest.raises(ValueError, match="generation arguments mismatch"):
        compare_gsm8k(stock, candidate)


def test_compare_gsm8k_rejects_near_but_not_exact_aggregate_accuracy(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    _write_gsm8k_result(stock, set())
    _write_gsm8k_result(candidate, set())
    results_path = stock / "results.json"
    results = json.loads(results_path.read_text(encoding="ascii"))
    results["exact_match"] = 5e-16
    results_path.write_text(json.dumps(results), encoding="ascii")

    with pytest.raises(ValueError, match="exact_match"):
        compare_gsm8k(stock, candidate)


def test_gsm8k_statistical_gate_rejects_significant_regression(
    tmp_path: Path,
) -> None:
    stock = tmp_path / "stock"
    candidate = tmp_path / "candidate"
    stock_correct = {f"gsm8k-{index:04d}" for index in range(1000)}
    candidate_correct = {f"gsm8k-{index:04d}" for index in range(900)}
    _write_gsm8k_result(stock, stock_correct)
    _write_gsm8k_result(candidate, candidate_correct)

    comparison = compare_gsm8k(stock, candidate)

    assert comparison.candidate_only_wins == 0
    assert comparison.stock_only_wins == 100
    assert comparison.mcnemar_p_value < 0.05
    assert comparison.delta_ci95[1] < 0
    assert not comparison.passed
    assert (
        compare_gsm8k_main(["--stock", str(stock), "--candidate", str(candidate)]) != 0
    )


def test_bootstrap_ci_rejects_non_boolean_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        paired_bootstrap_ci([], [])
    with pytest.raises(ValueError, match="booleans"):
        paired_bootstrap_ci([True], [1])  # type: ignore[list-item]
    assert math.isfinite(exact_mcnemar_p_value(1319, 0))
