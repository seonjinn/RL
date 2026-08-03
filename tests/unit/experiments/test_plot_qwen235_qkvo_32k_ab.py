import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.plot_qwen235_qkvo_32k_ab import (
    load_correctness_gate,
    summarize_pair_result,
)


def test_summarize_pair_result_requires_matched_completed_work(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "arm": "baseline",
                        "complete": True,
                        "generation_seconds": 512.0,
                        "gpu_count": 8,
                        "measurement_scope": "generation_calls",
                        "output_tokens": 2_097_152,
                        "tokens_per_second_per_gpu": 512.0,
                    },
                    {
                        "arm": "adaptive",
                        "complete": True,
                        "generation_seconds": 480.0,
                        "gpu_count": 8,
                        "measurement_scope": "generation_calls",
                        "output_tokens": 2_097_152,
                        "tokens_per_second_per_gpu": 546.133333,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_pair_result(summary)

    assert result["matched_output_tokens"] == 2_097_152
    assert result["rows"][0]["normalized_throughput"] == 1.0
    assert result["rows"][1]["normalized_throughput"] == pytest.approx(
        1.066666666, rel=1e-6
    )


def test_load_correctness_gate_reports_matched_non_regression(tmp_path: Path) -> None:
    gate = tmp_path / "gsm8k_correctness_gate.json"
    gate.write_text(
        json.dumps(
            {
                "status": "pass",
                "row_count": 1319,
                "baseline_accuracy": 1000 / 1319,
                "adaptive_accuracy": 1002 / 1319,
                "absolute_accuracy_delta": 2 / 1319,
                "paired": {
                    "adaptive_gains": 10,
                    "adaptive_losses": 8,
                    "ties": 1301,
                    "one_sided_p_value": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_correctness_gate(gate)

    assert result["matched_examples"] == 1319
    assert result["passed"] is True
    assert result["adaptive_accuracy"] == pytest.approx(1002 / 1319)


def test_load_correctness_gate_rejects_unpassed_gate(tmp_path: Path) -> None:
    gate = tmp_path / "gsm8k_correctness_gate.json"
    gate.write_text(
        json.dumps(
            {
                "status": "fail",
                "row_count": 1319,
                "baseline_accuracy": 1000 / 1319,
                "adaptive_accuracy": 950 / 1319,
                "absolute_accuracy_delta": -50 / 1319,
                "paired": {
                    "adaptive_gains": 1,
                    "adaptive_losses": 51,
                    "ties": 1267,
                    "one_sided_p_value": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correctness gate did not pass"):
        load_correctness_gate(gate)
