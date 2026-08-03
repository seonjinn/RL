from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.plot_qwen235_nemorl_ab import (
    summarize_three_arm_result,
)


def _summary() -> dict[str, object]:
    return {
        "runs": [
            {
                "arm": "baseline",
                "complete": True,
                "measurement_scope": "generation_calls",
                "gpu_count": 8,
                "output_tokens": 262144,
                "generation_seconds": 320.0,
                "tokens_per_second_per_gpu": 102.4,
            },
            {
                "arm": "trtllm_default",
                "complete": True,
                "measurement_scope": "generation_calls",
                "gpu_count": 8,
                "output_tokens": 262144,
                "generation_seconds": 300.0,
                "tokens_per_second_per_gpu": 109.2,
            },
            {
                "arm": "adaptive",
                "complete": True,
                "measurement_scope": "generation_calls",
                "gpu_count": 8,
                "output_tokens": 262144,
                "generation_seconds": 290.0,
                "tokens_per_second_per_gpu": 113.0,
            },
        ]
    }


def test_summarize_three_arm_result_normalizes_to_cutedsl(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    result = summarize_three_arm_result(summary_path)

    assert [row["arm"] for row in result["rows"]] == [
        "baseline",
        "trtllm_default",
        "adaptive",
    ]
    assert result["rows"][0]["normalized_throughput"] == pytest.approx(1.0)
    assert result["rows"][2]["normalized_throughput"] == pytest.approx(
        113.0 / 102.4
    )
    assert result["matched_gpu_count"] == 8
    assert result["matched_output_tokens"] == 262144


def test_summarize_three_arm_result_rejects_unmatched_work(tmp_path: Path) -> None:
    summary = _summary()
    summary["runs"][2]["output_tokens"] = 1  # type: ignore[index]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="output-token counts"):
        summarize_three_arm_result(summary_path)
