import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.plot_qwen30_nemorl_ab import (
    aggregate_summaries,
)


def _write_summary(
    path: Path,
    *,
    baseline_tps_gpu: float,
    adaptive_tps_gpu: float,
    output_tokens: int = 1000,
) -> None:
    payload = {
        "adaptive_vs_baseline_speedup": adaptive_tps_gpu / baseline_tps_gpu,
        "runs": [
            {
                "arm": "baseline",
                "complete": True,
                "generation_calls": 1,
                "generation_seconds": 10.0,
                "gpu_count": 8,
                "measurement_scope": "generation_calls",
                "output_tokens": output_tokens,
                "tokens_per_second_per_gpu": baseline_tps_gpu,
            },
            {
                "arm": "adaptive",
                "complete": True,
                "generation_calls": 1,
                "generation_seconds": 9.0,
                "gpu_count": 8,
                "measurement_scope": "generation_calls",
                "output_tokens": output_tokens,
                "tokens_per_second_per_gpu": adaptive_tps_gpu,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_summaries_reports_paired_and_median_results(tmp_path: Path) -> None:
    paths = [tmp_path / f"rep{index}.json" for index in range(1, 4)]
    _write_summary(paths[0], baseline_tps_gpu=100.0, adaptive_tps_gpu=99.0)
    _write_summary(paths[1], baseline_tps_gpu=101.0, adaptive_tps_gpu=101.0)
    _write_summary(paths[2], baseline_tps_gpu=102.0, adaptive_tps_gpu=100.0)

    result = aggregate_summaries(paths)

    assert len(result["repeats"]) == 3
    assert result["median"]["baseline_tokens_per_second_per_gpu"] == 101.0
    assert result["median"]["adaptive_tokens_per_second_per_gpu"] == 100.0
    assert result["median"]["paired_speedup"] == pytest.approx(0.99)


def test_aggregate_summaries_rejects_output_token_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "rep1.json"
    _write_summary(
        path,
        baseline_tps_gpu=100.0,
        adaptive_tps_gpu=101.0,
        output_tokens=1000,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"][1]["output_tokens"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="output token mismatch"):
        aggregate_summaries([path])
