import json
from pathlib import Path

import pytest

from experiments.mxfp8_adaptive_rollout_v0251.plot_qwen235_forced_32k_ab import (
    summarize_pair_result,
)


def test_summarize_pair_result_normalizes_to_cutedsl(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "arm": "baseline",
                        "complete": True,
                        "generation_seconds": 4.0,
                        "gpu_count": 8,
                        "measurement_scope": "generation_calls",
                        "output_tokens": 32768,
                        "tokens_per_second_per_gpu": 1024.0,
                    },
                    {
                        "arm": "adaptive",
                        "complete": True,
                        "generation_seconds": 5.0,
                        "gpu_count": 8,
                        "measurement_scope": "generation_calls",
                        "output_tokens": 32768,
                        "tokens_per_second_per_gpu": 819.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_pair_result(summary)

    assert result["matched_output_tokens"] == 32768
    assert result["rows"][0]["normalized_throughput"] == 1.0
    assert result["rows"][1]["normalized_throughput"] == pytest.approx(0.8)


def test_summarize_pair_result_rejects_unmatched_work(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "arm": arm,
                        "complete": True,
                        "generation_seconds": 1.0,
                        "gpu_count": 8,
                        "measurement_scope": "generation_calls",
                        "output_tokens": tokens,
                        "tokens_per_second_per_gpu": 1.0,
                    }
                    for arm, tokens in (("baseline", 32768), ("adaptive", 32767))
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output-token counts do not match"):
        summarize_pair_result(summary)
