"""Contracts for the stable Q30 fixed-versus-always report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.qwen3_30ba3b_fixed_vs_always_stable_200step_20260827.reporting.fixed_vs_always_report import (
    aggregate_history,
    build_comparisons,
    build_report,
    render_html,
)


def history(multiplier: float = 1.0) -> list[dict[str, object]]:
    return [
        {
            "_step": 2,
            "performance/tokens_per_sec_per_gpu": 9999.0,
            "performance/generation_tokens_per_sec_per_gpu": 9999.0,
        },
        {
            "_step": 3,
            "performance/tokens_per_sec_per_gpu": 100.0 * multiplier,
            "performance/generation_tokens_per_sec_per_gpu": 200.0 * multiplier,
            "timing/train/total_step_time": 20.0 / multiplier,
            "timing/train/generation": 10.0 / multiplier,
            "timing/train/policy_training": 4.0,
            "timing/train/policy_and_reference_logprobs": 3.0,
            "timing/train/prepare_for_generation/total": 2.0,
            "train/vllm/spec_acceptance_rate": 0.5,
            "train/vllm/spec_acceptance_length": 2.5,
        },
        {
            "_step": 200,
            "performance/tokens_per_sec_per_gpu": 120.0 * multiplier,
            "performance/generation_tokens_per_sec_per_gpu": 240.0 * multiplier,
            "timing/train/total_step_time": 18.0 / multiplier,
            "timing/train/generation": 8.0 / multiplier,
            "timing/train/policy_training": None,
            "timing/train/policy_and_reference_logprobs": 3.0,
            "timing/train/prepare_for_generation/total": 2.0,
            "vllm/spec_acceptance_rate": 0.7,
            "vllm/spec_acceptance_length": 3.5,
        },
        {"_step": 201, "performance/tokens_per_sec_per_gpu": 9999.0},
    ]


def complete_summary(multiplier: float) -> dict[str, object]:
    rows = []
    for step in range(3, 201):
        rows.append(
            {
                "_step": step,
                "performance/tokens_per_sec_per_gpu": 100.0 * multiplier,
                "performance/generation_tokens_per_sec_per_gpu": 200.0 * multiplier,
                "timing/train/total_step_time": 20.0 / multiplier,
                "timing/train/generation": 10.0 / multiplier,
            }
        )
    return aggregate_history(rows, 3, 200)


def test_aggregate_history_uses_closed_window_and_canonical_throughput() -> None:
    summary = aggregate_history(history(), 3, 200)

    assert summary["window"] == {"start_step": 3, "end_step": 200, "step_count": 198}
    assert summary["included_steps"] == [3, 200]
    assert summary["missing_steps"][0] == 4
    assert summary["missing_steps"][-1] == 199
    assert summary["completed"] is False
    assert summary["metrics"]["e2e_throughput_per_gpu"] == {
        "mean": 110.0,
        "valid_count": 2,
    }
    assert summary["metrics"]["generation_throughput_per_gpu"] == {
        "mean": 220.0,
        "valid_count": 2,
    }
    assert summary["metrics"]["policy_training_time_s"] == {
        "mean": 4.0,
        "valid_count": 1,
    }


def test_aggregate_history_never_reconstructs_missing_throughput() -> None:
    summary = aggregate_history(
        [{"_step": 3, "timing/train/total_step_time": 2.0}], 3, 3
    )

    assert summary["metrics"]["e2e_throughput_per_gpu"] == {
        "mean": None,
        "valid_count": 0,
    }


def test_comparisons_are_always_relative_to_same_drafter_fixed_arm() -> None:
    comparisons = build_comparisons(
        [
            {
                "variant": "dflash-fixed",
                "state": "finished",
                "summary": complete_summary(1.0),
            },
            {
                "variant": "dflash-always",
                "state": "finished",
                "summary": complete_summary(1.25),
            },
            {
                "variant": "dspark-fixed",
                "state": "finished",
                "summary": complete_summary(2.0),
            },
            {
                "variant": "dspark-always",
                "state": "finished",
                "summary": complete_summary(3.0),
            },
        ]
    )

    assert comparisons == [
        {
            "variant": "dflash-always",
            "fixed_baseline": "dflash-fixed",
            "status": "ready",
            "generation_throughput_speedup": 1.25,
            "generation_time_speedup": 1.25,
            "e2e_throughput_speedup": 1.25,
            "e2e_step_time_speedup": 1.25,
        },
        {
            "variant": "dspark-always",
            "fixed_baseline": "dspark-fixed",
            "status": "ready",
            "generation_throughput_speedup": 1.5,
            "generation_time_speedup": 1.5,
            "e2e_throughput_speedup": 1.5,
            "e2e_step_time_speedup": 1.5,
        },
    ]


def test_incomplete_pair_is_preliminary_without_speedup() -> None:
    comparisons = build_comparisons(
        [
            {
                "variant": "dflash-fixed",
                "state": "finished",
                "summary": complete_summary(1.0),
            },
            {
                "variant": "dflash-always",
                "state": "finished",
                "summary": aggregate_history(history(1.2), 3, 200),
            },
        ]
    )

    assert comparisons[0]["status"] == "preliminary"
    assert comparisons[0]["generation_throughput_speedup"] is None
    assert comparisons[0]["e2e_throughput_speedup"] is None


def test_failed_run_with_complete_history_is_never_ready() -> None:
    comparisons = build_comparisons(
        [
            {
                "variant": "dflash-fixed",
                "state": "finished",
                "summary": complete_summary(1.0),
            },
            {
                "variant": "dflash-always",
                "state": "failed",
                "summary": complete_summary(1.2),
            },
        ]
    )

    assert comparisons[0]["status"] == "preliminary"
    assert comparisons[0]["generation_time_speedup"] is None
    assert comparisons[0]["e2e_step_time_speedup"] is None


def test_sparse_comparison_metric_is_never_ready() -> None:
    sparse = complete_summary(1.2)
    sparse["metrics"]["generation_throughput_per_gpu"]["valid_count"] = 197
    comparisons = build_comparisons(
        [
            {
                "variant": "dflash-fixed",
                "state": "finished",
                "summary": complete_summary(1.0),
            },
            {
                "variant": "dflash-always",
                "state": "finished",
                "summary": sparse,
            },
        ]
    )

    assert comparisons[0]["status"] == "preliminary"
    assert comparisons[0]["generation_throughput_speedup"] is None
    assert comparisons[0]["e2e_throughput_speedup"] is None


def test_build_report_selects_latest_exact_launcher_retry() -> None:
    runs = [
        {
            "id": "old",
            "name": "q30ba3b-stable-200step-dflash-fixed-k5-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "created_at": "2026-08-27T01:00:00Z",
            "history": history(),
        },
        {
            "id": "new",
            "name": "q30ba3b-stable-200step-dflash-fixed-k5-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "created_at": "2026-08-27T02:00:00Z",
            "history": history(),
        },
        {"id": "foreign", "name": "dflash-fixed-other", "history": history()},
    ]

    report = build_report(runs, entity="nvidia", project="sna-specdec")

    assert len(report["runs"]) == 1
    assert report["runs"][0]["id"] == "new"
    assert report["runs"][0]["variant"] == "dflash-fixed"


def test_html_is_self_contained_and_discloses_steps_counts_and_methodology() -> None:
    summary = aggregate_history(history(), 3, 200)
    html = render_html(
        {
            "entity": "nvidia",
            "project": "sna-specdec",
            "group": "q30ba3b-fixed-vs-always-stable-200step-20260827",
            "runs": [{"variant": "dflash-fixed", "summary": summary}],
            "comparisons": [],
        }
    )

    assert "<style>" in html
    assert "https://" not in html
    assert "closed steps 3–200" in html
    assert "Included steps: 3, 200" in html
    assert "Missing steps: 4" in html
    assert "valid=2" in html
    assert "fixed means frozen drafter training" in html


def test_html_marks_failed_and_sparse_metric_runs_preliminary() -> None:
    sparse = complete_summary(1.2)
    sparse["metrics"]["e2e_step_time_s"]["valid_count"] = 197
    html = render_html(
        {
            "runs": [
                {
                    "variant": "dflash-always",
                    "state": "failed",
                    "summary": complete_summary(1.1),
                },
                {
                    "variant": "dspark-always",
                    "state": "finished",
                    "summary": sparse,
                },
            ],
            "comparisons": [],
        }
    )

    assert html.count("<td>preliminary</td>") == 2
    assert "<td>failed</td>" in html
    assert "<td>finished</td>" in html
    assert "3/4 full-window metrics" in html
    assert "<td>complete</td>" not in html


def test_offline_cli_writes_json_and_html(tmp_path: Path) -> None:
    module = Path(__file__).parents[1] / "fixed_vs_always_report.py"
    fixture = tmp_path / "history.json"
    fixture.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "id": "fixture",
                        "name": "q30ba3b-stable-200step-dflash-fixed-k5-cccccccccccccccccccccccccccccccc",
                        "created_at": "2026-08-27T03:00:00Z",
                        "history": history(),
                    }
                ]
            }
        )
    )
    json_output = tmp_path / "report.json"
    html_output = tmp_path / "report.html"

    result = subprocess.run(
        [
            sys.executable,
            str(module),
            "--history-json",
            str(fixture),
            "--json-output",
            str(json_output),
            "--html-output",
            str(html_output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(json_output.read_text())["runs"][0]["variant"] == "dflash-fixed"
    assert "Qwen3-30B-A3B" in html_output.read_text()
