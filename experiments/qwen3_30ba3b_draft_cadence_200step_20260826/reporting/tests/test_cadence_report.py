"""Contracts for the Qwen3-30B-A3B cadence report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.qwen3_30ba3b_draft_cadence_200step_20260826.reporting.cadence_report import (
    aggregate_history,
    build_comparisons,
    render_html,
)


FIXTURE = Path(__file__).parent / "fixtures" / "history.json"


def load_history() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text())["runs"][0]["history"]


def test_aggregate_history_uses_the_closed_3_to_200_window() -> None:
    """Catches a mutation that includes step 2 or 201, or excludes step 200."""
    summary = aggregate_history(load_history(), start_step=3, end_step=200)

    assert summary["window"] == {"start_step": 3, "end_step": 200, "step_count": 198}
    assert summary["included_steps"] == [3, 4, 200]
    assert summary["missing_steps"][0] == 5
    assert summary["missing_steps"][-1] == 199
    assert summary["metrics"]["e2e_throughput_per_gpu"] == {
        "mean": 200.0,
        "valid_count": 2,
    }
    assert summary["metrics"]["generation_throughput_per_gpu"] == {
        "mean": 100.0,
        "valid_count": 3,
    }


def test_aggregate_history_omits_nulls_per_metric_and_discloses_missing_steps() -> None:
    """Catches a mutation that treats null metrics as zero or hides missing steps."""
    summary = aggregate_history(load_history(), start_step=3, end_step=200)

    assert summary["metrics"]["e2e_step_time_s"] == {
        "mean": 15.0,
        "valid_count": 2,
    }
    assert summary["completed"] is False
    assert len(summary["missing_steps"]) == 195


def test_aggregate_history_requires_canonical_logged_throughput() -> None:
    """Catches a mutation that reconstructs throughput from token counts and time."""
    summary = aggregate_history(
        [
            {
                "_step": 3,
                "timing/train/total_step_time": 2.0,
                "train/mean_total_tokens_per_sample": 100.0,
            }
        ],
        start_step=3,
        end_step=3,
    )

    assert summary["metrics"]["e2e_throughput_per_gpu"] == {
        "mean": None,
        "valid_count": 0,
    }


def test_aggregate_history_accepts_logged_acceptance_key_aliases() -> None:
    """Catches a mutation that accepts only one historical W&B acceptance key."""
    summary = aggregate_history(load_history(), start_step=3, end_step=200)

    assert summary["metrics"]["acceptance_rate"] == {
        "mean": 0.5833333333333334,
        "valid_count": 3,
    }
    assert summary["metrics"]["mean_accepted_length"] == {
        "mean": 8.0,
        "valid_count": 3,
    }
    assert summary["cadence_reason_counts"] == {"always": 2, "fixed_interval": 1}


def test_build_comparisons_uses_the_matching_static_drafter() -> None:
    """Catches a mutation that compares an arm to the other drafter's static arm."""
    comparisons = build_comparisons(
        [
            {
                "variant": "dflash-static",
                "summary": {"completed": True, "metrics": {"e2e_throughput_per_gpu": {"mean": 100.0}, "generation_throughput_per_gpu": {"mean": 200.0}}},
            },
            {
                "variant": "dflash-always",
                "summary": {"completed": True, "metrics": {"e2e_throughput_per_gpu": {"mean": 125.0}, "generation_throughput_per_gpu": {"mean": 250.0}}},
            },
            {
                "variant": "dspark-static",
                "summary": {"completed": True, "metrics": {"e2e_throughput_per_gpu": {"mean": 50.0}, "generation_throughput_per_gpu": {"mean": 50.0}}},
            },
            {
                "variant": "dspark-fixed10",
                "summary": {"completed": True, "metrics": {"e2e_throughput_per_gpu": {"mean": 75.0}, "generation_throughput_per_gpu": {"mean": 100.0}}},
            },
        ]
    )

    assert comparisons == [
        {
            "variant": "dflash-always",
            "static_baseline": "dflash-static",
            "status": "ready",
            "e2e_throughput_speedup": 1.25,
            "generation_throughput_speedup": 1.25,
        },
        {
            "variant": "dspark-fixed10",
            "static_baseline": "dspark-static",
            "status": "ready",
            "e2e_throughput_speedup": 1.5,
            "generation_throughput_speedup": 2.0,
        },
    ]


def test_render_html_labels_incomplete_runs_and_waiting_baselines() -> None:
    """Catches a mutation that hides preliminary status or invents a baseline speedup."""
    summary = aggregate_history(load_history(), start_step=3, end_step=200)
    report = {
        "entity": "sna",
        "project": "sna-specdec",
        "group": "q30ba3b-draft-cadence-200step-20260826",
        "runs": [{"variant": "dflash-always", "summary": summary}],
        "comparisons": build_comparisons(
            [{"variant": "dflash-always", "summary": summary}]
        ),
    }

    html = render_html(report)

    assert "preliminary" in html
    assert "waiting static baseline" in html
    assert "Generation throughput" in html
    assert "E2E throughput" in html
    assert "cadence-relative comparison" in html


def test_cli_offline_output_never_serializes_wandb_api_key(tmp_path: Path) -> None:
    """Catches a mutation that copies WANDB_API_KEY into output or rendered HTML."""
    sentinel = "wandb-api-key-must-never-be-serialized"
    input_path = tmp_path / "runs.json"
    json_output = tmp_path / "report.json"
    html_output = tmp_path / "report.html"
    input_path.write_text(FIXTURE.read_text())
    module = Path(__file__).parents[1] / "cadence_report.py"
    result = subprocess.run(
        [
            sys.executable,
            str(module),
            "--entity",
            "sna",
            "--project",
            "sna-specdec",
            "--group",
            "q30ba3b-draft-cadence-200step-20260826",
            "--history-json",
            str(input_path),
            "--json-output",
            str(json_output),
            "--html-output",
            str(html_output),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "WANDB_API_KEY": sentinel},
    )

    assert result.returncode == 0, result.stderr
    assert sentinel not in result.stdout
    assert sentinel not in json_output.read_text()
    assert sentinel not in html_output.read_text()
