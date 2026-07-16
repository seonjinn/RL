from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.collect_results import (
    IncompleteWindowError,
    NoMatchingBaselineError,
    load_steps,
    match_baseline,
    render_reports,
    summarize_steps,
)


FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "experiments/vllm_0251_drafter_matrix/fixtures/sample_steps.jsonl"
)


def test_summarize_steps_excludes_warmup_and_averages_each_metric() -> None:
    summary = summarize_steps(load_steps(FIXTURE_PATH))

    assert summary.step_start == 2
    assert summary.step_end == 20
    assert summary.step_count == 19
    assert not summary.is_partial
    assert summary.e2e_time_s == pytest.approx(200.0 / 19.0)
    assert summary.generation_time_s == pytest.approx(105.0 / 19.0)
    assert summary.policy_time_s == pytest.approx(48.0 / 19.0)
    assert summary.logprob_time_s == pytest.approx(29.0 / 19.0)
    assert summary.throughput_tps == pytest.approx(2000.0 / 19.0)
    assert summary.generation_ratio == pytest.approx(9.7 / 19.0)
    assert summary.acceptance_rate == pytest.approx(11.6 / 19.0)
    assert summary.mean_accepted_length == pytest.approx(59.0 / 19.0)


def test_summarize_steps_rejects_incomplete_default_window() -> None:
    rows = list(load_steps(FIXTURE_PATH))[:-1]

    with pytest.raises(IncompleteWindowError, match="missing steps: 20"):
        summarize_steps(rows)


def test_summarize_steps_marks_explicit_partial_window() -> None:
    rows = list(load_steps(FIXTURE_PATH))[:-1]

    summary = summarize_steps(rows, allow_partial=True)

    assert summary.is_partial
    assert summary.step_count == 18


@pytest.mark.parametrize(
    "field",
    (
        "model",
        "recipe",
        "vllm_version",
        "container",
        "cluster",
        "temperature",
        "top_p",
        "max_osl",
        "cuda_graph_mode",
    ),
)
def test_match_baseline_requires_every_exact_identity_field(field: str) -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    candidate = replace(baseline, variant="eagle3_k3", runner="mrv2")
    mismatched = replace(baseline, **{field: _different_value(baseline, field)})

    with pytest.raises(NoMatchingBaselineError, match=field):
        match_baseline(candidate, [mismatched])


def test_match_baseline_computes_directional_speedups() -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    candidate = replace(
        baseline,
        variant="eagle3_k3",
        runner="mrv2",
        e2e_time_s=baseline.e2e_time_s / 1.25,
        generation_time_s=baseline.generation_time_s / 1.25,
        throughput_tps=baseline.throughput_tps * 1.25,
    )

    matched = match_baseline(candidate, [baseline])

    assert matched.e2e_time_speedup == 1.25
    assert matched.generation_time_speedup == 1.25
    assert matched.throughput_speedup == 1.25


def test_render_reports_is_deterministic_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    candidate = replace(
        baseline,
        variant="eagle3_k3",
        runner="mrv2",
        graph_mode="FULL",
        job_id="67890",
        log_path="logs/eagle3.log",
        wandb_url="https://wandb.example/eagle3",
        e2e_time_s=8.0,
        generation_time_s=4.0,
        throughput_tps=125.0,
    )
    candidate = match_baseline(candidate, [baseline])

    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"
    render_reports([candidate, baseline], csv_path, markdown_path)

    first_csv = csv_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    render_reports([baseline, candidate], csv_path, markdown_path)

    assert csv_path.read_text(encoding="utf-8") == first_csv
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    with csv_path.open(newline="", encoding="utf-8") as handle:
        report_rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in report_rows] == ["baseline", "eagle3_k3"]
    assert report_rows[1]["job_id"] == "67890"
    assert report_rows[1]["log_path"] == "logs/eagle3.log"
    assert report_rows[1]["wandb_url"] == "https://wandb.example/eagle3"
    assert report_rows[1]["runner"] == "mrv2"
    assert report_rows[1]["graph_mode"] == "FULL"
    assert (
        "| qwen30 | grpo-qwen3 | eagle3_k3 | lyris | 67890 | mrv2 | FULL |"
        in first_markdown
    )
    assert "logs/eagle3.log | https://wandb.example/eagle3 |" in first_markdown


def _different_value(summary: object, field: str) -> object:
    value = getattr(summary, field)
    if isinstance(value, str):
        return f"different-{value}"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.1
    raise AssertionError(f"Unsupported identity value for {field}: {value!r}")
