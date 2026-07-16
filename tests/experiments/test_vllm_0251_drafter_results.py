from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.vllm_0251_drafter_matrix.collect_results import (
    IncompleteWindowError,
    NoMatchingBaselineError,
    ReportRow,
    RunMetadata,
    load_report_row,
    load_steps,
    main,
    match_baseline,
    parse_step,
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
    assert summary.e2e_throughput_tps_per_gpu == pytest.approx(2000.0 / 19.0)
    assert summary.generation_throughput_tps_per_gpu == pytest.approx(4000.0 / 19.0)
    assert summary.generation_ratio == pytest.approx(9.7 / 19.0)
    assert summary.acceptance_rate == pytest.approx(11.6 / 19.0)
    assert summary.mean_accepted_length == pytest.approx(59.0 / 19.0)
    assert summary.metadata.cuda_graph_coverage == pytest.approx(0.75)


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
        "requested_cuda_graph_mode",
        "resolved_cuda_graph_mode",
        "runner",
    ),
)
def test_match_baseline_requires_every_exact_identity_field(field: str) -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    candidate = replace(
        baseline,
        metadata=replace(baseline.metadata, variant="eagle3_k3", runner="mrv2"),
    )
    mismatched = replace(
        baseline,
        metadata=replace(
            baseline.metadata,
            **{field: _different_value(baseline.metadata, field)},
        ),
    )

    with pytest.raises(NoMatchingBaselineError, match=field):
        match_baseline(candidate, [mismatched])


def test_match_baseline_computes_directional_time_and_throughput_speedups() -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    candidate = replace(
        baseline,
        metadata=replace(baseline.metadata, variant="eagle3_k3", runner="mrv2"),
        e2e_time_s=baseline.e2e_time_s / 1.25,
        generation_time_s=baseline.generation_time_s / 1.5,
        e2e_throughput_tps_per_gpu=baseline.e2e_throughput_tps_per_gpu * 1.25,
        generation_throughput_tps_per_gpu=(
            baseline.generation_throughput_tps_per_gpu * 1.5
        ),
    )

    matched = match_baseline(candidate, [baseline])

    assert matched.e2e_time_speedup == 1.25
    assert matched.generation_time_speedup == 1.5
    assert matched.e2e_throughput_speedup == 1.25
    assert matched.generation_throughput_speedup == 1.5


@pytest.mark.parametrize("variant", ("baseline", "baseline_mrv1"))
@pytest.mark.parametrize(
    ("acceptance_rate", "mean_accepted_length"), ((None, None), (0.0, 0.0))
)
def test_baselines_allow_missing_or_zero_specdec_metrics(
    variant: str,
    acceptance_rate: float | None,
    mean_accepted_length: float | None,
) -> None:
    data = _fixture_step_data()
    data["variant"] = variant
    data["acceptance_rate"] = acceptance_rate
    data["mean_accepted_length"] = mean_accepted_length

    step = parse_step(data)

    assert step.acceptance_rate == acceptance_rate
    assert step.mean_accepted_length == mean_accepted_length


@pytest.mark.parametrize("field", ("acceptance_rate", "mean_accepted_length"))
def test_non_baseline_completed_step_requires_every_specdec_metric(
    field: str,
) -> None:
    data = _fixture_step_data()
    data["variant"] = "eagle3_k3"
    data.pop(field)

    with pytest.raises(ValueError, match=field):
        parse_step(data)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("top_p", 1.1, "top_p"),
        ("e2e_time_s", 0.0, "e2e_time_s"),
        ("e2e_time_s", float("nan"), "e2e_time_s"),
        ("generation_time_s", 0.0, "generation_time_s"),
        ("e2e_throughput_tps_per_gpu", 0.0, "e2e_throughput"),
        ("generation_throughput_tps_per_gpu", 0.0, "generation_throughput"),
        ("generation_throughput_tps_per_gpu", float("inf"), "generation_throughput"),
        ("policy_time_s", -1.0, "policy_time_s"),
        ("generation_ratio", 1.1, "generation_ratio"),
        ("acceptance_rate", -0.1, "acceptance_rate"),
        ("mean_accepted_length", 0.0, "mean_accepted_length"),
    ),
)
def test_parse_step_rejects_invalid_completed_metrics(
    field: str, value: float, match: str
) -> None:
    data = _fixture_step_data()
    data["variant"] = "eagle3_k3"
    data[field] = value

    with pytest.raises(ValueError, match=match):
        parse_step(data)


@pytest.mark.parametrize(
    ("baseline_variant", "runner"),
    (("baseline", "mrv2"), ("baseline_mrv1", "mrv1")),
)
def test_cli_matches_candidates_to_both_baseline_variants(
    tmp_path: Path, baseline_variant: str, runner: str
) -> None:
    baseline_path = tmp_path / f"{baseline_variant}.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"
    _write_run(baseline_path, variant=baseline_variant, runner=runner)
    _write_run(candidate_path, variant="eagle3_k3", runner=runner)

    exit_code = main(
        [
            str(baseline_path),
            str(candidate_path),
            "--csv",
            str(csv_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        report_rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in report_rows} == {
        baseline_variant,
        "eagle3_k3",
    }


def test_load_report_row_preserves_failed_and_unsupported_records(
    tmp_path: Path,
) -> None:
    metadata = _fixture_step_data()
    metadata.pop("step")
    for field in (
        "e2e_time_s",
        "generation_time_s",
        "policy_time_s",
        "logprob_time_s",
        "e2e_throughput_tps_per_gpu",
        "generation_throughput_tps_per_gpu",
        "generation_ratio",
        "acceptance_rate",
        "mean_accepted_length",
    ):
        metadata.pop(field)
    failed_path = tmp_path / "failed.jsonl"
    unsupported_path = tmp_path / "unsupported.jsonl"
    failed_path.write_text(
        json.dumps({**metadata, "status": "failed", "reason": "OOM"}) + "\n",
        encoding="utf-8",
    )
    unsupported_path.write_text(
        json.dumps(
            {
                **metadata,
                "variant": "dflash_k3",
                "status": "unsupported",
                "reason": "no checkpoint",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    failed = load_report_row(failed_path)
    unsupported = load_report_row(unsupported_path)

    assert failed == ReportRow(
        metadata=RunMetadata.from_mapping(metadata, "test"),
        status="failed",
        reason="OOM",
    )
    assert unsupported.status == "unsupported"
    assert unsupported.reason == "no checkpoint"
    assert unsupported.summary is None


def test_render_reports_is_deterministic_and_reports_metrics_and_provenance(
    tmp_path: Path,
) -> None:
    fixture_summary = summarize_steps(load_steps(FIXTURE_PATH))
    baseline = replace(
        fixture_summary,
        metadata=replace(
            fixture_summary.metadata,
            resolved_cuda_graph_mode="FULL",
        ),
    )
    candidate = replace(
        baseline,
        metadata=replace(
            baseline.metadata,
            variant="eagle3_k3",
            runner="mrv2",
            resolved_cuda_graph_mode="FULL",
            job_id="67890",
            log_path="logs/eagle3.log",
            wandb_url="https://wandb.example/eagle3",
        ),
        e2e_time_s=8.0,
        generation_time_s=4.0,
        e2e_throughput_tps_per_gpu=125.0,
        generation_throughput_tps_per_gpu=250.0,
    )
    candidate = match_baseline(candidate, [baseline])
    failed = ReportRow(
        metadata=replace(baseline.metadata, variant="dflash_k3", job_id="98765"),
        status="failed",
        reason="OOM",
    )

    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"
    render_reports(
        [ReportRow.completed(candidate), ReportRow.completed(baseline), failed],
        csv_path,
        markdown_path,
    )

    first_csv = csv_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    render_reports(
        [failed, ReportRow.completed(baseline), ReportRow.completed(candidate)],
        csv_path,
        markdown_path,
    )

    assert csv_path.read_text(encoding="utf-8") == first_csv
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    with csv_path.open(newline="", encoding="utf-8") as handle:
        report_rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in report_rows] == [
        "baseline",
        "dflash_k3",
        "eagle3_k3",
    ]
    candidate_row = report_rows[2]
    assert candidate_row["status"] == "completed"
    assert candidate_row["job_id"] == "67890"
    assert candidate_row["log_path"] == "logs/eagle3.log"
    assert candidate_row["wandb_url"] == "https://wandb.example/eagle3"
    assert candidate_row["runner"] == "mrv2"
    assert candidate_row["requested_cuda_graph_mode"] == "FULL_AND_PIECEWISE"
    assert candidate_row["resolved_cuda_graph_mode"] == "FULL"
    assert candidate_row["cuda_graph_coverage"] == "0.75"
    assert candidate_row["e2e_throughput_tps_per_gpu"] == "125"
    assert candidate_row["generation_throughput_tps_per_gpu"] == "250"
    assert candidate_row["e2e_throughput_speedup"] == "1.1875"
    assert candidate_row["generation_throughput_speedup"] == "1.1875"
    assert report_rows[1]["status"] == "failed"
    assert report_rows[1]["reason"] == "OOM"
    for header in (
        "status",
        "reason",
        "e2e_time_s",
        "generation_time_s",
        "policy_time_s",
        "logprob_time_s",
        "e2e_throughput_tps_per_gpu",
        "generation_throughput_tps_per_gpu",
        "generation_ratio",
        "acceptance_rate",
        "mean_accepted_length",
        "requested_cuda_graph_mode",
        "resolved_cuda_graph_mode",
        "cuda_graph_coverage",
        "job_id",
        "log_path",
        "wandb_url",
        "runner",
    ):
        assert f"| {header} |" in first_markdown


def test_render_reports_uses_a_total_sort_key(tmp_path: Path) -> None:
    baseline = summarize_steps(load_steps(FIXTURE_PATH))
    alpha = ReportRow(
        metadata=replace(baseline.metadata, variant="failed"),
        status="failed",
        reason="alpha",
    )
    beta = replace(alpha, reason="beta")
    csv_path = tmp_path / "summary.csv"
    markdown_path = tmp_path / "summary.md"

    render_reports([beta, alpha], csv_path, markdown_path)
    first_csv = csv_path.read_text(encoding="utf-8")
    render_reports([alpha, beta], csv_path, markdown_path)

    assert csv_path.read_text(encoding="utf-8") == first_csv
    assert first_csv.index("alpha") < first_csv.index("beta")


def _fixture_step_data() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[1])


def _write_run(path: Path, *, variant: str, runner: str) -> None:
    records: list[str] = []
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["variant"] = variant
        record["runner"] = runner
        records.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _different_value(metadata: RunMetadata, field: str) -> object:
    value = getattr(metadata, field)
    if isinstance(value, str):
        return f"different-{value}"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.1
    raise AssertionError(f"Unsupported identity value for {field}: {value!r}")
