from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
PROVENANCE = {
    "nemo_rl_commit": "1" * 40,
    "bridge_commit": "2" * 40,
    "mcore_commit": "3" * 40,
    "te_commit": "4" * 40,
    "te_version": "2.16.0.dev0",
    "container_sha256": "5" * 64,
}
PARITY = {
    "router_topk_parity": True,
    "expert_count_parity": True,
    "parameter_delta_parity": True,
    "parameter_delta_max_abs_error": 0.0,
    "parameter_delta_max_rel_error": 0.0,
}


def _load_module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"reporting_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _load_exporter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    event_accumulator = ModuleType(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    event_processing = ModuleType("tensorboard.backend.event_processing")
    event_processing.event_accumulator = event_accumulator  # type: ignore[attr-defined]
    backend = ModuleType("tensorboard.backend")
    backend.event_processing = event_processing  # type: ignore[attr-defined]
    tensorboard = ModuleType("tensorboard")
    tensorboard.backend = backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorboard", tensorboard)
    monkeypatch.setitem(sys.modules, "tensorboard.backend", backend)
    monkeypatch.setitem(
        sys.modules, "tensorboard.backend.event_processing", event_processing
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorboard.backend.event_processing.event_accumulator",
        event_accumulator,
    )
    return _load_module("export_tensorboard")


def _real_scalar_values(
    exporter: ModuleType, steps: int = 5
) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {}
    for index, (canonical, aliases) in enumerate(
        exporter.CANONICAL_TAG_ALIASES.items(), start=1
    ):
        source = aliases[0]
        if canonical.startswith("cuda_graph/"):
            source = f"train/{canonical}"
        values[source] = {step: float(index + step) for step in range(1, steps + 1)}
    return values


def test_exporter_accepts_real_train_graph_tags_and_graph_free_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter(monkeypatch)
    values = _real_scalar_values(exporter)
    values.pop("train/cuda_graph/cache_misses")

    graph_rows = exporter._canonical_metrics(
        values,
        steps=5,
        require_graph_metrics=True,
    )
    graph_sources = {tag for tag in values if tag.startswith("train/cuda_graph/")}
    baseline_values = {
        tag: by_step for tag, by_step in values.items() if tag not in graph_sources
    }
    baseline_rows = exporter._canonical_metrics(
        baseline_values,
        steps=5,
        require_graph_metrics=False,
    )

    assert graph_rows[1]["cuda_graph/graph_calls"] > 0
    assert "cuda_graph/graph_calls" not in baseline_rows[1]
    assert baseline_rows[1]["timing/train/total_step_time"] > 0


def test_exporter_embeds_pairing_provenance_and_optional_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter(monkeypatch)
    event_path = tmp_path / "events"
    event_path.mkdir()
    output = tmp_path / "result.jsonl"
    monkeypatch.setattr(
        exporter, "_scalar_events", lambda _: _real_scalar_values(exporter)
    )

    exporter.export_events(
        [event_path],
        model="nano",
        dispatcher="hybridep",
        scope="attn",
        mode="nemorl",
        cluster="oci-hsg",
        profile="oci-hsg-gb200",
        phase="performance",
        steps=5,
        repeat=2,
        run_group="nano-attn-20260731",
        job_id="2474000",
        status="passed",
        provenance=PROVENANCE,
        parity=PARITY,
        output=output,
    )

    row = json.loads(output.read_text().splitlines()[0])
    assert row["profile"] == "oci-hsg-gb200"
    assert row["repeat"] == 2
    assert row["run_group"] == "nano-attn-20260731"
    assert row["provenance"] == PROVENANCE
    assert row["parity"] == PARITY


def test_baseline_export_omits_graph_metrics_and_parity_is_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = _load_exporter(monkeypatch)
    event_path = tmp_path / "events"
    event_path.mkdir()
    output = tmp_path / "baseline.jsonl"
    values = {
        tag: by_step
        for tag, by_step in _real_scalar_values(exporter).items()
        if not tag.startswith("train/cuda_graph/")
    }
    monkeypatch.setattr(exporter, "_scalar_events", lambda _: values)

    exporter.export_events(
        [event_path],
        model="nano",
        dispatcher="hybridep",
        scope="baseline",
        mode="nemorl",
        cluster="oci-hsg",
        profile="oci-hsg-gb200",
        phase="performance",
        steps=5,
        repeat=1,
        run_group="nano-performance",
        job_id="2473999",
        status="passed",
        provenance=PROVENANCE,
        output=output,
    )

    row = json.loads(output.read_text().splitlines()[0])
    assert row["graph_telemetry_status"] == "not_applicable"
    assert row["parity"] == {}
    assert not any(name.startswith("cuda_graph/") for name in row["metrics"])


def _complete_record(
    *,
    scope: str,
    repeat: int,
    job_id: str,
    e2e_step_time: float,
    throughput: float,
    step: int = 6,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    graph_metrics = {
        "train/cuda_graph/capture_count": 1,
        "train/cuda_graph/replay_count": 10,
        "train/cuda_graph/cache_hit_count": 10,
        "train/cuda_graph/cache_miss_count": 1,
        "train/cuda_graph/eviction_count": 0,
        "train/cuda_graph/fallback_count": 0,
        "train/cuda_graph/graph_calls": 10,
        "train/cuda_graph/eligible_calls": 10,
        "train/cuda_graph/logical_tokens": 80,
        "train/cuda_graph/padded_tokens": 96,
        "train/cuda_graph/capacity_tokens": 128,
        "train/cuda_graph/coverage": 1.0,
        "train/cuda_graph/capacity_utilization": 0.625,
        "train/cuda_graph/padding_utilization": 0.833333,
    }
    if scope == "baseline":
        graph_metrics = {"graph_telemetry_status": "not_applicable"}
    return {
        "model": "nano",
        "dispatcher": "hybridep",
        "scope": scope,
        "status": "passed",
        "mode": "nemorl",
        "cluster": "oci-hsg",
        "profile": "oci-hsg-gb200",
        "phase": "performance",
        "steps": 20,
        "step": step,
        "repeat": repeat,
        "run_group": "nano-performance-a",
        "job_id": job_id,
        "provenance": provenance or PROVENANCE,
        "parity": PARITY,
        "metrics": {
            "timing/train/total_step_time": e2e_step_time,
            "timing/train/generation": e2e_step_time / 2,
            "timing/train/policy_training": e2e_step_time / 4,
            "timing/train/policy_and_reference_logprobs": e2e_step_time / 8,
            "performance/tokens_per_sec_per_gpu": throughput,
            "performance/generation_tokens_per_sec_per_gpu": throughput / 2,
            "performance/policy_training_tokens_per_sec_per_gpu": throughput / 4,
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": throughput
            / 8,
            **graph_metrics,
            "train/reward": 0.8,
            "train/loss": 0.2,
            "train/gen_kl_error": 0.01,
            "train/token_mult_prob_error": 0.02,
            "train/policy_kl_error": 0.03,
            "train/js_divergence_error": 0.04,
            "train/sampling_importance_ratio": 1.0,
            "train/num_masked_seqs_by_logprob_error": 0,
            "train/grad_norm": 1.5,
        },
    }


def _complete_run(**kwargs: Any) -> list[dict[str, Any]]:
    return [_complete_record(step=step, **kwargs) for step in range(1, 21)]


def test_collector_preserves_runtime_and_correctness_metrics() -> None:
    collector = _load_module("collect_results")

    row = collector.normalize_record(
        _complete_record(
            scope="attn",
            repeat=1,
            job_id="graph-1",
            e2e_step_time=8.0,
            throughput=125.0,
        )
    )

    assert row["capture_count"] == 1
    assert row["replay_count"] == 10
    assert row["cache_hits"] == 10
    assert row["cache_misses"] == 1
    assert row["cache_evictions"] == 0
    assert row["fallback_count"] == 0
    assert row["policy_kl_error"] == 0.03
    assert row["js_divergence_error"] == 0.04
    assert row["sampling_importance_ratio"] == 1.0
    assert row["num_masked_seqs_by_logprob_error"] == 0
    assert row["parameter_delta_parity"] is True
    assert row["parameter_delta_max_abs_error"] == 0.0


def test_collection_refuses_to_replace_outputs_with_no_rows(tmp_path: Path) -> None:
    collector = _load_module("collect_results")
    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"
    output_json.write_text("good-json\n")
    output_csv.write_text("good-csv\n")

    with pytest.raises(ValueError, match="no result rows"):
        collector.write_results([], output_json=output_json, output_csv=output_csv)

    assert output_json.read_text() == "good-json\n"
    assert output_csv.read_text() == "good-csv\n"


def test_report_matches_exact_provenance_and_aggregates_repeat_deltas() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    records = [
        *_complete_run(
            scope="baseline",
            repeat=1,
            job_id="baseline-1",
            e2e_step_time=10.0,
            throughput=100.0,
        ),
        *_complete_run(
            scope="attn",
            repeat=1,
            job_id="graph-1",
            e2e_step_time=8.0,
            throughput=125.0,
        ),
        *_complete_run(
            scope="baseline",
            repeat=2,
            job_id="baseline-2",
            e2e_step_time=12.0,
            throughput=100.0,
        ),
        *_complete_run(
            scope="attn",
            repeat=2,
            job_id="graph-2",
            e2e_step_time=9.0,
            throughput=125.0,
        ),
    ]
    rows = [collector.normalize_record(record) for record in records]

    summaries = renderer.summarize_runs(rows)
    comparisons = renderer.build_matched_comparisons(summaries)

    assert all(not renderer.comparison_issues(summary) for summary in summaries)
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["scope"] == "attn"
    assert comparison["repeat_count"] == 2
    assert comparison["e2e_step_time_delta_pct_median"] == pytest.approx(-22.5)
    assert comparison["e2e_step_time_delta_pct_variance"] == pytest.approx(6.25)
    assert comparison["e2e_step_time_delta_pct_p95"] == pytest.approx(-20.25)


def test_partial_passed_run_is_not_comparison_eligible() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    row = collector.normalize_record(
        _complete_record(
            scope="attn",
            repeat=1,
            job_id="partial",
            e2e_step_time=8.0,
            throughput=125.0,
        )
    )

    summary = renderer.summarize_runs([row])[0]

    assert "expected 15 steady-state samples" in renderer.comparison_issues(summary)


def test_non_integer_runtime_counter_is_not_comparison_eligible() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    records = _complete_run(
        scope="attn",
        repeat=1,
        job_id="fractional-counter",
        e2e_step_time=8.0,
        throughput=125.0,
    )
    for record in records:
        record["metrics"]["train/cuda_graph/capture_count"] = 0.5
    rows = [collector.normalize_record(record) for record in records]

    summary = renderer.summarize_runs(rows)[0]

    assert "capture_count must be an integer" in renderer.comparison_issues(summary)


def test_nonpositive_performance_metric_is_not_comparison_eligible() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    records = _complete_run(
        scope="attn",
        repeat=1,
        job_id="negative-time",
        e2e_step_time=-1.0,
        throughput=125.0,
    )
    rows = [collector.normalize_record(record) for record in records]

    summary = renderer.summarize_runs(rows)[0]

    assert "e2e_step_time must be positive" in renderer.comparison_issues(summary)


def test_provenance_must_be_consistent_across_every_steady_step() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    records = _complete_run(
        scope="attn",
        repeat=1,
        job_id="mixed-provenance",
        e2e_step_time=8.0,
        throughput=125.0,
    )
    records[10]["provenance"] = {**PROVENANCE, "te_commit": "6" * 40}
    rows = [collector.normalize_record(record) for record in records]

    summary = renderer.summarize_runs(rows)[0]

    assert (
        "provenance differs across steady-state samples"
        in renderer.comparison_issues(summary)
    )


def test_report_renders_incomplete_rows_as_provisional_and_never_compares_them() -> (
    None
):
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    incomplete = _complete_record(
        scope="attn",
        repeat=1,
        job_id="incomplete-job",
        e2e_step_time=8.0,
        throughput=125.0,
    )
    incomplete.pop("parity")
    rows = [collector.normalize_record(incomplete)]

    report = renderer.render_html(rows)

    assert "Provisional / incomplete runs" in report
    assert "incomplete-job" in report
    assert "router_topk_parity" in report
    assert "No comparison-eligible matched baseline pairs." in report


def test_provenance_mismatch_prevents_matched_comparison() -> None:
    collector = _load_module("collect_results")
    renderer = _load_module("render_report")
    mismatched = {**PROVENANCE, "te_commit": "6" * 40}
    records = [
        *_complete_run(
            scope="baseline",
            repeat=1,
            job_id="baseline-1",
            e2e_step_time=10.0,
            throughput=100.0,
        ),
        *_complete_run(
            scope="attn",
            repeat=1,
            job_id="graph-1",
            e2e_step_time=8.0,
            throughput=125.0,
            provenance=mismatched,
        ),
    ]
    rows = [collector.normalize_record(record) for record in records]

    assert renderer.build_matched_comparisons(renderer.summarize_runs(rows)) == []


def test_missing_report_input_fails_closed(tmp_path: Path) -> None:
    renderer = _load_module("render_report")

    with pytest.raises(FileNotFoundError, match="normalized report input is missing"):
        renderer.read_rows(tmp_path / "missing.json")
