import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "research/qwen3_8b_dflash_refit_perf_matrix"


def _module() -> ModuleType:
    path = EXPERIMENT_DIR / "analyze_wandb.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_row(step: int, **overrides: float) -> dict[str, float]:
    row = {
        "_step": float(step),
        "timing/train/total_step_time": 10.0,
        "timing/train/policy_training": 5.0,
        "timing/train/weight_sync": 99.0,
        "timing/train/prepare_for_generation/total": 1.0,
        "timing/train/generation": 3.0,
        "train/total_num_tokens": 100.0,
        "performance/generation_tokens_per_sec_per_gpu": float(step),
        "train/vllm/spec_acceptance_rate": 0.5,
    }
    row.update(overrides)
    return row


def test_summary_merges_steps_and_uses_canonical_logged_generation_tps() -> None:
    analysis = _module()
    rows = [_required_row(step) for step in range(5, 30)]
    rows.append(
        {
            "_step": 5,
            "timing/train/policy_training": 7.0,
            "train/draft_loss": 2.0,
            "train/peak_memory_allocated_mb": 12.0,
        }
    )

    summary = analysis.summarize_history(
        rows,
        cell="gbs32_mbs1_online",
        gbs=32,
        arm="online",
        replicate=1,
        evidence={"draft_update_count": 25, "draft_refit_count": 25},
    )

    assert summary["steps"] == 25
    assert summary["included_steps"] == list(range(5, 30))
    assert summary["missing_steps"] == []
    assert summary["valid_counts"]["generation_tps"] == 25
    assert summary["e2e_seconds_per_sample"] == 0.3125
    assert summary["e2e_seconds_per_token"] == 0.1
    assert summary["policy_seconds_mean"] == pytest.approx(127.0 / 25.0)
    assert summary["refit_seconds_mean"] == 1.0
    assert summary["e2e_seconds_mean"] == 10.0
    assert summary["generation_tokens_per_second_per_gpu"] == 17.0
    assert summary["acceptance_rate_mean"] == 0.5
    assert summary["peak_memory_allocated_mb"] == 12.0
    assert summary["draft_loss_mean"] == 2.0
    assert summary["update_refit_correct"] is True


def test_fixed_summary_accepts_absent_draft_and_peak_metrics() -> None:
    analysis = _module()
    rows = [_required_row(step) for step in range(5, 30)]

    summary = analysis.summarize_history(
        rows,
        cell="gbs32_mbs1_fixed",
        gbs=32,
        arm="fixed",
        replicate=2,
        evidence={"draft_update_count": 0, "draft_refit_count": 0},
    )

    assert summary["peak_memory_allocated_mb"] is None
    assert summary["draft_loss_mean"] is None
    assert summary["update_refit_correct"] is True


def test_summary_fails_and_discloses_missing_required_step() -> None:
    analysis = _module()
    rows = [_required_row(step) for step in range(5, 30) if step != 17]

    with pytest.raises(ValueError, match=r"missing required steps: \[17\]"):
        analysis.summarize_history(
            rows,
            cell="gbs32_mbs1_fixed",
            gbs=32,
            arm="fixed",
            replicate=1,
            evidence={"draft_update_count": 0, "draft_refit_count": 0},
        )


def test_summary_fails_and_discloses_step_with_missing_required_metric() -> None:
    analysis = _module()
    rows = [_required_row(step) for step in range(5, 30)]
    del rows[8]["performance/generation_tokens_per_sec_per_gpu"]

    with pytest.raises(
        ValueError,
        match=r"generation_tps.*missing numeric values at steps \[13\]",
    ):
        analysis.summarize_history(
            rows,
            cell="gbs32_mbs1_fixed",
            gbs=32,
            arm="fixed",
            replicate=1,
            evidence={"draft_update_count": 0, "draft_refit_count": 0},
        )


def test_summary_retains_infinite_metric_as_broken_observation() -> None:
    analysis = _module()
    rows = [_required_row(step) for step in range(5, 30)]
    rows[0]["performance/generation_tokens_per_sec_per_gpu"] = math.inf

    summary = analysis.summarize_history(
        rows,
        cell="gbs32_mbs1_fixed",
        gbs=32,
        arm="fixed",
        replicate=1,
        evidence={"draft_update_count": 0, "draft_refit_count": 0},
    )

    assert math.isinf(summary["generation_tokens_per_second_per_gpu"])
    assert summary["valid_counts"]["generation_tps"] == 25


def test_load_history_requests_exact_closed_step_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _module()
    scan_kwargs: dict[str, object] = {}

    class FakeRun:
        def scan_history(self, **kwargs: object) -> list[dict[str, float]]:
            scan_kwargs.update(kwargs)
            return [{"_step": 5.0}]

    fake_api = SimpleNamespace(run=lambda run_path: FakeRun())
    fake_wandb = SimpleNamespace(Api=lambda timeout: fake_api)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rows = analysis._load_history("nvidia/project/run")

    assert rows == [{"_step": 5.0}]
    assert scan_kwargs["min_step"] == 5
    assert scan_kwargs["max_step"] == 30
    assert scan_kwargs["page_size"] == 1000


def test_pair_comparison_uses_fixed_as_the_denominator() -> None:
    analysis = _module()

    comparison = analysis.compare_pair(
        {
            "cell": "gbs64_mbs2_fixed",
            "replicate": 2,
            "e2e_seconds_per_token": 0.2,
        },
        {
            "cell": "gbs64_mbs2_online",
            "replicate": 2,
            "e2e_seconds_per_token": 0.25,
        },
    )

    assert comparison == {
        "shape": "gbs64_mbs2",
        "replicate": 2,
        "fixed_e2e_seconds_per_token": 0.2,
        "online_e2e_seconds_per_token": 0.25,
        "paired_delta_e2e_seconds_per_token": 0.04999999999999999,
        "online_overhead_percent": 25.0,
    }


def test_aggregate_discloses_all_three_paired_deltas_and_statistics() -> None:
    analysis = _module()
    comparisons = [
        {
            "shape": "gbs64_mbs2",
            "replicate": replicate,
            "paired_delta_e2e_seconds_per_token": delta,
            "online_overhead_percent": overhead,
        }
        for replicate, delta, overhead in [
            (1, 0.1, 10.0),
            (2, 0.2, 20.0),
            (3, 0.3, 30.0),
        ]
    ]

    aggregate = analysis.aggregate_pairs(comparisons)

    assert aggregate["shape"] == "gbs64_mbs2"
    assert aggregate["replicates"] == [1, 2, 3]
    assert aggregate["paired_deltas_e2e_seconds_per_token"] == [0.1, 0.2, 0.3]
    assert aggregate["paired_delta_mean"] == pytest.approx(0.2)
    assert aggregate["paired_delta_sample_stdev"] == pytest.approx(0.1)
    assert aggregate["paired_delta_95pct_ci"] == pytest.approx(
        [0.2 - 4.303 * 0.1 / math.sqrt(3), 0.2 + 4.303 * 0.1 / math.sqrt(3)]
    )
    assert aggregate["online_overhead_percent_mean"] == 20.0


def test_report_discloses_each_paired_replicate_and_aggregate(tmp_path: Path) -> None:
    analysis = _module()
    summaries = [
        {
            "cell": "gbs64_mbs2_fixed",
            "replicate": 1,
            "e2e_seconds_per_token": 0.2,
            "policy_seconds_mean": 1.0,
            "refit_seconds_mean": 0.0,
            "generation_tokens_per_second_per_gpu": 100.0,
            "acceptance_rate_mean": 0.5,
            "peak_memory_allocated_mb": None,
            "update_refit_correct": True,
        }
    ]
    comparisons = [
        {
            "shape": "gbs64_mbs2",
            "replicate": replicate,
            "paired_delta_e2e_seconds_per_token": delta,
            "online_overhead_percent": overhead,
        }
        for replicate, delta, overhead in [
            (1, 0.01, 5.0),
            (2, 0.02, 10.0),
            (3, 0.03, 15.0),
        ]
    ]
    aggregates = [analysis.aggregate_pairs(comparisons)]
    output_dir = tmp_path / "report"

    analysis._write_reports(output_dir, summaries, comparisons, aggregates)

    report = (output_dir / "README.md").read_text()
    assert "| gbs64_mbs2 | 1 | 0.010000 | 5.000 |" in report
    assert "| gbs64_mbs2 | 2 | 0.020000 | 10.000 |" in report
    assert "| gbs64_mbs2 | 3 | 0.030000 | 15.000 |" in report
    assert "Paired delta mean ± sample stdev" in report
    assert "95% CI" in report
