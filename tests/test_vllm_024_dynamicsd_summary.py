from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from experiments.vllm_024_upgrade.summarize_eagle3_dynamicsd import (
    RunSummary,
    _validate_manifest_rows,
    build_comparison_rows,
    main,
    summarize_history,
)


def _history(scale: float = 1.0) -> list[dict[str, float]]:
    return [
        {
            "_step": step,
            "timing/train/generation": 100.0 / scale,
            "timing/train/total_step_time": 200.0 / scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "train/vllm/spec_acceptance_rate": 0.5 if scale > 1 else 0.0,
            "train/vllm/spec_acceptance_length": 2.5 if scale > 1 else 1.0,
            "train/vllm/spec_num_drafts": 100.0 if scale > 1 else 0.0,
            "train/vllm/spec_num_draft_tokens": 300.0 if scale > 1 else 0.0,
            "train/vllm/spec_num_accepted_tokens": 150.0 if scale > 1 else 0.0,
            "train/reward": 0.4,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
        }
        for step in range(1, 21)
    ]


def test_summarize_history_uses_steps_2_through_20() -> None:
    summary = summarize_history("qwen32b", "baseline", _history())

    assert isinstance(summary, RunSummary)
    assert summary.complete
    assert summary.measured_steps == list(range(2, 21))
    assert summary.generation_time_s == 100.0
    assert summary.e2e_step_time_s == 200.0


def test_summarize_history_allows_baseline_without_specdec_metrics() -> None:
    history = _history()
    for row in history:
        del row["train/vllm/spec_acceptance_rate"]
        del row["train/vllm/spec_acceptance_length"]

    summary = summarize_history("qwen32b", "baseline", history)

    assert summary.complete
    assert summary.acceptance_rate is None
    assert summary.mean_acceptance_length is None


def test_summarize_history_rejects_missing_step_20() -> None:
    summary = summarize_history("qwen32b", "dynamic", _history(1.2)[:-1])

    assert not summary.complete
    assert summary.reason == "missing_steps:20"


def test_summarize_history_rejects_non_finite_required_metric() -> None:
    history = _history(1.2)
    history[1]["train/reward"] = math.nan

    summary = summarize_history("qwen32b", "dynamic", history)

    assert not summary.complete
    assert summary.reason == "non_finite_metrics:reward:2"


def test_summarize_history_requires_positive_specdec_evidence() -> None:
    history = _history(1.2)
    for row in history:
        row["train/vllm/spec_num_accepted_tokens"] = 0.0

    summary = summarize_history("qwen32b", "eagle3_k5", history)

    assert not summary.complete
    assert summary.reason == "missing_specdec_evidence"


def test_summarize_history_weights_specdec_ratios_by_counters() -> None:
    history = _history(1.2)
    history[1]["train/vllm/spec_num_drafts"] = 1.0
    history[1]["train/vllm/spec_num_draft_tokens"] = 10.0
    history[1]["train/vllm/spec_num_accepted_tokens"] = 5.0

    summary = summarize_history("qwen32b", "dynamic", history)

    expected_accepted = 5.0 + 18 * 150.0
    assert summary.acceptance_rate == pytest.approx(expected_accepted / (10.0 + 18 * 300.0))
    assert summary.mean_acceptance_length == pytest.approx(
        1.0 + expected_accepted / (1.0 + 18 * 100.0)
    )


def test_build_comparison_rows_matches_model_baseline() -> None:
    summaries = [
        summarize_history("qwen32b", "baseline", _history()),
        summarize_history("qwen32b", "eagle3_k5", _history(1.25)),
        summarize_history("qwen32b", "dynamic", _history(1.5)),
    ]

    rows = {row.variant: row for row in build_comparison_rows(summaries)}

    assert rows["dynamic"].generation_throughput_speedup_vs_baseline == 1.5
    assert rows["dynamic"].e2e_step_time_speedup_vs_baseline == 1.5
    assert rows["dynamic"].generation_throughput_speedup_vs_fixed == 1.2


def test_build_comparison_rows_rejects_incomplete_or_missing_baselines() -> None:
    incomplete_baseline = summarize_history("qwen32b", "baseline", _history()[:-1])

    with pytest.raises(ValueError, match="incomplete baseline"):
        build_comparison_rows([incomplete_baseline])

    with pytest.raises(ValueError, match="missing baseline"):
        build_comparison_rows([summarize_history("qwen32b", "dynamic", _history(1.2))])


def test_build_comparison_rows_fails_health_gate_outside_ten_percent() -> None:
    baseline = summarize_history("qwen32b", "baseline", _history())
    dynamic_history = _history(1.2)
    for row in dynamic_history:
        row["train/reward"] = 0.45
    dynamic = summarize_history("qwen32b", "dynamic", dynamic_history)
    fixed = summarize_history("qwen32b", "eagle3_k5", _history(1.1))

    rows = {row.variant: row for row in build_comparison_rows([baseline, fixed, dynamic])}

    assert not rows["dynamic"].health_gate_passed
    assert not rows["dynamic"].reward_health_passed


def test_build_comparison_rows_accepts_matching_zero_health_metrics() -> None:
    baseline_history = _history()
    dynamic_history = _history(1.2)
    for row in [*baseline_history, *dynamic_history]:
        row["train/gen_kl_error"] = 0.0
    baseline = summarize_history("qwen32b", "baseline", baseline_history)
    dynamic = summarize_history("qwen32b", "dynamic", dynamic_history)

    rows = {row.variant: row for row in build_comparison_rows([baseline, dynamic])}

    assert rows["dynamic"].kl_health_passed


def test_validate_manifest_rows_rejects_setup_mismatch_and_duplicates() -> None:
    rows = [
        {"model": "qwen32b", "variant": "baseline", "commit": "aaa", "nodes": "4"},
        {"model": "qwen32b", "variant": "dynamic", "commit": "bbb", "nodes": "4"},
    ]
    assert _validate_manifest_rows(rows) == "mismatched setup for model qwen32b"

    rows[1] = {"model": "qwen32b", "variant": "baseline", "commit": "aaa", "nodes": "4"}
    assert _validate_manifest_rows(rows) == "duplicate variant baseline for model qwen32b"


class _FakeRun:
    url = "https://wandb.example/runs/dynamic-run"

    def scan_history(self, *, keys: list[str]):
        assert keys[0] == "_step"
        assert "train/vllm/spec_num_drafts" in keys
        return iter({key: row[key] for key in keys} for row in _history(1.2))


class _FakeApi:
    def run(self, path: str) -> _FakeRun:
        assert path == "nvidia/nemorl-vllm024-dynamicsd-aws-dfw/dynamic-run"
        return _FakeRun()


class _MatrixRun:
    def __init__(self, scale: float, variant: str) -> None:
        self._scale = scale
        self._variant = variant
        self.url = f"https://wandb.example/runs/{scale}"

    def scan_history(self, *, keys: list[str]):
        if self._variant == "baseline":
            assert not any("spec_" in key for key in keys)
        else:
            assert "train/vllm/spec_num_drafts" in keys
        return iter({key: row[key] for key in keys} for row in _history(self._scale))


class _MatrixApi:
    def run(self, path: str) -> _MatrixRun:
        run_id = path.rsplit("/", maxsplit=1)[-1]
        scale = {"one-base": 1.0, "one-fixed": 1.2, "one-dynamic": 1.5,
                 "two-base": 1.0, "two-fixed": 1.1, "two-dynamic": 1.3}[run_id]
        variant = "baseline" if run_id.endswith("base") else (
            "eagle3_k5" if run_id.endswith("fixed") else "dynamic"
        )
        return _MatrixRun(scale, variant)


def test_main_writes_manifest_metadata_and_explicit_csv(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=["model", "variant", "job_id", "wandb_run_id", "wandb_url"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "qwen32b",
                "variant": "dynamic",
                "job_id": "12345",
                "wandb_run_id": "dynamic-run",
                "wandb_url": "https://submitted.example/dynamic-run",
            }
        )

    output_dir = tmp_path / "summary"
    exit_code = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)], api=_FakeApi()
    )

    assert exit_code == 1
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload[0]["job_id"] == "12345"
    assert payload[0]["wandb_run_id"] == "dynamic-run"
    assert payload[0]["wandb_url"] == "https://submitted.example/dynamic-run"
    assert payload[0]["reason"].startswith("comparison_failed:")
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        fieldnames = csv.DictReader(stream).fieldnames
        assert fieldnames is not None
        assert "job_id" in fieldnames


def test_main_keeps_metadata_attached_to_interleaved_models(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    rows = [
        ("qwen30ba3b", "baseline", "one-base", "one-base-job"),
        ("qwen32b", "baseline", "two-base", "two-base-job"),
        ("qwen30ba3b", "eagle3_k5", "one-fixed", "one-fixed-job"),
        ("qwen32b", "eagle3_k5", "two-fixed", "two-fixed-job"),
        ("qwen30ba3b", "dynamic", "one-dynamic", "one-dynamic-job"),
        ("qwen32b", "dynamic", "two-dynamic", "two-dynamic-job"),
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=["model", "variant", "job_id", "wandb_run_id", "wandb_url"],
        )
        writer.writeheader()
        for model, variant, run_id, job_id in rows:
            writer.writerow(
                {
                    "model": model,
                    "variant": variant,
                    "job_id": job_id,
                    "wandb_run_id": run_id,
                    "wandb_url": f"https://submitted.example/{run_id}",
                }
            )

    output_dir = tmp_path / "summary"
    assert main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=_MatrixApi()) == 0

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    dynamic_rows = {row["model"]: row for row in payload if row["variant"] == "dynamic"}
    assert dynamic_rows["qwen30ba3b"]["job_id"] == "one-dynamic-job"
    assert dynamic_rows["qwen32b"]["job_id"] == "two-dynamic-job"
