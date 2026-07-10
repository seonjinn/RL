# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def test_summarize_history_reports_valid_zero_acceptance() -> None:
    history = _history(1.2)
    for row in history:
        row["train/vllm/spec_num_accepted_tokens"] = 0.0

    summary = summarize_history("qwen32b", "eagle3_k5", history)

    assert summary.complete
    assert summary.reason == ""
    assert summary.acceptance_rate == 0.0
    assert summary.mean_acceptance_length == 1.0


@pytest.mark.parametrize(
    "variant",
    [
        "eagle3_k1",
        "eagle3_k2",
        "pard_k5",
        "pard_k16",
        "suffix_k32",
        "dflash_k15",
    ],
)
def test_summarize_history_requires_counters_for_every_specdec_variant(
    variant: str,
) -> None:
    history = _history(1.2)
    for row in history:
        del row["train/vllm/spec_num_drafts"]

    summary = summarize_history("qwen30ba3b", variant, history)

    assert not summary.complete
    assert summary.reason == "non_finite_metrics:num_drafts:2"


def test_summarize_history_weights_specdec_ratios_by_counters() -> None:
    history = _history(1.2)
    history[1]["train/vllm/spec_num_drafts"] = 1.0
    history[1]["train/vllm/spec_num_draft_tokens"] = 10.0
    history[1]["train/vllm/spec_num_accepted_tokens"] = 5.0

    summary = summarize_history("qwen32b", "dynamic", history)

    expected_accepted = 5.0 + 18 * 150.0
    assert summary.acceptance_rate == pytest.approx(
        expected_accepted / (10.0 + 18 * 300.0)
    )
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


def test_build_comparison_rows_distinguishes_draft_sampling_methods() -> None:
    summaries = [
        summarize_history("qwen30ba3b", "baseline", _history()),
        summarize_history(
            "qwen30ba3b",
            "eagle3_k5",
            _history(1.2),
            draft_sample_method="greedy",
        ),
        summarize_history(
            "qwen30ba3b",
            "eagle3_k5",
            _history(1.3),
            draft_sample_method="probabilistic",
        ),
    ]

    rows = build_comparison_rows(summaries)

    assert {(row.variant, row.draft_sample_method) for row in rows} == {
        ("baseline", "not_applicable"),
        ("eagle3_k5", "greedy"),
        ("eagle3_k5", "probabilistic"),
    }


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

    rows = {
        row.variant: row for row in build_comparison_rows([baseline, fixed, dynamic])
    }

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
    assert (
        _validate_manifest_rows(rows) == "duplicate variant baseline for model qwen32b"
    )


def test_validate_manifest_rows_matches_scheduler_and_graph_limits() -> None:
    common = {
        "model": "qwen30ba3b",
        "commit": "aaa",
        "nodes": "4",
        "max_num_seqs": "128",
        "output_max_model_len": "4096",
        "specdec_context_headroom_tokens": "32",
        "max_cudagraph_capture_size": "768",
        "cudagraph_capture_sizes": "[1,128,768]",
    }
    rows = [
        {**common, "variant": "baseline", "max_num_batched_tokens": "16384"},
        {**common, "variant": "pard_k16", "max_num_batched_tokens": "32768"},
    ]

    assert _validate_manifest_rows(rows) == "mismatched setup for model qwen30ba3b"


def test_validate_manifest_rows_matches_explicit_graph_shapes() -> None:
    common = {
        "model": "qwen235b",
        "commit": "aaa",
        "nodes": "16",
        "max_cudagraph_capture_size": "384",
    }
    rows = [
        {
            **common,
            "variant": "baseline",
            "cudagraph_capture_sizes": "[1,2,4,8,16,32,64,384]",
        },
        {
            **common,
            "variant": "eagle3_k5",
            "cudagraph_capture_sizes": "[1,2,4,8,16,32,64]",
        },
    ]

    assert _validate_manifest_rows(rows) == "mismatched setup for model qwen235b"


@pytest.mark.parametrize(
    ("field", "baseline_value", "candidate_value"),
    [
        ("num_prompts_per_step", "64", "16"),
        ("num_generations_per_prompt", "32", "16"),
        ("train_global_batch_size", "512", "256"),
        ("max_total_sequence_length", "4096", "40960"),
        ("max_new_tokens", "4096", "32768"),
    ],
)
def test_validate_manifest_rows_rejects_generation_geometry_mismatch(
    field: str,
    baseline_value: str,
    candidate_value: str,
) -> None:
    common = {
        "model": "qwen30ba3b",
        "commit": "aaa",
        "nodes": "4",
    }
    rows = [
        {**common, "variant": "baseline", field: baseline_value},
        {**common, "variant": "eagle3_k5", field: candidate_value},
    ]

    assert _validate_manifest_rows(rows) == "mismatched setup for model qwen30ba3b"


def test_validate_manifest_rows_allows_sampling_specific_variants() -> None:
    common = {
        "model": "qwen30ba3b",
        "variant": "eagle3_k5",
        "commit": "aaa",
        "nodes": "4",
        "rejection_sample_method": "standard",
    }
    rows = [
        {**common, "draft_sample_method": "greedy"},
        {**common, "draft_sample_method": "probabilistic"},
    ]

    assert _validate_manifest_rows(rows) is None


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
        scale = {
            "one-base": 1.0,
            "one-fixed": 1.2,
            "one-dynamic": 1.5,
            "two-base": 1.0,
            "two-fixed": 1.1,
            "two-dynamic": 1.3,
        }[run_id]
        variant = (
            "baseline"
            if run_id.endswith("base")
            else ("eagle3_k5" if run_id.endswith("fixed") else "dynamic")
        )
        return _MatrixRun(scale, variant)


class _UnhealthyRun(_MatrixRun):
    def scan_history(self, *, keys: list[str]):
        history = _history(self._scale)
        if self._variant == "dynamic":
            for row in history:
                row["train/reward"] = 0.0
        return iter({key: row[key] for key in keys} for row in history)


class _UnhealthyApi:
    def run(self, path: str) -> _UnhealthyRun:
        run_id = path.rsplit("/", maxsplit=1)[-1]
        if run_id == "base":
            return _UnhealthyRun(1.0, "baseline")
        assert run_id == "dynamic"
        return _UnhealthyRun(1.2, "dynamic")


class _SamplingMatrixApi:
    def run(self, path: str) -> _MatrixRun:
        run_id = path.rsplit("/", maxsplit=1)[-1]
        scale = {"base": 1.0, "greedy": 1.2, "probabilistic": 1.3}[run_id]
        variant = "baseline" if run_id == "base" else "eagle3_k5"
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
    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_MatrixApi(),
        )
        == 0
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    dynamic_rows = {row["model"]: row for row in payload if row["variant"] == "dynamic"}
    assert dynamic_rows["qwen30ba3b"]["job_id"] == "one-dynamic-job"
    assert dynamic_rows["qwen32b"]["job_id"] == "two-dynamic-job"


def test_main_keeps_sampling_specific_fixed_runs_distinct(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    rows = [
        ("baseline", "base", "base-job", "not_applicable", "not_applicable"),
        ("eagle3_k5", "greedy", "greedy-job", "standard", "greedy"),
        (
            "eagle3_k5",
            "probabilistic",
            "probabilistic-job",
            "standard",
            "probabilistic",
        ),
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=[
                "model",
                "variant",
                "job_id",
                "wandb_run_id",
                "rejection_sample_method",
                "draft_sample_method",
            ],
        )
        writer.writeheader()
        for variant, run_id, job_id, rejection_method, draft_method in rows:
            writer.writerow(
                {
                    "model": "qwen30ba3b",
                    "variant": variant,
                    "job_id": job_id,
                    "wandb_run_id": run_id,
                    "rejection_sample_method": rejection_method,
                    "draft_sample_method": draft_method,
                }
            )

    output_dir = tmp_path / "summary"
    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_SamplingMatrixApi(),
        )
        == 0
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    fixed_rows = {
        row["draft_sample_method"]: row
        for row in payload
        if row["variant"] == "eagle3_k5"
    }
    assert fixed_rows["greedy"]["job_id"] == "greedy-job"
    assert fixed_rows["probabilistic"]["job_id"] == "probabilistic-job"


def test_main_returns_nonzero_when_accuracy_health_gate_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=["model", "variant", "wandb_run_id"],
        )
        writer.writeheader()
        writer.writerow(
            {"model": "qwen32b", "variant": "baseline", "wandb_run_id": "base"}
        )
        writer.writerow(
            {
                "model": "qwen32b",
                "variant": "dynamic",
                "wandb_run_id": "dynamic",
            }
        )

    output_dir = tmp_path / "summary"
    exit_code = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=_UnhealthyApi(),
    )

    assert exit_code == 1
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    dynamic = next(row for row in payload if row["variant"] == "dynamic")
    assert dynamic["complete"]
    assert dynamic["health_gate_passed"] is False


def test_main_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "submissions.tsv"
    manifest.write_text("model\tvariant\twandb_run_id\n", encoding="utf-8")

    output_dir = tmp_path / "summary"
    exit_code = main(
        ["--manifest", str(manifest), "--output-dir", str(output_dir)],
        api=_UnhealthyApi(),
    )

    assert exit_code == 1
