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
from typing import Iterable, Mapping

import pytest

from experiments.vllm_024_upgrade import summarize_tail_gated_specdec
from experiments.vllm_024_upgrade.summarize_tail_gated_specdec import (
    CAPTURE_PROFILE_MANIFEST_HEADER,
    COHORT_FIELDS,
    LEGACY_C78A93C8_MANIFEST_HEADER,
    METRIC_KEYS,
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_ROW_FIELDS,
    RunSummary,
    _history_keys,
    _read_manifest,
    _scan_sparse_history,
    _validate_manifest_rows,
    build_comparison_rows,
    main,
    summarize_history,
)


EXPECTED_METRIC_KEYS = {
    "e2e_time": "timing/train/total_step_time",
    "generation_time": "timing/train/generation",
    "e2e_tps_gpu": "performance/tokens_per_sec_per_gpu",
    "generation_tps_gpu": "performance/generation_tokens_per_sec_per_gpu",
    "policy_time": "timing/train/policy_training",
    "logprob_time": "timing/train/policy_and_reference_logprobs",
    "reward": "train/reward",
    "response_length": "train/mean_gen_tokens_per_sample",
    "approx_kl": "train/gen_kl_error",
    "policy_loss": "train/loss",
    "num_drafts": "train/vllm/spec_num_drafts",
    "num_draft_tokens": "train/vllm/spec_num_draft_tokens",
    "num_accepted_tokens": "train/vllm/spec_num_accepted_tokens",
    "acceptance_rate": "train/vllm/spec_acceptance_rate",
    "mean_accept_len": "train/vllm/spec_acceptance_length",
    "gate_decisions": "train/vllm/tail_gate_decisions",
    "gate_activations": "train/vllm/tail_gate_activations",
    "gate_enabled_ratio": "train/vllm/tail_gate_enabled_step_ratio",
    "gate_advance_only_ratio": "train/vllm/tail_gate_advance_only_step_ratio",
    "activation_tick": "train/vllm/tail_gate_activation_tick",
    "activation_batch": "train/vllm/tail_gate_activation_batch",
    "activation_seq_len": "train/vllm/tail_gate_activation_seq_len",
    "predicted_speedup": "train/vllm/tail_gate_predicted_speedup",
    "activation_predicted_speedup": (
        "train/vllm/tail_gate_activation_predicted_speedup"
    ),
    "target_graph_ratio": "train/vllm/cudagraph_target_graph_call_ratio",
    "draft_graph_ratio": "train/vllm/cudagraph_draft_graph_call_ratio",
    "draft_prefill_graph_ratio": "train/vllm/cudagraph_draft_prefill_graph_call_ratio",
    "draft_decode_graph_ratio": "train/vllm/cudagraph_draft_decode_graph_call_ratio",
}


def test_scan_sparse_history_uses_unfiltered_wandb_rows() -> None:
    rows = [{"_step": 1, "metric": 2.0}, {"_step": 2, "metric": 3.0}]

    class SparseRun:
        def scan_history(self, *, keys: list[str] | None = None):
            return iter(rows if keys is None else [])

    assert _scan_sparse_history(SparseRun(), ["_step", "optional_metric"]) == rows


def _metadata(
    *,
    model: str = "qwen32b",
    runner: str = "v2",
    variant: str = "baseline_v2",
) -> dict[str, str]:
    is_qwen32 = model == "qwen32b"
    gated = "threshold" in variant or "roofline" in variant
    job_id = f"job-{model}-{variant}"
    run_dir = f"runs/{model}/{variant}"
    ray_log_root = f"{run_dir}/{job_id}-logs"
    return {
        "timestamp": "2026-07-10T12:00:00Z",
        "model": model,
        "variant": variant,
        "gate_mode": (
            "roofline" if "roofline" in variant else "threshold" if gated else "off"
        ),
        "k": "0" if variant.startswith("baseline_") else "5",
        "threshold": "32" if gated else "",
        "consecutive_checks": "10" if gated else "",
        "roofline_config_sha256": "cafebabe" if "roofline" in variant else "",
        "cluster": "lyris-gb200",
        "runtime": "nemo-rl",
        "runtime_version": "nightly-20260707",
        "runtime_commit": "abc123",
        "vllm_version": "0.24.0",
        "vllm_commit": "ee0da84a",
        "target_tp": "2" if is_qwen32 else "1",
        "draft_tp": "1",
        "dp": "8" if is_qwen32 else "16",
        "ep": "1",
        "temperature": "1.0",
        "top_p": "1.0",
        "max_osl": "4096",
        "max_model_len": "4128",
        "max_sequence_length": "4096",
        "num_prompts": "64",
        "num_generations": "32",
        "train_gbs": "512",
        "max_num_batched_tokens": "16384",
        "max_num_seqs": "1024",
        "recipe": (
            "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
            if is_qwen32
            else "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
        ),
        "container": "/containers/nemo.sqsh",
        "container_sha256": "deadbeef",
        "runner": runner,
        "graph_mode": "FULL_AND_PIECEWISE" if runner == "v2" else "PIECEWISE",
        "cuda_graph_enabled": "true",
        "enforce_eager": "false",
        "sampling": "standard",
        "draft_sample_method": (
            "not_applicable" if variant.startswith("baseline_") else "probabilistic"
        ),
        "job_id": job_id,
        "wandb_run_id": f"run-{model}-{variant}",
        "wandb_url": f"https://wandb.example/{model}/{variant}",
        "run_dir": run_dir,
        "slurm_log_path": f"{run_dir}/slurm-{job_id}.out",
        "ray_driver_log_path": f"{ray_log_root}/ray-driver.log",
        "ray_log_dir": f"{ray_log_root}/ray",
        "launcher_command": f"sbatch --model={model} --variant={variant}",
        "command": f"run --model={model} --variant={variant}",
    }


def _is_specdec(metadata: Mapping[str, str]) -> bool:
    return not metadata["variant"].startswith("baseline_")


def _is_gated(metadata: Mapping[str, str]) -> bool:
    return metadata["gate_mode"] in {"threshold", "roofline"}


def _history(
    metadata: Mapping[str, str],
    *,
    scale: float = 1.0,
    target_graph_ratio: float = 1.0,
    draft_graph_ratio: float = 1.0,
    activated: bool = True,
    predicted_speedup: float = 1.12,
    activation_predicted_speedup: float = 1.12,
    fallback_count: float | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for step in range(1, 21):
        row = {
            "_step": step,
            "timing/train/total_step_time": 200.0 / scale,
            "timing/train/generation": 100.0 / scale,
            "performance/tokens_per_sec_per_gpu": 25.0 * scale,
            "performance/generation_tokens_per_sec_per_gpu": 50.0 * scale,
            "timing/train/policy_training": 30.0,
            "timing/train/policy_and_reference_logprobs": 20.0,
            "train/reward": 0.4,
            "train/mean_gen_tokens_per_sample": 1024.0,
            "train/gen_kl_error": 0.01,
            "train/loss": 0.2,
            "train/vllm/cudagraph_target_graph_call_ratio": target_graph_ratio,
        }
        if _is_specdec(metadata):
            row.update(
                {
                    "train/vllm/spec_num_drafts": 100.0,
                    "train/vllm/spec_num_draft_tokens": 300.0,
                    "train/vllm/spec_num_accepted_tokens": 150.0,
                    "train/vllm/spec_acceptance_rate": 0.5,
                    "train/vllm/spec_acceptance_length": 2.5,
                }
            )
            if metadata["runner"] == "v1":
                row["train/vllm/cudagraph_draft_graph_call_ratio"] = draft_graph_ratio
            else:
                row["train/vllm/cudagraph_draft_prefill_graph_call_ratio"] = (
                    draft_graph_ratio
                )
                row["train/vllm/cudagraph_draft_decode_graph_call_ratio"] = (
                    draft_graph_ratio
                )
        if _is_gated(metadata):
            row.update(
                {
                    "train/vllm/tail_gate_decisions": 100.0,
                    "train/vllm/tail_gate_activations": (
                        float(metadata["dp"]) if activated else 0.0
                    ),
                    "train/vllm/tail_gate_enabled_step_ratio": (
                        0.25 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_advance_only_step_ratio": (
                        0.75 if activated else 1.0
                    ),
                    "train/vllm/tail_gate_activation_tick": (
                        17.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_activation_batch": (
                        16.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_activation_seq_len": (
                        8192.0 if activated else 0.0
                    ),
                    "train/vllm/tail_gate_predicted_speedup": (
                        predicted_speedup if activated else 0.0
                    ),
                }
            )
            if metadata["gate_mode"] == "roofline":
                row["train/vllm/tail_gate_activation_predicted_speedup"] = (
                    activation_predicted_speedup if activated else 0.0
                )
        if fallback_count is not None:
            row["train/vllm/cudagraph_target_fallback_missing_key"] = fallback_count
        rows.append(row)
    return rows


def _summary(
    metadata: dict[str, str],
    *,
    scale: float = 1.0,
    target_graph_ratio: float = 1.0,
    draft_graph_ratio: float = 1.0,
    activated: bool = True,
    predicted_speedup: float = 1.12,
    activation_predicted_speedup: float = 1.12,
    fallback_count: float | None = None,
) -> RunSummary:
    return summarize_history(
        metadata,
        _history(
            metadata,
            scale=scale,
            target_graph_ratio=target_graph_ratio,
            draft_graph_ratio=draft_graph_ratio,
            activated=activated,
            predicted_speedup=predicted_speedup,
            activation_predicted_speedup=activation_predicted_speedup,
            fallback_count=fallback_count,
        ),
    )


def test_available_cudagraph_fallback_counter_overrides_ratio_threshold() -> None:
    metadata = _metadata(variant="fastrl_threshold_v2_k5")
    summary = _summary(metadata, fallback_count=1.0)
    baseline = _summary(_metadata(variant="baseline_v2"))
    always_on = _summary(_metadata(variant="always_on_v2_k5"))

    candidate = next(
        row
        for row in build_comparison_rows([baseline, always_on, summary])
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.cuda_graph_fallback_count == 19.0
    assert candidate.cuda_graph_health_passed is False
    assert candidate.cuda_graph_evidence == "observed fallback counters=19"


def test_ratio_only_graph_evidence_is_phrased_as_measured_threshold() -> None:
    summary = _summary(_metadata(variant="baseline_v2"), target_graph_ratio=0.99)
    row = build_comparison_rows([summary])[0]

    assert row.cuda_graph_health_passed is True
    assert row.cuda_graph_fallback_count is None
    assert row.cuda_graph_evidence == (
        "fallback counters unavailable; measured graph-call ratio threshold >= 0.99"
    )


def test_gate_activation_count_cannot_exceed_worker_ranks() -> None:
    metadata = _metadata(variant="fastrl_threshold_v2_k5")
    history = _history(metadata)
    for record in history:
        record["train/vllm/tail_gate_activations"] = 17.0
    candidate_summary = summarize_history(metadata, history)
    baseline = _summary(_metadata(variant="baseline_v2"))
    always_on = _summary(_metadata(variant="always_on_v2_k5"))

    candidate = next(
        row
        for row in build_comparison_rows([baseline, always_on, candidate_summary])
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.gate_activation_health_passed is False


def _cohort(*, model: str = "qwen32b", runner: str = "v2") -> list[dict[str, str]]:
    variants = (
        ["baseline_v1", "always_on_v1_k5"]
        if runner == "v1"
        else ["baseline_v2", "always_on_v2_k5", "fastrl_threshold_v2_k5"]
    )
    return [
        _metadata(model=model, runner=runner, variant=variant) for variant in variants
    ]


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, delimiter="\t", fieldnames=list(materialized[0])
        )
        writer.writeheader()
        writer.writerows(materialized)


class _FakeRun:
    def __init__(self, history: list[dict[str, float]], url: str) -> None:
        self._history = history
        self.url = url
        self.requested_keys: list[str] = []

    def scan_history(self, *, keys: list[str]):
        self.requested_keys = keys
        return iter(
            {key: row[key] for key in keys if key in row} for row in self._history
        )


class _FakeApi:
    def __init__(self, histories: Mapping[str, list[dict[str, float]]]) -> None:
        self._histories = histories
        self.runs: dict[str, _FakeRun] = {}
        self.calls = 0

    def run(self, path: str) -> _FakeRun:
        self.calls += 1
        run_id = path.rsplit("/", maxsplit=1)[-1]
        run = _FakeRun(self._histories[run_id], f"https://wandb.example/api/{run_id}")
        self.runs[run_id] = run
        return run


def _api_for(rows: Iterable[dict[str, str]]) -> _FakeApi:
    return _FakeApi(
        {
            row["wandb_run_id"]: _history(
                row,
                scale=1.25 if not row["variant"].startswith("baseline_") else 1.0,
            )
            for row in rows
        }
    )


def test_metric_contract_uses_exact_production_emitter_keys() -> None:
    assert METRIC_KEYS == EXPECTED_METRIC_KEYS


def test_history_keys_are_variant_and_runner_aware() -> None:
    baseline = _metadata()
    v1_spec = _metadata(model="qwen30ba3b", runner="v1", variant="always_on_v1_k5")
    gated = _metadata(variant="fastrl_threshold_v2_k5")
    roofline = _metadata(variant="efficient_roofline_v2_k5")

    baseline_keys = _history_keys(baseline)
    assert "train/vllm/cudagraph_target_graph_call_ratio" in baseline_keys
    assert not any("spec_" in key or "tail_gate_" in key for key in baseline_keys)
    assert not any("cudagraph_draft" in key for key in baseline_keys)

    v1_keys = _history_keys(v1_spec)
    assert "train/vllm/spec_num_drafts" in v1_keys
    assert "train/vllm/cudagraph_draft_graph_call_ratio" in v1_keys
    assert not any("tail_gate_" in key for key in v1_keys)

    gated_keys = _history_keys(gated)
    assert "train/vllm/tail_gate_activations" in gated_keys
    assert "train/vllm/tail_gate_activation_predicted_speedup" not in gated_keys
    assert "train/vllm/cudagraph_draft_prefill_graph_call_ratio" in gated_keys
    assert "train/vllm/cudagraph_draft_decode_graph_call_ratio" in gated_keys

    assert "train/vllm/tail_gate_activation_predicted_speedup" in _history_keys(
        roofline
    )


def test_baseline_is_final_without_specdec_gate_or_draft_metrics() -> None:
    summary = _summary(_metadata())

    assert summary.status == "final"
    assert summary.steps == list(range(2, 21))
    assert summary.policy_time == 30.0
    assert summary.logprob_time == 20.0
    assert summary.policy_loss == 0.2
    assert summary.acceptance_rate is None
    assert summary.draft_prefill_graph_ratio is None


@pytest.mark.parametrize(
    ("variant", "missing_key", "reason_metric"),
    [
        ("always_on_v2_k5", "train/vllm/spec_num_drafts", "num_drafts"),
        (
            "fastrl_threshold_v2_k5",
            "train/vllm/tail_gate_activations",
            "gate_activations",
        ),
        (
            "always_on_v2_k5",
            "train/vllm/cudagraph_draft_decode_graph_call_ratio",
            "draft_decode_graph_ratio",
        ),
    ],
)
def test_variant_required_metrics_cannot_be_missing(
    variant: str, missing_key: str, reason_metric: str
) -> None:
    metadata = _metadata(variant=variant)
    history = _history(metadata)
    del history[5][missing_key]

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == f"missing_metric:{reason_metric}:6"


def test_activation_tick_is_required_only_for_gated_variants() -> None:
    gated = _metadata(variant="fastrl_threshold_v2_k5")
    baseline = _metadata()
    history = _history(gated)
    del history[5]["train/vllm/tail_gate_activation_tick"]

    summary = summarize_history(gated, history)

    assert summary.status == "partial"
    assert summary.reason == "missing_metric:activation_tick:6"
    assert "train/vllm/tail_gate_activation_tick" in _history_keys(gated)
    assert "train/vllm/tail_gate_activation_tick" not in _history_keys(baseline)


def test_nonfinite_production_metric_is_partial() -> None:
    metadata = _metadata()
    history = _history(metadata)
    history[8]["train/loss"] = math.nan

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == "non_finite_metric:policy_loss:9"


def test_comparison_key_uses_only_complete_explicit_cohort_schema() -> None:
    baseline_metadata = _metadata()
    candidate_metadata = _metadata(variant="always_on_v2_k5")
    candidate_metadata["command"] = "a deliberately different command string"

    rows = build_comparison_rows(
        [_summary(baseline_metadata), _summary(candidate_metadata, scale=1.25)]
    )

    candidate = next(row for row in rows if row.variant == "always_on_v2_k5")
    assert candidate.e2e_time_speedup_vs_baseline == 1.25


def test_comparison_rejects_mismatched_cudagraph_request_coverage() -> None:
    baseline_metadata = _metadata()
    candidate_metadata = _metadata(variant="always_on_v2_k5")
    baseline_metadata["cudagraph_max_requests"] = "256"
    candidate_metadata["cudagraph_max_requests"] = "128"

    with pytest.raises(ValueError, match="missing matched baseline"):
        build_comparison_rows(
            [_summary(baseline_metadata), _summary(candidate_metadata)]
        )


def test_comparison_rejects_mixed_cuda_graph_states() -> None:
    baseline_metadata = _metadata()
    candidate_metadata = _metadata(variant="always_on_v2_k5")
    candidate_metadata["cuda_graph_enabled"] = "false"
    candidate_metadata["enforce_eager"] = "true"

    with pytest.raises(ValueError, match="missing matched baseline"):
        build_comparison_rows(
            [_summary(baseline_metadata), _summary(candidate_metadata)]
        )


def test_same_variant_graph_mode_key_matches_only_graph_ablation_states() -> None:
    graph_on_metadata = _metadata(variant="always_on_v2_k5")
    graph_on_metadata.update(
        {
            "cudagraph_max_requests": "256",
            "cudagraph_max_tokens": "1536",
            "cudagraph_capture_sizes": "[6,12,24,1536]",
        }
    )
    graph_off_metadata = {
        **graph_on_metadata,
        "cuda_graph_enabled": "false",
        "enforce_eager": "true",
        "graph_mode": "NONE",
        "cudagraph_max_requests": "not_applicable",
        "cudagraph_max_tokens": "not_applicable",
        "cudagraph_capture_sizes": "not_applicable",
    }

    assert summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_on_metadata)
    ) == summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_off_metadata)
    )

    different_variant = {
        **graph_off_metadata,
        "variant": "fastrl_threshold_v2_k5",
    }
    different_recipe = {
        **graph_off_metadata,
        "recipe": "examples/configs/recipes/llm/performance/different.yaml",
    }
    different_draft_sampling = {
        **graph_off_metadata,
        "draft_sample_method": "greedy",
    }
    assert summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_on_metadata)
    ) != summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(different_variant)
    )
    assert summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_on_metadata)
    ) != summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(different_recipe)
    )
    assert summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_on_metadata)
    ) != summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(different_draft_sampling)
    )


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("target_checkpoint", "/lustre/test/other-target"),
        ("target_checkpoint_revision", "b" * 40),
        ("draft_checkpoint", "/lustre/test/other-draft"),
        ("runner", "v1"),
        ("target_tp", "4"),
        ("draft_tp", "2"),
        ("dp", "4"),
        ("ep", "2"),
        ("max_osl", "8192"),
        ("max_model_len", "8256"),
        ("max_sequence_length", "8192"),
        ("temperature", "0.7"),
        ("top_p", "0.9"),
        ("sampling", "different"),
        ("draft_sample_method", "greedy"),
        ("gate_mode", "threshold"),
        ("k", "7"),
        ("threshold", "16"),
        ("consecutive_checks", "3"),
        ("roofline_config_sha256", "different-roofline-config"),
    ],
)
def test_same_variant_graph_mode_key_requires_exact_non_graph_provenance(
    field: str, mismatched_value: str
) -> None:
    graph_on_metadata = _metadata(variant="efficient_roofline_v2_k5")
    graph_on_metadata.update(
        {
            "cudagraph_max_requests": "256",
            "cudagraph_max_tokens": "1536",
            "cudagraph_capture_sizes": "[6,12,24,1536]",
            "target_checkpoint": "/lustre/test/qwen32b-target",
            "target_checkpoint_revision": "a" * 40,
            "draft_checkpoint": "/lustre/test/eagle3",
        }
    )
    graph_off_metadata = {
        **graph_on_metadata,
        "cuda_graph_enabled": "false",
        "enforce_eager": "true",
        "graph_mode": "NONE",
        "cudagraph_max_requests": "not_applicable",
        "cudagraph_max_tokens": "not_applicable",
        "cudagraph_capture_sizes": "not_applicable",
    }

    graph_on_key = summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_on_metadata)
    )
    graph_off_key = summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(graph_off_metadata)
    )
    assert graph_on_key == graph_off_key
    assert dict(graph_on_key)[field] == graph_on_metadata[field]
    assert dict(graph_off_key)[field] == graph_off_metadata[field]

    mismatched_metadata = {**graph_off_metadata, field: mismatched_value}
    assert graph_on_key != summarize_tail_gated_specdec.same_variant_graph_mode_key(
        _summary(mismatched_metadata)
    )


def test_comparison_rejects_mixed_nonbaseline_draft_sample_methods() -> None:
    baseline = _metadata()
    always_on = _metadata(variant="always_on_v2_k5")
    threshold = _metadata(variant="fastrl_threshold_v2_k5")
    always_on["draft_sample_method"] = "greedy"

    with pytest.raises(ValueError, match="mixed draft_sample_method"):
        build_comparison_rows(
            [_summary(baseline), _summary(always_on), _summary(threshold)]
        )


@pytest.mark.parametrize(
    "field",
    [
        "cluster",
        "runtime_version",
        "runtime_commit",
        "vllm_version",
        "vllm_commit",
        "target_tp",
        "dp",
        "temperature",
        "max_osl",
        "max_model_len",
        "num_generations",
        "graph_mode",
        "sampling",
    ],
)
def test_cross_cohort_speedups_are_rejected(field: str) -> None:
    baseline = _metadata()
    candidate = _metadata(variant="always_on_v2_k5")
    candidate[field] = f"different-{candidate[field]}"

    with pytest.raises(ValueError, match="missing matched baseline"):
        build_comparison_rows([_summary(baseline), _summary(candidate)])


def test_manifest_rejects_every_missing_cohort_dimension() -> None:
    row = _metadata()

    for field in REQUIRED_MANIFEST_FIELDS:
        incomplete = {**row, field: ""}
        assert (
            _validate_manifest_rows([incomplete]) == f"missing manifest fields:{field}"
        )


@pytest.mark.parametrize(
    "field",
    [
        "run_dir",
        "slurm_log_path",
        "ray_driver_log_path",
        "ray_log_dir",
        "launcher_command",
        "command",
    ],
)
def test_legacy_manifest_allows_missing_mini_log_and_command_provenance(
    field: str,
) -> None:
    row = _metadata()
    del row[field]

    assert field not in REQUIRED_MANIFEST_FIELDS
    assert _validate_manifest_rows([row]) is None


def test_legacy_manifest_allows_missing_draft_sample_method() -> None:
    rows = _cohort()
    for row in rows:
        del row["draft_sample_method"]

    assert "draft_sample_method" not in REQUIRED_MANIFEST_FIELDS
    assert _validate_manifest_rows(rows) is None


def test_historical_manifest_without_mini_fields_remains_collectible(
    tmp_path: Path,
) -> None:
    rows = _cohort()
    mini_only_fields = (
        "draft_sample_method",
        "max_model_len",
        "cuda_graph_enabled",
        "enforce_eager",
        "run_dir",
        "slurm_log_path",
        "ray_driver_log_path",
        "ray_log_dir",
        "launcher_command",
    )
    for row in rows:
        for field in mini_only_fields:
            del row[field]
    manifest = tmp_path / "historical-submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "output"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_api_for(rows),
        )
        == 0
    )
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    baseline = next(row for row in payload if row["variant"] == "baseline_v2")
    always_on = next(row for row in payload if row["variant"] == "always_on_v2_k5")
    assert baseline["draft_sample_method"] == "not_applicable"
    assert always_on["draft_sample_method"] == "legacy_unspecified"


def test_c78a93c8_header_fixture_uses_explicit_legacy_sentinels(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "vllm_024_tail_gate"
        / "c78a93c8_submissions.tsv"
    )
    manifest = tmp_path / "submissions.tsv"
    manifest.write_bytes(fixture.read_bytes())
    output_dir = tmp_path / "output"

    assert fixture.read_text(encoding="utf-8").splitlines()[0].split("\t") == list(
        LEGACY_C78A93C8_MANIFEST_HEADER
    )
    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_api_for(_cohort()),
        )
        == 0
    )
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert {row["max_model_len"] for row in payload} == {"legacy_unrecorded"}
    assert (
        next(row for row in payload if row["variant"] == "baseline_v2")[
            "draft_sample_method"
        ]
        == "not_applicable"
    )


def test_capture_profile_manifest_schema_is_collectible(tmp_path: Path) -> None:
    metadata = _metadata(variant="always_on_v2_k5")
    metadata.update(
        {
            "cudagraph_max_requests": "256",
            "cudagraph_max_tokens": "1536",
            "cudagraph_capture_sizes": "[6,12,24,1536]",
            "checkout_path": "/lustre/test/nemo-rl",
            "ray_sub_path": "/lustre/test/nemo-rl/ray.sub",
            "target_checkpoint": "/lustre/test/qwen32",
            "target_checkpoint_revision": "a" * 40,
            "draft_checkpoint": "/lustre/test/eagle3",
            "command_argv_json": "[]",
            "launcher_argv_json": "[]",
        }
    )
    manifest = tmp_path / "capture-profile.tsv"
    ordered = {field: metadata.get(field, "") for field in CAPTURE_PROFILE_MANIFEST_HEADER}
    _write_manifest(manifest, [ordered])

    schema, rows = _read_manifest(manifest)

    assert schema.name == "capture_profile_v3"
    assert rows[0]["cudagraph_max_requests"] == "256"
    assert rows[0]["cudagraph_max_tokens"] == "1536"


@pytest.mark.parametrize("field", ["wandb_run_id", "job_id"])
def test_manifest_rejects_duplicate_run_identifiers_before_wandb_fetch(
    tmp_path: Path, field: str
) -> None:
    rows = _cohort()
    rows[1][field] = rows[0][field]
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "output"
    api = _api_for(rows)

    with pytest.raises(ValueError, match=rf"duplicate {field}"):
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=api,
        )

    assert api.calls == 0
    assert not output_dir.exists()


def test_manifest_rejects_engine_length_below_output_plus_headroom() -> None:
    row = _metadata()
    row["max_model_len"] = "4096"

    assert _validate_manifest_rows([row]) == (
        "max_model_len must be at least max_osl plus 32:4096:4096"
    )


@pytest.mark.parametrize(
    ("target_graph_ratio", "draft_graph_ratio"), [(0.98, 1.0), (1.0, 0.98)]
)
def test_graph_coverage_below_threshold_is_health_failed(
    target_graph_ratio: float, draft_graph_ratio: float
) -> None:
    metadata = _cohort()
    summaries = [
        _summary(
            row,
            target_graph_ratio=(
                target_graph_ratio
                if row["variant"] == "fastrl_threshold_v2_k5"
                else 1.0
            ),
            draft_graph_ratio=(
                draft_graph_ratio if row["variant"] == "fastrl_threshold_v2_k5" else 1.0
            ),
        )
        for row in metadata
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.cuda_graph_health_passed is False
    assert candidate.health_gate_passed is False
    assert candidate.status == "health_failed"


def test_gated_variant_must_activate_after_observable_off_period() -> None:
    metadata = _cohort()
    summaries = [
        _summary(row, activated=row["variant"] != "fastrl_threshold_v2_k5")
        for row in metadata
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.gate_activation_health_passed is False
    assert candidate.health_gate_passed is False
    assert candidate.status == "health_failed"


def test_gated_variant_cannot_report_always_on_behavior() -> None:
    cohort = _cohort()
    gated_metadata = cohort[-1]
    gated_history = _history(gated_metadata)
    for record in gated_history:
        record["train/vllm/tail_gate_enabled_step_ratio"] = 1.0
        record["train/vllm/tail_gate_advance_only_step_ratio"] = 0.0
    summaries = [
        _summary(cohort[0]),
        _summary(cohort[1]),
        summarize_history(gated_metadata, gated_history),
    ]

    candidate = next(
        row
        for row in build_comparison_rows(summaries)
        if row.variant == "fastrl_threshold_v2_k5"
    )

    assert candidate.gate_activation_health_passed is False
    assert candidate.status == "health_failed"


def test_roofline_requires_explicit_activation_predicted_speedup() -> None:
    metadata = _metadata(variant="efficient_roofline_v2_k5")
    history = _history(metadata)
    del history[5]["train/vllm/tail_gate_activation_predicted_speedup"]

    summary = summarize_history(metadata, history)

    assert summary.status == "partial"
    assert summary.reason == "missing_metric:activation_predicted_speedup:6"


@pytest.mark.parametrize(
    ("all_decision_speedup", "activation_speedup", "expected_status"),
    [(1.30, 1.01, "health_failed"), (0.90, 1.06, "final")],
)
def test_roofline_health_uses_only_activation_predicted_speedup(
    all_decision_speedup: float,
    activation_speedup: float,
    expected_status: str,
) -> None:
    baseline = _metadata()
    always_on = _metadata(variant="always_on_v2_k5")
    roofline = _metadata(variant="efficient_roofline_v2_k5")
    rows = build_comparison_rows(
        [
            _summary(baseline),
            _summary(always_on),
            _summary(
                roofline,
                predicted_speedup=all_decision_speedup,
                activation_predicted_speedup=activation_speedup,
            ),
        ]
    )

    candidate = next(row for row in rows if row.variant == "efficient_roofline_v2_k5")

    assert candidate.predicted_speedup == all_decision_speedup
    assert candidate.activation_predicted_speedup == activation_speedup
    assert candidate.status == expected_status


def test_main_renders_interleaved_models_and_runner_sections(tmp_path: Path) -> None:
    rows = [
        *_cohort(model="qwen32b", runner="v2"),
        *_cohort(model="qwen30ba3b", runner="v1"),
    ]
    interleaved = [rows[3], rows[0], rows[4], rows[2], rows[1]]
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, interleaved)
    output_dir = tmp_path / "output"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_api_for(rows),
        )
        == 0
    )

    fragment = (output_dir / "tail_gated_specdec.html").read_text(encoding="utf-8")
    assert "Model Runner V1" in fragment
    assert "Model Runner V2" in fragment
    assert "qwen30ba3b" in fragment
    assert "qwen32b" in fragment
    assert "Final finding:" in fragment


def test_findings_exclude_health_failed_rows(tmp_path: Path) -> None:
    rows = _cohort()
    histories = {
        row["wandb_run_id"]: _history(
            row,
            scale=2.0 if row["variant"] == "fastrl_threshold_v2_k5" else 1.0,
            activated=row["variant"] != "fastrl_threshold_v2_k5",
        )
        for row in rows
    }
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "output"

    assert (
        main(
            ["--manifest", str(manifest), "--output-dir", str(output_dir)],
            api=_FakeApi(histories),
        )
        == 1
    )

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    failed = next(row for row in payload if row["variant"] == "fastrl_threshold_v2_k5")
    assert failed["status"] == "health_failed"
    finding = next(
        line
        for line in (output_dir / "tail_gated_specdec.html")
        .read_text(encoding="utf-8")
        .splitlines()
        if "Final finding:" in line
    )
    assert "fastrl_threshold_v2_k5" not in finding


def test_output_directory_is_claimed_before_wandb_fetch(tmp_path: Path) -> None:
    rows = _cohort()
    manifest = tmp_path / "submissions.tsv"
    _write_manifest(manifest, rows)
    output_dir = tmp_path / "already-claimed"
    output_dir.mkdir()
    api = _api_for(rows)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--manifest", str(manifest), "--output-dir", str(output_dir)], api=api)

    assert api.calls == 0


def test_shuffled_manifest_produces_byte_identical_artifacts(tmp_path: Path) -> None:
    rows = [
        *_cohort(model="qwen32b", runner="v2"),
        *_cohort(model="qwen30ba3b", runner="v1"),
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_manifest = first_dir / "submissions.tsv"
    second_manifest = second_dir / "submissions.tsv"
    _write_manifest(first_manifest, rows)
    _write_manifest(second_manifest, reversed(rows))

    assert (
        main(
            [
                "--manifest",
                str(first_manifest),
                "--output-dir",
                str(first_dir / "output"),
            ],
            api=_api_for(rows),
        )
        == 0
    )
    assert (
        main(
            [
                "--manifest",
                str(second_manifest),
                "--output-dir",
                str(second_dir / "output"),
            ],
            api=_api_for(rows),
        )
        == 0
    )

    for filename in ("summary.csv", "summary.json", "tail_gated_specdec.html"):
        assert (first_dir / "output" / filename).read_bytes() == (
            second_dir / "output" / filename
        ).read_bytes()


def test_output_rows_include_full_metric_speedup_health_and_provenance_contract() -> (
    None
):
    rows = build_comparison_rows([_summary(row) for row in _cohort()])

    assert set(REQUIRED_ROW_FIELDS) == set(rows[0].to_dict())
    assert set(COHORT_FIELDS).issubset(rows[0].to_dict())
    baseline = next(row for row in rows if row.variant == "baseline_v2")
    assert baseline.to_dict()["draft_sample_method"] == "not_applicable"
    for field in (
        "run_dir",
        "slurm_log_path",
        "ray_driver_log_path",
        "ray_log_dir",
        "launcher_command",
        "command",
    ):
        assert baseline.to_dict()[field] == _metadata()[field]
