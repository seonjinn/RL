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

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
COLLECTOR = EXPERIMENT_DIR / "collect_cutedsl_ab_replicates.py"
WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT = 0.01
WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT = 0.02

CANONICAL_METRICS = {
    "timing/train/total_step_time": "timing/train/total_step_time",
    "timing/train/generation": "timing/train/generation",
    "timing/train/generation_finalize": "timing/train/generation_finalize",
    "timing/train/get_logprobs": "timing/train/policy_and_reference_logprobs",
    "timing/train/policy_training": "timing/train/policy_training",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "timing/train/prepare_for_generation/transfer_and_update_weights"
    ),
    "performance/tokens_per_sec_per_gpu": ("performance/tokens_per_sec_per_gpu"),
    "performance/generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu"
    ),
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
    "performance/policy_training_tokens_per_sec_per_gpu": (
        "performance/policy_training_tokens_per_sec_per_gpu"
    ),
    "train/total_num_tokens": "train/total_num_tokens",
    "train/global_valid_toks": "train/global_valid_toks",
    "train/mean_prompt_length": "train/mean_prompt_length",
    "train/num_valid_samples": "train/num_valid_samples",
    "train/total_turns": "train/total_turns",
}
RAW_FIELD_BY_CANONICAL_METRIC = {
    "timing/train/total_step_time": "total_step_seconds",
    "timing/train/generation": "generation_seconds",
    "timing/train/generation_finalize": "generation_finalize_seconds",
    "timing/train/get_logprobs": "logprob_seconds",
    "timing/train/policy_training": "policy_training_seconds",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "refit_transfer_update_seconds"
    ),
    "performance/tokens_per_sec_per_gpu": "e2e_tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu": (
        "generation_tokens_per_sec_per_gpu"
    ),
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
    "performance/policy_training_tokens_per_sec_per_gpu": (
        "policy_training_tokens_per_sec_per_gpu"
    ),
    "train/total_num_tokens": "total_num_tokens",
    "train/global_valid_toks": "global_valid_toks",
    "train/mean_prompt_length": "mean_prompt_length",
    "train/num_valid_samples": "num_valid_samples",
    "train/total_turns": "total_turns",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _raw_timing(
    *,
    job_id: str,
    arm: str,
    order_index: int,
    duration_ratio: float,
) -> dict[str, Any]:
    arm_duration_scale = duration_ratio if arm == "on" else 1.0
    measured_step_workload = []
    for offset, step in enumerate((6, 7, 8)):
        token_count = 1000.0 + offset * 100.0
        total_step_seconds = (10.0 + offset) * arm_duration_scale
        generation_seconds = (4.0 + offset) * arm_duration_scale
        logprob_seconds = (1.0 + offset / 10) * arm_duration_scale
        policy_training_seconds = (3.0 + offset / 10) * arm_duration_scale
        refit_seconds = (0.8 + offset / 10) * arm_duration_scale
        measured_step_workload.append(
            {
                "step": step,
                "total_step_seconds": total_step_seconds,
                "generation_seconds": generation_seconds,
                "generation_finalize_seconds": (0.4 + offset / 10) * arm_duration_scale,
                "logprob_seconds": logprob_seconds,
                "policy_training_seconds": policy_training_seconds,
                "refit_transfer_update_seconds": refit_seconds,
                "total_num_tokens": token_count,
                "global_valid_toks": token_count - 10.0,
                "mean_prompt_length": 128.0 + offset,
                "num_valid_samples": 2048.0,
                "total_turns": 2048.0,
                "e2e_tokens_per_sec_per_gpu": token_count / total_step_seconds / 4,
                "generation_tokens_per_sec_per_gpu": (
                    token_count / generation_seconds / 4
                ),
                "policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
                    token_count / logprob_seconds / 4
                ),
                "policy_training_tokens_per_sec_per_gpu": (
                    token_count / policy_training_seconds / 4
                ),
                "refit_effective_tokens_per_sec_per_gpu": (
                    token_count / refit_seconds / 4
                ),
            }
        )
    return {
        "run_id": job_id,
        "arm": arm,
        "order_index": order_index,
        "warmup_updates": 5,
        "measured_updates": 3,
        "training_gpu_count": 4,
        "resolved_metric_names": CANONICAL_METRICS,
        "policy_training_seconds": [
            row["policy_training_seconds"] for row in measured_step_workload
        ],
        "measured_component_series": {
            canonical_name: [
                {"step": row["step"], "value": row[row_field]}
                for row in measured_step_workload
            ]
            for canonical_name, row_field in RAW_FIELD_BY_CANONICAL_METRIC.items()
        },
        "measured_step_workload": measured_step_workload,
    }


def _workload_equivalence(
    raw_by_arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact_fields = ("mean_prompt_length", "num_valid_samples", "total_turns")
    exact_invariants_observed = all(
        [row[field] for row in raw_by_arm["on"]["measured_step_workload"]]
        == [row[field] for row in raw_by_arm["off"]["measured_step_workload"]]
        for field in exact_fields
    )
    metrics = {}
    observed = exact_invariants_observed
    for field in ("total_num_tokens", "global_valid_toks"):
        on = [row[field] for row in raw_by_arm["on"]["measured_step_workload"]]
        off = [row[field] for row in raw_by_arm["off"]["measured_step_workload"]]
        arm_total_relative_delta = abs(sum(on) - sum(off)) / (
            (sum(on) + sum(off)) / 2.0
        )
        max_paired_step_relative_delta = max(
            abs(on_value - off_value) / ((on_value + off_value) / 2.0)
            for on_value, off_value in zip(on, off, strict=True)
        )
        metrics[field] = {
            "on_total": sum(on),
            "off_total": sum(off),
            "arm_total_relative_delta": arm_total_relative_delta,
            "max_paired_step_relative_delta": max_paired_step_relative_delta,
        }
        observed = observed and (
            arm_total_relative_delta <= WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT
            and max_paired_step_relative_delta
            <= WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT
        )
    return {
        "schema_version": 2,
        "relative_delta_formula": "abs(on-off)/mean(on,off)",
        "required": True,
        "observed": observed,
        "actual_token_normalization_required": True,
        "normalization_metric": "train/total_num_tokens",
        "exact_observed_invariants": {
            "fields": list(exact_fields),
            "observed": exact_invariants_observed,
        },
        "prompt_sequence_identity_verified": False,
        "limits": {
            "arm_total_relative_delta": WORKLOAD_ARM_TOTAL_RELATIVE_DELTA_LIMIT,
            "paired_step_relative_delta": (WORKLOAD_PAIRED_STEP_RELATIVE_DELTA_LIMIT),
        },
        "metrics": metrics,
    }


def _renormalize_throughput(raw: dict[str, Any]) -> None:
    duration_by_throughput = {
        "e2e_tokens_per_sec_per_gpu": "total_step_seconds",
        "generation_tokens_per_sec_per_gpu": "generation_seconds",
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu": "logprob_seconds",
        "policy_training_tokens_per_sec_per_gpu": "policy_training_seconds",
        "refit_effective_tokens_per_sec_per_gpu": "refit_transfer_update_seconds",
    }
    for row in raw["measured_step_workload"]:
        for throughput_field, duration_field in duration_by_throughput.items():
            row[throughput_field] = (
                row["total_num_tokens"]
                / row[duration_field]
                / raw["training_gpu_count"]
            )


def _refresh_raw_projections(raw: dict[str, Any]) -> None:
    raw["policy_training_seconds"] = [
        row["policy_training_seconds"] for row in raw["measured_step_workload"]
    ]
    raw["measured_component_series"] = {
        canonical_name: [
            {"step": row["step"], "value": row[row_field]}
            for row in raw["measured_step_workload"]
        ]
        for canonical_name, row_field in RAW_FIELD_BY_CANONICAL_METRIC.items()
    }


def _refresh_summary_projections(
    summary: dict[str, Any], raw_by_arm: dict[str, dict[str, Any]]
) -> None:
    summary["median_policy_training_seconds"] = {
        arm: raw_by_arm[arm]["policy_training_seconds"][1] for arm in ("on", "off")
    }
    summary["median_normalized_throughput"] = {
        arm: raw_by_arm[arm]["measured_step_workload"][1][
            "policy_training_tokens_per_sec_per_gpu"
        ]
        for arm in ("on", "off")
    }
    summary["measured_total_num_tokens"] = {
        arm: [
            row["total_num_tokens"] for row in raw_by_arm[arm]["measured_step_workload"]
        ]
        for arm in ("on", "off")
    }


def _create_job(
    root: Path,
    *,
    job_id: str,
    replicate_index: int,
    timing_order: str,
    profile_enabled: bool,
    duration_ratio: float,
) -> Path:
    job_dir = root / job_id
    order = timing_order.split(",")
    raw_paths = []
    raw_by_arm = {}
    for order_index, arm in enumerate(order):
        raw_path = Path("timing") / f"{order_index}-{arm}" / "raw_timing.json"
        raw = _raw_timing(
            job_id=job_id,
            arm=arm,
            order_index=order_index,
            duration_ratio=duration_ratio,
        )
        _write_json(job_dir / raw_path, raw)
        raw_paths.append(str(raw_path))
        raw_by_arm[arm] = raw

    _write_json(
        job_dir / "status.json",
        {"run_id": job_id, "job_id": job_id, "exit_code": 0},
    )
    _write_json(
        job_dir / "benchmark_manifest.json",
        {
            "run_id": job_id,
            "replicate_index": replicate_index,
            "submission_group": "group-a",
            "timing_order": order,
            "warmup_updates": 5,
            "measured_updates": 3,
            "total_updates": 8,
            "profile_enabled": profile_enabled,
            "source_sha": "a" * 40,
            "upstream_ref": "origin/feature",
            "upstream_sha": "a" * 40,
            "image": "/images/nemo.sqsh",
            "image_sha256": "b" * 64,
            "base_config_sha256": "c" * 64,
            "artifact_revisions": {
                "model": {
                    "repo_id": "Qwen/Qwen3-30B-A3B",
                    "repo_type": None,
                    "revision": "d" * 40,
                },
                "dataset": {
                    "repo_id": "nvidia/OpenMathInstruct-2",
                    "repo_type": "dataset",
                    "revision": "e" * 40,
                    "split": "train_1M",
                    "num_rows": 1000000,
                },
            },
            "recipe": "recipes/cutedsl.yaml",
            "topology": {"num_nodes": 1, "gpus_per_node": 4},
            "fixed_config_evidence": {
                "on": {
                    "moe_grouped_gemm": True,
                    "grpo.val_period": 0,
                    "grpo.val_at_start": False,
                    "grpo.val_at_end": False,
                },
                "off": {
                    "moe_grouped_gemm": True,
                    "grpo.val_period": 0,
                    "grpo.val_at_start": False,
                    "grpo.val_at_end": False,
                },
            },
            "resolved_metric_names": {
                "on": CANONICAL_METRICS,
                "off": CANONICAL_METRICS,
            },
        },
    )
    _write_json(
        job_dir / "timing_summary.json",
        {
            "run_id": job_id,
            "timing_order": order,
            "raw_timing_files": raw_paths,
            "workload_metric": "train/total_num_tokens",
            "workload_equivalence": _workload_equivalence(raw_by_arm),
            "median_policy_training_seconds": {
                arm: raw_by_arm[arm]["policy_training_seconds"][1]
                for arm in ("on", "off")
            },
            "median_normalized_throughput": {
                arm: raw_by_arm[arm]["measured_step_workload"][1][
                    "policy_training_tokens_per_sec_per_gpu"
                ]
                for arm in ("on", "off")
            },
            "measured_total_num_tokens": {
                arm: [
                    item["total_num_tokens"]
                    for item in raw_by_arm[arm]["measured_step_workload"]
                ]
                for arm in ("on", "off")
            },
        },
    )
    if profile_enabled:
        _write_json(
            job_dir / "kernel_attribution.json",
            {
                "passed": True,
                "arms": {
                    "on": {
                        "kernel_evidence": "profiles/0-on/kernel_evidence.txt",
                        "fused_glu_match_count": 3,
                        "fused_dglu_match_count": 2,
                        "fused_quant_match_count": 4,
                        "fused_grouped_gemm_match_count": 9,
                        "baseline_expert_gemm_match_count": 0,
                    },
                    "off": {
                        "kernel_evidence": "profiles/1-off/kernel_evidence.txt",
                        "fused_glu_match_count": 0,
                        "fused_dglu_match_count": 0,
                        "fused_quant_match_count": 0,
                        "fused_grouped_gemm_match_count": 0,
                        "baseline_expert_gemm_match_count": 2,
                    },
                },
                "failures": [],
            },
        )
        for order_index, arm in enumerate(order):
            profile_dir = job_dir / "profiles" / f"{order_index}-{arm}"
            _write_json(
                profile_dir / "profile_summary.json",
                {
                    "arm": arm,
                    "order_index": order_index,
                    "nsight_report_count": 1,
                    "kernel_evidence": "kernel_evidence.txt",
                },
            )
            evidence = (
                "BlockScaledMoEGroupedGemmGluBiasKernel_object_at_0x1\n"
                "BlockScaledMoEGroupedGemmDgluDbiasKernel_object_at_0x2\n"
                "BlockScaledMoEGroupedGemmQuantKernel_object_at_0x3\n"
                if arm == "on"
                else "nvjet_sm100_128x128\n"
            )
            (profile_dir / "kernel_evidence.txt").write_text(evidence)
    return job_dir


def _create_valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    result_root = tmp_path / "benchmark-results"
    submission = tmp_path / "submission.jsonl"
    records = []
    for replicate_index, (order, duration_ratio) in enumerate(
        (
            ("on,off", 0.9),
            ("off,on", 1.0),
            ("on,off", 1.1),
        )
    ):
        job_id = str(100 + replicate_index)
        profile_enabled = replicate_index == 0
        _create_job(
            result_root,
            job_id=job_id,
            replicate_index=replicate_index,
            timing_order=order,
            profile_enabled=profile_enabled,
            duration_ratio=duration_ratio,
        )
        records.append(
            {
                "replicate_index": replicate_index,
                "timing_order": order,
                "profile_enabled": int(profile_enabled),
                "job_id": job_id,
                "submission_group": "group-a",
            }
        )
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    return submission, result_root


def _run_collector(
    submission: Path,
    result_root: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(submission),
            str(result_root),
            "--output-json",
            str(output_dir / "aggregate.json"),
            "--output-csv",
            str(output_dir / "aggregate.csv"),
            "--bootstrap-samples",
            "1000",
            "--bootstrap-seed",
            "2606",
        ],
        capture_output=True,
        text=True,
    )


def test_collector_writes_deterministic_paired_aggregate_json_and_csv(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    output_dir = tmp_path / "output"
    result = _run_collector(submission, result_root, output_dir)
    assert result.returncode == 0, result.stderr

    json_path = output_dir / "aggregate.json"
    csv_path = output_dir / "aggregate.csv"
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    aggregate = json.loads(first_json)
    assert aggregate["schema_version"] == 2
    assert aggregate["replicate_count"] == 3
    assert aggregate["replicate_indices"] == [0, 1, 2]
    assert aggregate["timing_orders"] == ["on,off", "off,on"]
    assert aggregate["profile_replicate"] == {
        "job_id": "100",
        "replicate_index": 0,
        "run_id": "100",
    }
    assert (
        aggregate["ratio_definition"]
        == "median(on measured steps) / median(off measured steps)"
    )
    assert set(aggregate["metrics"]) == {
        "e2e_duration",
        "generation_duration",
        "generation_finalize_duration",
        "logprob_duration",
        "policy_training_duration",
        "refit_duration",
        "e2e_throughput",
        "generation_throughput",
        "logprob_throughput",
        "policy_training_throughput",
        "refit_effective_throughput",
    }
    e2e_duration = aggregate["metrics"]["e2e_duration"]
    assert [item["ratio"] for item in e2e_duration["replicates"]] == [
        0.9,
        1.0,
        1.1,
    ]
    assert e2e_duration["median_ratio"] == 1.0
    assert e2e_duration["replicate_median_cv_percent"] == 10.0
    assert e2e_duration["paired_bootstrap_ci95"]["lower"] <= 1.0
    assert e2e_duration["paired_bootstrap_ci95"]["upper"] >= 1.0
    assert e2e_duration["recommendation"]["extend_to_six"] is True
    assert "CI crosses 1" in e2e_duration["recommendation"]["reasons"]
    assert set(e2e_duration["order_stratified"]) == {"off,on", "on,off"}
    assert e2e_duration["order_stratified"]["on,off"]["replicate_count"] == 2
    assert aggregate["metrics"]["e2e_throughput"]["median_ratio"] == 1.0

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["scope"] for row in rows} == {"aggregate", "order", "replicate"}
    assert len([row for row in rows if row["scope"] == "replicate"]) == 33
    assert len([row for row in rows if row["scope"] == "aggregate"]) == 11

    rerun = _run_collector(submission, result_root, output_dir)
    assert rerun.returncode == 0, rerun.stderr
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv


def test_collector_requires_three_distinct_completed_jobs(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = submission.read_text().splitlines()
    submission.write_text("\n".join(records[:2]) + "\n")
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "at least 3 distinct completed replicate jobs" in result.stderr

    submission.write_text("\n".join(records) + "\n")
    status_path = result_root / "101/status.json"
    status = json.loads(status_path.read_text())
    status["exit_code"] = 7
    _write_json(status_path, status)
    incomplete = _run_collector(submission, result_root, tmp_path / "output")
    assert incomplete.returncode != 0
    assert "job 101 is not completed successfully" in incomplete.stderr


def test_collector_requires_exactly_one_timing_summary_per_job(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    original = json.loads((result_root / "100/timing_summary.json").read_text())
    _write_json(result_root / "100/duplicate/timing_summary.json", original)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job 100 expected exactly one timing_summary.json, found 2" in result.stderr


def test_collector_rejects_functional_gate_before_loading_timing_artifacts(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "100/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["functional_gate"] = True
    manifest["performance_eligible"] = False
    _write_json(manifest_path, manifest)
    (result_root / "100/timing_summary.json").write_text("not JSON\n")

    result = _run_collector(submission, result_root, tmp_path / "output")

    assert result.returncode != 0
    assert "functional-gate evidence is not performance eligible" in result.stderr


def test_collector_requires_both_alternating_orders(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = [json.loads(line) for line in submission.read_text().splitlines()]
    records[1]["timing_order"] = "on,off"
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "submission timing order" in result.stderr


def test_collector_rejects_inconsistent_source_image_workload_and_metrics(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha"] = "c" * 40
    _write_json(manifest_path, manifest)
    source_result = _run_collector(submission, result_root, tmp_path / "output")
    assert source_result.returncode != 0
    assert "source identity differs across replicates" in source_result.stderr

    manifest["source_sha"] = "a" * 40
    manifest["image_sha256"] = "d" * 64
    _write_json(manifest_path, manifest)
    image_result = _run_collector(submission, result_root, tmp_path / "output")
    assert image_result.returncode != 0
    assert "image identity differs across replicates" in image_result.stderr

    manifest["image_sha256"] = "b" * 64
    manifest["topology"]["gpus_per_node"] = 8
    _write_json(manifest_path, manifest)
    workload_result = _run_collector(submission, result_root, tmp_path / "output")
    assert workload_result.returncode != 0
    assert "training_gpu_count differs from topology" in workload_result.stderr

    manifest["topology"]["gpus_per_node"] = 4
    manifest["resolved_metric_names"]["on"]["timing/train/get_logprobs"] = (
        "timing/train/get_logprobs"
    )
    manifest["resolved_metric_names"]["off"]["timing/train/get_logprobs"] = (
        "timing/train/get_logprobs"
    )
    _write_json(manifest_path, manifest)
    for raw_path in (result_root / "102/timing").glob("*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        raw["resolved_metric_names"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(raw_path, raw)
    metrics_result = _run_collector(submission, result_root, tmp_path / "output")
    assert metrics_result.returncode != 0
    assert "resolved metric names differ across replicates" in metrics_result.stderr


def test_collector_accepts_live_workload_within_predeclared_equivalence_bounds(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for job_id in ("100", "101", "102"):
        summary_path = result_root / job_id / "timing_summary.json"
        summary = json.loads(summary_path.read_text())
        raw_by_arm = {}
        for raw_path in (result_root / job_id / "timing").glob("*/raw_timing.json"):
            raw = json.loads(raw_path.read_text())
            arm = raw["arm"]
            if arm == "on":
                for row in raw["measured_step_workload"]:
                    row["total_num_tokens"] = float(
                        round(row["total_num_tokens"] * 1.005)
                    )
                    row["global_valid_toks"] = float(
                        round(row["global_valid_toks"] * 1.005)
                    )
                _renormalize_throughput(raw)
                _refresh_raw_projections(raw)
            _write_json(raw_path, raw)
            raw_by_arm[arm] = raw
        summary["workload_equivalence"] = _workload_equivalence(raw_by_arm)
        _refresh_summary_projections(summary, raw_by_arm)
        _write_json(summary_path, summary)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode == 0, result.stderr


def test_collector_rejects_out_of_bounds_or_tampered_workload_equivalence(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_dir = result_root / "101"
    raw_path = next((job_dir / "timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0]["total_num_tokens"] = float(
        round(raw["measured_step_workload"][0]["total_num_tokens"] * 1.03)
    )
    raw["measured_step_workload"][0]["global_valid_toks"] = float(
        round(raw["measured_step_workload"][0]["global_valid_toks"] * 1.03)
    )
    _renormalize_throughput(raw)
    _refresh_raw_projections(raw)
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "workload equivalence summary does not match raw timing" in result.stderr

    summary_path = job_dir / "timing_summary.json"
    summary = json.loads(summary_path.read_text())
    raw_by_arm = {
        json.loads(path.read_text())["arm"]: json.loads(path.read_text())
        for path in (job_dir / "timing").glob("*/raw_timing.json")
    }
    summary["workload_equivalence"] = _workload_equivalence(raw_by_arm)
    _refresh_summary_projections(summary, raw_by_arm)
    _write_json(summary_path, summary)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "workload equivalence limits exceeded" in result.stderr


def test_collector_rejects_measured_step_sequence_mismatch(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = next((result_root / "102/timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0]["step"] = 99
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "measured step sequence" in result.stderr


def test_collector_rejects_throughput_not_normalized_by_actual_tokens(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = next((result_root / "101/timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0]["policy_training_tokens_per_sec_per_gpu"] *= 1.01
    _refresh_raw_projections(raw)
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "is not normalized by actual total_num_tokens" in result.stderr


def test_collector_rejects_raw_rows_that_differ_from_component_series(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = next((result_root / "101/timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["measured_component_series"]["train/total_num_tokens"][0]["value"] += 1
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "measured row differs from component series" in result.stderr


def test_collector_rejects_same_wrong_measured_step_window_in_every_arm(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for raw_path in result_root.glob("*/timing/*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        for index, row in enumerate(raw["measured_step_workload"], start=1):
            row["step"] = index
        for series in raw["measured_component_series"].values():
            for index, point in enumerate(series, start=1):
                point["step"] = index
        _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "measured step sequence differs from manifest window" in result.stderr


def test_collector_rejects_mixed_submission_groups(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = [json.loads(line) for line in submission.read_text().splitlines()]
    records[-1]["submission_group"] = "group-b"
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["submission_group"] = "group-b"
    _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "submission group differs across replicates" in result.stderr


def test_collector_rejects_tampered_summary_projection(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    summary_path = result_root / "101/timing_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["median_policy_training_seconds"]["on"] += 0.25
    _write_json(summary_path, summary)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "median_policy_training_seconds differs from raw timing" in result.stderr


@pytest.mark.parametrize("field", ("num_valid_samples", "total_turns"))
def test_collector_rejects_nonintegral_observed_count(
    tmp_path: Path, field: str
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = next((result_root / "101/timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0][field] = 2048.5
    _refresh_raw_projections(raw)
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert f"{field} step 6 must be an integral count" in result.stderr


def test_collector_rejects_mixed_base_config_hashes(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_config_sha256"] = "d" * 64
    _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "workload identity differs across replicates" in result.stderr


def test_collector_rejects_mixed_model_revisions(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_revisions"]["model"]["revision"] = "f" * 40
    _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "workload identity differs across replicates" in result.stderr


def test_collector_rejects_noncanonical_dataset_row_count(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_revisions"]["dataset"]["num_rows"] = 999_999
    _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "dataset num_rows must equal 1000000" in result.stderr


def test_collector_requires_profile_and_kernel_attribution_for_designated_job(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    (result_root / "100/kernel_attribution.json").unlink()
    missing_result = _run_collector(submission, result_root, tmp_path / "output")
    assert missing_result.returncode != 0
    assert "designated profile job 100" in missing_result.stderr
    assert "kernel_attribution.json" in missing_result.stderr

    _write_json(
        result_root / "100/kernel_attribution.json",
        {"passed": False, "arms": {}, "failures": ["missing kernels"]},
    )
    failed_result = _run_collector(submission, result_root, tmp_path / "output")
    assert failed_result.returncode != 0
    assert "kernel attribution did not pass" in failed_result.stderr


@pytest.mark.parametrize(
    ("arm", "field", "invalid_value"),
    (
        ("on", "fused_glu_match_count", 0),
        ("on", "fused_dglu_match_count", 0),
        ("on", "fused_quant_match_count", 0),
        ("on", "fused_grouped_gemm_match_count", 0),
        ("off", "fused_glu_match_count", 1),
        ("off", "fused_dglu_match_count", 1),
        ("off", "fused_quant_match_count", 1),
        ("off", "fused_grouped_gemm_match_count", 1),
        ("off", "baseline_expert_gemm_match_count", 0),
        ("on", "quant_match_count", 4),
        ("off", "grouped_gemm_match_count", 2),
    ),
)
def test_collector_rejects_invalid_exact_kernel_attribution_schema(
    tmp_path: Path,
    arm: str,
    field: str,
    invalid_value: int,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    attribution_path = result_root / "100/kernel_attribution.json"
    attribution = json.loads(attribution_path.read_text())
    attribution["arms"][arm][field] = invalid_value
    _write_json(attribution_path, attribution)

    result = _run_collector(submission, result_root, tmp_path / "output")

    assert result.returncode != 0
    assert field in result.stderr


@pytest.mark.parametrize("unsafe_job_id", ("../outside", "/tmp/outside", "nested/job"))
def test_collector_rejects_job_ids_outside_benchmark_root(
    tmp_path: Path, unsafe_job_id: str
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = [json.loads(line) for line in submission.read_text().splitlines()]
    records[0]["job_id"] = unsafe_job_id
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job_id must be a safe single path component" in result.stderr


def test_collector_rejects_artifact_symlink_outside_benchmark_root(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    status_path = result_root / "100/status.json"
    outside_status = tmp_path / "outside-status.json"
    outside_status.write_text(status_path.read_text())
    status_path.unlink()
    status_path.symlink_to(outside_status)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job 100 status escapes benchmark result root" in result.stderr


def test_collector_requires_identical_nonempty_on_off_metric_names(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["resolved_metric_names"]["on"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(manifest_path, manifest)
        on_path = next(job_dir.glob("timing/*-on/raw_timing.json"))
        raw = json.loads(on_path.read_text())
        raw["resolved_metric_names"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(on_path, raw)
    asymmetric = _run_collector(submission, result_root, tmp_path / "output")
    assert asymmetric.returncode != 0
    assert "ON/OFF resolved metric names must match" in asymmetric.stderr

    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for arm in ("on", "off"):
            manifest["resolved_metric_names"][arm]["timing/train/get_logprobs"] = None
        _write_json(manifest_path, manifest)
        for raw_path in job_dir.glob("timing/*/raw_timing.json"):
            raw = json.loads(raw_path.read_text())
            raw["resolved_metric_names"]["timing/train/get_logprobs"] = None
            _write_json(raw_path, raw)
    empty = _run_collector(submission, result_root, tmp_path / "output")
    assert empty.returncode != 0
    assert "resolved metric name" in empty.stderr
    assert "nonempty string" in empty.stderr


def test_collector_rejects_mismatched_on_off_fixed_config_evidence(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for manifest_path in result_root.glob("*/benchmark_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest["fixed_config_evidence"]["off"]["grpo.val_period"] = 10
        _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "output")

    assert result.returncode != 0
    assert "ON/OFF fixed_config_evidence must match" in result.stderr


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    (
        ("source_sha", None, "source_sha must be a 40-character hexadecimal SHA"),
        ("image_sha256", "bad", "image_sha256 must be a 64-character hexadecimal SHA"),
        (
            "artifact_revisions",
            {},
            "artifact_revisions must contain model and dataset",
        ),
        ("recipe", "", "recipe must be a nonempty string"),
        ("topology", {}, "topology must be a nonempty object"),
        (
            "fixed_config_evidence",
            {},
            "fixed_config_evidence must contain ON/OFF objects",
        ),
    ),
)
def test_collector_rejects_missing_or_malformed_identity_fields(
    tmp_path: Path, field: str, invalid_value: Any, expected_error: str
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for manifest_path in result_root.glob("*/benchmark_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = invalid_value
        _write_json(manifest_path, manifest)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_requires_raw_artifacts_to_evidence_timing_order(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = result_root / "100/timing/0-on/raw_timing.json"
    raw = json.loads(raw_path.read_text())
    raw["order_index"] = 1
    _write_json(raw_path, raw)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "raw timing order_index does not evidence timing order" in result.stderr


def test_collector_resolves_unique_successful_restarted_job_run(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    restarted_dir = result_root / "100-r1"
    shutil.move(result_root / "100", restarted_dir)
    status_path = restarted_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["run_id"] = "100-r1"
    _write_json(status_path, status)
    for artifact in ("benchmark_manifest.json", "timing_summary.json"):
        artifact_path = restarted_dir / artifact
        value = json.loads(artifact_path.read_text())
        value["run_id"] = "100-r1"
        _write_json(artifact_path, value)
    for raw_path in restarted_dir.glob("timing/*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        raw["run_id"] = "100-r1"
        _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "output/aggregate.json").read_text())
    assert aggregate["profile_replicate"] == {
        "job_id": "100",
        "replicate_index": 0,
        "run_id": "100-r1",
    }


def test_collector_refuses_to_overwrite_submission_jsonl(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    original = submission.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(submission),
            str(result_root),
            "--output-json",
            str(submission),
            "--output-csv",
            str(tmp_path / "aggregate.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "output paths must not overwrite the submission JSONL" in result.stderr
    assert submission.read_bytes() == original
