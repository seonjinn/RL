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

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
COLLECTOR = EXPERIMENT_DIR / "collect_nemo2606_dependency_factorial.py"
CONTEXTS = ("g0a0", "g1a0", "g0a1", "g1a1")
CONTEXT_FACTORS = {
    "g0a0": 1.0,
    "g1a0": 0.8,
    "g0a1": 0.9,
    "g1a1": 0.6,
}
WARMUP_UPDATES = 5
MEASURED_UPDATES = 20
TOTAL_UPDATES = WARMUP_UPDATES + MEASURED_UPDATES
MEASURED_STEPS = tuple(range(WARMUP_UPDATES + 1, TOTAL_UPDATES + 1))
DIRECT_PATH_LIMITATION = (
    "a2a_only_without_cutedsl lacks OFF-arm temporal overlap proof; "
    "direct timing effect is associative only"
)
HARNESS_REPRESENTATIVE_LIMITATION = (
    "deterministic representative report; only one representative process/rank "
    "analyzed; no all-rank aggregation"
)
CANONICAL_METRICS = {
    "timing/train/total_step_time": "timing/train/total_step_time",
    "timing/train/generation": "timing/train/generation",
    "timing/train/generation_finalize": "timing/train/generation_finalize",
    "timing/train/policy_training": "timing/train/policy_training",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "timing/train/prepare_for_generation/transfer_and_update_weights"
    ),
    "performance/tokens_per_sec_per_gpu": "performance/tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu"
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
    "timing/train/policy_training": "policy_training_seconds",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "refit_transfer_update_seconds"
    ),
    "performance/tokens_per_sec_per_gpu": "e2e_tokens_per_sec_per_gpu",
    "performance/generation_tokens_per_sec_per_gpu": (
        "generation_tokens_per_sec_per_gpu"
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


def _context_flags(context: str) -> tuple[bool, bool]:
    return context[1] == "1", context[3] == "1"


def _raw_timing(
    *,
    job_id: str,
    arm: str,
    order_index: int,
    context: str,
    replicate_index: int,
) -> dict[str, Any]:
    context_factor = CONTEXT_FACTORS[context]
    cutedsl_factor = 1.0 if arm == "on" else 1.0 / 0.95
    replicate_factor = 1.0 + replicate_index * 0.01
    duration_factor = context_factor * cutedsl_factor * replicate_factor
    workload = []
    for offset, step in enumerate(MEASURED_STEPS):
        tokens = 1000.0 + offset * 100.0
        total_step = (10.0 + offset) * duration_factor
        generation = (4.0 + offset) * duration_factor
        policy = (3.0 + offset / 10.0) * duration_factor
        refit = (0.8 + offset / 10.0) * duration_factor
        workload.append(
            {
                "step": step,
                "total_step_seconds": total_step,
                "generation_seconds": generation,
                "generation_finalize_seconds": 0.4 * duration_factor,
                "policy_training_seconds": policy,
                "refit_transfer_update_seconds": refit,
                "total_num_tokens": tokens,
                "global_valid_toks": tokens - 10.0,
                "mean_prompt_length": 128.0 + offset,
                "num_valid_samples": 8.0,
                "total_turns": 8.0,
                "e2e_tokens_per_sec_per_gpu": tokens / total_step / 4.0,
                "generation_tokens_per_sec_per_gpu": tokens / generation / 4.0,
                "policy_training_tokens_per_sec_per_gpu": tokens / policy / 4.0,
                "refit_effective_tokens_per_sec_per_gpu": tokens / refit / 4.0,
            }
        )
    return {
        "run_id": job_id,
        "arm": arm,
        "order_index": order_index,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "training_gpu_count": 4,
        "resolved_metric_names": CANONICAL_METRICS,
        "policy_training_seconds": [row["policy_training_seconds"] for row in workload],
        "measured_component_series": {
            canonical_name: [
                {"step": row["step"], "value": row[field]} for row in workload
            ]
            for canonical_name, field in RAW_FIELD_BY_CANONICAL_METRIC.items()
        },
        "measured_step_workload": workload,
    }


def _workload_equivalence(raw_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = set(raw_by_arm) == {"on", "off"}
    return {
        "schema_version": 2,
        "required": required,
        "observed": True,
        "actual_token_normalization_required": True,
        "normalization_metric": "train/total_num_tokens",
        "exact_observed_invariants": {
            "fields": ["mean_prompt_length", "num_valid_samples", "total_turns"],
            "observed": True,
        },
        "prompt_sequence_identity_verified": False,
        "limits": {
            "arm_total_relative_delta": 0.01,
            "paired_step_relative_delta": 0.02,
        },
        "metrics": {
            field: {
                "on_total": sum(
                    row[field] for row in raw_by_arm["on"]["measured_step_workload"]
                ),
                "off_total": sum(
                    row[field]
                    for row in raw_by_arm.get("off", raw_by_arm["on"])[
                        "measured_step_workload"
                    ]
                ),
                "arm_total_relative_delta": 0.0,
                "max_paired_step_relative_delta": 0.0,
            }
            for field in ("total_num_tokens", "global_valid_toks")
        }
        if required
        else {},
        **({} if required else {"not_applicable_reason": "single timing arm"}),
    }


def _write_temporal_analyzer(
    job_dir: Path,
    job_id: str,
    *,
    temporal_overlap_verified: bool = True,
    overlap_duration_ns: int = 5000,
    a2a_overlap_ratio: float = 0.5,
    gemm_overlap_ratio: float = 0.25,
) -> None:
    profile_path = job_dir / "profiles/on.nsys-rep"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_bytes(f"synthetic-profile-{job_id}\n".encode())
    _write_json(
        job_dir / "a2a_temporal_overlap.json",
        {
            "schema_version": 1,
            "source_profile_sha256": hashlib.sha256(
                profile_path.read_bytes()
            ).hexdigest(),
            "a2a_interval_count": 7,
            "expert_gemm_interval_count": 11,
            "overlap_duration_ns": overlap_duration_ns,
            "a2a_overlap_ratio": a2a_overlap_ratio,
            "gemm_overlap_ratio": gemm_overlap_ratio,
            "temporal_overlap_verified": temporal_overlap_verified,
            "limitations": []
            if temporal_overlap_verified
            else ["synthetic analyzer did not prove temporal overlap"],
        },
    )


def _create_job(
    root: Path,
    *,
    job_id: str,
    context: str,
    replicate_index: int,
    profile_enabled: bool,
    temporal_overlap_verified: bool,
) -> None:
    full_cg_enabled, a2a_enabled = _context_flags(context)
    order = (
        ["on"]
        if full_cg_enabled
        else (["on", "off"] if replicate_index % 2 == 0 else ["off", "on"])
    )
    job_dir = root / job_id
    raw_by_arm = {}
    raw_paths = []
    for order_index, arm in enumerate(order):
        arm_dir = job_dir / "timing" / f"{order_index}-{arm}"
        raw = _raw_timing(
            job_id=job_id,
            arm=arm,
            order_index=order_index,
            context=context,
            replicate_index=replicate_index,
        )
        _write_json(arm_dir / "raw_timing.json", raw)
        metrics = {
            source_name: {
                str(point["step"]): point["value"]
                for point in raw["measured_component_series"][canonical_name]
            }
            for canonical_name, source_name in CANONICAL_METRICS.items()
        }
        if full_cg_enabled:
            all_steps = range(1, TOTAL_UPDATES + 1)
            metrics.update(
                {
                    "train/full_cuda_graph_warmup_calls": {
                        str(step): min(step, 3) for step in all_steps
                    },
                    "train/full_cuda_graph_capture_calls": {
                        str(step): int(step >= 3) for step in all_steps
                    },
                    "train/full_cuda_graph_replay_calls": {
                        str(step): max(0, step - 2) for step in all_steps
                    },
                    "train/full_cuda_graph_reset_calls": {
                        str(step): 0 for step in all_steps
                    },
                }
            )
        _write_json(arm_dir / "metrics.json", metrics)
        raw_by_arm[arm] = raw
        raw_paths.append(str((arm_dir / "raw_timing.json").relative_to(job_dir)))

    fixed_config = {
        "policy.megatron_cfg.moe_grouped_gemm": True,
        "policy.megatron_cfg.env_vars.CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "policy.megatron_cfg.overlap_moe_expert_parallel_comm": a2a_enabled,
        "policy.megatron_cfg.high_priority_a2a_comm_stream": a2a_enabled,
        "policy.megatron_cfg.delay_wgrad_compute": a2a_enabled,
        "policy.train_global_batch_size": 8,
        "policy.train_micro_batch_size": 1,
        "loss_fn.force_on_policy_ratio": True,
        "grpo.seq_logprob_error_threshold": None,
        "grpo.skip_reference_policy_logprobs_calculation": True,
        "loss_fn.reference_policy_kl_penalty": 0.0,
    }
    _write_json(
        job_dir / "status.json",
        {"run_id": job_id, "job_id": job_id, "exit_code": 0},
    )
    _write_json(
        job_dir / "benchmark_manifest.json",
        {
            "functional_gate": False,
            "performance_eligible": True,
            "run_id": job_id,
            "replicate_index": replicate_index,
            "submission_group": "factorial-group",
            "timing_order": order,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "total_updates": TOTAL_UPDATES,
            "profile_enabled": profile_enabled,
            "feature_context": context,
            "full_cg_enabled": full_cg_enabled,
            "a2a_enabled": a2a_enabled,
            "available_arms": order,
            "not_applicable_arms": {
                "off": "full-iteration CUDA Graph requires device-initiated CuTeDSL"
            }
            if full_cg_enabled
            else {},
            "aggregation_scope": (
                "context_single_arm"
                if full_cg_enabled
                else "context_local_cutedsl_pair"
            ),
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
                    "num_rows": 1_000_000,
                },
            },
            "recipe": "recipes/qwen3-30ba3b-noncolocated.yaml",
            "topology": {
                "num_nodes": 2,
                "gpus_per_node": 4,
                "segment_size": None,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 4,
            },
            "workload": {
                "train_global_batch_size": 8,
                "train_micro_batch_size": 1,
                "num_prompts_per_step": 4,
                "num_generations_per_prompt": 2,
            },
            "fixed_config_evidence": {arm: fixed_config for arm in order},
            "full_cg_config_evidence": {
                arm: {
                    "cuda_graph_impl": "full_iteration" if full_cg_enabled else "none",
                    "cuda_graph_warmup_steps": 2 if full_cg_enabled else None,
                    "cuda_graph_use_single_mempool": (
                        True if full_cg_enabled else None
                    ),
                }
                for arm in order
            },
            "resolved_metric_names": {arm: CANONICAL_METRICS for arm in order},
        },
    )
    equivalence = _workload_equivalence(raw_by_arm)
    _write_json(
        job_dir / "timing_summary.json",
        {
            "run_id": job_id,
            "timing_order": order,
            "raw_timing_files": raw_paths,
            "available_arms": order,
            "workload_metric": "train/total_num_tokens",
            "workload_equivalence": equivalence,
            "measured_total_num_tokens": {
                arm: [row["total_num_tokens"] for row in raw["measured_step_workload"]]
                for arm, raw in raw_by_arm.items()
            },
        },
    )
    if profile_enabled:
        _write_json(
            job_dir / "feature_attribution.json",
            {
                "feature_context": context,
                "full_cg_enabled": full_cg_enabled,
                "a2a_enabled": a2a_enabled,
                "counts": {
                    arm: {
                        "cuda_graph_launch_api": 3 if full_cg_enabled else 0,
                        "nccl_a2a_kernel": 7,
                    }
                    for arm in order
                },
                "kernel_presence_passed": True,
                "full_iteration_replay_verified": (True if full_cg_enabled else None),
                "a2a_temporal_overlap_verified": (False if a2a_enabled else None),
                "performance_claim_eligible": False,
            },
        )
        if not full_cg_enabled:
            _write_json(
                job_dir / "kernel_attribution.json",
                {
                    "passed": True,
                    "arms": {
                        "on": {
                            "fused_glu_match_count": 3,
                            "fused_dglu_match_count": 2,
                            "fused_quant_match_count": 4,
                            "fused_grouped_gemm_match_count": 9,
                            "baseline_expert_gemm_match_count": 0,
                        },
                        "off": {
                            "fused_glu_match_count": 0,
                            "fused_dglu_match_count": 0,
                            "fused_quant_match_count": 0,
                            "fused_grouped_gemm_match_count": 0,
                            "baseline_expert_gemm_match_count": 2,
                        },
                    },
                },
            )
    if profile_enabled:
        _write_temporal_analyzer(
            job_dir,
            job_id,
            temporal_overlap_verified=(
                temporal_overlap_verified if a2a_enabled else False
            ),
            overlap_duration_ns=5000 if a2a_enabled else 0,
            a2a_overlap_ratio=0.5 if a2a_enabled else 0.0,
            gemm_overlap_ratio=0.25 if a2a_enabled else 0.0,
        )


def _create_valid_inputs(
    tmp_path: Path, *, temporal_overlap_verified: bool = True
) -> tuple[Path, Path]:
    result_root = tmp_path / "results"
    submission = tmp_path / "submission.jsonl"
    records_by_replicate: dict[int, dict[str, dict[str, Any]]] = {}
    for replicate_index in range(3):
        records_by_replicate[replicate_index] = {}
        for context_index, context in enumerate(CONTEXTS):
            job_id = str(1000 + replicate_index * 10 + context_index)
            full_cg_enabled, a2a_enabled = _context_flags(context)
            profile_enabled = replicate_index == 0
            timing_order = (
                "on"
                if full_cg_enabled
                else ("on,off" if replicate_index % 2 == 0 else "off,on")
            )
            _create_job(
                result_root,
                job_id=job_id,
                context=context,
                replicate_index=replicate_index,
                profile_enabled=profile_enabled,
                temporal_overlap_verified=temporal_overlap_verified,
            )
            records_by_replicate[replicate_index][context] = {
                "factorial_context": context,
                "full_cg_enabled": full_cg_enabled,
                "a2a_enabled": a2a_enabled,
                "replicate_index": replicate_index,
                "timing_order": timing_order,
                "profile_enabled": profile_enabled,
                "job_id": job_id,
                "submission_group": "factorial-group",
            }
    records = []
    for replicate_index in range(3):
        offset = replicate_index % len(CONTEXTS)
        balanced_order = (*CONTEXTS[offset:], *CONTEXTS[:offset])
        records.extend(
            records_by_replicate[replicate_index][context] for context in balanced_order
        )
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    return submission, result_root


def _run_collector(
    submission: Path,
    result_root: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(submission),
            str(result_root),
            "--output-json",
            str(output),
            "--bootstrap-samples",
            "1000",
            "--bootstrap-seed",
            "2606",
        ],
        capture_output=True,
        text=True,
    )


def _submission_records(submission: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in submission.read_text().splitlines()]


def _write_submission(submission: Path, records: list[dict[str, Any]]) -> None:
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))


def _job_id(submission: Path, context: str, replicate_index: int) -> str:
    return next(
        record["job_id"]
        for record in _submission_records(submission)
        if record["factorial_context"] == context
        and record["replicate_index"] == replicate_index
    )


def _add_logprob_series(result_root: Path, value: float) -> None:
    logprob_metrics = {
        "timing/train/get_logprobs": "timing/train/policy_and_reference_logprobs",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
        ),
    }
    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for mapping in manifest["resolved_metric_names"].values():
            mapping.update(logprob_metrics)
        _write_json(manifest_path, manifest)
        for arm_dir in (job_dir / "timing").iterdir():
            raw_path = arm_dir / "raw_timing.json"
            raw = json.loads(raw_path.read_text())
            raw["resolved_metric_names"].update(logprob_metrics)
            raw["measured_component_series"].update(
                {
                    name: [
                        {"step": row["step"], "value": value}
                        for row in raw["measured_step_workload"]
                    ]
                    for name in logprob_metrics
                }
            )
            for row in raw["measured_step_workload"]:
                row["logprob_seconds"] = value
                row["policy_and_reference_logprobs_tokens_per_sec_per_gpu"] = value
            _write_json(raw_path, raw)
            metrics_path = arm_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            for source_name in logprob_metrics.values():
                metrics[source_name] = {str(step): value for step in MEASURED_STEPS}
            _write_json(metrics_path, metrics)


def test_collector_writes_dependency_constrained_factorial(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    output = tmp_path / "aggregate.json"

    result = _run_collector(submission, result_root, output)

    assert result.returncode == 0, result.stderr
    aggregate = json.loads(output.read_text())
    assert aggregate["schema_version"] == 1
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["claim_ready"] is False
    assert aggregate["provisional_reasons"] == [DIRECT_PATH_LIMITATION]
    assert aggregate["paired_replicate_indices"] == [0, 1, 2]
    assert aggregate["context_replicate_counts"] == {context: 3 for context in CONTEXTS}
    assert len(aggregate["cohort_evidence_digest"]) == 64
    assert len(aggregate["job_evidence_digests"]) == 12
    assert all(
        len(record["evidence_digest"]) == 64
        for record in aggregate["job_evidence_digests"]
    )
    assert aggregate["contract"] == {
        "policy_training_gpu_count": 4,
        "train_global_batch_size": 8,
        "train_micro_batch_size": 1,
        "expert_model_parallel_size": 4,
        "cross_context_arm": "on",
        "cutedsl_off_scope": ["g0a0", "g0a1"],
    }
    assert set(aggregate["factorial_effects"]) == {
        "e2e",
        "generation",
        "logprob",
        "policy_training",
        "refit",
    }
    assert aggregate["factorial_effects"]["logprob"] == {
        "status": "not_applicable",
        "reason": "disabled_by_supported_full_cg_slice",
    }
    policy = aggregate["factorial_effects"]["policy_training"]
    for measurement in ("duration", "throughput"):
        effects = policy[measurement]
        assert set(effects) == {
            "full_cg_at_a0",
            "full_cg_at_a1",
            "a2a_at_g0",
            "a2a_at_g1",
            "full_cg_main",
            "a2a_main",
            "interaction",
        }
        assert effects["interaction"]["replicate_count"] == 3
        assert effects["interaction"]["median_percent"] == pytest.approx(20.0)
        assert effects["interaction"]["min_percent"] == pytest.approx(20.0)
        assert effects["interaction"]["max_percent"] == pytest.approx(20.0)
        assert effects["interaction"]["bootstrap_ci95_percent"] == {
            "lower": pytest.approx(20.0),
            "upper": pytest.approx(20.0),
        }
    assert aggregate["effect_definitions"] == {
        "duration": "baseline / feature - 1",
        "throughput": "feature / baseline - 1",
        "main_effect": "geometric mean of the two dependency-valid conditional factors",
        "interaction": (
            "A2A optimization bundle factor with full-CG / factor without full-CG - 1"
        ),
    }
    assert aggregate["direct_path_effect_definitions"] == {
        "baseline": "g0a0 OFF (CuTeDSL OFF, full-CG OFF, A2A OFF)",
        "cutedsl_only": "g0a0 OFF -> g0a0 ON",
        "a2a_only_without_cutedsl": ("g0a0 OFF -> g0a1 OFF (A2A optimization bundle)"),
        "cutedsl_a2a": "g0a0 OFF -> g0a1 ON (A2A optimization bundle)",
        "cutedsl_full_cg": "g0a0 OFF -> g1a0 ON",
        "all_three_combined": "g0a0 OFF -> g1a1 ON",
    }
    expected_direct_percent = {
        "cutedsl_only": (1.0 / 0.95 - 1.0) * 100.0,
        "a2a_only_without_cutedsl": (1.0 / 0.9 - 1.0) * 100.0,
        "cutedsl_a2a": (1.0 / (0.95 * 0.9) - 1.0) * 100.0,
        "cutedsl_full_cg": (1.0 / (0.95 * 0.8) - 1.0) * 100.0,
        "all_three_combined": (1.0 / (0.95 * 0.6) - 1.0) * 100.0,
    }
    for component in ("e2e", "generation", "policy_training", "refit"):
        for measurement in ("duration", "throughput"):
            effects = aggregate["direct_path_effects"][component][measurement]
            assert set(effects) == set(expected_direct_percent)
            for effect, expected_percent in expected_direct_percent.items():
                assert effects[effect]["replicate_count"] == 3
                assert effects[effect]["claim_status"] == "provisional"
                assert effects[effect]["median_percent"] == pytest.approx(
                    expected_percent
                )
    assert aggregate["direct_path_effects"]["full_cg_without_cutedsl"] == {
        "status": "unsupported_dependency",
        "reason": "full-iteration CUDA Graph requires device-initiated CuTeDSL kernels",
    }
    assert aggregate["direct_path_effects"]["logprob"] == {
        "status": "not_applicable",
        "reason": "disabled_by_supported_full_cg_slice",
    }
    assert aggregate["dependency_notes"] == {
        "a2a_optimization_bundle": (
            "expert-parallel overlap + high-priority A2A stream + delayed wgrad compute"
        ),
        "common_baseline": "all three features disabled in g0a0 OFF",
        "a2a_only_without_cutedsl": (
            "timing effect is computed, but OFF-arm temporal overlap is not "
            "profiled; mechanistic attribution is provisional"
        ),
        "full_cg_without_cutedsl": (
            "unsupported because full-iteration CUDA Graph requires "
            "device-initiated CuTeDSL kernels"
        ),
        "incremental_effects": (
            "factorial_effects and cutedsl_g0_effects remain available"
        ),
        "profile_overlap_contrasts": (
            "representative-process exploratory evidence; not all-rank causal proof"
        ),
    }
    for level, full_cg_enabled in (("g0", False), ("g1", True)):
        contrast = aggregate["a2a_overlap_ratio_contrasts"][level]
        assert contrast["full_cg_enabled"] is full_cg_enabled
        assert contrast["baseline_context"] == f"{level}a0"
        assert contrast["overlap_context"] == f"{level}a1"
        assert contrast["evidence_scope"] == "representative_process_exploratory"
        assert contrast["all_pairs_increased"] is True
        assert contrast["median_baseline_ratio"] == pytest.approx(0.0)
        assert contrast["median_overlap_ratio"] == pytest.approx(0.5)
        assert contrast["median_absolute_increase"] == pytest.approx(0.5)
        assert contrast["claim_status"] == "claim_ready"
        assert contrast["paired_profile_replicates"] == [
            {
                "replicate_index": 0,
                "baseline_ratio": pytest.approx(0.0),
                "overlap_ratio": pytest.approx(0.5),
                "absolute_increase": pytest.approx(0.5),
                "relative_factor": None,
                "increased": True,
            }
        ]
    assert set(aggregate["cutedsl_g0_effects"]) == {"g0a0", "g0a1"}
    assert aggregate["cutedsl_g0_effects"]["g0a0"]["logprob"] == {
        "status": "not_applicable",
        "reason": "requires_separate_eager_paired_cohort",
    }
    assert aggregate["cutedsl_g0_effects"]["g0a0"]["policy_training"]["duration"][
        "median_percent"
    ] == pytest.approx((1.0 / 0.95 - 1.0) * 100.0)

    first = output.read_bytes()
    rerun = _run_collector(submission, result_root, output)
    assert rerun.returncode == 0, rerun.stderr
    assert output.read_bytes() == first


def test_collector_rejects_unverified_positive_a2a_temporal_overlap(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(
        tmp_path, temporal_overlap_verified=False
    )
    output = tmp_path / "aggregate.json"

    result = _run_collector(submission, result_root, output)

    assert result.returncode != 0
    assert (
        "temporal_overlap_verified is inconsistent with overlap evidence"
        in result.stderr
    )


@pytest.mark.parametrize("context", ("g0a0", "g1a1"))
def test_collector_keeps_missing_profile_temporal_analyzer_provisional(
    tmp_path: Path,
    context: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, context, 0)
    (result_root / job_id / "a2a_temporal_overlap.json").unlink()
    output = tmp_path / "aggregate.json"

    result = _run_collector(submission, result_root, output)

    assert result.returncode == 0, result.stderr
    aggregate = json.loads(output.read_text())
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["claim_ready"] is False
    assert aggregate["provisional_reasons"] == [
        DIRECT_PATH_LIMITATION,
        f"{context} job {job_id} lacks A2A temporal-overlap analysis",
    ]


def test_collector_rejects_a2a_analyzer_source_profile_digest_mismatch(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a1", 0)
    analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer["source_profile_sha256"] = "0" * 64
    _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "A2A source profile digest must match exactly one artifact" in result.stderr


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("a2a_overlap_ratio", 1.1, "a2a_overlap_ratio must be in (0, 1]"),
        ("temporal_overlap_verified", "yes", "must be boolean"),
    ],
)
def test_collector_rejects_malformed_a2a_temporal_analyzer(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a1", 0)
    analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer[field] = value
    _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_validates_optional_nonprofile_a2a_analyzer(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a1", 2)
    job_dir = result_root / job_id
    _write_temporal_analyzer(job_dir, job_id)
    analyzer_path = job_dir / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer["gemm_overlap_ratio"] = 0.0
    _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "gemm_overlap_ratio must be in (0, 1]" in result.stderr


@pytest.mark.parametrize(
    "artifact",
    ("job_directory", "manifest", "temporal_analyzer", "source_profile"),
)
def test_collector_rejects_internal_artifact_symlinks(
    tmp_path: Path,
    artifact: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a1", 0)
    job_dir = result_root / job_id
    if artifact == "job_directory":
        target = result_root / f"storage-{job_id}"
        job_dir.rename(target)
        job_dir.symlink_to(target, target_is_directory=True)
    elif artifact == "manifest":
        source = job_dir / "benchmark_manifest.json"
        target = result_root / _job_id(submission, "g1a1", 1) / "copied-manifest.json"
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)
    elif artifact == "temporal_analyzer":
        source = job_dir / "a2a_temporal_overlap.json"
        target = (
            result_root
            / _job_id(submission, "g1a1", 1)
            / "copied-temporal-analyzer.json"
        )
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)
    else:
        source = job_dir / "profiles/on.nsys-rep"
        target = (
            result_root / _job_id(submission, "g1a1", 1) / "profiles/copied.nsys-rep"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "must not contain symlinks" in result.stderr


def test_collector_requires_at_least_one_profile_replica_per_context(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = _submission_records(submission)
    profiled = next(
        record
        for record in records
        if record["factorial_context"] == "g1a1" and record["profile_enabled"]
    )
    profiled["profile_enabled"] = False
    _write_submission(submission, records)
    manifest_path = result_root / profiled["job_id"] / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile_enabled"] = False
    _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "context g1a1 requires at least one profile replicate" in result.stderr


def test_collector_allows_multiple_profile_replicas_per_a1_context(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = _submission_records(submission)
    extra = next(
        record
        for record in records
        if record["factorial_context"] == "g1a1" and record["replicate_index"] == 1
    )
    extra["profile_enabled"] = True
    _write_submission(submission, records)
    extra_job_dir = result_root / extra["job_id"]
    manifest_path = extra_job_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile_enabled"] = True
    _write_json(manifest_path, manifest)
    source_job_id = _job_id(submission, "g1a1", 0)
    source_attribution = json.loads(
        (result_root / source_job_id / "feature_attribution.json").read_text()
    )
    _write_json(extra_job_dir / "feature_attribution.json", source_attribution)
    _write_temporal_analyzer(extra_job_dir, extra["job_id"])

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["provisional_reasons"] == [DIRECT_PATH_LIMITATION]
    assert aggregate["context_replicate_counts"]["g1a1"] == 3


@pytest.mark.parametrize(
    ("remove", "expected_error"),
    [
        (("g1a1", 0), "context g1a1 requires at least 3 replicas"),
        (("all", 2), "context g0a0 requires at least 3 replicas"),
    ],
)
def test_collector_requires_four_contexts_with_three_paired_replicas(
    tmp_path: Path,
    remove: tuple[str, int],
    expected_error: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    context, replicate_index = remove
    records = [
        record
        for record in _submission_records(submission)
        if not (
            record["replicate_index"] == replicate_index
            and (context == "all" or record["factorial_context"] == context)
        )
    ]
    _write_submission(submission, records)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_rejects_unbalanced_context_submission_order(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = _submission_records(submission)
    records[0], records[1] = records[1], records[0]
    _write_submission(submission, records)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "context submission order for replica 0 is not balanced" in result.stderr


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("topology.expert_model_parallel_size", 8, "noncolocated EP4 contract"),
        ("workload.train_global_batch_size", 4, "train_global_batch_size must equal 8"),
        ("workload.train_micro_batch_size", 2, "train_micro_batch_size must equal 1"),
    ],
)
def test_collector_rejects_mismatched_factorial_contract(
    tmp_path: Path,
    field: str,
    value: int,
    expected_error: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a0", 1)
    path = result_root / job_id / "benchmark_manifest.json"
    manifest = json.loads(path.read_text())
    section, key = field.split(".", maxsplit=1)
    manifest[section][key] = value
    _write_json(path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    ("warmup_updates", "measured_updates", "expected_error"),
    [
        (4, MEASURED_UPDATES, "warmup_updates must be at least 5"),
        (WARMUP_UPDATES, 19, "measured_updates must be at least 20"),
    ],
)
def test_collector_rejects_short_performance_update_window(
    tmp_path: Path,
    warmup_updates: int,
    measured_updates: int,
    expected_error: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 1)
    manifest_path = result_root / job_id / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["warmup_updates"] = warmup_updates
    manifest["measured_updates"] = measured_updates
    manifest["total_updates"] = warmup_updates + measured_updates
    _write_json(manifest_path, manifest)
    for raw_path in (result_root / job_id / "timing").glob("*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        raw["warmup_updates"] = warmup_updates
        raw["measured_updates"] = measured_updates
        _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_requires_cuda_device_max_connections_32_string(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for arm_config in manifest["fixed_config_evidence"].values():
            arm_config["policy.megatron_cfg.env_vars.CUDA_DEVICE_MAX_CONNECTIONS"] = 32
        _write_json(manifest_path, manifest)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "CUDA_DEVICE_MAX_CONNECTIONS must equal string '32'" in result.stderr


def test_collector_rejects_raw_manifest_update_window_mismatch(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a1", 2)
    raw_path = next((result_root / job_id / "timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["warmup_updates"] += 1
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "raw update window differs" in result.stderr


def test_collector_rejects_non_four_policy_gpu_timing(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a1", 2)
    path = next((result_root / job_id / "timing").glob("*-on/raw_timing.json"))
    raw = json.loads(path.read_text())
    raw["training_gpu_count"] = 8
    _write_json(path, raw)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "policy training GPU count must equal 4" in result.stderr


def test_collector_rejects_hidden_full_cg_off_artifact(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a0", 0)
    job_dir = result_root / job_id
    on_path = next((job_dir / "timing").glob("*-on/raw_timing.json"))
    raw = json.loads(on_path.read_text())
    raw["arm"] = "off"
    raw["order_index"] = 1
    _write_json(job_dir / "timing/1-off/raw_timing.json", raw)
    _write_json(job_dir / "timing/1-off/metrics.json", {})

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "unexpected raw timing artifacts" in result.stderr


def test_collector_rejects_missing_or_tampered_metric_steps(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 1)
    arm_dir = next((result_root / job_id / "timing").glob("*-on"))
    metrics_path = arm_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["timing/train/total_step_time"][str(MEASURED_STEPS[0])] += 0.5
    _write_json(metrics_path, metrics)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "metrics.json differs from raw measured series" in result.stderr


def test_collector_rejects_canonical_metrics_aliased_to_same_source(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 0)
    job_dir = result_root / job_id
    manifest_path = job_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    duplicate_source = manifest["resolved_metric_names"]["on"][
        "timing/train/total_step_time"
    ]
    manifest["resolved_metric_names"]["on"]["timing/train/generation"] = (
        duplicate_source
    )
    _write_json(manifest_path, manifest)
    raw_path = next((job_dir / "timing").glob("*-on/raw_timing.json"))
    raw = json.loads(raw_path.read_text())
    raw["resolved_metric_names"]["timing/train/generation"] = duplicate_source
    _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "source names must be one-to-one" in result.stderr


def test_collector_rejects_cross_context_workload_drift(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a1", 2)
    arm_dir = next((result_root / job_id / "timing").glob("*-on"))
    raw_path = arm_dir / "raw_timing.json"
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0]["mean_prompt_length"] += 1
    raw["measured_component_series"]["train/mean_prompt_length"][0]["value"] += 1
    _write_json(raw_path, raw)
    metrics_path = arm_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["train/mean_prompt_length"][str(MEASURED_STEPS[0])] += 1
    _write_json(metrics_path, metrics)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "g0a0/g1a1 workload equivalence failed" in result.stderr


def test_collector_rejects_g0_off_cross_context_workload_drift(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for context, scale in (("g0a0", 0.994), ("g0a1", 1.006)):
        job_id = _job_id(submission, context, 1)
        arm_dir = next((result_root / job_id / "timing").glob("*-off"))
        raw_path = arm_dir / "raw_timing.json"
        raw = json.loads(raw_path.read_text())
        for row in raw["measured_step_workload"]:
            row["total_num_tokens"] *= scale
            row["global_valid_toks"] *= scale
            row["e2e_tokens_per_sec_per_gpu"] = (
                row["total_num_tokens"] / row["total_step_seconds"] / 4.0
            )
            row["generation_tokens_per_sec_per_gpu"] = (
                row["total_num_tokens"] / row["generation_seconds"] / 4.0
            )
            row["policy_training_tokens_per_sec_per_gpu"] = (
                row["total_num_tokens"] / row["policy_training_seconds"] / 4.0
            )
            row["refit_effective_tokens_per_sec_per_gpu"] = (
                row["total_num_tokens"] / row["refit_transfer_update_seconds"] / 4.0
            )
        raw["measured_component_series"] = {
            canonical_name: [
                {"step": row["step"], "value": row[field]}
                for row in raw["measured_step_workload"]
            ]
            for canonical_name, field in RAW_FIELD_BY_CANONICAL_METRIC.items()
        }
        _write_json(raw_path, raw)
        metrics_path = arm_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        for canonical_name, source_name in CANONICAL_METRICS.items():
            metrics[source_name] = {
                str(point["step"]): point["value"]
                for point in raw["measured_component_series"][canonical_name]
            }
        _write_json(metrics_path, metrics)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "g0a0/g0a1 OFF workload equivalence failed" in result.stderr


@pytest.mark.parametrize(
    ("metric", "final_value", "expected_error"),
    [
        ("train/full_cuda_graph_warmup_calls", 2, "warmup_calls must equal 3"),
        ("train/full_cuda_graph_capture_calls", 2, "capture_calls must equal 1"),
        ("train/full_cuda_graph_replay_calls", 1, "replay_calls must be at least 2"),
        ("train/full_cuda_graph_reset_calls", 1, "reset_calls must equal 0"),
    ],
)
def test_collector_rejects_invalid_full_cg_execution_evidence(
    tmp_path: Path,
    metric: str,
    final_value: int,
    expected_error: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a0", 2)
    path = next((result_root / job_id / "timing").glob("*-on/metrics.json"))
    metrics = json.loads(path.read_text())
    if metric == "train/full_cuda_graph_warmup_calls":
        metrics[metric] = {
            step: min(int(step), final_value) for step in metrics[metric]
        }
    elif metric == "train/full_cuda_graph_replay_calls":
        metrics[metric] = {
            step: min(value, final_value) for step, value in metrics[metric].items()
        }
    else:
        metrics[metric] = {step: final_value for step in metrics[metric]}
    _write_json(path, metrics)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize("failure", ("config", "kernel"))
def test_collector_rejects_missing_a2a_enabled_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a1", 0)
    if failure == "config":
        path = result_root / job_id / "benchmark_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["fixed_config_evidence"]["on"][
            "policy.megatron_cfg.high_priority_a2a_comm_stream"
        ] = False
        _write_json(path, manifest)
        expected_error = "A2A config"
    else:
        path = result_root / job_id / "feature_attribution.json"
        attribution = json.loads(path.read_text())
        attribution["counts"]["on"]["nccl_a2a_kernel"] = 0
        _write_json(path, attribution)
        expected_error = "lacks NCCL A2A kernel presence"

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_requires_a2a_kernel_presence_in_a0_profile(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 0)
    path = result_root / job_id / "feature_attribution.json"
    attribution = json.loads(path.read_text())
    attribution["counts"]["on"]["nccl_a2a_kernel"] = 0
    _write_json(path, attribution)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "lacks NCCL A2A kernel presence" in result.stderr


def test_collector_requires_cutedsl_profile_attribution_for_g0_effect(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 0)
    (result_root / job_id / "kernel_attribution.json").unlink()

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "CuTeDSL kernel attribution" in result.stderr


def test_collector_accepts_explicit_zero_logprob_series_as_not_applicable(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    _add_logprob_series(result_root, 0.0)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["factorial_effects"]["logprob"] == {
        "status": "not_applicable",
        "reason": "disabled_by_supported_full_cg_slice",
    }


def test_collector_rejects_positive_logprob_series_in_supported_full_cg_slice(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    _add_logprob_series(result_root, 0.01)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert "optional Logprob metrics must be exactly zero" in result.stderr


def test_collector_allows_unverified_a0_temporal_baseline(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for context in ("g0a0", "g1a0"):
        job_id = _job_id(submission, context, 0)
        analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
        analyzer = json.loads(analyzer_path.read_text())
        analyzer["temporal_overlap_verified"] = False
        analyzer["limitations"] = ["baseline overlap is observation-only"]
        _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["provisional_reasons"] == [DIRECT_PATH_LIMITATION]


def test_collector_rejects_verified_zero_overlap_baseline(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g0a0", 0)
    analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer["temporal_overlap_verified"] = True
    analyzer["limitations"] = []
    _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode != 0
    assert (
        "temporal_overlap_verified is inconsistent with overlap evidence"
        in result.stderr
    )


def test_collector_marks_representative_only_temporal_analysis_provisional(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    job_id = _job_id(submission, "g1a1", 0)
    analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer["limitations"] = [HARNESS_REPRESENTATIVE_LIMITATION]
    _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["provisional_reasons"] == [
        DIRECT_PATH_LIMITATION,
        f"g1a1 job {job_id} A2A analysis is representative-only; "
        "no all-rank aggregation",
    ]
    assert aggregate["a2a_overlap_ratio_contrasts"]["g1"]["claim_status"] == (
        "provisional"
    )


def test_collector_marks_nonincreasing_a2a_overlap_ratio_provisional(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for context in ("g0a0", "g0a1"):
        job_id = _job_id(submission, context, 0)
        analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
        analyzer = json.loads(analyzer_path.read_text())
        analyzer["overlap_duration_ns"] = 5000
        analyzer["a2a_overlap_ratio"] = 0.2
        analyzer["gemm_overlap_ratio"] = 0.1
        analyzer["temporal_overlap_verified"] = True
        analyzer["limitations"] = []
        _write_json(analyzer_path, analyzer)

    result = _run_collector(submission, result_root, tmp_path / "aggregate.json")

    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["claim_status"] == "provisional"
    assert aggregate["provisional_reasons"] == [
        DIRECT_PATH_LIMITATION,
        "g0 profile replica 0 A2A overlap ratio did not increase",
    ]
    contrast = aggregate["a2a_overlap_ratio_contrasts"]["g0"]
    assert contrast["claim_status"] == "provisional"
    assert contrast["all_pairs_increased"] is False
    assert contrast["median_absolute_increase"] == pytest.approx(0.0)
    assert aggregate["a2a_overlap_ratio_contrasts"]["g1"]["claim_status"] == (
        "claim_ready"
    )


def test_collector_canonical_digest_changes_with_validated_evidence(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    output = tmp_path / "aggregate.json"
    first = _run_collector(submission, result_root, output)
    assert first.returncode == 0, first.stderr
    first_digest = json.loads(output.read_text())["cohort_evidence_digest"]
    job_id = _job_id(submission, "g1a1", 0)
    analyzer_path = result_root / job_id / "a2a_temporal_overlap.json"
    analyzer = json.loads(analyzer_path.read_text())
    analyzer["a2a_overlap_ratio"] = 0.4
    _write_json(analyzer_path, analyzer)

    second = _run_collector(submission, result_root, output)

    assert second.returncode == 0, second.stderr
    second_digest = json.loads(output.read_text())["cohort_evidence_digest"]
    assert second_digest != first_digest
