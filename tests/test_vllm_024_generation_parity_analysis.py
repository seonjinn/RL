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

import pytest

from experiments.vllm_024_upgrade.analyze_generation_parity import (
    analyze_parity_rows,
    validate_metadata_contract,
)


def _row(
    prompt_id: str,
    sample_id: str,
    token_ids: list[int],
    *,
    truncated: bool = False,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "sample_id": sample_id,
        "token_ids": token_ids,
        "token_logprobs": [-0.25] * len(token_ids),
        "truncated": truncated,
    }


def test_greedy_gate_detects_a_later_token_mismatch() -> None:
    baseline = [_row("math-0", "0000", [1, 2, 3])]
    candidate = [_row("math-0", "0000", [1, 2, 4])]

    report = analyze_parity_rows(baseline, candidate, mode="greedy")

    assert report["status"] == "failed"
    assert report["checks"]["exact_sequence_match"]["passed"] is False


def test_sampled_gate_accepts_equal_unpaired_sequence_distributions() -> None:
    baseline = [
        _row("math-0", f"b-{index}", [1, 2 if index % 2 == 0 else 3])
        for index in range(40)
    ]
    candidate = [
        _row("math-0", f"c-{index}", [1, 3 if index % 2 == 0 else 2])
        for index in range(40)
    ]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=199,
        min_samples_per_prompt=32,
        seed=7,
    )

    assert report["status"] == "passed"
    assert report["checks"]["sequence_distribution"]["passed"] is True


def test_sampled_gate_rejects_material_selected_logprob_shift() -> None:
    baseline = [
        _row("math-0", f"b-{index}", [1, 2 if index % 2 == 0 else 3])
        for index in range(64)
    ]
    candidate = [
        _row("math-0", f"c-{index}", [1, 3 if index % 2 == 0 else 2])
        for index in range(64)
    ]
    for row in baseline:
        row["token_logprobs"] = [-0.1, -0.1]
    for row in candidate:
        row["token_logprobs"] = [-20.0, -20.0]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=199,
        min_samples_per_prompt=32,
        seed=23,
    )

    assert report["status"] == "failed"
    logprob_check = report["checks"]["selected_token_logprob_equivalence"]
    assert logprob_check["passed"] is False
    assert logprob_check["detected_shift"] is True
    assert logprob_check["p_value"] < logprob_check["alpha"]
    assert logprob_check["max_absolute_mean_delta"] == pytest.approx(19.9)


def test_sampled_gate_detects_a_later_token_distribution_shift() -> None:
    baseline = [
        _row("math-0", f"b-{index}", [1, 2 if index % 2 == 0 else 3])
        for index in range(64)
    ]
    candidate = [_row("math-0", f"c-{index}", [1, 4]) for index in range(64)]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=199,
        min_samples_per_prompt=32,
        seed=11,
    )

    assert report["status"] == "failed"
    sequence_check = report["checks"]["sequence_distribution"]
    assert sequence_check["passed"] is False
    assert sequence_check["p_value"] < 0.01


def test_sampled_gate_detects_a_shift_beyond_the_prefix_window() -> None:
    baseline = [_row("math-0", f"b-{index}", [1] * 64 + [2]) for index in range(64)]
    candidate = [_row("math-0", f"c-{index}", [1] * 64 + [3]) for index in range(64)]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=199,
        min_samples_per_prompt=32,
        max_positions=64,
        seed=19,
    )

    assert report["status"] == "failed"
    assert report["checks"]["sequence_distribution"]["detected_shift"] is True


def test_sampled_gate_detects_a_termination_rate_shift() -> None:
    baseline = [
        _row("math-0", f"b-{index}", [1, 2], truncated=False) for index in range(64)
    ]
    candidate = [
        _row("math-0", f"c-{index}", [1, 2], truncated=True) for index in range(64)
    ]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=99,
        min_samples_per_prompt=32,
        seed=13,
    )

    assert report["status"] == "failed"
    assert report["checks"]["truncation_rate_equivalence"]["passed"] is False


def test_sampled_gate_does_not_treat_failure_to_reject_as_equivalence() -> None:
    baseline = [
        _row("math-0", f"b-{index}", [1, 2 if index < 32 else 3]) for index in range(64)
    ]
    candidate = [
        _row("math-0", f"c-{index}", [1, 2 if index < 40 else 3]) for index in range(64)
    ]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=199,
        min_samples_per_prompt=32,
        sequence_mmd_margin=0.01,
        seed=17,
    )

    sequence_check = report["checks"]["sequence_distribution"]
    assert sequence_check["p_value"] >= 0.01
    assert sequence_check["equivalent"] is False
    assert report["status"] == "inconclusive"


def test_length_equivalence_uses_each_prompts_own_baseline_length() -> None:
    baseline = [
        *[_row("short", f"b-short-{index}", [1, 2]) for index in range(32)],
        *[_row("long", f"b-long-{index}", list(range(100))) for index in range(32)],
    ]
    candidate = [
        *[_row("short", f"c-short-{index}", [1, 2, 3]) for index in range(32)],
        *[_row("long", f"c-long-{index}", list(range(100))) for index in range(32)],
    ]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=99,
        min_samples_per_prompt=32,
    )

    assert report["checks"]["length_equivalence"]["passed"] is False


def test_sampled_gate_is_inconclusive_when_prompt_cohorts_are_too_small() -> None:
    baseline = [_row("math-0", "b-0", [1, 2])]
    candidate = [_row("math-0", "c-0", [1, 2])]

    report = analyze_parity_rows(
        baseline,
        candidate,
        mode="sampled",
        permutations=99,
        min_samples_per_prompt=32,
    )

    assert report["status"] == "inconclusive"


def test_gate_rejects_malformed_behavior_logprobs() -> None:
    malformed = [_row("math-0", "0000", [1, 2])]
    malformed[0]["token_logprobs"] = [-0.25]

    with pytest.raises(ValueError, match="token/logprob length mismatch"):
        analyze_parity_rows(malformed, malformed, mode="greedy")


def _metadata(*, draft_model: str | None = None) -> dict:
    return {
        "status": "passed",
        "git_commit": "abc123",
        "mode": "sampled",
        "prompt_count": 32,
        "samples_per_prompt": 64,
        "requested_samples": 2048,
        "batch_size": 8,
        "settings": {
            "model": "/models/qwen32",
            "tokenizer": "/models/qwen32",
            "draft_model": draft_model,
            "method": "eagle3",
            "num_speculative_tokens": 5,
            "target_tp": 2,
            "draft_tp": 1,
            "max_model_len": 4096,
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "num_nodes": 1,
            "gpus_per_node": 2,
            "gpu_memory_utilization": 0.8,
            "draft_sample_method": "greedy",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 128,
        },
    }


def test_metadata_contract_allows_only_specdec_specific_differences() -> None:
    baseline = _metadata()
    candidate = _metadata(draft_model="/models/qwen32-eagle3")

    validate_metadata_contract(baseline, candidate, expected_mode="sampled")


def test_metadata_contract_rejects_sampling_or_runtime_mismatch() -> None:
    baseline = _metadata()
    candidate = _metadata(draft_model="/models/qwen32-eagle3")
    candidate["settings"]["temperature"] = 0.0

    with pytest.raises(ValueError, match="temperature"):
        validate_metadata_contract(baseline, candidate, expected_mode="sampled")


def test_metadata_contract_rejects_failed_or_unclean_runs() -> None:
    baseline = _metadata()
    candidate = _metadata(draft_model="/models/qwen32-eagle3")
    candidate["cleanup_errors"] = ["worker shutdown failed"]

    with pytest.raises(ValueError, match="cleanup_errors"):
        validate_metadata_contract(baseline, candidate, expected_mode="sampled")
