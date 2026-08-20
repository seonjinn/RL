# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json
from pathlib import Path

import pytest

from research.qwen3_8b_forced_long_rollout.run_rollout import (
    load_manifest,
    required_decode_capture_sizes,
    sampling_kwargs,
    validate_runtime_versions,
    validate_prompt_lengths,
)


def test_manifest_requires_the_pinned_hash_and_exact_order(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    records = [
        {
            "logical_step": index + 1,
            "source_id": f"source-{index}",
            "prompt_sha256": str(index) * 64,
        }
        for index in range(5)
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_repository": "BytedTsinghua-SIA/DAPO-Math-17k",  # pragma: allowlist secret
                "dataset_revision": "revision",
                "num_responses": 5,
                "selection": "first-1000-source-order",
                "records": records,
            }
        )
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert load_manifest(manifest_path, expected_sha256=digest, count=5) == records

    with pytest.raises(ValueError, match="SHA256"):
        load_manifest(manifest_path, expected_sha256="0" * 64, count=5)


def test_forced_long_sampling_allows_eos_only_after_minimum() -> None:
    assert sampling_kwargs(min_tokens=28672, max_tokens=32768) == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_tokens": 28672,
        "max_tokens": 32768,
        "ignore_eos": False,
    }


def test_prompt_preflight_rejects_configured_and_context_overflow() -> None:
    validate_prompt_lengths([2048], max_input_tokens=2048, max_output_tokens=32768)

    with pytest.raises(ValueError, match="configured input limit"):
        validate_prompt_lengths([2049], max_input_tokens=2048, max_output_tokens=32768)

    with pytest.raises(ValueError, match="model context"):
        validate_prompt_lengths([8193], max_input_tokens=8193, max_output_tokens=32768)


def test_capture_sizes_cover_k5_and_k7_decode_shapes_for_eight_sequences() -> None:
    assert required_decode_capture_sizes(max_num_seqs=8, speculative_tokens=(5, 7)) == [
        1,
        2,
        4,
        6,
        8,
        12,
        16,
        18,
        24,
        30,
        32,
        36,
        40,
        42,
        48,
        56,
        64,
    ]


def test_runtime_requires_the_pinned_vllm_contract() -> None:
    assert validate_runtime_versions({"vllm": "0.25.1"}, expected_vllm="0.25.1") == {
        "vllm": "0.25.1"
    }

    with pytest.raises(RuntimeError, match="vLLM runtime mismatch"):
        validate_runtime_versions({"vllm": "0.27.1"}, expected_vllm="0.25.1")


def test_runtime_is_checked_inside_the_container_before_worker_launch() -> None:
    launcher = (Path(__file__).parents[1] / "run_oci_hsg.sbatch").read_text()
    sync_start = launcher.index("  bash -lc '\n")
    sync_end = launcher.index("\n  '\n", sync_start)
    worker_start = launcher.index("\nsrun \\\n", sync_end)
    sync_shell = launcher[sync_start:sync_end]
    required_order = [
        "unset VIRTUAL_ENV",
        "export UV_PROJECT_ENVIRONMENT=",
        "uv sync",
        'test -x "${UV_PROJECT_ENVIRONMENT}/bin/python"',
        '"${UV_PROJECT_ENVIRONMENT}/bin/python" -c',
    ]

    assert [sync_shell.index(fragment) for fragment in required_order] == sorted(
        sync_shell.index(fragment) for fragment in required_order
    )
    assert "test -x" not in launcher[sync_end:worker_start]
