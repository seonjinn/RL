# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json

import pytest

from research.qwen3_8b_forced_long_rollout.run_rollout import (
    load_manifest,
    sampling_kwargs,
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
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
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
