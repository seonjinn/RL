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

"""Unit tests for ``nemo_rl/utils/multimodal_payload_metrics.py``.

These tests pin the metric output that the Gate 9 dedup-verification
methodology relies on. Specifically they verify that for a given logical
batch:

* with ``deduplicate=True`` the unique-prompt count and tensor_mm bytes
  shrink in lockstep with how many duplicate prompts share an underlying
  tensor (the ``payload_bytes/<boundary>/tensor_mm`` term scales with the
  number of *unique* underlying tensors, not with the logical row count),
* logical_rows stays equal between dedup-on and dedup-off (the consumer
  still sees one row per prompt),
* ``infer_unique_prompt_count`` reads ``_dedup_prompt_idx`` when present
  and falls back to ``idx`` and finally ``default_rows``,
* the new boundary names introduced in
  ``nemo_rl/algorithms/grpo.py`` produce the expected metric keys.
"""

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.utils.multimodal_payload_metrics import (
    collect_multimodal_payload_metrics,
    infer_unique_prompt_count,
)


def _data_with_repeated_pixels(rollouts_per_prompt: int) -> dict:
    """Build a dict that simulates one image shared across rollouts."""
    img_tensor = torch.zeros(3, 16, 16, dtype=torch.bfloat16)
    pv = PackedTensor(
        [img_tensor.clone() for _ in range(rollouts_per_prompt)],
        dim_to_pack=0,
    )
    return {
        "input_ids": torch.zeros(rollouts_per_prompt, 8, dtype=torch.long),
        "input_lengths": torch.full((rollouts_per_prompt,), 8, dtype=torch.long),
        "pixel_values": pv,
        "_dedup_prompt_idx": torch.zeros(rollouts_per_prompt, dtype=torch.long),
    }


def test_collect_metrics_dedup_on_vs_off_same_logical_rows_smaller_bytes():
    """The core dedup-validation invariant.

    Two BatchedDataDicts with the same logical rows (4) and the same
    underlying image content (1 unique image shared across 4 rollouts):
    one represents the dedup-off layout (4 separate copies of the
    underlying tensor) and the other the dedup-on layout (1 unique tensor
    addressed by 4 logical indices). The metric helper must report:

    * identical ``payload_counts/.../logical_rows`` (4 in both cases),
    * identical ``payload_counts/.../unique_prompts`` (1 in both cases,
      since both runs carry the same ``_dedup_prompt_idx``),
    * ``payload_bytes/.../tensor_mm`` for the deduped variant == 1/4 of
      the dedup-off variant (1 unique tensor vs 4 logical copies, same
      element size).
    """
    rollouts_per_prompt = 4

    off = _data_with_repeated_pixels(rollouts_per_prompt)
    on = _data_with_repeated_pixels(rollouts_per_prompt)
    on["pixel_values"] = on["pixel_values"].deduplicate(
        on["_dedup_prompt_idx"]
    )

    m_off = collect_multimodal_payload_metrics(off, boundary="b")
    m_on = collect_multimodal_payload_metrics(on, boundary="b")

    assert m_off["payload_counts/b/logical_rows"] == rollouts_per_prompt
    assert m_on["payload_counts/b/logical_rows"] == rollouts_per_prompt
    assert m_off["payload_counts/b/unique_prompts"] == 1
    assert m_on["payload_counts/b/unique_prompts"] == 1

    assert m_on["payload_bytes/b/tensor_mm"] * rollouts_per_prompt == (
        m_off["payload_bytes/b/tensor_mm"]
    )
    assert m_on["payload_counts/b/unique_mm_items"] == 1
    assert (
        m_off["payload_counts/b/unique_mm_items"] == rollouts_per_prompt
    )


def test_infer_unique_prompt_count_prefers_dedup_idx_then_idx_then_default():
    """``infer_unique_prompt_count`` resolution order matches the methodology."""
    base = {
        "input_ids": torch.zeros(4, 8, dtype=torch.long),
    }
    assert infer_unique_prompt_count(base, default_rows=4) == 4

    base["idx"] = torch.tensor([10, 10, 11, 11])
    assert infer_unique_prompt_count(base, default_rows=4) == 2

    base["_dedup_prompt_idx"] = torch.tensor([0, 0, 0, 0])
    assert infer_unique_prompt_count(base, default_rows=4) == 1

    base["_dedup_prompt_idx"] = [7, 7, 8]
    assert infer_unique_prompt_count(base, default_rows=3) == 2


def test_boundary_names_used_in_grpo_produce_expected_keys():
    """Pin metric keys for each boundary the sync-rollout path emits.

    These keys are the ones a downstream parser (or the Gate 9
    dedup-validation script) will grep for; locking them down here
    catches accidental renames.
    """
    boundaries = (
        "driver_to_vllm_generation",
        "driver_to_policy_get_logprobs",
        "driver_to_policy_get_reference_policy_logprobs",
        "driver_to_policy_train",
        "driver_to_policy_calibrate_qkv_fp8_scales_post_train",
        "driver_to_policy_calibrate_qkv_fp8_scales_pre_refit",
    )
    payload = _data_with_repeated_pixels(2)
    for boundary in boundaries:
        m = collect_multimodal_payload_metrics(payload, boundary=boundary)
        for suffix in (
            "tensor_mm",
            "non_tensor_mm",
            "total_mm",
        ):
            assert f"payload_bytes/{boundary}/{suffix}" in m
        for suffix in (
            "logical_rows",
            "unique_prompts",
            "unique_mm_items",
        ):
            assert f"payload_counts/{boundary}/{suffix}" in m
        assert f"payload_ratio/{boundary}/logical_to_unique" in m


def test_collect_metrics_with_explicit_unique_prompts_overrides_inference():
    """When the caller supplies ``unique_prompts``, it overrides inference.

    The grpo sync path passes ``unique_prompts_for_policy`` (computed
    once on the source repeated_batch) to every downstream boundary so
    that, even after fields like ``_dedup_prompt_idx`` are dropped from
    ``logprob_data``, all four boundaries still report the *same* unique
    count for parity comparison.
    """
    payload = _data_with_repeated_pixels(4)
    payload.pop("_dedup_prompt_idx", None)
    m = collect_multimodal_payload_metrics(
        payload, boundary="b", unique_prompts=1
    )
    assert m["payload_counts/b/unique_prompts"] == 1
    assert m["payload_counts/b/logical_rows"] == 4
