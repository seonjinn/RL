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

"""Tests for vLLM worker helper functions."""

import pytest

from nemo_rl.models.generation.vllm.worker_utils import (
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
    validate_mxfp8_precision,
)


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )


@pytest.mark.parametrize("precision", [None, "auto", "bf16", "bfloat16"])
def test_validate_mxfp8_precision_rejects_non_fp8_precision(precision):
    with pytest.raises(ValueError, match="is_mx=True requires precision='fp8'"):
        validate_mxfp8_precision({"precision": precision, "is_mx": True})


@pytest.mark.parametrize(
    "vllm_cfg",
    [
        {"precision": "fp8", "is_mx": True},
        {"precision": "bfloat16", "is_mx": False},
        {"precision": "bfloat16"},
    ],
)
def test_validate_mxfp8_precision_accepts_supported_configs(vllm_cfg):
    validate_mxfp8_precision(vllm_cfg)
