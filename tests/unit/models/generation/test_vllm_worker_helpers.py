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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.worker_utils import (
    configure_refit_runtime,
    refit_cache_loader_routes_enabled,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
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


def test_configure_refit_runtime_preserves_additional_config():
    kwargs = {"additional_config": {"existing": "value"}}

    configure_refit_runtime({"refit_cache_loader_routes": True}, kwargs)

    assert kwargs["additional_config"]["existing"] == "value"
    assert refit_cache_loader_routes_enabled(
        SimpleNamespace(additional_config=kwargs["additional_config"])
    )


def test_refit_cache_loader_routes_defaults_to_disabled():
    assert not refit_cache_loader_routes_enabled(SimpleNamespace())


def test_refit_cache_loader_routes_rejects_non_boolean():
    config = SimpleNamespace(
        additional_config={"nemo_rl_refit_cache_loader_routes": "yes"}
    )

    with pytest.raises(TypeError, match="must be a boolean"):
        refit_cache_loader_routes_enabled(config)
