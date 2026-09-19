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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration


@pytest.mark.parametrize("needs_offload, engines", [(False, []), (True, [None])])
def test_finish_generation_noop_returns_literal_true(
    needs_offload: bool, engines: list[object | None]
) -> None:
    generation = SGLangGeneration.__new__(SGLangGeneration)
    generation.weight_synchronizer = None
    generation._async_loop = None
    generation._http_client = None
    generation._router_actor = None
    generation.num_gpus_per_engine = 1
    generation.num_gpus_per_node = 1
    generation.all_engines = engines
    generation.needs_offload = needs_offload

    assert generation.finish_generation() is True


def test_finish_generation_success_returns_literal_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = MagicMock(return_value=object())
    engine = SimpleNamespace(
        release_memory_occupation=SimpleNamespace(remote=release)
    )
    generation = SGLangGeneration.__new__(SGLangGeneration)
    generation.weight_synchronizer = None
    generation._async_loop = None
    generation._http_client = None
    generation._router_actor = None
    generation.num_gpus_per_engine = 1
    generation.num_gpus_per_node = 1
    generation.all_engines = [engine]
    generation.needs_offload = True
    ray_get = MagicMock(return_value=[None])
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get", ray_get
    )

    assert generation.finish_generation(tags=["weights"]) is True
    release.assert_called_once_with(tags=["weights"])
    ray_get.assert_called_once_with([release.return_value])
    generation.all_engines = []
