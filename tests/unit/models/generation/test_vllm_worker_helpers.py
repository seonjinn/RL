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

from typing import cast
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_worker import (
    VllmGenerationWorkerImpl,
    _refit_sleep_level,
)
from nemo_rl.models.generation.vllm.worker_utils import (
    find_tokenizer_required_architectures,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


def _refit_test_config(mode: str | None = None) -> VllmConfig:
    config: dict = {"vllm_cfg": {"async_engine": False}}
    if mode is not None:
        config["refit_cfg"] = {"memory_lifecycle": {"mode": mode}}
    return cast(VllmConfig, config)


def _sleep_test_worker(
    *, uses_specdec_deep_refit: bool
) -> tuple[VllmGenerationWorkerImpl, MagicMock]:
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.cfg = _refit_test_config(
        "specdec_deep_refit" if uses_specdec_deep_refit else None
    )
    worker.uses_specdec_deep_refit = uses_specdec_deep_refit  # type: ignore[attr-defined]
    fake_llm = MagicMock()
    worker.llm = fake_llm
    return worker, fake_llm


def test_refit_sleep_level_defaults_to_level_one() -> None:
    assert _refit_sleep_level(_refit_test_config()) == 1


def test_refit_sleep_level_selects_level_two_only_when_explicit() -> None:
    assert _refit_sleep_level(_refit_test_config("specdec_deep_refit")) == 2


def test_legacy_worker_sleep_remains_level_one_without_rpc() -> None:
    worker, fake_llm = _sleep_test_worker(uses_specdec_deep_refit=False)

    worker.sleep()

    fake_llm.sleep.assert_called_once_with(level=1)
    fake_llm.collective_rpc.assert_not_called()


def test_deep_refit_worker_sleep_selects_level_two() -> None:
    worker, fake_llm = _sleep_test_worker(uses_specdec_deep_refit=True)

    worker.sleep()

    fake_llm.sleep.assert_called_once_with(level=2)


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (None, []),
        ([], []),
        (["Gemma4ForCausalLM"], []),
        (
            ["Gemma4ForConditionalGeneration"],
            ["Gemma4ForConditionalGeneration"],
        ),
        (
            [
                "Gemma4ForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
            [
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
        ),
    ],
)
def test_find_tokenizer_required_architectures(architectures, expected):
    assert find_tokenizer_required_architectures(architectures) == expected


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
