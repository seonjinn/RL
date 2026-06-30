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

import importlib.util
from pathlib import Path

import pytest

SLEEP_PATH = Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/sleep.py"
SLEEP_SPEC = importlib.util.spec_from_file_location(
    "nemo_rl_vllm_sleep_under_test", SLEEP_PATH
)
assert SLEEP_SPEC is not None
SLEEP_MODULE = importlib.util.module_from_spec(SLEEP_SPEC)
assert SLEEP_SPEC.loader is not None
SLEEP_SPEC.loader.exec_module(SLEEP_MODULE)

get_vllm_sleep_level = SLEEP_MODULE.get_vllm_sleep_level
validate_vllm_sleep_level = SLEEP_MODULE.validate_vllm_sleep_level


def test_get_vllm_sleep_level_defaults_to_existing_behavior(monkeypatch):
    monkeypatch.delenv("NEMO_RL_VLLM_SLEEP_LEVEL", raising=False)

    assert get_vllm_sleep_level() == 1


def test_get_vllm_sleep_level_reads_environment(monkeypatch):
    monkeypatch.setenv("NEMO_RL_VLLM_SLEEP_LEVEL", "2")

    assert get_vllm_sleep_level() == 2


def test_get_vllm_sleep_level_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("NEMO_RL_VLLM_SLEEP_LEVEL", "not-an-int")

    with pytest.raises(ValueError, match="must be an integer"):
        get_vllm_sleep_level()


def test_get_vllm_sleep_level_rejects_non_positive_values(monkeypatch):
    monkeypatch.setenv("NEMO_RL_VLLM_SLEEP_LEVEL", "0")

    with pytest.raises(ValueError, match="must be one of"):
        get_vllm_sleep_level()


def test_get_vllm_sleep_level_rejects_unsupported_values(monkeypatch):
    monkeypatch.setenv("NEMO_RL_VLLM_SLEEP_LEVEL", "3")

    with pytest.raises(ValueError, match="must be one of"):
        get_vllm_sleep_level()


def test_validate_vllm_sleep_level_accepts_supported_values():
    assert validate_vllm_sleep_level(1) == 1
    assert validate_vllm_sleep_level(2) == 2
