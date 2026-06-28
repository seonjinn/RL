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

import os

_SUPPORTED_VLLM_SLEEP_LEVELS = (1, 2)


def validate_vllm_sleep_level(sleep_level: int) -> int:
    if sleep_level not in _SUPPORTED_VLLM_SLEEP_LEVELS:
        raise ValueError(
            "NEMO_RL_VLLM_SLEEP_LEVEL must be one of the sleep levels supported "
            f"by vLLM: {_SUPPORTED_VLLM_SLEEP_LEVELS}"
        )
    return sleep_level


def get_vllm_sleep_level() -> int:
    raw_level = os.environ.get("NEMO_RL_VLLM_SLEEP_LEVEL", "1")
    try:
        sleep_level = int(raw_level)
    except ValueError as exc:
        raise ValueError(
            "NEMO_RL_VLLM_SLEEP_LEVEL must be an integer sleep level supported by vLLM"
        ) from exc
    return validate_vllm_sleep_level(sleep_level)
