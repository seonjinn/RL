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

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from nemo_rl.environments.nemo_gym import NemoGym, setup_nemo_gym_config
from nemo_rl.models.generation.vllm import vllm_worker_async
from nemo_rl.models.generation.vllm.utils import validate_openai_sampling_request


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _gym_result(*, token_ids: list[int], logprobs: list[float]) -> dict[str, Any]:
    return {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": token_ids,
                    "generation_log_probs": logprobs,
                }
            ]
        },
        "responses_create_params": {"input": []},
    }


def _local_nemo_gym() -> Any:
    nemo_gym_cls = NemoGym.__ray_metadata__.modified_class
    nemo_gym = nemo_gym_cls.__new__(nemo_gym_cls)
    nemo_gym.cfg = {}
    return nemo_gym


def test_postprocess_rejects_generation_token_logprob_length_mismatch() -> None:
    nemo_gym = _local_nemo_gym()

    with pytest.raises(ValueError, match="token/logprob length mismatch"):
        nemo_gym._postprocess_nemo_gym_to_nemo_rl_result(
            _gym_result(token_ids=[3, 4], logprobs=[-0.25]),
            _Tokenizer(),
        )


def test_run_rollouts_rejects_nonfinite_logprobs_after_retry_exhaustion() -> None:
    nemo_gym = _local_nemo_gym()
    nemo_gym.rollout_max_attempts_to_avoid_lp_nan = 2
    nemo_gym.head_server_config = object()
    result = _gym_result(token_ids=[3], logprobs=[float("nan")])

    class _RolloutCollectionHelper:
        def __init__(self) -> None:
            self.calls = 0

        def run_examples(self, **_kwargs: Any) -> list[Any]:
            self.calls += 1

            async def _result() -> tuple[dict[str, int], dict[str, Any]]:
                return {"_rowidx": 0}, deepcopy(result)

            return [_result()]

    nemo_gym.rch = _RolloutCollectionHelper()

    with pytest.raises(RuntimeError, match="non-finite generation logprobs"):
        asyncio.run(nemo_gym.run_rollouts([{"_rowidx": 0}], _Tokenizer(), "test"))

    assert nemo_gym.rch.calls == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
        ("min_p", 0.1),
        ("logit_bias", {"3": 1.0}),
        ("allowed_token_ids", [3, 4]),
        ("bad_words", ["bad"]),
        ("use_beam_search", True),
        ("min_tokens", 1),
        ("thinking_token_budget", 128),
        ("structured_outputs", {"regex": "answer"}),
        ("response_format", {"type": "json_object"}),
        ("stop", ["END"]),
        ("stop_token_ids", [42]),
        ("include_stop_str_in_output", True),
        ("ignore_eos", True),
        ("repetition_detection", {"max_pattern_size": 4}),
        ("vllm_xargs", {"custom": 1}),
        ("truncate_prompt_tokens", 128),
        ("truncation_side", "left"),
        ("add_generation_prompt", False),
        ("continue_final_message", True),
        ("add_special_tokens", True),
        ("chat_template", "{{ messages }}"),
        ("documents", [{"title": "t", "text": "x"}]),
        ("media_io_kwargs", {"image": {"num_crops": 1}}),
        ("mm_processor_kwargs", {"num_crops": 1}),
    ],
)
def test_http_sampling_contract_rejects_unmodeled_distribution_modifiers(
    field: str, value: Any
) -> None:
    class _Request:
        top_k = None
        temperature = 1.0
        top_p = 0.9
        presence_penalty = 0.0
        frequency_penalty = 0.0
        repetition_penalty = None
        min_p = None
        logit_bias = None
        allowed_token_ids = None
        bad_words: list[str] = []
        use_beam_search = False

    request = _Request()
    setattr(request, field, value)

    with pytest.raises(ValueError, match=field):
            validate_openai_sampling_request(
                request,
                {"temperature": 1.0, "top_p": 0.9, "top_k": None},
            )


def test_http_sampling_contract_error_is_an_invalid_request_response() -> None:
    response = vllm_worker_async._openai_invalid_request_response(
        ValueError("presence_penalty changes the rollout distribution")
    )

    assert response.status_code == 400
    assert b'"type":"invalid_request_error"' in response.body
    assert b"presence_penalty changes the rollout distribution" in response.body


def test_http_sampling_contract_applies_configured_top_k() -> None:
    request = mock_request = type(
        "_Request",
        (),
        {"top_k": None, "temperature": 1.0, "top_p": 0.9},
    )()

    validate_openai_sampling_request(
        request,
        {"temperature": 1.0, "top_p": 0.9, "top_k": 50},
    )

    assert mock_request.top_k == 50


@pytest.mark.parametrize(
    ("field", "value"),
    [("stop_strings", ["END"]), ("stop_token_ids", [42])],
)
def test_nemo_gym_setup_rejects_configured_stop_criteria(
    field: str, value: Any
) -> None:
    class _Config:
        policy = {
            "generation": {
                "vllm_cfg": {},
                "stop_strings": None,
                "stop_token_ids": None,
            }
        }

    config = _Config()
    config.policy["generation"][field] = value

    with pytest.raises(ValueError, match=field):
        setup_nemo_gym_config(config, _Tokenizer())


def test_nemo_gym_setup_accepts_auto_inserted_eos_stop_token() -> None:
    class _Config:
        policy = {
            "generation": {
                "vllm_cfg": {},
                "stop_strings": None,
                "stop_token_ids": [_Tokenizer.eos_token_id],
            }
        }

    config = _Config()

    setup_nemo_gym_config(config, _Tokenizer())

    assert config.policy["generation"]["stop_token_ids"] is None
