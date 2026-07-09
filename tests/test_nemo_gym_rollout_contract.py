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

    async def _get_response_json(response: _Response) -> dict[str, Any]:
        return response.payload

    async def _raise_for_status(_response: _Response) -> None:
        return None

    nemo_gym._get_response_json = _get_response_json
    nemo_gym._is_request_debug_enabled = lambda: False
    nemo_gym._raise_for_status = _raise_for_status
    return nemo_gym


class _Response:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class _RolloutCollectionHelper:
    """Exercise the same proxy-task ownership boundary as Gym.run_examples."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def setup_server_client(self, _head_server_config: Any) -> Any:
        return self.client

    def run_examples(self, *, examples: list[dict[str, Any]], **_kwargs: Any) -> Any:
        async def _run_one(
            row: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            response = await self.client.post(
                server_name=row["agent_ref"]["name"],
                url_path="/run",
                json=row,
            )
            return row, response.payload

        return asyncio.as_completed([_run_one(row) for row in examples])


def _rollout_examples(count: int) -> list[dict[str, Any]]:
    return [
        {"_rowidx": rowidx, "agent_ref": {"name": f"agent-{rowidx}"}}
        for rowidx in range(count)
    ]


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

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, **_kwargs: Any) -> _Response:
            self.calls += 1
            return _Response(deepcopy(result))

    client = _Client()
    nemo_gym.rch = _RolloutCollectionHelper(client)

    with pytest.raises(RuntimeError, match="non-finite generation logprobs"):
        asyncio.run(nemo_gym.run_rollouts(_rollout_examples(1), _Tokenizer(), "test"))

    assert client.calls == 2


def test_run_rollouts_cancels_and_awaits_siblings_when_one_fails() -> None:
    async def _scenario() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        sibling_joined = asyncio.Event()
        release_sibling = asyncio.Event()

        class _Client:
            async def post(self, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
                if json["_rowidx"] == 0:
                    await sibling_started.wait()
                    raise RuntimeError("rollout failed")

                sibling_started.set()
                try:
                    await release_sibling.wait()
                except asyncio.CancelledError:
                    sibling_cancelled.set()
                    await asyncio.sleep(0)
                    sibling_joined.set()
                    raise
                return _Response(_gym_result(token_ids=[4], logprobs=[-0.2]))

        nemo_gym = _local_nemo_gym()
        nemo_gym.rollout_max_attempts_to_avoid_lp_nan = 1
        nemo_gym.head_server_config = object()
        nemo_gym.rch = _RolloutCollectionHelper(_Client())

        try:
            with pytest.raises(RuntimeError, match="rollout failed"):
                await nemo_gym.run_rollouts(_rollout_examples(2), _Tokenizer(), "test")

            assert sibling_cancelled.is_set()
            assert sibling_joined.is_set()
        finally:
            release_sibling.set()
            await asyncio.sleep(0)

    asyncio.run(_scenario())


def test_run_rollouts_cancels_and_awaits_siblings_when_parent_is_cancelled() -> None:
    async def _scenario() -> None:
        all_started = asyncio.Event()
        all_cancelled = asyncio.Event()
        all_joined = asyncio.Event()
        release_siblings = asyncio.Event()
        started_count = 0
        cancelled_count = 0
        joined_count = 0

        class _Client:
            async def post(self, **_kwargs: Any) -> _Response:
                nonlocal started_count, cancelled_count, joined_count
                started_count += 1
                if started_count == 2:
                    all_started.set()
                try:
                    await release_siblings.wait()
                except asyncio.CancelledError:
                    cancelled_count += 1
                    if cancelled_count == 2:
                        all_cancelled.set()
                    await asyncio.sleep(0)
                    joined_count += 1
                    if joined_count == 2:
                        all_joined.set()
                    raise
                return _Response(_gym_result(token_ids=[4], logprobs=[-0.2]))

        nemo_gym = _local_nemo_gym()
        nemo_gym.rollout_max_attempts_to_avoid_lp_nan = 1
        nemo_gym.head_server_config = object()
        nemo_gym.rch = _RolloutCollectionHelper(_Client())
        parent = asyncio.create_task(
            nemo_gym.run_rollouts(_rollout_examples(2), _Tokenizer(), "test")
        )

        try:
            await all_started.wait()
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent

            assert all_cancelled.is_set()
            assert all_joined.is_set()
        finally:
            release_siblings.set()
            await asyncio.sleep(0)

    asyncio.run(_scenario())


def test_run_rollouts_preserves_input_order_and_timing_metrics() -> None:
    async def _scenario() -> None:
        class _Client:
            async def post(self, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
                await asyncio.sleep(0.01 if json["_rowidx"] == 0 else 0)
                return _Response(
                    _gym_result(
                        token_ids=[json["_rowidx"] + 3],
                        logprobs=[-0.1],
                    )
                )

        nemo_gym = _local_nemo_gym()
        nemo_gym.rollout_max_attempts_to_avoid_lp_nan = 1
        nemo_gym.head_server_config = object()
        nemo_gym.rch = _RolloutCollectionHelper(_Client())

        results, timing_metrics = await nemo_gym.run_rollouts(
            _rollout_examples(2), _Tokenizer(), "test"
        )

        assert [result["message_log"][1]["token_ids"].item() for result in results] == [
            3,
            4,
        ]
        assert timing_metrics["test/await_results"] >= 0
        assert timing_metrics["test/postprocess_results"] >= 0
        assert timing_metrics["test/postprocess_results_pct"] >= 0

    asyncio.run(_scenario())


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
