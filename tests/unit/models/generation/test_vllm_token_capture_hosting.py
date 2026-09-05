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

"""S2 worker hosting: install_capture wiring, fan-outs, version stamping.

Marked nemo_gym (run with ``--nemo-gym-only``): the hosting seam imports
Gym's capture core. No engine or GPU is needed — the worker methods are
driven unbound against light fakes, and the VllmGeneration fan-outs against
a mock worker group.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.models.generation.interfaces import (  # noqa: E402
    GenerationLifecycleNotDispatchedError,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)


def _fake_worker(*, is_model_owner: bool = True) -> SimpleNamespace:
    """The attribute surface setup_token_capture touches, minus the engine."""
    worker = SimpleNamespace(
        is_model_owner=is_model_owner,
        token_capture=None,
        _rollout_weight_version=0,
        _staging_source=None,
        _prefix_cache={},
        _prefix_cache_lock=threading.Lock(),
    )
    worker.install_token_capture = lambda capture: setattr(
        worker, "token_capture", capture
    )
    return worker


def test_setup_token_capture_installs_capture_with_vllm_adapter(monkeypatch):
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap: MagicMock(name="dp_client"),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition: sink,
    )
    worker = _fake_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={"backend": "simple"}, staging_partition="rollout_staging"
        )
    )

    assert installed is True
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert worker.token_capture.adapter is not None
    # The adapter is the vLLM one (prefix ids enter via the worker's field).
    payload = worker.token_capture.adapter.enter_prefix({}, [1, 2])
    assert payload["required_prefix_token_ids"] == [1, 2]


def test_setup_token_capture_skips_non_model_owners(monkeypatch):
    worker = _fake_worker(is_model_owner=False)
    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging"
        )
    )
    assert installed is False
    assert worker.token_capture is None


def test_weight_version_is_stamped_from_worker_state(monkeypatch):
    """The install closure reads _rollout_weight_version live: a
    set_rollout_weight_version between calls changes the stamp."""
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition: sink,
    )
    worker = _fake_worker()
    asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging"
        )
    )

    assert (
        asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 4))
        is True
    )
    first = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c1", mode="text")
    )
    assert (
        asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 5))
        is True
    )
    second = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c2", mode="text")
    )

    assert (first.weight_version, second.weight_version) == (4, 5)

    coords = worker.token_capture.complete_call(
        first, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1]
    )
    assert coords.weight_version == 4
    assert sink.records[0].weight_version == 4


@pytest.mark.parametrize("version", [True, -1, 1.0, "1", None])
def test_worker_rejects_non_exact_or_negative_weight_versions(version):
    worker = _fake_worker()
    worker._rollout_weight_version = 9

    with pytest.raises(ValueError, match="exact nonnegative int"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, version)
        )

    assert worker._rollout_weight_version == 9


class _RemoteMethod:
    def __init__(self, result=True, *, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _Leader:
    def __init__(self, index: int) -> None:
        self.index = index
        self.setup_token_capture = _RemoteMethod()
        self.set_rollout_weight_version = _RemoteMethod()
        self.reset_prefix_cache_async = _RemoteMethod()
        self.pause_generation_async = _RemoteMethod()
        self.resume_generation_async = _RemoteMethod()
        self.update_weights_from_collective_async = _RemoteMethod()
        self.nccl_reshard_refit_async = _RemoteMethod()


def _generation_with_mock_group(*, async_engine: bool = True) -> VllmGeneration:
    gen = object.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    leaders = [_Leader(0), _Leader(1)]
    gen.worker_group = SimpleNamespace(
        workers=leaders,
        dp_size=2,
        shutdown=lambda **_: True,
    )
    gen.dp_size = 2
    gen._refit_membership = None
    gen._refit_commit_poison = None
    gen.weight_synchronizer = None
    return gen


def test_generation_setup_token_capture_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )
    assert gen.setup_token_capture({"backend": "simple"}, "rollout_staging") is True
    assert ray_timeouts == [300.0]
    for leader in gen.worker_group.workers:
        assert leader.setup_token_capture.calls == [
            {
                "dp_cfg": {"backend": "simple"},
                "staging_partition": "rollout_staging",
            }
        ]


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture({}, "rollout_staging")


def test_generation_set_rollout_weight_version_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )
    assert gen.set_rollout_weight_version(7, refit_timeout_s=17.5) is True
    assert ray_timeouts == [17.5]
    for leader in gen.worker_group.workers:
        assert leader.set_rollout_weight_version.calls == [{"version": 7}]


def test_generation_stamps_only_surviving_refit_leaders(monkeypatch):
    gen = _generation_with_mock_group()
    dead = gen.worker_group.workers[1]
    survivor = _Leader(2)
    gen.worker_group.workers.append(survivor)
    gen.dp_size = 3
    gen.worker_group.dp_size = 3
    gen._refit_membership = SimpleNamespace(
        shard_prefixes={0: 0, 2: 1}, workers_per_shard=1
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: futures,
    )

    assert gen.set_rollout_weight_version(8) is True

    assert gen.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 8}
    ]
    assert dead.set_rollout_weight_version.calls == []
    assert survivor.set_rollout_weight_version.calls == [{"version": 8}]


@pytest.mark.parametrize("result", [[True], [True, False], [True, None], []])
def test_generation_stamp_requires_exact_true_ack_from_every_survivor(
    monkeypatch, result
):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: result,
    )

    with pytest.raises(RuntimeError, match="acknowledgement|worker"):
        gen.set_rollout_weight_version(7)


def test_unknown_stamp_outcome_poison_prevents_retry_and_preserves_cause(monkeypatch):
    gen = _generation_with_mock_group()
    first_failure = TimeoutError("stamp RPC timed out")
    ray_calls = 0

    def _fail(_futures, *, timeout):
        nonlocal ray_calls
        del timeout
        ray_calls += 1
        raise first_failure

    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.get", _fail)

    with pytest.raises(TimeoutError) as first:
        gen.set_rollout_weight_version(7)
    assert first.value is first_failure

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        gen.set_rollout_weight_version(7)

    assert retry.value.__cause__ is first_failure
    assert ray_calls == 1
    for leader in gen.worker_group.workers:
        assert leader.set_rollout_weight_version.calls == [{"version": 7}]


def test_first_remote_failure_is_typed_not_dispatched_and_retry_safe(monkeypatch):
    gen = _generation_with_mock_group()
    first_failure = RuntimeError("first leader submission failed")
    first_method = _RemoteMethod(error=first_failure)
    gen.worker_group.workers[0].set_rollout_weight_version = first_method
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: futures,
    )

    with pytest.raises(GenerationLifecycleNotDispatchedError) as first:
        gen.set_rollout_weight_version(7)

    assert first.value.__cause__ is first_failure
    assert gen._refit_commit_poison is None
    assert first_method.calls == [{"version": 7}]
    assert gen.worker_group.workers[1].set_rollout_weight_version.calls == []

    first_method.error = None
    assert gen.set_rollout_weight_version(7) is True
    assert first_method.calls == [{"version": 7}, {"version": 7}]
    assert gen.worker_group.workers[1].set_rollout_weight_version.calls == [
        {"version": 7}
    ]


def test_later_remote_failure_poison_preserves_partial_dispatch_cause():
    gen = _generation_with_mock_group()
    partial_failure = RuntimeError("second leader submission failed")
    second_method = _RemoteMethod(error=partial_failure)
    gen.worker_group.workers[1].set_rollout_weight_version = second_method

    with pytest.raises(RuntimeError) as first:
        gen.set_rollout_weight_version(7)

    assert first.value is partial_failure
    assert gen._refit_commit_poison is partial_failure
    assert gen.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 7}
    ]
    assert second_method.calls == [{"version": 7}]

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        gen.set_rollout_weight_version(7)
    assert retry.value.__cause__ is partial_failure
    assert gen.worker_group.workers[0].set_rollout_weight_version.calls == [
        {"version": 7}
    ]
    assert second_method.calls == [{"version": 7}]


def test_poisoned_stamp_rejects_a_later_refit_before_dispatch(monkeypatch):
    gen = _generation_with_mock_group()
    first_failure = TimeoutError("stamp RPC timed out")
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: (_ for _ in ()).throw(first_failure),
    )
    with pytest.raises(TimeoutError):
        gen.set_rollout_weight_version(7)

    with pytest.raises(RuntimeError, match="poisoned") as caught:
        gen.update_weights_from_collective(refit_timeout_s=2.0)

    assert caught.value.__cause__ is first_failure
    for leader in gen.worker_group.workers:
        assert leader.update_weights_from_collective_async.calls == []


def test_cache_invalidation_is_bounded_exact_and_poisoning(monkeypatch):
    gen = _generation_with_mock_group()
    malformed = [True, None]
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or malformed,
    )

    with pytest.raises(RuntimeError, match="expected exactly True") as first:
        gen.invalidate_kv_cache(refit_timeout_s=6.25)

    assert ray_timeouts == [6.25]
    with pytest.raises(RuntimeError, match="poisoned") as retry:
        gen.set_rollout_weight_version(9)
    assert retry.value.__cause__ is first.value


def test_refit_pause_resume_are_bounded_and_use_only_survivor_leaders(monkeypatch):
    gen = _generation_with_mock_group()
    assert gen.supports_refit_pause_resume is True
    assert gen.supports_refit_pause_resume_timeout is True
    dead = gen.worker_group.workers[1]
    survivor = _Leader(2)
    gen.worker_group.workers.append(survivor)
    gen.dp_size = 3
    gen.worker_group.dp_size = 3
    gen._refit_membership = SimpleNamespace(
        shard_prefixes={0: 0, 2: 1}, workers_per_shard=1
    )
    ray_timeouts = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: ray_timeouts.append(timeout) or futures,
    )

    assert gen.pause_generation_for_refit(clear_cache=True, refit_timeout_s=4.5) is True
    assert gen.resume_generation_after_refit(refit_timeout_s=3.25) is True

    assert ray_timeouts == [4.5, 3.25]
    assert gen.worker_group.workers[0].pause_generation_async.calls == [
        {"clear_cache": True}
    ]
    assert survivor.pause_generation_async.calls == [{"clear_cache": True}]
    assert dead.pause_generation_async.calls == []
    assert gen.worker_group.workers[0].resume_generation_async.calls == [{}]
    assert survivor.resume_generation_async.calls == [{}]
    assert dead.resume_generation_async.calls == []


def test_refit_pause_requires_exact_ack_and_poison_blocks_resume(monkeypatch):
    gen = _generation_with_mock_group()
    malformed = [True, 1]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, *, timeout: malformed,
    )

    with pytest.raises(RuntimeError, match="expected exactly True") as first:
        gen.pause_generation_for_refit(clear_cache=True, refit_timeout_s=2.0)

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        gen.resume_generation_after_refit(refit_timeout_s=2.0)
    assert retry.value.__cause__ is first.value
    for leader in gen.worker_group.workers:
        assert leader.resume_generation_async.calls == []


# ---------------------------------------------------------------------------
# S4: the request-path hookup (begin -> finish/abort around a served call)
# ---------------------------------------------------------------------------


class _FakeRequest(SimpleNamespace):
    pass


def _worker_with_capture(sink: _MemorySink):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = _fake_worker()
    worker._capture_calls = {}
    worker._prefix_cache = {}
    worker._prefix_cache_lock = threading.Lock()
    worker._staging_source = None
    worker._delta_align_routed_experts = (
        VllmAsyncGenerationWorkerImpl._delta_align_routed_experts
    )
    for name in (
        "_fetch_chain_prefix",
        "_capture_admission",
        "_resolve_admission_prefix",
        "_enter_request_prefix",
    ):
        setattr(
            worker, name, getattr(VllmAsyncGenerationWorkerImpl, name).__get__(worker)
        )
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: worker._rollout_weight_version,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


class _MemoryPrefixSource:
    def __init__(self, deltas: dict[str, list[int]]) -> None:
        self.deltas = deltas
        self.calls: list[list[str]] = []

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        self.calls.append(list(staging_keys))
        return [token for key in staging_keys for token in self.deltas[key]]


def _served_content(gen_ids, logprobs):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{t}", "logprob": lp}
                        for t, lp in zip(gen_ids, logprobs)
                    ]
                },
            }
        ]
    }


def test_request_capture_round_trip_stages_and_rides_coords():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = _served_content([13, 14], [-0.1, -0.2])
    # Full-length routes on the served response must not survive the strip.
    content["choices"][0]["message"]["routed_experts"] = [[[0]]] * 5
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )
    # Bytes were staged before the coords existed (fail-closed ordering).
    assert len(sink.records) == 1
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13, 14]
    coords = content["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert (coords["delta_len"], coords["cum_len"]) == (5, 5)
    # Coords are token-free: hashes ride the wire, deltas stay in the sink.
    assert "token_ids_delta" not in coords
    assert coords["chain_hash"] == sink.records[0].chain_hash
    assert coords["cumulative_hash"] == sink.records[0].cumulative_hash
    # Logprobs and routes never transit worker -> gate; state map is drained.
    assert (
        "logprobs" not in content["choices"][0]
        or content["choices"][0]["logprobs"] is None
    )
    assert "routed_experts" not in content["choices"][0]["message"]
    assert worker._capture_calls == {}


def test_request_capture_token_in_prev_len_chains():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 3,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11, 12],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    spliced_prompt = [10, 11, 12, 20, 21]  # exact prefix + fresh suffix
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, spliced_prompt
    )
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([22], [-0.5])
    )
    coords = content["ng_commit_coords"]
    assert coords["parent_call_id"] == "c1"
    assert (coords["delta_len"], coords["cum_len"]) == (3, 6)
    assert sink.records[0].token_ids_delta == [20, 21, 22]


def _staging_chain_request(prev_len: int = 3) -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c3",
            "parent_call_id": "c2",
            "prev_len": prev_len,
            "mode": "token_in",
            "staging_chain": ["r0/c1", "r0/c2"],
            "parent_chain_hash": "2" * 64,
        },
        stream=False,
    )


def test_staging_chain_prefix_flows_through_adapter_and_begin_call():
    """The admission dict is read once and never mutated: the resolved prefix
    reaches the request via the adapter and begin_call via its keyword."""
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._staging_source = source
    request = _staging_chain_request()
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    worker._enter_request_prefix(request, prefix)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker,
        request,
        prefix + [20],
        admission=admission,
        prefix_token_ids=prefix,
    )

    assert prefix == [10, 11, 12]
    assert source.calls == [["r0/c1", "r0/c2"]]
    # Never patched back into the wire context.
    assert request.ng_capture == context_before
    assert admission.required_prefix_token_ids == []
    # enter_prefix is the production writer of the request field.
    assert request.required_prefix_token_ids == prefix
    call, prompt = worker._capture_calls[id(request)]
    assert call.prefix_token_ids == prefix
    assert prompt == [10, 11, 12, 20]


def test_inline_prefix_admission_resolves_without_a_fetch():
    worker = _worker_with_capture(_MemorySink())
    worker._staging_source = _MemoryPrefixSource({})
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 2,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    admission = worker._capture_admission(request)
    assert worker._resolve_admission_prefix(admission) == [10, 11]
    assert worker._staging_source.calls == []
    text_root = worker._capture_admission(
        _FakeRequest(
            ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
        )
    )
    assert worker._resolve_admission_prefix(text_root) == []


def test_staging_chain_cache_fetches_only_uncached_suffix():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._staging_source = source

    first = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(worker, ["r0/c1"])
    second = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(
        worker, ["r0/c1", "r0/c2"]
    )

    assert first == [10, 11]
    assert second == [10, 11, 12]
    assert source.calls == [["r0/c1"], ["r0/c2"]]


def test_staging_chain_prefix_length_mismatch_is_rejected_by_begin_call():
    """prev_len enforcement lives in Gym's begin_call, not in the worker."""
    worker = _worker_with_capture(_MemorySink())
    worker._staging_source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": []})
    request = _staging_chain_request(prev_len=3)
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    assert prefix == [10, 11]
    with pytest.raises(CaptureError, match="does not equal prev_len 3"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, prefix + [20], admission=admission, prefix_token_ids=prefix
        )

    assert request.ng_capture == context_before
    assert worker._capture_calls == {}


def test_staging_chain_admission_requires_the_resolved_prefix_keyword():
    worker = _worker_with_capture(_MemorySink())
    request = _staging_chain_request()

    with pytest.raises(CaptureError, match="pass the resolved prefix_token_ids"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11, 12, 20]
        )

    assert worker._capture_calls == {}


def test_request_capture_is_a_noop_without_context_or_capture():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, plain, [1, 2])
    content = {
        "choices": [{"message": {"role": "assistant"}, "logprobs": {"content": []}}]
    }
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, plain, dict(content)
    )
    assert "ng_commit_coords" not in out
    assert out["choices"][0]["logprobs"] is not None  # untouched off the capture path
    assert sink.records == []


def test_request_capture_abort_fails_the_call_and_drains_state():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [1, 2])
    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="engine_error"
    )
    assert worker._capture_calls == {}
    assert sink.records == []
    # A late finish after abort is a no-op (state already drained).
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([3], [-0.1])
    )
    assert "ng_commit_coords" not in out
