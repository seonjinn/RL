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

import json

import torch


def test_sample_payload_identity_is_valid_length_bound_and_typed() -> None:
    from nemo_rl.utils.r3_trace import sample_payload_identity

    input_ids = torch.tensor([11, 12, 99], dtype=torch.int64)
    routes = torch.tensor([[[0, 1]], [[2, 3]], [[3, 0]]], dtype=torch.int32)
    identity = sample_payload_identity(input_ids, routes, 2)

    input_ids[2] = 7
    routes[2] = 1
    assert sample_payload_identity(input_ids, routes, 2) == identity
    routes[1, 0, 0] = 3
    assert sample_payload_identity(input_ids, routes, 2) != identity


def test_graph_consumer_trace_contains_identity_without_content_or_addresses(
    tmp_path, monkeypatch
) -> None:
    from nemo_rl.utils.r3_trace import (
        r3_trace_stage,
        trace_router_replay_graph_consumer,
    )

    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "99")
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    with r3_trace_stage("train", enclosing_call_id=7):
        trace_router_replay_graph_consumer(
            action="replay_forward",
            layer_number=4,
            payload_idx=1,
            microbatch_generation=19,
            route_digest="a" * 64,
            physical_signature={
                "shape": [32, 2],
                "dtype": "torch.int64",
                "device_type": "cuda",
                "topk": 2,
                "num_experts": 8,
            },
            bank_id=7,
            graph_index=3,
            schedule_key=5,
            copy_generation=11,
            successful_graph_launch=True,
            capability_version="r3_router_cuda_graph_input_v1",
            source_sample_identities=(("sample-0", "b" * 64),),
        )

    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert record["event"] == "router_replay_graph_consumer"
    assert record["stage"] == "train"
    assert record["enclosing_call_id"] == 7
    assert record["action"] == "replay_forward"
    assert record["layer_number"] == 4
    assert record["payload_idx"] == 1
    assert record["microbatch_generation"] == 19
    assert record["route_digest"] == "a" * 64
    assert record["physical_signature"]["shape"] == [32, 2]
    assert record["bank_id"] == 7
    assert record["graph_index"] == 3
    assert record["schedule_key"] == 5
    assert record["copy_generation"] == 11
    assert record["successful_graph_launch"] is True
    assert record["capability_version"] == "r3_router_cuda_graph_input_v1"
    assert record["source_sample_identities"] == [
        {"key": "sample-0", "identity": "b" * 64}
    ]
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in (
        "input_ids",
        "prompt",
        "token_content",
        "preview",
        "data_ptr",
        "static_address",
    ):
        assert forbidden not in serialized


def test_graph_counter_trace_carries_detached_call_identity(
    tmp_path, monkeypatch
) -> None:
    from nemo_rl.utils.r3_trace import (
        current_r3_trace_call_identity,
        r3_trace_stage,
        trace_router_replay_graph_counters,
    )

    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "99")
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    with r3_trace_stage("train", enclosing_call_id=7):
        identity = current_r3_trace_call_identity()
        assert identity is not None
        assert identity.stage == "train"
        assert identity.trace_step >= 1
        assert identity.enclosing_call_id == 7
    assert current_r3_trace_call_identity() is None

    trace_router_replay_graph_counters(
        {"route_payloads_produced": 10},
        call_identity=identity,
        schedule_key=5,
        num_microbatches=5,
    )

    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert record["event"] == "router_replay_graph_counters"
    assert record["stage"] == "train"
    assert record["trace_step"] == identity.trace_step
    assert record["enclosing_call_id"] == 7
    assert record["schedule_key"] == 5
    assert record["num_microbatches"] == 5


def test_reduced_graph_counter_summary_uses_distinct_event(
    tmp_path, monkeypatch
) -> None:
    from nemo_rl.utils.r3_trace import trace_router_replay_graph_counter_summary

    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    trace_router_replay_graph_counter_summary(
        {"route_payloads_produced": 10},
        enclosing_call_id=7,
        local_calls=((1, 5, 5), (2, 5, 5)),
    )

    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert record["event"] == "router_replay_graph_counter_summary"
    assert record["scope"] == "global_reduced"
    assert "trace_step" not in record
    assert record["enclosing_call_id"] == 7
    assert record["local_calls"] == [
        {"trace_step": 1, "schedule_key": 5, "num_microbatches": 5},
        {"trace_step": 2, "schedule_key": 5, "num_microbatches": 5},
    ]
    assert record["local_trace_step_range"] == [1, 2]
    assert record["local_call_count"] == 2
    assert record["schedule_keys"] == [5]
