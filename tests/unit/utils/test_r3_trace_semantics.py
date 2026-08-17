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


def test_graph_consumer_trace_contains_identity_without_content_or_addresses(
    tmp_path, monkeypatch
) -> None:
    from nemo_rl.utils.r3_trace import (
        r3_trace_stage,
        trace_router_replay_graph_consumer,
    )

    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    with r3_trace_stage("train"):
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
        )

    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert record["event"] == "router_replay_graph_consumer"
    assert record["stage"] == "train"
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
