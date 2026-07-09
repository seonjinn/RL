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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def hidden_state_capture(monkeypatch):
    megatron = ModuleType("megatron")
    megatron_core = ModuleType("megatron.core")
    megatron_core.parallel_state = SimpleNamespace()
    megatron_utils = ModuleType("megatron.core.utils")
    megatron_utils.unwrap_model = lambda model: model

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", megatron_utils)

    module_path = (
        Path(__file__).parents[5] / "nemo_rl/models/megatron/draft/hidden_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hidden_capture_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.HiddenStateCapture


def test_send_tensor_uses_global_rank_for_subgroup(monkeypatch, hidden_state_capture):
    pp_group = object()
    global_rank_calls = []
    send_calls = []

    def get_global_rank(group, group_rank):
        global_rank_calls.append((group, group_rank))
        return {0: 1, 1: 3}[group_rank]

    def send(tensor, *, dst, group):
        send_calls.append((tensor, dst, group))

    monkeypatch.setattr(torch.distributed, "get_global_rank", get_global_rank)
    monkeypatch.setattr(torch.distributed, "send", send)

    payload = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    hidden_state_capture._send_tensor(payload, dst_rank=1, group=pp_group)

    assert global_rank_calls == [(pp_group, 1)]
    assert len(send_calls) == 2
    assert all(dst == 3 and group is pp_group for _, dst, group in send_calls)
    assert torch.equal(send_calls[1][0], payload.contiguous())


def test_recv_tensor_uses_global_rank_for_subgroup(monkeypatch, hidden_state_capture):
    pp_group = object()
    global_rank_calls = []
    recv_calls = []

    def get_global_rank(group, group_rank):
        global_rank_calls.append((group, group_rank))
        return {0: 1, 1: 3}[group_rank]

    def recv(tensor, *, src, group):
        recv_calls.append((tensor, src, group))
        if len(recv_calls) == 1:
            tensor.copy_(torch.tensor([2, 3, 4, 1], dtype=torch.int64))
        else:
            tensor.fill_(7)

    monkeypatch.setattr(torch.distributed, "get_global_rank", get_global_rank)
    monkeypatch.setattr(torch.distributed, "recv", recv)

    received = hidden_state_capture._recv_tensor(
        src_rank=0,
        group=pp_group,
        device=torch.device("cpu"),
    )

    assert global_rank_calls == [(pp_group, 0)]
    assert len(recv_calls) == 2
    assert all(src == 1 and group is pp_group for _, src, group in recv_calls)
    assert received.shape == (2, 3, 4)
    assert received.dtype is torch.bfloat16
    assert torch.equal(received, torch.full((2, 3, 4), 7, dtype=torch.bfloat16))
