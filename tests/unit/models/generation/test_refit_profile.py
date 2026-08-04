# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


def test_refit_phase_profiler_accumulates_wall_cuda_and_counter_metrics(
    monkeypatch,
):
    module_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/refit_profile.py"
    )
    spec = importlib.util.spec_from_file_location("refit_profile", module_path)
    assert spec is not None and spec.loader is not None
    refit_profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit_profile)

    timestamps = iter([10.0, 12.5])
    synchronize = MagicMock()

    class FakeEvent:
        def record(self):
            return None

        def elapsed_time(self, other):
            assert isinstance(other, FakeEvent)
            return 250.0

    monkeypatch.setattr(refit_profile.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(refit_profile.torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(refit_profile.torch.cuda, "Event", lambda **_: FakeEvent())

    profiler = refit_profile.RefitPhaseProfiler(enabled=True)
    with profiler.wall_phase("receive_and_load"):
        pass
    with profiler.cuda_phase("moe_layout_conversion"):
        pass
    profiler.increment("transport_clone_count", 3)
    profiler.increment("transport_clone_bytes", 4096)

    metrics = profiler.finish()

    assert metrics == {
        "moe_layout_conversion_event_count": 1,
        "moe_layout_conversion_gpu_sum_s": pytest.approx(0.25),
        "receive_and_load_s": pytest.approx(2.5),
        "transport_clone_bytes": 4096,
        "transport_clone_count": 3,
    }
    assert synchronize.call_count == 3


def test_refit_phase_profiler_is_noop_when_disabled(monkeypatch):
    module_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/refit_profile.py"
    )
    spec = importlib.util.spec_from_file_location("refit_profile", module_path)
    assert spec is not None and spec.loader is not None
    refit_profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit_profile)

    synchronize = MagicMock()
    event = MagicMock()
    monkeypatch.setattr(refit_profile.torch.accelerator, "synchronize", synchronize)
    monkeypatch.setattr(refit_profile.torch.cuda, "Event", event)

    profiler = refit_profile.RefitPhaseProfiler(enabled=False)
    with profiler.wall_phase("receive_and_load"):
        pass
    with profiler.cuda_phase("moe_layout_conversion"):
        pass
    profiler.increment("transport_clone_count", 3)

    assert profiler.finish() == {}
    synchronize.assert_not_called()
    event.assert_not_called()


def test_profile_weight_batch_records_payload_and_load_phase(monkeypatch):
    module_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/refit_profile.py"
    )
    spec = importlib.util.spec_from_file_location("refit_profile", module_path)
    assert spec is not None and spec.loader is not None
    refit_profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit_profile)

    profiler = MagicMock()
    profiler.cuda_phase.return_value = MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    )
    weights = [
        ("model.a", torch.zeros(3, dtype=torch.float32)),
        ("model.b", torch.zeros(5, dtype=torch.bfloat16)),
    ]
    load_weights = MagicMock()

    refit_profile.profile_weight_batch(profiler, weights, load_weights)

    profiler.increment.assert_any_call("received_batch_count")
    profiler.increment.assert_any_call("received_tensor_count", 2)
    profiler.increment.assert_any_call("received_weight_bytes", 22)
    profiler.cuda_phase.assert_called_once_with("load_weights")
    load_weights.assert_called_once_with(weights)


def test_emit_refit_profile_writes_parseable_stdout(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    module_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/refit_profile.py"
    )
    spec = importlib.util.spec_from_file_location("refit_profile", module_path)
    assert spec is not None and spec.loader is not None
    refit_profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit_profile)
    write = MagicMock()
    monkeypatch.setattr(refit_profile.os, "write", write)

    refit_profile.emit_refit_profile(
        rank=7,
        metrics={"receive_and_load_s": 1.25, "received_tensor_count": 3},
    )

    write.assert_called_once()
    fd, encoded_line = write.call_args.args
    assert fd == 1
    assert encoded_line.endswith(b"\n")
    prefix = b"[NRL_REFIT_PROFILE] "
    assert encoded_line.startswith(prefix)
    assert json.loads(encoded_line.removeprefix(prefix)) == {
        "rank": 7,
        "receive_and_load_s": 1.25,
        "received_tensor_count": 3,
    }


def test_refit_phase_profiler_does_not_mask_profiled_body_error(monkeypatch):
    module_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/refit_profile.py"
    )
    spec = importlib.util.spec_from_file_location("refit_profile", module_path)
    assert spec is not None and spec.loader is not None
    refit_profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit_profile)

    synchronize = MagicMock(side_effect=[None, RuntimeError("sync failed")])
    monkeypatch.setattr(refit_profile.torch.accelerator, "synchronize", synchronize)

    profiler = refit_profile.RefitPhaseProfiler(enabled=True)
    with pytest.raises(ValueError, match="body failed"):
        with profiler.wall_phase("receive_and_load"):
            raise ValueError("body failed")
