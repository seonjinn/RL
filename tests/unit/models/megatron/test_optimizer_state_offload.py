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
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeDistributedOptimizer:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.optimizer = SimpleNamespace(state={object(): {"exp_avg": object()}})


class _FakeChainedOptimizer:
    def __init__(self, optimizers: list[_FakeDistributedOptimizer]):
        self.chained_optimizers = optimizers


class _FakeOffloader:
    def __init__(self, optimizer: _FakeDistributedOptimizer):
        self.optimizer = optimizer
        self.adam_optimizer = optimizer.optimizer
        self._offloaded = False
        self.optimizer.events.append(f"create:{self.optimizer.name}")

    def mark_optimizer_states_initialized(self) -> None:
        self.optimizer.events.append(f"mark:{self.optimizer.name}")

    def offload(self) -> None:
        self.optimizer.events.append(f"offload:{self.optimizer.name}")
        self._offloaded = True

    def release_gpu_memory(self) -> None:
        self.optimizer.events.append(f"release:{self.optimizer.name}")

    def reload(self) -> None:
        self.optimizer.events.append(f"reload:{self.optimizer.name}")
        self._offloaded = False

    def sync_before_step(self) -> None:
        self.optimizer.events.append(f"wait:{self.optimizer.name}")


def _patch_offload_runtime(monkeypatch, events: list[str]):
    if importlib.util.find_spec("torch") is None:
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(synchronize=lambda: None)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    from nemo_rl.models.megatron import optimizer_state_offload

    monkeypatch.setattr(
        optimizer_state_offload,
        "DistributedOptimizer",
        _FakeDistributedOptimizer,
    )
    monkeypatch.setattr(
        optimizer_state_offload,
        "OptimizerStateOffloader",
        _FakeOffloader,
    )
    monkeypatch.setattr(
        optimizer_state_offload.torch.cuda,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    return optimizer_state_offload


def test_offload_waits_for_all_pinned_copies_before_releasing_storage(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)
    optimizer = _FakeChainedOptimizer(
        [
            _FakeDistributedOptimizer("dense", events),
            _FakeDistributedOptimizer("expert", events),
        ]
    )

    handled = module.move_distributed_optimizer_state(optimizer, "cpu")

    assert handled is True
    assert events == [
        "create:dense",
        "create:expert",
        "mark:dense",
        "offload:dense",
        "mark:expert",
        "offload:expert",
        "synchronize",
        "release:dense",
        "release:expert",
    ]


def test_restore_reuses_offloaders_and_joins_h2d_streams(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)
    optimizers = [
        _FakeDistributedOptimizer("dense", events),
        _FakeDistributedOptimizer("expert", events),
    ]
    optimizer = _FakeChainedOptimizer(optimizers)
    module.move_distributed_optimizer_state(optimizer, "cpu")
    events.clear()

    handled = module.move_distributed_optimizer_state(optimizer, "cuda")

    assert handled is True
    assert events == [
        "reload:dense",
        "reload:expert",
        "wait:dense",
        "wait:expert",
    ]


def test_repeated_cpu_move_is_a_noop_while_state_is_offloaded(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)
    optimizer = _FakeChainedOptimizer([_FakeDistributedOptimizer("dense", events)])
    module.move_distributed_optimizer_state(optimizer, "cpu")
    events.clear()

    handled = module.move_distributed_optimizer_state(optimizer, "cpu")

    assert handled is True
    assert events == []


def test_unsupported_optimizer_uses_existing_fallback(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)

    assert module.move_distributed_optimizer_state(object(), "cpu") is False
    assert events == []


def test_distributed_optimizer_without_supported_fused_adam_uses_fallback(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)

    class UnsupportedOffloader:
        def __init__(self, optimizer):
            raise AssertionError("requires TE FusedAdam")

    monkeypatch.setattr(module, "OptimizerStateOffloader", UnsupportedOffloader)
    optimizer = _FakeChainedOptimizer([_FakeDistributedOptimizer("dense", events)])

    assert module.move_distributed_optimizer_state(optimizer, "cpu") is False
    assert events == []


def test_invalid_device_is_rejected_before_mutating_optimizer(monkeypatch):
    events: list[str] = []
    module = _patch_offload_runtime(monkeypatch, events)
    optimizer = _FakeChainedOptimizer([_FakeDistributedOptimizer("dense", events)])

    with pytest.raises(ValueError, match="Only strings 'cpu' and 'cuda'"):
        module.move_distributed_optimizer_state(optimizer, "xpu")

    assert events == []
