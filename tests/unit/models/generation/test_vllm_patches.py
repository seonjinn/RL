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

PATCHES_PATH = Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/patches.py"
PATCHES_SPEC = importlib.util.spec_from_file_location(
    "nemo_rl_vllm_patches_under_test", PATCHES_PATH
)
assert PATCHES_SPEC is not None
PATCHES_MODULE = importlib.util.module_from_spec(PATCHES_SPEC)
assert PATCHES_SPEC.loader is not None
PATCHES_SPEC.loader.exec_module(PATCHES_MODULE)

_patch_vllm_parallel_state_timeout_content = (
    PATCHES_MODULE._patch_vllm_parallel_state_timeout_content
)
_patch_vllm_partial_wake_scheduler_content = (
    PATCHES_MODULE._patch_vllm_partial_wake_scheduler_content
)


_VLLM_ENGINE_CORE_SOURCE = """
class ModelExecutor:
    def __init__(self):
        self.is_sleeping = True

    def wake_up(self, tags):
        if tags is None or "kv_cache" in tags:
            self.is_sleeping = False


class EngineCore:
    def __init__(self):
        self.model_executor = ModelExecutor()
        self.resume_count = 0

    def resume_scheduler(self):
        self.resume_count += 1

    def wake_up(self, tags: list[str] | None = None):
        if tags is not None and "scheduling" in tags:
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Resume scheduling (applies to all levels)
        self.resume_scheduler()
"""


def test_vllm_partial_wake_patch_resumes_only_after_full_wake():
    patched = _patch_vllm_partial_wake_scheduler_content(
        _VLLM_ENGINE_CORE_SOURCE, "core.py"
    )
    namespace = {}
    exec(patched, namespace)
    engine = namespace["EngineCore"]()

    engine.wake_up(["weights"])
    assert engine.resume_count == 0

    engine.wake_up(["kv_cache"])
    assert engine.resume_count == 1


def test_vllm_partial_wake_patch_is_idempotent():
    patched = _patch_vllm_partial_wake_scheduler_content(
        _VLLM_ENGINE_CORE_SOURCE, "core.py"
    )

    assert _patch_vllm_partial_wake_scheduler_content(patched, "core.py") == patched


def test_vllm_partial_wake_patch_raises_when_resume_pattern_is_missing():
    with pytest.raises(RuntimeError, match="partial wake"):
        _patch_vllm_partial_wake_scheduler_content(
            "class EngineCore:\n    pass\n", "core.py"
        )


def test_vllm_timeout_patch_handles_current_parallel_state_import_shape():
    source = """
from vllm.distributed.utils import stateless_init_torch_distributed_process_group

def init_model_parallel_group(group_ranks, local_rank, backend):
    device_group = torch.distributed.new_group(
        ranks,
        backend=torch_distributed_backend,
    )
    return device_group
"""

    patched = _patch_vllm_parallel_state_timeout_content(source, "parallel_state.py")

    assert "get_current_vllm_config_or_none" in patched
    assert "device_timeout = _nemo_rl_get_distributed_timeout_or_none()" in patched
    assert "timeout=device_timeout" in patched


def test_vllm_timeout_patch_handles_legacy_cpu_timeout_import_shape():
    source = """
from vllm.distributed.utils import get_cpu_distributed_timeout_or_none

def init_model_parallel_group(group_ranks, local_rank, backend):
    timeout = get_cpu_distributed_timeout_or_none()
    cpu_group = torch.distributed.new_group(
        ranks,
        backend="gloo",
        timeout=timeout,
    )
    device_group = torch.distributed.new_group(
        ranks,
        backend=torch_distributed_backend,
    )
    return device_group
"""

    patched = _patch_vllm_parallel_state_timeout_content(source, "parallel_state.py")

    assert "get_cpu_distributed_timeout_or_none" in patched
    assert "device_timeout = _nemo_rl_get_distributed_timeout_or_none()" in patched
    assert "timeout=timeout" in patched
    assert "timeout=device_timeout" in patched


def test_vllm_timeout_patch_is_idempotent():
    source = """
from vllm.config import get_current_vllm_config_or_none

def _nemo_rl_get_distributed_timeout_or_none():
    return None

def init_model_parallel_group(group_ranks, local_rank, backend):
    device_timeout = _nemo_rl_get_distributed_timeout_or_none()
    device_group = torch.distributed.new_group(
        ranks,
        backend=torch_distributed_backend,
        timeout=device_timeout,
    )
    return device_group
"""

    patched = _patch_vllm_parallel_state_timeout_content(source, "parallel_state.py")

    assert patched == source


def test_vllm_timeout_patch_preserves_decorated_top_level_classes():
    source = """
from dataclasses import dataclass

@dataclass
class ProcessGroup:
    ranks: list[int]

def init_model_parallel_group(group_ranks, local_rank, backend):
    device_group = torch.distributed.new_group(
        ranks,
        backend=torch_distributed_backend,
    )
    return device_group
"""

    patched = _patch_vllm_parallel_state_timeout_content(source, "parallel_state.py")

    assert "@dataclass\nclass ProcessGroup:" in patched
    compile(patched, "parallel_state.py", "exec")


def test_vllm_timeout_patch_raises_when_device_group_pattern_is_missing():
    source = """
def init_model_parallel_group(group_ranks, local_rank, backend):
    return None
"""

    with pytest.raises(RuntimeError, match="device new_group"):
        _patch_vllm_parallel_state_timeout_content(source, "parallel_state.py")
