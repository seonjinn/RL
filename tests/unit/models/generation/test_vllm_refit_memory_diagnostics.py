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

import io
from dataclasses import asdict

import torch


def test_capture_refit_memory_snapshot_reports_exact_host_and_cuda_bytes(
    monkeypatch,
) -> None:
    from nemo_rl.models.generation.vllm import refit_memory_diagnostics as diagnostics

    proc_records = {
        "/proc/self/statm": "999 123 0 0 0 0 0\n",
        "/proc/meminfo": "MemTotal:       999999 kB\nMemAvailable:    4567 kB\n",
    }
    monkeypatch.setattr(
        diagnostics,
        "open",
        lambda path, *_args, **_kwargs: io.StringIO(proc_records[path]),
        raising=False,
    )
    monkeypatch.setattr(diagnostics.os, "sysconf", lambda _name: 4096)
    monkeypatch.setattr(diagnostics.os, "getpid", lambda: 321)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 111)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 222)

    snapshot = diagnostics.capture_refit_memory_snapshot("before_sleep")

    assert asdict(snapshot) == {
        "phase": "before_sleep",
        "pid": 321,
        "rss_bytes": 123 * 4096,
        "mem_available_bytes": 4567 * 1024,
        "cuda_allocated_bytes": 111,
        "cuda_reserved_bytes": 222,
    }


def test_capture_refit_memory_snapshot_uses_zero_cuda_bytes_when_unavailable(
    monkeypatch,
) -> None:
    from nemo_rl.models.generation.vllm import refit_memory_diagnostics as diagnostics

    proc_records = {
        "/proc/self/statm": "9 2 0 0 0 0 0\n",
        "/proc/meminfo": "MemAvailable:       3 kB\n",
    }
    monkeypatch.setattr(
        diagnostics,
        "open",
        lambda path, *_args, **_kwargs: io.StringIO(proc_records[path]),
        raising=False,
    )
    monkeypatch.setattr(diagnostics.os, "sysconf", lambda _name: 1024)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    snapshot = diagnostics.capture_refit_memory_snapshot("legacy_guard")

    assert snapshot.rss_bytes == 2048
    assert snapshot.mem_available_bytes == 3072
    assert snapshot.cuda_allocated_bytes == 0
    assert snapshot.cuda_reserved_bytes == 0
