# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


def test_uv_version_parser_accepts_platform_suffix() -> None:
    probe = Path(__file__).with_name("nightly_gpu_runtime_probe.py")
    namespace = runpy.run_path(str(probe))
    parser = namespace.get("_parse_uv_version")

    assert callable(parser), "runtime probe must expose a semantic uv version parser"
    parse_uv_version = cast(Callable[[str], str], parser)
    assert parse_uv_version("uv 0.11.28 (aarch64-unknown-linux-gnu)") == "0.11.28"


def test_cuda_graph_checksum_validation_rejects_empty_replay() -> None:
    probe = Path(__file__).with_name("nightly_gpu_runtime_probe.py")
    namespace = runpy.run_path(str(probe))
    validator = namespace.get("_validate_cuda_graph_checksums")

    assert callable(validator), "runtime probe must validate capture and replay effects"
    validate = cast(
        Callable[[float, float, float, float, float, float], None], validator
    )
    validate(4096.0, 8192.0, 16384.0, 4096.0, 8192.0, 16384.0)
    with pytest.raises(RuntimeError, match="first replay checksum mismatch"):
        validate(4096.0, 4096.0, 4096.0, 4096.0, 8192.0, 16384.0)


def test_launcher_stages_mcore_environment_in_writable_job_local_storage() -> None:
    launcher = (
        Path(__file__).parent / "scripts" / "validate_oci_nightly_gpu_runtime.sub"
    ).read_text()

    assert (
        'runtime_stage_root="/tmp/nemo-rl-nightly-mcore-smoke-${SLURM_JOB_ID}"'
        in launcher
    )
    assert 'cp -a -- "${source_project_root}/." "${runtime_project_root}/"' in launcher
    assert "--container-writable" in launcher
    assert "UV_CACHE_DIR=/root/.cache/uv" in launcher
    assert "for proxy_variable in HTTP_PROXY HTTPS_PROXY NO_PROXY" in launcher
    assert 'runtime_environment+=("${proxy_variable}=${!proxy_variable}")' in launcher
    assert '--directory "${runtime_project_root}"' in launcher
