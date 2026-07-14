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

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_JOB = PROJECT_ROOT / "scripts/smoke_nemo_rl_container.sbatch"


def test_smoke_bootstraps_the_locked_mcore_runtime() -> None:
    source = SMOKE_JOB.read_text()

    assert 'UV_PROJECT_ENVIRONMENT="/runtime/venv"' in source
    assert 'UV_BIN="/root/.local/bin/uv"' in source
    assert 'UV_BIN="/opt/nemo_rl_venv/bin/uv"' not in source
    assert 'CUDA_HOME="/usr/local/cuda"' in source
    assert 'NVTE_CUDA_ARCHS="100"' in source
    assert '"${UV_BIN}" sync --locked --extra mcore' in source
    assert '"${UV_BIN}" run --active --no-sync python' in source
    assert "scripts/smoke_nemo_rl_image.py" in source


def test_smoke_does_not_assume_optional_extras_are_in_the_base_venv() -> None:
    source = SMOKE_JOB.read_text()

    assert 'python_bin="/opt/nemo_rl_venv/bin/python"' not in source
    assert '"${python_bin}"' not in source


def test_smoke_records_source_and_image_identity() -> None:
    source = SMOKE_JOB.read_text()

    assert "CONTAINER_IMAGE_SHA256" in source
    assert 'git -C "${repo_root}" rev-parse HEAD' in source
    assert "sha256sum" in source
