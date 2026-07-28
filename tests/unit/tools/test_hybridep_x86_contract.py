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


DEEPEP_COMMIT = "f725d29699f5bda9ba789456bb9579af69844685"
OLD_X86_COMMIT = "29d31c095796f3c8ece47ee9cdcc167051bbeed9"


def test_linux_x86_uses_latest_hybridep_commit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pyproject = (project_root / "pyproject.toml").read_text()
    x86_dependencies = [
        line
        for line in pyproject.splitlines()
        if "deep_ep @ git+" in line and "platform_machine == 'x86_64'" in line
    ]

    assert len(x86_dependencies) == 4
    assert all(DEEPEP_COMMIT in line for line in x86_dependencies)
    assert OLD_X86_COMMIT not in pyproject


def test_lock_does_not_retain_the_pre_hybridep_x86_commit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    lock = (project_root / "uv.lock").read_text()

    assert OLD_X86_COMMIT not in lock
    assert f"DeepEP.git?rev={DEEPEP_COMMIT}" in lock

