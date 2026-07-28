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

"""Contracts for uv build isolation settings used by the MCore extra."""

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).parents[2]
FAST_HADAMARD_TRANSFORM = "fast-hadamard-transform"


def test_fast_hadamard_transform_uses_an_isolated_wheel_build() -> None:
    """A cold UV cache must provide the FHT build backend instead of image state."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    uv_config = config["tool"]["uv"]

    assert FAST_HADAMARD_TRANSFORM not in uv_config["no-build-isolation-package"]
    assert uv_config["extra-build-dependencies"][FAST_HADAMARD_TRANSFORM] == [
        {"requirement": "torch", "match-runtime": True},
        "wheel",
    ]
