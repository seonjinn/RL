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

"""Repository contracts for the pinned Megatron-Bridge dependency."""

import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
BRIDGE_PATH = Path("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge")
BRIDGE_COMMIT = "59c163cce9cb8cc209dcd0424b2b9de9d1be5027"
BRIDGE_TRANSFORMERS_REQUIREMENT = "transformers>=5.8.1,<5.9.0"


def test_root_bridge_gitlink_and_lock_metadata_match_corrected_requirement() -> None:
    """The staged Bridge pin and root lock metadata use the same Transformers range."""
    gitlink = subprocess.run(
        ["git", "ls-files", "--stage", "--", str(BRIDGE_PATH)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()

    assert gitlink[:3] == ["160000", BRIDGE_COMMIT, "0"]

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    bridge_manifest = next(
        metadata
        for metadata in lock["manifest"]["dependency-metadata"]
        if metadata["name"] == "megatron-bridge"
    )
    assert BRIDGE_TRANSFORMERS_REQUIREMENT in bridge_manifest["requires-dist"]

    bridge_package = next(
        package for package in lock["package"] if package["name"] == "megatron-bridge"
    )
    assert {
        "name": "transformers",
        "specifier": ">=5.8.1,<5.9.0",
    } in bridge_package["metadata"]["requires-dist"]
