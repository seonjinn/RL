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


def test_uv_version_parser_accepts_platform_suffix() -> None:
    probe = Path(__file__).with_name("nightly_gpu_runtime_probe.py")
    namespace = runpy.run_path(str(probe))
    parser = namespace.get("_parse_uv_version")

    assert callable(parser), "runtime probe must expose a semantic uv version parser"
    parse_uv_version = cast(Callable[[str], str], parser)
    assert parse_uv_version("uv 0.11.28 (aarch64-unknown-linux-gnu)") == "0.11.28"
