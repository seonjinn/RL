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

import re
from pathlib import Path


RAY_SUB = Path(__file__).parents[3] / "ray.sub"


def test_ray_sub_routes_cli_commands_through_configurable_executable() -> None:
    source = RAY_SUB.read_text(encoding="utf-8")

    assert "RAY_CLI=${RAY_CLI:-ray}" in source
    assert not re.search(r"^\s*ray (?:start|status|stop)\b", source, re.MULTILINE)
    assert source.count('"${RAY_CLI}"') == 5
