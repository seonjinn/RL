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

import argparse
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from tomlkit import array, dumps, inline_table, parse, table


def configure_custom_vllm_pyproject(
    source: str,
    *,
    source_path: str = "3rdparty/vllm",
) -> str:
    """Configure a pyproject to use one editable custom vLLM source."""
    document = parse(source)
    vllm_requirements = document["project"]["optional-dependencies"]["vllm"]
    retained_requirements = [
        requirement
        for requirement in vllm_requirements
        if canonicalize_name(Requirement(str(requirement)).name) != "vllm"
    ]
    retained_requirements.append("vllm")
    vllm_requirements.clear()
    vllm_requirements.extend(retained_requirements)

    tool = document.setdefault("tool", table())
    uv = tool.setdefault("uv", table())
    sources = uv.setdefault("sources", table())
    custom_source = inline_table()
    custom_source.update({"path": source_path, "editable": True})
    sources["vllm"] = custom_source

    no_build_isolation = uv.setdefault("no-build-isolation-package", array())
    retained_no_build_isolation = [
        package
        for package in no_build_isolation
        if canonicalize_name(str(package)) != "vllm"
    ]
    retained_no_build_isolation.append("vllm")
    no_build_isolation.clear()
    no_build_isolation.extend(retained_no_build_isolation)

    return dumps(document)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure pyproject.toml for an editable custom vLLM source."
    )
    parser.add_argument("pyproject", type=Path)
    args = parser.parse_args()

    source = args.pyproject.read_text(encoding="utf-8")
    configured = configure_custom_vllm_pyproject(source)
    args.pyproject.write_text(configured, encoding="utf-8")


if __name__ == "__main__":
    _main()
