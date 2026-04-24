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

from __future__ import annotations

import os
import subprocess

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv

DEFAULT_GATE6_VENV_DIR = "/lustre/fs1/portfolios/coreai/users/aroshanghias/tmp"


def main() -> int:
    os.environ["NEMO_RL_VENV_DIR"] = os.environ.get(
        "GATE6_VENV_DIR", DEFAULT_GATE6_VENV_DIR
    )
    force_rebuild = os.environ.get("GATE6_FORCE_REBUILD_VENV") == "1"
    python_path = create_local_venv(
        PY_EXECUTABLES.VLLM,
        "gate6_vllm_env_smoke",
        force_rebuild=force_rebuild,
    )
    print(f"VENV_ROOT={os.environ['NEMO_RL_VENV_DIR']}", flush=True)
    print(f"VENV_PYTHON={python_path}", flush=True)
    result = subprocess.run(
        [
            python_path,
            "-c",
            "import nemo_rl; import vllm; import msgspec; print('IMPORT_OK')",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
