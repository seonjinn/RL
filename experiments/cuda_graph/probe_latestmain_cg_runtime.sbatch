#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00

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

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the staged latest-main NeMo-RL image.}"
: "${MOUNTS:?Set MOUNTS for the source and cache filesystems.}"

srun \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --no-container-mount-home \
  bash -lc 'uv run --locked --extra mcore python - <<"PY"
import importlib
from pathlib import Path

for module_name in (
    "nemo_rl",
    "megatron.core",
    "transformer_engine.pytorch",
    "megatron.bridge.training.checkpointing",
):
    module = importlib.import_module(module_name)
    print(f"{module_name}_path={Path(module.__file__).resolve()}")
print("combined_import=passed")
PY'
