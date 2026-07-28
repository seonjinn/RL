#!/usr/bin/env bash
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

# Submit the committed base matrix without Slurm dependencies. Select a
# particular reusable file directly when varying a MoE experiment axis.
set -euo pipefail

: "${CLUSTER:?Set CLUSTER to ptyche or oci-hsg.}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/scopes/00_nocg.sh"
bash "${SCRIPT_DIR}/scopes/01_attn.sh"
bash "${SCRIPT_DIR}/scopes/02_mamba.sh"
bash "${SCRIPT_DIR}/scopes/03_attn_mamba.sh"
bash "${SCRIPT_DIR}/scopes/04_moe.sh"
bash "${SCRIPT_DIR}/scopes/05_attn_moe.sh"
bash "${SCRIPT_DIR}/scopes/06_mamba_moe.sh"
bash "${SCRIPT_DIR}/scopes/07_attn_mamba_moe.sh"
bash "${SCRIPT_DIR}/scopes/08_moe_router.sh"
bash "${SCRIPT_DIR}/scopes/09_attn_moe_router.sh"
bash "${SCRIPT_DIR}/scopes/10_mamba_moe_router.sh"
bash "${SCRIPT_DIR}/scopes/11_attn_mamba_moe_router.sh"
bash "${SCRIPT_DIR}/scopes/12_moe_router_preprocess.sh"
bash "${SCRIPT_DIR}/scopes/13_attn_moe_router_preprocess.sh"
bash "${SCRIPT_DIR}/scopes/14_mamba_moe_router_preprocess.sh"
bash "${SCRIPT_DIR}/scopes/15_attn_mamba_moe_router_preprocess.sh"
bash "${SCRIPT_DIR}/scopes/16_mlp.sh"
