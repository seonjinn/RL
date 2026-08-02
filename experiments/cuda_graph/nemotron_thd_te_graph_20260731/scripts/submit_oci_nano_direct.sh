#!/bin/bash
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

set -euo pipefail

ACCOUNT=${ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEGMENT_SIZE=${SEGMENT_SIZE:-}
STEPS=${STEPS:-5}
CUDA_GRAPH_MODULES=${CUDA_GRAPH_MODULES:-attn,mamba,moe_router}
CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-transformer_engine}
NVTE_DEBUG=${NVTE_DEBUG:-0}
NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd "${script_dir}/../../../.." && pwd -P)
SOURCE_ROOT=${SOURCE_ROOT:-${repository_root}}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260801_107e892b_20260801_5768359.sqsh}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/sna-cg-study/nemotron-thd-te-graph-20260801/runs/direct}
if [[ "${SOURCE_ROOT}" == /home/* ]]; then
  MOUNTS=${MOUNTS:-/lustre:/lustre,/home:/home}
else
  MOUNTS=${MOUNTS:-/lustre:/lustre}
fi

if [[ "${CUDA_GRAPH_IMPL}" == "none" ]]; then
  scope_name=baseline
  cuda_graph_overrides="++policy.megatron_cfg.cuda_graph_impl=none"
else
  scope_name=${CUDA_GRAPH_MODULES//,/-}
  cuda_graph_overrides="++policy.megatron_cfg.cuda_graph_impl=${CUDA_GRAPH_IMPL} ++policy.megatron_cfg.cuda_graph_modules=[${CUDA_GRAPH_MODULES}] ++policy.megatron_cfg.cuda_graph_warmup_steps=3 ++policy.megatron_cfg.thd_max_packed_sequences=16 ++policy.megatron_cfg.attention_backend=fused"
fi

debug_suffix=""
if [[ "${NVTE_DEBUG}" == "1" ]]; then
  [[ "${NVTE_DEBUG_LEVEL}" == "2" ]] || {
    echo "NVTE_DEBUG_LEVEL must be 2 when NVTE_DEBUG=1" >&2
    exit 2
  }
  debug_suffix=-nvte-debug2
fi

RUN_NAME="nano-${scope_name}-${STEPS}step-alltoall${debug_suffix}-${RUN_TAG}"
RUN_DIR=${EXPERIMENT_ROOT}/${RUN_NAME}
LOG_DIR=${RUN_DIR}/exp_logs
mkdir -p "${RUN_DIR}"

COMMAND="env NRL_FORCE_REBUILD_VENVS=true NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS=100a NVTE_DEBUG=${NVTE_DEBUG} NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL} NCCL_GRAPH_REGISTER=0 CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_DIR=${RUN_DIR}/wandb uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml \
  grpo.max_num_steps=${STEPS} \
  checkpointing.enabled=false \
  policy.sequence_packing.enabled=true \
  policy.dynamic_batching.enabled=false \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  logger.log_dir=${LOG_DIR} \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.wandb.project=sna-cg-study \
  logger.wandb.name=${RUN_NAME} \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  ${cuda_graph_overrides} \
  ++policy.generation.vllm_kwargs.moe_backend=triton"

export BASE_LOG_DIR=${RUN_DIR}
export COMMAND CONTAINER
export GPUS_PER_NODE
export MOUNTS

sbatch_args=(
  --nodes="${NUM_NODES}"
  --account="${ACCOUNT}"
  --job-name="cg-${RUN_NAME}"
  --partition="${PARTITION}"
  --time=01:00:00
  --gres="gpu:${GPUS_PER_NODE}"
  --output="${RUN_DIR}/slurm-%j.log"
)
if [[ -n "${SEGMENT_SIZE}" ]]; then
  sbatch_args+=(--segment="${SEGMENT_SIZE}")
fi

cd "${SOURCE_ROOT}"
sbatch --test-only "${sbatch_args[@]}" ray.sub
job_id=$(sbatch --parsable "${sbatch_args[@]}" ray.sub)

printf 'SLURM_JOB_ID=%s\n' "${job_id}"
printf 'RUN_NAME=%s\n' "${RUN_NAME}"
printf 'RUN_DIR=%s\n' "${RUN_DIR}"
