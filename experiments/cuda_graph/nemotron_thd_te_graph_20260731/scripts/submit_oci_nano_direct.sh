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
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
STEPS=${STEPS:-5}
CUDA_GRAPH_MODULES=${CUDA_GRAPH_MODULES:-attn,mamba,moe_router}
CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-transformer_engine}
NVTE_DEBUG=${NVTE_DEBUG:-0}
NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
MOE_FLEX_DISPATCHER_BACKEND=${MOE_FLEX_DISPATCHER_BACKEND:-}
HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-}
EXCLUDE=${EXCLUDE:-}
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

moe_preprocess_enabled=false
if [[ "${CUDA_GRAPH_IMPL}" != "none" && \
      ",${CUDA_GRAPH_MODULES}," == *",moe_preprocess,"* ]]; then
  if [[ ",${CUDA_GRAPH_MODULES}," != *",moe_router,"* ]]; then
    echo "CUDA_GRAPH_MODULES moe_preprocess requires moe_router" >&2
    exit 2
  fi
  moe_preprocess_enabled=true
fi

if [[ "${CUDA_GRAPH_IMPL}" == "none" ]]; then
  scope_name=baseline
  cuda_graph_overrides="++policy.megatron_cfg.cuda_graph_impl=none ++policy.megatron_cfg.attention_backend=fused"
else
  scope_name=${CUDA_GRAPH_MODULES//,/-}
  cuda_graph_overrides="++policy.megatron_cfg.cuda_graph_impl=${CUDA_GRAPH_IMPL} ++policy.megatron_cfg.cuda_graph_modules=[${CUDA_GRAPH_MODULES}] ++policy.megatron_cfg.cuda_graph_warmup_steps=3 ++policy.megatron_cfg.thd_max_packed_sequences=16 ++policy.megatron_cfg.attention_backend=fused"
fi

case "${MOE_TOKEN_DISPATCHER_TYPE}" in
  alltoall)
    [[ -z "${MOE_FLEX_DISPATCHER_BACKEND}" ]] || {
      echo "MOE_FLEX_DISPATCHER_BACKEND must be empty for alltoall" >&2
      exit 2
    }
    dispatcher_name=alltoall
    dispatcher_env=""
    dispatcher_overrides="policy.megatron_cfg.moe_token_dispatcher_type=alltoall"
    ;;
  flex)
    [[ "${MOE_FLEX_DISPATCHER_BACKEND}" == "hybridep" ]] || {
      echo "Direct flex smoke currently requires MOE_FLEX_DISPATCHER_BACKEND=hybridep" >&2
      exit 2
    }
    [[ "${HYBRID_EP_RANKS_PER_NVLINK_DOMAIN}" =~ ^[1-9][0-9]*$ ]] || {
      echo "HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must be a positive integer" >&2
      exit 2
    }
    dispatcher_name=hybridep
    dispatcher_env="USE_MNNVL=1 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${HYBRID_EP_RANKS_PER_NVLINK_DOMAIN}"
    hybridep_pad_uneven_dispatch_inputs=true
    if [[ "${moe_preprocess_enabled}" == "true" ]]; then
      hybridep_pad_uneven_dispatch_inputs=false
    fi
    dispatcher_overrides="policy.megatron_cfg.moe_token_dispatcher_type=flex ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep ++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=${hybridep_pad_uneven_dispatch_inputs}"
    ;;
  *)
    echo "MOE_TOKEN_DISPATCHER_TYPE must be alltoall or flex" >&2
    exit 2
    ;;
esac

debug_suffix=""
if [[ "${NVTE_DEBUG}" == "1" ]]; then
  [[ "${NVTE_DEBUG_LEVEL}" == "2" ]] || {
    echo "NVTE_DEBUG_LEVEL must be 2 when NVTE_DEBUG=1" >&2
    exit 2
  }
  debug_suffix=-nvte-debug2
fi

RUN_NAME="nano-${scope_name}-${STEPS}step-${dispatcher_name}${debug_suffix}-${RUN_TAG}"
RUN_DIR=${EXPERIMENT_ROOT}/${RUN_NAME}
LOG_DIR=${RUN_DIR}/exp_logs
mkdir -p "${RUN_DIR}"

COMMAND="env NRL_FORCE_REBUILD_VENVS=true NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS=100a NVTE_DEBUG=${NVTE_DEBUG} NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL} NCCL_GRAPH_REGISTER=0 CUDA_DEVICE_MAX_CONNECTIONS=1 ${dispatcher_env} WANDB_DIR=${RUN_DIR}/wandb uv run examples/run_grpo.py \
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
  ${dispatcher_overrides} \
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
  --time="${TIME_LIMIT}"
  --gres="gpu:${GPUS_PER_NODE}"
  --output="${RUN_DIR}/slurm-%j.log"
)
if [[ -n "${SEGMENT_SIZE}" ]]; then
  sbatch_args+=(--segment="${SEGMENT_SIZE}")
fi
if [[ -n "${EXCLUDE}" ]]; then
  sbatch_args+=(--exclude="${EXCLUDE}")
fi

cd "${SOURCE_ROOT}"
sbatch --test-only "${sbatch_args[@]}" ray.sub
job_id=$(sbatch --parsable "${sbatch_args[@]}" ray.sub)

printf 'SLURM_JOB_ID=%s\n' "${job_id}"
printf 'RUN_NAME=%s\n' "${RUN_NAME}"
printf 'RUN_DIR=%s\n' "${RUN_DIR}"
