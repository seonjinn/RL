#!/bin/bash

set -euo pipefail

: "${EXPERIMENT_OUTPUT_DIR:?Set EXPERIMENT_OUTPUT_DIR to a shared filesystem path}"

MAX_NUM_STEPS=${MAX_NUM_STEPS:-20}
DISPATCHER_MODE=${DISPATCHER_MODE:-hybridep}
LOGPROB_BATCH_SIZE=${LOGPROB_BATCH_SIZE:-2}
LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK_SIZE:-null}
RUN_NAME=${RUN_NAME:-qwen3-30ba3b-4n8g-${DISPATCHER_MODE}-lpb${LOGPROB_BATCH_SIZE}-lpc${LOGPROB_CHUNK_SIZE}}
WANDB_ENABLED=${WANDB_ENABLED:-true}

export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export USE_NIXL=0

declare -a dispatcher_overrides
case "${DISPATCHER_MODE}" in
  alltoall)
    unset HYBRID_EP_MULTINODE USE_MNNVL
    dispatcher_overrides=(
      "policy.megatron_cfg.moe_token_dispatcher_type=alltoall"
      "~policy.megatron_cfg.moe_flex_dispatcher_backend"
      "~policy.megatron_cfg.moe_hybridep_num_sms"
      "~policy.megatron_cfg.moe_hybridep_prepad_packed_inputs"
      "~policy.megatron_cfg.env_vars.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
      "~policy.megatron_cfg.env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"
      "~policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE"
      "~policy.megatron_cfg.env_vars.USE_MNNVL"
    )
    ;;
  hybridep)
    export HYBRID_EP_MULTINODE=0
    export USE_MNNVL=0
    dispatcher_overrides=(
      "policy.megatron_cfg.moe_token_dispatcher_type=flex"
      "policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep"
    )
    ;;
  *)
    echo "Unsupported DISPATCHER_MODE: ${DISPATCHER_MODE}" >&2
    exit 2
    ;;
esac

declare -a chunk_overrides=("policy.megatron_cfg.defer_fp32_logits=false")
if [[ "${LOGPROB_CHUNK_SIZE}" != "null" ]]; then
  chunk_overrides=("policy.megatron_cfg.defer_fp32_logits=true")
fi

UV_NO_SYNC=1 uv run env -u UV_NO_SYNC -u UV_FROZEN python examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml \
  grpo.max_num_steps="${MAX_NUM_STEPS}" \
  policy.logprob_batch_size="${LOGPROB_BATCH_SIZE}" \
  policy.logprob_chunk_size="${LOGPROB_CHUNK_SIZE}" \
  checkpointing.enabled=false \
  logger.log_dir="${EXPERIMENT_OUTPUT_DIR}/training" \
  logger.wandb_enabled="${WANDB_ENABLED}" \
  logger.wandb.name="${RUN_NAME}" \
  "${chunk_overrides[@]}" \
  "${dispatcher_overrides[@]}"
