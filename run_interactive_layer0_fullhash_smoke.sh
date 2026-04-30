#!/usr/bin/env bash

set -euo pipefail

NEMORL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${NEMORL}"

export CACHE_ROOT="${CACHE_ROOT:-/lustre/fs1/portfolios/coreai/users/aroshanghias/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export NRL_NEMOTRON_VL_DEBUG="${NRL_NEMOTRON_VL_DEBUG:-1}"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMORL}/3rdparty/vllm:${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ID="${RUN_ID:-interactive-layer0-fullhash-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/results/${RUN_ID}}"
export NRL_NEMOTRON_VL_DEBUG_DIR="${NRL_NEMOTRON_VL_DEBUG_DIR:-${RESULTS_DIR}/debug}"
export NRL_NEMOTRON_VL_RUN_LABEL="${NRL_NEMOTRON_VL_RUN_LABEL:-super}"

mkdir -p "${RESULTS_DIR}" "${NRL_NEMOTRON_VL_DEBUG_DIR}"

uv run --no-sync examples/run_vlm_grpo.py \
  --config examples/configs/nanov3_vision_rl_truncated.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=1 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.tensor_model_parallel_size=1 \
  policy.megatron_cfg.expert_model_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=false \
  policy.model_name=/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-truncated \
  policy.train_global_batch_size=1 \
  policy.sequence_packing.enabled=false \
  grpo.num_prompts_per_step=1 \
  grpo.num_generations_per_prompt=1 \
  grpo.seed=42 \
  grpo.max_num_steps=1 \
  grpo.overlong_filtering=false \
  grpo.deduplicate_multimodal_data=false \
  data.train.cache_dir=/lustre/fs1/portfolios/coreai/users/aroshanghias/data/mmpr_miniscule/processed \
  policy.generation.max_new_tokens=64 \
  policy.generation.temperature=1.0 \
  policy.generation.top_p=1.0 \
  policy.generation.bad_words='[]' \
  policy.megatron_cfg.scheduler.lr_warmup_iters=0 \
  checkpointing.checkpoint_dir="${RESULTS_DIR}" \
  logger.log_dir="${RESULTS_DIR}" \
  logger.wandb_enabled=false \
  logger.wandb.project=grpo-nanov3vl \
  logger.wandb.name="${RUN_ID}" \
  2>&1 | tee "${RESULTS_DIR}/run.log"
