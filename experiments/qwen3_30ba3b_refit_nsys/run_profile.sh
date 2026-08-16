#!/usr/bin/env bash

set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
SEGMENT_SIZE=2
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=180
# ===== END CONFIG =====

: "${REFIT_NSYS_ARM:?set REFIT_NSYS_ARM to bf16 or mxfp8}"

case "${REFIT_NSYS_ARM}" in
  bf16 | mxfp8) ;;
  *)
    echo "REFIT_NSYS_ARM must be bf16 or mxfp8, got ${REFIT_NSYS_ARM}" >&2
    exit 2
    ;;
esac

RESULTS_ROOT=${REFIT_NSYS_RESULTS_ROOT:-${PWD}/results/qwen3_30ba3b_refit_nsys}
RUN_ID=${SLURM_JOB_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${RESULTS_ROOT}/${REFIT_NSYS_ARM}/${RUN_ID}
mkdir -p "${RUN_ROOT}"

export NRL_REFIT_NVTX_DETAIL=1
export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker,vllm_generation_worker"
export NRL_NSYS_PROFILE_STEP_RANGE="2:3"
export NRL_NSYS_EXTRA_OPTIONS='{"cuda-memory-usage":"true","cpuctxsw":"none"}'
export RAY_LOG_SYNC_FREQUENCY=30

CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml

common_args=(
  --config "${CONFIG}"
  grpo.max_num_steps=4
  grpo.async_grpo.enabled=false
  grpo.async_grpo.in_flight_weight_updates=false
  policy.megatron_cfg.optimizer.lr=0.0
  policy.megatron_cfg.optimizer.min_lr=0.0
  policy.megatron_cfg.scheduler.lr_warmup_init=0.0
  checkpointing.enabled=false
  logger.log_dir="${RUN_ROOT}/logs"
  logger.wandb_enabled=false
  logger.tensorboard_enabled=true
  logger.monitor_gpus=false
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
  policy.generation.refit_transport=nccl_reshard
)

precision_args=()
scope_lock_args=()
if [[ "${REFIT_NSYS_ARM}" == "bf16" ]]; then
  precision_args=(
    policy.generation.vllm_cfg.precision=bfloat16
    '~policy.generation.vllm_cfg.is_mx'
    '~policy.generation.vllm_cfg.quantization_ignored_layer_kws'
  )
else
  scope_lock_args=(
    '~policy.generation.vllm_cfg.quantization_ignored_layer_kws'
    '++policy.generation.vllm_cfg.quantization_ignore_patterns=[model.layers.*.self_attn.*,model.layers.*.mlp.gate,lm_head]'
  )
fi

{
  echo "arm=${REFIT_NSYS_ARM}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "container=${CONTAINER:-unknown}"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "slurm_nodes=${SLURM_JOB_NODELIST:-unknown}"
  echo "profile_step_range=${NRL_NSYS_PROFILE_STEP_RANGE}"
  echo "cuda_graph=enabled"
  echo "moe_backend=flashinfer_trtllm"
  echo "refit_transport=nccl_reshard"
} > "${RUN_ROOT}/provenance.txt"

uv run examples/run_grpo.py \
  "${common_args[@]}" \
  "${precision_args[@]}" \
  "$@" \
  "${scope_lock_args[@]}" \
  2>&1 | tee "${RUN_ROOT}/run.log"
