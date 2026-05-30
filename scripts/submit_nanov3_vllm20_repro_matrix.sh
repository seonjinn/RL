#!/usr/bin/env bash

set -euo pipefail

# Submit the Nano v3 Omni vLLM20 reproduction matrix used in the DynamicCP
# runtime report. Run this script from the repo root on the target Slurm
# cluster after the 8K DynamicCP canary passes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${NEMORL}"

TARGET_CLUSTER="${TARGET_CLUSTER:-cw}"
SUBMIT_SET="${SUBMIT_SET:-canary}"
DRY_RUN="${DRY_RUN:-false}"
RUN_GROUP="${RUN_GROUP:-vllm20-repro-$(date +%Y%m%d-%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemotron-omni-dynamiccp-vllm20}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
VLLM_TP="${VLLM_TP:-2}"
COMMON_EXTRA_OVERRIDES="${COMMON_EXTRA_OVERRIDES:-policy.megatron_cfg.scheduler.lr_warmup_iters=0}"
REPRO_CACHE_ROOT="${REPRO_CACHE_ROOT:-${NEMORL}/.cache_vllm20_repro}"

case "${TARGET_CLUSTER}" in
  cw|cw-dfw)
    LAUNCHER="${LAUNCHER:-scripts/nanov3_vision_rl.sh}"
    ;;
  nrt|oci-nrt)
    LAUNCHER="${LAUNCHER:-scripts/nanov3_vision_rl_nrt.sh}"
    ;;
  *)
    echo "TARGET_CLUSTER must be one of: cw, cw-dfw, nrt, oci-nrt" >&2
    exit 1
    ;;
esac

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Launcher is not executable: ${LAUNCHER}" >&2
  exit 1
fi

sanitize_decimal() {
  echo "$1" | tr '.' 'p'
}

nodes_for() {
  local tp="$1"
  local cp="$2"
  local min_nodes="$3"
  local gpus_per_node="${GPUS_PER_NODE:-8}"
  local ranks=$((tp * cp))
  local nodes=$(((ranks + gpus_per_node - 1) / gpus_per_node))
  if (( nodes < min_nodes )); then
    nodes="${min_nodes}"
  fi
  if (( nodes < 1 )); then
    nodes=1
  fi
  echo "${nodes}"
}

submit_job() {
  local label="$1"
  local seq_len="$2"
  local tp="$3"
  local cp="$4"
  local min_nodes="$5"
  local mode="$6"
  local threshold="$7"
  local mb="$8"
  local prompts="$9"
  local train_gbs="${10}"
  local steps="${11:-${MAX_STEPS}}"

  local nodes
  nodes="$(nodes_for "${tp}" "${cp}" "${min_nodes}")"

  local dcp_tag=""
  local hybrid_enabled="false"
  local force_full_cp="false"
  local threshold_env=""
  local mb_env=""
  if [[ "${mode}" == "dyncp" ]]; then
    dcp_tag="-dyncp-th${threshold}-mb$(sanitize_decimal "${mb}")"
    hybrid_enabled="true"
    threshold_env="${threshold}"
    mb_env="${mb}"
  elif [[ "${mode}" != "nohcp" ]]; then
    echo "mode must be nohcp or dyncp, got ${mode}" >&2
    exit 1
  fi

  local job_name_base="${RUN_GROUP}-${label}${dcp_tag}"
  local extra="${COMMON_EXTRA_OVERRIDES}"
  extra+=" policy.megatron_cfg.tensor_model_parallel_size=${tp}"
  extra+=" policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}"
  extra+=" grpo.num_prompts_per_step=${prompts}"
  extra+=" policy.train_global_batch_size=${train_gbs}"
  extra+=" policy.generation.vllm_cfg.max_model_len=${seq_len}"
  extra+=" policy.generation.max_new_tokens=${seq_len}"
  extra+=" checkpointing.enabled=false"
  if (( seq_len >= 65536 )); then
    extra+=" policy.logprob_chunk_size=${LONG_LOGPROB_CHUNK_SIZE:-128}"
    extra+=" policy.generation.vllm_kwargs.max_num_batched_tokens=${LONG_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
  fi

  echo "submit ${job_name_base}: seq=${seq_len} tp=${tp} cp=${cp} nodes=${nodes} mode=${mode} th=${threshold:-na} mb=${mb:-na} prompts=${prompts} gbs=${train_gbs} steps=${steps}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '  NUM_NODES=%q CP_SIZE=%q POLICY_MAX_TOTAL_SEQUENCE_LENGTH=%q HYBRID_CP_ENABLED=%q HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK=%q HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER=%q NEMO_RL_ISOLATED_CACHE_ROOT=%q EXTRA_OVERRIDES_APPEND=%q %q\n' \
      "${nodes}" "${cp}" "${seq_len}" "${hybrid_enabled}" "${threshold_env}" "${mb_env}" "${REPRO_CACHE_ROOT}" "${extra}" "${LAUNCHER}"
    return
  fi

  NUM_NODES="${nodes}" \
  JOB_NAME_BASE="${job_name_base}" \
  NEMO_RL_ISOLATED_CACHE_ROOT="${REPRO_CACHE_ROOT}" \
  CP_SIZE="${cp}" \
  POLICY_MAX_TOTAL_SEQUENCE_LENGTH="${seq_len}" \
  HYBRID_CP_ENABLED="${hybrid_enabled}" \
  HYBRID_CP_FORCE_FULL_CP="${force_full_cp}" \
  HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK="${threshold_env}" \
  HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER="${mb_env}" \
  GRPO_MAX_NUM_STEPS="${steps}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENABLED="${WANDB_ENABLED}" \
  EXTRA_OVERRIDES_APPEND="${extra}" \
  bash "${LAUNCHER}"
}

add_canary() {
  submit_job "8k-cp4" 8192 8 4 4 dyncp 2048 1.0 512 2048 2
}

add_main_speedup_rows() {
  submit_job "8k-cp8-nohcp" 8192 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"

  submit_job "16k-cp8-nohcp" 16384 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"

  submit_job "32k-cp8-nohcp" 32768 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
}

add_main_baseline_rows() {
  submit_job "8k-cp8-nohcp" 8192 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8-nohcp" 16384 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8-nohcp" 32768 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
}

add_main_dyncp_rows() {
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_rows() {
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_baseline_rows() {
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
}

add_49k_dyncp_rows() {
  submit_job "49k-cp4" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}"
}

add_long_rows() {
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"

  submit_job "128k-tp4-cp16-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"

  submit_job "256k-tp4-cp16-nohcp" 262144 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
}

add_long_baseline_rows() {
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-nohcp" 262144 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
}

add_long_dyncp_rows() {
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
}

case "${SUBMIT_SET}" in
  canary)
    add_canary
    ;;
  main)
    add_main_speedup_rows
    ;;
  main_baseline)
    add_main_baseline_rows
    ;;
  main_dyncp)
    add_main_dyncp_rows
    ;;
  49k)
    add_49k_rows
    ;;
  49k_baseline)
    add_49k_baseline_rows
    ;;
  49k_dyncp)
    add_49k_dyncp_rows
    ;;
  long)
    add_long_rows
    ;;
  long_baseline)
    add_long_baseline_rows
    ;;
  long_dyncp)
    add_long_dyncp_rows
    ;;
  baselines)
    add_main_baseline_rows
    add_49k_baseline_rows
    add_long_baseline_rows
    ;;
  dyncp)
    add_main_dyncp_rows
    add_49k_dyncp_rows
    add_long_dyncp_rows
    ;;
  all)
    add_main_speedup_rows
    add_49k_rows
    add_long_rows
    ;;
  *)
    echo "Unknown SUBMIT_SET=${SUBMIT_SET}; expected canary, main, main_baseline, main_dyncp, 49k, 49k_baseline, 49k_dyncp, long, long_baseline, long_dyncp, baselines, dyncp, all" >&2
    exit 1
    ;;
esac
