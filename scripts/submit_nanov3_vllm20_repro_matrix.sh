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
GRPO_SEED="${GRPO_SEED:-42}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-}"
GENERATION_MIN_NEW_TOKENS="${GENERATION_MIN_NEW_TOKENS:-}"
WALLCLOCK_GENERATION_MAX_NEW_TOKENS="${WALLCLOCK_GENERATION_MAX_NEW_TOKENS:-8192}"
COMMON_EXTRA_OVERRIDES="${COMMON_EXTRA_OVERRIDES:-policy.megatron_cfg.scheduler.lr_warmup_iters=0 policy.generation.vllm_cfg.enforce_eager=false ++policy.generation.vllm_cfg.enable_prefix_caching=true}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE_LONG="${VLLM_DISABLE_CUSTOM_ALL_REDUCE_LONG:-true}"
CAP_MAX_TOKENS_TO_CONTEXT="${CAP_MAX_TOKENS_TO_CONTEXT:-auto}"
MCORE_DISABLE_TORCH_COMPILE_JIT="${MCORE_DISABLE_TORCH_COMPILE_JIT:-false}"
REPRO_CACHE_ROOT="${REPRO_CACHE_ROOT:-${NEMORL}/.cache_vllm20_repro}"
ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-auto}"

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

if [[ "${ENABLE_FLASHINFER_AUTOTUNE}" == "auto" ]]; then
  if [[ "${TARGET_CLUSTER}" == "nrt" || "${TARGET_CLUSTER}" == "oci-nrt" ]]; then
    # On OCI-NRT, FlashInfer MoE autotune can spend tens of minutes profiling
    # failed tactics before Step 1, which trips the cluster idle reaper. Keep
    # the production CW default unless the caller opts out explicitly.
    ENABLE_FLASHINFER_AUTOTUNE="false"
  else
    ENABLE_FLASHINFER_AUTOTUNE="true"
  fi
fi

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
  local max_new_tokens_override="${12:-}"
  local cap_max_tokens_to_context="${13:-${CAP_MAX_TOKENS_TO_CONTEXT}}"
  local policy_ep="${14:-}"

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
  local max_new_tokens="${max_new_tokens_override:-${GENERATION_MAX_NEW_TOKENS:-${seq_len}}}"
  if [[ "${cap_max_tokens_to_context}" == "auto" ]]; then
    if [[ "${max_new_tokens}" == "${seq_len}" ]]; then
      cap_max_tokens_to_context=1
    else
      cap_max_tokens_to_context=0
    fi
  fi
  extra+=" grpo.seed=${GRPO_SEED}"
  extra+=" grpo.max_num_steps=${steps}"
  extra+=" policy.megatron_cfg.tensor_model_parallel_size=${tp}"
  if [[ -n "${policy_ep}" ]]; then
    extra+=" policy.megatron_cfg.expert_model_parallel_size=${policy_ep}"
  fi
  extra+=" policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}"
  extra+=" grpo.num_prompts_per_step=${prompts}"
  extra+=" policy.train_global_batch_size=${train_gbs}"
  extra+=" policy.generation.vllm_cfg.max_model_len=${seq_len}"
  extra+=" policy.generation.max_new_tokens=${max_new_tokens}"
  # The launcher already maps GENERATION_MIN_NEW_TOKENS to the Hydra override.
  # Duplicating it here creates two policy.generation.min_new_tokens entries
  # and Hydra rejects the second append.
  extra+=" checkpointing.enabled=false"
  extra+=" checkpointing.checkpoint_must_save_by=null"
  if (( seq_len >= 65536 )); then
    extra+=" policy.logprob_chunk_size=${LONG_LOGPROB_CHUNK_SIZE:-128}"
    extra+=" policy.generation.vllm_kwargs.max_num_batched_tokens=${LONG_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
    if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE_LONG}" == "true" && "${extra}" != *"disable_custom_all_reduce"* ]]; then
      extra+=" ++policy.generation.vllm_kwargs.disable_custom_all_reduce=true"
    fi
  fi

  echo "submit ${job_name_base}: seq=${seq_len} max_new=${max_new_tokens} cap_ctx=${cap_max_tokens_to_context} tp=${tp} cp=${cp} nodes=${nodes} mode=${mode} th=${threshold:-na} mb=${mb:-na} prompts=${prompts} gbs=${train_gbs} steps=${steps}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '  NUM_NODES=%q CP_SIZE=%q POLICY_MAX_TOTAL_SEQUENCE_LENGTH=%q HYBRID_CP_ENABLED=%q HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK=%q HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER=%q NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT=%q MCORE_DISABLE_TORCH_COMPILE_JIT=%q ENABLE_FLASHINFER_AUTOTUNE=%q NEMO_RL_ISOLATED_CACHE_ROOT=%q EXTRA_OVERRIDES_APPEND=%q %q\n' \
      "${nodes}" "${cp}" "${seq_len}" "${hybrid_enabled}" "${threshold_env}" "${mb_env}" "${cap_max_tokens_to_context}" "${MCORE_DISABLE_TORCH_COMPILE_JIT}" "${ENABLE_FLASHINFER_AUTOTUNE}" "${REPRO_CACHE_ROOT}" "${extra}" "${LAUNCHER}"
    return
  fi

  NUM_NODES="${nodes}" \
  JOB_NAME_BASE="${job_name_base}" \
  NEMO_RL_ISOLATED_CACHE_ROOT="${REPRO_CACHE_ROOT}" \
  CP_SIZE="${cp}" \
  POLICY_EP="${policy_ep}" \
  POLICY_MAX_TOTAL_SEQUENCE_LENGTH="${seq_len}" \
  HYBRID_CP_ENABLED="${hybrid_enabled}" \
  HYBRID_CP_FORCE_FULL_CP="${force_full_cp}" \
  HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK="${threshold_env}" \
  HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER="${mb_env}" \
  NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT="${cap_max_tokens_to_context}" \
  MCORE_DISABLE_TORCH_COMPILE_JIT="${MCORE_DISABLE_TORCH_COMPILE_JIT}" \
  ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE}" \
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

add_8k_baseline_rows() {
  submit_job "8k-cp8-nohcp" 8192 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
}

add_main_dyncp_rows() {
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
}

add_nrt_cancelled_short_recovery_rows() {
  # Rerun only the short rows cancelled by the NRT scheduler before Step 1.
  submit_job "8k-cp8-nohcp" 8192 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8-nohcp" 32768 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
}

add_nrt_cancelled_long_recovery_rows() {
  # Rerun only the long rows cancelled by the NRT scheduler before Step 1.
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
}

add_nrt_reaper_recovery_exact_rows() {
  # Exact retry set for the NRT 20260601 idle-reaper cancellations. These
  # exclude rows that were still running from the original full matrix.
  submit_job "8k-cp8-nohcp" 8192 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8-nohcp" 16384 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8-nohcp" 32768 8 8 8 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  add_256k_safe_baseline_rows
}

add_nrt_missing_dyncp_recovery_rows() {
  # DynamicCP-only rows missing valid post-warmup NRT comparisons.
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8" 16384 8 8 8 dyncp 4096 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8" 32768 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
}

add_nrt_strict_missing_pair_rows() {
  # Counterparts missing from the e1d3890/autotune-off NRT strict matrix.
  # These avoid mixing old FlashInfer-autotune rows with the new rerun.
  submit_job "8k-cp8" 8192 8 8 8 dyncp 2048 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  add_256k_safe_dyncp_rows
}

add_49k_rows() {
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_baseline_rows() {
  submit_job "49k-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}"
}

add_49k_dyncp_rows() {
  submit_job "49k-cp4" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_dyncp_threshold_sweep_rows() {
  # Threshold is max sequence length per local CP rank. For CP=4 and 49K
  # sequences, the strict EP16/TP8 topology still floors local CP at 2. These
  # rows are useful for EP16 threshold sensitivity, not for local-CP1 testing.
  submit_job "49k-cp4-th12288" 49152 8 4 4 dyncp 12288 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th16384" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th24576" 49152 8 4 4 dyncp 24576 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th32768" 49152 8 4 4 dyncp 32768 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th49152" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_cp1_diagnostic_rows() {
  submit_job "49k-tp8-ep8-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
  submit_job "49k-tp8-ep8-cp4" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
}

add_49k_cp1_threshold_sweep_rows() {
  submit_job "49k-tp8-ep8-cp4-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
  submit_job "49k-tp8-ep8-cp4-th24576" 49152 8 4 4 dyncp 24576 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
  submit_job "49k-tp8-ep8-cp4-th32768" 49152 8 4 4 dyncp 32768 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
  submit_job "49k-tp8-ep8-cp4-th49152" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
}

add_49k_cp1_threshold_missing_rows() {
  submit_job "49k-tp8-ep8-cp4-th24576" 49152 8 4 4 dyncp 24576 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
  submit_job "49k-tp8-ep8-cp4-th32768" 49152 8 4 4 dyncp 32768 1.0 512 2048 "${MAX_STEPS}" "" "${CAP_MAX_TOKENS_TO_CONTEXT}" 8
}

add_long_rows() {
  add_64k_rows
  add_128k_rows
  add_256k_safe_rows
}

add_64k_rows() {
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
}

add_64k_baseline_rows() {
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
}

add_128k_rows() {
  submit_job "128k-tp4-cp16-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
}

add_long_no256_rows() {
  add_64k_rows
  add_128k_rows
}

add_256k_safe_rows() {
  # Keep the 256K model/context cap, but avoid deriving one full 256K packed
  # train microbatch. Earlier TP4/CP16 256K probes OOMed before useful metrics
  # with the default cap-derived packing budget.
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-nohcp-mb128k" 262144 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-mb128k" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_256k_safe_baseline_rows() {
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-nohcp-mb128k" 262144 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_256k_safe_dyncp_rows() {
  # DynamicCP-only variant used when a matching no-HCP 256K safe baseline is
  # already running or completed.
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-mb128k" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_long_baseline_rows() {
  submit_job "64k-tp2-cp32-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}"
  add_256k_safe_baseline_rows
}

add_long_dyncp_rows() {
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  add_256k_safe_dyncp_rows
}

add_long_dyncp_safe_rows() {
  submit_job "64k-tp2-cp32" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  add_256k_safe_dyncp_rows
}

add_64k_dyncp_threshold_sweep_rows() {
  submit_job "64k-tp2-cp32-th2048" 65536 2 32 8 dyncp 2048 1.0 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32-th4096" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}"
}

add_128k_dyncp_threshold_sweep_rows() {
  submit_job "128k-tp4-cp16-th8192" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th16384" 131072 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th32768" 131072 4 16 8 dyncp 32768 1.0 128 512 "${MAX_STEPS}"
}

add_256k_dyncp_threshold_sweep_rows() {
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-th16384" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-th32768" 262144 4 16 8 dyncp 32768 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_long_dyncp_threshold_sweep_rows() {
  add_64k_dyncp_threshold_sweep_rows
  add_128k_dyncp_threshold_sweep_rows
  add_256k_dyncp_threshold_sweep_rows
}

add_step_time_128k_rows() {
  submit_job "128k-stepcap-nohcp" 131072 4 16 8 nohcp "" "" 128 512 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
  submit_job "128k-stepcap" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
}

add_step_time_64k_rows() {
  submit_job "64k-stepcap-nohcp" 65536 2 32 8 nohcp "" "" 128 512 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
  submit_job "64k-stepcap" 65536 2 32 8 dyncp 4096 1.0 128 512 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
}

add_step_time_49k_rows() {
  submit_job "49k-stepcap-nohcp" 49152 8 4 4 nohcp "" "" 512 2048 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
  submit_job "49k-stepcap" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}" "${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}"
}

add_step_time_rows() {
  add_step_time_49k_rows
  add_step_time_64k_rows
  add_step_time_128k_rows
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
  8k_baseline)
    add_8k_baseline_rows
    ;;
  main_dyncp)
    add_main_dyncp_rows
    ;;
  nrt_cancelled_short_recovery)
    add_nrt_cancelled_short_recovery_rows
    ;;
  nrt_cancelled_long_recovery)
    add_nrt_cancelled_long_recovery_rows
    ;;
  nrt_reaper_recovery_exact)
    add_nrt_reaper_recovery_exact_rows
    ;;
  nrt_missing_dyncp_recovery)
    add_nrt_missing_dyncp_recovery_rows
    ;;
  nrt_strict_missing_pairs)
    add_nrt_strict_missing_pair_rows
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
  49k_dyncp_threshold_sweep)
    add_49k_dyncp_threshold_sweep_rows
    ;;
  49k_cp1_diagnostic)
    add_49k_cp1_diagnostic_rows
    ;;
  49k_cp1_threshold_sweep)
    add_49k_cp1_threshold_sweep_rows
    ;;
  49k_cp1_threshold_missing)
    add_49k_cp1_threshold_missing_rows
    ;;
  long)
    add_long_rows
    ;;
  long_no256)
    add_long_no256_rows
    ;;
  64k)
    add_64k_rows
    ;;
  64k_baseline)
    add_64k_baseline_rows
    ;;
  128k)
    add_128k_rows
    ;;
  256k_safe)
    add_256k_safe_rows
    ;;
  256k_safe_baseline)
    add_256k_safe_baseline_rows
    ;;
  256k_safe_dyncp)
    add_256k_safe_dyncp_rows
    ;;
  long_baseline)
    add_long_baseline_rows
    ;;
  long_dyncp)
    add_long_dyncp_rows
    ;;
  long_dyncp_safe)
    add_long_dyncp_safe_rows
    ;;
  64k_dyncp_threshold_sweep)
    add_64k_dyncp_threshold_sweep_rows
    ;;
  128k_dyncp_threshold_sweep)
    add_128k_dyncp_threshold_sweep_rows
    ;;
  256k_dyncp_threshold_sweep)
    add_256k_dyncp_threshold_sweep_rows
    ;;
  long_dyncp_threshold_sweep)
    add_long_dyncp_threshold_sweep_rows
    ;;
  step_time_128k)
    add_step_time_128k_rows
    ;;
  step_time_64k)
    add_step_time_64k_rows
    ;;
  step_time_49k)
    add_step_time_49k_rows
    ;;
  step_time)
    add_step_time_rows
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
    echo "Unknown SUBMIT_SET=${SUBMIT_SET}; expected canary, main, main_baseline, 8k_baseline, main_dyncp, nrt_cancelled_short_recovery, nrt_cancelled_long_recovery, nrt_reaper_recovery_exact, nrt_missing_dyncp_recovery, nrt_strict_missing_pairs, 49k, 49k_baseline, 49k_dyncp, 49k_dyncp_threshold_sweep, 49k_cp1_diagnostic, 49k_cp1_threshold_sweep, 49k_cp1_threshold_missing, long, long_no256, 64k, 64k_baseline, 128k, 256k_safe, 256k_safe_baseline, 256k_safe_dyncp, long_baseline, long_dyncp, long_dyncp_safe, 64k_dyncp_threshold_sweep, 128k_dyncp_threshold_sweep, 256k_dyncp_threshold_sweep, long_dyncp_threshold_sweep, step_time_49k, step_time_64k, step_time_128k, step_time, baselines, dyncp, all" >&2
    exit 1
    ;;
esac
