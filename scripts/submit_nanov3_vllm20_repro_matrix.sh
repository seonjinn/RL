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
CANARY_PROMPTS_PER_STEP="${CANARY_PROMPTS_PER_STEP:-64}"
CANARY_TRAIN_GBS="${CANARY_TRAIN_GBS:-512}"
CANARY_STEPS="${CANARY_STEPS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sna-nemotron-omni-dynamiccp-vllm20}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
POLICY_EP="${POLICY_EP:-8}"
GRPO_SEED="${GRPO_SEED:-42}"
USER_SET_GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS+x}"
GENERATION_MIN_NEW_TOKENS="${GENERATION_MIN_NEW_TOKENS:-}"
WALLCLOCK_GENERATION_MAX_NEW_TOKENS="${WALLCLOCK_GENERATION_MAX_NEW_TOKENS:-8192}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-${WALLCLOCK_GENERATION_MAX_NEW_TOKENS}}"
FORCE_ACTUAL_LONG_GENERATION="${FORCE_ACTUAL_LONG_GENERATION:-false}"
COMMON_EXTRA_OVERRIDES="${COMMON_EXTRA_OVERRIDES:-policy.megatron_cfg.scheduler.lr_warmup_iters=0 ++policy.megatron_cfg.distributed_timeout_minutes=60 ++policy.megatron_cfg.distributed_timeout_seconds_after_init=3600 policy.generation.vllm_cfg.enforce_eager=false ++policy.generation.vllm_cfg.enable_prefix_caching=true}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE_LONG="${VLLM_DISABLE_CUSTOM_ALL_REDUCE_LONG:-true}"
CAP_MAX_TOKENS_TO_CONTEXT="${CAP_MAX_TOKENS_TO_CONTEXT:-auto}"
MCORE_DISABLE_TORCH_COMPILE_JIT="${MCORE_DISABLE_TORCH_COMPILE_JIT:-false}"
SNAPSHOT_CODE="${SNAPSHOT_CODE:-1}"
REPRO_CACHE_ROOT="${REPRO_CACHE_ROOT:-${NEMORL}/.cache_vllm20_repro}"
ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-auto}"
SHARED_MEGATRON_CHECKPOINT_DIR="${SHARED_MEGATRON_CHECKPOINT_DIR:-${REPRO_CACHE_ROOT}/nemo_rl}"
SERIALIZE_SUBMISSIONS="${SERIALIZE_SUBMISSIONS:-true}"
MATRIX_BASE_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
LAST_SUBMITTED_JOB_ID=""

case "${TARGET_CLUSTER}" in
  cw|cw-dfw)
    LAUNCHER="${LAUNCHER:-scripts/nanov3_vision_rl.sh}"
    DEFAULT_MODEL_NAME="/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-0303/step_400"
    DEFAULT_CACHE_DIR="/lustre/fs1/portfolios/coreai/users/aroshanghias/data/mmpr_tiny/processed"
    ;;
  nrt|oci-nrt)
    LAUNCHER="${LAUNCHER:-scripts/nanov3_vision_rl_nrt.sh}"
    DEFAULT_MODEL_NAME="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/checkpoints/mpo-nanov3omni-mmpr-nanov2-filtered-conv3d-0303/step_400"
    DEFAULT_CACHE_DIR="${NEMORL}/.cache/mmpr_tiny"
    ;;
  *)
    echo "TARGET_CLUSTER must be one of: cw, cw-dfw, nrt, oci-nrt" >&2
    exit 1
    ;;
esac

if [[ "${ENABLE_FLASHINFER_AUTOTUNE}" == "auto" ]]; then
  # Keep CW/NRT comparison rows config-matched by default. Set
  # ENABLE_FLASHINFER_AUTOTUNE=true only for an explicit tuning study.
  ENABLE_FLASHINFER_AUTOTUNE="false"
fi

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Launcher is not executable: ${LAUNCHER}" >&2
  exit 1
fi

MODEL_NAME="${IMAGE_GRPO_MODEL_NAME:-${MODEL_NAME:-${DEFAULT_MODEL_NAME}}}"
CACHE_DIR="${IMAGE_GRPO_CACHE_DIR:-${CACHE_DIR:-${DEFAULT_CACHE_DIR}}}"

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
  local job_cache_root="${REPRO_CACHE_ROOT}/${job_name_base}"
  local extra="${COMMON_EXTRA_OVERRIDES}"
  local max_new_tokens
  if [[ "${FORCE_ACTUAL_LONG_GENERATION}" == "true" && -z "${max_new_tokens_override}" && -z "${USER_SET_GENERATION_MAX_NEW_TOKENS}" ]]; then
    max_new_tokens="${seq_len}"
  else
    max_new_tokens="${max_new_tokens_override:-${GENERATION_MAX_NEW_TOKENS}}"
  fi
  if [[ ! "${max_new_tokens}" =~ ^[0-9]+$ ]] || (( max_new_tokens < 1 )); then
    echo "max_new_tokens must be a positive integer, got ${max_new_tokens}" >&2
    exit 1
  fi
  if [[ "${FORCE_ACTUAL_LONG_GENERATION}" != "true" && "${GENERATION_MIN_NEW_TOKENS}" =~ ^[0-9]+$ ]] && (( GENERATION_MIN_NEW_TOKENS > max_new_tokens )); then
    echo "GENERATION_MIN_NEW_TOKENS=${GENERATION_MIN_NEW_TOKENS} exceeds capped max_new_tokens=${max_new_tokens}. Set FORCE_ACTUAL_LONG_GENERATION=true only for forced-length diagnostics." >&2
    exit 1
  fi
  if (( max_new_tokens > seq_len )); then
    max_new_tokens="${seq_len}"
  fi
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
  extra+=" policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP}"
  extra+=" policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}"
  if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
    extra+=" policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
  fi
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

  local dependency="${MATRIX_BASE_DEPENDENCY}"
  if [[ "${SERIALIZE_SUBMISSIONS}" == "true" && -n "${LAST_SUBMITTED_JOB_ID}" ]]; then
    if [[ -n "${dependency}" ]]; then
      dependency="${dependency},afterany:${LAST_SUBMITTED_JOB_ID}"
    else
      dependency="afterany:${LAST_SUBMITTED_JOB_ID}"
    fi
  fi

  echo "submit ${job_name_base}: seq=${seq_len} max_new=${max_new_tokens} force_actual_long=${FORCE_ACTUAL_LONG_GENERATION} cap_ctx=${cap_max_tokens_to_context} tp=${tp} cp=${cp} nodes=${nodes} mode=${mode} th=${threshold:-na} mb=${mb:-na} prompts=${prompts} gbs=${train_gbs} steps=${steps} serialize=${SERIALIZE_SUBMISSIONS} dependency=${dependency:-none}"
  echo "  model=${MODEL_NAME}"
  echo "  cache=${CACHE_DIR}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '  MODEL_NAME=%q CACHE_DIR=%q IMAGE_GRPO_MODEL_NAME=%q IMAGE_GRPO_CACHE_DIR=%q NUM_NODES=%q CP_SIZE=%q POLICY_EP=%q VLLM_GPU_MEMORY_UTILIZATION=%q SNAPSHOT_CODE=%q POLICY_MAX_TOTAL_SEQUENCE_LENGTH=%q HYBRID_CP_ENABLED=%q HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK=%q HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER=%q NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT=%q MCORE_DISABLE_TORCH_COMPILE_JIT=%q ENABLE_FLASHINFER_AUTOTUNE=%q NEMO_RL_ISOLATED_CACHE_ROOT=%q NEMO_RL_SHARED_MEGATRON_CHECKPOINT_DIR=%q GENERATION_MIN_NEW_TOKENS=%q EXTRA_OVERRIDES_APPEND=%q %q\n' \
      "${MODEL_NAME}" "${CACHE_DIR}" "${MODEL_NAME}" "${CACHE_DIR}" "${nodes}" "${cp}" "${POLICY_EP}" "${VLLM_GPU_MEMORY_UTILIZATION}" "${SNAPSHOT_CODE}" "${seq_len}" "${hybrid_enabled}" "${threshold_env}" "${mb_env}" "${cap_max_tokens_to_context}" "${MCORE_DISABLE_TORCH_COMPILE_JIT}" "${ENABLE_FLASHINFER_AUTOTUNE}" "${job_cache_root}" "${SHARED_MEGATRON_CHECKPOINT_DIR}" "${GENERATION_MIN_NEW_TOKENS}" "${extra}" "${LAUNCHER}"
    return
  fi

  local launcher_output
  launcher_output="$(MODEL_NAME="${MODEL_NAME}" \
    CACHE_DIR="${CACHE_DIR}" \
    IMAGE_GRPO_MODEL_NAME="${MODEL_NAME}" \
    IMAGE_GRPO_CACHE_DIR="${CACHE_DIR}" \
    NUM_NODES="${nodes}" \
    JOB_NAME_BASE="${job_name_base}" \
    NEMO_RL_ISOLATED_CACHE_ROOT="${job_cache_root}" \
    CP_SIZE="${cp}" \
    POLICY_EP="${POLICY_EP}" \
    POLICY_MAX_TOTAL_SEQUENCE_LENGTH="${seq_len}" \
    HYBRID_CP_ENABLED="${hybrid_enabled}" \
    HYBRID_CP_FORCE_FULL_CP="${force_full_cp}" \
    HYBRID_CP_MAX_SEQLEN_PER_DP_CP_RANK="${threshold_env}" \
    HYBRID_CP_MICROBATCH_BUDGET_MULTIPLIER="${mb_env}" \
    NEMO_RL_VLLM_CAP_MAX_TOKENS_TO_CONTEXT="${cap_max_tokens_to_context}" \
    MCORE_DISABLE_TORCH_COMPILE_JIT="${MCORE_DISABLE_TORCH_COMPILE_JIT}" \
    ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE}" \
    NEMO_RL_SHARED_MEGATRON_CHECKPOINT_DIR="${SHARED_MEGATRON_CHECKPOINT_DIR}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
    SNAPSHOT_CODE="${SNAPSHOT_CODE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENABLED="${WANDB_ENABLED}" \
    GENERATION_MIN_NEW_TOKENS="${GENERATION_MIN_NEW_TOKENS}" \
    SBATCH_DEPENDENCY="${dependency}" \
    EXTRA_OVERRIDES_APPEND="${extra}" \
    bash "${LAUNCHER}")"
  echo "${launcher_output}"
  local submitted_job_id
  submitted_job_id="$(printf '%s\n' "${launcher_output}" | awk '/Submitted batch job/{print $4}' | tail -1)"
  if [[ -n "${submitted_job_id}" ]]; then
    LAST_SUBMITTED_JOB_ID="${submitted_job_id%%;*}"
  elif [[ "${SERIALIZE_SUBMISSIONS}" == "true" ]]; then
    echo "Could not parse submitted job id from launcher output; cannot serialize following submissions." >&2
    exit 1
  fi
}

add_canary() {
  submit_job "8k-cp4" 8192 8 4 4 dyncp 2048 1.0 "${CANARY_PROMPTS_PER_STEP}" "${CANARY_TRAIN_GBS}" "${CANARY_STEPS}"
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
  # sequences, a full-sequence threshold (49152) collapses local CP to 1 and
  # can OOM. Sweep safer per-rank budgets first.
  submit_job "49k-cp4-th12288" 49152 8 4 4 dyncp 12288 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th16384" 49152 8 4 4 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th24576" 49152 8 4 4 dyncp 24576 1.0 512 2048 "${MAX_STEPS}"
}

add_49k_dyncp_high_threshold_sweep_rows() {
  # Memory-limit probe: 49152 is the highest useful threshold for a 49K cap.
  submit_job "49k-cp4-th32768" 49152 8 4 4 dyncp 32768 1.0 512 2048 "${MAX_STEPS}"
  submit_job "49k-cp4-th49152" 49152 8 4 4 dyncp 49152 1.0 512 2048 "${MAX_STEPS}"
}

add_small_dyncp_max_threshold_probe_rows() {
  # Memory-limit probes: threshold equal to the sequence cap is the highest
  # useful value for each short-context run.
  submit_job "8k-cp8-th8192" 8192 8 8 8 dyncp 8192 1.0 512 2048 "${MAX_STEPS}"
  submit_job "16k-cp8-th16384" 16384 8 8 8 dyncp 16384 1.0 512 2048 "${MAX_STEPS}"
  submit_job "32k-cp8-th32768" 32768 8 8 8 dyncp 32768 1.0 512 2048 "${MAX_STEPS}"
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

add_64k_dyncp_high_threshold_sweep_rows() {
  submit_job "64k-tp2-cp32-th8192" 65536 2 32 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32-th16384" 65536 2 32 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32-th32768" 65536 2 32 8 dyncp 32768 1.0 128 512 "${MAX_STEPS}"
  submit_job "64k-tp2-cp32-th65536" 65536 2 32 8 dyncp 65536 1.0 128 512 "${MAX_STEPS}"
}

add_128k_dyncp_threshold_sweep_rows() {
  submit_job "128k-tp4-cp16-th8192" 131072 4 16 8 dyncp 8192 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th16384" 131072 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th32768" 131072 4 16 8 dyncp 32768 1.0 128 512 "${MAX_STEPS}"
}

add_128k_dyncp_high_threshold_sweep_rows() {
  submit_job "128k-tp4-cp16-th65536" 131072 4 16 8 dyncp 65536 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th131072" 131072 4 16 8 dyncp 131072 1.0 128 512 "${MAX_STEPS}"
}

add_256k_dyncp_threshold_sweep_rows() {
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-th16384" 262144 4 16 8 dyncp 16384 1.0 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-th32768" 262144 4 16 8 dyncp 32768 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_256k_dyncp_high_threshold_sweep_rows() {
  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-th65536" 262144 4 16 8 dyncp 65536 1.0 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-th131072" 262144 4 16 8 dyncp 131072 1.0 128 512 "${MAX_STEPS}"
  submit_job "256k-tp4-cp16-th262144" 262144 4 16 8 dyncp 262144 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
}

add_long_dyncp_threshold_sweep_rows() {
  add_64k_dyncp_threshold_sweep_rows
  add_128k_dyncp_threshold_sweep_rows
  add_256k_dyncp_threshold_sweep_rows
}

add_long_dyncp_high_threshold_sweep_rows() {
  add_64k_dyncp_high_threshold_sweep_rows
  add_128k_dyncp_high_threshold_sweep_rows
  add_256k_dyncp_high_threshold_sweep_rows
}

add_long_dyncp_max_threshold_probe_rows() {
  submit_job "64k-tp2-cp32-th65536" 65536 2 32 8 dyncp 65536 1.0 128 512 "${MAX_STEPS}"
  submit_job "128k-tp4-cp16-th131072" 131072 4 16 8 dyncp 131072 1.0 128 512 "${MAX_STEPS}"

  local previous_extra="${COMMON_EXTRA_OVERRIDES}"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.train_mb_tokens=131072"
  COMMON_EXTRA_OVERRIDES+=" policy.sequence_packing.logprob_mb_tokens=131072"
  submit_job "256k-tp4-cp16-th262144" 262144 4 16 8 dyncp 262144 1.0 128 512 "${MAX_STEPS}"
  COMMON_EXTRA_OVERRIDES="${previous_extra}"
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
  49k_dyncp_high_threshold_sweep)
    add_49k_dyncp_high_threshold_sweep_rows
    ;;
  small_dyncp_max_threshold_probe)
    add_small_dyncp_max_threshold_probe_rows
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
  64k_dyncp_high_threshold_sweep)
    add_64k_dyncp_high_threshold_sweep_rows
    ;;
  128k_dyncp_threshold_sweep)
    add_128k_dyncp_threshold_sweep_rows
    ;;
  128k_dyncp_high_threshold_sweep)
    add_128k_dyncp_high_threshold_sweep_rows
    ;;
  256k_dyncp_threshold_sweep)
    add_256k_dyncp_threshold_sweep_rows
    ;;
  256k_dyncp_high_threshold_sweep)
    add_256k_dyncp_high_threshold_sweep_rows
    ;;
  long_dyncp_threshold_sweep)
    add_long_dyncp_threshold_sweep_rows
    ;;
  long_dyncp_high_threshold_sweep)
    add_long_dyncp_high_threshold_sweep_rows
    ;;
  long_dyncp_max_threshold_probe)
    add_long_dyncp_max_threshold_probe_rows
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
    echo "Unknown SUBMIT_SET=${SUBMIT_SET}; expected canary, main, main_baseline, 8k_baseline, main_dyncp, nrt_cancelled_short_recovery, nrt_cancelled_long_recovery, nrt_reaper_recovery_exact, nrt_missing_dyncp_recovery, nrt_strict_missing_pairs, 49k, 49k_baseline, 49k_dyncp, 49k_dyncp_threshold_sweep, 49k_dyncp_high_threshold_sweep, small_dyncp_max_threshold_probe, long, long_no256, 64k, 64k_baseline, 128k, 256k_safe, 256k_safe_baseline, 256k_safe_dyncp, long_baseline, long_dyncp, long_dyncp_safe, 64k_dyncp_threshold_sweep, 64k_dyncp_high_threshold_sweep, 128k_dyncp_threshold_sweep, 128k_dyncp_high_threshold_sweep, 256k_dyncp_threshold_sweep, 256k_dyncp_high_threshold_sweep, long_dyncp_threshold_sweep, long_dyncp_high_threshold_sweep, long_dyncp_max_threshold_probe, step_time_49k, step_time_64k, step_time_128k, step_time, baselines, dyncp, all" >&2
    exit 1
    ;;
esac
