#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

MODE="${1:-test-only}"
MODEL_SELECTION="${2:-all}"
VARIANT_SELECTION="${3:-all}"
MAX_STEPS="${MAX_STEPS:-20}"
STATIC_K="${STATIC_K:-5}"
DYNAMIC_SCHEDULE="${DYNAMIC_SCHEDULE:-[[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]]}"
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-standard}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
PARD_K16_MAX_NUM_BATCHED_TOKENS="${PARD_K16_MAX_NUM_BATCHED_TOKENS:-32768}"
ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-batch_long}"
USE_GRES="${USE_GRES:-true}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-dynamicsd-aws-dfw}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
OUTPUT_MAX_MODEL_LEN="${OUTPUT_MAX_MODEL_LEN:-}"
SPECDEC_CONTEXT_HEADROOM_TOKENS="${SPECDEC_CONTEXT_HEADROOM_TOKENS:-0}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-}"

if [[ "${REJECTION_SAMPLE_METHOD}" != "standard" ]]; then
  echo "ERROR: REJECTION_SAMPLE_METHOD must be standard (got ${REJECTION_SAMPLE_METHOD})" >&2
  exit 2
fi
case "${DRAFT_SAMPLE_METHOD}" in
  greedy|probabilistic)
    ;;
  *)
    echo "ERROR: DRAFT_SAMPLE_METHOD must be greedy or probabilistic (got ${DRAFT_SAMPLE_METHOD})" >&2
    exit 2
    ;;
esac
for numeric_override in \
  MAX_NUM_BATCHED_TOKENS \
  MAX_NUM_SEQS \
  OUTPUT_MAX_MODEL_LEN \
  MAX_CUDAGRAPH_CAPTURE_SIZE; do
  numeric_value="${!numeric_override}"
  if [[ -n "${numeric_value}" && ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${numeric_override} must be a positive integer" >&2
    exit 2
  fi
done
if [[ ! "${SPECDEC_CONTEXT_HEADROOM_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SPECDEC_CONTEXT_HEADROOM_TOKENS must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${CUDAGRAPH_CAPTURE_SIZES}" \
  && ! "${CUDAGRAPH_CAPTURE_SIZES}" =~ ^\[[1-9][0-9]*(,[1-9][0-9]*)*\]$ ]]; then
  echo "ERROR: CUDAGRAPH_CAPTURE_SIZES must be a comma-separated list of positive integers" >&2
  exit 2
fi
if [[ "${SPECDEC_CONTEXT_HEADROOM_TOKENS}" != "0" && -z "${OUTPUT_MAX_MODEL_LEN}" ]]; then
  echo "ERROR: OUTPUT_MAX_MODEL_LEN is required when reserving SpecDec context headroom" >&2
  exit 2
fi

if [[ -z "${REPO_DIR:-}" ]]; then
  logical_pwd="$(pwd -L)"
  repo_prefix="$(git rev-parse --show-prefix)"
  if [[ -n "${repo_prefix}" ]]; then
    REPO_DIR="${logical_pwd%/${repo_prefix%/}}"
  else
    REPO_DIR="${logical_pwd}"
  fi
fi
AWS_ROOT="${AWS_ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna}"
CONTAINER="${CONTAINER:-${AWS_ROOT}/containers/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-${AWS_ROOT}/hf_home}"
ARCTIC_OVERLAY="${ARCTIC_OVERLAY:-${AWS_ROOT}/python_overlays/arctic-inference-0.1.1-py313-aarch64}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${AWS_ROOT}/.secrets/wandb_api_key}"
WANDB_NETRC_HOME="${WANDB_NETRC_HOME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
RUN_TAG="${RUN_TAG:-vllm024-dynamicsd-step20-20260707}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_024_upgrade/runs/${RUN_TAG}}"
WALLTIME="${WALLTIME:-04:00:00}"
TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
CONTAINER_SHA256="${CONTAINER_SHA256:-}"

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

if [[ "${MODE}" == "submit" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -r "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
  elif [[ -n "${WANDB_NETRC_HOME}" && -r "${WANDB_NETRC_HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(awk '$1 == "password" {print $2; exit}' "${WANDB_NETRC_HOME}/.netrc")"
  else
    echo "ERROR: set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  export WANDB_API_KEY
fi

if [[ "${MODE}" == "submit" ]]; then
  if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty \
    || ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
    echo "ERROR: submit requires a clean tracked checkout" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
    experiments/vllm_024_upgrade/submit_eagle3_dynamicsd_step20.sh >/dev/null 2>&1; then
    echo "ERROR: launcher must be committed before submit" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
    echo "ERROR: HEAD is not present on a known remote branch" >&2
    exit 2
  fi
fi

if [[ -n "${NUM_PROMPTS_PER_STEP}" || -n "${NUM_GENERATIONS_PER_PROMPT}" \
  || -n "${TRAIN_GLOBAL_BATCH_SIZE}" ]]; then
  if [[ ! "${NUM_PROMPTS_PER_STEP}" =~ ^[1-9][0-9]*$ \
    || ! "${NUM_GENERATIONS_PER_PROMPT}" =~ ^[1-9][0-9]*$ \
    || ! "${TRAIN_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: prompt/generation overrides require positive NUM_PROMPTS_PER_STEP, NUM_GENERATIONS_PER_PROMPT, and TRAIN_GLOBAL_BATCH_SIZE" >&2
    exit 2
  fi
  total_trajectories=$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))
  if ((TRAIN_GLOBAL_BATCH_SIZE > total_trajectories)); then
    echo "ERROR: TRAIN_GLOBAL_BATCH_SIZE cannot exceed ${total_trajectories} trajectories" >&2
    exit 2
  fi
  if ((total_trajectories % TRAIN_GLOBAL_BATCH_SIZE != 0)); then
    echo "ERROR: ${total_trajectories} trajectories must be divisible by TRAIN_GLOBAL_BATCH_SIZE" >&2
    exit 2
  fi
fi

case "${MODEL_SELECTION}" in
  all) models=(qwen30ba3b qwen32b qwen235b) ;;
  qwen30ba3b|qwen32b|qwen235b) models=("${MODEL_SELECTION}") ;;
  *)
    echo "ERROR: model must be all, qwen30ba3b, qwen32b, or qwen235b" >&2
    exit 2
    ;;
esac

case "${VARIANT_SELECTION}" in
  all) variants=(baseline eagle3_k5 eagle3_k7 eagle3_k9 dynamic) ;;
  low-k) variants=(eagle3_k1 eagle3_k2) ;;
  aggressive) variants=(eagle3_k7 eagle3_k9) ;;
  compare) variants=(baseline eagle3_k5 eagle3_k7 eagle3_k9 suffix_k32 pard_k5 pard_k16 dflash_k15) ;;
  baseline|eagle3_k1|eagle3_k2|eagle3_k5|eagle3_k7|eagle3_k9|dynamic|suffix_k32|pard_k5|pard_k16|dflash_k15) variants=("${VARIANT_SELECTION}") ;;
  *)
    echo "ERROR: variant must be all, low-k, aggressive, compare, baseline, eagle3_k1, eagle3_k2, eagle3_k5, eagle3_k7, eagle3_k9, dynamic, suffix_k32, pard_k5, pard_k16, or dflash_k15" >&2
    exit 2
    ;;
esac

if [[ "${VARIANT_SELECTION}" == "compare" && "${MODEL_SELECTION}" != "qwen30ba3b" ]]; then
  echo "ERROR: compare currently supports only qwen30ba3b" >&2
  exit 2
fi
if [[ "${VARIANT_SELECTION}" == "dflash_k15" && "${MODEL_SELECTION}" != "qwen30ba3b" ]]; then
  echo "ERROR: dflash_k15 only supports qwen30ba3b" >&2
  exit 2
fi

submit_one() {
  local model="$1"
  local variant="$2"
  local recipe
  local draft_model
  local draft_k="${STATIC_K}"
  local draft_tp=1
  local dflash_cache
  local dflash_revision
  local manifest_rejection_sample_method="not_applicable"
  local manifest_draft_sample_method="not_applicable"
  local resolved_max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}"
  local nodes

  case "${model}" in
    qwen30ba3b)
      recipe="${QWEN30_RECIPE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml}"
      draft_model="${QWEN30_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
      nodes="${QWEN30_NODES:-4}"
      ;;
    qwen32b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      draft_model="${QWEN32_DRAFT_MODEL:-${HF_HOME}/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/dc84fe7ff1db31efa824776f49c141fc8195eb47}"
      nodes=4
      ;;
    qwen235b)
      recipe="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml"
      draft_model="${QWEN235_DRAFT_MODEL:-${HF_HOME}/hub/models--nvidia--Qwen3-235B-A22B-Eagle3/snapshots/33f3c01ce807376d1171301b9a148b1b28f239ba}"
      nodes=16
      ;;
  esac

  if [[ "${MODE}" != "dry-run" && ! -f "${REPO_DIR}/${recipe}" ]]; then
    echo "ERROR: recipe not found: ${REPO_DIR}/${recipe}" >&2
    exit 2
  fi
  if [[ "${MODE}" == "submit" ]] \
    && ! git -C "${REPO_DIR}" ls-files --error-unmatch "${recipe}" >/dev/null 2>&1; then
    echo "ERROR: recipe must be tracked before submit: ${recipe}" >&2
    exit 2
  fi

  case "${variant}" in
    suffix_k32)
      draft_model=""
      draft_k=32
      ;;
    pard_k5|pard_k16)
      case "${model}" in
        qwen30ba3b)
          draft_model="${QWEN30_PARD_MODEL:-${HF_HOME}/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136}"
          ;;
        qwen32b)
          draft_model="${QWEN32_PARD_MODEL:-${HF_HOME}/hub/models--amd--PARD-Qwen3-0.6B/snapshots/f9f650fbab180c26498817718f0db5cae8f25136}"
          draft_tp=2
          ;;
        *)
          echo "ERROR: ${variant} does not have a qualified PARD checkpoint for ${model}" >&2
          exit 2
          ;;
      esac
      ;;
    dflash_k15)
      case "${model}" in
        qwen30ba3b)
          if [[ -n "${QWEN30_DFLASH_MODEL:-}" ]]; then
            draft_model="${QWEN30_DFLASH_MODEL}"
          else
            dflash_cache="${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-speculator.dflash"
            dflash_revision="RESOLVED_FROM_REFS_MAIN"
            if [[ -r "${dflash_cache}/refs/main" ]]; then
              dflash_revision="$(<"${dflash_cache}/refs/main")"
            fi
            draft_model="${dflash_cache}/snapshots/${dflash_revision}"
          fi
          ;;
        *)
          echo "ERROR: ${variant} does not have a qualified DFlash checkpoint for ${model}" >&2
          exit 2
          ;;
      esac
      ;;
  esac
  if [[ -z "${resolved_max_num_batched_tokens}" ]] \
    && [[ "${variant}" == "pard_k16" || "${VARIANT_SELECTION}" == "compare" ]]; then
    resolved_max_num_batched_tokens="${PARD_K16_MAX_NUM_BATCHED_TOKENS}"
  fi

  if [[ "${MODE}" != "dry-run" && -n "${draft_model}" && "${variant}" != "baseline" && ! -d "${draft_model}" ]]; then
    echo "ERROR: draft model directory not found: ${draft_model}" >&2
    exit 2
  fi
  if [[ "${MODE}" != "dry-run" && "${variant}" == "suffix_k32" && ! -f "${ARCTIC_OVERLAY}/arctic_inference/suffix_decoding/__init__.py" ]]; then
    echo "ERROR: arctic-inference overlay not found: ${ARCTIC_OVERLAY}" >&2
    exit 2
  fi

  local run_dir="${EXPERIMENT_ROOT}/${model}/${variant}"
  local wandb_run_id="${RUN_TAG}-${ATTEMPT_ID}-${model}-${variant}"
  local wandb_name="${wandb_run_id}"
  local runtime_pythonpath="${REPO_DIR}"
  if [[ "${variant}" == "suffix_k32" ]]; then
    runtime_pythonpath="${ARCTIC_OVERLAY}:${REPO_DIR}"
  fi
  local triton_cache_dir="/tmp/nemorl-vllm024-triton-${RUN_TAG}-${model}-${variant}"
  local inductor_cache_dir="/tmp/nemorl-vllm024-inductor-${RUN_TAG}-${model}-${variant}"
  local venv_dir="/tmp/nemorl-vllm024-venvs-${RUN_TAG}-${ATTEMPT_ID}-${model}-${variant}"
  case "${variant}" in
    eagle3_k1)
      draft_k=1
      ;;
    eagle3_k2)
      draft_k=2
      ;;
    eagle3_k5)
      draft_k=5
      ;;
    eagle3_k7)
      draft_k=7
      ;;
    eagle3_k9)
      draft_k=9
      ;;
    suffix_k32)
      draft_k=32
      ;;
    pard_k5)
      draft_k=5
      ;;
    pard_k16)
      draft_k=16
      ;;
    dflash_k15)
      draft_k=15
      ;;
  esac

  local overrides=(
    "grpo.max_num_steps=${MAX_STEPS}"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
    "policy.generation.vllm_cfg.enforce_eager=false"
    "policy.generation.temperature=1.0"
    "policy.generation.top_p=1.0"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "cluster.num_nodes=${nodes}"
    "cluster.segment_size=${nodes}"
    "logger.wandb_enabled=true"
    "logger.tensorboard_enabled=false"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${wandb_name}"
    "++logger.wandb.entity=${WANDB_ENTITY}"
    "logger.log_dir=${run_dir}/nemo_logs"
  )
  if [[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE}" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.compilation_config.max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE}"
    )
  fi
  if [[ -n "${CUDAGRAPH_CAPTURE_SIZES}" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES}"
    )
  fi
  if [[ -n "${resolved_max_num_batched_tokens}" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.max_num_batched_tokens=${resolved_max_num_batched_tokens}"
    )
  fi
  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    overrides+=("++policy.generation.vllm_kwargs.max_num_seqs=${MAX_NUM_SEQS}")
  fi
  if [[ -n "${OUTPUT_MAX_MODEL_LEN}" ]]; then
    engine_max_model_len=$((OUTPUT_MAX_MODEL_LEN + SPECDEC_CONTEXT_HEADROOM_TOKENS))
    overrides+=(
      "++policy.generation._output_max_model_len=${OUTPUT_MAX_MODEL_LEN}"
      "policy.generation.vllm_cfg.max_model_len=${engine_max_model_len}"
    )
  fi
  if [[ -n "${NUM_PROMPTS_PER_STEP}" ]]; then
    overrides+=("grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}")
  fi
  if [[ -n "${NUM_GENERATIONS_PER_PROMPT}" ]]; then
    overrides+=("grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}")
  fi
  if [[ -n "${TRAIN_GLOBAL_BATCH_SIZE}" ]]; then
    overrides+=("policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}")
  fi
  if [[ -n "${MAX_TOTAL_SEQUENCE_LENGTH}" ]]; then
    overrides+=("policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}")
  fi
  if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    overrides+=("policy.generation.max_new_tokens=${MAX_NEW_TOKENS}")
  fi
  case "${variant}" in
    baseline)
      ;;
    suffix_k32)
      manifest_rejection_sample_method="${REJECTION_SAMPLE_METHOD}"
      overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.method=suffix"
        "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${REJECTION_SAMPLE_METHOD}"
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
      )
      ;;
    pard_k5|pard_k16)
      manifest_rejection_sample_method="${REJECTION_SAMPLE_METHOD}"
      manifest_draft_sample_method="${DRAFT_SAMPLE_METHOD}"
      overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.method=draft_model"
        "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=${draft_tp}"
        "++policy.generation.vllm_kwargs.speculative_config.parallel_drafting=true"
        "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${REJECTION_SAMPLE_METHOD}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD}"
        "++policy.generation.vllm_cfg.env_vars.NRL_VLLM_ENABLE_DRAFT_MODEL_CUDAGRAPH_PATCH=true"
      )
      ;;
    dflash_k15)
      manifest_rejection_sample_method="${REJECTION_SAMPLE_METHOD}"
      manifest_draft_sample_method="${DRAFT_SAMPLE_METHOD}"
      overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.method=dflash"
        "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
        "++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN"
        "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${REJECTION_SAMPLE_METHOD}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD}"
      )
      ;;
    *)
      manifest_rejection_sample_method="${REJECTION_SAMPLE_METHOD}"
      manifest_draft_sample_method="${DRAFT_SAMPLE_METHOD}"
      overrides+=(
        "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
        "++policy.generation.vllm_kwargs.speculative_config.model=${draft_model}"
        "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${draft_k}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
        "++policy.generation.vllm_kwargs.speculative_config.rejection_sample_method=${REJECTION_SAMPLE_METHOD}"
        "++policy.generation.vllm_kwargs.speculative_config.draft_sample_method=${DRAFT_SAMPLE_METHOD}"
      )
      if [[ "${variant}" == "dynamic" ]]; then
        overrides+=(
          "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=${DYNAMIC_SCHEDULE}"
        )
      fi
      ;;
  esac

  local command_env=(
    "WANDB_RUN_ID=${wandb_run_id}"
    "WANDB_RUN_GROUP=${RUN_TAG}"
    "WANDB_RESUME=never"
    "NEMO_RL_VENV_DIR=${venv_dir}"
    "NRL_FORCE_REBUILD_VENVS=true"
    # BaseVllmGenerationWorker assigns a distinct rendezvous window per engine.
    "PYTHONPATH=${runtime_pythonpath}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  if [[ "${NRL_IGNORE_TP_ACCURACY_CHECK:-0}" == "1" ]]; then
    command_env+=("NRL_IGNORE_TP_ACCURACY_CHECK=1")
  fi
  local command_parts=(
    env
    "${command_env[@]}"
    /opt/nemo_rl_venv/bin/python
    examples/run_grpo.py
    --config
    "${recipe}"
    "${overrides[@]}"
  )
  local command
  printf -v command '%q ' "${command_parts[@]}"
  command="${command% }"

  local environment=(
    "CONTAINER=${CONTAINER}"
    "MOUNTS=/lustre:/lustre"
    "CONTAINER_WORKDIR=${REPO_DIR}"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=${GPUS_PER_NODE}"
    "HF_HOME=${HF_HOME}"
    "PYTHONPATH=${runtime_pythonpath}"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
    "TMPDIR=${TMPDIR}"
    "TRITON_CACHE_DIR=${triton_cache_dir}"
    "TORCHINDUCTOR_CACHE_DIR=${inductor_cache_dir}"
  )
  if [[ "${model}" == "qwen235b" ]]; then
    environment+=("NRL_DISABLE_VLLM_PORT_OVERRIDE=1")
  fi
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${nodes}"
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment="${nodes}"
    --dependency=
    --job-name="${ACCOUNT}-nemorl.dynamicsd-${model}-${variant}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )
  if [[ "${USE_GRES}" == "true" ]]; then
    sbatch_args+=(--gres="gpu:${GPUS_PER_NODE}")
  fi

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] env'
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${REPO_DIR}/ray.sub"
      printf '[DRY-RUN] command %s\n' "${command}"
      printf '[DRY-RUN] wandb https://wandb.ai/%s/%s/runs/%s\n' \
        "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${wandb_run_id}"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local manifest="${EXPERIMENT_ROOT}/submissions.tsv"
      local manifest_header=$'timestamp\tmodel\tvariant\tjob_id\tnodes\tsegment\tcommit\twandb_run_id\twandb_url\trecipe\tdraft_model\tcontainer\tcontainer_sha256\tmax_steps\tstatic_k\tdynamic_schedule\trejection_sample_method\tdraft_sample_method\tmax_num_batched_tokens\tmax_num_seqs\toutput_max_model_len\tspecdec_context_headroom_tokens\tmax_cudagraph_capture_size\tcudagraph_capture_sizes\tnum_prompts_per_step\tnum_generations_per_prompt\ttrain_global_batch_size\tmax_total_sequence_length\tmax_new_tokens\tcommand'
      if [[ -f "${manifest}" ]]; then
        local existing_manifest_header
        existing_manifest_header="$(head -n 1 "${manifest}")"
        if [[ "${existing_manifest_header}" != "${manifest_header}" ]]; then
          echo "ERROR: submissions manifest header mismatch: ${manifest}" >&2
          exit 2
        fi
      fi
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" "${REPO_DIR}/ray.sub")"
      if [[ ! -f "${manifest}" ]]; then
        printf '%s\n' "${manifest_header}" > "${manifest}"
      fi
      local resolved_container
      resolved_container="$(readlink -f "${CONTAINER}")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${model}" "${variant}" "${job_id}" \
        "${nodes}" "${nodes}" "$(git -C "${REPO_DIR}" rev-parse HEAD)" \
        "${wandb_run_id}" "https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${wandb_run_id}" \
        "${recipe}" "${draft_model}" "${resolved_container}" "${CONTAINER_SHA256}" \
        "${MAX_STEPS}" "${draft_k}" "${DYNAMIC_SCHEDULE}" \
        "${manifest_rejection_sample_method}" "${manifest_draft_sample_method}" \
        "${resolved_max_num_batched_tokens}" "${MAX_NUM_SEQS}" \
        "${OUTPUT_MAX_MODEL_LEN}" "${SPECDEC_CONTEXT_HEADROOM_TOKENS}" \
        "${MAX_CUDAGRAPH_CAPTURE_SIZE}" "${CUDAGRAPH_CAPTURE_SIZES}" \
        "${NUM_PROMPTS_PER_STEP}" \
        "${NUM_GENERATIONS_PER_PROMPT}" "${TRAIN_GLOBAL_BATCH_SIZE}" \
        "${MAX_TOTAL_SEQUENCE_LENGTH}" "${MAX_NEW_TOKENS}" \
        "${command}" >> "${manifest}"
      ;;
    *)
      echo "ERROR: mode must be dry-run, test-only, or submit" >&2
      exit 2
      ;;
  esac
}

if [[ "${MODE}" != "dry-run" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}"
fi
for model in "${models[@]}"; do
  for variant in "${variants[@]}"; do
    submit_one "${model}" "${variant}"
  done
done
