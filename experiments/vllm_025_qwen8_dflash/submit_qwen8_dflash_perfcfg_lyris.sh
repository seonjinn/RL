#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
MODEL_PROFILE="${MODEL_PROFILE:-qwen8}"
VARIANT="${VARIANT:-dflash_k16}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-gb200}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MAX_STEPS="${MAX_STEPS:-20}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-vllm025-qwen8-dflash}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/.secrets/wandb_api_key}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm025-dflash-perfcfg-20260715}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260715.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
MEGATRON_CHECKPOINT_DIR="${MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl-v025-dflash-perfcfg-20260715}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs}"

case "${MODEL_PROFILE}" in
  qwen8)
    MODEL_TAG=qwen8
    RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g.yaml
    NUM_NODES=2
    SEGMENT_SIZE=2
    DFLASH_VARIANT=dflash_k16
    DFLASH_TOKENS=16
    TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
    DRAFT_SNAPSHOT="${DRAFT_SNAPSHOT:-${HF_HOME}/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}"
    ;;
  qwen30ba3b)
    MODEL_TAG=qwen30ba3b
    RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
    NUM_NODES=4
    SEGMENT_SIZE=4
    DFLASH_VARIANT=dflash_k7
    DFLASH_TOKENS=7
    TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
    DRAFT_SNAPSHOT="${DRAFT_SNAPSHOT:-${HF_HOME}/hub/models--inference-optimization--Qwen3-30B-A3B-speculator.dflash/snapshots/2247bb71fb6ac89b75f44ec2c049c811bfd54ca5}"
    ;;
  *)
    printf 'MODEL_PROFILE must be qwen8 or qwen30ba3b; got %s\n' "${MODEL_PROFILE}" >&2
    exit 2
    ;;
esac

RUN_TAG="${RUN_TAG:-${MODEL_TAG}-v025-perfcfg-${VARIANT}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}}"

case "${VARIANT}" in
  baseline)
    SPECULATIVE_TOKENS=0
    ;;
  "${DFLASH_VARIANT}")
    SPECULATIVE_TOKENS="${DFLASH_TOKENS}"
    ;;
  *)
    printf 'VARIANT must be baseline or %s; got %s\n' "${DFLASH_VARIANT}" "${VARIANT}" >&2
    exit 2
    ;;
esac

has_safetensors_checkpoint() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/model.safetensors" || \
     -f "${checkpoint_dir}/model.safetensors.index.json" ]]
}

overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "checkpointing.enabled=false"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "policy.generation.temperature=1.0"
  "policy.generation.top_p=1.0"
  "cluster.segment_size=${SEGMENT_SIZE}"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.tensorboard_enabled=false"
  "logger.log_dir=${RUN_DIR}/nemo_logs"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  overrides+=(
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${RUN_TAG}"
  )
fi

if [[ "${SPECULATIVE_TOKENS}" -gt 0 ]]; then
  capture_sizes=()
  for num_requests in 1 2 4 8 16 32 64; do
    capture_sizes+=("$((num_requests * (SPECULATIVE_TOKENS + 1)))")
  done
  printf -v capture_sizes_csv '%s,' "${capture_sizes[@]}"
  capture_sizes_csv="[${capture_sizes_csv%,}]"

  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=dflash"
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_SNAPSHOT}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${SPECULATIVE_TOKENS}"
    "++policy.generation.vllm_kwargs.speculative_config.max_model_len=4096"
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
    "++policy.generation.vllm_kwargs.speculative_config.attention_backend=FLASH_ATTN"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL"
    "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes_csv}"
  )
fi

command_parts=(
  env
  "VLLM_USE_V2_MODEL_RUNNER=1"
  "WANDB_RUN_GROUP=${RUN_TAG}"
  "WANDB_RESUME=never"
  "NRL_DISABLE_VLLM_PORT_OVERRIDE=1"
  "NRL_DISABLE_NUMA_MEMBIND=1"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "NRL_MEGATRON_CHECKPOINT_DIR=${MEGATRON_CHECKPOINT_DIR}"
  "PYTHONPATH=${REPO_DIR}"
  "NEMO_RL_VENV_DIR=/tmp/nemorl-v025-dflash-${RUN_TAG}"
  "NRL_FORCE_REBUILD_VENVS=true"
  "UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/uv_cache"
  "UV_LOCK_TIMEOUT=900"
  "TRITON_CACHE_DIR=/tmp/nemorl-v025-qwen8-dflash-triton-${RUN_TAG}"
  "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v025-qwen8-dflash-inductor-${RUN_TAG}"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
  "RAY_LOG_SYNC_FREQUENCY=30"
)

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  command_parts+=(
    "WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE}"
    bash
    -c
    'set +x; export WANDB_API_KEY="$(< "${WANDB_API_KEY_FILE}")"; exec "$@"'
    nemo-rl-with-wandb-key
  )
fi

command_parts+=(
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config
  "${RECIPE}"
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes="${NUM_NODES}"
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --segment="${SEGMENT_SIZE}"
  --dependency=
  --job-name="${ACCOUNT}-nemorl.${MODEL_TAG}-${VARIANT}"
  --output="${RUN_DIR}/slurm-%j.out"
  --comment=metrics
)

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] command %s\n' "${command}"
    printf '[DRY-RUN] environment BASE_LOG_DIR=%s\n' "${BASE_LOG_DIR}"
    printf '[DRY-RUN] sbatch'
    printf ' %s' "${sbatch_args[@]}"
    printf ' %s\n' "${REPO_DIR}/ray.sub"
    ;;
  test-only|submit)
    if [[ ! -f "${CONTAINER}" ]]; then
      printf 'Container does not exist: %s\n' "${CONTAINER}" >&2
      exit 2
    fi
    if [[ "${WANDB_ENABLED}" == "true" && ! -r "${WANDB_API_KEY_FILE}" ]]; then
      printf 'W&B API key file is unavailable: %s\n' "${WANDB_API_KEY_FILE}" >&2
      exit 2
    fi
    if ! has_safetensors_checkpoint "${TARGET_SNAPSHOT}"; then
      printf 'Target snapshot is incomplete: %s\n' "${TARGET_SNAPSHOT}" >&2
      exit 2
    fi
    if [[ "${SPECULATIVE_TOKENS}" -gt 0 ]] && \
       ! has_safetensors_checkpoint "${DRAFT_SNAPSHOT}"; then
      printf 'DFlash snapshot is incomplete: %s\n' "${DRAFT_SNAPSHOT}" >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty || \
       ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
      printf 'Submission requires a clean tracked checkout\n' >&2
      exit 2
    fi
    required_tracked_files=(
      "${RECIPE}"
      experiments/vllm_025_qwen8_dflash/submit_qwen8_dflash_perfcfg_lyris.sh
    )
    for tracked_file in "${required_tracked_files[@]}"; do
      if ! git -C "${REPO_DIR}" ls-files --error-unmatch "${tracked_file}" >/dev/null 2>&1; then
        printf 'Submission dependency must be committed: %s\n' "${tracked_file}" >&2
        exit 2
      fi
    done
    if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
      printf 'HEAD must be pushed before submission\n' >&2
      exit 2
    fi

    mkdir -p "${RUN_DIR}"
    {
      printf 'run_tag=%s\nmodel_profile=%s\nvariant=%s\n' "${RUN_TAG}" "${MODEL_PROFILE}" "${VARIANT}"
      printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
      printf 'bridge_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)"
      printf 'megatron_lm_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)"
      printf 'container=%s\n' "${CONTAINER}"
      printf 'recipe=%s\n' "${RECIPE}"
      printf 'target_snapshot=%s\ndraft_snapshot=%s\n' "${TARGET_SNAPSHOT}" "${DRAFT_SNAPSHOT}"
      printf 'megatron_checkpoint_dir=%s\n' "${MEGATRON_CHECKPOINT_DIR}"
      printf 'num_speculative_tokens=%s\n' "${SPECULATIVE_TOKENS}"
      printf 'max_steps=%s\nnum_nodes=%s\nsegment=%s\n' "${MAX_STEPS}" "${NUM_NODES}" "${SEGMENT_SIZE}"
      printf 'cuda_graph_enabled=true\ncudagraph_mode=%s\n' "$([[ "${SPECULATIVE_TOKENS}" -gt 0 ]] && printf FULL || printf recipe-default)"
      printf 'temperature=1.0\ntop_p=1.0\nwandb_project=%s\n' "${WANDB_PROJECT}"
      printf 'command=%s\n' "${command}"
    } > "${RUN_DIR}/provenance.txt"

    export COMMAND="${command}" CONTAINER MOUNTS BASE_LOG_DIR
    if [[ "${MODE}" == "test-only" ]]; then
      (
        cd "${REPO_DIR}"
        sbatch --test-only "${sbatch_args[@]}" ray.sub
      )
    else
      job_id="$(
        cd "${REPO_DIR}"
        sbatch --parsable "${sbatch_args[@]}" ray.sub
      )"
      printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
    fi
    ;;
  *)
    printf 'MODE must be dry-run, test-only, or submit; got %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
