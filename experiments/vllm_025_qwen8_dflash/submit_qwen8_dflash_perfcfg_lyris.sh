#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
VARIANT="${VARIANT:-dflash_k16}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MAX_STEPS="${MAX_STEPS:-20}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-vllm025-qwen8-dflash}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm025-dflash-perfcfg-20260715}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
DRAFT_SNAPSHOT="${DRAFT_SNAPSHOT:-${HF_HOME}/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}"
RUN_TAG="${RUN_TAG:-qwen8-v025-perfcfg-${VARIANT}-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna/nemorl_reference_runs}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}}"

case "${VARIANT}" in
  baseline)
    SPECULATIVE_TOKENS=0
    ;;
  dflash_k16)
    SPECULATIVE_TOKENS=16
    ;;
  *)
    printf 'VARIANT must be baseline or dflash_k16; got %s\n' "${VARIANT}" >&2
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
  "cluster.segment_size=2"
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
  "NRL_MEGATRON_CHECKPOINT_DIR=${HF_HOME}/nemo_rl"
  "PYTHONPATH=${REPO_DIR}"
  "NEMO_RL_VENV_DIR=/tmp/nemorl-v025-qwen8-dflash-${RUN_TAG}"
  "NRL_FORCE_REBUILD_VENVS=true"
  "UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/uv_cache"
  "UV_LOCK_TIMEOUT=900"
  "TRITON_CACHE_DIR=/tmp/nemorl-v025-qwen8-dflash-triton-${RUN_TAG}"
  "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v025-qwen8-dflash-inductor-${RUN_TAG}"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
  "RAY_LOG_SYNC_FREQUENCY=30"
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config
  examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g.yaml
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=2
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --segment=2
  --dependency=
  --job-name="${ACCOUNT}-nemorl.qwen8-${VARIANT}"
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
      examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g.yaml
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
      printf 'run_tag=%s\nvariant=%s\n' "${RUN_TAG}" "${VARIANT}"
      printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
      printf 'bridge_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)"
      printf 'megatron_lm_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)"
      printf 'container=%s\n' "${CONTAINER}"
      printf 'recipe=examples/configs/recipes/llm/performance/grpo-qwen3-8b-2n4g.yaml\n'
      printf 'target_snapshot=%s\ndraft_snapshot=%s\n' "${TARGET_SNAPSHOT}" "${DRAFT_SNAPSHOT}"
      printf 'num_speculative_tokens=%s\n' "${SPECULATIVE_TOKENS}"
      printf 'max_steps=%s\nnum_nodes=2\nsegment=2\n' "${MAX_STEPS}"
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
