#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
VARIANT="${VARIANT:-eagle3_k3}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-4}"
SEGMENT="${SEGMENT:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
MAX_STEPS="${MAX_STEPS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
CUDAGRAPH_METRICS="${CUDAGRAPH_METRICS:-false}"
CAPTURE_PROFILE="${CAPTURE_PROFILE:-native}"
DYNAMIC_SD_SCHEDULE="${DYNAMIC_SD_SCHEDULE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-vllm0251-eagle3-perfcfg}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm0251-eagle3-fullcg-20260715}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HF_HOME}/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
DRAFT_SNAPSHOT="${DRAFT_SNAPSHOT:-${HF_HOME}/hub/models--RedHatAI--Qwen3-30B-A3B-Thinking-2507-speculator.eagle3/snapshots/a7ec796dd65236f1ecd4ed2958a7f0689e5da5cf}"
RUN_TAG="${RUN_TAG:-qwen30-v0251-${VARIANT}-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_0251_eagle3_perfcfg/runs}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}}"

case "${VARIANT}" in
  baseline) SPECULATIVE_TOKENS=0 ;;
  eagle3_k1) SPECULATIVE_TOKENS=1 ;;
  eagle3_k3) SPECULATIVE_TOKENS=3 ;;
  eagle3_k5) SPECULATIVE_TOKENS=5 ;;
  eagle3_k7) SPECULATIVE_TOKENS=7 ;;
  eagle3_k9) SPECULATIVE_TOKENS=9 ;;
  *)
    printf 'VARIANT must be baseline or eagle3_k{1,3,5,7,9}; got %s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${NUM_NODES}" != "4" || "${SEGMENT}" != "4" ]]; then
  printf 'Qwen3-30B-A3B 4n4g runs require NUM_NODES=4 and SEGMENT=4\n' >&2
  exit 2
fi
if [[ "${GPUS_PER_NODE}" != "4" ]]; then
  printf 'Qwen3-30B-A3B 4n4g runs require GPUS_PER_NODE=4\n' >&2
  exit 2
fi
if [[ "${CUDAGRAPH_METRICS}" != "true" && "${CUDAGRAPH_METRICS}" != "false" ]]; then
  printf 'CUDAGRAPH_METRICS must be true or false; got %s\n' "${CUDAGRAPH_METRICS}" >&2
  exit 2
fi
if [[ "${CUDAGRAPH_METRICS}" == "true" ]]; then
  printf 'CUDAGRAPH_METRICS is only supported by async vLLM recipes\n' >&2
  exit 2
fi
if [[ "${CAPTURE_PROFILE}" != "native" && "${CAPTURE_PROFILE}" != "compact" ]]; then
  printf 'CAPTURE_PROFILE must be native or compact; got %s\n' "${CAPTURE_PROFILE}" >&2
  exit 2
fi
if [[ -n "${DYNAMIC_SD_SCHEDULE}" && "${SPECULATIVE_TOKENS}" -eq 0 ]]; then
  printf 'DYNAMIC_SD_SCHEDULE requires an Eagle-3 variant\n' >&2
  exit 2
fi
if [[ -n "${DYNAMIC_SD_SCHEDULE}" && "${CAPTURE_PROFILE}" != "native" ]]; then
  printf 'DynamicSD requires CAPTURE_PROFILE=native\n' >&2
  exit 2
fi

overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "checkpointing.enabled=false"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE"
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
  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_SNAPSHOT}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${SPECULATIVE_TOKENS}"
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
  )
  if [[ "${CAPTURE_PROFILE}" == "compact" ]]; then
    capture_sizes=()
    for num_requests in 1 2 4 8 16 32 64; do
      capture_sizes+=("$((num_requests * (SPECULATIVE_TOKENS + 1)))")
    done
    printf -v capture_sizes_csv '%s,' "${capture_sizes[@]}"
    capture_sizes_csv="[${capture_sizes_csv%,}]"
    overrides+=(
      "++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=${capture_sizes_csv}"
    )
  fi
  if [[ -n "${DYNAMIC_SD_SCHEDULE}" ]]; then
    overrides+=(
      "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens_per_batch_size=${DYNAMIC_SD_SCHEDULE}"
    )
  fi
fi

command_env=(
  "VLLM_USE_V2_MODEL_RUNNER=1"
  "WANDB_RUN_GROUP=vllm0251-eagle3-fullcg"
  "WANDB_RESUME=never"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "PYTHONPATH=${REPO_DIR}"
  "NEMO_RL_VENV_DIR=/tmp/nemorl-v0251-${RUN_TAG}"
  "NRL_FORCE_REBUILD_VENVS=true"
  "TRITON_CACHE_DIR=/tmp/nemorl-v0251-triton-${RUN_TAG}"
  "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v0251-inductor-${RUN_TAG}"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
)
if [[ -n "${DYNAMIC_SD_SCHEDULE}" ]]; then
  command_env+=(
    "NRL_VENV_POST_SYNC_SCRIPT=${REPO_DIR}/experiments/vllm_0251_eagle3_perfcfg/apply_vllm0251_dynamic_sd_cg_fix.py"
    "NRL_VENV_POST_SYNC_TARGET=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
  )
fi

command_parts=(
  env
  "${command_env[@]}"
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --dependency=
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=4
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --segment=4
  --job-name="${ACCOUNT}-nemorl.q30-v0251-${VARIANT}"
  --output="${RUN_DIR}/slurm-%j.out"
  --comment=metrics
)

case "${MODE}" in
  dry-run)
    printf '[DRY-RUN] command %s\n' "${command}"
    printf '[DRY-RUN] environment BASE_LOG_DIR=%s GPUS_PER_NODE=%s\n' \
      "${BASE_LOG_DIR}" "${GPUS_PER_NODE}"
    printf '[DRY-RUN] sbatch'
    printf ' %s' "${sbatch_args[@]}"
    printf ' %s\n' "${REPO_DIR}/ray.sub"
    ;;
  test-only|submit)
    if [[ ! -f "${TARGET_SNAPSHOT}/model.safetensors.index.json" ]]; then
      printf 'Target snapshot is missing: %s\n' "${TARGET_SNAPSHOT}" >&2
      exit 2
    fi
    if [[ "${SPECULATIVE_TOKENS}" -gt 0 && ! -d "${DRAFT_SNAPSHOT}" ]]; then
      printf 'Eagle-3 drafter snapshot is missing: %s\n' "${DRAFT_SNAPSHOT}" >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty || \
       ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
      printf 'Submission requires a clean tracked checkout\n' >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
      printf 'HEAD must be pushed before submission\n' >&2
      exit 2
    fi
    mkdir -p "${RUN_DIR}"
    {
      printf 'run_tag=%s\n' "${RUN_TAG}"
      printf 'variant=%s\n' "${VARIANT}"
      printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
      printf 'container=%s\n' "${CONTAINER}"
      printf 'recipe=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml\n'
      printf 'target_snapshot=%s\n' "${TARGET_SNAPSHOT}"
      printf 'draft_snapshot=%s\n' "${DRAFT_SNAPSHOT}"
      printf 'num_speculative_tokens=%s\n' "${SPECULATIVE_TOKENS}"
      printf 'dynamic_sd_schedule=%s\n' "${DYNAMIC_SD_SCHEDULE:-disabled}"
      printf 'cudagraph_mode=FULL_AND_PIECEWISE\n'
      printf 'capture_profile=%s\n' "${CAPTURE_PROFILE}"
      printf 'vllm_use_v2_model_runner=1\n'
      printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
      printf 'command=%s\n' "${command}"
    } > "${RUN_DIR}/provenance.txt"
    export COMMAND="${command}" CONTAINER MOUNTS BASE_LOG_DIR GPUS_PER_NODE
    if [[ "${MODE}" == "test-only" ]]; then
      (cd "${REPO_DIR}" && sbatch --test-only "${sbatch_args[@]}" ray.sub)
    else
      job_id="$(cd "${REPO_DIR}" && sbatch --parsable "${sbatch_args[@]}" ray.sub)"
      printf 'job_id=%s\nrun_dir=%s\n' "${job_id}" "${RUN_DIR}"
    fi
    ;;
  *)
    printf 'MODE must be dry-run, test-only, or submit; got %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
