#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-dry-run}"
VARIANT="${VARIANT:-eagle3_k3}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-16}"
SEGMENT="${SEGMENT:-16}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_STEPS="${MAX_STEPS:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
CUDAGRAPH_METRICS="${CUDAGRAPH_METRICS:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-vllm025-q235-specdec}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm025-thinking-eagle3-20260714}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
TARGET_SNAPSHOT="${TARGET_SNAPSHOT:-${HF_HOME}/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65}"
DRAFT_SNAPSHOT="${DRAFT_SNAPSHOT:-${HF_HOME}/hub/models--RedHatAI--Qwen3-235B-A22B-Thinking-2507-speculator.eagle3/snapshots/3c0c5cbad8e1fa7ce9e6fb6a1b0a35458b124e87}"
RUN_TAG="${RUN_TAG:-q235-v025-thinking-${VARIANT}-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_025_q235_specdec/runs}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${RUN_DIR}}"
BASELINE_JOB_ID="${BASELINE_JOB_ID:-2375433}"

case "${VARIANT}" in
  baseline)
    SPECULATIVE_TOKENS=0
    ;;
  eagle3_k1)
    SPECULATIVE_TOKENS=1
    ;;
  eagle3_k3)
    SPECULATIVE_TOKENS=3
    ;;
  eagle3_k5)
    SPECULATIVE_TOKENS=5
    ;;
  eagle3_k7)
    SPECULATIVE_TOKENS=7
    ;;
  eagle3_k9)
    SPECULATIVE_TOKENS=9
    ;;
  *)
    printf 'VARIANT must be baseline, eagle3_k1, eagle3_k3, eagle3_k5, eagle3_k7, or eagle3_k9; got %s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${NUM_NODES}" != "16" || "${SEGMENT}" != "16" ]]; then
  printf 'Qwen3-235B performance runs require NUM_NODES=16 and SEGMENT=16\n' >&2
  exit 2
fi

if [[ "${CUDAGRAPH_METRICS}" != "true" && "${CUDAGRAPH_METRICS}" != "false" ]]; then
  printf 'CUDAGRAPH_METRICS must be true or false; got %s\n' "${CUDAGRAPH_METRICS}" >&2
  exit 2
fi

overrides=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "checkpointing.enabled=false"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "policy.generation.temperature=1.0"
  "policy.generation.top_p=1.0"
  "cluster.segment_size=16"
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

if [[ "${CUDAGRAPH_METRICS}" == "true" ]]; then
  overrides+=("++policy.generation.vllm_kwargs.cudagraph_metrics=true")
fi

if [[ "${SPECULATIVE_TOKENS}" -gt 0 ]]; then
  overrides+=(
    "++policy.generation.vllm_kwargs.speculative_config.method=eagle3"
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT_SNAPSHOT}"
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${SPECULATIVE_TOKENS}"
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1"
  )
fi

command_env=(
  "WANDB_RUN_GROUP=${RUN_TAG}"
  "WANDB_RESUME=never"
  "NRL_DISABLE_VLLM_PORT_OVERRIDE=1"
  "NRL_DISABLE_NUMA_MEMBIND=1"
  "NRL_DEBUG_REFERENCE_MODEL_SETUP=1"
  "NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS=300"
  "NRL_REFERENCE_SETUP_MARKER_DIR=${RUN_DIR}/reference_setup"
  "NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800"
  "PYTHONFAULTHANDLER=1"
  "RAY_DEDUP_LOGS=0"
  "RAY_LOG_SYNC_FREQUENCY=30"
  "NCCL_DEBUG=WARN"
  "TORCH_NCCL_TRACE_BUFFER_SIZE=2000"
  "TORCH_NCCL_DUMP_ON_TIMEOUT=1"
  "TORCH_NCCL_DESYNC_DEBUG=1"
  "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=60000"
  "TORCH_FR_DUMP_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_"
  "TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_"
  "TORCH_INCLUDE_STACK_TRACE=1"
  "TORCH_INCLUDE_ONLY_ACTIVE=0"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "NRL_MEGATRON_CHECKPOINT_DIR=${HF_HOME}/nemo_rl"
  "PYTHONPATH=${REPO_DIR}"
  "NEMO_RL_VENV_DIR=/tmp/nemorl-v025-q235-thinking-${RUN_TAG}"
  "NRL_FORCE_REBUILD_VENVS=true"
  "UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/uv_cache"
  "UV_LOCK_TIMEOUT=900"
  "TRITON_CACHE_DIR=/tmp/nemorl-v025-q235-thinking-triton-${RUN_TAG}"
  "TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v025-q235-thinking-inductor-${RUN_TAG}"
)

command_parts=(
  env
  "${command_env[@]}"
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config
  examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml
  "${overrides[@]}"
)
printf -v command '%q ' "${command_parts[@]}"
command="${command% }"

sbatch_args=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes=16
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --segment=16
  --dependency=
  --job-name="${ACCOUNT}-nemorl.q235-thinking-${VARIANT}"
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
    if [[ ! -f "${TARGET_SNAPSHOT}/model.safetensors.index.json" ]]; then
      printf 'Target snapshot is missing its safetensors index: %s\n' "${TARGET_SNAPSHOT}" >&2
      exit 2
    fi
    if [[ ! -d "${DRAFT_SNAPSHOT}" && "${SPECULATIVE_TOKENS}" -gt 0 ]]; then
      printf 'Thinking drafter snapshot is missing: %s\n' "${DRAFT_SNAPSHOT}" >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" diff --quiet --ignore-submodules=dirty || \
       ! git -C "${REPO_DIR}" diff --cached --quiet --ignore-submodules=dirty; then
      printf 'Submission requires a clean tracked checkout\n' >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" ls-files --error-unmatch \
      experiments/vllm_025_q235_specdec/submit_q235_thinking_eagle3_ptyche.sh >/dev/null 2>&1; then
      printf 'Launcher must be committed before submission\n' >&2
      exit 2
    fi
    if ! git -C "${REPO_DIR}" branch -r --contains HEAD | grep -q .; then
      printf 'HEAD must be pushed before submission\n' >&2
      exit 2
    fi

    mkdir -p "${RUN_DIR}/reference_setup" "${RUN_DIR}/torch_nccl"
    {
      printf 'run_tag=%s\n' "${RUN_TAG}"
      printf 'variant=%s\n' "${VARIANT}"
      printf 'baseline_job_id=%s\n' "${BASELINE_JOB_ID}"
      printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
      printf 'bridge_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)"
      printf 'megatron_lm_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)"
      printf 'container=%s\n' "${CONTAINER}"
      printf 'target_model=Qwen/Qwen3-235B-A22B\n'
      printf 'target_snapshot=%s\n' "${TARGET_SNAPSHOT}"
      printf 'draft_snapshot=%s\n' "${DRAFT_SNAPSHOT}"
      printf 'num_speculative_tokens=%s\n' "${SPECULATIVE_TOKENS}"
      printf 'max_steps=%s\n' "${MAX_STEPS}"
      printf 'num_nodes=16\nsegment=16\n'
      printf 'cuda_graph_enabled=true\n'
      printf 'cudagraph_metrics=%s\n' "${CUDAGRAPH_METRICS}"
      printf 'numa_cpu_affinity=true\nnuma_membind=false\n'
      printf 'temperature=1.0\ntop_p=1.0\n'
      printf 'wandb_enabled=%s\n' "${WANDB_ENABLED}"
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
      printf '%s\n' "job_id=${job_id}" "run_dir=${RUN_DIR}"
    fi
    ;;
  *)
    printf 'MODE must be dry-run, test-only, or submit; got %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
