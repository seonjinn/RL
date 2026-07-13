#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-16}"
SEGMENT="${SEGMENT:-16}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_STEPS="${MAX_STEPS:-1}"
NUMA_MEMBIND_MODE="${NUMA_MEMBIND_MODE:-off}"
DRY_RUN="${DRY_RUN:-0}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm025-numa-diagnostics-20260713}"
RUN_TAG="${RUN_TAG:-q235-v025-numa-${NUMA_MEMBIND_MODE}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/experiments/vllm_025_q235_numa_oom/runs/${RUN_TAG}}"

case "${NUMA_MEMBIND_MODE}" in
  on)
    DISABLE_NUMA_MEMBIND=0
    ;;
  off)
    DISABLE_NUMA_MEMBIND=1
    ;;
  *)
    printf 'NUMA_MEMBIND_MODE must be on or off, got: %s\n' "${NUMA_MEMBIND_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}/reference_setup" "${RUN_DIR}/torch_nccl"
{
  printf 'run_tag=%s\n' "${RUN_TAG}"
  printf 'repo_head=%s\n' "$(git -C "${REPO_DIR}" rev-parse HEAD)"
  printf 'bridge_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)"
  printf 'megatron_lm_head=%s\n' "$(git -C "${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)"
  printf 'container=%s\n' "${CONTAINER}"
  printf 'numa_membind_mode=%s\n' "${NUMA_MEMBIND_MODE}"
  printf 'num_nodes=%s\nsegment=%s\nmax_steps=%s\n' "${NUM_NODES}" "${SEGMENT}" "${MAX_STEPS}"
} > "${RUN_DIR}/provenance.txt"

COMMAND="env \
WANDB_RUN_GROUP=${RUN_TAG} \
WANDB_RESUME=never \
NRL_DISABLE_VLLM_PORT_OVERRIDE=1 \
NRL_DISABLE_NUMA_MEMBIND=${DISABLE_NUMA_MEMBIND} \
NRL_DEBUG_REFERENCE_MODEL_SETUP=1 \
NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS=300 \
NRL_REFERENCE_SETUP_MARKER_DIR=${RUN_DIR}/reference_setup \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800 \
PYTHONFAULTHANDLER=1 \
RAY_DEDUP_LOGS=0 \
RAY_LOG_SYNC_FREQUENCY=30 \
NCCL_DEBUG=WARN \
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_DESYNC_DEBUG=1 \
TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=60000 \
TORCH_FR_DUMP_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_ \
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_ \
TORCH_INCLUDE_STACK_TRACE=1 \
TORCH_INCLUDE_ONLY_ACTIVE=0 \
HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home \
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/nemo_rl \
PYTHONPATH=${REPO_DIR} \
NEMO_RL_VENV_DIR=/tmp/nemorl-v025-numa-${RUN_TAG} \
NRL_FORCE_REBUILD_VENVS=true \
UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/uv_cache \
UV_LOCK_TIMEOUT=900 \
TRITON_CACHE_DIR=/tmp/nemorl-v025-numa-triton-${RUN_TAG} \
TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v025-numa-inductor-${RUN_TAG} \
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml \
grpo.max_num_steps=${MAX_STEPS} \
checkpointing.enabled=false \
checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints \
policy.generation.vllm_cfg.enforce_eager=false \
logger.wandb_enabled=false \
logger.tensorboard_enabled=false \
logger.log_dir=${RUN_DIR}/nemo_logs"

export COMMAND CONTAINER MOUNTS

sbatch_args=(
  --parsable
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes="${NUM_NODES}"
  --ntasks-per-node=1
  --exclusive
  --time="${TIME_LIMIT}"
  --segment="${SEGMENT}"
  --dependency=
  --job-name="${ACCOUNT}-nemorl.q235-v025-numa-${NUMA_MEMBIND_MODE}"
  --output="${RUN_DIR}/slurm-%j.out"
  --comment=metrics
)

if [[ "${DRY_RUN}" == "1" ]]; then
  sbatch_args+=(--test-only)
fi

job_id="$({
  cd "${REPO_DIR}"
  sbatch "${sbatch_args[@]}" ray.sub
})"

printf '%s\n' "job_id=${job_id}" "run_dir=${RUN_DIR}"
