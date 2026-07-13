#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-36x2-a01r}"
NUM_NODES="${NUM_NODES:-16}"
SEGMENT="${SEGMENT:-16}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260704.sqsh}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/coreai_dlalgo_llm/users/sna/RL-vllm024-fresh-main-ad23-20260712}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
RUN_TAG="${RUN_TAG:-q235-v024-ref-init-diag-ptyche-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/experiments/vllm_024_fresh_main_oom/runs/${RUN_TAG}}"

PRETRAINED_CHECKPOINT="${NRL_MEGATRON_CHECKPOINT_DIR}/Qwen/Qwen3-235B-A22B/iter_0000000"
for marker in metadata.json run_config.yaml; do
  if [[ ! -f "${PRETRAINED_CHECKPOINT}/${marker}" ]]; then
    printf 'Missing shared pretrained checkpoint marker: %s\n' \
      "${PRETRAINED_CHECKPOINT}/${marker}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_DIR}/torch_nccl"

COMMAND="env \
WANDB_RUN_GROUP=${RUN_TAG} \
WANDB_RESUME=never \
NRL_DISABLE_VLLM_PORT_OVERRIDE=1 \
NRL_DEBUG_REFERENCE_MODEL_SETUP=1 \
NRL_REFERENCE_SETUP_STACK_DUMP_SECONDS=600 \
NRL_MEGATRON_NCCL_TIMEOUT_SECONDS=1800 \
PYTHONFAULTHANDLER=1 \
NCCL_DEBUG=WARN \
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_DESYNC_DEBUG=1 \
TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=60000 \
TORCH_FR_DUMP_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_ \
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${RUN_DIR}/torch_nccl/trace_rank_ \
TORCH_INCLUDE_STACK_TRACE=1 \
TORCH_INCLUDE_ONLY_ACTIVE=0 \
HF_HOME=${HF_HOME} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
PYTHONPATH=${REPO_DIR} \
NEMO_RL_VENV_DIR=/tmp/nemorl-v024-ref-init-diag-${RUN_TAG} \
NRL_FORCE_REBUILD_VENVS=true \
UV_CACHE_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/uv_cache \
UV_LOCK_TIMEOUT=900 \
TRITON_CACHE_DIR=/tmp/nemorl-v024-ref-init-diag-triton-${RUN_TAG} \
TORCHINDUCTOR_CACHE_DIR=/tmp/nemorl-v024-ref-init-diag-inductor-${RUN_TAG} \
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml \
grpo.max_num_steps=20 \
checkpointing.enabled=false \
checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints \
policy.generation.vllm_cfg.enforce_eager=false \
logger.wandb_enabled=false \
logger.tensorboard_enabled=false \
logger.log_dir=${RUN_DIR}/nemo_logs"

export COMMAND CONTAINER MOUNTS

job_id="$({
  cd "${REPO_DIR}"
  sbatch \
    --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --exclusive \
    --time="${TIME_LIMIT}" \
    --segment="${SEGMENT}" \
    --dependency= \
    --job-name="${ACCOUNT}-nemorl.q235-ref-init-diag" \
    --output="${RUN_DIR}/slurm-%j.out" \
    --comment=metrics \
    ray.sub
})"

printf '%s\n' "job_id=${job_id}" "run_dir=${RUN_DIR}"
