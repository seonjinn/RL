#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:4}"
SEGMENT="${SEGMENT:-${NUM_NODES}}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_WORKER} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
JOB_TAG="${JOB_TAG:-nsys-seqpack-alltoall-p${NUM_PROMPTS:-8}g${NUM_GENERATIONS:-8}-s5to7}"

CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml}"
CONTAINER="${CONTAINER:-${SCRIPT_DIR}/nemo_rl_nightly.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${SCRIPT_DIR}/nrl_megatron_ckpts_20260528_qwen30ba3b_seqpack_alltoall}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
UV_PYTHON="${UV_PYTHON:-3.12.13}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.driver_venvs/qwen30ba3b_nsys_seqpack_alltoall_py312}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-8}"
NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-5:8}"
NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-megatron_policy_worker}"
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=true \
UV_PYTHON=${UV_PYTHON} \
UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT} \
PYTHONPATH=${SCRIPT_DIR}:${PYTHONPATH:-} \
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE} \
NRL_NSYS_WORKER_PATTERNS=${NRL_NSYS_WORKER_PATTERNS} \
LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} \
uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${SCRIPT_DIR} python ./examples/run_grpo.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=${NUM_PROMPTS} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS} \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
policy.sequence_packing.enabled=true \
policy.megatron_cfg.moe_enable_deepep=false \
policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
grpo.max_num_steps=${MAX_STEPS} \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='Qwen30B_A3B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_seqpack_alltoall_nsys_p${NUM_PROMPTS}_g${NUM_GENERATIONS}_${MAX_STEPS}step'"

CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE}" \
NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS}" \
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY}" \
UV_PYTHON="${UV_PYTHON}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
MOUNTS="${MOUNTS}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen30ba3b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time=02:00:00 \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  ray.sub
