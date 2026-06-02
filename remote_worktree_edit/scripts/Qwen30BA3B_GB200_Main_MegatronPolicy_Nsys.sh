#!/bin/bash
set -euo pipefail

# Reusable NeMo-RL Nsight Systems launcher for Qwen3-30B-A3B on main.
#
# Default profile:
#   - recipe: examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml
#   - nodes: 4, GPUs/node: 4
#   - GRPO max_steps: 10
#   - nsys worker: megatron_policy_worker
#   - nsys step range: 2:5
#   - recipe shape: num_prompts_per_step=64, num_generations_per_prompt=32,
#     train_global_batch_size=2048
#
# Run on the OCI login node:
#   cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-main-20260527-online-draft
#   bash Qwen30BA3B_GB200_Main_MegatronPolicy_Nsys.sh
#
# Common overrides:
#   CONTAINER=/path/to/nemo_rl_nightly_YYYYMMDD.sqsh bash Qwen30BA3B_GB200_Main_MegatronPolicy_Nsys.sh
#   NRL_NSYS_PROFILE_STEP_RANGE=5:8 MAX_STEPS=12 bash Qwen30BA3B_GB200_Main_MegatronPolicy_Nsys.sh
#   DRY_RUN=1 bash Qwen30BA3B_GB200_Main_MegatronPolicy_Nsys.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEMO_RL_DIR="${NEMO_RL_DIR:-${SCRIPT_DIR}}"
CONFIG_FILE="${CONFIG_FILE:-examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml}"
RAY_SUB="${RAY_SUB:-${NEMO_RL_DIR}/ray_oci.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  RAY_SUB="${NEMO_RL_DIR}/ray.sub"
fi

ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SEGMENT="${SEGMENT:-${NUM_NODES}}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
GRES_FLAG="${GRES_FLAG:---gres=gpu:${GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((GPUS_PER_NODE * 16))}"
SBATCH_RESOURCE_ARGS="${SBATCH_RESOURCE_ARGS:---ntasks-per-node=1 --cpus-per-task=${CPUS_PER_TASK} --mem=0}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS * NUM_GENERATIONS))}"
MAX_STEPS="${MAX_STEPS:-10}"
NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-2:5}"
NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-megatron_policy_worker}"
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"

UV_PYTHON="${UV_PYTHON:-3.13.13}"
RAY_PYTHON_VERSION="${RAY_PYTHON_VERSION:-3.13.13}"
RAY_VERSION="${RAY_VERSION:-2.54.0}"
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT:-${NEMO_RL_DIR}/.driver_venvs/qwen30ba3b_main_megatron_nsys_py313}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${NEMO_RL_DIR}/nrl_megatron_ckpts_qwen30ba3b_main_nsys}"

CONTAINER="${CONTAINER:-${NEMO_RL_DIR}/nemo_rl_nightly_20260602.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"
MOUNTS="${MOUNTS:-/lustre:/lustre}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${NEMO_RL_DIR}/logs/qwen30ba3b_grpo_nsys_main_megatron_step2to5_max10}"
WANDB_PROJECT="${WANDB_PROJECT:-sync-grpo-gb200_oci-benchmark}"
WANDB_NAME="${WANDB_NAME:-Qwen30B_A3B_Main_N${NUM_NODES}xG${GPUS_PER_NODE}_main_megatron_policy_nsys_s${NRL_NSYS_PROFILE_STEP_RANGE/:/to}_ms${MAX_STEPS}_gbs${TRAIN_GLOBAL_BATCH_SIZE}}"
JOB_TAG="${JOB_TAG:-main-megpolicy-nsys-s${NRL_NSYS_PROFILE_STEP_RANGE/:/to}-ms${MAX_STEPS}-gbs${TRAIN_GLOBAL_BATCH_SIZE}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}" "${BASE_LOG_DIR}"

if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: Ray sbatch template not found: ${RAY_SUB}" >&2
  exit 2
fi

if [[ ! -s "${CONTAINER}" ]]; then
  cat >&2 <<EOF
ERROR: container not found or empty: ${CONTAINER}

For current NeMo-RL main, use a CUDA13-capable nightly image. The old
20260502 image exposes CUDA 12.9 and can fail while building CUDA13 deps.
Override CONTAINER=... if the nightly image has a different name.
EOF
  exit 2
fi

read -r -d '' COMMAND <<EOF || true
set -euo pipefail
export NRL_FORCE_REBUILD_VENVS=true
export UV_PYTHON=${UV_PYTHON}
export UV_PROJECT_ENVIRONMENT=${DRIVER_UV_PROJECT_ENVIRONMENT}
export PYTHONPATH=${NEMO_RL_DIR}:\${PYTHONPATH:-}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR}
export NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE}
export NRL_NSYS_WORKER_PATTERNS=${NRL_NSYS_WORKER_PATTERNS}
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}

echo "[NSYS] workdir=${NEMO_RL_DIR}"
echo "[NSYS] config=${CONFIG_FILE}"
echo "[NSYS] profile_step_range=${NRL_NSYS_PROFILE_STEP_RANGE}"
echo "[NSYS] worker_patterns=${NRL_NSYS_WORKER_PATTERNS}"
echo "[NSYS] max_steps=${MAX_STEPS}"
echo "[NSYS] NRL_FORCE_REBUILD_VENVS=true"

RUN_ARGS=(
  --config ${CONFIG_FILE}
  cluster.num_nodes=${NUM_NODES}
  cluster.gpus_per_node=${GPUS_PER_NODE}
  grpo.async_grpo.enabled=false
  grpo.val_period=1000
  checkpointing.enabled=false
  grpo.num_prompts_per_step=${NUM_PROMPTS}
  grpo.num_generations_per_prompt=${NUM_GENERATIONS}
  policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}
  policy.sequence_packing.enabled=true
  policy.megatron_cfg.moe_enable_deepep=false
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall
  grpo.max_num_steps=${MAX_STEPS}
  logger.wandb_enabled=true
  logger.wandb.project=${WANDB_PROJECT}
  logger.wandb.name=${WANDB_NAME}
)
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_OVERRIDES} )
  RUN_ARGS+=("\${EXTRA_ARGS[@]}")
fi

uv run --python ${UV_PYTHON} --locked --extra mcore --directory ${NEMO_RL_DIR} python ./examples/run_grpo.py "\${RUN_ARGS[@]}"
EOF

echo "=========================================="
echo "Submitting Qwen3-30B-A3B Megatron policy nsys run"
echo "  NEMO_RL_DIR=${NEMO_RL_DIR}"
echo "  branch/commit: $(cd "${NEMO_RL_DIR}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown) $(cd "${NEMO_RL_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  CONFIG_FILE=${CONFIG_FILE}"
echo "  CONTAINER=${CONTAINER}"
echo "  NUM_NODES=${NUM_NODES}, GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "  NUM_PROMPTS=${NUM_PROMPTS}, NUM_GENERATIONS=${NUM_GENERATIONS}, GBS=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE}"
echo "  NRL_NSYS_WORKER_PATTERNS=${NRL_NSYS_WORKER_PATTERNS}"
echo "  BASE_LOG_DIR=${BASE_LOG_DIR}"
echo "  WANDB_NAME=${WANDB_NAME}"
echo "=========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] COMMAND:"
  printf '%s\n' "${COMMAND}"
  echo "[DRY_RUN] sbatch template: ${RAY_SUB}"
  exit 0
fi

cd "${NEMO_RL_DIR}"

CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
BASE_LOG_DIR="${BASE_LOG_DIR}" \
RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY}" \
RAY_PYTHON_VERSION="${RAY_PYTHON_VERSION}" \
RAY_VERSION="${RAY_VERSION}" \
UV_PYTHON="${UV_PYTHON}" \
DRIVER_UV_PROJECT_ENVIRONMENT="${DRIVER_UV_PROJECT_ENVIRONMENT}" \
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE}" \
NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
COMMAND="${COMMAND}" \
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="qwen30ba3b-${JOB_TAG}-N${NUM_NODES}xG${GPUS_PER_NODE}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  ${GRES_FLAG} \
  ${SBATCH_RESOURCE_ARGS} \
  ${SBATCH_EXTRA_ARGS} \
  --segment "${SEGMENT}" \
  "${RAY_SUB}"
