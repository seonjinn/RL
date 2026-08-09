#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
MODE=${MODE:?MODE is required: legacy or nccl}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
TOTAL_NODES=${TOTAL_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GEN_NODES=${GEN_NODES:-4}
VLLM_TP=${VLLM_TP:-4}
VLLM_PP=${VLLM_PP:-2}
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-4}
TRAIN_CP=${TRAIN_CP:-2}
TRAIN_EP=${TRAIN_EP:-8}
TRAIN_ETP=${TRAIN_ETP:-1}
MAX_STEPS=${MAX_STEPS:-20}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false}
VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
RUNTIME_ROOT=${RUNTIME_ROOT:-${WORK_ROOT}/containers/nemo-rl-nightly-cw-fallback-20260808}
CONTAINER=${CONTAINER:-${RUNTIME_ROOT}/nemo_rl_nightly_20260805_15171871.sqsh}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/pr3477-qwen235b-refit-ab/${RUN_SUFFIX}/${MODE}}
WORKER_VENV_TAG=${WORKER_VENV_TAG:-${RUN_SUFFIX}}
WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-${CACHE_ROOT}/worker-venvs/${WORKER_VENV_TAG}}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/pr3477-qwen235b-refit-ab/results/${RUN_SUFFIX}/${MODE}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3477-qwen235b-refit-ab}
WANDB_NAME=${WANDB_NAME:-qwen235b-${MODE}-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml

case "${MODE}" in
  legacy) REFIT_TRANSPORT=null ;;
  nccl) REFIT_TRANSPORT=nccl_reshard ;;
  *) echo "MODE must be legacy or nccl" >&2; exit 2 ;;
esac

case "${ACTION}" in
  dry-run) SBATCH_ACTION=(--test-only); PRINT_ONLY=1 ;;
  test-only) SBATCH_ACTION=(--test-only); PRINT_ONLY=0 ;;
  submit) SBATCH_ACTION=(); PRINT_ONLY=0 ;;
  *) echo "ACTION must be dry-run, test-only, or submit" >&2; exit 2 ;;
esac

if (( TOTAL_NODES != 8 || GPUS_PER_NODE != 8 || GEN_NODES != 4 )); then
  echo "The reportable A/B requires 8 nodes, 8 GPUs/node, and 4 generation nodes" >&2
  exit 2
fi
if (( VLLM_TP * VLLM_PP != GPUS_PER_NODE )); then
  echo "The B200 A/B requires one vLLM engine per node (VLLM_TP*VLLM_PP=${GPUS_PER_NODE})" >&2
  exit 2
fi

TRAIN_WORLD_SIZE=$(((TOTAL_NODES - GEN_NODES) * GPUS_PER_NODE))
GEN_WORLD_SIZE=$((GEN_NODES * GPUS_PER_NODE))
DENSE_MODEL_PARALLEL_SIZE=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))
EXPERT_MODEL_PARALLEL_SIZE=$((TRAIN_ETP * TRAIN_EP * TRAIN_PP))
if (( TRAIN_WORLD_SIZE % DENSE_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Trainer world size ${TRAIN_WORLD_SIZE} is not divisible by TP*PP*CP=${DENSE_MODEL_PARALLEL_SIZE}" >&2
  exit 2
fi
if (( TRAIN_WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Trainer world size ${TRAIN_WORLD_SIZE} is not divisible by ETP*EP*PP=${EXPERT_MODEL_PARALLEL_SIZE}" >&2
  exit 2
fi

git -C "${REPO}" pull --ff-only
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi

REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
for path in \
  "${REPO}/${CONFIG}" \
  "${REPO}/ray.sub" \
  "${CONTAINER}"; do
  test -e "${path}"
done

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/.cache/huggingface"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ -f "${HOME}/.netrc" ]]; then
  (umask 077; awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
      }
    }
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  (umask 077; printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}")
fi
test -s "${WANDB_KEY_FILE}"

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
mode=${MODE}
repo_sha=${REPO_SHA}
config=${CONFIG}
refit_transport=${REFIT_TRANSPORT}
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
trainer_nodes=$((TOTAL_NODES - GEN_NODES))
generation_nodes=${GEN_NODES}
generation_world_size=${GEN_WORLD_SIZE}
generation_tensor_parallel_size=${VLLM_TP}
generation_pipeline_parallel_size=${VLLM_PP}
generation_data_parallel_size=$((GEN_WORLD_SIZE / (VLLM_TP * VLLM_PP)))
trainer_world_size=${TRAIN_WORLD_SIZE}
trainer_tp=${TRAIN_TP}
trainer_pp=${TRAIN_PP}
trainer_cp=${TRAIN_CP}
trainer_ep=${TRAIN_EP}
trainer_etp=${TRAIN_ETP}
max_steps=${MAX_STEPS}
seed=42
train_global_batch_size=512
container=${CONTAINER}
driver_runtime=uv_run_frozen
worker_venv_dir=${WORKER_VENV_ROOT}
vllm_allreduce_use_symm_mem=${VLLM_ALLREDUCE_USE_SYMM_MEM}
vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${WORK_ROOT}/.cache/huggingface
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS}
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export NCCL_NVLS_ENABLE=0
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export RAY_CGRAPH_get_timeout=2400
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_LOCK_TIMEOUT=7200
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}
export WANDB_API_KEY="\$(cat ${WANDB_KEY_FILE})"
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.segment_size=1 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  policy.generation.refit_transport=${REFIT_TRANSPORT} \
  policy.megatron_cfg.tensor_model_parallel_size=${TRAIN_TP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${TRAIN_PP} \
  policy.megatron_cfg.context_parallel_size=${TRAIN_CP} \
  policy.megatron_cfg.expert_tensor_parallel_size=${TRAIN_ETP} \
  policy.megatron_cfg.expert_model_parallel_size=${TRAIN_EP} \
  policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
  policy.megatron_cfg.moe_flex_dispatcher_backend=deepep \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.pipeline_parallel_size=${VLLM_PP} \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.use_tqdm=false \
  +policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  policy.train_global_batch_size=512 \
  grpo.seed=42 \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=true \
  ++logger.wandb.entity=${WANDB_ENTITY} \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE
export BASE_LOG_DIR=${EXPERIMENT_ROOT}

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --gpus-per-node="${GPUS_PER_NODE}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --job-name="sna-p3477-235b-${MODE}-${MAX_STEPS}s"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"environment and Qwen3-235B initialization"}}'
)

printf 'mode=%s\nrepo=%s\nsha=%s\nresult=%s\n' \
  "${MODE}" "${REPO}" "${REPO_SHA}" "${EXPERIMENT_ROOT}"

if (( PRINT_ONLY == 1 )); then
  printf 'SBATCH:'
  printf ' %q' sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
  printf '\nCOMMAND:\n%s\n' "${COMMAND}"
  exit 0
fi

exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
