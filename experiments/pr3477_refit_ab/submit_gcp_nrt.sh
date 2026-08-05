#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
MODE=${MODE:?MODE is required: legacy or nccl}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
TOTAL_NODES=${TOTAL_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GEN_NODES=${GEN_NODES:-2}
MAX_STEPS=${MAX_STEPS:-20}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
RUNTIME_ROOT=${RUNTIME_ROOT:-${WORK_ROOT}/containers/nemo-rl-nightly-refresh}
CONTAINER=${CONTAINER:-${RUNTIME_ROOT}/nemo_rl_nightly_20260730_483099.sqsh}
PYTHON_OVERLAY=${PYTHON_OVERLAY:-${RUNTIME_ROOT}/python-overlay-483099}
ROOT_CACHE_OVERLAY=${ROOT_CACHE_OVERLAY:-${RUNTIME_ROOT}/root-cache-overlay-483099}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/pr3477-refit-ab/${MODE}}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/pr3477-refit-ab/results/${MODE}-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3477-refit-ab}
WANDB_NAME=${WANDB_NAME:-${MODE}-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

case "${MODE}" in
  legacy) REFIT_TRANSPORT=null ;;
  nccl) REFIT_TRANSPORT=nccl_reshard ;;
  *) echo "MODE must be legacy or nccl" >&2; exit 2 ;;
esac

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

if (( TOTAL_NODES != 4 || GPUS_PER_NODE != 8 || GEN_NODES != 2 )); then
  echo "The reportable A/B requires TOTAL_NODES=4, GPUS_PER_NODE=8, GEN_NODES=2" >&2
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
  "${CONTAINER}" \
  "${PYTHON_OVERLAY}" \
  "${ROOT_CACHE_OVERLAY}"; do
  test -e "${path}"
done

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}" "${WORK_ROOT}/.cache/huggingface"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  (umask 077; printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}")
elif [[ ! -s "${WANDB_KEY_FILE}" && -f "${HOME}/.netrc" ]]; then
  (umask 077; awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' \
    "${HOME}/.netrc" >"${WANDB_KEY_FILE}")
fi
test -s "${WANDB_KEY_FILE}"

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
mode=${MODE}
repo_sha=${REPO_SHA}
config=${CONFIG}
refit_transport=${REFIT_TRANSPORT}
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
generation_nodes=${GEN_NODES}
max_steps=${MAX_STEPS}
container=${CONTAINER}
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${WORK_ROOT}/.cache/huggingface
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=${CACHE_ROOT}/worker-venvs
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
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
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  policy.generation.vllm_cfg.expert_parallel_size=1 \
  policy.generation.vllm_cfg.use_tqdm=false \
  policy.train_global_batch_size=2048 \
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
export MOUNTS=/lustre:/lustre,${PYTHON_OVERLAY}:/root/.local/share/uv/python,${ROOT_CACHE_OVERLAY}:/root/.cache
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
  --job-name="sna-p3477-${MODE}-${MAX_STEPS}s"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"environment and Qwen3-30B initialization"}}'
)

printf 'mode=%s\nrepo=%s\nsha=%s\nresult=%s\n' \
  "${MODE}" "${REPO}" "${REPO_SHA}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
