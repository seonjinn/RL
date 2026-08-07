#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

ACTION=${ACTION:-test-only}
ACCOUNT=${SLURM_ACCOUNT:-coreai_chef_posttrain}
PARTITION=${PARTITION:-batch}
TOTAL_NODES=${TOTAL_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GEN_NODES=${GEN_NODES:-2}
MAX_STEPS=${MAX_STEPS:-2}
WALLTIME=${WALLTIME:-04:00:00}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

WORK_ROOT=${WORK_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna}
RUNTIME_ROOT=${RUNTIME_ROOT:-${WORK_ROOT}/containers/nemo-rl-nightly-refresh}
CONTAINER=${CONTAINER:-${RUNTIME_ROOT}/nemo_rl_nightly_20260730_483099.sqsh}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/mopd_nano_fast/.cache/pr3477-nano-smoke/exact-head}
PYTHON_OVERLAY=${PYTHON_OVERLAY:-${CACHE_ROOT}/uv-python}
ROOT_CACHE_OVERLAY=${ROOT_CACHE_OVERLAY:-${RUNTIME_ROOT}/root-cache-overlay-483099}
UV_BIN_DIR=${UV_BIN_DIR:-${WORK_ROOT}/tools/uv-current}
MODEL_PATH=${MODEL_PATH:-${WORK_ROOT}/nemo-evaluator-rundirs/nano_v35/conversions/Ultra-SFTb2-512K-hermes20k-lr2e-5-iter_0005000/hf}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${WORK_ROOT}/experiments/pr3477-nano-smoke/results/${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_PROJECT=${WANDB_PROJECT:-sna-pr3477-nano-smoke}
WANDB_NAME=${WANDB_NAME:-nano-bf16-train-mxfp8-rollout-nccl-${MAX_STEPS}step-${RUN_SUFFIX}}
WANDB_ENTITY=${WANDB_ENTITY:-nvidia}
CONFIG=examples/configs/recipes/llm/grpo-nanov3-30BA3B-4n8g-megatron-mxfp8-rollout-nccl-smoke.yaml

case "${ACTION}" in
  submit) SBATCH_ACTION=() ;;
  test-only) SBATCH_ACTION=(--test-only) ;;
  *) echo "ACTION must be submit or test-only" >&2; exit 2 ;;
esac

if (( TOTAL_NODES != 4 || GPUS_PER_NODE != 8 || GEN_NODES != 2 )); then
  echo "This smoke requires four 8-GPU nodes split into two trainer and two generation nodes" >&2
  exit 2
fi

git -C "${REPO}" pull --ff-only
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"

for path in \
  "${REPO}/${CONFIG}" \
  "${REPO}/ray.sub" \
  "${REPO}/3rdparty/Gym-workspace/Gym/pyproject.toml" \
  "${CONTAINER}" \
  "${PYTHON_OVERLAY}" \
  "${ROOT_CACHE_OVERLAY}" \
  "${UV_BIN_DIR}/uv" \
  "${MODEL_PATH}"; do
  test -e "${path}"
done

REPO_SHA=$(git -C "${REPO}" rev-parse HEAD)
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
repo_sha=${REPO_SHA}
config=${CONFIG}
model_path=${MODEL_PATH}
training_precision=bfloat16
rollout_precision=mxfp8
refit_transport=nccl_reshard
total_nodes=${TOTAL_NODES}
gpus_per_node=${GPUS_PER_NODE}
generation_nodes=${GEN_NODES}
max_steps=${MAX_STEPS}
container=${CONTAINER}
python_overlay=${PYTHON_OVERLAY}
uv_version=$(${UV_BIN_DIR}/uv --version)
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${WORK_ROOT}/.cache/huggingface
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${CACHE_ROOT}/worker-venvs
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export PATH=${UV_BIN_DIR}:\${PATH}
export UV_CACHE_DIR=/root/.cache/uv
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
export WANDB_API_KEY="\$(cat ${WANDB_KEY_FILE})"
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
uv run --frozen examples/run_grpo.py \
  --config ${CONFIG} \
  policy.model_name=${MODEL_PATH} \
  policy.tokenizer.name=${MODEL_PATH} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
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
  --job-name="sna-p3477-nano-${MAX_STEPS}s"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"environment and Nemotron Nano initialization"}}'
)

printf 'repo=%s\nsha=%s\nresult=%s\n' "${REPO}" "${REPO_SHA}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
