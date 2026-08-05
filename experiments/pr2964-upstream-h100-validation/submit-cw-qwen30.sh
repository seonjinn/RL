#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(git rev-parse --show-toplevel)
EXPECTED_BRANCH=sna/pr2964-upstream-h100-validation-20260805
EXPECTED_BRIDGE_COMMIT=573e088c9c6740082c39744e03dc5b009e730ed4
EXPECTED_MCORE_COMMIT=6513e3e23d6b5eda6a1c934990b15e804237732b
DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685

ACCOUNT=coreai_dlalgo_nemorl
PARTITION=batch
NUM_ACTOR_NODES=4
GPUS_PER_NODE=8
SEGMENT_SIZE=4
MAX_STEPS=${MAX_STEPS:-5}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}

EXPERIMENT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/pr2964-upstream-h100-validation
RUN_NAME=${RUN_NAME:-qwen3-30ba3b-cw-h100-4n8g-${MAX_STEPS}step-upstream-only-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${EXPERIMENT_ROOT}/runs/${RUN_NAME}
ACTOR_VENV_ROOT=${RUN_ROOT}/actor-venvs
DRIVER_VENV=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-x86-b200-h100/cw-dfw/driver-venv
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-20260727/nemo_rl_nightly_20260727_14418344.sqsh
DEEPEP_WHEEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-x86-b200-h100/cw-dfw/q30-triplet-main-d3f3eb5-20260729/artifacts/deepep/hybridep-f725d29699f5bda9ba789456bb9579af69844685-sm90-14636236/deep_ep-1.2.1+f725d29-cp313-cp313-linux_x86_64.whl
NCCL_WHEEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-x86-b200-h100/cw-dfw/q30-triplet-main-d3f3eb5-20260729/artifacts/nccl/nvidia-nccl-cu13-2.30.4-14633389/nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_18_x86_64.whl
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
OVERLAY=/tmp/nemo-rl-pr2964-upstream-h100

BRIDGE_ROOT=${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE_ROOT=${BRIDGE_ROOT}/3rdparty/Megatron-LM
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g-hybridep-upstream-validation.yaml

if [[ $(git branch --show-current) != "${EXPECTED_BRANCH}" ]]; then
  printf 'Expected branch %s; found %s\n' "${EXPECTED_BRANCH}" "$(git branch --show-current)" >&2
  exit 2
fi
if [[ -n $(git status --porcelain) ]]; then
  printf 'Refusing to submit a dirty checkout.\n' >&2
  git status --short >&2
  exit 2
fi

git pull --ff-only --recurse-submodules=no
git submodule sync --recursive
git submodule update --init --recursive

BRIDGE_COMMIT=$(git -C "${BRIDGE_ROOT}" rev-parse HEAD)
MCORE_COMMIT=$(git -C "${MCORE_ROOT}" rev-parse HEAD)
BRIDGE_URL=$(git -C "${BRIDGE_ROOT}" remote get-url origin)
MCORE_URL=$(git -C "${MCORE_ROOT}" remote get-url origin)
if [[ ${BRIDGE_COMMIT} != "${EXPECTED_BRIDGE_COMMIT}" ]]; then
  printf 'Unexpected Bridge commit: %s\n' "${BRIDGE_COMMIT}" >&2
  exit 2
fi
if [[ ${MCORE_COMMIT} != "${EXPECTED_MCORE_COMMIT}" ]]; then
  printf 'Unexpected Megatron-LM commit: %s\n' "${MCORE_COMMIT}" >&2
  exit 2
fi
if [[ ${BRIDGE_URL} != https://github.com/NVIDIA-NeMo/Megatron-Bridge.git ]]; then
  printf 'Unexpected Bridge URL: %s\n' "${BRIDGE_URL}" >&2
  exit 2
fi
if [[ ${MCORE_URL} != https://github.com/NVIDIA/Megatron-LM.git ]]; then
  printf 'Unexpected Megatron-LM URL: %s\n' "${MCORE_URL}" >&2
  exit 2
fi

for artifact in "${CONTAINER}" "${DRIVER_VENV}/bin/python" "${DEEPEP_WHEEL}" "${NCCL_WHEEL}"; do
  if [[ ! -e ${artifact} ]]; then
    printf 'Required artifact is missing: %s\n' "${artifact}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/ray"

read -r -d '' SETUP_COMMAND <<SETUP_EOF || true
set -euo pipefail
OVERLAY=${OVERLAY}
DRIVER_PYTHON=${DRIVER_VENV}/bin/python
if [[ \$(dirname -- "\${OVERLAY}") != /tmp || \$(basename -- "\${OVERLAY}") != nemo-rl-pr2964-upstream-h100 ]]; then
  printf 'Unsafe overlay path: %s\n' "\${OVERLAY}" >&2
  exit 2
fi
rm -rf -- "\${OVERLAY}"
mkdir -p "\${OVERLAY}"
UV_NO_CONFIG=1 uv pip install --python "\${DRIVER_PYTHON}" --target "\${OVERLAY}" --reinstall --no-deps --no-index "${NCCL_WHEEL}" "${DEEPEP_WHEEL}"
PYTHONPATH="\${OVERLAY}:${PROJECT_ROOT}:${BRIDGE_ROOT}/src:${MCORE_ROOT}" "\${DRIVER_PYTHON}" -c "import deep_ep, deep_ep_cpp, hybrid_ep_cpp; from deep_ep import HybridEPBuffer; print('UPSTREAM_ONLY_HYBRIDEP_IMPORT_OK', deep_ep.__file__)"
SETUP_EOF

read -r -d '' COMMAND <<COMMAND_EOF || true
set -euo pipefail
export PYTHONPATH=${OVERLAY}:${PROJECT_ROOT}:${BRIDGE_ROOT}/src:${MCORE_ROOT}:\${PYTHONPATH:-}
export LD_LIBRARY_PATH=${OVERLAY}/nvidia/nccl/lib:\${LD_LIBRARY_PATH:-}
export UV_NO_SYNC=1
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${ACTOR_VENV_ROOT}
${DRIVER_VENV}/bin/python -c "import nemo_rl, megatron.bridge, megatron.core; print('UPSTREAM_ONLY_STACK_IMPORT_OK', nemo_rl.__file__, megatron.bridge.__file__, megatron.core.__file__)"
if ${DRIVER_VENV}/bin/python -c 'import pytest' >/dev/null 2>&1; then
  ${DRIVER_VENV}/bin/python -m pytest -q tests/unit/models/megatron/test_megatron_data.py
else
  printf 'PYTEST_SKIPPED driver environment does not include pytest\n'
fi
env UV_PROJECT_ENVIRONMENT=${DRIVER_VENV} uv run --no-sync examples/run_grpo.py \
  --config ${CONFIG} \
  grpo.max_num_steps=${MAX_STEPS} \
  checkpointing.enabled=false \
  logger.log_dir=${RUN_ROOT}/training \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true \
  ++deepep_override=${DEEPEP_COMMIT}
COMMAND_EOF

export CONTAINER
export HF_HOME
export HF_DATASETS_CACHE=${HF_HOME}/cache
export MOUNTS=${PROJECT_ROOT}:${PROJECT_ROOT},/lustre:/lustre
export COMMAND
export SETUP_COMMAND
export RAY_VENV=${DRIVER_VENV}
export NEMO_RL_VENV_DIR=${ACTOR_VENV_ROOT}
export NRL_FORCE_REBUILD_VENVS=true
export BASE_LOG_DIR=${RUN_ROOT}/ray
export NCCL_NVLS_ENABLE=0
export UV_NO_SYNC=1

printf '%s\n' \
  "submitted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "run_name=${RUN_NAME}" \
  "rl_commit=$(git rev-parse HEAD)" \
  "bridge_url=${BRIDGE_URL}" \
  "bridge_commit=${BRIDGE_COMMIT}" \
  "megatron_lm_url=${MCORE_URL}" \
  "megatron_lm_commit=${MCORE_COMMIT}" \
  "deepep_commit=${DEEPEP_COMMIT}" \
  "container=${CONTAINER}" \
  "config=${CONFIG}" \
  "nodes=${NUM_ACTOR_NODES}" \
  "gpus_per_node=${GPUS_PER_NODE}" \
  "max_steps=${MAX_STEPS}" > "${RUN_ROOT}/submission.env"

SBATCH_ARGS=(
  --nodes="${NUM_ACTOR_NODES}"
  --account="${ACCOUNT}"
  --job-name="${ACCOUNT}.pr2964-upstream-q30"
  --partition="${PARTITION}"
  --time="${TIME_LIMIT}"
  --gres="gpu:${GPUS_PER_NODE}"
  --segment="${SEGMENT_SIZE}"
  --output="${RUN_ROOT}/slurm-%j.out"
)

if [[ ${1:-} == --test-only ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" ray.sub
else
  sbatch "${SBATCH_ARGS[@]}" ray.sub
fi
