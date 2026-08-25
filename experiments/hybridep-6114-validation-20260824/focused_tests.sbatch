#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=hybridep-6114-focused

set -euo pipefail

readonly EXPECTED_MEGATRON_LM_COMMIT=377ad24cd05f41686fafe2e6747f47678b8581c8

: "${CONTAINER:?Set CONTAINER to the staged immutable .sqsh path}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the validation checkout}"
: "${RESULTS_DIR:?Set RESULTS_DIR to a shared result directory}"

megatron_lm_root="${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
megatron_lm_commit=$(git -C "${megatron_lm_root}" rev-parse HEAD)
if [[ ${megatron_lm_commit} != "${EXPECTED_MEGATRON_LM_COMMIT}" ]]; then
  echo "Expected Megatron-LM ${EXPECTED_MEGATRON_LM_COMMIT}, got ${megatron_lm_commit}" >&2
  exit 1
fi

node_local_root="/raid/scratch/${USER}/nemo-rl-hybridep-6114-focused-${SLURM_JOB_ID}"
main_venv="${node_local_root}/main-venv"
mkdir -p "${RESULTS_DIR}" "${node_local_root}"

export UV_PROJECT_ENVIRONMENT="${main_venv}"
export NEMO_RL_VENV_DIR="${node_local_root}/worker-venvs"
export NRL_FORCE_REBUILD_VENVS=true
export HYBRID_EP_MULTINODE=1
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_CUDA_ARCHS=90
export MAX_JOBS=8

mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${RESULTS_DIR}:${RESULTS_DIR},/raid/scratch:/raid/scratch"
test_command=$(printf '%q ' \
  bash -c '
    set -euo pipefail
    export TMS_CUDA_MAJOR="$(${CUDA_HOME:-/usr/local/cuda}/bin/nvcc --version | sed -n '\''s/.*release \([0-9][0-9]*\).*/\1/p'\'' | head -1)"
    test -n "${TMS_CUDA_MAJOR}"
    bash "${PROJECT_ROOT}/experiments/hybridep-6114-validation-20260824/run_focused_tests.sh"
  ')

export PROJECT_ROOT
export MEGATRON_LM_ROOT="${megatron_lm_root}"
export RESULTS_DIR

srun \
  --gpus-per-node=1 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${mounts}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "${test_command}" \
  2>&1 | tee "${RESULTS_DIR}/focused-tests.log"
