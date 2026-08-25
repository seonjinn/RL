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
uv_cache="${node_local_root}/uv-cache"
mkdir -p "${RESULTS_DIR}" "${node_local_root}" "${uv_cache}"

export UV_PROJECT_ENVIRONMENT="${main_venv}"
export UV_CACHE_DIR="${uv_cache}"
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
    cd "${PROJECT_ROOT}"
    uv run --locked pytest -q \
      tests/unit/models/megatron/test_megatron_setup.py -k hybridep
    uv run --locked pytest -q \
      tests/unit/models/megatron/test_megatron_data.py -k "alignment_padding or pad_full_seq_to"
    uv run --locked pytest -q \
      tests/unit/tools/test_hybridep_default_8g_recipes.py
    cd "${MEGATRON_LM_ROOT}"
    PYTHONPATH="${MEGATRON_LM_ROOT}:${PYTHONPATH:-}" \
      "${UV_PROJECT_ENVIRONMENT}/bin/python" -m pytest -q \
      "tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_expert_bias_token_counts_with_padding_mask"
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
