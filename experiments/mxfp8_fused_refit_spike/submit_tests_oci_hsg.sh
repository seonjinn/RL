#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3294-fused-refit-spike-20260823}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-microbench-20260823}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/experiments/mxfp8-fused-refit-spike}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/tests}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/sna}
VENV_KEY=${VENV_KEY:-v0251}
RUN_WORKER_TESTS=${RUN_WORKER_TESTS:-1}

if [[ "${ACTION}" == render ]]; then
  printf 'tests=MXFP8-shuffle-parity,pointer-stability\nhardware=GB200\n'
  exit 0
fi
case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

git -C "${REPO}" fetch origin "${BRANCH}"
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
git -C "${REPO}" checkout --detach "${REMOTE_HEAD}"
git -C "${REPO}" submodule update --init --recursive
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -e "${CONTAINER}"
mkdir -p "${RUN_ROOT}"

VLLM_VENV=${VLLM_VENV:-${LOCAL_SCRATCH}/nemo-rl-worker-cache/mxfp8-layout-vllm-tests-${VENV_KEY}}
MCORE_VENV=${MCORE_VENV:-${LOCAL_SCRATCH}/nemo-rl-worker-cache/mxfp8-layout-mcore-tests-${VENV_KEY}}
WORKER_TEST_COMMAND=""
if [[ "${RUN_WORKER_TESTS}" == 1 ]]; then
  WORKER_TEST_COMMAND=$(cat <<EOF
export UV_PROJECT_ENVIRONMENT=${MCORE_VENV}
uv sync --locked --extra mcore --group test --no-install-project
PYTHONPATH=${REPO} ${MCORE_VENV}/bin/python -m pytest -q \\
  tests/unit/models/policy/test_megatron_worker.py::test_iter_params_batches_expert_prequantization_and_reuses_scratch \\
  --mcore-only \\
  | tee ${RUN_ROOT}/pytest-worker.txt
EOF
)
fi
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
mkdir -p "\${UV_CACHE_DIR}" "\${UV_PYTHON_INSTALL_DIR}"
export UV_PROJECT_ENVIRONMENT=${VLLM_VENV}
uv sync --locked --extra vllm --group test --no-install-project
PYTHONPATH=${REPO} ${VLLM_VENV}/bin/python -m pytest -q \
  tests/unit/models/generation/test_mxfp8_prequant.py::test_batched_expert_prequantization_preserves_wire_entries_and_reuses_scratch \
  tests/unit/models/generation/test_mxfp8_prequant.py::test_batched_expert_prequantization_bounds_batch_and_has_stable_order \
  tests/unit/models/generation/test_mxfp8_prequant.py::test_batched_moe_shuffle_matches_per_expert \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_batched_moe_shuffle_matches_per_expert \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_process_mxfp8_moe_refit_uses_batched_flashinfer_shuffle \
  tests/unit/models/generation/test_vllm_fp8_quantization.py::test_process_weights_after_loading_copies_in_place_on_refit \
  tests/unit/models/generation/test_vllm_refit_loader.py::test_refit_loader_cache_matches_equivalent_tensor_views \
  tests/unit/models/generation/test_vllm_refit_loader.py::test_refit_loader_cache_batches_mxfp8_expert_replay \
  --vllm-only \
  | tee ${RUN_ROOT}/pytest.txt
${WORKER_TEST_COMMAND}
EOF
)
export COMMAND CONTAINER
export MOUNTS=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch

SBATCH_ARGS=(
  --nodes=1
  --ntasks=1
  --gres=gpu:4
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time=01:00:00
  --job-name="${ACCOUNT}-mxfp8-fused-refit-tests"
  --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${NODELIST:-}" ]]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" \
  --wrap='srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" bash -lc "$COMMAND"'
