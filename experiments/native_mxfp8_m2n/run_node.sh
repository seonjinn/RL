#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?}"
: "${SOURCE_SHA:?}"
: "${RESULT_DIR:?}"
LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${SLURM_JOB_USER:-${USER}}/native-mxfp8-m2n}
mkdir -p "${LOCAL_ROOT}/source" "${LOCAL_ROOT}/cache" "${RESULT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR=${LOCAL_ROOT}/cache/uv
export XDG_CACHE_HOME=${LOCAL_ROOT}/cache
export TRITON_CACHE_DIR=${LOCAL_ROOT}/cache/triton
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_ROOT}/cache/inductor
export TMPDIR=${LOCAL_ROOT}/tmp
mkdir -p "${TMPDIR}"
SOURCE_DIR=${LOCAL_ROOT}/source/${SOURCE_SHA}
(
  flock 9
  if [[ ! -f "${SOURCE_DIR}/.complete" ]]; then
    mkdir -p "${SOURCE_DIR}"
    git -C "${REPO}" archive "${SOURCE_SHA}" | tar -xf - -C "${SOURCE_DIR}"
    touch "${SOURCE_DIR}/.complete"
  fi
) 9>"${LOCAL_ROOT}/source/${SOURCE_SHA}.lock"

PYTHON_BIN=${PYTHON_BIN:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}
export PYTHONPATH=${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}
cd "${SOURCE_DIR}"
"${PYTHON_BIN}" experiments/native_mxfp8_m2n/probe.py >"${RESULT_DIR}/runtime-${SLURM_PROCID:-0}.json"
"${PYTHON_BIN}" -c 'import torch; assert torch.__version__; assert torch.cuda.is_available()'
if [[ "${TASK:-transport}" == adapter-unit || "${TASK:-transport}" == adapter-gpu ]]; then
  TEST_DEPS=${LOCAL_ROOT}/pytest-9.1.1
  (
    flock 9
    if [[ ! -f "${TEST_DEPS}/.complete" ]]; then
      uv pip install --python "${PYTHON_BIN}" --target "${TEST_DEPS}" --no-deps \
        pytest==9.1.1 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.19.2
      touch "${TEST_DEPS}/.complete"
    fi
  ) 9>"${TEST_DEPS}.lock"
  export PYTHONPATH=${PYTHONPATH}:${TEST_DEPS}
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  selector=${TEST_FILTER:-binds_dense_and_routed_checkpoint_components}
  if [[ "${TASK}" == adapter-gpu ]]; then
    selector=${TEST_FILTER:-native_cuda_dense_and_routed_refit}
  fi
  exec "${PYTHON_BIN}" -m pytest --confcutdir=tests/unit/models/generation \
    -q -o addopts='' tests/unit/models/generation/test_vllm_refit_adapter.py \
    -k "${selector}"
fi

for backend in ${BACKENDS:-python native native-grouped}; do
  MASTER_PORT=$((MASTER_PORT + 10))
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${SLURM_JOB_NUM_NODES:-1}" \
    --nproc_per_node=4 \
    --node_rank="${SLURM_PROCID:-0}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    experiments/native_mxfp8_m2n/benchmark.py \
    --backend "${backend}" --output "${RESULT_DIR}/${backend}.json"
done
