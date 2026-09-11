#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?}"
: "${SOURCE_SHA:?}"
: "${RESULT_DIR:?}"
LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/native-mxfp8-m2n}
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

if [[ -z "${PYTHON_BIN:-}" ]]; then
  shopt -s nullglob
  for candidate in /opt/ray_venvs/*/bin/python; do
    if [[ "${candidate,,}" == *vllm* ]]; then
      PYTHON_BIN=${candidate}
      break
    fi
  done
fi
: "${PYTHON_BIN:?No vLLM actor interpreter found; set PYTHON_BIN explicitly}"
export PYTHONPATH=${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}
cd "${SOURCE_DIR}"
"${PYTHON_BIN}" experiments/native_mxfp8_m2n/probe.py >"${RESULT_DIR}/runtime-${SLURM_PROCID:-0}.json"

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
