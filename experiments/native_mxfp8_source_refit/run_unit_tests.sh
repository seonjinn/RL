#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
PYTHON=${PYTHON:-/opt/nemo_rl_venv/bin/python}
RUN_BRIDGE_TESTS=${RUN_BRIDGE_TESTS:-0}
BRIDGE=${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=${BRIDGE}/3rdparty/Megatron-LM

case "${RUN_BRIDGE_TESTS}" in
  0|1) ;;
  *) echo "RUN_BRIDGE_TESTS must be 0 or 1" >&2; exit 2 ;;
esac

cd "${REPO}"
PYTHONPATH="${REPO}:${BRIDGE}/src:${MCORE}" "${PYTHON}" -m pytest -q \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py

if [[ "${RUN_BRIDGE_TESTS}" == 1 ]]; then
  cd "${BRIDGE}"
  PYTHONPATH="${BRIDGE}/src:${MCORE}:${REPO}" "${PYTHON}" -m pytest -q \
    tests/unit_tests/models/test_fp8_param_export.py
else
  echo "Bridge tests skipped; set RUN_BRIDGE_TESTS=1 in a TE-enabled environment."
fi
