#!/usr/bin/env bash

set -euo pipefail

REPO=${REPO:-/home/${USER}/RL-qwen235-mxfp8-ab-20260909}
PYTHON=${PYTHON:-/opt/nemo_rl_venv/bin/python}
BRIDGE=${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=${BRIDGE}/3rdparty/Megatron-LM

cd "${REPO}"
PYTHONPATH="${REPO}:${BRIDGE}/src:${MCORE}" "${PYTHON}" -m pytest -q \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py

cd "${BRIDGE}"
PYTHONPATH="${BRIDGE}/src:${MCORE}:${REPO}" "${PYTHON}" -m pytest -q \
  tests/unit_tests/models/test_fp8_param_export.py
