#!/bin/bash

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the validation checkout}"
: "${MEGATRON_LM_ROOT:?Set MEGATRON_LM_ROOT to the Megatron-LM checkout}"
: "${UV_PROJECT_ENVIRONMENT:?Set UV_PROJECT_ENVIRONMENT to the NeMo-RL environment}"

export PYTEST_ADDOPTS=
readonly pytest_plugin_args=(-p no:pytest-shard)
readonly python="${UV_PROJECT_ENVIRONMENT}/bin/python"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${python}" -m pytest "${pytest_plugin_args[@]}" -q \
  tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig
"${python}" -m pytest "${pytest_plugin_args[@]}" -q \
  tests/unit/models/megatron/test_megatron_data.py::TestPrepareVlmBatchForMegatron
"${python}" -m pytest "${pytest_plugin_args[@]}" -q \
  tests/unit/tools/test_hybridep_default_8g_recipes.py

cd "${MEGATRON_LM_ROOT}"
PYTHONPATH="${MEGATRON_LM_ROOT}:${PYTHONPATH:-}" \
  "${python}" -m pytest "${pytest_plugin_args[@]}" -q \
  "tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_expert_bias_token_counts_with_padding_mask"
