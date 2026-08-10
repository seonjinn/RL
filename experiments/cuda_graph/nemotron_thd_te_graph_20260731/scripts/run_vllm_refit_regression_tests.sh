#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <staged-python> <absolute-result-root>" >&2
  exit 2
fi

runtime_python=$1
result_root=$2
if [[ "${runtime_python}" != /* || ! -x "${runtime_python}" ]]; then
  echo "Staged Python must be an absolute executable" >&2
  exit 2
fi
if [[ "${result_root}" != /* || -e "${result_root}" || -L "${result_root}" ]]; then
  echo "Result root must be a new absolute path" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd "${script_dir}/../../../.." && pwd -P)
mkdir -m 0700 -- "${result_root}"
cd "${project_root}"

env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  "${runtime_python}" -m pytest -q -p no:cacheprovider --maxfail=0 \
  "--basetemp=${result_root}/tmp" \
  "--junitxml=${result_root}/vllm-refit-regressions.xml" \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_detach_pending_layerwise_weights_owns_main_and_draft_views \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_collective_refit_uses_one_layerwise_reload_lifecycle \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_weight_reload_targets_only_include_refit_owned_drafter \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_ipc_refit_owns_weights_before_ack_allows_buffer_reuse \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_ipc_reload_roots_include_refit_owned_drafter \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_reload_keeps_vllm_config_active_during_layerwise_processing
