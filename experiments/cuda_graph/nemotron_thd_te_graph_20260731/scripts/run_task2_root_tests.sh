#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

report_runtime_stage_phase() {
  local phase=$1
  local status=$2
  printf "RUNTIME_STAGE_PHASE=%s STATUS=%s TIMESTAMP_UTC=%s\n" \
    "${phase}" "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

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
  echo "Task 2 result root must be a new absolute path" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd "${script_dir}/../../../.." && pwd -P)
mcore_root=${project_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
mcore_test=tests/unit_tests/data/test_dataset_utils.py
if [[ -L "${mcore_root}" || ! -d "${mcore_root}" || ! -f "${mcore_root}/${mcore_test}" ]]; then
  echo "Megatron-Core dataset helper test source is missing or unsafe" >&2
  exit 2
fi
unit_result_file=tests/unit/unit_results.json
unit_result_dir=tests/unit/unit_results
for generated_path in "${unit_result_file}" "${unit_result_dir}"; do
  if [[ -e "${generated_path}" || -L "${generated_path}" ]]; then
    echo "Task 2 generated result path must not preexist: ${generated_path}" >&2
    exit 2
  fi
done
mkdir -m 0700 -- "${result_root}"

report_runtime_stage_phase root_tests start
pytest_status=0
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  "${runtime_python}" -m pytest -q -p no:cacheprovider \
  "--basetemp=${result_root}/tmp" \
  "--junitxml=${result_root}/task-2-root.xml" \
  tests/unit/experiments/test_validate_te_runtime.py \
  tests/unit/experiments/test_runtime_attestation.py \
  tests/unit/experiments/test_container_harness_hardening.py \
  tests/unit/experiments/test_mcore_standalone_driver.py \
  tests/unit/experiments/test_matrix_submitters.py \
  tests/unit/experiments/test_nemotron_thd_te_graph_launchers.py \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_detach_pending_layerwise_weights_owns_main_and_draft_views \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_collective_refit_uses_one_layerwise_reload_lifecycle \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_weight_reload_targets_only_include_refit_owned_drafter \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_base_ipc_refit_owns_weights_before_ack_allows_buffer_reuse \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_ipc_reload_roots_include_refit_owned_drafter \
  tests/unit/models/generation/test_vllm_modelopt_real_quant_config.py::test_real_quant_reload_keeps_vllm_config_active_during_layerwise_processing \
  || pytest_status=$?
if ((pytest_status != 0)); then
  exit "${pytest_status}"
fi
report_runtime_stage_phase root_tests done
report_runtime_stage_phase root_test_cleanup start
rm -rf -- "${result_root}/tmp"
rm -f -- "${unit_result_file}"
rm -rf -- "${unit_result_dir}"
report_runtime_stage_phase root_test_cleanup done

report_runtime_stage_phase mcore_tests start
mcore_pytest_status=0
(
  cd "${mcore_root}"
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${runtime_python}" -m pytest -q -p no:cacheprovider \
    "--basetemp=${result_root}/mcore-tmp" \
    "--junitxml=${result_root}/task-2-mcore.xml" \
    "${mcore_test}"
) || mcore_pytest_status=$?
if ((mcore_pytest_status != 0)); then
  exit "${mcore_pytest_status}"
fi
report_runtime_stage_phase mcore_tests done
report_runtime_stage_phase mcore_test_cleanup start
rm -rf -- "${result_root}/mcore-tmp"
report_runtime_stage_phase mcore_test_cleanup done
