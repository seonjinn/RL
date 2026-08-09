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
  echo "Task 2 result root must be a new absolute path" >&2
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
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/data/test_dataset_utils.py || pytest_status=$?
if ((pytest_status != 0)); then
  exit "${pytest_status}"
fi
rm -rf -- "${result_root}/tmp"
rm -f -- "${unit_result_file}"
rm -rf -- "${unit_result_dir}"
