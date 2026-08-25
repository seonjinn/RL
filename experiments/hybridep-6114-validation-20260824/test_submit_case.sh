#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_case.sh"
FOCUSED_SCRIPT="${SCRIPT_DIR}/focused_tests.sbatch"

assert_contains() {
  local output=$1
  local expected=$2

  if [[ ${output} != *"${expected}"* ]]; then
    echo "Expected dry-run output to contain: ${expected}" >&2
    exit 1
  fi
}

dry_run_case() {
  local case_name=$1

  CONTAINER=/tmp/nemo-rl.sqsh \
    HF_HOME=/tmp/hf-home \
    RESULTS_ROOT=/tmp/hybridep-results \
    VALIDATION_DRY_RUN=1 \
    "${SUBMIT_SCRIPT}" "${case_name}"
}

qwen30_pp1_cp1=$(dry_run_case qwen30_pp1_cp1)
assert_contains "${qwen30_pp1_cp1}" "--nodes=4"
assert_contains "${qwen30_pp1_cp1}" "--gpus-per-node=8"
assert_contains "${qwen30_pp1_cp1}" "pipeline_model_parallel_size=1"
assert_contains "${qwen30_pp1_cp1}" "context_parallel_size=1"
assert_contains "${qwen30_pp1_cp1}" "moe_hybridep_prepad_packed_inputs=true"

qwen30_pp2_cp2=$(dry_run_case qwen30_pp2_cp2)
assert_contains "${qwen30_pp2_cp2}" "--nodes=4"
assert_contains "${qwen30_pp2_cp2}" "pipeline_model_parallel_size=2"
assert_contains "${qwen30_pp2_cp2}" "context_parallel_size=2"
assert_contains "${qwen30_pp2_cp2}" "moe_hybridep_prepad_packed_inputs=false"

qwen235_pp8_cp2=$(dry_run_case qwen235_pp8_cp2)
assert_contains "${qwen235_pp8_cp2}" "--nodes=16"
assert_contains "${qwen235_pp8_cp2}" "pipeline_model_parallel_size=8"
assert_contains "${qwen235_pp8_cp2}" "context_parallel_size=2"

super_pp1_cp1=$(dry_run_case super_pp1_cp1)
assert_contains "${super_pp1_cp1}" "--nodes=32"
assert_contains "${super_pp1_cp1}" "moe_router_enable_expert_bias=true"

for output in \
  "${qwen30_pp1_cp1}" \
  "${qwen30_pp2_cp2}" \
  "${qwen235_pp8_cp2}" \
  "${super_pp1_cp1}"; do
  assert_contains "${output}" "grpo.max_num_steps=3"
  assert_contains "${output}" "--shared"
  assert_contains "${output}" "NRL_FORCE_REBUILD_VENVS=true"
  assert_contains "${output}" "HYBRID_EP_MULTINODE=1"
  assert_contains "${output}" "TMS_CUDA_MAJOR"
  if [[ ${output} == *"NEMO_RL_VENV_DIR="* || ${output} == *"UV_PROJECT_ENVIRONMENT="* ]]; then
    echo "Model jobs must use the nightly container's node-local environment paths" >&2
    exit 1
  fi
  if [[ ${output} == *"UV_CACHE_DIR_OVERRIDE="* ]]; then
    echo "Model jobs must reuse the container-baked uv cache" >&2
    exit 1
  fi
done

echo "submit_case dry-run contract passed"

focused_content=$(<"${FOCUSED_SCRIPT}")
assert_contains "${focused_content}" "#SBATCH --gpus-per-node=1"
assert_contains "${focused_content}" "export HYBRID_EP_MULTINODE=1"
assert_contains "${focused_content}" "export TMS_CUDA_MAJOR="
if [[ ${focused_content} == *"export UV_CACHE_DIR="* ]]; then
  echo "Focused tests must reuse the container-baked uv cache" >&2
  exit 1
fi
focused_nemo_pytest_count=$(grep -c "uv run --locked pytest -q" "${FOCUSED_SCRIPT}")
if [[ ${focused_nemo_pytest_count} -ne 3 ]]; then
  echo "Focused test must run three independently filtered NeMo-RL pytest commands" >&2
  exit 1
fi
if [[ ${focused_content} == *"--exclusive"* ]]; then
  echo "Focused test must not request exclusive-node access" >&2
  exit 1
fi

echo "focused test allocation contract passed"
