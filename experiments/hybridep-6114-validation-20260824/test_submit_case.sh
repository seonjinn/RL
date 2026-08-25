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

assert_not_contains() {
  local output=$1
  local unexpected=$2

  if [[ ${output} == *"${unexpected}"* ]]; then
    echo "Expected dry-run output not to contain: ${unexpected}" >&2
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
assert_contains "${qwen30_pp2_cp2}" "policy.make_sequence_length_divisible_by=4"
assert_contains "${qwen30_pp2_cp2}" "moe_hybridep_prepad_packed_inputs=false"

qwen30_batch_short=$(TIME_LIMIT=02:00:00 dry_run_case qwen30_pp2_cp2)
assert_contains "${qwen30_batch_short}" "--time=02:00:00"

qwen235_pp8_cp2=$(dry_run_case qwen235_pp8_cp2)
assert_contains "${qwen235_pp8_cp2}" "--nodes=16"
assert_contains "${qwen235_pp8_cp2}" "pipeline_model_parallel_size=8"
assert_contains "${qwen235_pp8_cp2}" "context_parallel_size=2"
assert_contains "${qwen235_pp8_cp2}" "policy.make_sequence_length_divisible_by=4"

super_pp1_cp1=$(dry_run_case super_pp1_cp1)
assert_contains "${super_pp1_cp1}" "--nodes=32"
assert_contains "${super_pp1_cp1}" "moe_router_enable_expert_bias=true"

qwen30_async_baseline=$(dry_run_case qwen30_async_baseline)
assert_contains "${qwen30_async_baseline}" "grpo-qwen3-30ba3b-4n8g-async-1off.yaml"
assert_contains "${qwen30_async_baseline}" "--nodes=4"
assert_contains "${qwen30_async_baseline}" "moe_token_dispatcher_type=alltoall"
assert_contains "${qwen30_async_baseline}" "moe_flex_dispatcher_backend=deepep"
assert_contains "${qwen30_async_baseline}" "moe_hybridep_prepad_packed_inputs=false"

qwen30_async_hybridep=$(dry_run_case qwen30_async_hybridep)
assert_contains "${qwen30_async_hybridep}" "grpo-qwen3-30ba3b-4n8g-async-1off.yaml"
assert_contains "${qwen30_async_hybridep}" "--nodes=4"
assert_not_contains "${qwen30_async_hybridep}" "moe_token_dispatcher_type=alltoall"

qwen235_async_baseline=$(dry_run_case qwen235_async_baseline)
assert_contains "${qwen235_async_baseline}" "grpo-qwen3-235b-32n8g-async-1off.yaml"
assert_contains "${qwen235_async_baseline}" "--nodes=32"
assert_contains "${qwen235_async_baseline}" "moe_token_dispatcher_type=alltoall"
assert_contains "${qwen235_async_baseline}" "moe_flex_dispatcher_backend=deepep"
assert_contains "${qwen235_async_baseline}" "moe_hybridep_prepad_packed_inputs=false"

qwen235_async_hybridep=$(dry_run_case qwen235_async_hybridep)
assert_contains "${qwen235_async_hybridep}" "grpo-qwen3-235b-32n8g-async-1off.yaml"
assert_contains "${qwen235_async_hybridep}" "--nodes=32"
assert_not_contains "${qwen235_async_hybridep}" "moe_token_dispatcher_type=alltoall"

super_async_baseline=$(dry_run_case super_async_baseline)
assert_contains "${super_async_baseline}" "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml"
assert_contains "${super_async_baseline}" "--nodes=32"
assert_contains "${super_async_baseline}" "moe_token_dispatcher_type=alltoall"
assert_contains "${super_async_baseline}" "moe_flex_dispatcher_backend=deepep"
assert_contains "${super_async_baseline}" "moe_hybridep_prepad_packed_inputs=false"

super_async_hybridep=$(dry_run_case super_async_hybridep)
assert_contains "${super_async_hybridep}" "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml"
assert_contains "${super_async_hybridep}" "--nodes=32"
assert_not_contains "${super_async_hybridep}" "moe_token_dispatcher_type=alltoall"

for output in \
  "${qwen30_pp1_cp1}" \
  "${qwen30_pp2_cp2}" \
  "${qwen235_pp8_cp2}" \
  "${super_pp1_cp1}" \
  "${qwen30_async_baseline}" \
  "${qwen30_async_hybridep}" \
  "${qwen235_async_baseline}" \
  "${qwen235_async_hybridep}" \
  "${super_async_baseline}" \
  "${super_async_hybridep}"; do
  assert_contains "${output}" "grpo.max_num_steps=3"
  assert_contains "${output}" "NRL_FORCE_REBUILD_VENVS=true"
  assert_contains "${output}" "HYBRID_EP_MULTINODE=1"
  assert_contains "${output}" "TMS_CUDA_MAJOR"
  if [[ ${output} == *"--exclusive"* || ${output} == *"--shared"* ]]; then
    echo "Model jobs must rely on the cluster's default shared allocation" >&2
    exit 1
  fi
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
if [[ ${focused_content} == *"--exclusive"* ]]; then
  echo "Focused test must not request exclusive-node access" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/test_run_focused_tests.sh"

echo "focused test allocation contract passed"
