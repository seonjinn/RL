#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

fail() {
  echo "$*" >&2
  exit 2
}

run_sbatch_without_reserved_environment() {
  local -a clean_environment=(env)
  local exported_name

  while IFS= read -r exported_name; do
    if [[ "${exported_name}" == SBATCH_* ]]; then
      clean_environment+=(-u "${exported_name}")
    fi
  done < <(compgen -e)
  "${clean_environment[@]}" "$@"
}

: "${ACCOUNT:?Set ACCOUNT to the Slurm account}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to an absolute shared-storage path}"
: "${ATTESTED_NEMORL_SHA:?Set ATTESTED_NEMORL_SHA to the runtime source commit}"
: "${BACKEND:?Set BACKEND to auto, triton, or both}"
: "${CONTAINER:?Set CONTAINER to the immutable Enroot image path}"
: "${CONTAINER_SHA256:?Set CONTAINER_SHA256 to the attested image digest}"
: "${EXPECTED_BRIDGE_SHA:?Set EXPECTED_BRIDGE_SHA to the runtime Bridge commit}"
: "${EXPECTED_MCORE_SHA:?Set EXPECTED_MCORE_SHA to the runtime MCore commit}"
: "${EXPECTED_NEMORL_SHA:?Set EXPECTED_NEMORL_SHA to the exact source commit}"
: "${EXPECTED_TE_SHA:?Set EXPECTED_TE_SHA to the runtime TE commit}"
: "${EXPECTED_TE_VERSION_BASE_SHA:?Set EXPECTED_TE_VERSION_BASE_SHA}"
: "${HF_HOME:?Set HF_HOME to the populated offline Hugging Face cache}"
: "${MODEL_PATH:?Set MODEL_PATH to the exact local model snapshot}"
: "${PARTITION:?Set PARTITION to the Slurm partition}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the exact NeMo-RL checkout}"
: "${RUNTIME_ATTESTATION:?Set RUNTIME_ATTESTATION to the exact passed JSON}"
: "${RUNTIME_PREFLIGHT_JOB_ID:?Set RUNTIME_PREFLIGHT_JOB_ID}"
: "${UV_EXECUTABLE:?Set UV_EXECUTABLE to the attested staged uv}"

TEST_ONLY=${TEST_ONLY:-0}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
runtime_stage_root=$(dirname "$(dirname "${UV_EXECUTABLE}")")
VLLM_PYTHON=${runtime_stage_root}/vllm-environment/bin/python
case "${TEST_ONLY}:${SBATCH_TEST_ONLY}" in
  0:0|0:1|1:0) ;;
  1:1) fail "TEST_ONLY and SBATCH_TEST_ONLY are mutually exclusive" ;;
  *) fail "TEST_ONLY and SBATCH_TEST_ONLY must be 0 or 1" ;;
esac
case "${BACKEND}" in
  auto|triton|both) ;;
  *) fail "BACKEND must be auto, triton, or both" ;;
esac
for path in "${ARTIFACT_DIR}" "${CONTAINER}" "${HF_HOME}" "${MODEL_PATH}" \
  "${PROJECT_ROOT}" "${RUNTIME_ATTESTATION}" "${UV_EXECUTABLE}" \
  "${VLLM_PYTHON}"; do
  [[ "${path}" == /* ]] || fail "All runtime paths must be absolute: ${path}"
done
[[ "${EXPECTED_NEMORL_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
  fail "EXPECTED_NEMORL_SHA must be a full lowercase commit SHA"
for commit in "${ATTESTED_NEMORL_SHA}" "${EXPECTED_BRIDGE_SHA}" \
  "${EXPECTED_MCORE_SHA}" "${EXPECTED_TE_SHA}" \
  "${EXPECTED_TE_VERSION_BASE_SHA}"; do
  [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "Runtime commits must be full SHAs"
done
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "CONTAINER_SHA256 must be a full lowercase SHA256"
[[ "${RUNTIME_PREFLIGHT_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || \
  fail "RUNTIME_PREFLIGHT_JOB_ID must be positive"

launcher=$(realpath "${BASH_SOURCE[0]}") || fail "Cannot resolve launcher path"
diagnostic=${PROJECT_ROOT}/tools/model_diagnostics/6.vllm_routed_experts_completeness.py
runtime_attestation_validator=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py
actual_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD) || fail "Cannot read source SHA"
[[ "${actual_sha}" == "${EXPECTED_NEMORL_SHA}" ]] || \
  fail "NeMo-RL source SHA mismatch: expected ${EXPECTED_NEMORL_SHA}, got ${actual_sha}"
runtime_attestation_command=(
  python3
  "${runtime_attestation_validator}"
  --attestation "${RUNTIME_ATTESTATION}"
  --container "${CONTAINER}"
  --expected-container-sha256 "${CONTAINER_SHA256}"
  --nemo-rl-commit "${ATTESTED_NEMORL_SHA}"
  --bridge-commit "${EXPECTED_BRIDGE_SHA}"
  --mcore-commit "${EXPECTED_MCORE_SHA}"
  --uv-lock "${PROJECT_ROOT}/uv.lock"
  --expected-te-commit "${EXPECTED_TE_SHA}"
  --expected-te-version-base-commit "${EXPECTED_TE_VERSION_BASE_SHA}"
  --expected-device-count 4
  --expected-python-version 3.13.14
  --expected-python-install-dir "$(dirname "${RUNTIME_ATTESTATION}")/uv-python-installations"
  --expected-uv-version 0.11.28
  --expected-uv-executable "${UV_EXECUTABLE}"
  --expected-nvte-with-nccl-ep 0
  --expected-runtime-attestation-job-id "${RUNTIME_PREFLIGHT_JOB_ID}"
  --runtime-feature-set dropless_hybridep_nano16
  --excluded-packages fast-hadamard-transform
  --torch-cuda-arch-list 10.0a
  --nvte-cuda-archs 100a
)
printf 'RUNTIME_ATTESTATION_COMMAND:'
printf ' %q' "${runtime_attestation_command[@]}"
printf '\n'

build_diagnostic_command() {
  local backend=$1
  diagnostic_result_path=${ARTIFACT_DIR}/vllm-routed-experts-${backend}-${SLURM_JOB_ID:-JOB_ID}.json
  diagnostic_command=(
    "${VLLM_PYTHON}"
    "${diagnostic}"
    "${MODEL_PATH}"
    --num-prompts 1
    --max-tokens 2
    --max-model-len 128
    --prompt-repeat 1
    --tensor-parallel-size 4
    --dtype bfloat16
    --gpu-memory-utilization 0.3
    --enforce-eager
    --output "${diagnostic_result_path}"
    --llm-kwarg distributed_executor_backend=mp
    --llm-kwarg load_format=dummy
    --llm-kwarg max_num_batched_tokens=128
    --llm-kwarg max_num_seqs=1
    --llm-kwarg seed=123
  )
  if [[ "${backend}" == "triton" ]]; then
    diagnostic_command+=(--llm-kwarg moe_backend=triton)
  fi
}

if [[ "${BACKEND}" == both ]]; then
  backends=(auto triton)
else
  backends=("${BACKEND}")
fi
for backend in "${backends[@]}"; do
  cache_root=${ARTIFACT_DIR}/cache-${backend}-${SLURM_JOB_ID:-JOB_ID}
  printf 'CACHE_ROOT[%s]: %s\n' "${backend}" "${cache_root}"
  diagnostic_command=()
  build_diagnostic_command "${backend}"
  if [[ "${BACKEND}" == both ]]; then
    printf 'DIAGNOSTIC_COMMAND[%s]:' "${backend}"
  else
    printf 'DIAGNOSTIC_COMMAND:'
  fi
  printf ' %q' "${diagnostic_command[@]}"
  printf '\n'
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ -f "${CONTAINER}" && ! -L "${CONTAINER}" ]] || \
    fail "CONTAINER must be an existing regular non-symlink file"
  [[ -x "${VLLM_PYTHON}" ]] || fail "VLLM_PYTHON must be executable"
  [[ -d "${MODEL_PATH}" && -d "${HF_HOME}" ]] || \
    fail "MODEL_PATH and HF_HOME must be existing directories"
  [[ -f "${diagnostic}" && ! -L "${diagnostic}" ]] || \
    fail "Diagnostic must be an existing regular non-symlink file"
  NVTE_WITH_NCCL_EP=0 "${runtime_attestation_command[@]}" >/dev/null
  mkdir -p "${ARTIFACT_DIR}"
  export FLASHINFER_NO_DOWNLOAD=1
  export HF_DATASETS_OFFLINE=1
  export HF_HOME
  export HF_HUB_CACHE=${HF_HOME}/hub
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_OFFLINE=1
  export HF_MODULES_CACHE=${HF_HOME}/modules
  export TRANSFORMERS_OFFLINE=1
  job_status=0
  artifact_failure=0
  for backend in "${backends[@]}"; do
    cache_root=${ARTIFACT_DIR}/cache-${backend}-${SLURM_JOB_ID}
    mkdir -p \
      "${cache_root}/cuda" \
      "${cache_root}/home" \
      "${cache_root}/torchinductor" \
      "${cache_root}/triton" \
      "${cache_root}/vllm" \
      "${cache_root}/xdg"
    export CUDA_CACHE_PATH=${cache_root}/cuda
    export HOME=${cache_root}/home
    export TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor
    export TRITON_CACHE_DIR=${cache_root}/triton
    export VLLM_CACHE_ROOT=${cache_root}/vllm
    export XDG_CACHE_HOME=${cache_root}/xdg
    diagnostic_command=()
    build_diagnostic_command "${backend}"
    if srun \
        --nodes=1 \
        --ntasks=1 \
        --kill-on-bad-exit=1 \
        --no-container-mount-home \
        --container-image="${CONTAINER}" \
        --container-mounts=/lustre:/lustre \
        --container-env=CUDA_CACHE_PATH,FLASHINFER_NO_DOWNLOAD,HF_DATASETS_OFFLINE,HF_HOME,HF_HUB_CACHE,HF_HUB_DISABLE_IMPLICIT_TOKEN,HF_HUB_DISABLE_TELEMETRY,HF_HUB_OFFLINE,HF_MODULES_CACHE,HOME,TORCHINDUCTOR_CACHE_DIR,TRANSFORMERS_OFFLINE,TRITON_CACHE_DIR,VLLM_CACHE_ROOT,XDG_CACHE_HOME \
        --export=ALL \
        "${diagnostic_command[@]}"; then
      backend_status=0
    else
      backend_status=$?
    fi
    printf 'BACKEND_EXIT[%s]: %s\n' "${backend}" "${backend_status}"
    if [[ ! -s "${diagnostic_result_path}" || -L "${diagnostic_result_path}" ]] || \
       ! python3 -c 'import json, sys; json.load(open(sys.argv[1]))' \
         "${diagnostic_result_path}"; then
      printf 'RESULT_STATUS[%s]: missing_or_invalid\n' "${backend}" >&2
      artifact_failure=1
    else
      printf 'RESULT_STATUS[%s]: published\n' "${backend}"
    fi
    if [[ "${BACKEND}" != both || "${backend}" == triton ]]; then
      job_status=${backend_status}
    fi
  done
  if [[ "${artifact_failure}" != 0 ]]; then
    exit 1
  fi
  exit "${job_status}"
fi

if [[ "${TEST_ONLY}" != 1 ]]; then
  [[ -f "${CONTAINER}" && ! -L "${CONTAINER}" ]] || \
    fail "CONTAINER must be an existing regular non-symlink file"
  [[ -x "${VLLM_PYTHON}" ]] || fail "VLLM_PYTHON must be executable"
  [[ -d "${MODEL_PATH}" && -d "${HF_HOME}" ]] || \
    fail "MODEL_PATH and HF_HOME must be existing directories"
  [[ -f "${diagnostic}" && ! -L "${diagnostic}" ]] || \
    fail "Diagnostic must be an existing regular non-symlink file"
  [[ -f "${runtime_attestation_validator}" && ! -L "${runtime_attestation_validator}" ]] || \
    fail "Runtime attestation validator is missing or unsafe"
  git -C "${PROJECT_ROOT}" diff --quiet --ignore-submodules=none || \
    fail "NeMo-RL checkout has unstaged changes"
  git -C "${PROJECT_ROOT}" diff --cached --quiet --ignore-submodules=none || \
    fail "NeMo-RL checkout has staged changes"
  [[ -z "$(git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard)" ]] || \
    fail "NeMo-RL checkout has untracked files"
  NVTE_WITH_NCCL_EP=0 "${runtime_attestation_command[@]}" >/dev/null
fi

sbatch_command=(
  sbatch
  --parsable
  "--partition=${PARTITION}"
  "--account=${ACCOUNT}"
  --nodes=1
  --time=00:30:00
  "--job-name=${ACCOUNT}-vllm-routes-${BACKEND}"
  "--output=${ARTIFACT_DIR}/vllm-routed-experts-${BACKEND}-%j.log"
  "--export=ACCOUNT=${ACCOUNT},ARTIFACT_DIR=${ARTIFACT_DIR},ATTESTED_NEMORL_SHA=${ATTESTED_NEMORL_SHA},BACKEND=${BACKEND},CONTAINER=${CONTAINER},CONTAINER_SHA256=${CONTAINER_SHA256},EXPECTED_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA},EXPECTED_MCORE_SHA=${EXPECTED_MCORE_SHA},EXPECTED_NEMORL_SHA=${EXPECTED_NEMORL_SHA},EXPECTED_TE_SHA=${EXPECTED_TE_SHA},EXPECTED_TE_VERSION_BASE_SHA=${EXPECTED_TE_VERSION_BASE_SHA},HF_HOME=${HF_HOME},MODEL_PATH=${MODEL_PATH},PARTITION=${PARTITION},PROJECT_ROOT=${PROJECT_ROOT},RUNTIME_ATTESTATION=${RUNTIME_ATTESTATION},RUNTIME_PREFLIGHT_JOB_ID=${RUNTIME_PREFLIGHT_JOB_ID},UV_EXECUTABLE=${UV_EXECUTABLE},PATH=${PATH:-/usr/bin:/bin}"
  "${launcher}"
)
if [[ "${SBATCH_TEST_ONLY}" == 1 ]]; then
  sbatch_command=(sbatch --test-only "${sbatch_command[@]:2}")
fi
printf 'SBATCH:'
printf ' %q' "${sbatch_command[@]}"
printf '\n'
if [[ "${TEST_ONLY}" == 1 ]]; then
  echo "TEST_ONLY: no submission performed"
  exit 0
fi
if [[ "${SBATCH_TEST_ONLY}" != 1 ]]; then
  mkdir -p "${ARTIFACT_DIR}"
fi
run_sbatch_without_reserved_environment "${sbatch_command[@]}"
