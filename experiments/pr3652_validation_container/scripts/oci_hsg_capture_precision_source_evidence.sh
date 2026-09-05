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

set -Eeuo pipefail

if (( $# != 9 )); then
  echo "Usage: $0 SEMANTIC_WORKTREE EXPECTED_REPO_SHA COMPRESSED_TENSORS_SOURCE_ROOT MODELOPT_LIGHTNING_SOURCE_ROOT TRANSFORMER_ENGINE_SOURCE_ROOT STAGED_METADATA_ROOT SCRATCH_DIRECTORY CAPTURED_BASE EXPECTED_TOOLING_SHA" >&2
  exit 2
fi

readonly SEMANTIC_WORKTREE=$1
readonly EXPECTED_REPO_SHA=$2
readonly COMPRESSED_TENSORS_SOURCE_ROOT=$3
readonly MODELOPT_LIGHTNING_SOURCE_ROOT=$4
readonly TRANSFORMER_ENGINE_SOURCE_ROOT=$5
readonly STAGED_METADATA_ROOT=$6
readonly SCRATCH_DIRECTORY=$7
readonly CAPTURED_BASE=$8
readonly EXPECTED_TOOLING_SHA=$9
readonly MAIN_PYTHON=/opt/nemo_rl_venv/bin/python
readonly CAPTURE_TOOL=${SEMANTIC_WORKTREE}/tools/capture_precision_policy_source_evidence.py
readonly OUTPUT_DIRECTORY=${SCRATCH_DIRECTORY}/captured
readonly EXPECTED_RAW_MANIFEST_SHA256=d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly EXPECTED_CONTAINER_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
PUBLISH_STAGE_DIRECTORY=
RUN_RECEIPT_STAGE=

require_clean_git_tree() {
  local path=$1
  local expected_sha=$2
  local worktree_status

  [[ "${path}" = /* ]]
  test ! -L "${path}"
  test "$(git -C "${path}" rev-parse --is-inside-work-tree)" = true
  test "$(git -C "${path}" rev-parse --show-toplevel)" = "${path}"
  test "$(git -C "${path}" rev-parse HEAD)" = "${expected_sha}"
  worktree_status=$(git -C "${path}" status --porcelain)
  test -z "${worktree_status}"
}

require_pinned_source_tree() {
  local path=$1
  local expected_sha=$2
  local expected_origin=$3
  local origin_count
  local origin_urls

  require_clean_git_tree "${path}" "${expected_sha}"
  origin_urls=$(git -C "${path}" config --get-all remote.origin.url)
  origin_count=$(printf '%s\n' "${origin_urls}" | sed '/^$/d' | wc -l | tr -d ' ')
  test "${origin_count}" = 1
  test "${origin_urls}" = "${expected_origin}"
}

validate_raw_receipt() {
  local file_count
  local manifest_line_count
  local manifest_sha256
  local symlink_path

  test -d "${STAGED_METADATA_ROOT}"
  test ! -L "${STAGED_METADATA_ROOT}"
  test -f "${STAGED_METADATA_ROOT}/SHA256SUMS"
  test ! -L "${STAGED_METADATA_ROOT}/SHA256SUMS"
  symlink_path=$(find "${STAGED_METADATA_ROOT}" -type l -print -quit)
  test -z "${symlink_path}"
  file_count=$(find "${STAGED_METADATA_ROOT}" -type f ! -path "${STAGED_METADATA_ROOT}/SHA256SUMS" -print | wc -l | tr -d ' ')
  test "${file_count}" = 19
  manifest_line_count=$(wc -l <"${STAGED_METADATA_ROOT}/SHA256SUMS" | tr -d ' ')
  test "${manifest_line_count}" = 19
  manifest_sha256=$(sha256sum "${STAGED_METADATA_ROOT}/SHA256SUMS" | awk '{print $1}')
  test "${manifest_sha256}" = "${EXPECTED_RAW_MANIFEST_SHA256}"
  (
    cd "${STAGED_METADATA_ROOT}"
    awk 'NF != 2 || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }' SHA256SUMS
    diff -u \
      <(awk '{print $2}' SHA256SUMS | LC_ALL=C sort) \
      <(find . -type f ! -name SHA256SUMS -print | sed 's#^\./##' | LC_ALL=C sort)
    sha256sum --check --strict SHA256SUMS
  )
}

validate_output_directory() {
  local actual_files
  local expected_files

  test -d "${OUTPUT_DIRECTORY}"
  test -z "$(find "${OUTPUT_DIRECTORY}" -type l -print -quit)"
  actual_files=$(find "${OUTPUT_DIRECTORY}" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort)
  expected_files=$'producer_implementations.json\nsource_format_evidence.json'
  test "${actual_files}" = "${expected_files}"
  test -s "${OUTPUT_DIRECTORY}/producer_implementations.json"
  test -s "${OUTPUT_DIRECTORY}/source_format_evidence.json"
}

validate_published_directory() {
  local directory=$1
  local actual_files
  local expected_files

  test -d "${directory}"
  test ! -L "${directory}"
  test -z "$(find "${directory}" -type l -print -quit)"
  actual_files=$(find "${directory}" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort)
  expected_files=$'MANIFEST.sha256\nproducer_implementations.json\nsource_format_evidence.json'
  test "${actual_files}" = "${expected_files}"
  test -s "${directory}/producer_implementations.json"
  test -s "${directory}/source_format_evidence.json"
  cmp -s "${OUTPUT_DIRECTORY}/MANIFEST.sha256" "${directory}/MANIFEST.sha256"
  (
    cd "${directory}"
    sha256sum --check --strict MANIFEST.sha256
  )
}

cleanup() {
  local exit_status=$?

  trap - EXIT
  if [[ -n ${PUBLISH_STAGE_DIRECTORY} && -e ${PUBLISH_STAGE_DIRECTORY} ]]; then
    chmod u+w "${PUBLISH_STAGE_DIRECTORY}"
    rm -rf -- "${PUBLISH_STAGE_DIRECTORY}"
  fi
  if [[ -n ${RUN_RECEIPT_STAGE} && -e ${RUN_RECEIPT_STAGE} ]]; then
    rm -f -- "${RUN_RECEIPT_STAGE}"
  fi
  exit "${exit_status}"
}

case ${SCRATCH_DIRECTORY} in
  /raid/scratch/nemo-rl-semantic-precision-evidence/oci-capture-[0-9]*) ;;
  *)
    echo 'SCRATCH_DIRECTORY must be the job-local OCI capture scratch directory' >&2
    exit 2
    ;;
esac
case ${CAPTURED_BASE} in
  /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/captured) ;;
  *)
    echo 'CAPTURED_BASE must be the pinned OCI evidence publication root' >&2
    exit 2
    ;;
esac
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi
test "${SCRATCH_DIRECTORY}" = "/raid/scratch/nemo-rl-semantic-precision-evidence/oci-capture-${SLURM_JOB_ID}"
test "${TMPDIR:-}" = "${SCRATCH_DIRECTORY}/tmp"
test "${PYTHONPYCACHEPREFIX:-}" = "${SCRATCH_DIRECTORY}/pycache"
test "${XDG_CACHE_HOME:-}" = "${SCRATCH_DIRECTORY}/xdg-cache"
test "${UV_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/uv-cache"
test "${TORCHINDUCTOR_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/torchinductor-cache"
test "${TRITON_CACHE_DIR:-}" = "${SCRATCH_DIRECTORY}/triton-cache"
test -x "${MAIN_PYTHON}"
test -f "${CAPTURE_TOOL}"
test -d "${CAPTURED_BASE}"
test -d "${CAPTURED_BASE}/runs"
test ! -e "${OUTPUT_DIRECTORY}"
require_clean_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
require_pinned_source_tree "${COMPRESSED_TENSORS_SOURCE_ROOT}" f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 https://github.com/vllm-project/compressed-tensors.git
require_pinned_source_tree "${MODELOPT_LIGHTNING_SOURCE_ROOT}" c897fbeaaff66d53d61033f107885b7c5432f235 https://github.com/NVIDIA/Model-Optimizer.git
require_pinned_source_tree "${TRANSFORMER_ENGINE_SOURCE_ROOT}" 42b840051647eef89761a16dfdff87e82bb253ab https://github.com/NVIDIA/TransformerEngine.git
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)" = b11414c71b15e54d333eb49346ed199f20fa9021
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Automodel-workspace/Automodel" rev-parse HEAD)" = 1814c6c93a66b9d59d254960ef6a99a64249b671
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)" = 7c9c3a027c503ae9ae1e8ad7b14397abb8269378
validate_raw_receipt
trap cleanup EXIT

mkdir -p "${OUTPUT_DIRECTORY}"
PYTHONPATH="${SEMANTIC_WORKTREE}" "${MAIN_PYTHON}" "${CAPTURE_TOOL}" \
  --repository-root "${SEMANTIC_WORKTREE}" \
  --compressed-tensors-source-root "${COMPRESSED_TENSORS_SOURCE_ROOT}" \
  --modelopt-lightning-source-root "${MODELOPT_LIGHTNING_SOURCE_ROOT}" \
  --staged-metadata-root "${STAGED_METADATA_ROOT}" \
  --transformer-engine-source-root "${TRANSFORMER_ENGINE_SOURCE_ROOT}" \
  --output-directory "${OUTPUT_DIRECTORY}" \
  --inspect-runtime
PYTHONPATH="${SEMANTIC_WORKTREE}" "${MAIN_PYTHON}" "${CAPTURE_TOOL}" \
  --repository-root "${SEMANTIC_WORKTREE}" \
  --compressed-tensors-source-root "${COMPRESSED_TENSORS_SOURCE_ROOT}" \
  --modelopt-lightning-source-root "${MODELOPT_LIGHTNING_SOURCE_ROOT}" \
  --staged-metadata-root "${STAGED_METADATA_ROOT}" \
  --transformer-engine-source-root "${TRANSFORMER_ENGINE_SOURCE_ROOT}" \
  --output-directory "${OUTPUT_DIRECTORY}" \
  --check \
  --inspect-runtime

require_clean_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
require_pinned_source_tree "${COMPRESSED_TENSORS_SOURCE_ROOT}" f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0 https://github.com/vllm-project/compressed-tensors.git
require_pinned_source_tree "${MODELOPT_LIGHTNING_SOURCE_ROOT}" c897fbeaaff66d53d61033f107885b7c5432f235 https://github.com/NVIDIA/Model-Optimizer.git
require_pinned_source_tree "${TRANSFORMER_ENGINE_SOURCE_ROOT}" 42b840051647eef89761a16dfdff87e82bb253ab https://github.com/NVIDIA/TransformerEngine.git
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)" = b11414c71b15e54d333eb49346ed199f20fa9021
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Automodel-workspace/Automodel" rev-parse HEAD)" = 1814c6c93a66b9d59d254960ef6a99a64249b671
test "$(git -C "${SEMANTIC_WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)" = 7c9c3a027c503ae9ae1e8ad7b14397abb8269378
validate_raw_receipt
validate_output_directory
(
  cd "${OUTPUT_DIRECTORY}"
  sha256sum producer_implementations.json source_format_evidence.json | LC_ALL=C sort -k2 >MANIFEST.sha256
)
test "$(wc -l <"${OUTPUT_DIRECTORY}/MANIFEST.sha256" | tr -d ' ')" = 2
OUTPUT_ID=$(sha256sum "${OUTPUT_DIRECTORY}/MANIFEST.sha256" | awk '{print $1}')
readonly OUTPUT_ID
[[ "${OUTPUT_ID}" =~ ^[0-9a-f]{64}$ ]]
readonly PUBLISH_DIRECTORY=${CAPTURED_BASE}/sha256-${OUTPUT_ID}
PUBLISH_STAGE_DIRECTORY=${CAPTURED_BASE}/.sha256-${OUTPUT_ID}.job-${SLURM_JOB_ID}.stage
readonly RUN_RECEIPT=${CAPTURED_BASE}/runs/${SLURM_JOB_ID}.json
RUN_RECEIPT_STAGE=${CAPTURED_BASE}/runs/.${SLURM_JOB_ID}.json.stage

test ! -e "${PUBLISH_STAGE_DIRECTORY}"
test ! -e "${RUN_RECEIPT_STAGE}"
mkdir "${PUBLISH_STAGE_DIRECTORY}"
cp -- \
  "${OUTPUT_DIRECTORY}/source_format_evidence.json" \
  "${OUTPUT_DIRECTORY}/producer_implementations.json" \
  "${OUTPUT_DIRECTORY}/MANIFEST.sha256" \
  "${PUBLISH_STAGE_DIRECTORY}/"
validate_published_directory "${PUBLISH_STAGE_DIRECTORY}"
chmod 444 "${PUBLISH_STAGE_DIRECTORY}"/*
chmod 555 "${PUBLISH_STAGE_DIRECTORY}"

printf '%s\n' \
  '{' \
  "  \"schema_version\": \"precision-policy-source-capture-run.v1\"," \
  "  \"slurm_job_id\": \"${SLURM_JOB_ID}\"," \
  "  \"semantic_repository_sha\": \"${EXPECTED_REPO_SHA}\"," \
  "  \"tooling_repository_sha\": \"${EXPECTED_TOOLING_SHA}\"," \
  "  \"container_sha256\": \"${EXPECTED_CONTAINER_SHA256}\"," \
  "  \"raw_manifest_sha256\": \"${EXPECTED_RAW_MANIFEST_SHA256}\"," \
  "  \"output_id\": \"${OUTPUT_ID}\"," \
  "  \"artifact_directory\": \"${PUBLISH_DIRECTORY}\"" \
  '}' >"${RUN_RECEIPT_STAGE}"
chmod 444 "${RUN_RECEIPT_STAGE}"

mv -Tn -- "${PUBLISH_STAGE_DIRECTORY}" "${PUBLISH_DIRECTORY}"
if [[ -e ${PUBLISH_STAGE_DIRECTORY} ]]; then
  validate_published_directory "${PUBLISH_DIRECTORY}"
  chmod u+w "${PUBLISH_STAGE_DIRECTORY}"
  rm -rf -- "${PUBLISH_STAGE_DIRECTORY}"
fi
PUBLISH_STAGE_DIRECTORY=
validate_published_directory "${PUBLISH_DIRECTORY}"

mv -Tn -- "${RUN_RECEIPT_STAGE}" "${RUN_RECEIPT}"
if [[ -e ${RUN_RECEIPT_STAGE} ]]; then
  cmp -s "${RUN_RECEIPT_STAGE}" "${RUN_RECEIPT}"
  rm -f -- "${RUN_RECEIPT_STAGE}"
fi
RUN_RECEIPT_STAGE=
trap - EXIT
