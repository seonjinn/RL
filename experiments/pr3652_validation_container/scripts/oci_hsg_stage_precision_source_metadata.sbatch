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
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=nemotron_n3_post-stage_precision_metadata
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw/logs/stage-slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw/logs/stage-slurm-%j.err

set -Eeuo pipefail

export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${SCRIPT_ROOT:?Set SCRIPT_ROOT to the absolute clean staging worktree root}"
: "${EXPECTED_TOOLING_SHA:?Set EXPECTED_TOOLING_SHA to the staging worktree HEAD}"
: "${SEMANTIC_WORKTREE:?Set SEMANTIC_WORKTREE to the absolute clean semantic worktree root}"
: "${EXPECTED_REPO_SHA:?Set EXPECTED_REPO_SHA to the semantic worktree HEAD}"

readonly CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_CONTAINER_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly CONTAINER_SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly CONTAINER_PYTHON=/opt/nemo_rl_venv/bin/python
readonly OUTPUT_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/semantic-precision-refit/source-evidence/raw
readonly EXPECTED_MANIFEST_SHA256=d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b
readonly RAW_MANIFEST_FILENAME=SHA256SUMS
readonly EXPECTED_FINAL=${OUTPUT_ROOT}/sha256-${EXPECTED_MANIFEST_SHA256}
readonly SRUN=/cm/local/apps/slurm/current/bin/srun
readonly -a SNAPSHOT_BLOBS=(
  tools/stage_precision_policy_source_metadata.py
  tools/capture_precision_policy_source_evidence.py
  nemo_rl/precision_policy/semantic.py
  nemo_rl/precision_policy/source_formats.py
)

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly SCRATCH_DIRECTORY=/raid/scratch/nemo-rl-semantic-precision-metadata/oci-stage-${SLURM_JOB_ID}
readonly SNAPSHOT_ROOT=${SCRATCH_DIRECTORY}/semantic-repo

require_clean_git_tree() {
  local path=$1
  local expected_sha=$2
  local status

  [[ "${path}" = /* ]]
  test ! -L "${path}"
  test "$(git -C "${path}" rev-parse --is-inside-work-tree)" = true
  test "$(git -C "${path}" rev-parse --show-toplevel)" = "${path}"
  test "$(git -C "${path}" rev-parse HEAD)" = "${expected_sha}"
  status=$(git -C "${path}" status --porcelain)
  test -z "${status}"
}

validate_image_receipt() {
  local metadata=${CONTAINER}.metadata.txt
  local completion_marker=${CONTAINER}.complete

  test -f "${CONTAINER}"
  test ! -L "${CONTAINER}"
  test -f "${metadata}"
  test ! -L "${metadata}"
  test "$(wc -l <"${metadata}")" -eq 7
  grep -Fx -- "source=${CONTAINER_SOURCE}" "${metadata}" >/dev/null
  grep -Fx -- 'source_cluster=ptyche' "${metadata}" >/dev/null
  grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata}" >/dev/null
  grep -Fx -- 'source_smoke_job=2721177' "${metadata}" >/dev/null
  grep -Ex -- 'download_job=[0-9]+' "${metadata}" >/dev/null
  grep -Ex -- 'downloaded_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${metadata}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_CONTAINER_SHA256}" "${metadata}" >/dev/null

  test -f "${completion_marker}"
  test ! -L "${completion_marker}"
  test "$(wc -l <"${completion_marker}")" -eq 4
  grep -Fx -- "destination=${CONTAINER}" "${completion_marker}" >/dev/null
  grep -Fx -- "source=${CONTAINER_SOURCE}" "${completion_marker}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_CONTAINER_SHA256}" "${completion_marker}" >/dev/null
  grep -Ex -- 'completed_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${completion_marker}" >/dev/null

  printf 'expected_image_sha256=%s\n' "${EXPECTED_CONTAINER_SHA256}"
  printf 'image_metadata_sha256=%s\n' "$(sha256sum "${metadata}" | awk '{print $1}')"
  printf 'image_completion_marker_sha256=%s\n' "$(sha256sum "${completion_marker}" | awk '{print $1}')"
  printf 'validated_image_receipt=%s\n' "${CONTAINER}"
}

verify_archived_blob() {
  local relative_path=$1
  local snapshot=${SNAPSHOT_ROOT}/${relative_path}
  local expected_blob
  local actual_blob
  local snapshot_sha256

  test -f "${snapshot}"
  test ! -L "${snapshot}"
  expected_blob=$(git -C "${SEMANTIC_WORKTREE}" rev-parse "${EXPECTED_REPO_SHA}:${relative_path}")
  actual_blob=$(git -C "${SEMANTIC_WORKTREE}" hash-object "${snapshot}")
  test "${actual_blob}" = "${expected_blob}"
  snapshot_sha256=$(sha256sum "${snapshot}" | awk '{print $1}')
  printf 'snapshot_path=%s git_blob=%s sha256=%s\n' "${relative_path}" "${expected_blob}" "${snapshot_sha256}"
}

cleanup() {
  local exit_status=$?

  trap - EXIT
  rm -rf -- "${SCRATCH_DIRECTORY}"
  exit "${exit_status}"
}

for authorization_variable in \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HF_API_TOKEN \
  HUGGINGFACE_TOKEN \
  NVIDIA_API_KEY \
  NGC_API_KEY \
  AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN; do
  unset "${authorization_variable}"
  if [[ -n ${!authorization_variable:-} ]]; then
    echo "Authorization environment could not be cleared: ${authorization_variable}" >&2
    exit 2
  fi
done

require_clean_git_tree "${SCRIPT_ROOT}" "${EXPECTED_TOOLING_SHA}"
require_clean_git_tree "${SEMANTIC_WORKTREE}" "${EXPECTED_REPO_SHA}"
validate_image_receipt
test -d "${OUTPUT_ROOT}"
test ! -L "${OUTPUT_ROOT}"
test -d "${OUTPUT_ROOT}/logs"
test ! -e "${SCRATCH_DIRECTORY}"
mkdir -p "${SNAPSHOT_ROOT}"
trap cleanup EXIT

git -C "${SEMANTIC_WORKTREE}" archive --format=tar "${EXPECTED_REPO_SHA}" | tar -x -C "${SNAPSHOT_ROOT}"
for relative_path in "${SNAPSHOT_BLOBS[@]}"; do
  verify_archived_blob "${relative_path}"
done

export TMPDIR=${SCRATCH_DIRECTORY}/tmp
export PYTHONPYCACHEPREFIX=${SCRATCH_DIRECTORY}/pycache
export XDG_CACHE_HOME=${SCRATCH_DIRECTORY}/xdg-cache
export UV_CACHE_DIR=${SCRATCH_DIRECTORY}/uv-cache
export HF_HOME=${SCRATCH_DIRECTORY}/hf-cache
export PYTHONPATH=${SNAPSHOT_ROOT}
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export NO_VCS_VERSION=1
export SLURM_EXPORT_ENV=ALL
mkdir -p \
  "${TMPDIR}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}"

"${SRUN}" \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid/scratch:/raid/scratch \
  --container-remap-root \
  /bin/bash -s -- \
  "${SNAPSHOT_ROOT}" \
  "${OUTPUT_ROOT}" \
  "${EXPECTED_FINAL}" \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${RAW_MANIFEST_FILENAME}" \
  "${CONTAINER_PYTHON}" <<'CONTAINER_BODY'
set -Eeuo pipefail

readonly SNAPSHOT_ROOT=$1
readonly OUTPUT_ROOT=$2
readonly EXPECTED_FINAL=$3
readonly EXPECTED_MANIFEST_SHA256=$4
readonly RAW_MANIFEST_FILENAME=$5
readonly CONTAINER_PYTHON=$6
readonly STAGER=${SNAPSHOT_ROOT}/tools/stage_precision_policy_source_metadata.py

write_expected_payload_paths() {
  cat <<'PAYLOAD_PATHS'
checkpoints/kimi_k2/config.json
checkpoints/kimi_k2/model.safetensors.index.json
checkpoints/kimi_k2/safetensors_header_manifest.json
checkpoints/kimi_k25/config.json
checkpoints/kimi_k25/model.safetensors.index.json
checkpoints/kimi_k25/safetensors_header_manifest.json
checkpoints/kimi_k3/config.json
checkpoints/kimi_k3/model.safetensors.index.json
checkpoints/kimi_k3/safetensors_header_manifest.json
checkpoints/nemotron_lightning_nvfp4/config.json
checkpoints/nemotron_lightning_nvfp4/model.safetensors.index.json
checkpoints/nemotron_lightning_nvfp4/safetensors_header_manifest.json
checkpoints/qwen3_bf16/config.json
checkpoints/qwen3_bf16/model.safetensors.index.json
checkpoints/qwen3_bf16/safetensors_header_manifest.json
checkpoints/qwen_a95b_fp8/config.json
checkpoints/qwen_a95b_fp8/model.safetensors.index.json
checkpoints/qwen_a95b_fp8/safetensors_header_byte_lengths.json
checkpoints/qwen_a95b_fp8/safetensors_header_manifest.json
PAYLOAD_PATHS
}

write_expected_directories() {
  cat <<'DIRECTORIES'
.
checkpoints
checkpoints/kimi_k2
checkpoints/kimi_k25
checkpoints/kimi_k3
checkpoints/nemotron_lightning_nvfp4
checkpoints/qwen3_bf16
checkpoints/qwen_a95b_fp8
DIRECTORIES
}

validate_published_tree() {
  local root=$1
  local actual_directories=${TMPDIR}/actual-directories
  local actual_files=${TMPDIR}/actual-files
  local actual_payload_paths=${TMPDIR}/actual-payload-paths
  local expected_directories=${TMPDIR}/expected-directories
  local expected_files=${TMPDIR}/expected-files
  local expected_payload_paths=${TMPDIR}/expected-payload-paths
  local manifest=${root}/${RAW_MANIFEST_FILENAME}
  local manifest_paths=${TMPDIR}/manifest-paths
  local manifest_sha256
  local unexpected_special

  test "${root}" = "${EXPECTED_FINAL}"
  test -d "${root}"
  test ! -L "${root}"
  test -f "${manifest}"
  test ! -L "${manifest}"
  test -z "$(find "${root}" -type l -print -quit)"
  unexpected_special=$(find "${root}" ! -type d ! -type f ! -type l -print -quit)
  test -z "${unexpected_special}"

  write_expected_payload_paths | LC_ALL=C sort >"${expected_payload_paths}"
  write_expected_directories | LC_ALL=C sort >"${expected_directories}"
  {
    cat "${expected_payload_paths}"
    printf '%s\n' "${RAW_MANIFEST_FILENAME}"
  } | LC_ALL=C sort >"${expected_files}"
  (
    cd "${root}"
    find . -type d -print | sed 's#^\./##; s#^$#.#' | LC_ALL=C sort >"${actual_directories}"
    find . -type f -print | sed 's#^\./##' | LC_ALL=C sort >"${actual_files}"
    find checkpoints -type f -print | LC_ALL=C sort >"${actual_payload_paths}"
  )
  diff -u "${expected_directories}" "${actual_directories}"
  diff -u "${expected_files}" "${actual_files}"
  diff -u "${expected_payload_paths}" "${actual_payload_paths}"

  test "$(wc -l <"${manifest}" | tr -d ' ')" = 19
  awk 'NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }' "${manifest}"
  awk '{print $2}' "${manifest}" | LC_ALL=C sort >"${manifest_paths}"
  diff -u "${expected_payload_paths}" "${manifest_paths}"
  manifest_sha256=$(sha256sum "${manifest}" | awk '{print $1}')
  test "${manifest_sha256}" = "${EXPECTED_MANIFEST_SHA256}"
  (
    cd "${root}"
    sha256sum --check --strict "${RAW_MANIFEST_FILENAME}"
  )
}

test "${PYTHONPATH}" = "${SNAPSHOT_ROOT}"
test "${PYTHONNOUSERSITE}" = 1
test "${PYTHONSAFEPATH}" = 1
unset \
  NRL_CONTAINER \
  NRL_NSYS_WORKER_PATTERNS \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HF_API_TOKEN \
  HUGGINGFACE_TOKEN \
  NVIDIA_API_KEY \
  NGC_API_KEY \
  AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN

runtime_version=$("${CONTAINER_PYTHON}" -P -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
IFS=. read -r runtime_major runtime_minor runtime_patch <<<"${runtime_version}"
[[ ${runtime_major} =~ ^[0-9]+$ && ${runtime_minor} =~ ^[0-9]+$ && ${runtime_patch} =~ ^[0-9]+$ ]]
if (( runtime_major != 3 || runtime_minor < 12 )); then
  echo "Python >=3.12,<4 is required; found ${runtime_version}" >&2
  exit 2
fi
printf 'python_runtime_path=%s\n' "${CONTAINER_PYTHON}"
printf 'python_runtime_version=%s\n' "${runtime_version}"

"${CONTAINER_PYTHON}" -P - "${SNAPSHOT_ROOT}" <<'PYTHON_IMPORT_PROBE'
import importlib
import pathlib
import sys

snapshot_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected = {
    "tools.stage_precision_policy_source_metadata": "tools/stage_precision_policy_source_metadata.py",
    "tools.capture_precision_policy_source_evidence": "tools/capture_precision_policy_source_evidence.py",
    "nemo_rl.precision_policy.semantic": "nemo_rl/precision_policy/semantic.py",
    "nemo_rl.precision_policy.source_formats": "nemo_rl/precision_policy/source_formats.py",
}
for module_name, relative_path in expected.items():
    module = importlib.import_module(module_name)
    actual_path = pathlib.Path(module.__file__).resolve(strict=True)
    expected_path = (snapshot_root / relative_path).resolve(strict=True)
    if actual_path != expected_path or not actual_path.is_relative_to(snapshot_root):
        raise SystemExit(
            f"import escaped committed snapshot: {module_name} -> {actual_path}"
        )
    print(f"import_origin={module_name}:{actual_path}")
PYTHON_IMPORT_PROBE

first_output=$("${CONTAINER_PYTHON}" -P "${STAGER}" --output-root "${OUTPUT_ROOT}")
test "${first_output}" = "${EXPECTED_FINAL}"
printf 'first_stage_output=%s\n' "${first_output}"
validate_published_tree "${first_output}"

second_output=$(
  env \
    http_proxy=http://127.0.0.1:9 \
    https_proxy=http://127.0.0.1:9 \
    HTTP_PROXY=http://127.0.0.1:9 \
    HTTPS_PROXY=http://127.0.0.1:9 \
    all_proxy=http://127.0.0.1:9 \
    ALL_PROXY=http://127.0.0.1:9 \
    no_proxy= \
    NO_PROXY= \
    "${CONTAINER_PYTHON}" -P "${STAGER}" --output-root "${OUTPUT_ROOT}"
)
test "${second_output}" = "${first_output}"
printf 'second_stage_output=%s\n' "${second_output}"
validate_published_tree "${second_output}"
CONTAINER_BODY

test -d "${EXPECTED_FINAL}"
printf 'staged_metadata=%s\n' "${EXPECTED_FINAL}"
