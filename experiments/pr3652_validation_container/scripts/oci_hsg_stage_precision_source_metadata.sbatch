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
: "${PRODUCT_WORKTREE:?Set PRODUCT_WORKTREE to the absolute clean product worktree root}"
: "${EXPECTED_PRODUCT_SHA:?Set EXPECTED_PRODUCT_SHA to the product worktree HEAD}"

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
  tools/precision_policy_source_artifacts.py
)

if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  echo 'SLURM_JOB_ID must be a decimal job ID' >&2
  exit 2
fi

readonly SCRATCH_ROOT=/raid/scratch
readonly USER_SCRATCH_ROOT=${SCRATCH_ROOT}/sna
readonly SCRATCH_BASE=${USER_SCRATCH_ROOT}/nemo-rl-semantic-precision-metadata
readonly SCRATCH_DIRECTORY=${SCRATCH_BASE}/oci-stage-${SLURM_JOB_ID}
readonly SNAPSHOT_ROOT=${SCRATCH_DIRECTORY}/product-repo
readonly SCRATCH_OWNER_FILE=${SCRATCH_DIRECTORY}/.stage-owner
SCRATCH_OWNED=0
SCRATCH_DIRECTORY_IDENTITY=
SCRATCH_OWNER_TOKEN=

die() {
  echo "ERROR: $*" >&2
  exit 2
}

file_identity() {
  local path=$1
  local identity

  if identity=$(stat -Lc '%d:%i' "${path}" 2>/dev/null); then
    :
  elif identity=$(stat -f '%d:%i' "${path}" 2>/dev/null); then
    :
  else
    return 1
  fi
  [[ "${identity}" =~ ^[0-9]+:[0-9]+$ ]] || return 1
  printf '%s\n' "${identity}"
}

file_mode() {
  local path=$1
  local mode

  if mode=$(stat -Lc '%a' "${path}" 2>/dev/null); then
    :
  elif mode=$(stat -f '%Lp' "${path}" 2>/dev/null); then
    :
  else
    return 1
  fi
  [[ "${mode}" =~ ^[0-7]{3,4}$ ]] || return 1
  printf '%s\n' "${mode}"
}

require_canonical_directory() {
  local path=$1
  local resolved

  [[ -d "${path}" && ! -L "${path}" ]] ||
    die "required scratch parent is not a plain directory: ${path}"
  resolved=$(cd "${path}" && pwd -P) ||
    die "required scratch parent is not accessible: ${path}"
  [[ "${resolved}" == "${path}" ]] ||
    die "required scratch parent is not canonical: ${path}"
}

ensure_owned_private_directory() {
  local path=$1
  local mode

  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    mkdir -m 700 "${path}" 2>/dev/null || true
  fi
  require_canonical_directory "${path}"
  [[ -O "${path}" ]] || die "scratch directory is not owned by the job user: ${path}"
  mode=$(file_mode "${path}") || die "could not inspect scratch permissions: ${path}"
  (((8#${mode} & 8#022) == 0)) ||
    die "scratch directory must not be group- or other-writable: ${path}"
}

require_job_scratch_lifecycle() {
  local actual_identity
  local actual_owner_token=

  if [[ ! -d "${SCRATCH_DIRECTORY}" || -L "${SCRATCH_DIRECTORY}" ||
    ! -d "${TMPDIR}" || -L "${TMPDIR}" ||
    ! -f "${SCRATCH_OWNER_FILE}" || -L "${SCRATCH_OWNER_FILE}" ]]; then
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  fi
  actual_identity=$(file_identity "${SCRATCH_DIRECTORY}") || {
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  }
  IFS= read -r actual_owner_token <"${SCRATCH_OWNER_FILE}" || true
  if [[ "${actual_identity}" != "${SCRATCH_DIRECTORY_IDENTITY}" ||
    "${actual_owner_token}" != "${SCRATCH_OWNER_TOKEN}" ]]; then
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  fi
}

require_clean_git_tree() {
  local path=$1
  local expected_sha=$2
  local inside_work_tree
  local top_level
  local actual_sha
  local git_status

  [[ "${path}" = /* ]]
  test ! -L "${path}"
  inside_work_tree=$(git -C "${path}" rev-parse --is-inside-work-tree) || {
    echo "ERROR: could not inspect Git worktree state: ${path}" >&2
    return 1
  }
  [[ "${inside_work_tree}" == true ]] || {
    echo "ERROR: path is not inside a Git worktree: ${path}" >&2
    return 1
  }
  top_level=$(git -C "${path}" rev-parse --show-toplevel) || {
    echo "ERROR: could not resolve Git worktree root: ${path}" >&2
    return 1
  }
  [[ "${top_level}" == "${path}" ]] || {
    echo "ERROR: path is not the exact Git worktree root: ${path}" >&2
    return 1
  }
  actual_sha=$(git -C "${path}" rev-parse HEAD) || {
    echo "ERROR: could not resolve Git HEAD: ${path}" >&2
    return 1
  }
  [[ "${actual_sha}" == "${expected_sha}" ]] || {
    echo "ERROR: Git HEAD does not match the expected commit: ${path}" >&2
    return 1
  }
  git_status=$(git -C "${path}" status --porcelain) || {
    echo "ERROR: could not inspect Git worktree cleanliness: ${path}" >&2
    return 1
  }
  [[ -z "${git_status}" ]] || {
    echo "ERROR: Git worktree is not clean: ${path}" >&2
    return 1
  }
}

validate_image_receipt() {
  local metadata=${CONTAINER}.metadata.txt
  local completion_marker=${CONTAINER}.complete
  local metadata_line_count
  local completion_line_count
  local metadata_sha256
  local completion_marker_sha256

  test -f "${CONTAINER}"
  test ! -L "${CONTAINER}"
  test -f "${metadata}"
  test ! -L "${metadata}"
  metadata_line_count=$(wc -l <"${metadata}") || {
    echo "ERROR: could not count image metadata receipt lines: ${metadata}" >&2
    return 1
  }
  metadata_line_count=${metadata_line_count//[[:space:]]/}
  [[ "${metadata_line_count}" == 7 ]] || {
    echo "ERROR: image metadata receipt must contain exactly 7 lines: ${metadata}" >&2
    return 1
  }
  grep -Fx -- "source=${CONTAINER_SOURCE}" "${metadata}" >/dev/null
  grep -Fx -- 'source_cluster=ptyche' "${metadata}" >/dev/null
  grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata}" >/dev/null
  grep -Fx -- 'source_smoke_job=2721177' "${metadata}" >/dev/null
  grep -Ex -- 'download_job=[0-9]+' "${metadata}" >/dev/null
  grep -Ex -- 'downloaded_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${metadata}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_CONTAINER_SHA256}" "${metadata}" >/dev/null

  test -f "${completion_marker}"
  test ! -L "${completion_marker}"
  completion_line_count=$(wc -l <"${completion_marker}") || {
    echo "ERROR: could not count image completion receipt lines: ${completion_marker}" >&2
    return 1
  }
  completion_line_count=${completion_line_count//[[:space:]]/}
  [[ "${completion_line_count}" == 4 ]] || {
    echo "ERROR: image completion receipt must contain exactly 4 lines: ${completion_marker}" >&2
    return 1
  }
  grep -Fx -- "destination=${CONTAINER}" "${completion_marker}" >/dev/null
  grep -Fx -- "source=${CONTAINER_SOURCE}" "${completion_marker}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_CONTAINER_SHA256}" "${completion_marker}" >/dev/null
  grep -Ex -- 'completed_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${completion_marker}" >/dev/null

  metadata_sha256=$(sha256sum "${metadata}" | awk '{print $1}') || {
    echo "ERROR: could not hash image metadata receipt: ${metadata}" >&2
    return 1
  }
  completion_marker_sha256=$(sha256sum "${completion_marker}" | awk '{print $1}') || {
    echo "ERROR: could not hash image completion receipt: ${completion_marker}" >&2
    return 1
  }
  printf 'expected_image_sha256=%s\n' "${EXPECTED_CONTAINER_SHA256}"
  printf 'image_metadata_sha256=%s\n' "${metadata_sha256}"
  printf 'image_completion_marker_sha256=%s\n' "${completion_marker_sha256}"
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
  expected_blob=$(git -C "${PRODUCT_WORKTREE}" rev-parse "${EXPECTED_PRODUCT_SHA}:${relative_path}")
  actual_blob=$(git -C "${PRODUCT_WORKTREE}" hash-object "${snapshot}")
  test "${actual_blob}" = "${expected_blob}"
  snapshot_sha256=$(sha256sum "${snapshot}" | awk '{print $1}')
  printf 'snapshot_path=%s git_blob=%s sha256=%s\n' "${relative_path}" "${expected_blob}" "${snapshot_sha256}"
}

cleanup() {
  local exit_status=$?
  local cleanup_status=0
  local quarantine_directory=
  local quarantine_identity=
  local actual_quarantine_identity=
  local quarantined_scratch=
  local actual_scratch_identity=
  local actual_owner_token=

  trap - EXIT
  set +e
  if ((SCRATCH_OWNED == 1)); then
    quarantine_directory=$(mktemp -d "${SCRATCH_BASE}/.cleanup-${SLURM_JOB_ID}.XXXXXX")
    if [[ -d "${quarantine_directory}" && ! -L "${quarantine_directory}" ]]; then
      chmod 700 "${quarantine_directory}" || cleanup_status=97
      quarantine_identity=$(file_identity "${quarantine_directory}")
      quarantined_scratch=${quarantine_directory}/$(basename "${SCRATCH_DIRECTORY}")
      if mv "${SCRATCH_DIRECTORY}" "${quarantine_directory}/"; then
        actual_scratch_identity=$(file_identity "${quarantined_scratch}")
        if [[ -f "${quarantined_scratch}/.stage-owner" &&
          ! -L "${quarantined_scratch}/.stage-owner" ]]; then
          IFS= read -r actual_owner_token \
            <"${quarantined_scratch}/.stage-owner" || true
        fi
        if [[ "${actual_scratch_identity}" == "${SCRATCH_DIRECTORY_IDENTITY}" &&
          "${actual_owner_token}" == "${SCRATCH_OWNER_TOKEN}" ]]; then
          rm -rf -- "${quarantined_scratch}" || cleanup_status=97
        else
          echo "ERROR: preserving unverified quarantined job scratch: ${quarantined_scratch}" >&2
          cleanup_status=97
        fi
      else
        echo "ERROR: could not quarantine job scratch: ${SCRATCH_DIRECTORY}" >&2
        cleanup_status=97
      fi
      actual_quarantine_identity=$(file_identity "${quarantine_directory}")
      if [[ -n "${quarantine_identity}" &&
        "${actual_quarantine_identity}" == "${quarantine_identity}" ]]; then
        rmdir "${quarantine_directory}" 2>/dev/null || {
          ((cleanup_status != 0)) || cleanup_status=97
        }
      else
        echo "ERROR: refusing to remove replaced scratch quarantine: ${quarantine_directory}" >&2
        cleanup_status=97
      fi
    else
      echo "ERROR: could not allocate private scratch quarantine under ${SCRATCH_BASE}" >&2
      cleanup_status=97
    fi
  fi
  if ((exit_status == 0 && cleanup_status != 0)); then
    exit_status=${cleanup_status}
  fi
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
require_clean_git_tree "${PRODUCT_WORKTREE}" "${EXPECTED_PRODUCT_SHA}"
validate_image_receipt
test -d "${OUTPUT_ROOT}"
test ! -L "${OUTPUT_ROOT}"
test -d "${OUTPUT_ROOT}/logs"
require_canonical_directory "${SCRATCH_ROOT}"
ensure_owned_private_directory "${USER_SCRATCH_ROOT}"
ensure_owned_private_directory "${SCRATCH_BASE}"
trap cleanup EXIT
if ! mkdir -m 700 "${SCRATCH_DIRECTORY}"; then
  die "could not atomically claim job-owned scratch: ${SCRATCH_DIRECTORY}"
fi
SCRATCH_OWNED=1
SCRATCH_DIRECTORY_IDENTITY=$(file_identity "${SCRATCH_DIRECTORY}") ||
  die "could not identify job-owned scratch: ${SCRATCH_DIRECTORY}"
SCRATCH_OWNER_TOKEN="job-${SLURM_JOB_ID}:pid-$$:tooling-${EXPECTED_TOOLING_SHA}"
(set -o noclobber; printf '%s\n' "${SCRATCH_OWNER_TOKEN}" >"${SCRATCH_OWNER_FILE}") ||
  die "could not create job-owned scratch marker: ${SCRATCH_OWNER_FILE}"
chmod 600 "${SCRATCH_OWNER_FILE}"
mkdir "${SNAPSHOT_ROOT}"

git -C "${PRODUCT_WORKTREE}" archive \
  --format=tar \
  "${EXPECTED_PRODUCT_SHA}" \
  -- \
  "${SNAPSHOT_BLOBS[@]}" \
  | tar -x -C "${SNAPSHOT_ROOT}"
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
require_job_scratch_lifecycle

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
  "${CONTAINER_PYTHON}" \
  "${SCRATCH_DIRECTORY}" \
  "${SCRATCH_DIRECTORY_IDENTITY}" \
  "${SCRATCH_OWNER_TOKEN}" <<'CONTAINER_BODY'
set -Eeuo pipefail

readonly SNAPSHOT_ROOT=$1
readonly OUTPUT_ROOT=$2
readonly EXPECTED_FINAL=$3
readonly EXPECTED_MANIFEST_SHA256=$4
readonly RAW_MANIFEST_FILENAME=$5
readonly CONTAINER_PYTHON=$6
readonly SCRATCH_DIRECTORY=$7
readonly EXPECTED_SCRATCH_IDENTITY=$8
readonly EXPECTED_SCRATCH_OWNER_TOKEN=$9
readonly SCRATCH_OWNER_FILE=${SCRATCH_DIRECTORY}/.stage-owner
readonly STAGER=${SNAPSHOT_ROOT}/tools/stage_precision_policy_source_metadata.py

require_container_scratch_lifecycle() {
  local actual_identity
  local actual_owner_token=

  if [[ ! -d "${SCRATCH_DIRECTORY}" || -L "${SCRATCH_DIRECTORY}" ||
    ! -d "${TMPDIR}" || -L "${TMPDIR}" ||
    ! -f "${SCRATCH_OWNER_FILE}" || -L "${SCRATCH_OWNER_FILE}" ]]; then
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  fi
  if actual_identity=$(stat -Lc '%d:%i' "${SCRATCH_DIRECTORY}" 2>/dev/null); then
    :
  elif actual_identity=$(stat -f '%d:%i' "${SCRATCH_DIRECTORY}" 2>/dev/null); then
    :
  else
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  fi
  IFS= read -r actual_owner_token <"${SCRATCH_OWNER_FILE}" || true
  if [[ "${actual_identity}" != "${EXPECTED_SCRATCH_IDENTITY}" ||
    "${actual_owner_token}" != "${EXPECTED_SCRATCH_OWNER_TOKEN}" ]]; then
    echo "ERROR: job-owned scratch lifecycle violated during container stage: ${SCRATCH_DIRECTORY}" >&2
    return 2
  fi
}

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
  local manifest=${root}/${RAW_MANIFEST_FILENAME}
  local manifest_sha256
  local manifest_line_count
  local symlink_path
  local unexpected_special
  local expected_payload_paths
  local expected_directories
  local expected_files
  local actual_directories
  local actual_files
  local actual_payload_paths
  local manifest_paths

  test "${root}" = "${EXPECTED_FINAL}"
  test -d "${root}"
  test ! -L "${root}"
  test -f "${manifest}"
  test ! -L "${manifest}"
  symlink_path=$(find "${root}" -type l -print -quit) || {
    echo 'published metadata symlink enumeration failed' >&2
    return 1
  }
  [[ -z "${symlink_path}" ]] || {
    echo 'published metadata tree contains a symlink' >&2
    return 1
  }
  unexpected_special=$(find "${root}" ! -type d ! -type f ! -type l -print -quit) || {
    echo 'published metadata special-file enumeration failed' >&2
    return 1
  }
  [[ -z "${unexpected_special}" ]] || {
    echo 'published metadata tree contains a special file' >&2
    return 1
  }

  expected_payload_paths=$(write_expected_payload_paths | LC_ALL=C sort) ||
    return 1
  expected_directories=$(write_expected_directories | LC_ALL=C sort) ||
    return 1
  expected_files=$(
    {
      printf '%s\n' "${expected_payload_paths}"
      printf '%s\n' "${RAW_MANIFEST_FILENAME}"
    } | LC_ALL=C sort
  ) || return 1
  actual_directories=$(
    cd "${root}" || exit
    find . -type d -print | sed 's#^\./##; s#^$#.#' | LC_ALL=C sort
  ) || {
    echo 'published metadata directory enumeration failed' >&2
    return 1
  }
  actual_files=$(
    cd "${root}" || exit
    find . -type f -print | sed 's#^\./##' | LC_ALL=C sort
  ) || {
    echo 'published metadata file enumeration failed' >&2
    return 1
  }
  actual_payload_paths=$(
    cd "${root}" || exit
    find checkpoints -type f -print | LC_ALL=C sort
  ) || {
    echo 'published metadata payload enumeration failed' >&2
    return 1
  }
  if [[ "${actual_directories}" != "${expected_directories}" ||
    "${actual_files}" != "${expected_files}" ||
    "${actual_payload_paths}" != "${expected_payload_paths}" ]]; then
    echo 'published metadata topology differs from the exact expected tree' >&2
    return 1
  fi

  manifest_line_count=$(wc -l <"${manifest}") || {
    echo 'published metadata manifest line count failed' >&2
    return 1
  }
  manifest_line_count=${manifest_line_count//[[:space:]]/}
  [[ "${manifest_line_count}" == 19 ]] || {
    echo 'published metadata manifest must contain exactly 19 lines' >&2
    return 1
  }
  awk 'NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }' "${manifest}"
  manifest_paths=$(awk '{print $2}' "${manifest}" | LC_ALL=C sort) || {
    echo 'published metadata manifest path enumeration failed' >&2
    return 1
  }
  if [[ "${manifest_paths}" != "${expected_payload_paths}" ]]; then
    echo 'published metadata manifest paths differ from the exact payload tree' >&2
    return 1
  fi
  manifest_sha256=$(sha256sum "${manifest}" | awk '{print $1}') || {
    echo 'published metadata manifest hashing failed' >&2
    return 1
  }
  [[ "${manifest_sha256}" == "${EXPECTED_MANIFEST_SHA256}" ]] || {
    echo 'published metadata manifest digest differs from the expected digest' >&2
    return 1
  }
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

runtime_version=$("${CONTAINER_PYTHON}" -S -P -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
IFS=. read -r runtime_major runtime_minor runtime_patch <<<"${runtime_version}"
[[ ${runtime_major} =~ ^[0-9]+$ && ${runtime_minor} =~ ^[0-9]+$ && ${runtime_patch} =~ ^[0-9]+$ ]]
if (( runtime_major != 3 || runtime_minor < 12 )); then
  echo "Python >=3.12,<4 is required; found ${runtime_version}" >&2
  exit 2
fi
printf 'python_runtime_path=%s\n' "${CONTAINER_PYTHON}"
printf 'python_runtime_version=%s\n' "${runtime_version}"

"${CONTAINER_PYTHON}" -S -P - "${SNAPSHOT_ROOT}" <<'PYTHON_IMPORT_PROBE'
import importlib
import importlib.abc
import pathlib
import sys

snapshot_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
blocked = (
    "nemo_rl",
    "pydantic",
    "typing_extensions",
    "tools.capture_precision_policy_source_evidence",
)
attempted: list[str] = []


def is_blocked(module_name: str) -> bool:
    return any(
        module_name == blocked_name
        or module_name.startswith(f"{blocked_name}.")
        for blocked_name in blocked
    )


class BlockRuntimeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if is_blocked(fullname):
            attempted.append(fullname)
            raise ImportError(f"blocked import attempted: {fullname}")
        return None


if not sys.flags.no_site:
    raise SystemExit("metadata import probe requires Python -S")
if not sys.flags.safe_path:
    raise SystemExit("metadata import probe requires Python -P")
preloaded = sorted(name for name in sys.modules if is_blocked(name))
if preloaded:
    raise SystemExit(f"blocked imports loaded before probe: {', '.join(preloaded)}")
sys.meta_path.insert(0, BlockRuntimeDependencies())
expected = {
    "tools.stage_precision_policy_source_metadata": "tools/stage_precision_policy_source_metadata.py",
    "tools.precision_policy_source_artifacts": "tools/precision_policy_source_artifacts.py",
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
if attempted:
    raise SystemExit(f"blocked imports attempted: {', '.join(dict.fromkeys(attempted))}")
loaded = sorted(name for name in sys.modules if is_blocked(name))
if loaded:
    raise SystemExit(f"blocked imports loaded: {', '.join(loaded)}")
print("blocked_imports=not-attempted-not-loaded")
PYTHON_IMPORT_PROBE

run_stager_site_free_twice() {
  "${CONTAINER_PYTHON}" -S -P - \
    "${SNAPSHOT_ROOT}" \
    "${STAGER}" \
    "${OUTPUT_ROOT}" \
    "${SCRATCH_DIRECTORY}" \
    "${TMPDIR}" \
    "${EXPECTED_SCRATCH_IDENTITY}" \
    "${EXPECTED_SCRATCH_OWNER_TOKEN}" <<'PYTHON_STAGER_RUNNER'
import contextlib
import importlib
import importlib.abc
import io
import os
import pathlib
import stat
import sys

snapshot_root_argument = pathlib.Path(sys.argv[1])
stager_argument = pathlib.Path(sys.argv[2])
output_root = sys.argv[3]
scratch_directory = pathlib.Path(sys.argv[4])
temporary_directory = pathlib.Path(sys.argv[5])
expected_scratch_identity = tuple(int(part) for part in sys.argv[6].split(":"))
expected_scratch_owner_token = sys.argv[7]
scratch_owner_file = scratch_directory / ".stage-owner"


def scratch_lifecycle_error() -> SystemExit:
    return SystemExit(
        f"job-owned scratch lifecycle violated during container stage: "
        f"{scratch_directory}"
    )


def require_scratch_lifecycle() -> None:
    owner_descriptor = None
    try:
        scratch_stat = os.lstat(scratch_directory)
        temporary_stat = os.lstat(temporary_directory)
        if (
            not stat.S_ISDIR(scratch_stat.st_mode)
            or not stat.S_ISDIR(temporary_stat.st_mode)
            or (scratch_stat.st_dev, scratch_stat.st_ino)
            != expected_scratch_identity
        ):
            raise scratch_lifecycle_error()
        owner_descriptor = os.open(
            scratch_owner_file,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        owner_opened_stat = os.fstat(owner_descriptor)
        if not stat.S_ISREG(owner_opened_stat.st_mode) or owner_opened_stat.st_size > 512:
            raise scratch_lifecycle_error()
        owner_bytes = os.read(owner_descriptor, 513)
        owner_final_stat = os.fstat(owner_descriptor)
        owner_path_stat = os.lstat(scratch_owner_file)
        owner_identity = (
            owner_opened_stat.st_dev,
            owner_opened_stat.st_ino,
            owner_opened_stat.st_mode,
            owner_opened_stat.st_size,
            owner_opened_stat.st_mtime_ns,
            owner_opened_stat.st_ctime_ns,
        )
        if (
            owner_bytes != f"{expected_scratch_owner_token}\n".encode()
            or (
                owner_final_stat.st_dev,
                owner_final_stat.st_ino,
                owner_final_stat.st_mode,
                owner_final_stat.st_size,
                owner_final_stat.st_mtime_ns,
                owner_final_stat.st_ctime_ns,
            )
            != owner_identity
            or (
                owner_path_stat.st_dev,
                owner_path_stat.st_ino,
                owner_path_stat.st_mode,
                owner_path_stat.st_size,
                owner_path_stat.st_mtime_ns,
                owner_path_stat.st_ctime_ns,
            )
            != owner_identity
        ):
            raise scratch_lifecycle_error()
    except (OSError, ValueError):
        raise scratch_lifecycle_error() from None
    finally:
        if owner_descriptor is not None:
            try:
                os.close(owner_descriptor)
            except OSError:
                raise scratch_lifecycle_error() from None


if len(expected_scratch_identity) != 2:
    raise scratch_lifecycle_error()
require_scratch_lifecycle()
try:
    snapshot_root = snapshot_root_argument.resolve(strict=True)
    stager = stager_argument.resolve(strict=True)
except OSError:
    raise scratch_lifecycle_error() from None
expected_stager = (
    snapshot_root / "tools/stage_precision_policy_source_metadata.py"
).resolve(strict=True)
expected_leaf = (
    snapshot_root / "tools/precision_policy_source_artifacts.py"
).resolve(strict=True)
blocked = (
    "nemo_rl",
    "pydantic",
    "typing_extensions",
    "tools.capture_precision_policy_source_evidence",
)
attempted: list[str] = []


def is_blocked(module_name: str) -> bool:
    return any(
        module_name == blocked_name
        or module_name.startswith(f"{blocked_name}.")
        for blocked_name in blocked
    )


class BlockRuntimeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if is_blocked(fullname):
            attempted.append(fullname)
            raise ImportError(f"blocked import attempted: {fullname}")
        return None


if not sys.flags.no_site:
    raise SystemExit("metadata stager requires Python -S")
if not sys.flags.safe_path:
    raise SystemExit("metadata stager requires Python -P")
if stager != expected_stager or not stager.is_relative_to(snapshot_root):
    raise SystemExit(f"stager escaped committed snapshot: {stager}")
preloaded = sorted(name for name in sys.modules if is_blocked(name))
if preloaded:
    raise SystemExit(f"blocked imports loaded before stager: {', '.join(preloaded)}")
sys.meta_path.insert(0, BlockRuntimeDependencies())
expected_modules = {
    "tools.stage_precision_policy_source_metadata": expected_stager,
    "tools.precision_policy_source_artifacts": expected_leaf,
}
verified_modules = {}
verified_module_files = {}
for module_name, expected_path in expected_modules.items():
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise SystemExit(f"metadata module has no origin: {module_name}")
    actual_path = pathlib.Path(module_file).resolve(strict=True)
    if actual_path != expected_path or not actual_path.is_relative_to(snapshot_root):
        raise SystemExit(f"metadata module escaped committed snapshot: {module_name} -> {actual_path}")
    verified_modules[module_name] = module
    verified_module_files[module_name] = module_file


def require_runtime_identity() -> None:
    if attempted:
        raise SystemExit(
            f"blocked imports attempted: {', '.join(dict.fromkeys(attempted))}"
        )
    loaded = sorted(name for name in sys.modules if is_blocked(name))
    if loaded:
        raise SystemExit(f"blocked imports loaded: {', '.join(loaded)}")
    for module_name, module in verified_modules.items():
        if sys.modules.get(module_name) is not module:
            raise SystemExit(f"verified metadata module identity changed: {module_name}")
        if getattr(module, "__file__", None) != verified_module_files[module_name]:
            raise SystemExit(f"verified metadata module origin changed: {module_name}")


stager_module = verified_modules["tools.stage_precision_policy_source_metadata"]
stager_main = getattr(stager_module, "main", None)
if not callable(stager_main):
    raise SystemExit("metadata stager main is not callable")


def invoke_stager() -> str:
    require_runtime_identity()
    sys.argv = [str(stager), "--output-root", output_root]
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        exit_status = stager_main()
    require_runtime_identity()
    if exit_status != 0:
        raise SystemExit(exit_status)
    output_lines = captured_stdout.getvalue().splitlines()
    if len(output_lines) != 1:
        raise SystemExit("metadata stager must print exactly one output path")
    return output_lines[0]


first_output = invoke_stager()
require_scratch_lifecycle()
for variable in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ[variable] = "http://127.0.0.1:9"
for variable in ("no_proxy", "NO_PROXY"):
    os.environ[variable] = ""
second_output = invoke_stager()
require_scratch_lifecycle()
print(first_output)
print(second_output)
PYTHON_STAGER_RUNNER
}

stage_outputs=$(run_stager_site_free_twice)
require_container_scratch_lifecycle
if [[ "${stage_outputs}" != *$'\n'* || "${stage_outputs#*$'\n'}" == *$'\n'* ]]; then
  echo 'metadata stager runner must return exactly two output paths' >&2
  exit 2
fi
first_output=${stage_outputs%%$'\n'*}
second_output=${stage_outputs#*$'\n'}
test "${first_output}" = "${EXPECTED_FINAL}"
printf 'first_stage_output=%s\n' "${first_output}"
validate_published_tree "${first_output}"

test "${second_output}" = "${first_output}"
printf 'second_stage_output=%s\n' "${second_output}"
validate_published_tree "${second_output}"
CONTAINER_BODY

require_job_scratch_lifecycle
test -d "${EXPECTED_FINAL}"
printf 'staged_metadata=%s\n' "${EXPECTED_FINAL}"
