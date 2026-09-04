#!/bin/bash
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --time=02:00:00
#SBATCH --job-name=nemotron_n3_post-download_validated_nightly
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-download/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/oci-download/slurm-%j.err

set -euo pipefail

readonly SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly DESTINATION=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly PARTIAL=${DESTINATION}.partial.${SLURM_JOB_ID}
readonly METADATA=${DESTINATION}.metadata.txt
readonly PARTIAL_METADATA=${METADATA}.partial.${SLURM_JOB_ID}
readonly COMPLETION_MARKER=${DESTINATION}.complete
readonly PARTIAL_COMPLETION_MARKER=${COMPLETION_MARKER}.partial.${SLURM_JOB_ID}

created_destination=0
created_metadata=0
created_completion_marker=0

same_file() {
  [[ -e "$1" && -e "$2" && "$1" -ef "$2" ]]
}

remove_owned_final_path() {
  local partial_path=$1
  local final_path=$2

  if same_file "${partial_path}" "${final_path}"; then
    rm -f -- "${final_path}"
  fi
}

image_matches_expected() {
  local image_path=$1

  test -f "${image_path}"
  test "$(sha256sum "${image_path}" | awk '{print $1}')" = "${EXPECTED_SHA256}"
}

metadata_matches_expected() {
  local metadata_path=$1

  test -f "${metadata_path}"
  grep -Fx -- "source=${SOURCE}" "${metadata_path}" >/dev/null
  grep -Fx -- 'source_cluster=ptyche' "${metadata_path}" >/dev/null
  grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata_path}" >/dev/null
  grep -Fx -- 'source_smoke_job=2721177' "${metadata_path}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_SHA256}" "${metadata_path}" >/dev/null
}

completion_marker_matches_expected() {
  local marker_path=$1

  test -f "${marker_path}"
  grep -Fx -- "destination=${DESTINATION}" "${marker_path}" >/dev/null
  grep -Fx -- "source=${SOURCE}" "${marker_path}" >/dev/null
  grep -Fx -- "sha256=${EXPECTED_SHA256}" "${marker_path}" >/dev/null
}

cleanup() {
  local exit_status=$?

  trap - EXIT
  if (( exit_status != 0 )); then
    if (( created_completion_marker )); then
      remove_owned_final_path "${PARTIAL_COMPLETION_MARKER}" "${COMPLETION_MARKER}"
    fi
    if (( created_metadata )); then
      remove_owned_final_path "${PARTIAL_METADATA}" "${METADATA}"
    fi
    if (( created_destination )); then
      remove_owned_final_path "${PARTIAL}" "${DESTINATION}"
    fi
  fi
  rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}" "${PARTIAL_COMPLETION_MARKER}"
  exit "${exit_status}"
}
trap cleanup EXIT

command -v rclone >/dev/null
command -v sha256sum >/dev/null

export GOMAXPROCS=8
if [[ -e "${DESTINATION}" ]]; then
  image_matches_expected "${DESTINATION}"
else
  /cm/local/apps/slurm/current/bin/srun -J download-validated-nightly \
    rclone copyto \
    --verbose \
    --checksum \
    --checkers=8 \
    --transfers=4 \
    --multi-thread-streams=32 \
    --buffer-size=128M \
    "${SOURCE}" \
    "${PARTIAL}"

  image_matches_expected "${PARTIAL}"
  if ln -T "${PARTIAL}" "${DESTINATION}"; then
    created_destination=1
  elif [[ -e "${DESTINATION}" ]]; then
    image_matches_expected "${DESTINATION}"
  else
    echo "Unable to publish immutable destination: ${DESTINATION}" >&2
    exit 1
  fi
fi

{
  echo "source=${SOURCE}"
  echo "source_cluster=ptyche"
  echo "source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh"
  echo "source_smoke_job=2721177"
  echo "download_job=${SLURM_JOB_ID}"
  echo "downloaded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sha256=${EXPECTED_SHA256}"
} >"${PARTIAL_METADATA}"

if ln -T "${PARTIAL_METADATA}" "${METADATA}"; then
  created_metadata=1
elif [[ -e "${METADATA}" ]]; then
  metadata_matches_expected "${METADATA}"
else
  echo "Unable to publish immutable metadata: ${METADATA}" >&2
  exit 1
fi

{
  echo "destination=${DESTINATION}"
  echo "source=${SOURCE}"
  echo "sha256=${EXPECTED_SHA256}"
  echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${PARTIAL_COMPLETION_MARKER}"

if ln -T "${PARTIAL_COMPLETION_MARKER}" "${COMPLETION_MARKER}"; then
  created_completion_marker=1
elif [[ -e "${COMPLETION_MARKER}" ]]; then
  completion_marker_matches_expected "${COMPLETION_MARKER}"
else
  echo "Unable to publish immutable completion marker: ${COMPLETION_MARKER}" >&2
  exit 1
fi

image_matches_expected "${DESTINATION}"
metadata_matches_expected "${METADATA}"
completion_marker_matches_expected "${COMPLETION_MARKER}"

rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}" "${PARTIAL_COMPLETION_MARKER}"
trap - EXIT

ls -lh "${DESTINATION}" "${METADATA}" "${COMPLETION_MARKER}"
