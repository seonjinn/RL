#!/bin/bash

set -euo pipefail

readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly DESTINATION=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly METADATA=${DESTINATION}.metadata.txt
readonly COMPLETION_MARKER=${DESTINATION}.complete

if (( $# == 0 )); then
  image_path=${DESTINATION}
  metadata_path=${METADATA}
  marker_path=${COMPLETION_MARKER}
elif (( $# == 3 )); then
  image_path=$1
  metadata_path=$2
  marker_path=$3
else
  echo "Usage: $0 [IMAGE METADATA COMPLETION_MARKER]" >&2
  exit 2
fi

test "${image_path}" = "${DESTINATION}"
test "${metadata_path}" = "${METADATA}"
test "${marker_path}" = "${COMPLETION_MARKER}"

command -v sha256sum >/dev/null
test -f "${image_path}"
test "$(sha256sum "${image_path}" | awk '{print $1}')" = "${EXPECTED_SHA256}"

test -f "${metadata_path}"
test "$(wc -l <"${metadata_path}")" -eq 7
grep -Fx -- "source=${SOURCE}" "${metadata_path}" >/dev/null
grep -Fx -- 'source_cluster=ptyche' "${metadata_path}" >/dev/null
grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata_path}" >/dev/null
grep -Fx -- 'source_smoke_job=2721177' "${metadata_path}" >/dev/null
grep -Ex -- 'download_job=[0-9]+' "${metadata_path}" >/dev/null
grep -Ex -- 'downloaded_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${metadata_path}" >/dev/null
grep -Fx -- "sha256=${EXPECTED_SHA256}" "${metadata_path}" >/dev/null

test -f "${marker_path}"
test "$(wc -l <"${marker_path}")" -eq 4
grep -Fx -- "destination=${DESTINATION}" "${marker_path}" >/dev/null
grep -Fx -- "source=${SOURCE}" "${marker_path}" >/dev/null
grep -Fx -- "sha256=${EXPECTED_SHA256}" "${marker_path}" >/dev/null
grep -Ex -- 'completed_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${marker_path}" >/dev/null

printf 'validated_image=%s\n' "${image_path}"
