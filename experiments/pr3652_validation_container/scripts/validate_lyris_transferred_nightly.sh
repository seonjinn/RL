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

readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly EXPECTED_RCLONE_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a
readonly SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly DESTINATION=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly METADATA=${DESTINATION}.metadata.txt
readonly COMPLETION_MARKER=${DESTINATION}.complete

if (( $# != 4 )); then
  echo "Usage: $0 IMAGE METADATA COMPLETION_MARKER EXPECTED_TOOLING_SHA" >&2
  exit 2
fi

readonly image_path=$1
readonly metadata_path=$2
readonly marker_path=$3
readonly expected_tooling_sha=$4

[[ ${expected_tooling_sha} =~ ^[0-9a-f]{40}$ ]]
test "${image_path}" = "${DESTINATION}"
test "${metadata_path}" = "${METADATA}"
test "${marker_path}" = "${COMPLETION_MARKER}"

command -v sha256sum >/dev/null
test -f "${image_path}"
test ! -L "${image_path}"
test "$(sha256sum "${image_path}" | awk '{print $1}')" = "${EXPECTED_SHA256}"

test -f "${metadata_path}"
test ! -L "${metadata_path}"
test "$(wc -l <"${metadata_path}" | tr -d ' ')" -eq 10
grep -Fx -- "source=${SOURCE}" "${metadata_path}" >/dev/null
grep -Fx -- 'source_cluster=ptyche' "${metadata_path}" >/dev/null
grep -Fx -- 'source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh' "${metadata_path}" >/dev/null
grep -Fx -- 'source_smoke_job=2721177' "${metadata_path}" >/dev/null
grep -Fx -- 'destination_cluster=lyris' "${metadata_path}" >/dev/null
grep -Ex -- 'download_job=[0-9]+' "${metadata_path}" >/dev/null
grep -Ex -- 'downloaded_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${metadata_path}" >/dev/null
grep -Fx -- "sha256=${EXPECTED_SHA256}" "${metadata_path}" >/dev/null
grep -Fx -- "tooling_commit=${expected_tooling_sha}" "${metadata_path}" >/dev/null
grep -Fx -- "rclone_sha256=${EXPECTED_RCLONE_SHA256}" "${metadata_path}" >/dev/null

test -f "${marker_path}"
test ! -L "${marker_path}"
test "$(wc -l <"${marker_path}" | tr -d ' ')" -eq 6
grep -Fx -- "destination=${DESTINATION}" "${marker_path}" >/dev/null
grep -Fx -- "source=${SOURCE}" "${marker_path}" >/dev/null
grep -Fx -- "sha256=${EXPECTED_SHA256}" "${marker_path}" >/dev/null
grep -Fx -- "tooling_commit=${expected_tooling_sha}" "${marker_path}" >/dev/null
grep -Fx -- "rclone_sha256=${EXPECTED_RCLONE_SHA256}" "${marker_path}" >/dev/null
grep -Ex -- 'completed_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z' "${marker_path}" >/dev/null

printf 'validated_image=%s\n' "${image_path}"
printf 'validated_sha256=%s\n' "${EXPECTED_SHA256}"
printf 'validated_tooling_commit=%s\n' "${expected_tooling_sha}"
