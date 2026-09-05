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

# shellcheck disable=SC2016  # Assertions below intentionally match literal shell source.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_DIRECTORY=${script_directory}
readonly WRAPPER=${SCRIPT_DIRECTORY}/submit_lyris_download_validated_nightly.sh
readonly BATCH=${SCRIPT_DIRECTORY}/lyris_download_validated_nightly.sbatch
readonly VALIDATOR=${SCRIPT_DIRECTORY}/validate_lyris_transferred_nightly.sh
readonly EXPECTED_SOURCE=pbss-team-nemo-ci-s3:nemo-ci/nemo-rl/sna/cross-cluster/validated-containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_DESTINATION=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904_c6edc455e0fac52d.sqsh
readonly EXPECTED_SHA256=c6edc455e0fac52db4212003f58dec15c8d267f11183f30ec2e1dcfc7d2fb20e
readonly EXPECTED_RCLONE=/home/sna/.local/libexec/nemo-rl/rclone/sha256-a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a/rclone
readonly EXPECTED_RCLONE_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a
readonly EXPECTED_LOG_DIRECTORY=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/container-transfer/nemo_rl_nightly_20260904_c6edc455/lyris-download

fail() {
  printf 'Lyris validated-nightly static test failed: %s\n' "$1" >&2
  exit 1
}

require_literal() {
  local needle=$1
  local file=$2

  grep -Fq -- "${needle}" "${file}" ||
    fail "missing literal in ${file##*/}: ${needle}"
}

forbid_literal() {
  local needle=$1
  local file=$2

  if grep -Fq -- "${needle}" "${file}"; then
    fail "forbidden literal in ${file##*/}: ${needle}"
  fi
}

for file in "${WRAPPER}" "${BATCH}" "${VALIDATOR}"; do
  test -f "${file}" || fail "missing owned script: ${file##*/}"
  test -x "${file}" || fail "owned script is not executable: ${file##*/}"
  bash -n "${file}"
  require_literal 'Copyright (c) 2026, NVIDIA CORPORATION.' "${file}"
  forbid_literal 'printenv' "${file}"
done

require_literal '#SBATCH --account=coreai_dlalgo_llm' "${BATCH}"
require_literal '#SBATCH --partition=gb200' "${BATCH}"
require_literal "readonly SOURCE=${EXPECTED_SOURCE}" "${BATCH}"
require_literal "readonly DESTINATION=${EXPECTED_DESTINATION}" "${BATCH}"
require_literal "readonly EXPECTED_SHA256=${EXPECTED_SHA256}" "${BATCH}"
require_literal "readonly RCLONE_SOURCE=${EXPECTED_RCLONE}" "${BATCH}"
require_literal "readonly EXPECTED_RCLONE_SHA256=${EXPECTED_RCLONE_SHA256}" "${BATCH}"
require_literal 'readonly EXPECTED_COMPUTE_ARCHITECTURE=aarch64' "${BATCH}"
require_literal 'readonly VALIDATOR_RELATIVE_PATH=experiments/pr3652_validation_container/scripts/validate_lyris_transferred_nightly.sh' "${BATCH}"
require_literal 'git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${VALIDATOR_RELATIVE_PATH}"' "${BATCH}"
require_literal 'test "${runtime_rclone_sha256}" = "${EXPECTED_RCLONE_SHA256}"' "${BATCH}"
require_literal 'ln -T "${PARTIAL}" "${DESTINATION}"' "${BATCH}"
require_literal 'trap cleanup EXIT' "${BATCH}"
require_literal 'rm -f -- "${PARTIAL}" "${PARTIAL_METADATA}" "${PARTIAL_COMPLETION_MARKER}"' "${BATCH}"
require_literal 'rm -rf -- "${JOB_SCRATCH_DIRECTORY}"' "${BATCH}"
require_literal 'tooling_commit=${EXPECTED_TOOLING_SHA}' "${BATCH}"
forbid_literal '/lustre/fs1/' "${BATCH}"
forbid_literal '--export=ALL' "${WRAPPER}"

require_literal "readonly LOG_DIRECTORY=${EXPECTED_LOG_DIRECTORY}" "${WRAPPER}"
require_literal 'test "${EXPECTED_TOOLING_SHA}" = "${TOOLING_UPSTREAM_SHA}"' "${WRAPPER}"
require_literal 'git -C "${SCRIPT_ROOT}" show "${EXPECTED_TOOLING_SHA}:${BATCH_RELATIVE_PATH}"' "${WRAPPER}"
require_literal '/usr/bin/env -i' "${WRAPPER}"
require_literal '--test-only' "${WRAPPER}"
require_literal '--export="${EXPORTS}"' "${WRAPPER}"

require_literal "readonly SOURCE=${EXPECTED_SOURCE}" "${VALIDATOR}"
require_literal "readonly DESTINATION=${EXPECTED_DESTINATION}" "${VALIDATOR}"
require_literal "readonly EXPECTED_SHA256=${EXPECTED_SHA256}" "${VALIDATOR}"
require_literal "readonly EXPECTED_RCLONE_SHA256=${EXPECTED_RCLONE_SHA256}" "${VALIDATOR}"
require_literal 'test ! -L "${image_path}"' "${VALIDATOR}"
require_literal 'grep -Fx -- "tooling_commit=${expected_tooling_sha}" "${metadata_path}"' "${VALIDATOR}"
require_literal 'grep -Fx -- "rclone_sha256=${EXPECTED_RCLONE_SHA256}" "${metadata_path}"' "${VALIDATOR}"

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/lyris-validated-nightly-test.XXXXXXXX")
cleanup() {
  rm -rf -- "${temporary_directory}"
}
trap cleanup EXIT

readonly TEST_IMAGE=${temporary_directory}/nightly.sqsh
readonly TEST_METADATA=${TEST_IMAGE}.metadata.txt
readonly TEST_MARKER=${TEST_IMAGE}.complete
readonly TEST_TOOLING_SHA=1111111111111111111111111111111111111111
readonly TEST_VALIDATOR=${temporary_directory}/validate.sh
readonly STUB_DIRECTORY=${temporary_directory}/bin
mkdir -p "${STUB_DIRECTORY}"

sed "s#^readonly DESTINATION=.*#readonly DESTINATION=${TEST_IMAGE}#" \
  "${VALIDATOR}" >"${TEST_VALIDATOR}"
chmod 700 "${TEST_VALIDATOR}"
printf 'fixture\n' >"${TEST_IMAGE}"
cat >"${TEST_METADATA}" <<EOF
source=${EXPECTED_SOURCE}
source_cluster=ptyche
source_path=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260904.sqsh
source_smoke_job=2721177
destination_cluster=lyris
download_job=12345
downloaded_at_utc=2026-09-05T01:02:03Z
sha256=${EXPECTED_SHA256}
tooling_commit=${TEST_TOOLING_SHA}
rclone_sha256=${EXPECTED_RCLONE_SHA256}
EOF
cat >"${TEST_MARKER}" <<EOF
destination=${TEST_IMAGE}
source=${EXPECTED_SOURCE}
sha256=${EXPECTED_SHA256}
tooling_commit=${TEST_TOOLING_SHA}
rclone_sha256=${EXPECTED_RCLONE_SHA256}
completed_at_utc=2026-09-05T01:02:04Z
EOF
cat >"${STUB_DIRECTORY}/sha256sum" <<EOF
#!/bin/bash
printf '%s  %s\n' "\${TEST_ACTUAL_SHA256:-${EXPECTED_SHA256}}" "\$1"
EOF
chmod 700 "${STUB_DIRECTORY}/sha256sum"

PATH=${STUB_DIRECTORY}:/usr/bin:/bin \
  "${TEST_VALIDATOR}" \
  "${TEST_IMAGE}" \
  "${TEST_METADATA}" \
  "${TEST_MARKER}" \
  "${TEST_TOOLING_SHA}" >/dev/null

if TEST_ACTUAL_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  PATH=${STUB_DIRECTORY}:/usr/bin:/bin \
  "${TEST_VALIDATOR}" \
  "${TEST_IMAGE}" \
  "${TEST_METADATA}" \
  "${TEST_MARKER}" \
  "${TEST_TOOLING_SHA}" >/dev/null 2>&1; then
  fail 'validator accepted the wrong image digest'
fi

mv "${TEST_IMAGE}" "${TEST_IMAGE}.real"
ln -s "${TEST_IMAGE}.real" "${TEST_IMAGE}"
if PATH=${STUB_DIRECTORY}:/usr/bin:/bin \
  "${TEST_VALIDATOR}" \
  "${TEST_IMAGE}" \
  "${TEST_METADATA}" \
  "${TEST_MARKER}" \
  "${TEST_TOOLING_SHA}" >/dev/null 2>&1; then
  fail 'validator accepted a symlinked image'
fi

printf 'Lyris validated-nightly static tests passed\n'
