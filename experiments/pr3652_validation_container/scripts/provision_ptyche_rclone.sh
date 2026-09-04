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

readonly SOURCE=/usr/bin/rclone
readonly EXPECTED_SHA256=dc1ec3109000e4d36c8d14efac6d4c4158d1b860853cb75c09dcc9f6dded420b
readonly DESTINATION=/home/sna/.local/libexec/nemo-rl/rclone/sha256-dc1ec3109000e4d36c8d14efac6d4c4158d1b860853cb75c09dcc9f6dded420b/rclone
readonly DESTINATION_DIRECTORY=${DESTINATION%/*}

require_command() {
  local command_name=$1
  local action=$2

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Ptyche rclone %s failed: required command is unavailable: %s\n' \
      "${action}" \
      "${command_name}" >&2
    return 127
  fi
}

require_regular_executable() {
  local path=$1
  local action=$2

  if [[ ! -f "${path}" || -L "${path}" || ! -x "${path}" ]]; then
    printf 'Ptyche rclone %s failed: required regular executable is missing: %s\n' \
      "${action}" \
      "${path}" >&2
    return 1
  fi
}

verify_binary() {
  local path=$1
  local action=$2
  local actual_sha256
  local exit_status

  require_regular_executable "${path}" "${action}"
  actual_sha256=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    printf 'Ptyche rclone %s failed: SHA256 did not match the expected content address\n' \
      "${action}" >&2
    return 1
  fi
  if "${path}" version >/dev/null 2>&1; then
    return 0
  else
    exit_status=$?
  fi
  printf 'Ptyche rclone %s failed: binary is incompatible with this host\n' \
    "${action}" >&2
  return "${exit_status}"
}

cleanup_temporary_file() {
  local exit_status=$?
  local cleanup_exit_status

  trap - EXIT
  if [[ -n ${temporary_path:-} && ( -e ${temporary_path} || -L ${temporary_path} ) ]]; then
    if rm -f -- "${temporary_path}"; then
      :
    else
      cleanup_exit_status=$?
      echo 'Ptyche rclone stage cleanup failed: could not remove the exact temporary file' >&2
      if (( exit_status == 0 )); then
        exit_status=${cleanup_exit_status}
      fi
    fi
  fi
  exit "${exit_status}"
}

if (( $# > 1 )); then
  echo "Usage: $0 [check|stage]" >&2
  exit 2
fi
readonly ACTION=${1:-check}
if [[ ${ACTION} != check && ${ACTION} != stage ]]; then
  echo "Usage: $0 [check|stage]" >&2
  exit 2
fi

require_command sha256sum "${ACTION}"
require_command awk "${ACTION}"

if [[ ${ACTION} == check ]]; then
  verify_binary "${DESTINATION}" check
  printf 'Ptyche rclone verified: %s\n' "${DESTINATION}"
  exit 0
fi

for command_name in cp chmod ln mkdir mktemp rm; do
  require_command "${command_name}" stage
done
verify_binary "${SOURCE}" stage
if [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
  printf 'Ptyche rclone stage failed: destination already exists; refusing to overwrite: %s\n' \
    "${DESTINATION}" >&2
  exit 1
fi

umask 077
mkdir -p -- "${DESTINATION_DIRECTORY}"
if [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
  printf 'Ptyche rclone stage failed: destination already exists; refusing to overwrite: %s\n' \
    "${DESTINATION}" >&2
  exit 1
fi

temporary_path=$(mktemp "${DESTINATION_DIRECTORY}/.rclone.stage.XXXXXXXX")
trap cleanup_temporary_file EXIT
cp -- "${SOURCE}" "${temporary_path}"
chmod 500 "${temporary_path}"
verify_binary "${temporary_path}" stage
if ! ln -T -- "${temporary_path}" "${DESTINATION}"; then
  echo 'Ptyche rclone stage failed: could not publish the binary without overwriting a path' >&2
  exit 1
fi
rm -f -- "${temporary_path}"
temporary_path=
trap - EXIT
verify_binary "${DESTINATION}" stage
printf 'Ptyche rclone staged and verified: %s\n' "${DESTINATION}"
