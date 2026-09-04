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

readonly ARCHIVE_URL=https://downloads.rclone.org/v1.75.0/rclone-v1.75.0-linux-arm64.zip
readonly EXPECTED_ARCHIVE_SHA256=d0ad88ba4c8e285b7c9efa591e0ab643280a91741e13c27f3a9c0957ccfa5203
readonly EXPECTED_SHA256=a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a
readonly DESTINATION=/home/sna/.local/libexec/nemo-rl/rclone/sha256-a7094d6e48c6c26cb069175ae93ee221db7dabfa18f57cb6bf3d3d5e1fb1cf3a/rclone
readonly DESTINATION_DIRECTORY=${DESTINATION%/*}
readonly ARCHIVE_NAME=rclone-v1.75.0-linux-arm64.zip
readonly EXTRACTED_BINARY_RELATIVE_PATH=rclone-v1.75.0-linux-arm64/rclone

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
  local binary_description

  require_regular_executable "${path}" "${action}"
  actual_sha256=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    printf 'Ptyche rclone %s failed: SHA256 did not match the expected content address\n' \
      "${action}" >&2
    return 1
  fi
  binary_description=$(LC_ALL=C file -b "${path}")
  if [[ ${binary_description} != *'ELF 64-bit LSB executable, ARM aarch64'* ]]; then
    printf 'Ptyche rclone %s failed: binary is not Linux ARM64\n' \
      "${action}" >&2
    return 1
  fi
}

cleanup_stage() {
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
  if [[ -n ${download_directory:-} && -d ${download_directory} ]]; then
    if rm -r -- "${download_directory}"; then
      :
    else
      cleanup_exit_status=$?
      echo 'Ptyche rclone stage cleanup failed: could not remove the exact download directory' >&2
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
require_command file "${ACTION}"

if [[ ${ACTION} == check ]]; then
  verify_binary "${DESTINATION}" check
  printf 'Ptyche rclone verified: %s\n' "${DESTINATION}"
  exit 0
fi

for command_name in cp chmod curl ln mkdir mktemp rm unzip; do
  require_command "${command_name}" stage
done
if [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
  printf 'Ptyche rclone stage failed: destination already exists; refusing to overwrite: %s\n' \
    "${DESTINATION}" >&2
  exit 1
fi

umask 077
download_directory=$(mktemp -d /tmp/nemo-rl-rclone-arm64.XXXXXXXX)
trap cleanup_stage EXIT
archive_path=${download_directory}/${ARCHIVE_NAME}
extraction_directory=${download_directory}/extracted
mkdir -p -- "${extraction_directory}"
curl \
  --proto '=https' \
  --tlsv1.2 \
  --connect-timeout 20 \
  --max-time 300 \
  --retry 2 \
  --retry-connrefused \
  --fail \
  --location \
  --silent \
  --show-error \
  --output "${archive_path}" \
  "${ARCHIVE_URL}"
archive_sha256=$(sha256sum "${archive_path}" | awk '{print $1}')
if [[ ${archive_sha256} != "${EXPECTED_ARCHIVE_SHA256}" ]]; then
  echo 'Ptyche rclone stage failed: archive SHA256 did not match the pinned release' >&2
  exit 1
fi
unzip -q "${archive_path}" -d "${extraction_directory}"
source_binary=${extraction_directory}/${EXTRACTED_BINARY_RELATIVE_PATH}
verify_binary "${source_binary}" stage

mkdir -p -- "${DESTINATION_DIRECTORY}"
if [[ -e ${DESTINATION} || -L ${DESTINATION} ]]; then
  printf 'Ptyche rclone stage failed: destination already exists; refusing to overwrite: %s\n' \
    "${DESTINATION}" >&2
  exit 1
fi

temporary_path=$(mktemp "${DESTINATION_DIRECTORY}/.rclone.stage.XXXXXXXX")
cp -- "${source_binary}" "${temporary_path}"
chmod 500 "${temporary_path}"
verify_binary "${temporary_path}" stage
if ! ln -T -- "${temporary_path}" "${DESTINATION}"; then
  echo 'Ptyche rclone stage failed: could not publish the binary without overwriting a path' >&2
  exit 1
fi
rm -f -- "${temporary_path}"
temporary_path=
verify_binary "${DESTINATION}" stage
rm -r -- "${download_directory}"
download_directory=
trap - EXIT
printf 'Ptyche rclone staged and verified: %s\n' "${DESTINATION}"
