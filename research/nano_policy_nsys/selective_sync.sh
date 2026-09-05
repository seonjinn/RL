#!/usr/bin/env bash

set -euo pipefail

: "${NSYS_RESULT_DIR:?Set NSYS_RESULT_DIR to a durable result directory}"

output_dir=${NSYS_RESULT_DIR}/$(hostname)
mkdir -p "${output_dir}"

while true; do
  for report in /tmp/ray/session_latest/logs/*megatron_policy_worker*.nsys-rep; do
    [[ -f "${report}" ]] || continue
    size_file=/tmp/.$(basename "${report}").last_size
    current_size=$(stat -c %s "${report}" 2>/dev/null || echo 0)
    previous_size=$(cat "${size_file}" 2>/dev/null || echo -1)
    printf '%s\n' "${current_size}" > "${size_file}"
    if [[ "${current_size}" -gt 0 && "${current_size}" == "${previous_size}" ]]; then
      cp -f "${report}" "${output_dir}/$(basename "${report}")"
    fi
  done
  sleep 15
done
