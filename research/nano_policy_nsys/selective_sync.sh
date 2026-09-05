#!/usr/bin/env bash

set -euo pipefail

: "${NSYS_RESULT_DIR:?Set NSYS_RESULT_DIR to a durable result directory}"

shopt -s nullglob
ray_sessions=(/tmp/ray/session_[0-9]*)
ray_root=/tmp/ray
if (( ${#ray_sessions[@]} == 0 )); then
  raylet_pid=$(pgrep -o -f '/raylet .*--temp-dir=/tmp/ray' || true)
  [[ -n "${raylet_pid}" ]] || exit 1
  ray_root=/proc/${raylet_pid}/root/tmp/ray
fi

output_dir=${NSYS_RESULT_DIR}/$(hostname)
mkdir -p "${output_dir}"

while true; do
  for report in "${ray_root}"/session_[0-9]*/logs/*megatron_policy_worker*.nsys-rep; do
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
