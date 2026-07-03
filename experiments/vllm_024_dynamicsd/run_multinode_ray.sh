#!/usr/bin/env bash
set -euo pipefail

: "${HEAD_IP:?HEAD_IP is required}"
: "${RAY_PORT:?RAY_PORT is required}"
: "${RAY_SYNC_DIR:?RAY_SYNC_DIR is required}"

rank="${SLURM_PROCID:-0}"
world_size="${SLURM_NTASKS:-1}"
gpus_per_node="${GPUS_PER_NODE:-4}"
done_file="${RAY_SYNC_DIR}/done"

mkdir -p "${RAY_SYNC_DIR}"

if [[ "${rank}" == "0" ]]; then
  cleanup() {
    touch "${done_file}"
    ray stop --force >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${gpus_per_node}" \
    --disable-usage-stats

  deadline=$((SECONDS + 180))
  while (( world_size > 1 )); do
    ready_count="$(find "${RAY_SYNC_DIR}" -maxdepth 1 -name 'worker-*' | wc -l | tr -d ' ')"
    if (( ready_count >= world_size - 1 )); then
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for Ray workers: ${ready_count}/$((world_size - 1))" >&2
      exit 4
    fi
    sleep 2
  done

  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  "$@"
else
  cleanup() {
    ray stop --force >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  ray start \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --num-gpus="${gpus_per_node}" \
    --disable-usage-stats
  touch "${RAY_SYNC_DIR}/worker-${rank}"
  while [[ ! -f "${done_file}" ]]; do
    sleep 2
  done
fi
