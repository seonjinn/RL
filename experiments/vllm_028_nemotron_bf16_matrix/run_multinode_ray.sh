#!/usr/bin/env bash
set -euo pipefail

: "${HEAD_IP:?HEAD_IP must identify the Ray head node}"
: "${RAY_PORT:?RAY_PORT must be set}"
: "${RAY_SYNC_DIR:?RAY_SYNC_DIR must be set}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${USER:?USER must be set}"

readonly RAY_TEMP_ROOT="${RAY_TEMP_ROOT:-/raid/scratch/${USER}/vllm028_${SLURM_JOB_ID}/ray}"
cleanup_ray_temp() {
  if [[ "${RAY_TEMP_ROOT}" == "/raid/scratch/${USER}/vllm028_${SLURM_JOB_ID}/ray" ]]; then
    rm -rf -- "${RAY_TEMP_ROOT}"
  fi
}
trap cleanup_ray_temp EXIT

rank=${SLURM_PROCID:?SLURM_PROCID must be set}
world_size=${SLURM_NTASKS:?SLURM_NTASKS must be set}
node_ip=$(hostname --ip-address | awk '{print $1}')
ray_cli=(python3 -m ray.scripts.scripts)

mkdir -p "${RAY_SYNC_DIR}"
mkdir -p "${RAY_TEMP_ROOT}"
if [[ "${rank}" -eq 0 ]]; then
  "${ray_cli[@]}" start \
    --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --temp-dir="${RAY_TEMP_ROOT}" \
    --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
    --disable-usage-stats

  deadline=$((SECONDS + 300))
  while [[ $(find "${RAY_SYNC_DIR}" -maxdepth 1 -type f -name 'worker-*' | wc -l) -lt $((world_size - 1)) ]]; do
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      echo "Timed out waiting for Ray workers" >&2
      touch "${RAY_SYNC_DIR}/done"
      exit 1
    fi
    sleep 2
  done

  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  status=0
  "$@" || status=$?
  touch "${RAY_SYNC_DIR}/done"
  "${ray_cli[@]}" stop --force || true
  exit "${status}"
fi

"${ray_cli[@]}" start \
  --address="${HEAD_IP}:${RAY_PORT}" \
  --node-ip-address="${node_ip}" \
  --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
  --disable-usage-stats
touch "${RAY_SYNC_DIR}/worker-${rank}"

deadline=$((SECONDS + 14400))
while [[ ! -e "${RAY_SYNC_DIR}/done" ]]; do
  if [[ "${SECONDS}" -ge "${deadline}" ]]; then
    echo "Timed out waiting for benchmark completion" >&2
    "${ray_cli[@]}" stop --force || true
    exit 1
  fi
  sleep 5
done
"${ray_cli[@]}" stop --force || true
