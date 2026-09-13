#!/usr/bin/env bash
set -euo pipefail
: "${CONTAINER:?}"
: "${SOURCE_REPO:?}"
: "${SOURCE_SHA:?}"
: "${RESULT_DIR:?}"

for arm in default trim; do
  export PROBE_ARM="${arm}"
  srun --ntasks=4 --cpus-per-task=8 --kill-on-bad-exit=1 --wait=30 \
    --container-image="${CONTAINER}" \
    --container-mounts="/home/sna:/home/sna,/raid/scratch:/raid/scratch" \
    --output="${RESULT_DIR}/${arm}-%N-%t.log" \
    timeout --signal=TERM --kill-after=30s 8m bash -s <<'WORKER'
set -euo pipefail
root="/raid/scratch/sna/optimizer-retention/${SLURM_JOB_ID}/${PROBE_ARM}/${SLURM_LOCALID}"
mkdir -p "${root}"
git -C "${SOURCE_REPO}" archive "${SOURCE_SHA}" \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  experiments/precision_matrix_refresh_20260905/probe_optimizer_cpu_copy.py \
  experiments/precision_matrix_refresh_20260905/probe_optimizer_retention.py \
  | tar -xf - -C "${root}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
cd "${root}"
args=()
if [[ "${PROBE_ARM}" == trim ]]; then args+=(--trim); fi
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
  experiments/precision_matrix_refresh_20260905/probe_optimizer_retention.py "${args[@]}"
WORKER
done
