#!/bin/bash

set -Eeuo pipefail

mode=${1:-}
if [[ "${mode}" != "--test-only" && -n "${mode}" ]]; then
  echo "usage: $0 [--test-only]" >&2
  exit 2
fi

: "${REMOTE_REPO:?Set exact clean /home checkout}"
: "${EXPECTED_HEAD:?Set exact harness commit}"
: "${FINAL_ROOT:?Set fresh /lustre result root}"
: "${CONTAINER:?Set immutable container path}"
: "${CONTAINER_SHA256:?Set immutable container SHA256}"
: "${TARGET_SNAPSHOT:?Set exact Qwen3-8B target snapshot}"
: "${DFLASH_SNAPSHOT:?Set exact DFlash snapshot}"
: "${DSPARK_SNAPSHOT:?Set exact DSpark snapshot}"
: "${SBATCH_ACCOUNT:?Set refreshed FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"
: "${WANDB_PROJECT:?Set W&B project}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_dspark_cp2_packed_smoke"
readonly runner="${experiment}/run_oci_hsg.sbatch"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_ROOT}" == /lustre/* ]]
test "${SBATCH_ACCOUNT}" = nemotron_n3_post

run_id() {
  printf 'q8-cp2-pack-%s-' "$1"
  python3 -c 'import secrets; print(secrets.token_hex(5))'
}

jobs=()
for arm in dflash dspark; do
  if [[ "${arm}" == dflash ]]; then
    drafter_snapshot=${DFLASH_SNAPSHOT}
  else
    drafter_snapshot=${DSPARK_SNAPSHOT}
  fi
  wandb_run_id=$(run_id "${arm}")
  exports="ALL,ARM=${arm},REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_ROOT}/${arm},CONTAINER=${CONTAINER},CONTAINER_SHA256=${CONTAINER_SHA256},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${drafter_snapshot},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_ID=${wandb_run_id}"
  options=(
    --account="${SBATCH_ACCOUNT}"
    --partition=batch
    --qos=normal
    --time=01:00:00
    --nodes=1
    --exclusive
    --gres=gpu:4
    --output="/raid/scratch/q8-cp2-pack-${arm}-%j.out"
    --job-name="q8-cp2-pack-${arm}"
    --export="${exports}"
  )
  if [[ "${mode}" == --test-only ]]; then
    sbatch --test-only "${options[@]}" "${runner}" >&2
  else
    job_id=$(sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1)
    jobs+=("${job_id}")
    printf '%s_job_id=%s\n%s_wandb_run_id=%s\n' \
      "${arm}" "${job_id}" "${arm}" "${wandb_run_id}"
  fi
done

if [[ "${mode}" == --test-only ]]; then
  echo "submission_mode=test-only jobs_submitted=0"
else
  test "${#jobs[@]}" -eq 2
  if [[ "${jobs[0]}" == "${jobs[1]}" ]]; then
    echo "duplicate job IDs returned by sbatch" >&2
    exit 1
  fi
  printf 'submitted_jobs=%s\n' "${jobs[*]}"
fi
