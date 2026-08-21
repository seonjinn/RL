#!/bin/bash

set -Eeuo pipefail

mode=${1:-}
if [[ "${mode}" != "--test-only" && -n "${mode}" ]]; then
  echo "usage: $0 [--test-only]" >&2
  exit 2
fi

: "${REMOTE_REPO:?Set exact clean /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre matrix result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_refit_perf_matrix"
readonly runner="${experiment}/run_pair_oci_hsg.sbatch"
readonly wandb_project=sna-nemo-rl-online-drafter

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_ROOT}" == /lustre/* ]]

run_id() {
  local shape=$1
  local replicate=$2
  local arm=$3
  printf 'q8-%s-r%s-%s-' "${shape}" "${replicate}" "${arm}"
  python3 -c 'import secrets; print(secrets.token_hex(4))'
}

submit_pair() {
  local shape=$1
  local replicate=$2
  local first_arm=$3
  local fixed_id=$4
  local online_id=$5
  local exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_ROOT}/${shape}/replicate-${replicate},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},PAIR_SHAPE=${shape},REPLICATE=${replicate},FIRST_ARM=${first_arm},FIXED_WANDB_RUN_ID=${fixed_id},ONLINE_WANDB_RUN_ID=${online_id},WANDB_PROJECT=${wandb_project}"
  local options=(
    --account="${SBATCH_ACCOUNT}"
    --output="/raid/scratch/dflash-refit-matrix-%j.out"
    --job-name="q8-refit-${shape}-r${replicate}"
    --export="${exports}"
  )
  if [[ "${mode}" == "--test-only" ]]; then
    sbatch --test-only "${options[@]}" "${runner}" >&2
  else
    sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
  fi
}

jobs=()
for shape in gbs32_mbs1 gbs64_mbs1 gbs64_mbs2; do
  for replicate in 1 2 3; do
    if [[ "${replicate}" == 2 ]]; then
      first_arm=online
    else
      first_arm=fixed
    fi
    fixed_id=$(run_id "${shape}" "${replicate}" fixed)
    online_id=$(run_id "${shape}" "${replicate}" online)
    test "${fixed_id}" != "${online_id}"
    if [[ "${mode}" == "--test-only" ]]; then
      submit_pair "${shape}" "${replicate}" "${first_arm}" "${fixed_id}" "${online_id}"
    else
      jobs+=("$(submit_pair "${shape}" "${replicate}" "${first_arm}" "${fixed_id}" "${online_id}")")
      echo "${shape}_r${replicate}_fixed_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${fixed_id}"
      echo "${shape}_r${replicate}_online_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${online_id}"
    fi
  done
done

if [[ "${mode}" == "--test-only" ]]; then
  echo "submission_mode=test-only jobs_submitted=0"
else
  printf 'submitted_pair_jobs=%s\n' "${jobs[*]}"
  "${experiment}/monitor_matrix.sh" "${jobs[@]}"
fi
