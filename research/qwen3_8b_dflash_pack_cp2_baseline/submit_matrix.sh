#!/bin/bash

set -Eeuo pipefail

mode=${1:-}
if [[ "${mode}" != "--test-only" && -n "${mode}" ]]; then
  echo "usage: $0 [--test-only]" >&2
  exit 2
fi
: "${REMOTE_REPO:?Set exact clean /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set DFlash snapshot}"
: "${SBATCH_ACCOUNT:?Set refreshed FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_pack_cp2_baseline"
readonly runner="${experiment}/run_pair_oci_hsg.sbatch"
readonly wandb_project=sna-nemo-rl-dflash-pack-cp2-baseline
[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_ROOT}" == /lustre/* ]]
test "${SBATCH_ACCOUNT}" = nemotron_n3_post

run_id() {
  printf 'q8-pack-cp2-r%s-%s-' "$1" "$2"
  python3 -c 'import secrets; print(secrets.token_hex(5))'
}

jobs=()
all_ids=()
for replicate in 1 2 3; do
  if [[ "${replicate}" == 2 ]]; then first_arm=online; else first_arm=fixed; fi
  fixed_id=$(run_id "${replicate}" fixed)
  online_id=$(run_id "${replicate}" online)
  all_ids+=("${fixed_id}" "${online_id}")
  exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_ROOT}/replicate-${replicate},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},REPLICATE=${replicate},FIRST_ARM=${first_arm},FIXED_WANDB_RUN_ID=${fixed_id},ONLINE_WANDB_RUN_ID=${online_id},WANDB_PROJECT=${wandb_project}"
  options=(--account="${SBATCH_ACCOUNT}" --partition=batch --qos=normal \
    --time=04:00:00 \
    --nodes=1 --exclusive --gres=gpu:4 \
    --output="/raid/scratch/dflash-pack-cp2-%j.out" \
    --job-name="q8-pack-cp2-r${replicate}" --export="${exports}")
  if [[ "${mode}" == --test-only ]]; then
    sbatch --test-only "${options[@]}" "${runner}" >&2
  else
    jobs+=("$(sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1)")
    echo "replicate_${replicate}_fixed_wandb_id=${fixed_id}"
    echo "replicate_${replicate}_online_wandb_id=${online_id}"
  fi
done

test "$(printf '%s\n' "${all_ids[@]}" | sort -u | wc -l | tr -d ' ')" -eq 6
if [[ "${mode}" == --test-only ]]; then
  echo "submission_mode=test-only jobs_submitted=0"
else
  test "${#jobs[@]}" -eq 3
  test "$(printf '%s\n' "${jobs[@]}" | sort -u | wc -l | tr -d ' ')" -eq 3
  printf 'submitted_pair_jobs=%s\n' "${jobs[*]}"
  "${experiment}/monitor_matrix.sh" "${jobs[@]}"
fi
