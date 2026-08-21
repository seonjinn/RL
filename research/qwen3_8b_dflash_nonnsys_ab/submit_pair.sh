#!/bin/bash

set -Eeuo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre pair result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_ab"
readonly runner="${experiment}/run_oci_hsg.sbatch"
readonly wandb_project=sna-nemo-rl-online-drafter
fixed_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly fixed_wandb_id
online_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly online_wandb_id
test "${fixed_wandb_id}" != "${online_wandb_id}"

submit() {
  local arm=$1
  local wandb_id=$2
  local final_dir="${FINAL_ROOT}/${arm}"
  local exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},ARM=${arm},WANDB_RUN_ID=${wandb_id},WANDB_PROJECT=${wandb_project}"
  local options=(
    --account="${SBATCH_ACCOUNT}"
    --time=02:00:00
    --output="/raid/scratch/dflash-ab-%j.out"
    --job-name="q8-dflash-ab-${arm}"
    --export="${exports}"
  )
  sbatch --test-only "${options[@]}" "${runner}" >&2
  sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
}

fixed_job="$(submit fixed "${fixed_wandb_id}")"
online_job="$(submit online "${online_wandb_id}")"

echo "fixed_job=${fixed_job}"
echo "online_job=${online_job}"
echo "fixed_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${fixed_wandb_id}"
echo "online_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${online_wandb_id}"
