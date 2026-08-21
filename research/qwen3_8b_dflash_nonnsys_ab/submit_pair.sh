#!/bin/bash

set -Eeuo pipefail

: "${FIXED_REMOTE_REPO:?Set exact clean fixed /home checkout}"
: "${ONLINE_REMOTE_REPO:?Set exact clean online /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre pair result root}"
: "${PARITY_PROOF:?Set immutable-container resolved parity proof}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly expected_container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly fixed_experiment="${FIXED_REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_ab"
readonly experiment="${fixed_experiment}"
readonly parity_authority="${PARITY_AUTHORITY:-${fixed_experiment}/resolved_parity.py}"
readonly monitor_script="${MONITOR_SCRIPT:-${experiment}/monitor_pair.sh}"
readonly wandb_project=sna-nemo-rl-online-drafter
[[ "${FIXED_REMOTE_REPO}" == /home/* || -n "${PARITY_AUTHORITY:-}" ]]
[[ "${ONLINE_REMOTE_REPO}" == /home/* || -n "${PARITY_AUTHORITY:-}" ]]
test "${FIXED_REMOTE_REPO}" != "${ONLINE_REMOTE_REPO}"
python3 "${parity_authority}" validate-proof \
  --proof "${PARITY_PROOF}" \
  --expected-head "${EXPECTED_HEAD}" \
  --target-snapshot "${TARGET_SNAPSHOT}" \
  --drafter-snapshot "${DRAFTER_SNAPSHOT}" \
  --container-sha256 "${expected_container_sha}" \
  --wandb-project "${wandb_project}"
fixed_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly fixed_wandb_id
online_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly online_wandb_id
test "${fixed_wandb_id}" != "${online_wandb_id}"

submit() {
  local arm=$1
  local wandb_id=$2
  local test_only=$3
  local remote_repo=$4
  local runner_path="${remote_repo}/research/qwen3_8b_dflash_nonnsys_ab/run_oci_hsg.sbatch"
  local final_dir="${FINAL_ROOT}/${arm}"
  local exports="ALL,REMOTE_REPO=${remote_repo},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},ARM=${arm},WANDB_RUN_ID=${wandb_id},WANDB_PROJECT=${wandb_project}"
  local options=(
    --account="${SBATCH_ACCOUNT}"
    --time=02:00:00
    --output="/raid/scratch/dflash-ab-%j.out"
    --job-name="q8-dflash-ab-${arm}"
    --export="${exports}"
  )
  if [[ "${test_only}" == 1 ]]; then
    sbatch --test-only "${options[@]}" "${runner_path}" >&2
  else
    sbatch --parsable "${options[@]}" "${runner_path}" | cut -d';' -f1
  fi
}

submit fixed "${fixed_wandb_id}" 1 "${FIXED_REMOTE_REPO}"
submit online "${online_wandb_id}" 1 "${ONLINE_REMOTE_REPO}"
fixed_job="$(submit fixed "${fixed_wandb_id}" 0 "${FIXED_REMOTE_REPO}")"
if ! online_job="$(submit online "${online_wandb_id}" 0 "${ONLINE_REMOTE_REPO}")"; then
  if scancel "${fixed_job}"; then
    echo "online submission failed; cancelled fixed job ${fixed_job}" >&2
  else
    echo "online submission failed; could not cancel fixed job ${fixed_job}" >&2
  fi
  exit 1
fi

echo "fixed_job=${fixed_job}"
echo "online_job=${online_job}"
echo "fixed_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${fixed_wandb_id}"
echo "online_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${online_wandb_id}"
echo "monitor=${monitor_script} ${fixed_job} ${online_job}"
"${monitor_script}" "${fixed_job}" "${online_job}"
