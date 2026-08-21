#!/bin/bash

set -Eeuo pipefail

: "${BASE_REMOTE_REPO:?Set exact clean base /home checkout}"
: "${OPTIMIZED_REMOTE_REPO:?Set exact clean optimized /home checkout}"
: "${BASE_EXPECTED_HEAD:?Set base harness SHA}"
: "${OPTIMIZED_EXPECTED_HEAD:?Set optimized harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre pair result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly base_product_head=79e80af96a13522e6049658663a8c40ab21e8314
readonly optimized_product_head=f909e3d124bb663db4099e88f6846e55b0500912
readonly base_repo="$(realpath "${BASE_REMOTE_REPO}")"
readonly optimized_repo="$(realpath "${OPTIMIZED_REMOTE_REPO}")"
readonly experiment_path=research/qwen3_8b_dflash_nonnsys_ab
readonly source_parity="${base_repo}/${experiment_path}/source_parity.py"
readonly monitor_script="${base_repo}/${experiment_path}/monitor_pair.sh"
readonly wandb_project=sna-nemo-rl-online-drafter
readonly parity_proof="${FINAL_ROOT}/source-parity.json"

[[ "${base_repo}" == /home/* ]]
[[ "${optimized_repo}" == /home/* ]]
[[ "${FINAL_ROOT}" == /lustre/* ]]
test "${base_repo}" != "${optimized_repo}"
test ! -e "${FINAL_ROOT}"

python3 "${source_parity}" check \
  --base-checkout "${base_repo}" \
  --optimized-checkout "${optimized_repo}" \
  --base-harness-head "${BASE_EXPECTED_HEAD}" \
  --optimized-harness-head "${OPTIMIZED_EXPECTED_HEAD}" \
  --proof "${parity_proof}"

base_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly base_wandb_id
optimized_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly optimized_wandb_id
test "${base_wandb_id}" != "${optimized_wandb_id}"

submit() {
  local source_arm=$1
  local wandb_id=$2
  local test_only=$3
  local remote_repo=$4
  local product_head=$5
  local expected_head=$6
  local runner_path="${remote_repo}/${experiment_path}/run_oci_hsg.sbatch"
  local final_dir="${FINAL_ROOT}/${source_arm}"
  local exports="ALL,REMOTE_REPO=${remote_repo},EXPECTED_HEAD=${expected_head},PRODUCT_HEAD=${product_head},SOURCE_ARM=${source_arm},PARITY_PROOF=${parity_proof},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},ARM=online,WANDB_RUN_ID=${wandb_id},WANDB_PROJECT=${wandb_project}"
  local options=(
    --account="${SBATCH_ACCOUNT}"
    --time=02:00:00
    --output="/raid/scratch/dflash-source-ab-%j.out"
    --job-name="q8-dflash-src-${source_arm}"
    --export="${exports}"
  )
  if [[ "${test_only}" == 1 ]]; then
    sbatch --test-only "${options[@]}" "${runner_path}" >&2
  else
    sbatch --parsable "${options[@]}" "${runner_path}" | cut -d';' -f1
  fi
}

submit "base" "${base_wandb_id}" 1 "${base_repo}" \
  "${base_product_head}" "${BASE_EXPECTED_HEAD}"
submit "optimized" "${optimized_wandb_id}" 1 "${optimized_repo}" \
  "${optimized_product_head}" "${OPTIMIZED_EXPECTED_HEAD}"
base_job="$(submit "base" "${base_wandb_id}" 0 "${base_repo}" \
  "${base_product_head}" "${BASE_EXPECTED_HEAD}")"
if ! optimized_job="$(submit "optimized" "${optimized_wandb_id}" 0 \
  "${optimized_repo}" "${optimized_product_head}" \
  "${OPTIMIZED_EXPECTED_HEAD}")"; then
  scancel "${base_job}" || true
  echo "optimized submission failed; cancelled base job ${base_job}" >&2
  exit 1
fi

echo "base_job=${base_job}"
echo "optimized_job=${optimized_job}"
echo "base_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${base_wandb_id}"
echo "optimized_wandb=https://wandb.ai/nvidia/${wandb_project}/runs/${optimized_wandb_id}"
"${monitor_script}" "${base_job}" "${optimized_job}"
