#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed composition SHA}"
: "${FINAL_DIR:?Set /lustre result directory}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_fixed_dense_control"
readonly runner="${experiment}/run_oci_hsg.sbatch"
readonly wandb_project=sna-nemo-rl-online-drafter
readonly wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_DIR},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},WANDB_RUN_ID=${wandb_id},WANDB_PROJECT=${wandb_project},SOURCE_GATE_SHA=${EXPECTED_HEAD}"

submit() {
  local stage=$1
  local dependency=$2
  local walltime=$3
  local milestone=$4
  local deadline=$5
  local is_gate=$6
  local resume=$7
  local options=(--account="${SBATCH_ACCOUNT}" --time="${walltime}" \
    --output="/raid/scratch/fixed-control-%j.out" \
    --job-name="q8-fixed-k7-${stage}" \
    --export="${exports},STAGE_MIN_STEP=${milestone},STAGE_DEADLINE=${deadline},IS_GATE=${is_gate},WANDB_RESUME=${resume}")
  if [[ -n "${dependency}" ]]; then options+=(--dependency="afterok:${dependency}"); fi
  sbatch --test-only "${options[@]}" "${runner}" >&2
  sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
}

gate_id="$(submit gate "" 01:00:00 2 00:00:50:00 1 allow)"
echo "CHAIN gate=${gate_id}"
previous="${gate_id}"
for spec in \
  "segment1 04:00:00 350 00:03:30:00" \
  "segment2 04:00:00 700 00:03:30:00" \
  "segment3 04:00:00 1000 00:03:30:00"; do
  read -r stage walltime milestone deadline <<<"${spec}"
  previous="$(submit "${stage}" "${previous}" "${walltime}" \
    "${milestone}" "${deadline}" 0 must)"
  echo "CHAIN ${stage}=${previous}"
done
echo "WANDB_URL=https://wandb.ai/nvidia/${wandb_project}/runs/${wandb_id}"
