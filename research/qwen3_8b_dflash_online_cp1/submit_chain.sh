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

readonly mode="${1:-gate}"
readonly gate_dependency="${2:-}"
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_online_cp1"
readonly gate="${experiment}/run_gate_oci_hsg.sbatch"
readonly resume="${experiment}/run_resume_oci_hsg.sbatch"
if [[ "${mode}" == gate ]]; then
  wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
elif [[ "${mode}" == continue && -n "${gate_dependency}" ]]; then
  wandb_id="$(python3 "${experiment}/resume_contract.py" \
    --checkpoint-dir "${FINAL_DIR}/checkpoints" \
    --manifest "${FINAL_DIR}/gate-manifest.json" \
    --git-sha "${EXPECTED_HEAD}" \
    --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
    --drafter-revision 9b41424b7109f9c5413454f481b09a82b85333f4 \
    --container-sha256 6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44 \
    --print-manifest-wandb-id)"
else
  echo "usage: $0 gate | $0 continue GATE_JOB_ID" >&2
  exit 2
fi
readonly wandb_id
readonly exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_DIR},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},WANDB_RUN_ID=${wandb_id}"

submit() {
  local stage=$1
  local dependency=$2
  local runner=$3
  local walltime=$4
  local milestone=$5
  local deadline=$6
  local options=(--account="${SBATCH_ACCOUNT}" --time="${walltime}" \
    --output="/raid/scratch/nrl-online-%j.out" --job-name="q8-online-k7-${stage}" \
    --export="${exports},STAGE_MIN_STEP=${milestone},STAGE_DEADLINE=${deadline}")
  if [[ -n "${dependency}" ]]; then options+=(--dependency="afterok:${dependency}"); fi
  sbatch --test-only "${options[@]}" "${runner}" >&2
  sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
}

if [[ "${mode}" == gate ]]; then
  gate_id="$(submit gate "" "${gate}" 01:00:00 2 00:00:10:00)"
  echo "CHAIN gate=${gate_id}"
else
  previous="${gate_dependency}"
  for spec in \
    "segment1 04:00:00 350 00:03:30:00" \
    "segment2 04:00:00 700 00:03:30:00" \
    "segment3 04:00:00 1000 00:03:30:00"; do
    read -r stage walltime milestone deadline <<<"${spec}"
    previous="$(submit "${stage}" "${previous}" "${resume}" \
      "${walltime}" "${milestone}" "${deadline}")"
    echo "CHAIN ${stage}=${previous}"
  done
fi
echo "WANDB_URL=https://wandb.ai/nvidia/nemo-rl-specdec-eval/runs/${wandb_id}"
