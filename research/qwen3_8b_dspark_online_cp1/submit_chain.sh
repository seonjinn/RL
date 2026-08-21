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

readonly mode="${1:-smoke}"
readonly arm="${2:?Set dspark-k5 or dspark-k7}"
case "${arm}" in
  dspark-k5) num_speculative_tokens=5 ;;
  dspark-k7) num_speculative_tokens=7 ;;
  *) echo "arm must be dspark-k5 or dspark-k7" >&2; exit 2 ;;
esac
readonly num_speculative_tokens
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dspark_online_cp1"
readonly runner="${experiment}/run_segment_oci_hsg.sbatch"
readonly smoke_proof="${FINAL_DIR}/smoke-proof.json"
readonly common="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${FINAL_DIR},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},ARM_NAME=${arm},NUM_SPECULATIVE_TOKENS=${num_speculative_tokens}"

[[ "${FINAL_DIR}" == */"${arm}" ]]

submit() {
  local stage=$1 previous=$2 walltime=$3 milestone=$4 deadline=$5 resume=$6 run_id=$7
  local options=(--account="${SBATCH_ACCOUNT}" --time="${walltime}"
    --output="/raid/scratch/nrl-dspark-online-%j.out"
    --job-name="q8-${arm}-${stage}"
    --export="${common},WANDB_RUN_ID=${run_id},WANDB_RESUME=${resume},STAGE_MODE=${stage},STAGE_MIN_STEP=${milestone},STAGE_DEADLINE=${deadline}")
  if [[ -n "${previous}" ]]; then options+=(--dependency="afterok:${previous}"); fi
  sbatch --test-only "${options[@]}" "${runner}" >&2
  sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
}

if [[ "${mode}" == smoke ]]; then
  run_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
  job_id="$(submit smoke "" 04:00:00 2 00:03:30:00 allow "${run_id}")"
  echo "CHAIN smoke=${job_id}"
elif [[ "${mode}" == start ]]; then
  python3 "${experiment}/resume_contract.py" \
    --checkpoint-dir "${FINAL_DIR}/checkpoints" --smoke-proof "${smoke_proof}" \
    --git-sha "${EXPECTED_HEAD}" \
    --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
    --drafter-revision 03326e5043815da1f81b109078b2889737c26017 \
    --container-sha256 6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44 \
    --num-speculative-tokens "${num_speculative_tokens}" --validate-smoke-proof
  run_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
  previous="$(submit start "" 04:00:00 350 00:03:30:00 allow "${run_id}")"
  echo "CHAIN to350=${previous}"
  for spec in \
    "resume 700 00:03:30:00" \
    "resume 1000 00:03:30:00"; do
    read -r stage milestone deadline <<<"${spec}"
    previous="$(submit "${stage}" "${previous}" 04:00:00 "${milestone}" "${deadline}" must "${run_id}")"
    echo "CHAIN to${milestone}=${previous}"
  done
else
  echo "usage: $0 smoke ARM | $0 start ARM" >&2
  exit 2
fi
echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${run_id}"
