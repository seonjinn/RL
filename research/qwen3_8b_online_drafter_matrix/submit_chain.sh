#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed matrix SHA}"
: "${RESULT_ROOT:?Set fresh /lustre result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly arm="${1:?usage: submit_chain.sh ARM}"
readonly experiment="${REMOTE_REPO}/research/qwen3_8b_online_drafter_matrix"
readonly runner="${experiment}/run_oci_hsg.sbatch"
python3 "${experiment}/runtime_contract.py" --arm "${arm}" --print-config >/dev/null
readonly final_dir="${RESULT_ROOT}/${arm}"
test ! -e "${final_dir}"
wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
readonly wandb_id
readonly exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},MATRIX_ARM=${arm},WANDB_RUN_ID=${wandb_id}"

submit() {
  local stage=$1 dependency=$2 walltime=$3 milestone=$4 deadline=$5 gate=$6 resume=$7
  local options=(--account="${SBATCH_ACCOUNT}" --time="${walltime}" \
    --output="/raid/scratch/matrix-%j.out" --job-name="q8-${arm}-${stage}" \
    --export="${exports},STAGE_MIN_STEP=${milestone},STAGE_DEADLINE=${deadline},IS_GATE=${gate},WANDB_RESUME=${resume}")
  if [[ -n "${dependency}" ]]; then options+=(--dependency="afterok:${dependency}"); fi
  sbatch --test-only "${options[@]}" "${runner}" >&2
  sbatch --parsable "${options[@]}" "${runner}" | cut -d';' -f1
}

previous="$(submit gate "" 01:00:00 2 00:00:50:00 1 allow)"
echo "CHAIN ${arm} gate=${previous}"
for spec in \
  "segment1 04:00:00 350 00:03:30:00" \
  "segment2 04:00:00 700 00:03:30:00" \
  "segment3 04:00:00 1000 00:03:30:00"; do
  read -r stage walltime milestone deadline <<<"${spec}"
  previous="$(submit "${stage}" "${previous}" "${walltime}" "${milestone}" "${deadline}" 0 must)"
  echo "CHAIN ${arm} ${stage}=${previous}"
done
echo "WANDB_URL=https://wandb.ai/nvidia/sna-nemo-rl-online-drafter/runs/${wandb_id}"
