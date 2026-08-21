#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set exact /home checkout}"
: "${EXPECTED_HEAD:?Set signed execution SHA}"
: "${RUN_ROOT:?Set fresh result parent}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

readonly experiment="${REMOTE_REPO}/research/qwen3_8b_dflash_nonnsys_sanity_b"
readonly online_runner="${experiment}/run_online_oci_hsg.sbatch"
readonly fixed_runner="${experiment}/run_fixed_oci_hsg.sbatch"
readonly wandb_project=sna-nemo-rl-online-drafter
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
online_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(6))')"
fixed_wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(6))')"
readonly timestamp online_wandb_id fixed_wandb_id
readonly online_final_dir="${RUN_ROOT}/${timestamp}-online-${online_wandb_id}"
readonly fixed_final_dir="${RUN_ROOT}/${timestamp}-fixed-${fixed_wandb_id}"
readonly submission_manifest="${RUN_ROOT}/${timestamp}-submission.json"

if [[ "${online_wandb_id}" == "${fixed_wandb_id}" ]]; then
  echo "fresh W&B IDs unexpectedly collided" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

readonly common_exports="REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},WANDB_PROJECT=${wandb_project},WANDB_RESUME=allow"
readonly online_exports="ALL,${common_exports},FINAL_DIR=${online_final_dir},WANDB_RUN_ID=${online_wandb_id}"
readonly fixed_exports="ALL,${common_exports},FINAL_DIR=${fixed_final_dir},WANDB_RUN_ID=${fixed_wandb_id},IS_GATE=1,STAGE_MIN_STEP=50,STAGE_DEADLINE=00:00:50:00,SOURCE_GATE_SHA=${EXPECTED_HEAD}"
readonly online_options=(
  --account="${SBATCH_ACCOUNT}"
  --time=01:00:00
  --output=/raid/scratch/q8-online-nonnsys-sanity-b-%j.out
  --job-name=q8-online-k7-nonnsys-sanity-b
  --export="${online_exports}"
)
readonly fixed_options=(
  --account="${SBATCH_ACCOUNT}"
  --time=01:00:00
  --output=/raid/scratch/q8-fixed-nonnsys-sanity-b-%j.out
  --job-name=q8-fixed-k7-nonnsys-sanity-b
  --export="${fixed_exports}"
)

# Validate both scheduler requests before either independent actual submission.
sbatch --test-only "${online_options[@]}" "${online_runner}" >&2
sbatch --test-only "${fixed_options[@]}" "${fixed_runner}" >&2
online_job_id="$(sbatch --parsable "${online_options[@]}" "${online_runner}" | cut -d';' -f1)"
fixed_job_id="$(sbatch --parsable "${fixed_options[@]}" "${fixed_runner}" | cut -d';' -f1)"
readonly online_job_id fixed_job_id

python3 "${experiment}/write_submission_manifest.py" \
  --output "${submission_manifest}" --expected-head "${EXPECTED_HEAD}" \
  --online-job-id "${online_job_id}" --fixed-job-id "${fixed_job_id}" \
  --online-wandb-run-id "${online_wandb_id}" \
  --fixed-wandb-run-id "${fixed_wandb_id}" \
  --online-final-dir "${online_final_dir}" --fixed-final-dir "${fixed_final_dir}"

echo "ONLINE_JOB_ID=${online_job_id}"
echo "FIXED_JOB_ID=${fixed_job_id}"
echo "ONLINE_WANDB_URL=https://wandb.ai/nvidia/${wandb_project}/runs/${online_wandb_id}"
echo "FIXED_WANDB_URL=https://wandb.ai/nvidia/${wandb_project}/runs/${fixed_wandb_id}"
echo "ONLINE_FINAL_DIR=${online_final_dir}"
echo "FIXED_FINAL_DIR=${fixed_final_dir}"
echo "SUBMISSION_MANIFEST=${submission_manifest}"
