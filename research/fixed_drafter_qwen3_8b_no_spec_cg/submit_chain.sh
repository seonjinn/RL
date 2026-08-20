#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

gate_runner="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/run_oci_hsg.sbatch"
resume_runner="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/run_resume_oci_hsg.sbatch"
gate_run_dir="${RUN_ROOT}/gate"
checkpoint_dir="${gate_run_dir}/checkpoints"
gate_manifest="${gate_run_dir}/gate-manifest.json"
common="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},TRAINING_HORIZON_STEPS=1000,CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT}"

gate_exports="${common},RUN_DIR=${gate_run_dir}"
sbatch --test-only --export="${gate_exports}" "${gate_runner}"
submitted="$(sbatch --parsable --export="${gate_exports}" "${gate_runner}")"
previous_job_id="${submitted%%;*}"
echo "baseline gate job=${previous_job_id}"

for chunk in 1 2 3 4; do
  run_dir="${RUN_ROOT}/chunk$(printf '%02d' "${chunk}")"
  exports="${common},RUN_DIR=${run_dir},CHECKPOINT_DIR=${checkpoint_dir},GATE_MANIFEST=${gate_manifest}"
  sbatch --test-only \
    --dependency="afterok:${previous_job_id}" \
    --export="${exports}" \
    "${resume_runner}"
  submitted="$(sbatch --parsable \
    --dependency="afterok:${previous_job_id}" \
    --export="${exports}" \
    "${resume_runner}")"
  previous_job_id="${submitted%%;*}"
  echo "baseline chunk=${chunk} job=${previous_job_id}"
done
