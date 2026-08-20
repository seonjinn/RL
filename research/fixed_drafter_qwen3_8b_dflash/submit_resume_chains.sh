#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DRAFTER_SNAPSHOT:?Set DRAFTER_SNAPSHOT}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

runner="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/run_resume_oci_hsg.sbatch"
training_horizon_steps=1000

for dflash_k in 5 7; do
  gate_job_var="GATE_JOB_K${dflash_k}"
  gate_run_var="GATE_RUN_DIR_K${dflash_k}"
  gate_job_id="${!gate_job_var:?Set ${gate_job_var}}"
  gate_run_dir="${!gate_run_var:?Set ${gate_run_var}}"
  gate_state="$(sacct -X -j "${gate_job_id}" --noheader --format=State | awk 'NF {print $1; exit}')"
  if [[ "${gate_state}" != COMPLETED ]]; then
    echo "Gate ${gate_job_id} for K=${dflash_k} is ${gate_state:-unknown}, not COMPLETED." >&2
    exit 1
  fi

  checkpoint_dir="${gate_run_dir}/checkpoints"
  gate_manifest="${gate_run_dir}/gate-manifest.json"
  previous_job_id="${gate_job_id}"

  for chunk in 1 2 3 4; do
    segment_run_dir="${RUN_ROOT}/k$(printf '%03d' "${dflash_k}")/chunk$(printf '%02d' "${chunk}")"
    exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},RUN_DIR=${segment_run_dir},CHECKPOINT_DIR=${checkpoint_dir},GATE_MANIFEST=${gate_manifest},TRAINING_HORIZON_STEPS=${training_horizon_steps},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},DFLASH_K=${dflash_k}"

    sbatch --test-only \
      --dependency="afterok:${previous_job_id}" \
      --export="${exports}" \
      "${runner}"
    submitted="$(sbatch --parsable \
      --dependency="afterok:${previous_job_id}" \
      --export="${exports}" \
      "${runner}")"
    previous_job_id="${submitted%%;*}"
    echo "K=${dflash_k} chunk=${chunk} job=${previous_job_id}"
  done
done
