#!/bin/bash

set -euo pipefail

: "${REMOTE_REPO:?Set REMOTE_REPO to an exact /home checkout}"
: "${EXPECTED_HEAD:?Set EXPECTED_HEAD}"
: "${FINAL_ROOT:?Set FINAL_ROOT under /lustre}"
: "${CONTAINER:?Set CONTAINER}"
: "${TARGET_SNAPSHOT:?Set TARGET_SNAPSHOT}"
: "${DRAFTER_SNAPSHOT:?Set DRAFTER_SNAPSHOT}"
: "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT after the FairShare check}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

[[ "${REMOTE_REPO}" == /home/* ]]
[[ "${FINAL_ROOT}" == /lustre/* ]]

readonly dflash_gate="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/run_oci_hsg.sbatch"
readonly dflash_resume="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_dflash/run_resume_oci_hsg.sbatch"
readonly baseline_gate="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/run_oci_hsg.sbatch"
readonly baseline_resume="${REMOTE_REPO}/research/fixed_drafter_qwen3_8b_no_spec_cg/run_resume_oci_hsg.sbatch"

submit_job() {
  local arm=$1
  local stage=$2
  local dependency=$3
  local runner=$4
  local dflash_k=$5
  local final_dir="${FINAL_ROOT}/${arm}"
  local exports="ALL,REMOTE_REPO=${REMOTE_REPO},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${final_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT}"
  if [[ -n "${dflash_k}" ]]; then
    exports+=",DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT},DFLASH_K=${dflash_k}"
  fi
  local options=(--account="${SBATCH_ACCOUNT}" --job-name="q8-4k-${arm}-${stage}" --export="${exports}")
  if [[ -n "${dependency}" ]]; then
    options+=(--dependency="afterok:${dependency}")
  fi
  planner="$(sbatch --test-only "${options[@]}" "${runner}" 2>&1)"
  echo "TEST_ONLY arm=${arm} stage=${stage} dependency=${dependency:-none} ${planner}" >&2
  submitted="$(sbatch --parsable "${options[@]}" "${runner}")"
  echo "${submitted%%;*}"
}

submit_arm() {
  local arm=$1
  local dflash_k=$2
  local gate_runner=$3
  local resume_runner=$4
  local gate
  gate="$(submit_job "${arm}" gate "" "${gate_runner}" "${dflash_k}")"
  echo "CHAIN arm=${arm} gate=${gate}"
  local previous=${gate}
  local segment
  for segment in 1 2 3 4; do
    previous="$(submit_job "${arm}" "resume${segment}" "${previous}" "${resume_runner}" "${dflash_k}")"
    echo "CHAIN arm=${arm} resume${segment}=${previous}"
  done
}

submit_arm baseline "" "${baseline_gate}" "${baseline_resume}"
submit_arm dflash_k5 5 "${dflash_gate}" "${dflash_resume}"
submit_arm dflash_k7 7 "${dflash_gate}" "${dflash_resume}"
