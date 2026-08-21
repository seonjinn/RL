#!/bin/bash

set -Eeuo pipefail

: "${PARITY_REMOTE_REPO:?Set exact clean parity /home checkout}"
: "${FIXED_REMOTE_REPO:?Set exact clean fixed /home checkout}"
: "${ONLINE_REMOTE_REPO:?Set exact clean online /home checkout}"
: "${EXPECTED_HEAD:?Set signed harness SHA}"
: "${FINAL_ROOT:?Set fresh /lustre pair result root}"
: "${CONTAINER:?Set immutable container}"
: "${TARGET_SNAPSHOT:?Set exact target snapshot}"
: "${DRAFTER_SNAPSHOT:?Set exact drafter snapshot}"
: "${SBATCH_ACCOUNT:?Set best eligible FairShare account}"
: "${WANDB_API_KEY:?W&B cluster secret is required}"

canonical_checkout() {
  local checkout=$1
  test -d "${checkout}"
  realpath "${checkout}"
}

checkout_identity() {
  python3 -c 'import os, sys; value = os.stat(sys.argv[1]); print(f"{value.st_dev}:{value.st_ino}")' "$1"
}

parity_remote_repo="$(canonical_checkout "${PARITY_REMOTE_REPO}")"
readonly parity_remote_repo
fixed_remote_repo="$(canonical_checkout "${FIXED_REMOTE_REPO}")"
readonly fixed_remote_repo
online_remote_repo="$(canonical_checkout "${ONLINE_REMOTE_REPO}")"
readonly online_remote_repo
parity_identity="$(checkout_identity "${parity_remote_repo}")"
readonly parity_identity
fixed_identity="$(checkout_identity "${fixed_remote_repo}")"
readonly fixed_identity
online_identity="$(checkout_identity "${online_remote_repo}")"
readonly online_identity

if [[ "${parity_remote_repo}" == "${fixed_remote_repo}" || \
  "${parity_remote_repo}" == "${online_remote_repo}" || \
  "${fixed_remote_repo}" == "${online_remote_repo}" || \
  "${parity_identity}" == "${fixed_identity}" || \
  "${parity_identity}" == "${online_identity}" || \
  "${fixed_identity}" == "${online_identity}" ]]; then
  echo "checkout alias: parity, fixed, and online must be distinct canonical directories" >&2
  exit 1
fi

readonly expected_container_sha=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
readonly parity_experiment="${parity_remote_repo}/research/qwen3_8b_dflash_nonnsys_ab"
readonly fixed_experiment="${fixed_remote_repo}/research/qwen3_8b_dflash_nonnsys_ab"
readonly experiment="${fixed_experiment}"
readonly parity_authority="${PARITY_AUTHORITY:-${parity_experiment}/resolved_parity.py}"
readonly monitor_script="${MONITOR_SCRIPT:-${experiment}/monitor_pair.sh}"
readonly wandb_project=sna-nemo-rl-online-drafter
readonly parity_dir="${FINAL_ROOT}/parity"
readonly parity_proof="${parity_dir}/resolved-parity.json"
readonly online_config="${parity_remote_repo}/research/qwen3_8b_dflash_online_cp1/config.yaml"
readonly fixed_config="${parity_remote_repo}/research/qwen3_8b_dflash_fixed_dense_control/config.yaml"

if [[ -z "${PARITY_AUTHORITY:-}" ]]; then
  [[ "${parity_remote_repo}" == /home/* ]]
  [[ "${fixed_remote_repo}" == /home/* ]]
  [[ "${online_remote_repo}" == /home/* ]]
  [[ "${FINAL_ROOT}" == /lustre/* ]]
fi
test ! -e "${FINAL_ROOT}"

submit_parity() {
  local test_only=$1
  local runner_path="${parity_experiment}/run_parity_oci_hsg.sbatch"
  local exports="ALL,REMOTE_REPO=${parity_remote_repo},EXPECTED_HEAD=${EXPECTED_HEAD},FINAL_DIR=${parity_dir},CONTAINER=${CONTAINER},TARGET_SNAPSHOT=${TARGET_SNAPSHOT},DRAFTER_SNAPSHOT=${DRAFTER_SNAPSHOT}"
  local options=(
    --account="${SBATCH_ACCOUNT}"
    --export="${exports}"
  )
  if [[ "${test_only}" == 1 ]]; then
    sbatch --test-only "${options[@]}" "${runner_path}" >&2
  else
    sbatch --parsable "${options[@]}" "${runner_path}" | cut -d';' -f1
  fi
}

wait_for_parity() {
  local job_id=$1
  local seen=0
  for pass in {1..20}; do
    sleep 60
    echo "parity_monitoring_pass=${pass} timestamp=$(date -Is)"
    local records
    records="$(
      sacct -j "${job_id}" -n -X -P \
        -o JobIDRaw,JobName,State,ExitCode,Elapsed
    )"
    printf '%s\n' "${records}"
    while IFS='|' read -r record_job_id job_name state exit_code elapsed; do
      [[ "${record_job_id}" == "${job_id}" ]] || continue
      seen=1
      case "${state}" in
        COMPLETED) return 0 ;;
        FAILED* | CANCELLED* | TIMEOUT* | NODE_FAIL* | OUT_OF_MEMORY* | \
          PREEMPTED* | BOOT_FAIL* | DEADLINE* | REVOKED*)
          echo "parity_terminal_failure=${record_job_id}|${job_name}|${state}|${exit_code}|${elapsed}" >&2
          return 1
          ;;
      esac
    done <<< "${records}"
  done
  if ((seen == 0)); then
    echo "unseen_parity_job=${job_id}" >&2
  else
    echo "parity_job_not_terminal_after_20_passes=${job_id}" >&2
  fi
  return 1
}

submit_parity 1
parity_job="$(submit_parity 0)"
readonly parity_job
[[ "${parity_job}" =~ ^[0-9]+$ ]]
echo "parity_job=${parity_job}"
wait_for_parity "${parity_job}"
python3 "${parity_authority}" validate-proof \
  --proof "${parity_proof}" \
  --expected-head "${EXPECTED_HEAD}" \
  --target-snapshot "${TARGET_SNAPSHOT}" \
  --drafter-snapshot "${DRAFTER_SNAPSHOT}" \
  --container-sha256 "${expected_container_sha}" \
  --wandb-project "${wandb_project}" \
  --parity-job-id "${parity_job}" \
  --online-config "${online_config}" \
  --fixed-config "${fixed_config}"

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

submit fixed "${fixed_wandb_id}" 1 "${fixed_remote_repo}"
submit online "${online_wandb_id}" 1 "${online_remote_repo}"
fixed_job="$(submit fixed "${fixed_wandb_id}" 0 "${fixed_remote_repo}")"
if ! online_job="$(submit online "${online_wandb_id}" 0 "${online_remote_repo}")"; then
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
