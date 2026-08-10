#!/bin/bash

set -euo pipefail

mode=${1:-}
round_count=${2:-20}
case "${mode}" in
  plan|submit) ;;
  *) printf 'Usage: %s {plan|submit} [ROUND_COUNT]\n' "$0" >&2; exit 2 ;;
esac
case "${round_count}" in
  ''|*[!0-9]*) printf 'ROUND_COUNT must be an integer from 1 through 24\n' >&2; exit 2 ;;
esac
if ((round_count < 1 || round_count > 24)); then
  printf 'ROUND_COUNT must be an integer from 1 through 24\n' >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
round_launcher=${ROUND_LAUNCHER_OVERRIDE:-${script_dir}/submit_super_4hour.sh}
manifest=${CHAIN_MANIFEST_OVERRIDE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-super-200step-20260809/job-chain.env}

if [[ "${mode}" == plan ]]; then
  for ((round = 1; round <= round_count; round += 1)); do
    if ((round == 1)); then
      printf 'round=%s dependency=none\n' "${round}"
    else
      printf 'round=%s dependency=afterok:<round-%s-job>\n' "${round}" "$((round - 1))"
    fi
  done
  exit 0
fi

bash "${round_launcher}" test-only 1

mkdir -p "$(dirname "${manifest}")"
manifest_tmp=${manifest}.tmp.$$
trap 'rm -f -- "${manifest_tmp}"' EXIT
: > "${manifest_tmp}"

previous_job_id=
for ((round = 1; round <= round_count; round += 1)); do
  if [[ -n "${previous_job_id}" ]]; then
    dependency=afterok:${previous_job_id}
  else
    dependency=
  fi

  job_output=$(JOB_DEPENDENCY="${dependency}" bash "${round_launcher}" submit "${round}")
  job_id=$(printf '%s\n' "${job_output}" | tail -n 1)
  case "${job_id}" in
    ''|*[!0-9]*)
      printf 'Round %s returned an invalid job ID: %s\n' "${round}" "${job_id}" >&2
      exit 1
      ;;
  esac

  printf 'hybridep_round%s=%s\n' "${round}" "${job_id}" >> "${manifest_tmp}"
  printf 'submitted_round=%s job_id=%s dependency=%s\n' \
    "${round}" "${job_id}" "${dependency:-none}"
  previous_job_id=${job_id}
done

mv "${manifest_tmp}" "${manifest}"
trap - EXIT
