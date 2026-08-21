#!/bin/bash

set -Eeuo pipefail

readonly fixed_job=${1:?Pass the fixed job ID}
readonly online_job=${2:?Pass the online job ID}
[[ "${fixed_job}" =~ ^[0-9]+$ ]]
[[ "${online_job}" =~ ^[0-9]+$ ]]

seen_fixed=0
seen_online=0
for pass in 1 2 3 4 5; do
  sleep 60
  echo "monitoring_pass=${pass} timestamp=$(date -Is)"
  records="$(
    sacct -j "${fixed_job},${online_job}" -n -X -P \
      -o JobIDRaw,JobName,State,ExitCode,Elapsed
  )"
  printf '%s\n' "${records}"
  completed_fixed=0
  completed_online=0
  failed=0
  while IFS='|' read -r job_id job_name state exit_code elapsed; do
    case "${job_id}" in
      "${fixed_job}") seen_fixed=1 ;;
      "${online_job}") seen_online=1 ;;
      *) continue ;;
    esac
    case "${state}" in
      COMPLETED)
        if [[ "${job_id}" == "${fixed_job}" ]]; then
          completed_fixed=1
        else
          completed_online=1
        fi
        ;;
      FAILED* | CANCELLED* | TIMEOUT* | NODE_FAIL* | OUT_OF_MEMORY* | \
        PREEMPTED* | BOOT_FAIL* | DEADLINE* | REVOKED*)
        echo "terminal_failure=${job_id}|${job_name}|${state}|${exit_code}|${elapsed}" >&2
        failed=1
        ;;
    esac
  done <<< "${records}"
  if ((failed != 0)); then
    exit 1
  fi
  if ((completed_fixed == 1 && completed_online == 1)); then
    echo "both_jobs_completed=yes"
    exit 0
  fi
done

if ((seen_fixed == 0 || seen_online == 0)); then
  missing=()
  ((seen_fixed == 1)) || missing+=("${fixed_job}")
  ((seen_online == 1)) || missing+=("${online_job}")
  (IFS=,; echo "unseen_jobs=${missing[*]}" >&2)
  exit 1
fi
