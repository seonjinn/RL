#!/bin/bash

set -Eeuo pipefail

test "$#" -ge 1
jobs=("$@")
for job in "${jobs[@]}"; do
  [[ "${job}" =~ ^[0-9]+$ ]]
done
job_csv=$(IFS=,; echo "${jobs[*]}")

seen="|"
for pass in 1 2 3 4 5; do
  sleep 60
  echo "monitoring_pass=${pass} timestamp=$(date -Is)"
  records="$(
    sacct -j "${job_csv}" -n -X -P \
      -o JobIDRaw,JobName,State,ExitCode,Elapsed
  )"
  printf '%s\n' "${records}"
  completed=0
  while IFS='|' read -r job_id job_name state exit_code elapsed; do
    [[ "|${job_csv//,/|}|" == *"|${job_id}|"* ]] || continue
    seen="${seen}${job_id}|"
    case "${state}" in
      COMPLETED) completed=$((completed + 1)) ;;
      FAILED* | CANCELLED* | TIMEOUT* | NODE_FAIL* | OUT_OF_MEMORY* | \
        PREEMPTED* | BOOT_FAIL* | DEADLINE* | REVOKED*)
        echo "terminal_failure=${job_id}|${job_name}|${state}|${exit_code}|${elapsed}" >&2
        exit 1
        ;;
    esac
  done <<< "${records}"
  if ((completed == ${#jobs[@]})); then
    echo "all_jobs_completed=yes"
    exit 0
  fi
done

for job in "${jobs[@]}"; do
  if [[ "${seen}" != *"|${job}|"* ]]; then
    echo "unseen_job=${job}" >&2
    exit 1
  fi
done
echo "five_minute_monitor_complete=yes"
