#!/bin/bash

set -Eeuo pipefail
test "$#" -eq 3
jobs=("$@")
test "$(printf '%s\n' "${jobs[@]}" | sort -u | wc -l | tr -d ' ')" -eq 3 || {
  echo "duplicate job ID" >&2
  exit 2
}
for job in "${jobs[@]}"; do [[ "${job}" =~ ^[0-9]+$ ]]; done
job_csv=$(IFS=,; echo "${jobs[*]}")
seen="|"
for pass in 1 2 3 4 5; do
  sleep 60
  echo "monitoring_pass=${pass} timestamp=$(date -Is)"
  records=$(sacct -j "${job_csv}" -n -X -P -o JobIDRaw,JobName,State,ExitCode,Elapsed)
  printf '%s\n' "${records}"
  while IFS='|' read -r job_id job_name state exit_code elapsed; do
    [[ "|${job_csv//,/|}|" == *"|${job_id}|"* ]] || continue
    seen="${seen}${job_id}|"
    case "${state}" in
      FAILED* | CANCELLED* | TIMEOUT* | NODE_FAIL* | OUT_OF_MEMORY* | PREEMPTED*)
        echo "terminal_failure=${job_id}|${job_name}|${state}|${exit_code}|${elapsed}" >&2
        exit 1
        ;;
    esac
  done <<< "${records}"
done
for job in "${jobs[@]}"; do
  [[ "${seen}" == *"|${job}|"* ]] || { echo "unseen_job=${job}" >&2; exit 1; }
done
echo "five_minute_monitor_complete=yes"
