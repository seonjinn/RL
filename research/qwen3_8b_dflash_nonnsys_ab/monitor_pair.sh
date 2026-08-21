#!/bin/bash

set -Eeuo pipefail

readonly fixed_job=${1:?Pass the fixed job ID}
readonly online_job=${2:?Pass the online job ID}
[[ "${fixed_job}" =~ ^[0-9]+$ ]]
[[ "${online_job}" =~ ^[0-9]+$ ]]

for pass in 1 2 3 4 5; do
  sleep 60
  echo "monitoring_pass=${pass} timestamp=$(date -Is)"
  squeue -j "${fixed_job},${online_job}" -h -o '%i|%j|%T|%M|%r|%N'
done
