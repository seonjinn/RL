#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/ray/session/logs/nsight" "$scratch/output"
printf 'policy-profile-1' >"$scratch/ray/session/logs/nsight/megatron_policy_worker_3:4_123.nsys-rep"
printf 'policy-profile-2' >"$scratch/ray/session/logs/nsight/megatron_policy_worker_3:4_456.nsys-rep"
printf 'generation-profile' >"$scratch/ray/session/logs/nsight/vllm_generation_worker_3:4_789.nsys-rep"
bash "$script_dir/capture_policy_nsys.sh" "$scratch/ray" "$scratch/output" 7
[[ $(find "$scratch/output" -name '*.nsys-rep' | wc -l) -eq 1 ]]
[[ -s "$scratch/output/node-7.nsys-rep" ]]
grep -Fq 'node_rank=7' "$scratch/output/node-7.txt"
grep -Eq '^size_bytes=[1-9][0-9]*$' "$scratch/output/node-7.txt"
echo 'Policy nsys capture test passed'
