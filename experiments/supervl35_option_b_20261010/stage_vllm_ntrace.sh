#!/bin/bash
set -euo pipefail

# Stage one small source archive instead of thousands of files on shared storage.
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=${NTRACE_SOURCE:?Set NTRACE_SOURCE to a pinned ntrace checkout}
destination=${1:?Pass an empty local staging directory}
test -f "$source_dir/pyproject.toml"
git -C "$source_dir" merge-base --is-ancestor \
    2a63da2c4603f26c77f78b057b945e92dd9e210b HEAD
test ! -e "$destination"
mkdir -p "$destination/configs"
experiment=$repo/experiments/supervl35_option_b_20261010
cp "$experiment/bench_vllm_generation.py" \
    "$experiment/ntrace_vllm_worker.py" \
    "$experiment/run_vllm_ab.sh" "$destination/"
cp "$experiment/configs/bf16.yaml" "$destination/configs/bf16.yaml"
cp "$experiment/configs/moe.yaml" "$destination/configs/moe.yaml"
git -C "$source_dir" rev-parse HEAD >"$destination/ntrace-commit.txt"
git -C "$source_dir" archive --format=tar HEAD | gzip -n >"$destination/ntrace-source.tar.gz"
(cd "$destination" && sha256sum ntrace-source.tar.gz >ntrace-source.tar.gz.sha256)
printf 'staged=%s ntrace_commit=%s\n' "$destination" "$(cat "$destination/ntrace-commit.txt")"
