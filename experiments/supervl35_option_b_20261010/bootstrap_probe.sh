#!/bin/bash
set -euo pipefail
umask 077
source_commit=$1
sample=$2
output=$3
mkdir -p "$output"
output=$(realpath "$output")
sample=$(realpath "$sample")
exec >"$output/bootstrap.log" 2>&1
checkout="$PWD/source-${source_commit}"
if ! test -d "$checkout/.git"; then
    git clone --depth 1 --no-checkout --single-branch \
        --branch sna/supervl35-optionb-20261010 \
        https://github.com/seonjinn/RL.git "$checkout"
fi
git -C "$checkout" fetch --depth 1 origin "$source_commit"
git -C "$checkout" checkout --detach "$source_commit"
test "$(git -C "$checkout" rev-parse HEAD)" = "$source_commit"
git -C "$checkout" submodule update --init --recursive --depth 1
git -C "$checkout" submodule status --recursive >"$output/submodules.txt"
archive="/raid/scratch/sna/supervl35-bootstrap-${SLURM_JOB_ID}/source.tar"
mkdir -p "$(dirname "$archive")"
tar --exclude=.git -cf "$archive" -C "$checkout" .
bash "$checkout/experiments/supervl35_option_b_20261010/runtime_probe.sh" \
    "$archive" "$output/runtime" "$sample"
