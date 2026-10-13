#!/bin/bash
set -euo pipefail

[[ $# -eq 3 ]] || { echo "Usage: capture_policy_nsys.sh RAY_ROOT OUTPUT_DIR NODE_RANK" >&2; exit 2; }
ray_root=$1
output_dir=$2
rank=$3
[[ $rank =~ ^[0-9]+$ ]] || { echo "NODE_RANK must be a nonnegative integer" >&2; exit 2; }

# The Ray session is node-local. Keep one representative policy trace per node.
report=$(find "$ray_root" -type f -name '*megatron_policy_worker*.nsys-rep' -print -quit)
if [[ -z $report ]]; then
    echo "No policy nsys report on node rank $rank" >&2
    exit 0
fi

mkdir -p "$output_dir"
destination="$output_dir/node-${rank}.nsys-rep"
cp "$report" "$destination.tmp"
mv "$destination.tmp" "$destination"
size_bytes=$(wc -c <"$destination" | tr -d '[:space:]')
printf 'node_rank=%s\nhost=%s\nsource=%s\nsize_bytes=%s\n' \
    "$rank" "$(hostname)" "$report" "$size_bytes" >"$output_dir/node-${rank}.txt.tmp"
mv "$output_dir/node-${rank}.txt.tmp" "$output_dir/node-${rank}.txt"
echo "Saved policy nsys report: $destination"
