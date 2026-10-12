#!/bin/bash
set -euo pipefail

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$launcher_dir" rev-parse --show-toplevel)
spool=$(mktemp -d)
trap 'rm -r "$spool"' EXIT
cp "$launcher_dir/submit_runtime_sbatch.sh" "$spool/submit_runtime_sbatch.sh"

srun() { printf '%s\n' "$@"; }
export -f srun
output=$(SLURM_NNODES=1 bash "$spool/submit_runtime_sbatch.sh" --run \
    "$(git -C "$repo" rev-parse HEAD)" "$repo" "$spool" /tmp/test-image.sqsh)
actual=$(printf '%s\n' "$output" | tail -n 2 | head -n 1)
expected="$launcher_dir/prepare_runtime.sh"
[[ "$actual" == "$expected" ]] || {
    printf 'Expected runtime script %s, got %s\n' "$expected" "$actual" >&2
    exit 1
}
