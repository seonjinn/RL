#!/bin/bash
set -euo pipefail

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
run="$tmp/runs/42/bf16-bf16"
mkdir -p "$run"
printf 'train step 1/20\n' >"$tmp/node.log"
tar -czf "$run/node-0.tar.gz" -C "$tmp" node.log

NRL_DURABLE_ROOT="$tmp" \
    bash "$launcher_dir/capture_running_driver_log.sh" 42 bf16-bf16
cmp "$tmp/node.log" "$run/driver.log"

printf 'an older copy\n' >"$run/driver.log"
NRL_DURABLE_ROOT="$tmp" \
    bash "$launcher_dir/capture_running_driver_log.sh" 42 bf16-bf16
cmp "$tmp/node.log" "$run/driver.log"

printf 'corrupt archive\n' >"$run/node-0.tar.gz"
if NRL_DURABLE_ROOT="$tmp" \
    bash "$launcher_dir/capture_running_driver_log.sh" 42 bf16-bf16 2>/dev/null; then
    echo "Expected corrupt archive extraction to fail" >&2
    exit 1
fi
cmp "$tmp/node.log" "$run/driver.log"
