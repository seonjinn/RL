#!/bin/bash
set -euo pipefail

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper="$launcher_dir/sync_driver_log.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
source_log="$tmp/node.log"
driver_log="$tmp/output/driver.log"

printf 'first step\n' >"$source_log"
bash "$helper" "$source_log" "$driver_log" --once
cmp "$source_log" "$driver_log"

printf 'second step\n' >>"$source_log"
bash "$helper" "$source_log" "$driver_log" --once
bash "$helper" "$source_log" "$driver_log" --once
cmp "$source_log" "$driver_log"

if bash "$helper" "$tmp/missing.log" "$driver_log" --once 2>/dev/null; then
    echo "A missing source log was accepted" >&2
    exit 1
fi

printf 'short\n' >"$source_log"
if bash "$helper" "$source_log" "$driver_log" --once 2>/dev/null; then
    echo "A truncated source log was accepted" >&2
    exit 1
fi

rm "$driver_log"
printf 'live start\n' >"$source_log"
bash "$helper" "$source_log" "$driver_log" 1 &
mirror_pid=$!
trap 'kill "$mirror_pid" 2>/dev/null || true; wait "$mirror_pid" 2>/dev/null || true; rm -rf "$tmp"' EXIT
for _ in 1 2 3 4 5; do
    [[ -f "$driver_log" ]] && break
    sleep 1
done
[[ -f "$driver_log" ]]
printf 'live update\n' >>"$source_log"
for _ in 1 2 3 4 5; do
    cmp -s "$source_log" "$driver_log" && break
    sleep 1
done
cmp "$source_log" "$driver_log"
