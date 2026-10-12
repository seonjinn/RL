#!/bin/bash
set -euo pipefail
umask 077

[[ $# -eq 3 ]] || { echo "Usage: sync_driver_log.sh SOURCE DEST (--once|SECONDS)" >&2; exit 2; }
source_log=$1
driver_log=$2
mode=$3
if [[ "$mode" != --once && ! "$mode" =~ ^[1-9][0-9]*$ ]]; then
    echo "Sync interval must be a positive number of seconds" >&2
    exit 2
fi

sync_once() {
    [[ -f "$source_log" ]] || { echo "Missing source log: $source_log" >&2; return 1; }
    mkdir -p "$(dirname -- "$driver_log")"
    [[ -e "$driver_log" ]] || : >"$driver_log"

    local source_size destination_size
    source_size=$(wc -c <"$source_log")
    destination_size=$(wc -c <"$driver_log")
    if (( destination_size > source_size )); then
        echo "Source log was truncated; refusing to append to $driver_log" >&2
        return 1
    fi
    if (( destination_size < source_size )); then
        tail -c +"$((destination_size + 1))" -- "$source_log" >>"$driver_log"
    fi
}

if [[ "$mode" == --once ]]; then
    sync_once
else
    while true; do
        sync_once
        sleep "$mode"
    done
fi
