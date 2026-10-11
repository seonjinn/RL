#!/bin/bash
set -euo pipefail
umask 077
output=$1
precision_recipe=$2
mkdir -p "$output"
exec >"$output/inspection.log" 2>&1
date -u
cp "$precision_recipe" "$output/te_precision.yaml"
sha256sum "$precision_recipe" >"$output/te_precision.sha256"
shift 2
for root in "$@"; do
    printf 'SEARCH_ROOT %s\n' "$root"
    if test -d "$root"; then
        find "$root" -maxdepth 4 \( -name hf_home -o -name checkpoints -o -name .cache -o -name source -o -name .git \) -prune -o \
            -type f -name '*7776525*' -print >"$output/$(basename "$root")-image-paths.txt"
    fi
done
printf '%s\n' INSPECTION_COMPLETE
