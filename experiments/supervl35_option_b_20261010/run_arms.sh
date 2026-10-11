#!/bin/bash
set -euo pipefail
source_commit=$1
shift
launcher=$(realpath "$(dirname "$0")/run_node.sh")
for arm in "$@"; do
    bash "$launcher" "$arm" "$source_commit"
done
