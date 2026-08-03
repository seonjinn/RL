#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ "$#" == 7 ]] || { echo "usage: run_r3_validated_command.sh RECORD_PYTHON RUN_LOG_DIR REPO_ROOT UV DRIVER_FILE DRIVER_SHA256 CHECKER_SHA256" >&2; exit 2; }
record_python=$1
shift
exec "${record_python}" "${script_dir}/run_r3_validated_command.py" "$@"
