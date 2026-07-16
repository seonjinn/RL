#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--worker" ]]; then
  python_bin="/opt/nemo_rl_venv/bin/python"
else
  python_bin="${PYTHON_BIN:-python3}"
fi
exec "${python_bin}" "${script_dir}/stage_drafters.py" "$@"
