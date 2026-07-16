#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/nemo_rl_venv/bin/python "${script_dir}/stage_drafters.py" "$@"
