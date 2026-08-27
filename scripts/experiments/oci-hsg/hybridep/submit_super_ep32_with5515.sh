#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:-"${SCRIPT_DIR}/models/nemotron3-super-120ba12b-32n4g-ep32-with5515.env"}

exec "${SCRIPT_DIR}/submit_super_ep32_no5515.sh" "${PROFILE}"
