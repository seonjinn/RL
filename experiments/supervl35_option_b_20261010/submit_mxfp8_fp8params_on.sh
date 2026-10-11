#!/bin/bash
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/submit_arm.sh" mxfp8-fp8params-on
