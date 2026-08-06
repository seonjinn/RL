#!/bin/bash

QUANT_MODE=w4a16
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/nvfp4-matched-transport-common.sh" "$@"
