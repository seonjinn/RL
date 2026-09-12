#!/usr/bin/env bash
set -euo pipefail
: "${SWE_NODE_ROOT:?}" "${SWE_SOURCE_ROOT:?}"
mkdir -p "${SWE_NODE_ROOT}/tmp" "${SWE_NODE_ROOT}/Gym"
# Gym creates evaluator repos, environments, locks and results relative to its
# source. Bind this private node-local copy over the checkout inside Pyxis.
cp -a "${SWE_SOURCE_ROOT}/3rdparty/Gym-workspace/Gym/." "${SWE_NODE_ROOT}/Gym/"
printf '[SWE-GATE] staged isolated Gym source on %s\n' "$(hostname)"
