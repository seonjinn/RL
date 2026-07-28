#!/usr/bin/env bash
# Prepare an exact, recursively initialized source tree before a strict runtime
# probe. This script intentionally performs no scheduler submission.
set -euo pipefail

BRANCH="experiment/latestmain-pr5672-nano-matrix-20260727"
REMOTE="${REMOTE:-git@github-seonjinn:seonjinn/RL.git}"
DESTINATION="${1:?Usage: $0 /absolute/path/to/fresh-nemo-rl-clone}"

if [[ -d "${DESTINATION}/.git" ]]; then
  git -C "${DESTINATION}" fetch origin "${BRANCH}"
  git -C "${DESTINATION}" switch "${BRANCH}"
  git -C "${DESTINATION}" pull --ff-only origin "${BRANCH}"
else
  git clone --branch "${BRANCH}" "${REMOTE}" "${DESTINATION}"
fi

cd "${DESTINATION}"
git submodule sync --recursive
git submodule update --init --recursive

GYM_PATH=3rdparty/Gym-workspace/Gym
BRIDGE_PATH=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE_PATH="${BRIDGE_PATH}/3rdparty/Megatron-LM"
if [[ ! -d "${GYM_PATH}" ]]; then
  echo "Gym workspace is missing after recursive submodule initialization" >&2
  exit 1
fi
if [[ ! -d "${BRIDGE_PATH}" || ! -d "${MCORE_PATH}" ]]; then
  echo "Bridge or nested Megatron-LM workspace is missing after initialization" >&2
  exit 1
fi

printf 'root_sha=%s\n' "$(git rev-parse HEAD)"
printf 'bridge_sha=%s\n' "$(git -C "${BRIDGE_PATH}" rev-parse HEAD)"
printf 'mcore_sha=%s\n' "$(git -C "${MCORE_PATH}" rev-parse HEAD)"
printf 'strict_runtime_probe_ready=%s\n' "${DESTINATION}"
