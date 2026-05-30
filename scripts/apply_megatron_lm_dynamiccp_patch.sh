#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM}"
PATCH_FILE="${PATCH_FILE:-${SCRIPT_DIR}/patches/megatron_lm_dynamiccp_nano_omni.patch}"

if [[ ! -d "${MEGATRON_LM_DIR}/.git" && ! -f "${MEGATRON_LM_DIR}/.git" ]]; then
  echo "Megatron-LM git checkout not found: ${MEGATRON_LM_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

if git -C "${MEGATRON_LM_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Megatron-LM DynamicCP patch is already applied."
  exit 0
fi

git -C "${MEGATRON_LM_DIR}" apply --check "${PATCH_FILE}"
git -C "${MEGATRON_LM_DIR}" am "${PATCH_FILE}"

echo "Applied Megatron-LM DynamicCP patch in ${MEGATRON_LM_DIR}"
