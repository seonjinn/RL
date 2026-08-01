#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

fail() {
  echo "$*" >&2
  exit 1
}

: "${SOURCE_ROOT:?Set SOURCE_ROOT to the clean NeMo-RL checkout}"
: "${SNAPSHOT_STORE:?Set SNAPSHOT_STORE to an absolute persistent directory}"
: "${EXPECTED_NEMORL_SHA:?Set EXPECTED_NEMORL_SHA}"
: "${EXPECTED_BRIDGE_SHA:?Set EXPECTED_BRIDGE_SHA}"
: "${EXPECTED_MCORE_SHA:?Set EXPECTED_MCORE_SHA}"

[[ "${SOURCE_ROOT}" == /* ]] || fail "SOURCE_ROOT must be absolute"
[[ "${SNAPSHOT_STORE}" == /* ]] || fail "SNAPSHOT_STORE must be absolute"
source_root=$(cd "${SOURCE_ROOT}" && pwd -P)
bridge_root=${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_root=${bridge_root}/3rdparty/Megatron-LM
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
verifier=${script_dir}/verify_source_provenance.sh

"${verifier}" \
  "${source_root}" "${EXPECTED_NEMORL_SHA}" \
  "${bridge_root}" "${EXPECTED_BRIDGE_SHA}" \
  "${mcore_root}" "${EXPECTED_MCORE_SHA}"

read -r outer_mode outer_type outer_gitlink _ < <(
  git -C "${source_root}" ls-tree HEAD \
    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
)
[[ "${outer_mode:-}" == "160000" && "${outer_type:-}" == "commit" && \
   "${outer_gitlink:-}" == "${EXPECTED_BRIDGE_SHA}" ]] || \
  fail "Outer Bridge gitlink does not match EXPECTED_BRIDGE_SHA"
read -r bridge_mode bridge_type bridge_gitlink _ < <(
  git -C "${bridge_root}" ls-tree HEAD 3rdparty/Megatron-LM
)
[[ "${bridge_mode:-}" == "160000" && "${bridge_type:-}" == "commit" && \
   "${bridge_gitlink:-}" == "${EXPECTED_MCORE_SHA}" ]] || \
  fail "Bridge MCore gitlink does not match EXPECTED_MCORE_SHA"

snapshot_name=${EXPECTED_NEMORL_SHA:0:12}-${EXPECTED_BRIDGE_SHA:0:12}-${EXPECTED_MCORE_SHA:0:12}
snapshot=${SNAPSHOT_STORE%/}/${snapshot_name}
manifest=${snapshot}/.source-manifest.env
mkdir -p "${SNAPSHOT_STORE}"

if [[ -e "${snapshot}" ]]; then
  [[ -f "${manifest}" ]] || fail "Existing snapshot has no source manifest"
  "${snapshot}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/verify_source_provenance.sh" \
    "${snapshot}" "${EXPECTED_NEMORL_SHA}" \
    "${snapshot}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    "${EXPECTED_BRIDGE_SHA}" \
    "${snapshot}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" \
    "${EXPECTED_MCORE_SHA}"
  echo "SOURCE_SNAPSHOT=${snapshot}"
  echo "SOURCE_MANIFEST=${manifest}"
  exit 0
fi

temporary=$(mktemp -d "${SNAPSHOT_STORE%/}/.source-snapshot.XXXXXX")
cleanup() {
  if [[ -n "${temporary:-}" && -d "${temporary}" ]]; then
    rm -rf -- "${temporary}"
  fi
}
trap cleanup EXIT

git clone --quiet --no-hardlinks --no-checkout "${source_root}" "${temporary}"
git -C "${temporary}" checkout --quiet --detach "${EXPECTED_NEMORL_SHA}"
git clone --quiet --no-hardlinks --no-checkout \
  "${bridge_root}" \
  "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
git -C "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
  checkout --quiet --detach "${EXPECTED_BRIDGE_SHA}"
git clone --quiet --no-hardlinks --no-checkout \
  "${mcore_root}" \
  "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
git -C "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" \
  checkout --quiet --detach "${EXPECTED_MCORE_SHA}"

snapshot_verifier=${temporary}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/verify_source_provenance.sh
"${snapshot_verifier}" \
  "${temporary}" "${EXPECTED_NEMORL_SHA}" \
  "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
  "${EXPECTED_BRIDGE_SHA}" \
  "${temporary}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" \
  "${EXPECTED_MCORE_SHA}"

uv_lock_sha256=$(sha256sum "${temporary}/uv.lock" | awk '{print $1}')
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '.source-manifest.env\n' >>"${temporary}/.git/info/exclude"
{
  printf 'nemo_rl_commit=%s\n' "${EXPECTED_NEMORL_SHA}"
  printf 'bridge_commit=%s\n' "${EXPECTED_BRIDGE_SHA}"
  printf 'mcore_commit=%s\n' "${EXPECTED_MCORE_SHA}"
  printf 'outer_bridge_gitlink=%s\n' "${outer_gitlink}"
  printf 'bridge_mcore_gitlink=%s\n' "${bridge_gitlink}"
  printf 'uv_lock_sha256=%s\n' "${uv_lock_sha256}"
  printf 'created_utc=%s\n' "${created_utc}"
} >"${temporary}/.source-manifest.env"

mv "${temporary}" "${snapshot}"
temporary=
trap - EXIT
echo "SOURCE_SNAPSHOT=${snapshot}"
echo "SOURCE_MANIFEST=${manifest}"
