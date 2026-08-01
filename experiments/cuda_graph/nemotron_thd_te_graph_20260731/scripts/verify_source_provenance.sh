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

if (($# != 6)); then
  fail "usage: verify_source_provenance.sh REPO SHA BRIDGE SHA MCORE SHA"
fi

repositories=("$1" "$3" "$5")
expected_commits=("$2" "$4" "$6")
labels=("NeMo-RL" "Megatron-Bridge" "Megatron-LM")

for index in 0 1 2; do
  repository=${repositories[${index}]}
  expected_commit=${expected_commits[${index}]}
  label=${labels[${index}]}
  [[ "${repository}" == /* ]] || fail "${label} repository must be absolute"
  [[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || \
    fail "${label} expected commit must be a full lowercase SHA"
  [[ -d "${repository}/.git" || -f "${repository}/.git" ]] || \
    fail "${label} repository is not a Git worktree: ${repository}"
  actual_commit=$(git -C "${repository}" rev-parse HEAD)
  [[ "${actual_commit}" == "${expected_commit}" ]] || \
    fail "${label} source SHA mismatch: expected ${expected_commit}, got ${actual_commit}"
  git -C "${repository}" diff --quiet --ignore-submodules=none || \
    fail "${label} source worktree has unstaged changes"
  git -C "${repository}" diff --cached --quiet --ignore-submodules=none || \
    fail "${label} source worktree has staged changes"
  [[ -z "$(git -C "${repository}" ls-files --others --exclude-standard)" ]] || \
    fail "${label} source worktree has untracked files"
done

echo "SOURCE_PROVENANCE_VERIFIED=true"
