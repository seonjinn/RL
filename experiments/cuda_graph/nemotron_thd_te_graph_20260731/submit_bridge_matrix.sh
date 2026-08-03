#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

set -euo pipefail
fail() { echo "$*" >&2; exit 2; }
[[ -z "${COMMAND:-}" && -z "${BRIDGE_COMMAND:-}" ]] || \
  fail "Raw command payloads are forbidden by the typed matrix runner"
[[ "${BRIDGE_TEST_ROWS:-}" == bridge_forward_only_eval_8 ]] || \
  fail "Unknown test row: BRIDGE_TEST_ROWS must be bridge_forward_only_eval_8"
: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"
: "${PROFILE_FILE:?Set PROFILE_FILE to a validated cluster profile}"
: "${BRIDGE_CANDIDATE_SHA:?Set BRIDGE_CANDIDATE_SHA to the resolved pushed commit}"
[[ "${BRIDGE_CANDIDATE_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
  fail "BRIDGE_CANDIDATE_SHA must be one lowercase 40-character SHA"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${script_dir}/../../.." && pwd -P)
bridge_root=${repo_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
mcore_root=${bridge_root}/3rdparty/Megatron-LM
matrix=${script_dir}/bridge_test_matrix.json
driver=${script_dir}/scripts/run_mcore_training.py

profile_output=$(python3 "${script_dir}/profile_snapshot.py" \
  --profile-dir "${script_dir}/profiles" --cluster "${CLUSTER}" \
  --profile-file "${PROFILE_FILE}") || fail "Cluster profile rejected"
while IFS=$'\t' read -r field value; do
  case "${field}" in
    PROFILE_ID|ACCOUNT|PARTITION|CONTAINER|CONTAINER_SHA256|MOUNTS|SBATCH_GPUS_PER_NODE|SBATCH_GRES|SBATCH_SEGMENT_SIZE|TIME_LIMIT|RUNTIME_ATTESTATION|RUNTIME_PREFLIGHT_JOB_ID|EXPECTED_TE_SHA|EXPECTED_TE_VERSION_BASE_SHA|EXPECTED_NEMORL_SHA|EXPECTED_BRIDGE_SHA|EXPECTED_MCORE_SHA|RUN_LOG_ROOT|PROFILE_SHA256)
      printf -v "${field}" '%s' "${value}"
      ;;
    *) fail "Cluster profile snapshot returned an unknown field" ;;
  esac
done <<<"${profile_output}"
[[ "${RUN_LOG_ROOT}" == /* ]] || fail "RUN_LOG_ROOT must be absolute"
[[ "${PARTITION}" == batch ]] || fail "Typed matrix jobs require PARTITION=batch"
[[ "${SBATCH_GPUS_PER_NODE}" == 4 ]] || fail "Bridge row requires four GPUs per node"
for field in ACCOUNT CONTAINER CONTAINER_SHA256 MOUNTS RUNTIME_ATTESTATION EXPECTED_TE_SHA EXPECTED_TE_VERSION_BASE_SHA RUN_LOG_ROOT; do
  value=${!field:-}
  [[ -n "${value}" && "${value}" != *"__REQUIRED"* ]] || fail "Profile field ${field} is unresolved"
done

remote_sha=$(git -C "${bridge_root}" ls-remote origin refs/heads/sna/thd-cg-hybrid-nemotron-20260731 | awk 'NF == 2 {print $1}')
[[ "${remote_sha}" =~ ^[0-9a-f]{40}$ ]] || fail "Bridge branch did not resolve to exactly one pushed SHA"
[[ "${remote_sha}" == "${BRIDGE_CANDIDATE_SHA}" ]] || fail "Bridge candidate is absent from the pushed remote branch"
candidate_mcore_sha=$(git -C "${bridge_root}" ls-tree "${BRIDGE_CANDIDATE_SHA}" 3rdparty/Megatron-LM | awk '$2 == "commit" {print $3}')
[[ "${candidate_mcore_sha}" =~ ^[0-9a-f]{40}$ ]] || fail "Bridge candidate lacks one nested MCore gitlink"
git -C "${mcore_root}" cat-file -e "${candidate_mcore_sha}^{commit}" || \
  fail "Bridge candidate nested MCore commit is unavailable"
root_branch=$(git -C "${repo_root}" branch --show-current)
[[ "${root_branch}" == experiment/thd-cg-hybrid-nemotron-20260731 ]] || \
  fail "NeMo-RL runner must use the Task 2 infrastructure branch"
root_sha=$(git -C "${repo_root}" rev-parse HEAD)
remote_root_sha=$(git -C "${repo_root}" ls-remote seonjinn \
  refs/heads/experiment/thd-cg-hybrid-nemotron-20260731 | awk 'NF == 2 {print $1}')
[[ "${remote_root_sha}" == "${root_sha}" ]] || \
  fail "NeMo-RL runner infrastructure is not pushed at the local HEAD"
git -C "${repo_root}" diff --quiet --ignore-submodules=dirty || \
  fail "NeMo-RL source has unstaged tracked changes"
git -C "${repo_root}" diff --cached --quiet --ignore-submodules=dirty || \
  fail "NeMo-RL source has staged changes"
[[ -z "$(git -C "${repo_root}" ls-files --others --exclude-standard)" ]] || \
  fail "NeMo-RL source has untracked files"
integration_sha=${EXPECTED_BRIDGE_SHA}
source_provenance_verifier=${script_dir}/scripts/verify_source_provenance.sh
runtime_attestation_command=${script_dir}/verify_runtime_attestation.py

snapshot=${RUN_LOG_ROOT}/source-snapshots/bridge/${BRIDGE_CANDIDATE_SHA}
mkdir -p "$(dirname "${snapshot}")"
if [[ ! -d "${snapshot}" ]]; then
  temporary_snapshot=$(mktemp -d "$(dirname "${snapshot}")/.${BRIDGE_CANDIDATE_SHA}.XXXXXX")
  git -C "${bridge_root}" archive "${BRIDGE_CANDIDATE_SHA}" | tar -x -C "${temporary_snapshot}"
  mkdir -p "${temporary_snapshot}/3rdparty/Megatron-LM"
  git -C "${mcore_root}" archive "${candidate_mcore_sha}" | \
    tar -x -C "${temporary_snapshot}/3rdparty/Megatron-LM"
  printf '%s\n' "${BRIDGE_CANDIDATE_SHA}" >"${temporary_snapshot}/.candidate-sha"
  printf '%s\n' "${candidate_mcore_sha}" >"${temporary_snapshot}/.candidate-mcore-sha"
  mv "${temporary_snapshot}" "${snapshot}"
fi

intent_dir=${RUN_LOG_ROOT}/submission-intents/bridge/${BRIDGE_CANDIDATE_SHA}
mkdir -p "${intent_dir}"
intent=${intent_dir}/$(date -u +%Y%m%dT%H%M%SZ)-$$.json
python3 - "${intent}" "${BRIDGE_CANDIDATE_SHA}" "${integration_sha}" "${candidate_mcore_sha}" "${PROFILE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps({
    "schema_version": 1, "candidate_kind": "bridge", "candidate_sha": sys.argv[2],
    "integration_sha": sys.argv[3], "candidate_mcore_sha": sys.argv[4],
    "profile_sha256": sys.argv[5], "rows": ["bridge_forward_only_eval_8"],
}, sort_keys=True) + "\n")
temporary.replace(path)
PY

exports="ALL,TEST_ROW_ID=bridge_forward_only_eval_8,TEST_WORLD_SIZE=8,TEST_NUM_NODES=2,TEST_GPUS_PER_NODE=4,CANDIDATE_KIND=bridge,CANDIDATE_SHA=${BRIDGE_CANDIDATE_SHA},INTEGRATION_SHA=${integration_sha},CANDIDATE_SOURCE_ROOT=${snapshot},RUN_LOG_ROOT=${RUN_LOG_ROOT},TEST_MATRIX=${matrix},RUNNER_PATH=${driver},CONTAINER=${CONTAINER},CONTAINER_SHA256=${CONTAINER_SHA256},MOUNTS=${MOUNTS},EXPECTED_TE_SHA=${EXPECTED_TE_SHA},EXPECTED_TE_VERSION_BASE_SHA=${EXPECTED_TE_VERSION_BASE_SHA},RUNTIME_ATTESTATION=${RUNTIME_ATTESTATION},SUBMISSION_INTENT=${intent},REPO_ROOT=${repo_root},EXPECTED_NEMORL_SHA=${EXPECTED_NEMORL_SHA},EXPECTED_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA},EXPECTED_MCORE_SHA=${EXPECTED_MCORE_SHA},SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier},RUNTIME_ATTESTATION_COMMAND=${runtime_attestation_command}"
command=(sbatch --parsable --nodes=2 "--account=${ACCOUNT}" "--partition=${PARTITION}" "--time=${TIME_LIMIT}" --job-name=bridge-forward-only-eval "--output=${RUN_LOG_ROOT}/slurm/bridge-forward-only-eval-%j.log" "--export=${exports}")
[[ "${SBATCH_GRES}" == none ]] || command+=("--gres=${SBATCH_GRES}")
[[ -z "${SBATCH_SEGMENT_SIZE}" ]] || command+=("--segment=${SBATCH_SEGMENT_SIZE}")
[[ "${SBATCH_TEST_ONLY:-0}" == 1 ]] && command+=(--test-only)
command+=("${script_dir}/scripts/run_bridge_scope.sub")
mkdir -p "${RUN_LOG_ROOT}/slurm"
output=$("${command[@]}")
printf 'ROW: bridge_forward_only_eval_8\nSBATCH_OUTPUT: %s\n' "${output}"
