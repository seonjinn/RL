#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

fail() {
  echo "$*" >&2
  exit 2
}

[[ -z "${COMMAND:-}" && -z "${MCORE_COMMAND:-}" ]] || \
  fail "Raw command payloads are forbidden by the typed matrix runner"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${script_dir}/../../.." && pwd -P)
matrix=${script_dir}/mcore_test_matrix.json
driver=${script_dir}/scripts/run_mcore_training.py
rows=${MCORE_TEST_ROWS:-}
[[ -n "${rows}" ]] || fail "MCORE_TEST_ROWS must contain one or more literal row IDs"

selection=$(python3 - "${driver}" "${matrix}" "${rows}" <<'PY'
import importlib.util
import sys
from pathlib import Path

driver_path, matrix_path, requested = sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_mcore_training", driver_path)
if spec is None or spec.loader is None:
    raise SystemExit("unable to load typed MCore driver")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
matrix = module.load_matrix(Path(matrix_path), candidate_kind="mcore")
row_ids = requested.split()
if not row_ids or len(set(row_ids)) != len(row_ids):
    raise SystemExit("test row selection must be non-empty and unique")
for row_id in row_ids:
    if row_id not in matrix:
        raise SystemExit(f"unknown test row: {row_id}")
    row = matrix[row_id]
    allocation = next(((n, g) for n, g in row.allocations if g == 4), row.allocations[0])
    print(f"{row_id}\t{row.world_size}\t{allocation[0]}\t{allocation[1]}")
PY
) || fail "Unknown test row or invalid MCore matrix"

: "${CLUSTER:?Set CLUSTER to ptyche, oci-hsg, or lyris}"
: "${PROFILE_FILE:?Set PROFILE_FILE to a validated cluster profile}"
: "${MCORE_CANDIDATE_SHA:?Set MCORE_CANDIDATE_SHA to the resolved pushed commit}"
[[ "${MCORE_CANDIDATE_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
  fail "MCORE_CANDIDATE_SHA must be one lowercase 40-character SHA"
case "${CLUSTER}" in
  ptyche|oci-hsg|lyris) ;;
  *) fail "CLUSTER must be ptyche, oci-hsg, or lyris" ;;
esac

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
for field in ACCOUNT CONTAINER CONTAINER_SHA256 MOUNTS RUNTIME_ATTESTATION EXPECTED_TE_SHA EXPECTED_TE_VERSION_BASE_SHA RUN_LOG_ROOT; do
  value=${!field:-}
  [[ -n "${value}" && "${value}" != *"__REQUIRED"* ]] || fail "Profile field ${field} is unresolved"
done

runtime_contract=$(SELECTION="${selection}" python3 - "${RUNTIME_ATTESTATION}" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
feature_set = payload.get("runtime_feature_set")
excluded = payload.get("excluded_packages")
torch_arch = payload.get("torch_cuda_arch_list")
nvte_arch = payload.get("nvte_cuda_archs")
rows = [line.split("\t", 1)[0] for line in os.environ["SELECTION"].splitlines()]
feature_exclusions = {
    "te_eval_capability_8": [
        "causal-conv1d",
        "deep-ep",
        "fast-hadamard-transform",
        "mamba-ssm",
    ],
    "dropless_hybridep_nano16": ["fast-hadamard-transform"],
    "dropless_alltoall_qwen30_16": ["deep-ep", "fast-hadamard-transform"],
    "dropless_alltoall_super32": ["deep-ep", "fast-hadamard-transform"],
    "dropless_hybridep_qwen235_64": ["fast-hadamard-transform"],
}
expected_excluded = feature_exclusions.get(feature_set)
if expected_excluded is None or rows != [feature_set]:
    raise SystemExit("runtime attestation must authorize the exact selected row")
if excluded != expected_excluded or torch_arch != "10.0a" or nvte_arch != "100a":
    raise SystemExit("runtime attestation feature contract mismatch")
print("\t".join((feature_set, ",".join(excluded), torch_arch, nvte_arch)))
PY
) || fail "Runtime feature contract rejected"
IFS=$'\t' read -r RUNTIME_FEATURE_SET RUNTIME_EXCLUDED_PACKAGES \
  TORCH_CUDA_ARCH_LIST NVTE_CUDA_ARCHS <<<"${runtime_contract}"
[[ -n "${NVTE_CUDA_ARCHS}" ]] || fail "Runtime feature contract is incomplete"

mcore_root=${repo_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
remote_sha=$(git -C "${mcore_root}" ls-remote origin refs/heads/sj/thd-cg-hybrid-nemotron-main-20260806 | awk 'NF == 2 {print $1}')
[[ "${remote_sha}" =~ ^[0-9a-f]{40}$ ]] || fail "Candidate branch did not resolve to exactly one pushed SHA"
[[ "${remote_sha}" == "${MCORE_CANDIDATE_SHA}" ]] || fail "Candidate SHA does not match the pushed remote branch"
root_branch=$(git -C "${repo_root}" branch --show-current)
[[ "${root_branch}" == experiment/thd-cg-hybrid-nemotron-main-20260806 ]] || \
  fail "NeMo-RL runner must use the HybridEP integration branch"
root_sha=$(git -C "${repo_root}" rev-parse HEAD)
remote_root_sha=$(git -C "${repo_root}" ls-remote seonjinn \
  refs/heads/experiment/thd-cg-hybrid-nemotron-main-20260806 | awk 'NF == 2 {print $1}')
[[ "${remote_root_sha}" == "${root_sha}" ]] || \
  fail "NeMo-RL runner infrastructure is not pushed at the local HEAD"
git -C "${repo_root}" diff --quiet --ignore-submodules=dirty || \
  fail "NeMo-RL source has unstaged tracked changes"
git -C "${repo_root}" diff --cached --quiet --ignore-submodules=dirty || \
  fail "NeMo-RL source has staged changes"
[[ -z "$(git -C "${repo_root}" ls-files --others --exclude-standard)" ]] || \
  fail "NeMo-RL source has untracked files"
integration_sha=${EXPECTED_MCORE_SHA}
source_provenance_verifier=${script_dir}/scripts/verify_source_provenance.sh
runtime_attestation_command=${script_dir}/verify_runtime_attestation.py

artifacts=$(SELECTION=${selection} python3 - "${driver}" "${mcore_root}" \
  "${RUN_LOG_ROOT}" "${MCORE_CANDIDATE_SHA}" "${integration_sha}" \
  "${PROFILE_SHA256}" "${RUNTIME_FEATURE_SET}" "${RUNTIME_EXCLUDED_PACKAGES}" \
  "${TORCH_CUDA_ARCH_LIST}" "${NVTE_CUDA_ARCHS}" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

driver_path, repository, run_log_root, candidate_sha, integration_sha, profile_sha256, feature_set, excluded, torch_arch, nvte_arch = sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_mcore_training", driver_path)
if spec is None or spec.loader is None:
    raise SystemExit("unable to load typed MCore driver")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
payload = {
    "schema_version": 1,
    "candidate_kind": "mcore",
    "candidate_sha": candidate_sha,
    "integration_sha": integration_sha,
    "profile_sha256": profile_sha256,
    "runtime_feature_set": feature_set,
    "excluded_packages": excluded.split(","),
    "torch_cuda_arch_list": torch_arch,
    "nvte_cuda_archs": nvte_arch,
    "rows": [line.split("\t")[0] for line in os.environ["SELECTION"].splitlines()],
}
artifacts = module.prepare_candidate_submission(
    archive_sources=((Path(repository), candidate_sha, Path(".")),),
    run_log_root=Path(run_log_root),
    candidate_kind="mcore",
    candidate_sha=candidate_sha,
    intent_payload=payload,
)
print("\t".join((str(artifacts.snapshot_root), artifacts.snapshot_sha256, str(artifacts.intent_path), artifacts.intent_sha256)))
PY
 ) || fail "Failed to publish immutable submission artifacts"
IFS=$'\t' read -r snapshot snapshot_sha256 intent intent_sha256 <<<"${artifacts}"
[[ "${snapshot_sha256}" =~ ^[0-9a-f]{64}$ && "${intent_sha256}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "Submission artifact digests are invalid"

while IFS=$'\t' read -r row_id world_size num_nodes gpus_per_node; do
  [[ "${SBATCH_GPUS_PER_NODE}" == "${gpus_per_node}" ]] || fail "Profile/allocation GPU mismatch"
  exports="ALL,TEST_ROW_ID=${row_id},TEST_WORLD_SIZE=${world_size},TEST_NUM_NODES=${num_nodes},TEST_GPUS_PER_NODE=${gpus_per_node},CANDIDATE_KIND=mcore,CANDIDATE_SHA=${MCORE_CANDIDATE_SHA},INTEGRATION_SHA=${integration_sha},CANDIDATE_SOURCE_ROOT=${snapshot},CANDIDATE_SNAPSHOT_SHA256=${snapshot_sha256},RUN_LOG_ROOT=${RUN_LOG_ROOT},TEST_MATRIX=${matrix},RUNNER_PATH=${driver},CONTAINER=${CONTAINER},CONTAINER_SHA256=${CONTAINER_SHA256},MOUNTS=${MOUNTS},EXPECTED_TE_SHA=${EXPECTED_TE_SHA},EXPECTED_TE_VERSION_BASE_SHA=${EXPECTED_TE_VERSION_BASE_SHA},RUNTIME_ATTESTATION=${RUNTIME_ATTESTATION},SUBMISSION_INTENT=${intent},SUBMISSION_INTENT_SHA256=${intent_sha256},REPO_ROOT=${repo_root},EXPECTED_NEMORL_SHA=${EXPECTED_NEMORL_SHA},EXPECTED_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA},EXPECTED_MCORE_SHA=${EXPECTED_MCORE_SHA},SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier},RUNTIME_ATTESTATION_COMMAND=${runtime_attestation_command},RUNTIME_FEATURE_SET=${RUNTIME_FEATURE_SET},RUNTIME_EXCLUDED_PACKAGES=${RUNTIME_EXCLUDED_PACKAGES},TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST},NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"
  command=(sbatch --parsable "--nodes=${num_nodes}" "--account=${ACCOUNT}" "--partition=${PARTITION}" "--time=${TIME_LIMIT}" "--job-name=mcore-${row_id}" "--output=${RUN_LOG_ROOT}/slurm/mcore-${row_id}-%j.log" "--export=${exports}")
  [[ "${SBATCH_GRES}" == none ]] || command+=("--gres=${SBATCH_GRES}")
  [[ -z "${SBATCH_SEGMENT_SIZE}" ]] || command+=("--segment=${SBATCH_SEGMENT_SIZE}")
  [[ "${SBATCH_TEST_ONLY:-0}" == 1 ]] && command+=(--test-only)
  command+=("${script_dir}/scripts/run_mcore_scope.sub")
  mkdir -p "${RUN_LOG_ROOT}/slurm"
  output=$("${command[@]}")
  printf 'ROW: %s\nSBATCH_OUTPUT: %s\n' "${row_id}" "${output}"
done <<<"${selection}"
