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
slurm_segment_helper=${script_dir}/slurm_segment.py
[[ -f "${slurm_segment_helper}" ]] || fail "Missing SLURM segment resolver"

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

runtime_contract=$(python3 - "${RUNTIME_ATTESTATION}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
excluded = payload.get("excluded_packages")
expected_excluded = [
    "causal-conv1d", "deep-ep", "fast-hadamard-transform", "mamba-ssm",
]
if payload.get("runtime_feature_set") != "bridge_forward_only_eval_8":
    raise SystemExit("runtime attestation does not authorize the Bridge row")
if excluded != expected_excluded or payload.get("torch_cuda_arch_list") != "10.0a" or payload.get("nvte_cuda_archs") != "100a":
    raise SystemExit("runtime attestation feature contract mismatch")
print("\t".join(("bridge_forward_only_eval_8", ",".join(excluded), "10.0a", "100a")))
PY
) || fail "Runtime feature contract rejected"
IFS=$'\t' read -r RUNTIME_FEATURE_SET RUNTIME_EXCLUDED_PACKAGES \
  TORCH_CUDA_ARCH_LIST NVTE_CUDA_ARCHS <<<"${runtime_contract}"
[[ -n "${NVTE_CUDA_ARCHS}" ]] || fail "Runtime feature contract is incomplete"
segment_size=$(python3 "${slurm_segment_helper}" \
  --cluster "${CLUSTER}" --num-nodes 2 \
  --configured "${SBATCH_SEGMENT_SIZE}") || fail "SLURM segment resolution failed"

remote_sha=$(git -C "${bridge_root}" ls-remote fork refs/heads/sna/thd-cg-hybrid-nemotron-main-20260806 | awk 'NF == 2 {print $1}')
[[ "${remote_sha}" =~ ^[0-9a-f]{40}$ ]] || fail "Bridge branch did not resolve to exactly one pushed SHA"
[[ "${remote_sha}" == "${BRIDGE_CANDIDATE_SHA}" ]] || fail "Bridge candidate is absent from the pushed remote branch"
candidate_mcore_sha=$(git -C "${bridge_root}" ls-tree "${BRIDGE_CANDIDATE_SHA}" 3rdparty/Megatron-LM | awk '$2 == "commit" {print $3}')
[[ "${candidate_mcore_sha}" =~ ^[0-9a-f]{40}$ ]] || fail "Bridge candidate lacks one nested MCore gitlink"
git -C "${mcore_root}" cat-file -e "${candidate_mcore_sha}^{commit}" || \
  fail "Bridge candidate nested MCore commit is unavailable"
root_branch=$(git -C "${repo_root}" branch --show-current)
[[ "${root_branch}" == experiment/thd-cg-hybrid-nemotron-main-20260806 ]] || \
  fail "NeMo-RL runner must use the Task 2 infrastructure branch"
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
integration_sha=${EXPECTED_BRIDGE_SHA}
source_provenance_verifier=${script_dir}/scripts/verify_source_provenance.sh
runtime_attestation_command=${script_dir}/verify_runtime_attestation.py

artifacts=$(python3 - "${driver}" "${bridge_root}" "${mcore_root}" \
  "${RUN_LOG_ROOT}" "${BRIDGE_CANDIDATE_SHA}" "${integration_sha}" \
  "${candidate_mcore_sha}" "${PROFILE_SHA256}" "${RUNTIME_FEATURE_SET}" \
  "${RUNTIME_EXCLUDED_PACKAGES}" "${TORCH_CUDA_ARCH_LIST}" "${NVTE_CUDA_ARCHS}" <<'PY'
import importlib.util
import sys
from pathlib import Path

driver_path, bridge_root, mcore_root, run_log_root, candidate_sha, integration_sha, candidate_mcore_sha, profile_sha256, feature_set, excluded, torch_arch, nvte_arch = sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_mcore_training", driver_path)
if spec is None or spec.loader is None:
    raise SystemExit("unable to load typed MCore driver")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
payload = {
    "schema_version": 1, "candidate_kind": "bridge", "candidate_sha": candidate_sha,
    "integration_sha": integration_sha, "candidate_mcore_sha": candidate_mcore_sha,
    "profile_sha256": profile_sha256, "runtime_feature_set": feature_set,
    "excluded_packages": excluded.split(","), "torch_cuda_arch_list": torch_arch,
    "nvte_cuda_archs": nvte_arch, "rows": ["bridge_forward_only_eval_8"],
}
artifacts = module.prepare_candidate_submission(
    archive_sources=(
        (Path(bridge_root), candidate_sha, Path(".")),
        (Path(mcore_root), candidate_mcore_sha, Path("3rdparty/Megatron-LM")),
    ),
    run_log_root=Path(run_log_root), candidate_kind="bridge",
    candidate_sha=candidate_sha, intent_payload=payload,
)
print("\t".join((str(artifacts.snapshot_root), artifacts.snapshot_sha256, str(artifacts.intent_path), artifacts.intent_sha256)))
PY
 ) || fail "Failed to publish immutable submission artifacts"
IFS=$'\t' read -r snapshot snapshot_sha256 intent intent_sha256 <<<"${artifacts}"
[[ "${snapshot_sha256}" =~ ^[0-9a-f]{64}$ && "${intent_sha256}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "Submission artifact digests are invalid"

exports="ALL,TEST_ROW_ID=bridge_forward_only_eval_8,TEST_WORLD_SIZE=8,TEST_NUM_NODES=2,TEST_GPUS_PER_NODE=4,CANDIDATE_KIND=bridge,CANDIDATE_SHA=${BRIDGE_CANDIDATE_SHA},INTEGRATION_SHA=${integration_sha},CANDIDATE_SOURCE_ROOT=${snapshot},CANDIDATE_SNAPSHOT_SHA256=${snapshot_sha256},RUN_LOG_ROOT=${RUN_LOG_ROOT},TEST_MATRIX=${matrix},RUNNER_PATH=${driver},CONTAINER=${CONTAINER},CONTAINER_SHA256=${CONTAINER_SHA256},MOUNTS=${MOUNTS},EXPECTED_TE_SHA=${EXPECTED_TE_SHA},EXPECTED_TE_VERSION_BASE_SHA=${EXPECTED_TE_VERSION_BASE_SHA},RUNTIME_ATTESTATION=${RUNTIME_ATTESTATION},SUBMISSION_INTENT=${intent},SUBMISSION_INTENT_SHA256=${intent_sha256},REPO_ROOT=${repo_root},EXPECTED_NEMORL_SHA=${EXPECTED_NEMORL_SHA},EXPECTED_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA},EXPECTED_MCORE_SHA=${EXPECTED_MCORE_SHA},SOURCE_PROVENANCE_VERIFIER=${source_provenance_verifier},RUNTIME_ATTESTATION_COMMAND=${runtime_attestation_command},RUNTIME_FEATURE_SET=${RUNTIME_FEATURE_SET},RUNTIME_EXCLUDED_PACKAGES=${RUNTIME_EXCLUDED_PACKAGES},TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST},NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"
command=(sbatch --parsable --nodes=2 "--account=${ACCOUNT}" "--partition=${PARTITION}" "--time=${TIME_LIMIT}" --job-name=bridge-forward-only-eval "--output=${RUN_LOG_ROOT}/slurm/bridge-forward-only-eval-%j.log" "--export=${exports}")
if [[ "${SBATCH_GRES}" == none ]]; then
  command+=("--gpus-per-node=${SBATCH_GPUS_PER_NODE}")
else
  command+=("--gres=${SBATCH_GRES}")
fi
[[ -z "${segment_size}" ]] || command+=("--segment=${segment_size}")
[[ "${SBATCH_TEST_ONLY:-0}" == 1 ]] && command+=(--test-only)
command+=("${script_dir}/scripts/run_bridge_scope.sub")
mkdir -p "${RUN_LOG_ROOT}/slurm"
output=$("${command[@]}")
printf 'ROW: bridge_forward_only_eval_8\nSBATCH_OUTPUT: %s\n' "${output}"
