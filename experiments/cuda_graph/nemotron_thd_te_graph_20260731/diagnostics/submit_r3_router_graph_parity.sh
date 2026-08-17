#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

fail() {
  echo "$*" >&2
  exit 2
}

run_sbatch_without_ambient_state() {
  local -a clean=(env)
  local name
  while IFS= read -r name; do
    case "${name}" in
      SBATCH_*|NRL_*) clean+=(-u "${name}") ;;
    esac
  done < <(compgen -e)
  "${clean[@]}" "$@"
}

load_profile_snapshot() {
  local helper output field value
  local -a command
  helper=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/profile_snapshot.py
  [[ -f "${helper}" && ! -L "${helper}" ]] || fail "Canonical profile snapshot helper is missing or unsafe"
  command=(
    python3 "${helper}"
    --profile-dir "$(dirname "${PROFILE_FILE}")"
    --cluster oci-hsg
    --profile-file "${PROFILE_FILE}"
  )
  if [[ -n "${VALIDATED_PROFILE_SHA256:-}" ]]; then
    [[ "${VALIDATED_PROFILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
      fail "VALIDATED_PROFILE_SHA256 must be a full lowercase SHA256"
    command+=(--expected-sha256 "${VALIDATED_PROFILE_SHA256}")
  fi
  output=$("${command[@]}") || fail "Canonical profile snapshot validation failed"
  PROFILE_SHA256=
  while IFS=$'\t' read -r field value; do
    case "${field}" in
      PROFILE_SHA256|PROFILE_ID|ACCOUNT|PARTITION|CONTAINER|CONTAINER_SHA256|MOUNTS|SBATCH_GPUS_PER_NODE|SBATCH_GRES|SBATCH_SEGMENT_SIZE|TIME_LIMIT|RUNTIME_ATTESTATION|RUNTIME_PREFLIGHT_JOB_ID|UV_EXECUTABLE|EXPECTED_TE_SHA|EXPECTED_TE_VERSION_BASE_SHA|EXPECTED_NEMORL_SHA|EXPECTED_BRIDGE_SHA|EXPECTED_MCORE_SHA|RUN_LOG_ROOT)
        printf -v "${field}" '%s' "${value}"
        ;;
      *) fail "Canonical profile snapshot returned an unknown field" ;;
    esac
  done <<<"${output}"
  [[ "${PROFILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "Profile snapshot omitted its digest"
}

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the exact NeMo-RL checkout}"
: "${PROFILE_FILE:?Set PROFILE_FILE to the exact OCI-HSG profile}"
: "${FROZEN_BATCH:?Set FROZEN_BATCH to train_data_step*.jsonl}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to an absolute immutable output directory}"
: "${CONFIG:?Set CONFIG to the exact GRPO recipe}"
: "${HF_HOME:?Set HF_HOME to the populated offline Hugging Face cache}"

TEST_ONLY=${TEST_ONLY:-0}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-0}
case "${TEST_ONLY}:${SBATCH_TEST_ONLY}" in
  0:0|0:1|1:0) ;;
  1:1) fail "TEST_ONLY and SBATCH_TEST_ONLY are mutually exclusive" ;;
  *) fail "TEST_ONLY and SBATCH_TEST_ONLY must be 0 or 1" ;;
esac
for path in "${PROJECT_ROOT}" "${PROFILE_FILE}" "${FROZEN_BATCH}" \
  "${ARTIFACT_DIR}" "${CONFIG}" "${HF_HOME}"; do
  [[ "${path}" == /* ]] || fail "All parity paths must be absolute: ${path}"
done
[[ -f "${PROFILE_FILE}" && ! -L "${PROFILE_FILE}" ]] || fail "Profile is missing or unsafe"
load_profile_snapshot

[[ "${PROFILE_ID}" == oci-hsg ]] || fail "R3 parity requires PROFILE_ID=oci-hsg"
[[ "${SBATCH_GPUS_PER_NODE}" == 4 && "${SBATCH_GRES}" == gpu:4 ]] || \
  fail "R3 parity requires exactly 4 GPUs per node"
for required in ACCOUNT PARTITION CONTAINER CONTAINER_SHA256 MOUNTS TIME_LIMIT \
  RUNTIME_ATTESTATION RUNTIME_PREFLIGHT_JOB_ID UV_EXECUTABLE EXPECTED_TE_SHA \
  EXPECTED_TE_VERSION_BASE_SHA EXPECTED_NEMORL_SHA EXPECTED_BRIDGE_SHA \
  EXPECTED_MCORE_SHA; do
  [[ -n "${!required:-}" ]] || fail "Profile field ${required} is required"
done
runtime_stage_root=${UV_EXECUTABLE%/uv/uv}
[[ "${runtime_stage_root}" != "${UV_EXECUTABLE}" ]] || \
  fail "UV_EXECUTABLE must use staged-runtimes/<sha256>/uv/uv"
canonical_runtime_python=${runtime_stage_root}/environment/bin/python
if [[ -n "${RUNTIME_PYTHON:-}" && "${RUNTIME_PYTHON}" != "${canonical_runtime_python}" ]]; then
  fail "RUNTIME_PYTHON must equal the canonical staged environment Python"
fi
RUNTIME_PYTHON=${canonical_runtime_python}
VLLM_RUNTIME_PYTHON=${runtime_stage_root}/vllm-environment/bin/python
for commit in "${EXPECTED_TE_SHA}" "${EXPECTED_TE_VERSION_BASE_SHA}" \
  "${EXPECTED_NEMORL_SHA}" "${EXPECTED_BRIDGE_SHA}" "${EXPECTED_MCORE_SHA}"; do
  [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "Runtime commits must be full lowercase SHAs"
done
[[ "${CONTAINER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "Container SHA256 is invalid"
[[ "${RUNTIME_PREFLIGHT_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || fail "Runtime attestation job ID is invalid"
[[ "${FROZEN_BATCH}" =~ /train_data_step[0-9]+\.jsonl$ ]] || fail "Frozen source name is invalid"

actual_sha=$(git -C "${PROJECT_ROOT}" rev-parse HEAD) || fail "Cannot read source SHA"
[[ "${actual_sha}" == "${EXPECTED_NEMORL_SHA}" ]] || \
  fail "NeMo-RL source SHA mismatch: expected ${EXPECTED_NEMORL_SHA}, got ${actual_sha}"
frozen_sha=$(python3 - "${FROZEN_BATCH}" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("Frozen JSONL must be a regular non-symlink")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
) || fail "Cannot hash frozen JSONL"

runtime_contract=$(python3 - "${RUNTIME_ATTESTATION}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("Runtime attestation must be a regular non-symlink")
payload = json.loads(path.read_text())
if payload.get("runtime_feature_set") != "dropless_hybridep_nano16_r3_router_graph_v1":
    raise SystemExit("runtime feature mismatch")
capability = (payload.get("mcore_capabilities") or {}).get("router_replay_cuda_graph_input")
if capability != "r3_router_cuda_graph_input_v1":
    raise SystemExit("runtime capability mismatch")
print(payload["runtime_feature_set"] + ":" + capability)
PY
) || fail "Runtime attestation runtime feature/capability check failed"
[[ "${runtime_contract}" == dropless_hybridep_nano16_r3_router_graph_v1:r3_router_cuda_graph_input_v1 ]] || \
  fail "Runtime attestation contract is not exact"

verifier=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/verify_runtime_attestation.py
attestation_command=(
  python3 "${verifier}"
  --attestation "${RUNTIME_ATTESTATION}"
  --container "${CONTAINER}"
  --expected-container-sha256 "${CONTAINER_SHA256}"
  --nemo-rl-commit "${EXPECTED_NEMORL_SHA}"
  --bridge-commit "${EXPECTED_BRIDGE_SHA}"
  --mcore-commit "${EXPECTED_MCORE_SHA}"
  --uv-lock "${PROJECT_ROOT}/uv.lock"
  --expected-te-commit "${EXPECTED_TE_SHA}"
  --expected-te-version-base-commit "${EXPECTED_TE_VERSION_BASE_SHA}"
  --expected-device-count 4
  --expected-python-version 3.13.14
  --expected-python-install-dir "$(dirname "${RUNTIME_ATTESTATION}")/uv-python-installations"
  --expected-uv-version 0.11.28
  --expected-uv-executable "${UV_EXECUTABLE}"
  --expected-nvte-with-nccl-ep 0
  --expected-runtime-attestation-job-id "${RUNTIME_PREFLIGHT_JOB_ID}"
  --runtime-feature-set dropless_hybridep_nano16_r3_router_graph_v1
  --excluded-packages fast-hadamard-transform
  --torch-cuda-arch-list 10.0a
  --nvte-cuda-archs 100a
)
[[ -f "${verifier}" && ! -L "${verifier}" ]] || fail "Canonical runtime verifier is missing or unsafe"
NVTE_WITH_NCCL_EP=0 "${attestation_command[@]}" >/dev/null
job_name=${ACCOUNT}-r3-router-parity
contract="profile_sha256=${PROFILE_SHA256} dropless_hybridep_nano16_r3_router_graph_v1 r3_router_cuda_graph_input_v1 NRL_ROUTER_REPLAY_VALIDATE=1 ++policy.router_replay.enabled=true ++policy.generation.vllm_kwargs.moe_backend=triton policy.precision=bfloat16 policy.megatron_cfg.moe_token_dispatcher_type=flex ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep ++policy.megatron_cfg.thd_max_packed_sequences=16 ++policy.megatron_cfg.cuda_graph_modules=[moe_router] cluster.num_nodes=4 cluster.gpus_per_node=4"
driver=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_r3_router_graph_parity.py
run_nemorl_scope=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/run_nemorl_scope.sub
source_verifier=${PROJECT_ROOT}/experiments/cuda_graph/nemotron_thd_te_graph_20260731/scripts/verify_source_provenance.sh
output=${ARTIFACT_DIR}/r3-router-graph-parity-${frozen_sha}.json
driver_argv=(
  env
  NRL_R3_PARITY_EXPORT=0
  NRL_ROUTER_REPLAY_VALIDATE=1
  "${RUNTIME_PYTHON}"
  "${driver}"
  --config "${CONFIG}"
  --frozen-batch "${FROZEN_BATCH}"
  --expected-source-sha "${frozen_sha}"
  --runtime-attestation "${RUNTIME_ATTESTATION}"
  --profile-sha256 "${PROFILE_SHA256}"
  --output "${output}"
  ++policy.router_replay.enabled=true
  ++policy.generation.vllm_kwargs.moe_backend=triton
  policy.precision=bfloat16
  policy.megatron_cfg.moe_token_dispatcher_type=flex
  ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
  ++policy.megatron_cfg.thd_max_packed_sequences=16
  '++policy.megatron_cfg.cuda_graph_modules=[moe_router]'
  ++policy.megatron_cfg.cuda_graph_impl=transformer_engine
  loss_fn.reference_policy_kl_penalty=0.0
  grpo.skip_reference_policy_logprobs_calculation=true
  cluster.num_nodes=4
  cluster.gpus_per_node=4
)
printf -v driver_command '%q ' "${driver_argv[@]}"
driver_command=${driver_command% }
printf -v attestation_command_string '%q ' "${attestation_command[@]}"
attestation_command_string=${attestation_command_string% }
pinned_uv_version=$(sed -nE 's/^ARG UV_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$/\1/p' "${PROJECT_ROOT}/docker/Dockerfile")
managed_python_version=$(tr -d '[:space:]' <"${PROJECT_ROOT}/.python-version")
wrapper_argv=(
  env
  "R3_PARITY_CONTRACT=${contract}"
  "COMMAND=${driver_command}"
  "CONTAINER=${CONTAINER}"
  "CONTAINER_SHA256=${CONTAINER_SHA256}"
  "MOUNTS=${MOUNTS}"
  "RUNTIME_ATTESTATION_COMMAND=${attestation_command_string}"
  "REPO_ROOT=${PROJECT_ROOT}"
  "EXPECTED_NEMORL_SHA=${EXPECTED_NEMORL_SHA}"
  "EXPECTED_BRIDGE_SHA=${EXPECTED_BRIDGE_SHA}"
  "EXPECTED_MCORE_SHA=${EXPECTED_MCORE_SHA}"
  "SOURCE_PROVENANCE_VERIFIER=${source_verifier}"
  "PINNED_UV_VERSION=${pinned_uv_version}"
  "UV_EXECUTABLE=${UV_EXECUTABLE}"
  "RUNTIME_PYTHON=${RUNTIME_PYTHON}"
  "UV_PYTHON=${managed_python_version}"
  "UV_PYTHON_INSTALL_DIR=$(dirname "${RUNTIME_ATTESTATION}")/uv-python-installations"
  UV_MANAGED_PYTHON=1
  UV_PYTHON_DOWNLOADS=never
  NVTE_WITH_NCCL_EP=0
  "BASE_LOG_DIR=${ARTIFACT_DIR}/ray"
  "NRL_MEGATRON_CHECKPOINT_DIR=$(dirname "${RUNTIME_ATTESTATION}")/megatron-checkpoints"
  "NEMO_RL_MCORE_PY_EXECUTABLE=${RUNTIME_PYTHON}"
  "NEMO_RL_VLLM_PY_EXECUTABLE=${VLLM_RUNTIME_PYTHON}"
  "HF_HOME=${HF_HOME}"
  "HF_HUB_CACHE=${HF_HOME}/hub"
  "HF_MODULES_CACHE=${HF_HOME}/modules"
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  HF_HUB_DISABLE_TELEMETRY=1
  GPUS_PER_NODE=4
  bash "${run_nemorl_scope}"
)
printf -v wrapper_command '%q ' "${wrapper_argv[@]}"
wrapper_command=${wrapper_command% }
payload=(
  sbatch
  --parsable
  "--partition=${PARTITION}"
  "--account=${ACCOUNT}"
  --nodes=4
  --gres=gpu:4
  "--time=${TIME_LIMIT}"
  "--job-name=${job_name}"
  "--comment=${contract}"
  "--output=${ARTIFACT_DIR}/r3-router-graph-parity-%j.log"
  --export=NONE
  "--chdir=${PROJECT_ROOT}"
  "--wrap=${wrapper_command}"
)
printf 'PAYLOAD:'
printf ' %q' "${payload[@]}"
printf '\n'
printf 'CONTRACT: %s\n' "${contract}"

if [[ "${TEST_ONLY}" == 1 ]]; then
  echo "TEST_ONLY: no submission performed"
  exit 0
fi
actual_payload=("${payload[@]}")
if [[ "${SBATCH_TEST_ONLY}" == 1 ]]; then
  actual_payload=(sbatch --test-only "${payload[@]:2}")
else
  mkdir -p "${ARTIFACT_DIR}"
fi
run_sbatch_without_ambient_state "${actual_payload[@]}"
