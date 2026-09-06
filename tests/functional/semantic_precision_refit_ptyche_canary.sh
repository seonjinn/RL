#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
readonly NEMO_RL_PYTHON="${NEMO_RL_PYTHON:-/opt/nemo_rl_venv/bin/python}"
readonly DEFAULT_VLLM_PYTHON="/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
readonly MEGATRON_PYTHON="${MEGATRON_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
readonly KNOWN_VLLM_PYTHONS=(
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
)
readonly SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/source-input/source.tar}"
readonly SOURCE_PROVENANCE_MANIFEST="${SOURCE_PROVENANCE_MANIFEST:-/source-input/source-provenance.prepare.manifest}"
readonly CANARY_WORK_ROOT="${CANARY_WORK_ROOT:-}"

readonly PRECISION_TESTS=(tests/unit/precision_policy)
readonly REFIT_PLAN_TESTS=(tests/unit/weight_sync/test_refit_plan.py)
readonly REFIT_RUNTIME_TESTS=(
    tests/unit/weight_sync/test_refit_supervisor.py
    tests/unit/weight_sync/test_collective_refit_supervision.py
    tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py
)
readonly GENERATION_LIFECYCLE_TESTS=(
    tests/unit/models/generation/test_lifecycle_ack_contract.py
    tests/unit/models/generation/test_prepare_refit_info_deadline_contract.py
)
readonly GRPO_FAIL_FAST_TESTS=(tests/unit/algorithms/test_grpo_refit_supervision.py)
readonly GYM_STARTUP_FAIL_FAST_TESTS=(
    tests/unit/environments/test_nemo_gym_utils.py::test_start_nemo_gym_actor_returns_pending_without_waiting
    tests/unit/environments/test_nemo_gym_utils.py::test_start_nemo_gym_actor_submission_failure_kills_actor_and_preserves_error
    tests/unit/environments/test_nemo_gym_utils.py::test_finish_nemo_gym_actor_waits_then_installs_tokenizer
    tests/unit/environments/test_nemo_gym_utils.py::test_abort_nemo_gym_actor_kills_without_queued_shutdown
    tests/unit/environments/test_nemo_gym_utils.py::test_concurrent_nemo_gym_startup_abort_never_waits_and_kills_late_actor
    tests/unit/environments/test_nemo_gym_utils.py::test_concurrent_nemo_gym_startup_wait_returns_published_pending_actor
    tests/unit/environments/test_nemo_gym_utils.py::test_concurrent_nemo_gym_startup_abort_does_not_wait_for_actor_kill
    tests/unit/environments/test_nemo_gym_utils.py::test_compatibility_spinup_failure_aborts_actor_and_preserves_error
    tests/unit/algorithms/test_grpo.py::test_setup_refits_noncolocated_megatron_while_nemo_gym_waits
    tests/unit/algorithms/test_grpo.py::test_setup_refit_failure_aborts_pending_nemo_gym_without_waiting
    tests/unit/algorithms/test_distillation.py::test_distillation_setup_failure_aborts_pending_nemo_gym_without_waiting
    tests/unit/single_controller/test_setup.py::TestSetup::test_megatron_setup
    tests/unit/single_controller/test_setup.py::TestSetup::test_refit_failure_aborts_pending_nemo_gym_without_waiting
)
readonly TOKEN_REFIT_FAIL_FAST_TESTS=(
    tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_broadcasts_native_refit_pause_and_resume
    tests/unit/models/generation/test_vllm_generation.py::test_vllm_generation_rejects_partial_refit_pause_and_resume
    tests/unit/models/generation/test_vllm_generation.py::test_async_vllm_worker_propagates_prefix_reset_acknowledgement
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_worker_rejects_non_exact_or_negative_weight_versions
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_generation_stamps_only_surviving_refit_leaders
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_generation_stamp_requires_exact_true_ack_from_every_survivor
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_unknown_stamp_outcome_poison_prevents_retry_and_preserves_cause
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_first_remote_failure_is_typed_not_dispatched_and_retry_safe
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_later_remote_failure_poison_preserves_partial_dispatch_cause
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_poisoned_stamp_rejects_a_later_refit_before_dispatch
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_cache_invalidation_is_bounded_exact_and_poisoning
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_refit_pause_resume_are_bounded_and_use_only_survivor_leaders
    tests/unit/models/generation/test_vllm_refit_lifecycle.py::test_refit_pause_requires_exact_ack_and_poison_blocks_resume
    tests/unit/single_controller/test_refit_recovery.py::TestRefitCommitPhase
    tests/unit/single_controller/test_setup.py::TestSetup::test_token_capture_initial_stamp_uses_restored_trainer_version_and_deadline
    tests/unit/single_controller/test_single_controller_actor.py::test_sync_weights_honors_recompute_kv_cache_config
    tests/unit/single_controller/test_single_controller_actor.py::test_sync_weights_calibrates_and_forwards_fp8_kv_scales
    tests/unit/weight_sync/test_reshard_rebuild.py::TestRefitDispatchExcludesTheDeadShard::test_refit_commit_fanouts_skip_the_dead_shard
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_blocking_refit_pause_is_bounded_poisoned_and_never_retried
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_not_dispatched_pause_is_retry_safe_at_collector_boundary
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_malformed_refit_pause_ack_poison_blocks_lifecycle_retry
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_blocking_refit_resume_is_bounded_poisoned_and_never_retried
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_vllm_cache_failure_is_fatal_and_keeps_refit_gate_closed
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_supported_cache_malformed_ack_is_fatal_and_keeps_refit_gate_closed
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_unsupported_cache_noop_warns_and_reopens_refit_gate
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_pause_resume_capability_requires_exact_bool
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_pause_resume_capability_requires_paired_method_overrides
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_legacy_paired_pause_resume_omits_timeout_keyword
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_pause_resume_timeout_capability_requires_exact_bool
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_pause_resume_timeout_capability_requires_pause_support
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_cache_without_backend_timeout_is_outer_bounded_and_poisoned
    tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_dynamo_cache_invalidation_failure_is_fatal_and_keeps_refit_gate_closed
)

TEMP_DIR=""
VLLM_PYTHON="${VLLM_PYTHON:-}"

die() {
    printf 'CANARY ERROR: %s\n' "$*" >&2
    exit 1
}

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

emit_test_inventory() {
    printf '%s\n' \
        "${PRECISION_TESTS[@]}" \
        "${REFIT_PLAN_TESTS[@]}" \
        "${REFIT_RUNTIME_TESTS[@]}" \
        "${GENERATION_LIFECYCLE_TESTS[@]}" \
        "${GRPO_FAIL_FAST_TESTS[@]}" \
        "${GYM_STARTUP_FAIL_FAST_TESTS[@]}" \
        "${TOKEN_REFIT_FAIL_FAST_TESTS[@]}"
}

check_provenance_binding() {
    local actual_archive_sha actual_inventory_sha actual_manifest_sha

    [[ "${EXPECTED_REPO_SHA:-}" =~ ^[0-9a-f]{40}$ ]] \
        || die 'EXPECTED_REPO_SHA must be a full 40-character lowercase commit SHA'
    [[ "${EXPECTED_ARCHIVE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
        || die 'EXPECTED_ARCHIVE_SHA256 must be a lowercase SHA-256 digest'
    [[ "${EXPECTED_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
        || die 'EXPECTED_MANIFEST_SHA256 must be a lowercase SHA-256 digest'
    [[ "${EXPECTED_TEST_INVENTORY_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
        || die 'EXPECTED_TEST_INVENTORY_SHA256 must be a lowercase SHA-256 digest'
    [[ -r "$SOURCE_ARCHIVE" ]] || die "source archive is unreadable: ${SOURCE_ARCHIVE}"
    [[ -r "$SOURCE_PROVENANCE_MANIFEST" ]] \
        || die "source provenance manifest is unreadable: ${SOURCE_PROVENANCE_MANIFEST}"

    actual_archive_sha="$(sha256_file "$SOURCE_ARCHIVE")"
    actual_manifest_sha="$(sha256_file "$SOURCE_PROVENANCE_MANIFEST")"
    actual_inventory_sha="$(emit_test_inventory | sha256sum | awk '{print $1}')"
    [[ "$actual_archive_sha" == "$EXPECTED_ARCHIVE_SHA256" ]] \
        || die "source archive digest changed: ${actual_archive_sha}"
    [[ "$actual_manifest_sha" == "$EXPECTED_MANIFEST_SHA256" ]] \
        || die "source provenance manifest digest changed: ${actual_manifest_sha}"
    [[ "$actual_inventory_sha" == "$EXPECTED_TEST_INVENTORY_SHA256" ]] \
        || die "archived canary test inventory changed: ${actual_inventory_sha}"
    grep -Fqx -- "revision=${EXPECTED_REPO_SHA}" "$SOURCE_PROVENANCE_MANIFEST" \
        || die 'source provenance manifest revision does not match EXPECTED_REPO_SHA'
    grep -Fqx -- "archive_sha256=${EXPECTED_ARCHIVE_SHA256}" "$SOURCE_PROVENANCE_MANIFEST" \
        || die 'source provenance manifest archive digest does not match'
    grep -Fqx -- "test_inventory_sha256=${EXPECTED_TEST_INVENTORY_SHA256}" "$SOURCE_PROVENANCE_MANIFEST" \
        || die 'source provenance manifest test inventory digest does not match'

    printf 'revision_sha=%s\narchive_sha256=%s\nmanifest_sha256=%s\ntest_inventory_sha256=%s\n' \
        "$EXPECTED_REPO_SHA" \
        "$actual_archive_sha" \
        "$actual_manifest_sha" \
        "$actual_inventory_sha"
}

check_python_executable() {
    local label=$1
    local python_path=$2
    if [[ "$python_path" == */* ]]; then
        [[ -x "$python_path" ]] || die "${label} is not executable: ${python_path}"
    else
        command -v "$python_path" >/dev/null 2>&1 \
            || die "${label} is not on PATH: ${python_path}"
    fi
    run_command timeout --signal=TERM --kill-after=10s 45s "$python_path" --version
}

select_vllm_python() {
    if [[ -z "${VLLM_PYTHON:-}" ]]; then
        VLLM_PYTHON="$DEFAULT_VLLM_PYTHON"
        for candidate in "${KNOWN_VLLM_PYTHONS[@]}"; do
            if [[ -x "$candidate" ]]; then
                VLLM_PYTHON="$candidate"
                break
            fi
        done
    fi
    check_python_executable VLLM_PYTHON "$VLLM_PYTHON"
}

cleanup() {
    local status=$?
    local actual_archive_sha=''
    local actual_manifest_sha=''
    local post_failure=0

    set +e
    actual_archive_sha="$(sha256_file "$SOURCE_ARCHIVE" 2>/dev/null)" || post_failure=1
    actual_manifest_sha="$(sha256_file "$SOURCE_PROVENANCE_MANIFEST" 2>/dev/null)" \
        || post_failure=1
    if [[ "$actual_archive_sha" != "${EXPECTED_ARCHIVE_SHA256:-}" ]]; then
        printf 'CANARY ERROR: source archive changed during canary: %s\n' "$actual_archive_sha" >&2
        post_failure=1
    fi
    if [[ "$actual_manifest_sha" != "${EXPECTED_MANIFEST_SHA256:-}" ]]; then
        printf 'CANARY ERROR: provenance manifest changed during canary: %s\n' "$actual_manifest_sha" >&2
        post_failure=1
    fi
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        if [[ "$CANARY_WORK_ROOT" =~ ^/raid/scratch/spr\.[0-9]+/[0-9a-f]{40}$ && "$TEMP_DIR" == "${CANARY_WORK_ROOT}/runtime" ]]; then
            if ! rm -rf -- "$TEMP_DIR" || [[ -e "$TEMP_DIR" ]]; then
                printf 'CANARY ERROR: failed to remove temp directory: %s\n' "$TEMP_DIR" >&2
                post_failure=1
            fi
        else
            printf 'CANARY ERROR: refusing to remove unexpected temp path: %s\n' "$TEMP_DIR" >&2
            post_failure=1
        fi
    fi

    if [[ "$status" -eq 0 && "$post_failure" -ne 0 ]]; then
        status=1
    fi
    exit "$status"
}

command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is unavailable'
command -v timeout >/dev/null 2>&1 || die 'GNU timeout is unavailable'
[[ "$CANARY_WORK_ROOT" =~ ^/raid/scratch/spr\.[0-9]+/[0-9a-f]{40}$ ]] \
    || die "CANARY_WORK_ROOT is outside the allowed scratch namespace: ${CANARY_WORK_ROOT}"
[[ "${CANARY_WORK_ROOT##*/}" == "${EXPECTED_REPO_SHA:-}" ]] \
    || die 'CANARY_WORK_ROOT is not bound to EXPECTED_REPO_SHA'
[[ "$REPO_ROOT" == "${CANARY_WORK_ROOT}/source" ]] \
    || die "canary must run from the wrapper-extracted source: ${REPO_ROOT}"

check_provenance_binding
check_python_executable NEMO_RL_PYTHON "$NEMO_RL_PYTHON"
select_vllm_python
check_python_executable MEGATRON_PYTHON "$MEGATRON_PYTHON"

TEMP_DIR="${CANARY_WORK_ROOT}/runtime"
[[ ! -e "$TEMP_DIR" ]] || die "runtime temp path already exists: ${TEMP_DIR}"
mkdir -m 700 -- "$TEMP_DIR"
trap cleanup EXIT

readonly SHORT_TMP_DIR="${CANARY_WORK_ROOT%/*}/t"
mkdir -p -- \
    "$SHORT_TMP_DIR" \
    "$TEMP_DIR/pycache" \
    "$TEMP_DIR/xdg-cache" \
    "$TEMP_DIR/uv-cache" \
    "$TEMP_DIR/triton-cache" \
    "$TEMP_DIR/torchinductor-cache" \
    "$TEMP_DIR/torch-extensions" \
    "$TEMP_DIR/huggingface" \
    "$TEMP_DIR/ray" \
    "$TEMP_DIR/junit"
export TMPDIR="$SHORT_TMP_DIR"
export PYTHONPYCACHEPREFIX="$TEMP_DIR/pycache"
export XDG_CACHE_HOME="$TEMP_DIR/xdg-cache"
export UV_CACHE_DIR="$TEMP_DIR/uv-cache"
export TRITON_CACHE_DIR="$TEMP_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$TEMP_DIR/torchinductor-cache"
export TORCH_EXTENSIONS_DIR="$TEMP_DIR/torch-extensions"
export HF_HOME="$TEMP_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$TEMP_DIR/huggingface/hub"
export HF_DATASETS_CACHE="$TEMP_DIR/huggingface/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO_ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cd -- "$REPO_ROOT"

run_command timeout --signal=TERM --kill-after=10s 60s "$NEMO_RL_PYTHON" - <<'PY'
from pathlib import Path

import nemo_rl
import pytest_asyncio
import pytest_timeout
import torch
import transformers

repo_root = Path.cwd().resolve()
nemo_rl_path = Path(nemo_rl.__file__).resolve()
if not nemo_rl_path.is_relative_to(repo_root):
    raise SystemExit(
        f"CANARY ERROR: nemo_rl imported outside the exact source: {nemo_rl_path}"
    )
print(f"nemo_rl_source={nemo_rl_path}")
count = torch.cuda.device_count()
print(f"visible_gpu_count={count}")
if count != 4:
    raise SystemExit(f"CANARY ERROR: expected exactly 4 visible GPUs, found {count}")
if not torch.cuda.is_available():
    raise SystemExit("CANARY ERROR: CUDA is not available")
for index in range(count):
    print(f"gpu_{index}_name={torch.cuda.get_device_name(index)}")
print(f"pytest_asyncio_version={getattr(pytest_asyncio, '__version__', 'unknown')}")
print(f"pytest_timeout_import={pytest_timeout.__name__}")
print(f"torch_version={torch.__version__}")
print(f"cuda_runtime_version={torch.version.cuda or 'unavailable'}")
print(f"transformers_version={transformers.__version__}")
PY

printf '\n=== vLLM worker interpreter probe ===\n'
run_command timeout --signal=TERM --kill-after=10s 60s "$VLLM_PYTHON" - <<'PY'
import vllm

version = getattr(vllm, "__version__", None)
if not version:
    raise SystemExit("CANARY ERROR: vLLM imported but did not expose __version__")
print(f"vllm_version={version}")
PY

printf '\n=== Megatron worker interpreter probe ===\n'
run_command timeout --signal=TERM --kill-after=10s 60s "$MEGATRON_PYTHON" - <<'PY'
import megatron.core
import transformer_engine

version = getattr(transformer_engine, "__version__", None)
if not version:
    raise SystemExit("Transformer Engine did not expose __version__")
print(f"transformer_engine_version={version}")
print("megatron_core_import=ok")
PY

if ! timeout --signal=TERM --kill-after=10s 60s "$NEMO_RL_PYTHON" -m pytest --version >/dev/null 2>&1; then
    die "DEPENDENCY MISSING: pytest is unavailable in ${NEMO_RL_PYTHON}"
fi
if ! timeout --signal=TERM --kill-after=10s 60s "$NEMO_RL_PYTHON" -m ruff --version >/dev/null 2>&1; then
    die "DEPENDENCY MISSING: ruff is unavailable in ${NEMO_RL_PYTHON}"
fi

for test_path in \
    "${PRECISION_TESTS[@]}" \
    "${REFIT_PLAN_TESTS[@]}" \
    "${REFIT_RUNTIME_TESTS[@]}" \
    "${GENERATION_LIFECYCLE_TESTS[@]}" \
    "${GRPO_FAIL_FAST_TESTS[@]}" \
    "${GYM_STARTUP_FAIL_FAST_TESTS[@]}" \
    "${TOKEN_REFIT_FAIL_FAST_TESTS[@]}"; do
    test_file="${test_path%%::*}"
    [[ -e "$test_file" ]] || die "archived source is missing focused test file: ${test_file}"
done

verify_junit_no_skip() {
    local junit_path=$1
    local expected_minimum=$2
    run_command timeout --signal=TERM --kill-after=5s 30s \
        "$NEMO_RL_PYTHON" - "$junit_path" "$expected_minimum" <<'PY'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
expected_minimum = int(sys.argv[2])
root = ET.parse(path).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
if tests < expected_minimum:
    raise SystemExit(
        f"CANARY ERROR: {path} ran {tests} tests, expected at least {expected_minimum}"
    )
if skipped:
    raise SystemExit(f"CANARY ERROR: {path} contains {skipped} skipped tests")
print(f"junit_tests={tests} junit_skipped={skipped} junit_path={path}")
PY
}

run_pytest_group() {
    local label=$1
    local timeout_s=$2
    local expected_minimum=$3
    shift 3
    local junit_name junit_path
    junit_name="${label//[^[:alnum:]]/_}"
    junit_path="${TEMP_DIR}/junit/${junit_name}.xml"
    printf '\n=== %s ===\n' "$label"
    run_command timeout --signal=TERM --kill-after=30s "${timeout_s}s" \
        "$NEMO_RL_PYTHON" -m pytest -q -rs --strict-config --strict-markers \
        --maxfail=0 \
        -p no:cacheprovider -p pytest_asyncio.plugin -p pytest_timeout --noconftest \
        --confcutdir="$REPO_ROOT/tests/unit" --junitxml="$junit_path" "$@"
    verify_junit_no_skip "$junit_path" "$expected_minimum"
}

run_pytest_group 'Gym startup and refit fail-fast contracts' 120 \
    "${#GYM_STARTUP_FAIL_FAST_TESTS[@]}" "${GYM_STARTUP_FAIL_FAST_TESTS[@]}"
run_pytest_group 'precision policy contracts' 120 1 "${PRECISION_TESTS[@]}"
run_pytest_group 'refit plan contracts' 90 1 "${REFIT_PLAN_TESTS[@]}"
run_pytest_group 'refit supervisor collective and checkpoint contracts' 120 3 \
    "${REFIT_RUNTIME_TESTS[@]}"
run_pytest_group 'generation lifecycle contracts' 60 2 \
    "${GENERATION_LIFECYCLE_TESTS[@]}"
run_pytest_group 'GRPO fail-fast contracts' 60 1 "${GRPO_FAIL_FAST_TESTS[@]}"
run_pytest_group 'token capture and backend-general refit fail-fast contracts' 180 \
    "${#TOKEN_REFIT_FAIL_FAST_TESTS[@]}" "${TOKEN_REFIT_FAIL_FAST_TESTS[@]}"

readonly RUFF_SLICES=(
    tests/unit/environments/test_nemo_gym_utils.py
    tests/unit/algorithms/test_distillation.py
    tests/unit/algorithms/test_grpo.py
    tests/unit/single_controller/test_setup.py
    tests/unit/models/generation/test_vllm_refit_lifecycle.py
    nemo_rl/precision_policy
    nemo_rl/weight_sync/refit_plan.py
    nemo_rl/weight_sync/refit_supervisor.py
    nemo_rl/weight_sync/collective_weight_synchronizer.py
    nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py
    nemo_rl/models/generation/interfaces.py
    nemo_rl/models/generation/vllm/vllm_generation.py
    nemo_rl/models/generation/vllm/vllm_worker_async.py
    nemo_rl/models/generation/trtllm/trtllm_generation.py
    nemo_rl/algorithms/async_utils/trajectory_collector.py
    nemo_rl/algorithms/distillation.py
    nemo_rl/algorithms/single_controller.py
    nemo_rl/algorithms/single_controller_utils/setup.py
    nemo_rl/algorithms/grpo.py
    nemo_rl/environments/nemo_gym.py
)
for source_path in "${RUFF_SLICES[@]}"; do
    [[ -e "$source_path" ]] || die "archived source is missing lint path: ${source_path}"
done

printf '\n=== ruff check ===\n'
run_command timeout --signal=TERM --kill-after=10s 60s \
    "$NEMO_RL_PYTHON" -m ruff check "${RUFF_SLICES[@]}"
printf '\n=== ruff format check ===\n'
run_command timeout --signal=TERM --kill-after=10s 60s \
    "$NEMO_RL_PYTHON" -m ruff format --check "${RUFF_SLICES[@]}"

mapfile -t PY_COMPILE_FILES < <(
    find nemo_rl/precision_policy -type f -name '*.py' -print
)
PY_COMPILE_FILES+=(
    tests/unit/environments/test_nemo_gym_utils.py
    tests/unit/algorithms/test_distillation.py
    tests/unit/algorithms/test_grpo.py
    tests/unit/single_controller/test_setup.py
    tests/unit/models/generation/test_vllm_refit_lifecycle.py
    nemo_rl/weight_sync/refit_plan.py
    nemo_rl/weight_sync/refit_supervisor.py
    nemo_rl/weight_sync/collective_weight_synchronizer.py
    nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py
    nemo_rl/models/generation/interfaces.py
    nemo_rl/models/generation/vllm/vllm_generation.py
    nemo_rl/models/generation/vllm/vllm_worker_async.py
    nemo_rl/models/generation/trtllm/trtllm_generation.py
    nemo_rl/algorithms/async_utils/trajectory_collector.py
    nemo_rl/algorithms/distillation.py
    nemo_rl/algorithms/single_controller.py
    nemo_rl/algorithms/single_controller_utils/setup.py
    nemo_rl/algorithms/grpo.py
    nemo_rl/environments/nemo_gym.py
)
printf '\n=== py_compile ===\n'
run_command timeout --signal=TERM --kill-after=10s 60s \
    "$NEMO_RL_PYTHON" -m py_compile "${PY_COMPILE_FILES[@]}"

printf '\nCANARY PASS: semantic precision/refit Ptyche canary completed.\n'
