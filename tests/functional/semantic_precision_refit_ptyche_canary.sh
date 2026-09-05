#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
readonly SCRATCH_ROOT="/raid/scratch"
readonly NEMO_RL_PYTHON="${NEMO_RL_PYTHON:-/opt/nemo_rl_venv/bin/python}"
readonly DEFAULT_VLLM_PYTHON="/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
readonly MEGATRON_PYTHON="${MEGATRON_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
readonly KNOWN_VLLM_PYTHONS=(
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
)

TEMP_DIR=""
VLLM_PYTHON="${VLLM_PYTHON:-}"

die() {
    printf 'CANARY ERROR: %s\n' "$*"
    exit 1
}

run_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

check_clean_worktree() {
    local status
    status="$(git --no-optional-locks -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
    if [[ -n "$status" ]]; then
        printf 'CANARY ERROR: worktree is not clean:\n%s\n' "$status"
        return 1
    fi
}

check_revision_binding() {
    local head expected remote

    [[ "${EXPECTED_REPO_SHA:-}" =~ ^[0-9a-f]{40}$ ]] \
        || die 'EXPECTED_REPO_SHA must be a full 40-character lowercase commit SHA'

    expected="$(git --no-optional-locks -C "$REPO_ROOT" rev-parse --verify "${EXPECTED_REPO_SHA}^{commit}")" \
        || die "EXPECTED_REPO_SHA does not resolve to a commit: ${EXPECTED_REPO_SHA}"
    head="$(git --no-optional-locks -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')" \
        || die 'unable to resolve the current worktree HEAD'
    [[ "$head" == "$expected" ]] \
        || die "worktree HEAD ${head} does not equal EXPECTED_REPO_SHA ${EXPECTED_REPO_SHA}"

    if [[ -n "${EXPECTED_REMOTE_REF:-}" ]]; then
        remote="$(git --no-optional-locks -C "$REPO_ROOT" rev-parse --verify "${EXPECTED_REMOTE_REF}^{commit}")" \
            || die "EXPECTED_REMOTE_REF does not resolve to a commit: ${EXPECTED_REMOTE_REF}"
        [[ "$remote" == "$expected" ]] \
            || die "remote ref ${remote} does not equal EXPECTED_REPO_SHA ${EXPECTED_REPO_SHA}"
        printf 'remote_ref=%s\n' "$EXPECTED_REMOTE_REF"
    fi

    printf 'revision_sha=%s\n' "$expected"
}

check_python_executable() {
    local label=$1
    local python_path=$2
    if [[ "$python_path" == */* ]]; then
        [[ -x "$python_path" ]] \
            || die "${label} is not executable: ${python_path}"
    else
        command -v "$python_path" >/dev/null 2>&1 \
            || die "${label} is not on PATH: ${python_path}"
    fi
    run_command "$python_path" --version
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
    local post_head=''
    local post_remote=''
    local post_status=''
    local post_failure=0

    set +e
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        case "$TEMP_DIR" in
            /raid/scratch/semantic_precision_refit_ptyche_canary.*)
                if ! rm -rf -- "$TEMP_DIR"; then
                    printf 'CANARY ERROR: failed to remove temp directory: %s\n' "$TEMP_DIR"
                    post_failure=1
                elif [[ -e "$TEMP_DIR" ]]; then
                    printf 'CANARY ERROR: temp directory remains after cleanup: %s\n' "$TEMP_DIR"
                    post_failure=1
                fi
                ;;
            *)
                printf 'CANARY ERROR: refusing to remove unexpected temp path: %s\n' "$TEMP_DIR"
                post_failure=1
                ;;
        esac
    fi

    post_head="$(git --no-optional-locks -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" \
        || post_failure=1
    if [[ -n "${EXPECTED_REMOTE_REF:-}" ]]; then
        post_remote="$(git --no-optional-locks -C "$REPO_ROOT" rev-parse --verify "${EXPECTED_REMOTE_REF}^{commit}" 2>/dev/null)" \
            || post_failure=1
    fi
    post_status="$(git --no-optional-locks -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all 2>/dev/null)" \
        || post_failure=1
    if [[ "$post_head" != "${EXPECTED_REPO_SHA:-}" ]]; then
        printf 'CANARY ERROR: worktree HEAD changed during canary: %s\n' "$post_head"
        post_failure=1
    fi
    if [[ -n "${EXPECTED_REMOTE_REF:-}" && "$post_remote" != "${EXPECTED_REPO_SHA:-}" ]]; then
        printf 'CANARY ERROR: remote ref changed during canary: %s\n' "$post_remote"
        post_failure=1
    fi
    if [[ -n "$post_status" ]]; then
        printf 'CANARY ERROR: worktree is dirty after canary:\n%s\n' "$post_status"
        post_failure=1
    fi

    if [[ "$status" -eq 0 && "$post_failure" -ne 0 ]]; then
        status=1
    fi
    exit "$status"
}

check_revision_binding
check_clean_worktree || die 'worktree must be clean before creating the immutable archive'
check_python_executable NEMO_RL_PYTHON "$NEMO_RL_PYTHON"
select_vllm_python
check_python_executable MEGATRON_PYTHON "$MEGATRON_PYTHON"

[[ -d "$SCRATCH_ROOT" ]] \
    || die "required scratch root does not exist: ${SCRATCH_ROOT}"
TEMP_DIR="$(mktemp -d -- "${SCRATCH_ROOT}/semantic_precision_refit_ptyche_canary.XXXXXXXX")" \
    || die "unable to create a temporary directory under ${SCRATCH_ROOT}"
trap cleanup EXIT
readonly EXTRACTED_ROOT="${TEMP_DIR}/source"
mkdir -- "$EXTRACTED_ROOT"

printf 'archive_root=%s\n' "$EXTRACTED_ROOT"
printf '+ git --no-optional-locks -C %q archive --format=tar %q | tar -xf - -C %q\n' \
    "$REPO_ROOT" "$EXPECTED_REPO_SHA" "$EXTRACTED_ROOT"
git --no-optional-locks -C "$REPO_ROOT" archive --format=tar "$EXPECTED_REPO_SHA" \
    | tar -xf - -C "$EXTRACTED_ROOT"

mkdir -p -- \
    "$TEMP_DIR/tmp" \
    "$TEMP_DIR/pycache" \
    "$TEMP_DIR/xdg-cache" \
    "$TEMP_DIR/uv-cache" \
    "$TEMP_DIR/triton-cache" \
    "$TEMP_DIR/torchinductor-cache" \
    "$TEMP_DIR/torch-extensions" \
    "$TEMP_DIR/huggingface" \
    "$TEMP_DIR/ray"
export TMPDIR="$TEMP_DIR/tmp"
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
export PYTHONPATH="$EXTRACTED_ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cd -- "$EXTRACTED_ROOT"

run_command "$NEMO_RL_PYTHON" - <<'PY'
import torch
import transformers

count = torch.cuda.device_count()
print(f"visible_gpu_count={count}")
if count != 4:
    raise SystemExit(f"CANARY ERROR: expected exactly 4 visible GPUs, found {count}")
if not torch.cuda.is_available():
    raise SystemExit("CANARY ERROR: CUDA is not available")
for index in range(count):
    print(f"gpu_{index}_name={torch.cuda.get_device_name(index)}")
print(f"torch_version={torch.__version__}")
print(f"cuda_runtime_version={torch.version.cuda or 'unavailable'}")
print(f"transformers_version={transformers.__version__}")
PY

printf '\n=== vLLM worker interpreter probe ===\n'
run_command "$VLLM_PYTHON" - <<'PY'
import vllm

version = getattr(vllm, "__version__", None)
if not version:
    raise SystemExit("CANARY ERROR: vLLM imported but did not expose __version__")
print(f"vllm_version={version}")
PY

printf '\n=== Megatron worker interpreter probe ===\n'
run_command "$MEGATRON_PYTHON" - <<'PY'
import megatron.core
import transformer_engine

version = getattr(transformer_engine, "__version__", None)
if not version:
    raise SystemExit("Transformer Engine did not expose __version__")
print(f"transformer_engine_version={version}")
print("megatron_core_import=ok")
PY

if ! "$NEMO_RL_PYTHON" -m pytest --version >/dev/null 2>&1; then
    die "DEPENDENCY MISSING: pytest is unavailable in ${NEMO_RL_PYTHON}"
fi
if ! "$NEMO_RL_PYTHON" -m ruff --version >/dev/null 2>&1; then
    die "DEPENDENCY MISSING: ruff is unavailable in ${NEMO_RL_PYTHON}"
fi

readonly PRECISION_TESTS=(
    tests/unit/precision_policy
)
readonly REFIT_PLAN_TESTS=(
    tests/unit/weight_sync/test_refit_plan.py
)
readonly REFIT_RUNTIME_TESTS=(
    tests/unit/weight_sync/test_refit_supervisor.py
    tests/unit/weight_sync/test_collective_refit_supervision.py
    tests/unit/weight_sync/test_checkpoint_engine_weight_synchronizer.py
)
readonly GENERATION_LIFECYCLE_TESTS=(
    tests/unit/models/generation/test_lifecycle_ack_contract.py
    tests/unit/models/generation/test_prepare_refit_info_deadline_contract.py
)
readonly GRPO_FAIL_FAST_TESTS=(
    tests/unit/algorithms/test_grpo_refit_supervision.py
)

for test_path in \
    "${PRECISION_TESTS[@]}" \
    "${REFIT_PLAN_TESTS[@]}" \
    "${REFIT_RUNTIME_TESTS[@]}" \
    "${GENERATION_LIFECYCLE_TESTS[@]}" \
    "${GRPO_FAIL_FAST_TESTS[@]}"; do
    [[ -e "$test_path" ]] || die "archived source is missing focused test path: ${test_path}"
done

run_pytest_group() {
    local label=$1
    shift
    printf '\n=== %s ===\n' "$label"
    # --noconftest is mandatory: these self-contained tests must not load tests/unit/conftest.py.
    run_command "$NEMO_RL_PYTHON" -m pytest -q --noconftest \
        --confcutdir="$EXTRACTED_ROOT/tests/unit" "$@"
}

run_pytest_group 'precision policy contracts' "${PRECISION_TESTS[@]}"
run_pytest_group 'refit plan contracts' "${REFIT_PLAN_TESTS[@]}"
run_pytest_group 'refit supervisor, collective, and checkpoint contracts' \
    "${REFIT_RUNTIME_TESTS[@]}"
run_pytest_group 'generation lifecycle contracts' "${GENERATION_LIFECYCLE_TESTS[@]}"
run_pytest_group 'GRPO fail-fast contracts' "${GRPO_FAIL_FAST_TESTS[@]}"

readonly RUFF_SLICES=(
    nemo_rl/precision_policy
    nemo_rl/weight_sync/refit_plan.py
    nemo_rl/weight_sync/refit_supervisor.py
    nemo_rl/weight_sync/collective_weight_synchronizer.py
    nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py
    nemo_rl/models/generation/interfaces.py
    nemo_rl/models/generation/vllm/vllm_generation.py
    nemo_rl/models/generation/trtllm/trtllm_generation.py
    nemo_rl/algorithms/grpo.py
)
for source_path in "${RUFF_SLICES[@]}"; do
    [[ -e "$source_path" ]] || die "archived source is missing lint path: ${source_path}"
done

printf '\n=== ruff check ===\n'
run_command "$NEMO_RL_PYTHON" -m ruff check "${RUFF_SLICES[@]}"
printf '\n=== ruff format check ===\n'
run_command "$NEMO_RL_PYTHON" -m ruff format --check "${RUFF_SLICES[@]}"

mapfile -t PY_COMPILE_FILES < <(
    find nemo_rl/precision_policy -type f -name '*.py' -print
)
PY_COMPILE_FILES+=(
    nemo_rl/weight_sync/refit_plan.py
    nemo_rl/weight_sync/refit_supervisor.py
    nemo_rl/weight_sync/collective_weight_synchronizer.py
    nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py
    nemo_rl/models/generation/interfaces.py
    nemo_rl/models/generation/vllm/vllm_generation.py
    nemo_rl/models/generation/trtllm/trtllm_generation.py
    nemo_rl/algorithms/grpo.py
)
printf '\n=== py_compile ===\n'
run_command "$NEMO_RL_PYTHON" -m py_compile "${PY_COMPILE_FILES[@]}"

printf '\nCANARY PASS: semantic precision/refit Ptyche canary completed.\n'
