#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
PROVENANCE_DIR="$EXP_DIR/provenance"
TRACE_DIR="$EXP_DIR/traces"
RUN_LOG="$EXP_DIR/run.log"
REPORT="$EXP_DIR/task3_report.json"
TRACE_FILE_LIST="$EXP_DIR/trace_files.txt"
INVENTORY="$EXP_DIR/qwen3_30ba3b_tp1_inventory.json"
INVENTORY_ERROR="$EXP_DIR/qwen3_30ba3b_tp1_inventory.stderr"
TRACE_RECIPE="$PROJECT_ROOT/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-adaptive-trace.yaml"
SOURCE_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml"
BOOTSTRAP_CONFIG_NAME="qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"
EXPECTED_VLLM_VERSION=0.20.2
EXPECTED_FLASHINFER_VERSION=0.6.8.post1
EXPECTED_VLLM_SOURCE_COMMIT=bc5881924556fcf830f8158815d5a62cef0fbcba
EXPECTED_VLLM_SOURCE_URL=https://github.com/seonjinn/vllm.git
EMPTY_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

die() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" > /dev/null 2>&1 || die "required command not found: $1"
}

assert_fixed() {
    local expected=$1
    local file=$2
    grep -Fq -- "$expected" "$file" ||
        die "expected '$expected' in $file"
}

assert_regex() {
    local expected=$1
    local file=$2
    grep -Eq -- "$expected" "$file" ||
        die "expected regex '$expected' in $file"
}

write_not_applicable_report() {
    local reason=$1
    printf \
        '{"schema_version":1,"status":"not-applicable","workload":"Qwen/Qwen3-30B-A3B TP1 4n4g MXFP8 rollout","reason":"%s","fallback":"Nemotron 3 Ultra TP4","promoted_qwen_manifest":null}\n' \
        "$reason" > "$REPORT"
    printf '[NOT-APPLICABLE] %s; use Nemotron 3 Ultra TP4 fallback. Report: %s\n' \
        "$reason" "$REPORT" >&2
}

write_inventory_ready_report() {
    local shape_count=$1
    printf \
        '{"schema_version":1,"status":"inventory-ready","workload":"Qwen/Qwen3-30B-A3B TP1 4n4g MXFP8 rollout","eligible_shapes":%s,"next_stage":"shmoo","fallback":null,"promoted_qwen_manifest":null}\n' \
        "$shape_count" > "$REPORT"
}

for command_name in cmp find git grep install mktemp realpath sha256sum sort uv; do
    require_command "$command_name"
done

CONTAINER_SHA256=${NEMO_RL_CONTAINER_SHA256:-}
[[ "$CONTAINER_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    die "NEMO_RL_CONTAINER_SHA256 must be the lowercase SHA256 of the immutable runtime container"

if [[ -n "${NEMO_RL_CONTAINER_IMAGE:-}" ]]; then
    [[ -f "$NEMO_RL_CONTAINER_IMAGE" ]] ||
        die "NEMO_RL_CONTAINER_IMAGE is not a file: $NEMO_RL_CONTAINER_IMAGE"
    ACTUAL_CONTAINER_SHA256=$(sha256sum -- "$NEMO_RL_CONTAINER_IMAGE")
    ACTUAL_CONTAINER_SHA256=${ACTUAL_CONTAINER_SHA256%% *}
    [[ "$ACTUAL_CONTAINER_SHA256" == "$CONTAINER_SHA256" ]] ||
        die "runtime container SHA256 does not match NEMO_RL_CONTAINER_SHA256"
fi

rm -rf "$EXP_DIR"
mkdir -p "$PROVENANCE_DIR" "$TRACE_DIR"
touch "$RUN_LOG"
export NEMO_RL_MXFP8_TRACE_DIR="$TRACE_DIR"

cd "$PROJECT_ROOT"
NEMO_RL_COMMIT=$(git rev-parse HEAD)
RECIPE_SNAPSHOT="$PROVENANCE_DIR/$(basename "$SOURCE_RECIPE")"
EMPTY_HINTS="$PROVENANCE_DIR/qwen3_30ba3b_tp1_trace_bootstrap.empty_hints"
git show "$NEMO_RL_COMMIT:$SOURCE_RECIPE" > "$RECIPE_SNAPSHOT"
: > "$EMPTY_HINTS"
chmod 0444 "$RECIPE_SNAPSHOT" "$EMPTY_HINTS"
SOURCE_MANIFEST_SHA256=$(sha256sum -- "$RECIPE_SNAPSHOT")
SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256%% *}
SOURCE_HINT_SHA256=$(sha256sum -- "$EMPTY_HINTS")
SOURCE_HINT_SHA256=${SOURCE_HINT_SHA256%% *}
[[ "$SOURCE_HINT_SHA256" == "$EMPTY_SHA256" ]] ||
    die "empty bootstrap hints SHA256 changed"

RUNTIME_METADATA=$(
    uv run --locked --extra vllm python -c \
        'import importlib.metadata, pathlib, sys, vllm; print(f"PYTHON_EXECUTABLE={sys.executable}"); print(f"VLLM_VERSION={vllm.__version__}"); print(f"FLASHINFER_VERSION={importlib.metadata.version(\"flashinfer-python\")}"); print(f"VLLM_FILE={pathlib.Path(vllm.__file__).resolve()}"); print(f"VLLM_SOURCE_ROOT={pathlib.Path(vllm.__file__).resolve().parents[1]}")'
)
printf '%s\n' "$RUNTIME_METADATA" | tee -a "$RUN_LOG"
assert_fixed "VLLM_VERSION=$EXPECTED_VLLM_VERSION" "$RUN_LOG"
assert_fixed "FLASHINFER_VERSION=$EXPECTED_FLASHINFER_VERSION" "$RUN_LOG"

VLLM_FILE=$(printf '%s\n' "$RUNTIME_METADATA" | sed -n 's/^VLLM_FILE=//p')
VLLM_SOURCE_ROOT=$(printf '%s\n' "$RUNTIME_METADATA" | sed -n 's/^VLLM_SOURCE_ROOT=//p')
[[ -n "$VLLM_FILE" && -f "$VLLM_FILE" ]] ||
    die "unable to resolve the imported custom vLLM package"
[[ -n "$VLLM_SOURCE_ROOT" ]] ||
    die "unable to resolve the custom vLLM source root"
VLLM_GIT_ROOT=$(git -C "$VLLM_SOURCE_ROOT" rev-parse --show-toplevel 2> /dev/null) ||
    die "imported vLLM is not backed by a Git checkout: $VLLM_SOURCE_ROOT"
[[ "$(realpath "$VLLM_GIT_ROOT")" == "$(realpath "$VLLM_SOURCE_ROOT")" ]] ||
    die "imported vLLM package is not rooted in its expected source checkout"
VLLM_SOURCE_COMMIT=$(git -C "$VLLM_SOURCE_ROOT" rev-parse HEAD)
[[ "$VLLM_SOURCE_COMMIT" == "$EXPECTED_VLLM_SOURCE_COMMIT" ]] ||
    die "custom vLLM source commit is $VLLM_SOURCE_COMMIT, expected $EXPECTED_VLLM_SOURCE_COMMIT"

SOURCE_URL_FOUND=0
while IFS= read -r remote_name; do
    remote_url=$(git -C "$VLLM_SOURCE_ROOT" remote get-url "$remote_name")
    if [[ "$remote_url" == "$EXPECTED_VLLM_SOURCE_URL" ]]; then
        SOURCE_URL_FOUND=1
        break
    fi
done < <(git -C "$VLLM_SOURCE_ROOT" remote)
(( SOURCE_URL_FOUND == 1 )) ||
    die "custom vLLM checkout has no remote at $EXPECTED_VLLM_SOURCE_URL"

{
    printf 'NEMO_RL_COMMIT=%s\n' "$NEMO_RL_COMMIT"
    printf 'VLLM_SOURCE_COMMIT=%s\n' "$VLLM_SOURCE_COMMIT"
    printf 'VLLM_SOURCE_URL=%s\n' "$EXPECTED_VLLM_SOURCE_URL"
    printf 'CONTAINER_SHA256=%s\n' "$CONTAINER_SHA256"
    printf 'SOURCE_MANIFEST_SHA256=%s\n' "$SOURCE_MANIFEST_SHA256"
    printf 'SOURCE_HINT_SHA256=%s\n' "$SOURCE_HINT_SHA256"
} | tee -a "$RUN_LOG"

OFFLINE_SHMOO="$VLLM_SOURCE_ROOT/tools/mxfp8/offline_shmoo.py"
[[ -f "$OFFLINE_SHMOO" ]] ||
    die "custom vLLM offline qualification tool not found: $OFFLINE_SHMOO"
TEMP_DIR=$(mktemp -d "$EXP_DIR/bootstrap.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT
STAGED_BOOTSTRAP="$TEMP_DIR/$BOOTSTRAP_CONFIG_NAME"
BOOTSTRAP_SHA256=$(
    uv run --locked --extra vllm python "$OFFLINE_SHMOO" \
        trace-bootstrap-qwen3-30ba3b-tp1 \
        --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
        --source-hint-sha256 "$SOURCE_HINT_SHA256" \
        --container-sha256 "$CONTAINER_SHA256" \
        --output "$STAGED_BOOTSTRAP"
)
[[ "$BOOTSTRAP_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    die "trace bootstrap command did not return a SHA256"
ACTUAL_BOOTSTRAP_SHA256=$(sha256sum -- "$STAGED_BOOTSTRAP")
ACTUAL_BOOTSTRAP_SHA256=${ACTUAL_BOOTSTRAP_SHA256%% *}
[[ "$ACTUAL_BOOTSTRAP_SHA256" == "$BOOTSTRAP_SHA256" ]] ||
    die "generated bootstrap bytes do not match the command's SHA256"

TACTIC_CONFIG_DIR="$(dirname "$VLLM_FILE")/model_executor/kernels/linear/mxfp8/tactic_configs"
[[ -d "$TACTIC_CONFIG_DIR" && -w "$TACTIC_CONFIG_DIR" ]] ||
    die "custom vLLM package tactic config directory is not writable: $TACTIC_CONFIG_DIR"
BOOTSTRAP_MANIFEST="$TACTIC_CONFIG_DIR/$BOOTSTRAP_CONFIG_NAME"
if [[ -e "$BOOTSTRAP_MANIFEST" ]]; then
    cmp -s "$STAGED_BOOTSTRAP" "$BOOTSTRAP_MANIFEST" ||
        die "existing package-relative Qwen trace bootstrap has different bytes"
else
    install -m 0444 "$STAGED_BOOTSTRAP" "$BOOTSTRAP_MANIFEST"
fi

uv run --locked --extra vllm python - \
    "$BOOTSTRAP_MANIFEST" "$BOOTSTRAP_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
payload = manifest_path.read_bytes()
assert hashlib.sha256(payload).hexdigest() == expected_sha256
manifest = json.loads(payload)
assert manifest["compatibility"]["model"] == "Qwen/Qwen3-30B-A3B"
assert manifest["compatibility"]["tensor_parallel_size"] == 1
assert manifest["policy"]["default_tactic"] == -1
assert manifest["policy"]["pad_to_128"] is False
assert manifest["tactics"] == {"8x4": [], "128x4": []}
assert (
    manifest["provenance"]["qualification_scope"]
    == "nemo_rl_qwen3_30ba3b_mxfp8_rollout_trace_bootstrap"
)
PY

{
    printf 'VLLM_MXFP8_DENSE_CONFIG_FILE=%s\n' "$BOOTSTRAP_CONFIG_NAME"
    printf 'CONFIG_SHA256=%s\n' "$BOOTSTRAP_SHA256"
} | tee -a "$RUN_LOG"

set +e
uv run --locked --extra vllm examples/run_grpo.py \
    --config "$TRACE_RECIPE" \
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE=$BOOTSTRAP_CONFIG_NAME" \
    2>&1 | tee -a "$RUN_LOG"
ROLLOUT_STATUS=${PIPESTATUS[0]}
set -e
(( ROLLOUT_STATUS == 0 )) ||
    die "Qwen 4n4g MXFP8 trace rollout failed with status $ROLLOUT_STATUS"

assert_fixed "VLLM_VERSION=$EXPECTED_VLLM_VERSION" "$RUN_LOG"
assert_fixed "FLASHINFER_VERSION=$EXPECTED_FLASHINFER_VERSION" "$RUN_LOG"
assert_fixed "VLLM_SOURCE_COMMIT=$EXPECTED_VLLM_SOURCE_COMMIT" "$RUN_LOG"
assert_fixed "VLLM_SOURCE_URL=$EXPECTED_VLLM_SOURCE_URL" "$RUN_LOG"
assert_fixed "CONFIG_SHA256=$BOOTSTRAP_SHA256" "$RUN_LOG"
assert_fixed "VLLM_MXFP8_DENSE_CONFIG_FILE=$BOOTSTRAP_CONFIG_NAME" "$RUN_LOG"
assert_regex \
    "MXFP8 dense config path=.*${BOOTSTRAP_CONFIG_NAME} sha256=${BOOTSTRAP_SHA256}" \
    "$RUN_LOG"

mapfile -d '' -t TRACE_FILES < <(
    find "$TRACE_DIR" -maxdepth 1 -type f \
        \( -name 'adaptive_dispatch_*_*.jsonl' \
        -o -name 'dense_shapes_*_*.jsonl' \) \
        -print0 |
        sort -z
)
if (( ${#TRACE_FILES[@]} == 0 )); then
    write_not_applicable_report "no dense MXFP8 trace files were emitted"
    exit 3
fi
printf '%s\n' "${TRACE_FILES[@]}" > "$TRACE_FILE_LIST"

uv run --locked --extra vllm python - \
    "$BOOTSTRAP_SHA256" "${TRACE_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

expected_sha256 = sys.argv[1]
trace_paths = tuple(Path(value) for value in sys.argv[2:])
assert trace_paths
for trace_path in trace_paths:
    prefix = (
        "adaptive_dispatch"
        if trace_path.name.startswith("adaptive_dispatch_")
        else "dense_shapes"
    )
    record_count = 0
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            expected_name = (
                f"{prefix}_{record['hostname']}_{record['pid']}.jsonl"
            )
            assert trace_path.name == expected_name
            assert record["config_sha256"] == expected_sha256
            record_count += 1
    assert record_count > 0
PY

TRACE_ARGS=()
for trace_file in "${TRACE_FILES[@]}"; do
    TRACE_ARGS+=(--trace "$trace_file")
done

set +e
uv run --locked --extra vllm python "$OFFLINE_SHMOO" inventory \
    --bootstrap-manifest "$BOOTSTRAP_MANIFEST" \
    "${TRACE_ARGS[@]}" \
    --output "$INVENTORY" \
    2> "$INVENTORY_ERROR"
INVENTORY_STATUS=$?
set -e
if (( INVENTORY_STATUS != 0 )); then
    cat "$INVENTORY_ERROR" >&2
    if grep -Fq 'zero eligible dense MXFP8 trace records' "$INVENTORY_ERROR"; then
        write_not_applicable_report "trace files contain zero eligible dense MXFP8 records"
        exit 3
    fi
    exit "$INVENTORY_STATUS"
fi

ELIGIBLE_SHAPES=$(
    uv run --locked --extra vllm python -c \
        'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["shapes"]))' \
        "$INVENTORY"
)
[[ "$ELIGIBLE_SHAPES" =~ ^[1-9][0-9]*$ ]] ||
    die "inventory did not produce a positive eligible shape count"
write_inventory_ready_report "$ELIGIBLE_SHAPES"
printf '[PASS] %s Qwen TP1 shapes are ready for the custom vLLM shmoo stage. Report: %s\n' \
    "$ELIGIBLE_SHAPES" "$REPORT"
