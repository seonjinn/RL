#!/bin/bash

set -euo pipefail
umask 027

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
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

write_not_applicable_report() {
    local reason=$1
    printf \
        '{"schema_version":1,"status":"not-applicable","workload":"Qwen/Qwen3-30B-A3B TP1 4n4g MXFP8 rollout","run_id":"%s","reason":"%s","fallback":"Nemotron 3 Ultra TP4","promoted_qwen_manifest":null}\n' \
        "$RUN_ID" "$reason" > "$REPORT"
    printf '[NOT-APPLICABLE] %s; use Nemotron 3 Ultra TP4 fallback. Report: %s\n' \
        "$reason" "$REPORT" >&2
}

write_inventory_ready_report() {
    local shape_count=$1
    printf \
        '{"schema_version":1,"status":"inventory-ready","workload":"Qwen/Qwen3-30B-A3B TP1 4n4g MXFP8 rollout","run_id":"%s","eligible_shapes":%s,"next_stage":"shmoo","fallback":null,"promoted_qwen_manifest":null}\n' \
        "$RUN_ID" "$shape_count" > "$REPORT"
}

for command_name in find git grep realpath sed sha256sum sort stat uv; do
    require_command "$command_name"
done

RUN_ID=${NEMO_RL_MXFP8_RUN_ID:-}
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
    die "NEMO_RL_MXFP8_RUN_ID must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"

SHARED_ROOT_INPUT=${NEMO_RL_MXFP8_SHARED_ROOT:-}
[[ "$SHARED_ROOT_INPUT" == /* ]] ||
    die "NEMO_RL_MXFP8_SHARED_ROOT must be an existing absolute Lustre directory"
[[ -d "$SHARED_ROOT_INPUT" ]] ||
    die "NEMO_RL_MXFP8_SHARED_ROOT is not a directory: $SHARED_ROOT_INPUT"
SHARED_ROOT=$(realpath "$SHARED_ROOT_INPUT")
[[ "$SHARED_ROOT" == /lustre/* ]] ||
    die "NEMO_RL_MXFP8_SHARED_ROOT must resolve below /lustre"
SHARED_FILESYSTEM_TYPE=$(stat -f -c %T -- "$SHARED_ROOT")
[[ "$SHARED_FILESYSTEM_TYPE" == "lustre" ]] ||
    die "NEMO_RL_MXFP8_SHARED_ROOT is on $SHARED_FILESYSTEM_TYPE, expected Lustre"
[[ -w "$SHARED_ROOT" ]] ||
    die "NEMO_RL_MXFP8_SHARED_ROOT is not writable: $SHARED_ROOT"

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

NEMO_TRACKED_STATUS=$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=no)
[[ -z "$NEMO_TRACKED_STATUS" ]] ||
    die "NeMo-RL tracked worktree is dirty; commit or restore tracked changes before tracing"
NEMO_RL_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

RUNTIME_METADATA=$(
    uv run --locked --extra vllm python -c \
        'import importlib.metadata, pathlib, sys, vllm; print(f"PYTHON_EXECUTABLE={sys.executable}"); print(f"VLLM_VERSION={vllm.__version__}"); print(f"FLASHINFER_VERSION={importlib.metadata.version(\"flashinfer-python\")}"); print(f"VLLM_FILE={pathlib.Path(vllm.__file__).resolve()}"); print(f"VLLM_SOURCE_ROOT={pathlib.Path(vllm.__file__).resolve().parents[1]}")'
)
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
VLLM_TRACKED_STATUS=$(
    git -C "$VLLM_SOURCE_ROOT" status --porcelain=v1 --untracked-files=no
)
[[ -z "$VLLM_TRACKED_STATUS" ]] ||
    die "custom vLLM tracked worktree is dirty; commit or restore tracked changes before tracing"
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

RUN_DIR="$SHARED_ROOT/$RUN_ID"
if ! mkdir -- "$RUN_DIR"; then
    die "immutable Task 3 run directory already exists or cannot be created: $RUN_DIR"
fi
PROVENANCE_DIR="$RUN_DIR/provenance"
TRACE_DIR="$RUN_DIR/traces"
RECEIPT_DIR="$RUN_DIR/startup_receipts"
RECEIPT_HOOK_DIR="$RUN_DIR/receipt_hook"
RUN_LOG="$RUN_DIR/run.log"
REPORT="$RUN_DIR/task3_report.json"
TRACE_FILE_LIST="$RUN_DIR/trace_files.txt"
RECEIPT_INVENTORY="$RUN_DIR/startup_receipt_inventory.json"
INVENTORY="$RUN_DIR/qwen3_30ba3b_tp1_inventory.json"
INVENTORY_ERROR="$RUN_DIR/qwen3_30ba3b_tp1_inventory.stderr"
COMPOSED_TRACE_RECIPE="$PROVENANCE_DIR/composed_trace_recipe.yaml"
RESOLVED_RUNTIME_CONFIG="$PROVENANCE_DIR/resolved_runtime_config.yaml"
mkdir -- "$PROVENANCE_DIR" "$TRACE_DIR" "$RECEIPT_DIR" "$RECEIPT_HOOK_DIR"
touch "$RUN_LOG"
export NEMO_RL_MXFP8_TRACE_DIR="$TRACE_DIR"

cd "$PROJECT_ROOT"
printf '%s\n' "$RUNTIME_METADATA" | tee -a "$RUN_LOG"
assert_fixed "VLLM_VERSION=$EXPECTED_VLLM_VERSION" "$RUN_LOG"
assert_fixed "FLASHINFER_VERSION=$EXPECTED_FLASHINFER_VERSION" "$RUN_LOG"

uv run --locked --extra vllm python -c \
    'import sys; from omegaconf import OmegaConf; from nemo_rl.utils.config import register_omegaconf_resolvers; register_omegaconf_resolvers(); from tools.config_cli import load_config; config = load_config(sys.argv[1]); OmegaConf.save(config, sys.argv[2], resolve=True)' \
    "$TRACE_RECIPE" "$COMPOSED_TRACE_RECIPE"
read -r EXPECTED_NUM_NODES EXPECTED_GPUS_PER_NODE ROLLOUT_TP ROLLOUT_PP < <(
    uv run --locked --extra vllm python -c \
        'import sys, yaml; config = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); cluster = config["cluster"]; vllm = config["policy"]["generation"]["vllm_cfg"]; print(cluster["num_nodes"], cluster["gpus_per_node"], vllm["tensor_parallel_size"], vllm["pipeline_parallel_size"])' \
        "$COMPOSED_TRACE_RECIPE"
)
[[ "$ROLLOUT_TP" == "1" && "$ROLLOUT_PP" == "1" ]] ||
    die "Task 3 requires vLLM TP1/PP1, got TP$ROLLOUT_TP/PP$ROLLOUT_PP"
EXPECTED_ROLLOUT_ACTORS=$((EXPECTED_NUM_NODES * EXPECTED_GPUS_PER_NODE))
[[ "$EXPECTED_ROLLOUT_ACTORS" == "16" ]] ||
    die "Task 3 exact 4n4g recipe must create 16 rollout actors"

RECIPE_SNAPSHOT="$PROVENANCE_DIR/$(basename "$SOURCE_RECIPE")"
EMPTY_HINTS="$PROVENANCE_DIR/qwen3_30ba3b_tp1_trace_bootstrap.empty_hints"
git show "$NEMO_RL_COMMIT:$SOURCE_RECIPE" > "$RECIPE_SNAPSHOT"
: > "$EMPTY_HINTS"
chmod 0440 "$RECIPE_SNAPSHOT" "$EMPTY_HINTS" "$COMPOSED_TRACE_RECIPE"
SOURCE_MANIFEST_SHA256=$(sha256sum -- "$RECIPE_SNAPSHOT")
SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256%% *}
SOURCE_HINT_SHA256=$(sha256sum -- "$EMPTY_HINTS")
SOURCE_HINT_SHA256=${SOURCE_HINT_SHA256%% *}
[[ "$SOURCE_HINT_SHA256" == "$EMPTY_SHA256" ]] ||
    die "empty bootstrap hints SHA256 changed"

OFFLINE_SHMOO="$VLLM_SOURCE_ROOT/tools/mxfp8/offline_shmoo.py"
[[ -f "$OFFLINE_SHMOO" ]] ||
    die "custom vLLM offline qualification tool not found: $OFFLINE_SHMOO"
# Safety correction: generate once on shared Lustre instead of mutating a
# package-relative path that would only be visible on the driver node.
BOOTSTRAP_MANIFEST="$RUN_DIR/$BOOTSTRAP_CONFIG_NAME"
[[ "$BOOTSTRAP_MANIFEST" == /* ]] ||
    die "bootstrap manifest must be an absolute shared path"
BOOTSTRAP_SHA256=$(
    uv run --locked --extra vllm python "$OFFLINE_SHMOO" \
        trace-bootstrap-qwen3-30ba3b-tp1 \
        --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
        --source-hint-sha256 "$SOURCE_HINT_SHA256" \
        --container-sha256 "$CONTAINER_SHA256" \
        --output "$BOOTSTRAP_MANIFEST"
)
[[ "$BOOTSTRAP_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    die "trace bootstrap command did not return a SHA256"
ACTUAL_BOOTSTRAP_SHA256=$(sha256sum -- "$BOOTSTRAP_MANIFEST")
ACTUAL_BOOTSTRAP_SHA256=${ACTUAL_BOOTSTRAP_SHA256%% *}
[[ "$ACTUAL_BOOTSTRAP_SHA256" == "$BOOTSTRAP_SHA256" ]] ||
    die "generated bootstrap bytes do not match the command's SHA256"
chmod 0440 "$BOOTSTRAP_MANIFEST"

uv run --locked --extra vllm python - \
    "$BOOTSTRAP_MANIFEST" "$BOOTSTRAP_SHA256" <<'PY_BOOTSTRAP_VALIDATOR'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_sha256 = sys.argv[2]
payload = manifest_path.read_bytes()
assert manifest_path.is_absolute()
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
PY_BOOTSTRAP_VALIDATOR

uv run --locked --extra vllm python -c \
    'import sys; from omegaconf import OmegaConf; from nemo_rl.utils.config import register_omegaconf_resolvers; register_omegaconf_resolvers(); from tools.config_cli import load_config; config = load_config(sys.argv[1]); OmegaConf.update(config, "policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE", sys.argv[2], force_add=True); OmegaConf.save(config, sys.argv[3], resolve=True)' \
    "$TRACE_RECIPE" "$BOOTSTRAP_MANIFEST" "$RESOLVED_RUNTIME_CONFIG"
RESOLVED_CONFIG_SHA256=$(sha256sum -- "$RESOLVED_RUNTIME_CONFIG")
RESOLVED_CONFIG_SHA256=${RESOLVED_CONFIG_SHA256%% *}
chmod 0440 "$RESOLVED_RUNTIME_CONFIG"

{
    printf 'RUN_ID=%s\n' "$RUN_ID"
    printf 'RUN_DIR=%s\n' "$RUN_DIR"
    printf 'NEMO_RL_COMMIT=%s\n' "$NEMO_RL_COMMIT"
    printf 'VLLM_SOURCE_COMMIT=%s\n' "$VLLM_SOURCE_COMMIT"
    printf 'VLLM_SOURCE_URL=%s\n' "$EXPECTED_VLLM_SOURCE_URL"
    printf 'CONTAINER_SHA256=%s\n' "$CONTAINER_SHA256"
    printf 'RESOLVED_CONFIG_SHA256=%s\n' "$RESOLVED_CONFIG_SHA256"
    printf 'SOURCE_MANIFEST_SHA256=%s\n' "$SOURCE_MANIFEST_SHA256"
    printf 'SOURCE_HINT_SHA256=%s\n' "$SOURCE_HINT_SHA256"
    printf 'EXPECTED_ROLLOUT_ACTORS=%s\n' "$EXPECTED_ROLLOUT_ACTORS"
} | tee -a "$RUN_LOG"

RECEIPT_HOOK="$RECEIPT_HOOK_DIR/sitecustomize.py"
tee "$RECEIPT_HOOK" > /dev/null <<'PY_RECEIPT_HOOK'
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


config_reference = os.environ.get("VLLM_MXFP8_DENSE_CONFIG_FILE", "")
rank_reference = os.environ.get("RANK", "")
receipt_dir_reference = os.environ.get("NEMO_RL_MXFP8_RECEIPT_DIR", "")
if config_reference and rank_reference and receipt_dir_reference:
    def git_commit(source_root: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    config_path = Path(config_reference)
    if not config_path.is_absolute():
        raise RuntimeError("Task 3 startup receipt requires an absolute config path")
    config_bytes = config_path.read_bytes()
    hostname = socket.gethostname()
    pid = os.getpid()
    vllm_spec = importlib.util.find_spec("vllm")
    nemo_rl_spec = importlib.util.find_spec("nemo_rl")
    if (
        vllm_spec is None
        or vllm_spec.origin is None
        or nemo_rl_spec is None
        or nemo_rl_spec.origin is None
    ):
        raise RuntimeError("Task 3 startup receipt cannot locate runtime packages")
    vllm_file = Path(vllm_spec.origin).resolve()
    vllm_source_root = vllm_file.parents[1]
    nemo_rl_file = Path(nemo_rl_spec.origin).resolve()
    nemo_rl_source_root = nemo_rl_file.parents[1]
    document = {
        "schema_version": 1,
        "rank": int(rank_reference),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "node_rank": int(os.environ["NODE_RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "hostname": hostname,
        "pid": pid,
        "python_executable": sys.executable,
        "nemo_rl_commit": git_commit(nemo_rl_source_root),
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_source_commit": git_commit(vllm_source_root),
        "vllm_source_root": str(vllm_source_root),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    receipt_dir = Path(receipt_dir_reference)
    output = receipt_dir / (
        f"startup_receipt_rank{document['rank']:04d}_{hostname}_{pid}.json"
    )
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o640,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(
        "NEMO_RL_MXFP8_STARTUP_RECEIPT="
        + json.dumps(document, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
PY_RECEIPT_HOOK
chmod 0440 "$RECEIPT_HOOK"

export NEMO_RL_MXFP8_RECEIPT_DIR="$RECEIPT_DIR"
export PYTHONPATH="$RECEIPT_HOOK_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAY_DEDUP_LOGS=0
{
    printf 'VLLM_MXFP8_DENSE_CONFIG_FILE=%s\n' "$BOOTSTRAP_MANIFEST"
    printf 'CONFIG_SHA256=%s\n' "$BOOTSTRAP_SHA256"
} | tee -a "$RUN_LOG"

set +e
uv run --locked --extra vllm examples/run_grpo.py \
    --config "$TRACE_RECIPE" \
    "++policy.generation.vllm_cfg.env_vars.VLLM_MXFP8_DENSE_CONFIG_FILE=$BOOTSTRAP_MANIFEST" \
    2>&1 | tee -a "$RUN_LOG"
ROLLOUT_STATUS=${PIPESTATUS[0]}
set -e
(( ROLLOUT_STATUS == 0 )) ||
    die "Qwen 4n4g MXFP8 trace rollout failed with status $ROLLOUT_STATUS"

assert_fixed "VLLM_VERSION=$EXPECTED_VLLM_VERSION" "$RUN_LOG"
assert_fixed "FLASHINFER_VERSION=$EXPECTED_FLASHINFER_VERSION" "$RUN_LOG"
assert_fixed "NEMO_RL_COMMIT=$NEMO_RL_COMMIT" "$RUN_LOG"
assert_fixed "VLLM_SOURCE_COMMIT=$EXPECTED_VLLM_SOURCE_COMMIT" "$RUN_LOG"
assert_fixed "VLLM_SOURCE_URL=$EXPECTED_VLLM_SOURCE_URL" "$RUN_LOG"
assert_fixed "CONFIG_SHA256=$BOOTSTRAP_SHA256" "$RUN_LOG"
assert_fixed "VLLM_MXFP8_DENSE_CONFIG_FILE=$BOOTSTRAP_MANIFEST" "$RUN_LOG"

uv run --locked --extra vllm python - \
    "$RECEIPT_DIR" \
    "$EXPECTED_ROLLOUT_ACTORS" \
    "$EXPECTED_NUM_NODES" \
    "$EXPECTED_GPUS_PER_NODE" \
    "$NEMO_RL_COMMIT" \
    "$EXPECTED_VLLM_SOURCE_COMMIT" \
    "$EXPECTED_VLLM_VERSION" \
    "$EXPECTED_FLASHINFER_VERSION" \
    "$BOOTSTRAP_MANIFEST" \
    "$BOOTSTRAP_SHA256" \
    "$RECEIPT_INVENTORY" <<'PY_RECEIPT_VALIDATOR'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

receipt_dir = Path(sys.argv[1])
expected_actor_count = int(sys.argv[2])
expected_node_count = int(sys.argv[3])
expected_gpus_per_node = int(sys.argv[4])
expected_nemo_commit = sys.argv[5]
expected_vllm_commit = sys.argv[6]
expected_vllm_version = sys.argv[7]
expected_flashinfer_version = sys.argv[8]
expected_config_path = sys.argv[9]
expected_config_sha256 = sys.argv[10]
inventory_path = Path(sys.argv[11])

receipt_paths = sorted(receipt_dir.glob("startup_receipt_rank*_*.json"))
assert len(receipt_paths) == expected_actor_count, (
    len(receipt_paths),
    expected_actor_count,
)
receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
assert {receipt["schema_version"] for receipt in receipts} == {1}
expected_ranks = set(range(expected_actor_count))
actual_ranks = [receipt["rank"] for receipt in receipts]
assert set(actual_ranks) == expected_ranks
assert len(actual_ranks) == len(set(actual_ranks))
actor_processes = [
    (receipt["hostname"], receipt["pid"])
    for receipt in receipts
]
assert len(actor_processes) == len(set(actor_processes))
assert {receipt["world_size"] for receipt in receipts} == {expected_actor_count}
assert {receipt["nemo_rl_commit"] for receipt in receipts} == {
    expected_nemo_commit
}
assert {receipt["vllm_source_commit"] for receipt in receipts} == {
    expected_vllm_commit
}
assert {receipt["vllm_version"] for receipt in receipts} == {
    expected_vllm_version
}
assert {receipt["flashinfer_version"] for receipt in receipts} == {
    expected_flashinfer_version
}
assert {receipt["config_path"] for receipt in receipts} == {
    expected_config_path
}
assert Path(expected_config_path).is_absolute()
assert {receipt["config_sha256"] for receipt in receipts} == {
    expected_config_sha256
}
assert len({receipt["python_executable"] for receipt in receipts}) == 1
assert len({receipt["vllm_source_root"] for receipt in receipts}) == 1

host_counts = Counter(receipt["hostname"] for receipt in receipts)
assert len(host_counts) == expected_node_count
assert set(host_counts.values()) == {expected_gpus_per_node}
host_node_ranks: dict[str, set[int]] = defaultdict(set)
host_local_ranks: dict[str, set[int]] = defaultdict(set)
node_rank_hosts: dict[int, set[str]] = defaultdict(set)
for receipt in receipts:
    hostname = receipt["hostname"]
    node_rank = receipt["node_rank"]
    host_node_ranks[hostname].add(node_rank)
    host_local_ranks[hostname].add(receipt["local_rank"])
    node_rank_hosts[node_rank].add(hostname)
assert set(node_rank_hosts) == set(range(expected_node_count))
assert all(len(hosts) == 1 for hosts in node_rank_hosts.values())
assert all(len(node_ranks) == 1 for node_ranks in host_node_ranks.values())
assert all(
    local_ranks == set(range(expected_gpus_per_node))
    for local_ranks in host_local_ranks.values()
)

inventory = {
    "schema_version": 1,
    "expected_actor_count": expected_actor_count,
    "receipts": sorted(receipts, key=lambda receipt: receipt["rank"]),
}
with inventory_path.open("x", encoding="utf-8") as stream:
    json.dump(inventory, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
print(f"validated {len(receipts)} rollout startup receipts")
PY_RECEIPT_VALIDATOR

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
    "$BOOTSTRAP_SHA256" "$RECEIPT_INVENTORY" "${TRACE_FILES[@]}" <<'PY_TRACE_VALIDATOR'
import json
import sys
from pathlib import Path

expected_sha256 = sys.argv[1]
receipt_inventory = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
trace_paths = tuple(Path(value) for value in sys.argv[3:])
actor_processes = {
    (receipt["hostname"], receipt["pid"])
    for receipt in receipt_inventory["receipts"]
}
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
            assert (record["hostname"], record["pid"]) in actor_processes
            assert record["config_sha256"] == expected_sha256
            record_count += 1
    assert record_count > 0
PY_TRACE_VALIDATOR

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
