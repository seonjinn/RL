#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONTRACT=$EXP_ROOT/manifests/config_contract.tsv
VALIDATOR=$SCRIPT_DIR/validate_config_contract.py
RUNNER=$SCRIPT_DIR/run_force_on_policy_benchmark.sbatch
SUBMITTER=$SCRIPT_DIR/submit_force_on_policy_matrix.sh

test -f "$CONTRACT"
test -f "$VALIDATOR"
test -f "$RUNNER"
test -f "$SUBMITTER"

PYCACHE_DIR=$(mktemp -d)
trap 'rm -rf "$PYCACHE_DIR"' EXIT
PYTHONPYCACHEPREFIX=$PYCACHE_DIR python3 -m py_compile "$VALIDATOR"
grep -q 'grpo.max_num_steps=' "$VALIDATOR"
grep -q 'checkpointing.enabled=false' "$VALIDATOR"
bash -n "$RUNNER"
bash -n "$SUBMITTER"

test "$(awk 'END {print NR - 1}' "$CONTRACT")" -eq 6
test "$(awk -F '\t' 'NR > 1 {print $5}' "$CONTRACT" | sort | uniq -c | tr -s ' ' | sed 's/^ //')" = $'3 false\n3 true'
test "$(awk -F '\t' 'NR > 1 && $4 == "async1off" {count++} END {print count + 0}' "$CONTRACT")" -eq 6
test "$(awk -F '\t' 'NR > 1 && $7 == 4 {count++} END {print count + 0}' "$CONTRACT")" -eq 6
test "$(awk -F '\t' 'NR > 1 && $9 == 2048 {count++} END {print count + 0}' "$CONTRACT")" -eq 6
test "$(awk -F '\t' 'NR > 1 && $10 == 20 {count++} END {print count + 0}' "$CONTRACT")" -eq 6
test "$(awk -F '\t' 'NR > 1 && $2 == "llama3.1-8b" && $8 == "none" {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $2 == "qwen3-30ba3b" && $8 == 2 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $2 == "qwen3-32b" && $8 == "none" {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $6 == 2 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $6 == 4 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $6 == 8 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 {print $3}' "$CONTRACT" | sort -u | wc -l | tr -d ' ')" -eq 3
if awk -F '\t' 'NR > 1 {print $3}' "$CONTRACT" | grep -Eq -- '-[0-9]+n8g'; then
    printf 'Eight-GPU recipe found in native-4g contract.\n' >&2
    exit 1
fi

grep -q 'train_nodes' "$VALIDATOR"
grep -q 'segment is None' "$VALIDATOR"
grep -q 'train_nodes % case.segment == 0' "$VALIDATOR"

grep -q 'd4cfecf90db41cdf142629963b54b67ab479ab02' "$RUNNER"
grep -q 'bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510' "$RUNNER"
grep -q 'nemo_rl_nightly_20260630_0215.sqsh' "$RUNNER"
grep -q 'grpo.max_num_steps=20' "$RUNNER"
grep -q 'checkpointing.enabled=false' "$RUNNER"
grep -q 'policy.train_global_batch_size=2048' "$RUNNER"
grep -q 'loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO}' "$RUNNER"
grep -q 'logger.wandb_enabled=true' "$RUNNER"
grep -q 'topology_override=' "$RUNNER"
grep -q 'segment_args=()' "$SUBMITTER"
grep -q 'segment_export=' "$SUBMITTER"
grep -Fq '"${segment_args[@]}"' "$SUBMITTER"
grep -q 'pretyche_force_on_policy_ratio_async_retry_20260707' "$RUNNER"
grep -q 'pretyche_force_on_policy_ratio_async_retry_20260707' "$SUBMITTER"
grep -q 'TEST_ONLY' "$SUBMITTER"
grep -q 'Refusing duplicate submission' "$SUBMITTER"

if grep -Fq '        --segment="$segment" \' "$SUBMITTER"; then
    printf 'Submitter still passes an unconditional segment.\n' >&2
    exit 1
fi
if grep -Fq 'cluster.segment_size=${SEGMENT_SIZE}' "$RUNNER"; then
    printf 'Runner still passes an unconditional topology override.\n' >&2
    exit 1
fi

if grep -Eq -- '--gres|RAY_CGRAPH_get_timeout|distributed_timeout|NCCL_TIMEOUT|moe_flex_dispatcher_backend|hybridep|moe_backend=' \
    "$RUNNER" "$SUBMITTER"; then
    printf 'Disallowed benchmark override found.\n' >&2
    exit 1
fi

printf 'MATRIX_CONTRACT_OK\n'
