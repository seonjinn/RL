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

test "$(awk 'END {print NR - 1}' "$CONTRACT")" -eq 8
test "$(awk -F '\t' 'NR > 1 {print $5}' "$CONTRACT" | sort | uniq -c | tr -s ' ' | sed 's/^ //')" = $'4 false\n4 true'
test "$(awk -F '\t' 'NR > 1 && $7 == 4 {count++} END {print count + 0}' "$CONTRACT")" -eq 8
test "$(awk -F '\t' 'NR > 1 && $9 == 2048 {count++} END {print count + 0}' "$CONTRACT")" -eq 8
test "$(awk -F '\t' 'NR > 1 && $10 == 20 {count++} END {print count + 0}' "$CONTRACT")" -eq 8
test "$(awk -F '\t' 'NR > 1 && $6 == $8 {count++} END {print count + 0}' "$CONTRACT")" -eq 8
test "$(awk -F '\t' 'NR > 1 && $6 == 2 {count++} END {print count + 0}' "$CONTRACT")" -eq 4
test "$(awk -F '\t' 'NR > 1 && $6 == 4 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 && $6 == 8 {count++} END {print count + 0}' "$CONTRACT")" -eq 2
test "$(awk -F '\t' 'NR > 1 {print $3}' "$CONTRACT" | sort -u | wc -l | tr -d ' ')" -eq 4
! awk -F '\t' 'NR > 1 {print $3}' "$CONTRACT" | grep -Eq -- '-[0-9]+n8g'

grep -q 'd4cfecf90db41cdf142629963b54b67ab479ab02' "$RUNNER"
grep -q 'bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510' "$RUNNER"
grep -q 'nemo_rl_nightly_20260630_0215.sqsh' "$RUNNER"
grep -q 'grpo.max_num_steps=20' "$RUNNER"
grep -q 'checkpointing.enabled=false' "$RUNNER"
grep -q 'policy.train_global_batch_size=2048' "$RUNNER"
grep -q 'loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO}' "$RUNNER"
grep -q 'logger.wandb_enabled=true' "$RUNNER"
grep -q 'TEST_ONLY' "$SUBMITTER"
grep -q 'Refusing duplicate submission' "$SUBMITTER"

! grep -Eq -- '--gres|RAY_CGRAPH_get_timeout|distributed_timeout|NCCL_TIMEOUT|moe_flex_dispatcher_backend|hybridep|moe_backend=' \
    "$RUNNER" "$SUBMITTER"

printf 'MATRIX_CONTRACT_OK\n'
