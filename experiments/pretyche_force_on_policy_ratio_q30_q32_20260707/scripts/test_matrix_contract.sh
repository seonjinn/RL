#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONTRACT=$EXP_ROOT/manifests/config_contract.tsv
VALIDATOR=$SCRIPT_DIR/validate_config_contract.py
VALIDATOR_JOB=$SCRIPT_DIR/validate_config_contract.sbatch
RUNNER=$SCRIPT_DIR/run_force_on_policy_benchmark.sbatch
SUBMITTER=$SCRIPT_DIR/submit_force_on_policy_matrix.sh
WATCHER=$SCRIPT_DIR/watch_smoke_and_submit_performance.sh

test -f "$CONTRACT"
test -f "$VALIDATOR"
test -f "$VALIDATOR_JOB"
test -f "$RUNNER"
test -f "$SUBMITTER"
test -f "$WATCHER"

PYCACHE_DIR=$(mktemp -d)
trap 'rm -rf "$PYCACHE_DIR"' EXIT
PYTHONPYCACHEPREFIX=$PYCACHE_DIR python3 -m py_compile "$VALIDATOR"
bash -n "$VALIDATOR_JOB"
bash -n "$RUNNER"
bash -n "$SUBMITTER"
bash -n "$WATCHER"

test "$(awk 'END {print NR - 1}' "$CONTRACT")" -eq 4
test "$(awk -F '\t' 'NR > 1 {print $2}' "$CONTRACT" | sort -u | tr '\n' ' ')" = "qwen3-30ba3b qwen3-32b "
test "$(awk -F '\t' 'NR > 1 {print $4}' "$CONTRACT" | sort | uniq -c | tr -s ' ' | sed 's/^ //')" = $'2 false\n2 true'
test "$(awk -F '\t' 'NR > 1 && $5 == 4 && $6 == 4 {count++} END {print count + 0}' "$CONTRACT")" -eq 4
test "$(awk -F '\t' 'NR > 1 && $7 == 2048 {count++} END {print count + 0}' "$CONTRACT")" -eq 4
test "$(awk -F '\t' 'NR > 1 && $8 == 20 {count++} END {print count + 0}' "$CONTRACT")" -eq 4

! grep -Eq 'RAY_CGRAPH_get_timeout|distributed_timeout|NCCL_TIMEOUT|moe_flex_dispatcher_backend|hybridep' \
    "$VALIDATOR" "$VALIDATOR_JOB"

grep -q 'd4cfecf90db41cdf142629963b54b67ab479ab02' "$RUNNER"
grep -q 'nemo_rl_nightly_20260630_0215.sqsh' "$RUNNER"
grep -q 'nemo_rl_nightly_20260630_0215.sqsh' "$VALIDATOR_JOB"
grep -q 'nemo_rl_nightly_20260630_0215.sqsh' "$SUBMITTER"
grep -q 'bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510' "$RUNNER"
grep -q 'bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510' "$VALIDATOR_JOB"
grep -q 'bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510' "$SUBMITTER"
grep -q 'trap cleanup_unit_results EXIT' "$VALIDATOR_JOB"
grep -q 'policy.train_global_batch_size=2048' "$RUNNER"
grep -q 'loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO}' "$RUNNER"
grep -q 'checkpointing.enabled=false' "$RUNNER"
grep -q 'logger.wandb_enabled=true' "$RUNNER"
grep -q 'TEST_ONLY' "$SUBMITTER"
grep -q 'SMOKE_ONLY' "$SUBMITTER"
grep -q 'AFTEROK_JOB_ID' "$SUBMITTER"
grep -q -- '--dependency=afterok:' "$SUBMITTER"
grep -q 'verify_smoke_gate' "$WATCHER"
grep -q 'SMOKE_ONLY=0' "$WATCHER"
grep -q 'force_on_policy_ratio enabled' "$SUBMITTER"
grep -q 'Skipping prev_logprobs (force_on_policy_ratio=True)' "$SUBMITTER"

! grep -Eq -- '--gres|RAY_CGRAPH_get_timeout|distributed_timeout|NCCL_TIMEOUT|moe_flex_dispatcher_backend|hybridep' \
    "$RUNNER" "$SUBMITTER"

printf 'MATRIX_CONTRACT_OK\n'
