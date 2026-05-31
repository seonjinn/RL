#!/bin/bash
# Extract per-stage timings from job 11825251 driver log
# Usage: ssh cw-dfw-cs-001-vscode-02 'bash -s' < extract_hybridep_fuseloss_timings.sh

set -u
JOB=${1:-11825251}
REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/repos/nemo-rl-qwen-swe-fuseloss-port
LOG=$REPO/$JOB-logs/ray-driver.log

echo "=== job $JOB ==="
echo
echo "=== sanity: GYM-RESTORE fired ==="
grep -m1 "GYM-RESTORE" $REPO/$JOB-logs/ray-head.log || echo "(not found in ray-head.log)"
echo
echo "=== sanity: SequencePackingFusionLossWrapper instantiated ==="
grep -m2 -E "FusionLossWrapper|fuse_loss" "$LOG" | head -5
echo
echo "=== sanity: HybridEP flex dispatcher active ==="
grep -m3 -iE "HybridEP|moe_flex|hybridep" "$LOG" | head -5
echo
echo "=== step landings (elapsed_steps) ==="
grep -oE 'elapsed_steps=[0-9]+' "$LOG" | sort -u | tail -25
echo
echo "=== per-stage timing rows (last 20) ==="
grep -E "TIMING|policy_training|policy_logprob|generation|weight_sync|step.*finished" "$LOG" | tail -40
echo
echo "=== failures (if any) ==="
grep -E "Traceback|Error|FAILED|Killed|RuntimeError|OOM|FileNotFoundError|missing.*pyproject" "$LOG" | head -10
