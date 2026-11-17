#!/bin/bash
# Q30 SWE V2 tail-drop A/B at GBS 32, staleness 2, L0, no drafter. Run on the HSG login node.
# Both arms run the same NeMo-RL commit, Gym 5b42623075 and nv-OpenHands a8b3da24df (the run-ahead
# control). Only data.train.data_path differs. SFX names a replicate (for example -r2).
#   WANT=<full sha> ARMS=both [SFX=-r2] [ACCOUNT=...] bash submit.sh
set -o pipefail
U=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
WANT=${WANT:?set WANT to the full commit sha}
BR=sna/swe-stale2-taildrop-1116
ARMS=${ARMS:?set ARMS to s2, tail or both}
DATA=$U/datasets/swe/balanced_language_drop5_tail5.jsonl
export ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl} ENTRY=examples/run_grpo_single_controller.py NUM_VLLM_REPLICAS=16

[ -s "$DATA" ] || { echo "missing $DATA"; exit 1; }

sync_clone() {
  local c=$1
  [ -z "$(squeue -u sna -h -o %Z | grep "/$c$")" ] || { echo "$c busy"; return 1; }
  cd $U/$c || return 1
  git fetch -q origin $BR || return 1
  git checkout -q -B $BR $WANT || { echo "$c checkout failed"; return 1; }
  git submodule sync --recursive -q || { echo "$c submodule sync failed"; return 1; }
  git submodule update --init --recursive -q || { echo "$c submodule failed"; return 1; }
  [[ $(git rev-parse HEAD) == $WANT ]] || { echo "$c HEAD mismatch"; return 1; }
  [[ $(git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD) == $(git rev-parse HEAD:3rdparty/Gym-workspace/Gym) ]] || { echo "$c Gym pin mismatch"; return 1; }
  git diff --cached --quiet HEAD || { echo "$c index differs from HEAD"; return 1; }
  echo "$c HEAD=$(git rev-parse --short=10 HEAD) Gym=$(git -C 3rdparty/Gym-workspace/Gym rev-parse --short=10 HEAD)"
}

jobid() { sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p' | tail -1; }
SWE=experiments/specdec_perf_20261020/submit_swe.sh
C=experiments/specdec_perf_20261020/configs/qwen3_30ba3b_swe
sub() {
  local out id
  out=$(cd $U/$1 && $SWE $C/$2.yaml swe-taildrop1116-sc-r16-rf-eval450-$3${SFX} 20 2>&1)
  id=$(echo "$out" | jobid)
  echo "$(date +%H:%M) $1 $3${SFX} -> ${id:-NONE} $(echo "$out" | grep -iE 'rror|Gym venvs' | tail -1)"
}

declare -A CLONE=([s2]=integ-verify6 [tail]=integ-verify7)
declare -A CFG=([s2]=swe_traj_l0_staleness2_drop5 [tail]=swe_traj_l0_staleness2_droptail)
declare -A RUN=([s2]=drop5-baseline-l0-staleness2 [tail]=droptail-baseline-l0-staleness2)

[[ $ARMS == both ]] && ARMS="s2 tail"
for a in $ARMS; do sync_clone ${CLONE[$a]} || exit 1; done
for a in $ARMS; do sub ${CLONE[$a]} ${CFG[$a]} ${RUN[$a]}; done
echo ALL_SUBMITTED----