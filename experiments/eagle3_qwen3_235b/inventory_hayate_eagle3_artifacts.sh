#!/usr/bin/env bash
set -euo pipefail

# Inventory Hayate/Hiso Eagle3-related worktrees and artifacts.
#
# Run this on a host where the relevant /lustre paths are mounted, or set
# REMOTE_HOST to run through ssh:
#
#   REMOTE_HOST=cw-dfw-cs-001-vscode-01 \
#   bash experiments/eagle3_qwen3_235b/inventory_hayate_eagle3_artifacts.sh
#
# The original oci-hsg hostname was not resolvable from this workspace, so this
# script keeps paths configurable.

MODEL_OPT_DIR="${MODEL_OPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer}"
MODEL_OPT_CANDIDATES="${MODEL_OPT_CANDIDATES:-$MODEL_OPT_DIR /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/Model-Optimizer /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/NVIDIA/TensorRT-Model-Optimizer-worktrees/eagle3}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge}"
NEMO_RL_DIR="${NEMO_RL_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/code/nemo-rl-internal-worktrees/feat-eagle3-online-specdec}"
DRAFT_MODELS_DIR="${DRAFT_MODELS_DIR:-$NEMO_RL_DIR/models}"
REMOTE_HOST="${REMOTE_HOST:-}"
HAYATE_INVENTORY_OUT="${HAYATE_INVENTORY_OUT:-}"

if [[ -n "$HAYATE_INVENTORY_OUT" ]]; then
  mkdir -p "$(dirname "$HAYATE_INVENTORY_OUT")"
  exec > "$HAYATE_INVENTORY_OUT"
fi

remote_script='
set -euo pipefail
echo "## Host"
hostname || true
date || true

echo
echo "## Model-Optimizer"
for candidate in $MODEL_OPT_CANDIDATES; do
  if [ -d "$candidate" ]; then
    MODEL_OPT_DIR="$candidate"
    break
  fi
done
if [ -d "$MODEL_OPT_DIR" ]; then
  echo "path=$MODEL_OPT_DIR"
  git --git-dir="$MODEL_OPT_DIR/.git" --work-tree="$MODEL_OPT_DIR" branch --show-current 2>/dev/null || true
  git --git-dir="$MODEL_OPT_DIR/.git" --work-tree="$MODEL_OPT_DIR" log -5 --oneline --decorate 2>/dev/null || true
  echo "-- changed files --"
  git --git-dir="$MODEL_OPT_DIR/.git" --work-tree="$MODEL_OPT_DIR" status --short examples/speculative_decoding 2>/dev/null || true
  git --git-dir="$MODEL_OPT_DIR/.git" --work-tree="$MODEL_OPT_DIR" diff --name-status HEAD -- examples/speculative_decoding 2>/dev/null || true
  echo "-- qwen3 eagle configs --"
  find "$MODEL_OPT_DIR/examples/speculative_decoding" -maxdepth 1 -type f -name "eagle_config_qwen3*.json" 2>/dev/null | sort | sed -n "1,120p" || true
  echo "-- dapo/generation/slurm files --"
  find "$MODEL_OPT_DIR/examples/speculative_decoding/prepare_input_conversations" "$MODEL_OPT_DIR/examples/speculative_decoding/slurm" -maxdepth 1 -type f 2>/dev/null | sort | sed -n "1,160p" || true
  echo "-- eagle3 experiment files --"
  find "$MODEL_OPT_DIR/experiments/eagle3" -maxdepth 3 -type f 2>/dev/null | sort | sed -n "1,120p" || true
else
  echo "missing: $MODEL_OPT_DIR"
fi

echo
echo "## SpecForge"
if [ -d "$SPEC_FORGE_DIR" ]; then
  echo "path=$SPEC_FORGE_DIR"
  git --git-dir="$SPEC_FORGE_DIR/.git" --work-tree="$SPEC_FORGE_DIR" branch --show-current 2>/dev/null || true
  git --git-dir="$SPEC_FORGE_DIR/.git" --work-tree="$SPEC_FORGE_DIR" log -5 --oneline --decorate 2>/dev/null || true
  echo "-- qwen3 configs --"
  find "$SPEC_FORGE_DIR/configs" -maxdepth 1 -type f -iname "*qwen3*eagle3*.json" 2>/dev/null | sort || true
  echo "-- output configs --"
  find "$SPEC_FORGE_DIR/outputs" -maxdepth 3 -type f -name config.json 2>/dev/null | sort | sed -n "1,120p" || true
else
  echo "missing: $SPEC_FORGE_DIR"
fi

echo
echo "## NeMo-RL"
if [ -d "$NEMO_RL_DIR" ]; then
  echo "path=$NEMO_RL_DIR"
  git --git-dir="$NEMO_RL_DIR/.git" --work-tree="$NEMO_RL_DIR" branch --show-current 2>/dev/null || true
  git --git-dir="$NEMO_RL_DIR/.git" --work-tree="$NEMO_RL_DIR" log -5 --oneline --decorate 2>/dev/null || true
  echo "-- specdec references --"
  find "$NEMO_RL_DIR" -path "*/__pycache__" -prune -o -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" \) -print 2>/dev/null \
    | xargs grep -nE "specdec|speculative_config|Eagle3|EAGLE3|draft_model" 2>/dev/null \
    | sed -n "1,160p" || true
  echo "-- draft model configs --"
  find "$DRAFT_MODELS_DIR" -maxdepth 5 -type f -name config.json 2>/dev/null | sort | sed -n "1,160p" || true
else
  echo "missing: $NEMO_RL_DIR"
fi
'

if [[ -n "$REMOTE_HOST" ]]; then
  ssh -S none -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST" \
    "MODEL_OPT_DIR='$MODEL_OPT_DIR' MODEL_OPT_CANDIDATES='$MODEL_OPT_CANDIDATES' SPEC_FORGE_DIR='$SPEC_FORGE_DIR' NEMO_RL_DIR='$NEMO_RL_DIR' DRAFT_MODELS_DIR='$DRAFT_MODELS_DIR' bash -s" \
    <<< "$remote_script"
else
  MODEL_OPT_DIR="$MODEL_OPT_DIR" MODEL_OPT_CANDIDATES="$MODEL_OPT_CANDIDATES" SPEC_FORGE_DIR="$SPEC_FORGE_DIR" NEMO_RL_DIR="$NEMO_RL_DIR" DRAFT_MODELS_DIR="$DRAFT_MODELS_DIR" bash -s \
    <<< "$remote_script"
fi
