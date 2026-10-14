#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {qwen|minimax} {hybridep|deepep} MAX_STEPS" >&2
  exit 2
fi

case "$1" in
  qwen)
    readonly CONFIG=examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml
    readonly NODES=4
    ;;
  minimax)
    readonly CONFIG=examples/configs/recipes/llm/grpo-minimax-m27-dapo-8n8g-automodel.yaml
    readonly NODES=8
    ;;
  *) echo "Unknown model: $1" >&2; exit 2 ;;
esac

case "$2" in
  hybridep|deepep) readonly DISPATCHER=$2 ;;
  *) echo "Unknown dispatcher: $2" >&2; exit 2 ;;
esac

[[ "$3" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_STEPS must be positive" >&2; exit 2; }
readonly MAX_STEPS=$3
REPO=$(git rev-parse --show-toplevel)
readonly REPO
readonly SOURCE_COMMIT=19666f86ca3b1988880b2b8b9107c575608c4ab0
git -C "$REPO" merge-base --is-ancestor "$SOURCE_COMMIT" HEAD || { echo "Wrong source base" >&2; exit 1; }
git -C "$REPO" diff --quiet "$SOURCE_COMMIT" HEAD -- . \
  ':(exclude)experiments/release_0_8_hybridep_oom_20261014' || { echo "Release source has changed" >&2; exit 1; }
CHECKOUT_COMMIT=$(git -C "$REPO" rev-parse HEAD)
readonly CHECKOUT_COMMIT

readonly ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
readonly EXP_ROOT=$ROOT/r0p8_hybridep_oom_20261014
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
RUN_NAME="${1}-${DISPATCHER}-${MAX_STEPS}step-$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_NAME
readonly RUN_DIR=$EXP_ROOT/results/$RUN_NAME

test -r "$IMAGE"
test -f "$REPO/$CONFIG"
mkdir -p "$RUN_DIR/logs" "$EXP_ROOT/slurm" "$EXP_ROOT/ray_logs"

export CONTAINER=$IMAGE
export MOUNTS=/lustre:/lustre,/home:/home
export GPUS_PER_NODE=8
export HF_HOME=$ROOT/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export BASE_LOG_DIR=$EXP_ROOT/ray_logs
export RAY_LOG_SYNC_FREQUENCY=30

cat > "$RUN_DIR/manifest.txt" <<EOF
source_commit=$SOURCE_COMMIT
checkout_commit=$CHECKOUT_COMMIT
container=$IMAGE
config=$CONFIG
dispatcher=$DISPATCHER
nodes=$NODES
gpus_per_node=$GPUS_PER_NODE
max_steps=$MAX_STEPS
EOF

export COMMAND="set -euo pipefail; cd '$REPO'; NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py \
  --config '$CONFIG' \
  grpo.max_num_steps=$MAX_STEPS \
  checkpointing.enabled=false \
  policy.automodel_cfg.automodel_kwargs.backend.dispatcher=$DISPATCHER \
  logger.log_dir='$RUN_DIR/logs' \
  logger.wandb_enabled=false \
  logger.monitor_gpus=true \
  2>&1 | tee '$RUN_DIR/run.log'"

readonly JOB_NAME=coreai_dlalgo_nemorl.r08-${1}-${DISPATCHER}
readonly OUTPUT=$EXP_ROOT/slurm/%x-%j.out
readonly SUBMIT_ARGS=(
  --nodes="$NODES"
  --account=coreai_dlalgo_nemorl
  --job-name="$JOB_NAME"
  --partition=batch
  --time=04:00:00
  --gres=gpu:8
  --exclusive
  --comment=metrics
  --output="$OUTPUT"
)

cd "$REPO"
sbatch --test-only "${SUBMIT_ARGS[@]}" ray.sub
JOB_ID=$(sbatch --parsable "${SUBMIT_ARGS[@]}" ray.sub)
readonly JOB_ID
printf 'job_id=%s\nrun_dir=%s\n' "$JOB_ID" "$RUN_DIR" | tee -a "$RUN_DIR/manifest.txt"
