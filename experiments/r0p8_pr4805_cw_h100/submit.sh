#!/usr/bin/env bash

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
readonly REPO
readonly PR_COMMIT=dbad3b423a6b9c0137660499f916f6965cfd99d7
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
readonly HF_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home
readonly ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_pr4805_cw_h100
readonly CONFIG=$REPO/examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
readonly ACCOUNT=coreai_dlalgo_nemorl
readonly BRANCH=sna/r0p8-pr4805-cw-h100-20261021
readonly JOB_NAME=$ACCOUNT.r08-q30-fi-trtllm

cd "$REPO"
git -c fetch.recurseSubmodules=false pull --ff-only origin "$BRANCH"
git merge-base --is-ancestor "$PR_COMMIT" HEAD
test -z "$(git status --porcelain)"
test -r "$IMAGE"
test -r "$CONFIG"
test -r "$HF_CACHE/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json"
grep -q 'moe_backend: flashinfer_trtllm' "$CONFIG"
grep -q 'NVLINK_DOMAIN_SIZE: "8"' "$CONFIG"
grep -q 'USE_MNNVL: "0"' "$CONFIG"

SOURCE_COMMIT=$(git rev-parse HEAD)
CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)
RUN_NAME=qwen3-30ba3b-bf16-flashinfer-trtllm-4n8g-2step-$(date -u +%Y%m%dT%H%M%SZ)
readonly SOURCE_COMMIT CONFIG_SHA256 RUN_NAME
readonly RUN_DIR=$ROOT/results/$RUN_NAME
mkdir -p "$RUN_DIR/logs" "$ROOT/ray_logs" "$ROOT/slurm"

export CONTAINER=$IMAGE
export MOUNTS=/lustre:/lustre,/home:/home
export GPUS_PER_NODE=8
export HF_HOME=$HF_CACHE
export HF_HUB_CACHE=$HF_CACHE/hub
export HF_DATASETS_CACHE=$HF_CACHE/datasets
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export BASE_LOG_DIR=$ROOT/ray_logs
export RAY_LOG_SYNC_FREQUENCY=30
unset HF_HUB_OFFLINE NRL_FORCE_REBUILD_VENVS

cat > "$RUN_DIR/manifest.txt" <<EOF
pr=4805
pr_merge_commit=$PR_COMMIT
source_commit=$SOURCE_COMMIT
config=$CONFIG
config_sha256=$CONFIG_SHA256
container=$IMAGE
hf_home=$HF_HOME
model=Qwen/Qwen3-30B-A3B
precision=bfloat16
generation_backend=vllm
moe_backend=flashinfer_trtllm
nodes=4
gpus_per_node=8
generation_tp=2
policy_ep=8
max_steps=2
partition=batch
walltime=04:00:00
EOF

export COMMAND="set -euo pipefail; cd /opt/nemo-rl; test -f 3rdparty/Gym-workspace/Gym/pyproject.toml; uv run --no-sync examples/run_grpo.py \
  --config '$CONFIG' \
  grpo.max_num_steps=2 \
  checkpointing.enabled=false \
  logger.log_dir='$RUN_DIR/logs' \
  logger.wandb_enabled=false \
  > '$RUN_DIR/run.log' 2>&1"

readonly COMMENT='metrics {"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Qwen3-30B-A3B checkpoint loading and vLLM FlashInfer TRTLLM kernel warmup on H100"}}'
readonly OUTPUT=$ROOT/slurm/%x-%j.out
readonly SUBMIT_ARGS=(
  --nodes=4
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --partition=batch
  --time=04:00:00
  --gres=gpu:8
  --exclusive
  --segment=16
  --comment="$COMMENT"
  --output="$OUTPUT"
)

sbatch --test-only "${SUBMIT_ARGS[@]}" ray.sub
JOB_ID=$(sbatch --parsable "${SUBMIT_ARGS[@]}" ray.sub)
printf 'job_id=%s\nrun_dir=%s\n' "$JOB_ID" "$RUN_DIR" | tee -a "$RUN_DIR/manifest.txt"
