#!/usr/bin/env bash
set -euo pipefail

readonly MODEL=${1:?use 32b or 235b}
REPO=$(git rev-parse --show-toplevel)
readonly REPO
readonly BRANCH=sna/r0p8-qwen-async-reshard-20261022
readonly BASE_COMMIT=99ddc5ad711a0bbea6b145c3a8b4f062ef2b410c
readonly EXPERIMENT=$REPO/experiments/r0p8_qwen_async_reshard_20261022
readonly ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_qwen_async_reshard_20261022
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-b0617dab1.squashfs
readonly HF_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home
readonly DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/datasets
readonly MODULES_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/modules
readonly ACCOUNT=coreai_dlalgo_llm

case "$MODEL" in
  32b)
    readonly NODES=8
    readonly MODEL_NAME=Qwen/Qwen3-32B
    readonly SNAPSHOT=9216db5781bf21249d130ec9da846c4624c16137
    readonly JOB_NAME=$ACCOUNT.r08-q32-async-reshard
    export NRL_IGNORE_TP_ACCURACY_CHECK=1
    ;;
  235b)
    readonly NODES=32
    readonly MODEL_NAME=Qwen/Qwen3-235B-A22B
    readonly SNAPSHOT=8efa61729e24bd65b1d152b5ab5409052aa80e65
    readonly JOB_NAME=$ACCOUNT.r08-q235-async-reshard
    export NCCL_NVLS_ENABLE=0
    ;;
  *)
    printf 'unknown model: %s\n' "$MODEL" >&2
    exit 2
    ;;
esac

cd "$REPO"
git -c fetch.recurseSubmodules=false pull --ff-only origin "$BRANCH"
git merge-base --is-ancestor "$BASE_COMMIT" HEAD
test -z "$(git status --porcelain)"
test -r "$IMAGE"
test -r "$HF_CACHE/hub/models--${MODEL_NAME//\//--}/snapshots/$SNAPSHOT/config.json"
test -w "$DATASETS_CACHE"
mkdir -p "$MODULES_CACHE" "$ROOT/results" "$ROOT/ray_logs" "$ROOT/slurm"

readonly CONFIG=$REPO/examples/configs/recipes/llm/performance/grpo-qwen3-${MODEL}-${NODES}n8g-async-1off.yaml
test -r "$CONFIG"
SOURCE_COMMIT=$(git rev-parse HEAD)
CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)
RUN_NAME=qwen3-${MODEL}-${NODES}n8g-async-1off-triton-reshard-10step-$(date -u +%Y%m%dT%H%M%SZ)
readonly SOURCE_COMMIT CONFIG_SHA256 RUN_NAME
readonly RUN_DIR=$ROOT/results/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

export CONTAINER=$IMAGE
export MOUNTS=/lustre:/lustre,/home:/home
export GPUS_PER_NODE=8
export HF_HOME=$HF_CACHE
export HF_HUB_CACHE=$HF_CACHE/hub
export HF_DATASETS_CACHE=$DATASETS_CACHE
export HF_MODULES_CACHE=$MODULES_CACHE
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH=$REPO
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export BASE_LOG_DIR=$ROOT/ray_logs
export RAY_LOG_SYNC_FREQUENCY=30
export COMMAND="bash '$EXPERIMENT/run_driver.sh' '$MODEL' '$RUN_DIR'"

cat > "$RUN_DIR/manifest.txt" <<EOF
release_branch=r0.8.0
release_base_commit=$BASE_COMMIT
source_commit=$SOURCE_COMMIT
model=$MODEL_NAME
config=$CONFIG
config_sha256=$CONFIG_SHA256
container=$IMAGE
container_source_commit=b0617dab1
source_override=PYTHONPATH:$REPO
generation_moe_backend=triton
refit_transport=nccl_reshard
async_grpo=true
max_steps=10
nodes=$NODES
gpus_per_node=8
account=$ACCOUNT
partition=batch
walltime=04:00:00
hf_home=$HF_HOME
hf_datasets_cache=$HF_DATASETS_CACHE
command=$COMMAND
EOF

readonly COMMENT="metrics {\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"60\",\"reason\":\"data_loading\",\"description\":\"r0.8 Qwen3 ${MODEL} checkpoint loading and async NCCL reshard startup\"}}"
readonly SUBMIT_ARGS=(
  --nodes="$NODES"
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --partition=batch
  --time=04:00:00
  --gres=gpu:8
  --exclusive
  --segment=16
  --comment="$COMMENT"
  --output="$ROOT/slurm/%x-%j.out"
)

sbatch --test-only "${SUBMIT_ARGS[@]}" ray.sub
JOB_ID=$(sbatch --parsable "${SUBMIT_ARGS[@]}" ray.sub)
readonly JOB_ID
printf 'job_id=%s\nrun_dir=%s\n' "$JOB_ID" "$RUN_DIR" | tee -a "$RUN_DIR/manifest.txt"
