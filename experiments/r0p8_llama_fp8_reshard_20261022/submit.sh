#!/usr/bin/env bash
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
readonly REPO
readonly BRANCH=sna/r0p8-llama-fp8-reshard-smoke-20261022
readonly CONFIG=examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g-fp8-async-1off.yaml
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-b0617dab1.squashfs
readonly HF_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home
readonly DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/datasets
readonly MODULES_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/modules
readonly ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_llama_fp8_reshard_20261022
readonly ACCOUNT=coreai_dlalgo_llm
readonly JOB_NAME=$ACCOUNT.r08-llama8-fp8-reshard
readonly MODEL_SNAPSHOT=$HF_CACHE/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/config.json

cd "$REPO"
git -c fetch.recurseSubmodules=false pull --ff-only origin "$BRANCH"
git merge-base --is-ancestor 8c606fcfd3241ec9c65f8f67ee6ef43163d527ac HEAD
test -z "$(git status --porcelain)"
test -r "$IMAGE"
test -r "$MODEL_SNAPSHOT"
test -w "$DATASETS_CACHE"
mkdir -p "$MODULES_CACHE" "$ROOT/results" "$ROOT/ray_logs" "$ROOT/slurm"

SOURCE_COMMIT=$(git rev-parse HEAD)
CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)
RUN_NAME=llama3.1-8b-fp8-async-1off-reshard-10step-$(date -u +%Y%m%dT%H%M%SZ)
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
export PYTHONPATH=$REPO:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export BASE_LOG_DIR=$ROOT/ray_logs
export RAY_LOG_SYNC_FREQUENCY=30
export COMMAND="bash '$REPO/experiments/r0p8_llama_fp8_reshard_20261022/run_driver.sh' '$RUN_DIR'"

cat > "$RUN_DIR/manifest.txt" <<EOF
release_branch=r0.8.0
source_commit=$SOURCE_COMMIT
model=meta-llama/Llama-3.1-8B-Instruct
config=$CONFIG
config_sha256=$CONFIG_SHA256
container=$IMAGE
refit_transport=nccl_reshard
train_fp8_param=true
generation_precision=fp8
max_steps=10
nodes=2
gpus_per_node=8
account=$ACCOUNT
partition=batch
walltime=04:00:00
hf_home=$HF_HOME
hf_datasets_cache=$HF_DATASETS_CACHE
command=$COMMAND
EOF

readonly COMMENT="metrics {\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"60\",\"reason\":\"data_loading\",\"description\":\"r0.8 Llama3.1 FP8 checkpoint loading and NCCL reshard startup\"}}"
readonly SUBMIT_ARGS=(
  --nodes=2
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
