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
readonly PATCHED_LOCAL_LOGPROBS=${PATCHED_LOCAL_LOGPROBS:-0}
case "$PATCHED_LOCAL_LOGPROBS" in
  0)
    git -C "$REPO" diff --quiet "$SOURCE_COMMIT" HEAD -- . \
      ':(exclude)experiments/release_0_8_hybridep_oom_20261014' || { echo "Release source has changed; use PATCHED_LOCAL_LOGPROBS=1 for the logprob fix" >&2; exit 1; }
    ;;
  1)
    git -C "$REPO" diff --quiet "$SOURCE_COMMIT" HEAD -- . \
      ':(exclude)experiments/release_0_8_hybridep_oom_20261014' \
      ':(exclude)nemo_rl/distributed/model_utils.py' \
      ':(exclude)nemo_rl/models/automodel/train.py' \
      ':(exclude)tests/unit/distributed/test_local_logprob_memory.py' \
      ':(exclude)tests/unit/models/automodel/test_automodel_train.py' || { echo "Unexpected release source change" >&2; exit 1; }
    git -C "$REPO" diff --quiet HEAD -- nemo_rl/distributed/model_utils.py nemo_rl/models/automodel/train.py || { echo "Commit the patched source files before submission" >&2; exit 1; }
    ;;
  *) echo "PATCHED_LOCAL_LOGPROBS must be 0 or 1" >&2; exit 2 ;;
esac
CHECKOUT_COMMIT=$(git -C "$REPO" rev-parse HEAD)
readonly CHECKOUT_COMMIT

readonly ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
readonly EXP_ROOT=$ROOT/r0p8_hybridep_oom_20261014
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
readonly ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
readonly PARTITION=${PARTITION:-batch}
readonly WALLTIME=${WALLTIME:-04:00:00}
readonly SEGMENT=${SEGMENT:-}
readonly DEPENDENCY=${DEPENDENCY:-}
readonly OPTIMIZER_FOREACH=${OPTIMIZER_FOREACH:-default}
readonly LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK_SIZE:-default}
readonly VLLM_KV_CACHE_BYTES=${VLLM_KV_CACHE_BYTES:-default}
readonly POLICY_CPU_OFFLOAD=${POLICY_CPU_OFFLOAD:-default}
readonly POLICY_TP=${POLICY_TP:-default}
readonly POLICY_EP=${POLICY_EP:-default}
if [[ "$1" == minimax ]]; then
  readonly GPU_IDLE_EXEMPT_MINS=${GPU_IDLE_EXEMPT_MINS:-60}
else
  readonly GPU_IDLE_EXEMPT_MINS=${GPU_IDLE_EXEMPT_MINS:-0}
fi
case "$GPU_IDLE_EXEMPT_MINS" in
  0) readonly JOB_COMMENT=metrics ;;
  *)
    [[ "$1" == minimax ]] || { echo "GPU idle exemption is configured only for the measured MiniMax initialization" >&2; exit 2; }
    [[ "$GPU_IDLE_EXEMPT_MINS" =~ ^[1-9][0-9]*$ ]] && (( GPU_IDLE_EXEMPT_MINS > 30 && GPU_IDLE_EXEMPT_MINS <= 60 )) || {
      echo "GPU_IDLE_EXEMPT_MINS must be 0 or 31-60 (CW account cap is 60 minutes)" >&2
      exit 2
    }
    readonly JOB_COMMENT="metrics {\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"$GPU_IDLE_EXEMPT_MINS\",\"reason\":\"data_loading\",\"description\":\"MiniMax M2.7 64-GPU model loading and policy worker initialization exceeded 39 minutes in job 20250970\"}}"
    ;;
esac
case "$OPTIMIZER_FOREACH" in
  default) readonly FOREACH_OVERRIDE='' ;;
  false)
    [[ "$1" == minimax ]] || { echo "foreach override is only valid for MiniMax AdamW" >&2; exit 2; }
    readonly FOREACH_OVERRIDE='+policy.optimizer.kwargs.foreach=false'
    ;;
  *) echo "OPTIMIZER_FOREACH must be default or false" >&2; exit 2 ;;
esac
case "$LOGPROB_CHUNK_SIZE" in
  default) readonly LOGPROB_CHUNK_OVERRIDE='' ;;
  *)
    [[ "$LOGPROB_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "LOGPROB_CHUNK_SIZE must be default or a positive integer" >&2; exit 2; }
    readonly LOGPROB_CHUNK_OVERRIDE="policy.logprob_chunk_size=$LOGPROB_CHUNK_SIZE"
    ;;
esac
case "$VLLM_KV_CACHE_BYTES" in
  default) readonly VLLM_KV_OVERRIDE='' ;;
  *)
    [[ "$1" == qwen && "$VLLM_KV_CACHE_BYTES" =~ ^[1-9][0-9]*$ ]] || { echo "VLLM_KV_CACHE_BYTES requires Qwen and a positive byte count" >&2; exit 2; }
    readonly VLLM_KV_OVERRIDE="++policy.generation.vllm_kwargs.kv_cache_memory_bytes=$VLLM_KV_CACHE_BYTES"
    ;;
esac
case "$POLICY_CPU_OFFLOAD" in
  default) readonly CPU_OFFLOAD_OVERRIDE='' ;;
  true)
    [[ "$1" == minimax ]] || { echo "POLICY_CPU_OFFLOAD override is only valid for MiniMax" >&2; exit 2; }
    readonly CPU_OFFLOAD_OVERRIDE='policy.automodel_cfg.cpu_offload=true'
    ;;
  *) echo "POLICY_CPU_OFFLOAD must be default or true" >&2; exit 2 ;;
esac
case "$POLICY_TP" in
  default) readonly POLICY_TP_OVERRIDE='' ;;
  *)
    [[ "$POLICY_TP" =~ ^[1-9][0-9]*$ ]] || { echo "POLICY_TP must be default or a positive integer" >&2; exit 2; }
    readonly POLICY_TP_OVERRIDE="policy.automodel_cfg.tensor_parallel_size=$POLICY_TP"
    ;;
esac
case "$POLICY_EP" in
  default) readonly POLICY_EP_OVERRIDE='' ;;
  *)
    [[ "$POLICY_EP" =~ ^[1-9][0-9]*$ ]] || { echo "POLICY_EP must be default or a positive integer" >&2; exit 2; }
    readonly POLICY_EP_OVERRIDE="policy.automodel_cfg.expert_parallel_size=$POLICY_EP"
    ;;
esac
RUN_NAME="${1}-${DISPATCHER}-patch${PATCHED_LOCAL_LOGPROBS}-chunk${LOGPROB_CHUNK_SIZE}-kv${VLLM_KV_CACHE_BYTES}-offload${POLICY_CPU_OFFLOAD}-tp${POLICY_TP}-ep${POLICY_EP}-${OPTIMIZER_FOREACH}-${MAX_STEPS}step-$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_NAME
readonly RUN_DIR=$EXP_ROOT/results/$RUN_NAME

test -r "$IMAGE"
test -f "$REPO/$CONFIG"
mkdir -p "$RUN_DIR/logs" "$EXP_ROOT/slurm" "$EXP_ROOT/ray_logs"

export CONTAINER=$IMAGE
export MOUNTS=/lustre:/lustre,/home:/home
if [[ "$PATCHED_LOCAL_LOGPROBS" == 1 ]]; then
  export MOUNTS+=",$REPO/nemo_rl/distributed/model_utils.py:/opt/nemo-rl/nemo_rl/distributed/model_utils.py,$REPO/nemo_rl/models/automodel/train.py:/opt/nemo-rl/nemo_rl/models/automodel/train.py"
fi
export GPUS_PER_NODE=8
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_genai/nemo-rl/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE
if [[ "$1" == minimax ]]; then
  test -r "$HF_HUB_CACHE/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b/config.json"
else
  test -d "$HF_HUB_CACHE/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots"
fi
export BASE_LOG_DIR=$EXP_ROOT/ray_logs
export RAY_LOG_SYNC_FREQUENCY=30

cat > "$RUN_DIR/manifest.txt" <<EOF
source_commit=$SOURCE_COMMIT
checkout_commit=$CHECKOUT_COMMIT
container=$IMAGE
account=$ACCOUNT
hf_home=$HF_HOME
hf_hub_cache=$HF_HUB_CACHE
hf_hub_disable_implicit_token=$HF_HUB_DISABLE_IMPLICIT_TOKEN
config=$CONFIG
dispatcher=$DISPATCHER
nodes=$NODES
gpus_per_node=$GPUS_PER_NODE
max_steps=$MAX_STEPS
optimizer_foreach=$OPTIMIZER_FOREACH
patched_local_logprobs=$PATCHED_LOCAL_LOGPROBS
logprob_chunk_size=$LOGPROB_CHUNK_SIZE
vllm_kv_cache_bytes=$VLLM_KV_CACHE_BYTES
policy_cpu_offload=$POLICY_CPU_OFFLOAD
policy_tp=$POLICY_TP
policy_ep=$POLICY_EP
dependency=$DEPENDENCY
partition=$PARTITION
walltime=$WALLTIME
segment=$SEGMENT
job_comment=$JOB_COMMENT
EOF

export COMMAND="set -euo pipefail; cd /opt/nemo-rl; test -f 3rdparty/Gym-workspace/Gym/pyproject.toml; uv run --no-sync examples/run_grpo.py \
  --config '$CONFIG' \
  grpo.max_num_steps=$MAX_STEPS \
  checkpointing.enabled=false \
  policy.automodel_cfg.automodel_kwargs.backend.dispatcher=$DISPATCHER \
  $FOREACH_OVERRIDE \
  $LOGPROB_CHUNK_OVERRIDE \
  $VLLM_KV_OVERRIDE \
  $CPU_OFFLOAD_OVERRIDE \
  $POLICY_TP_OVERRIDE \
  $POLICY_EP_OVERRIDE \
  logger.log_dir='$RUN_DIR/logs' \
  logger.wandb_enabled=false \
  logger.monitor_gpus=true \
  2>&1 | tee '$RUN_DIR/run.log'"

readonly JOB_NAME=$ACCOUNT.r08-${1}-${DISPATCHER}
readonly OUTPUT=$EXP_ROOT/slurm/%x-%j.out
SUBMIT_ARGS=(
  --nodes="$NODES"
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --partition="$PARTITION"
  --time="$WALLTIME"
  --gres=gpu:8
  --exclusive
  --comment="$JOB_COMMENT"
  --output="$OUTPUT"
)
if [[ -n "$DEPENDENCY" ]]; then
  SUBMIT_ARGS+=(--dependency="$DEPENDENCY")
fi
if [[ -n "$SEGMENT" ]]; then
  [[ "$SEGMENT" =~ ^[1-9][0-9]*$ ]] || { echo "SEGMENT must be a positive integer" >&2; exit 2; }
  SUBMIT_ARGS+=(--segment "$SEGMENT")
fi

cd "$REPO"
sbatch --test-only "${SUBMIT_ARGS[@]}" ray.sub
JOB_ID=$(sbatch --parsable "${SUBMIT_ARGS[@]}" ray.sub)
readonly JOB_ID
printf 'job_id=%s\nrun_dir=%s\n' "$JOB_ID" "$RUN_DIR" | tee -a "$RUN_DIR/manifest.txt"
