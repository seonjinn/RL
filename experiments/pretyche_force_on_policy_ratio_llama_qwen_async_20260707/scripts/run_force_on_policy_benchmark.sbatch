#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=36x2-a01r
#SBATCH --exclusive
#SBATCH --comment=metrics
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/users/sna/pretyche_force_on_policy_ratio_async_retry_20260707/slurm/%x-%j.out

set -euo pipefail

: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${RUN_KEY:?RUN_KEY is required}"
: "${MODEL:?MODEL is required}"
: "${MODE:?MODE is required}"
: "${FORCE_ON_POLICY_RATIO:?FORCE_ON_POLICY_RATIO is required}"
: "${EXPECTED_NODES:?EXPECTED_NODES is required}"
: "${EXPECTED_RUNNER_SHA:?EXPECTED_RUNNER_SHA is required}"

SEGMENT_SIZE=${SEGMENT_SIZE-}

[[ $FORCE_ON_POLICY_RATIO == false || $FORCE_ON_POLICY_RATIO == true ]]
test "$MODE" = async1off
test "$SLURM_NNODES" = "$EXPECTED_NODES"
case "$MODEL" in
    qwen3-30ba3b)
        test "$SEGMENT_SIZE" = 2
        ;;
    llama3.1-8b|qwen3-32b)
        test -z "$SEGMENT_SIZE"
        ;;
    *)
        printf 'Unsupported model: %s\n' "$MODEL" >&2
        exit 2
        ;;
esac

BASE=/lustre/fsw/coreai_dlalgo_llm/users/sna
REPO=$BASE/nemo-rl-main-pr3030-q235-20260701
EXP_ROOT=$BASE/pretyche_force_on_policy_ratio_async_retry_20260707
RESULT_DIR=$EXP_ROOT/results/$RUN_KEY/20step
LOG_DIR=$RESULT_DIR/logs
RUN_LOG=$RESULT_DIR/run.log
CONTAINER=$BASE/containers/nemo_rl_nightly_20260630_0215.sqsh
RUNNER=$EXP_ROOT/scripts/run_force_on_policy_benchmark.sbatch
CONFIG_PATH=examples/configs/recipes/llm/performance/${CONFIG_NAME}.yaml
EXPECTED_REPO_SHA=d4cfecf90db41cdf142629963b54b67ab479ab02
EXPECTED_CONTAINER_SHA=bf841732e6615aca7a00a6c4ba47d7298a118137fc914296a4083172132ff510

mkdir -p "$EXP_ROOT/slurm" "$LOG_DIR" "$BASE/uv_cache"
if [[ -r "$HOME/.nemo_rl_tokens" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.nemo_rl_tokens"
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in ~/.nemo_rl_tokens}"
: "${HF_TOKEN:?HF_TOKEN must be set in ~/.nemo_rl_tokens}"

test -s "$CONTAINER"
test "$(sha256sum "$CONTAINER" | awk '{print $1}')" = "$EXPECTED_CONTAINER_SHA"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" rev-parse '@{u}')" = "$EXPECTED_REPO_SHA"
test "$(git -C "$REPO" remote get-url origin)" = https://github.com/seonjinn/RL.git
test -z "$(git -C "$REPO" status --porcelain --ignore-submodules=untracked)"
! git -C "$REPO" submodule status --recursive | grep -Eq '^[-+U]'

export CONTAINER
export MOUNTS=/lustre:/lustre,/project:/project
export GPUS_PER_NODE=4
export BASE_LOG_DIR=$EXP_ROOT/slurm
export RAY_LOG_SYNC_FREQUENCY=30
export HF_HOME=$BASE/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}
export NRL_MEGATRON_CHECKPOINT_DIR=$BASE/megatron_checkpoints/$EXPECTED_REPO_SHA/force_on_policy_ratio_async_retry/$RUN_KEY
export UV_CACHE_DIR=$BASE/uv_cache
export UV_HTTP_TIMEOUT=300
export UV_LOCK_TIMEOUT=3600
export XDG_CACHE_HOME=/tmp/nemo_rl_force_ratio_${RUN_KEY}_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
export CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export FLASHINFER_CACHE_DIR=$XDG_CACHE_HOME/flashinfer
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export NVTE_CUDA_ARCHS=100
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray
export NCCL_NVLS_ENABLE=0
export WANDB_MODE=online

mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"

topology_override=
if [[ -n $SEGMENT_SIZE ]]; then
    topology_override="cluster.segment_size=$SEGMENT_SIZE"
fi

export COMMAND="set -euo pipefail; cd '$REPO'; \
if [[ -r \"\$HOME/.nemo_rl_tokens\" ]]; then source \"\$HOME/.nemo_rl_tokens\"; fi; \
actual_repo_sha=\$(git rev-parse HEAD); \
echo RUN_KEY='$RUN_KEY'; \
echo MODEL='$MODEL'; \
echo MODE='$MODE'; \
echo FORCE_ON_POLICY_RATIO='$FORCE_ON_POLICY_RATIO'; \
echo SEGMENT_SIZE='${SEGMENT_SIZE:-none}'; \
echo REPO_SHA=\$actual_repo_sha; \
echo CONTAINER_SHA='$EXPECTED_CONTAINER_SHA'; \
test \"\$actual_repo_sha\" = '$EXPECTED_REPO_SHA'; \
test \"\$(git rev-parse '@{u}')\" = '$EXPECTED_REPO_SHA'; \
test -z \"\$(git status --porcelain --ignore-submodules=untracked)\"; \
! git submodule status --recursive | grep -Eq '^[-+U]'; \
git submodule status --recursive; \
export HF_HOME='$HF_HOME'; \
export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'; \
export HF_HUB_CACHE='$HF_HUB_CACHE'; \
export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'; \
export NRL_MEGATRON_CHECKPOINT_DIR='$NRL_MEGATRON_CHECKPOINT_DIR'; \
export HF_TOKEN=\"\${HF_TOKEN:-}\"; \
export HUGGING_FACE_HUB_TOKEN=\"\${HUGGING_FACE_HUB_TOKEN:-\$HF_TOKEN}\"; \
export WANDB_API_KEY=\"\${WANDB_API_KEY:-}\"; \
export WANDB_MODE=online; \
export UV_CACHE_DIR='$UV_CACHE_DIR'; \
export UV_HTTP_TIMEOUT='$UV_HTTP_TIMEOUT'; \
export UV_LOCK_TIMEOUT='$UV_LOCK_TIMEOUT'; \
export XDG_CACHE_HOME='$XDG_CACHE_HOME'; \
export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'; \
export TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR'; \
export CUDA_CACHE_PATH='$CUDA_CACHE_PATH'; \
export FLASHINFER_CACHE_DIR='$FLASHINFER_CACHE_DIR'; \
export VLLM_CACHE_ROOT='$VLLM_CACHE_ROOT'; \
export NVTE_CUDA_ARCHS='$NVTE_CUDA_ARCHS'; \
export NCCL_NVLS_ENABLE='$NCCL_NVLS_ENABLE'; \
if [[ -f /usr/local/cuda-13.2/compat/lib.real/libcuda.so.1 ]]; then export LD_PRELOAD=\"/usr/local/cuda-13.2/compat/lib.real/libcuda.so.1\${LD_PRELOAD:+ }\${LD_PRELOAD:-}\"; fi; \
NRL_FORCE_REBUILD_VENVS=false uv run --locked examples/run_grpo.py \
  --config '$CONFIG_PATH' \
  grpo.max_num_steps=20 \
  checkpointing.enabled=false \
  logger.log_dir='$LOG_DIR' \
  logger.wandb_enabled=true \
  logger.wandb.project='sna-force-on-policy-llama-qwen-async-gb200' \
  logger.wandb.name='pretyche-force-ratio-retry-${RUN_KEY}-20s' \
  policy.train_global_batch_size=2048 \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  ${topology_override:+$topology_override }2>&1 | tee '$RUN_LOG'"

cd "$REPO"
exec bash ray.sub
