#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MAX_STEPS=${MAX_STEPS:-5}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-pr3294-fused-refit-spike-20260823}
GIT_REMOTE=${GIT_REMOTE:-fork}
EXPECTED_HEAD=${EXPECTED_HEAD:-}

if [[ "${ACTION}" == render ]]; then
  cat <<EOF
model=Qwen3-30B-A3B
mode=sync_colocated_cuda_ipc
arm=mxfp8_full_pr3294
max_steps=${MAX_STEPS}
profile_step=3
cuda_graphs=enabled
EOF
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-/home/sna/RL-pr3294-fused-refit-spike-20260823}
CONTAINER=${CONTAINER:-${ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh}
HF_HOME=${HF_HOME:-${ROOT}/hf_home}
WANDB_HOME=${WANDB_HOME:-${ROOT}/wandb_netrc_home}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/experiments/mxfp8-fused-refit-spike}
RUN_ROOT=${RUN_ROOT:-${RESULT_ROOT}/${RUN_GROUP}/profile}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n3_post}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/sna}
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml

git -C "${REPO}" fetch "${GIT_REMOTE}" "${BRANCH}"
git -C "${REPO}" pull --ff-only "${GIT_REMOTE}" "${BRANCH}"
git -C "${REPO}" submodule update --init --recursive
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "${GIT_REMOTE}/${BRANCH}")
test "${LOCAL_HEAD}" = "${REMOTE_HEAD}"
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)"
if git -C "${REPO}" submodule status --recursive | grep -q '^[+-U]'; then
  echo "All submodules must be initialized at pinned revisions" >&2
  exit 2
fi
for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${HF_HOME}" "${WANDB_HOME}/.netrc"; do
  test -e "${path}"
done

mkdir -p "${RUN_ROOT}/logs"
cat >"${RUN_ROOT}/metadata.env" <<EOF
source_commit=${LOCAL_HEAD}
branch=${BRANCH}
container=${CONTAINER}
hardware=GB200
model=Qwen3-30B-A3B
mode=sync_colocated_cuda_ipc
arm=mxfp8_full_pr3294
nodes=4
gpus_per_node=4
training_precision=bf16
rollout_precision=mxfp8
quantization_scope=routed_experts_only
cuda_graphs=enabled
moe_backend=flashinfer_trtllm
refit_prequantize=true
refit_persistent_ipc_buffers=true
refit_batched_moe_shuffle=true
refit_loader_route_cache=true
profile_step=3
max_steps=${MAX_STEPS}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=\${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export NRL_MXFP8_BATCHED_SHUFFLE=1
export NRL_MXFP8_SHUFFLE_VERIFY=0
export NRL_NSYS_WORKER_PATTERNS='megatron_policy_worker,vllm_generation_worker'
export NRL_NSYS_PROFILE_STEP_RANGE='3:4'
export NRL_NSYS_EXTRA_OPTIONS='{"gpu-metrics-device":"all","cuda-memory-usage":"true","cpuctxsw":"none"}'
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VENV_DIR=${LOCAL_SCRATCH}/nemo-rl-worker-cache/pr3294-fused-refit-${LOCAL_HEAD}
export NVTE_CUDA_ARCHS=100
export PYTHONPATH=${REPO}
export TORCH_CUDA_ARCH_LIST=10.0
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
export UV_LOCK_TIMEOUT=7200
unset UV_PROJECT_ENVIRONMENT WANDB_API_KEY
mkdir -p "\${NEMO_RL_VENV_DIR}" "\${UV_CACHE_DIR}" "\${UV_PYTHON_INSTALL_DIR}"
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config ${CONFIG} \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4 \
  cluster.segment_size=4 \
  policy.precision=bfloat16 \
  policy.train_global_batch_size=2048 \
  grpo.async_grpo.enabled=false \
  data_plane.enabled=false \
  policy.generation.colocated.enabled=true \
  policy.generation.refit_transport=null \
  policy.generation.real_quant_export_cpu_offload=false \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  policy.generation.vllm_cfg.precision=fp8 \
  ++policy.generation.vllm_cfg.is_mx=true \
  ++policy.generation.vllm_cfg.refit_prequantize=true \
  ++policy.generation.vllm_cfg.refit_cache_loader_routes=true \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  ++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \
  ++policy.refit_buffer_size_gb=4 \
  ++policy.refit_persistent_ipc_buffers=true \
  ++policy.megatron_cfg.refit_slim_offload_after=true \
  policy.megatron_cfg.expert_tensor_parallel_size=1 \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.use_importance_sampling_correction=true \
  ++grpo.skip_reference_policy_logprobs_calculation=false \
  grpo.max_num_steps=${MAX_STEPS} \
  grpo.seed=42 \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.log_dir=${RUN_ROOT}/logs \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true
EOF
)

export CONTAINER
export MOUNTS=/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc
export CONTAINER_REMAP_ROOT=1
export COMMAND
export GPUS_PER_NODE=4
export BASE_LOG_DIR=${RUN_ROOT}
export RAY_LOG_SYNC_FREQUENCY=30
export SETUP_COMMAND="mkdir -p ${LOCAL_SCRATCH}/nemo-rl-worker-cache ${LOCAL_SCRATCH}/uv-cache ${LOCAL_SCRATCH}/uv-python"

SBATCH_ARGS=(
  --nodes=4
  --gres=gpu:4
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment=4
  --job-name="${ACCOUNT}-mxfp8-fused-refit-profile"
  --output="${RUN_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"Steady-state MXFP8 refit profile"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' \
  "${REPO}" "${LOCAL_HEAD}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
