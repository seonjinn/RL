#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 main|pr [--test-only]}"
mode="${2:-}"

case "${variant}" in
  main)
    branch="codex/pr3294-no-pr-vllm029-20260929"
    source_label="main-no-pr3294"
    ;;
  pr)
    branch="sna/pr-mxfp8-refit-optimization"
    source_label="pr3294"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

case "${mode}" in
  "") sbatch_mode=() ;;
  --test-only) sbatch_mode=(--test-only) ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

readonly repo="${REPO:-/home/${USER}/nemo-rl-pr3294-matched-${variant}-20260929}"
readonly remote="${REMOTE:-origin}"
readonly account="${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}"
readonly container="${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/nemo-rl/images/vllm029-20260929/nemo_rl_nightly_vllm029_20260929.sqsh}"
readonly shared_root="${SHARED_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}}"
readonly result_root="${RESULT_ROOT:-${shared_root}/results/pr3294-vllm029-qwen235-matched-pr-ab}"
readonly config="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g-mxfp8-rollout.yaml"
readonly slurm_bin="/cm/local/apps/slurm/25.11/bin"
readonly run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly model_cache="${shared_root}/hf_home/hub/models--Qwen--Qwen3-235B-A22B"

git -C "${repo}" pull --ff-only "${remote}" "${branch}"
readonly code_sha="$(git -C "${repo}" rev-parse HEAD)"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=no --ignore-submodules=none)"
test -f "${repo}/${config}"
test -f "${container}"
test -f "/home/${USER}/.netrc"
test -f "${model_cache}/refs/main"
readonly model_revision="$(<"${model_cache}/refs/main")"
readonly model_path="${model_cache}/snapshots/${model_revision}"
test -d "${model_path}"

readonly run_name="pr3294-vllm029-qwen235-${source_label}-20s-${run_tag}-${code_sha:0:9}"
readonly run_root="${result_root}/${run_name}"
readonly local_root="/raid/scratch/${USER}/${run_name}"

mkdir -p "${run_root}"
printf 'source_sha=%s\ncontainer=%s\nconfig=%s\nvariant=%s\nmodel_revision=%s\n' \
  "${code_sha}" "${container}" "${config}" "${variant}" "${model_revision}" \
  >"${run_root}/provenance.txt"

export PATH="${slurm_bin}:/usr/local/bin:/usr/bin:/bin"
export CONTAINER="${container}"
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=144
export BASE_LOG_DIR="${run_root}"
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,/home/${USER}/.netrc:/root/.netrc,${repo}/nemo_rl:/opt/nemo-rl/nemo_rl,${repo}/examples:/opt/nemo-rl/examples,${repo}/tests:/opt/nemo-rl/tests"
export SETUP_COMMAND="set -euo pipefail; rm -rf ${local_root}; mkdir -p ${local_root}/{tmp,venvs,vllm,triton,inductor,ray}"
export COMMAND="set -euo pipefail; \
cd /opt/nemo-rl; \
export HOME=/root; \
export HF_HOME=${shared_root}/hf_home; \
export HF_DATASETS_CACHE=\${HF_HOME}/cache; \
export TMPDIR=${local_root}/tmp; \
export NEMO_RL_VENV_DIR=${local_root}/venvs; \
export VLLM_CACHE_ROOT=${local_root}/vllm; \
export TRITON_CACHE_DIR=${local_root}/triton; \
export TORCHINDUCTOR_CACHE_DIR=${local_root}/inductor; \
export RAY_TMPDIR=${local_root}/ray; \
export NRL_FORCE_REBUILD_VENVS=true; \
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0; \
export NCCL_NVLS_ENABLE=0; \
export RAY_CGRAPH_get_timeout=2400; \
export NVTE_CUDA_ARCHS=100; \
export TORCH_CUDA_ARCH_LIST=10.0; \
export UV_PYTHON=/opt/nemo_rl_venv/bin/python; \
export UV_NO_MANAGED_PYTHON=1; \
export UV_LOCK_TIMEOUT=7200; \
unset UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR WANDB_API_KEY; \
printf 'NEMO_RL_SOURCE_COMMIT=%s\\n' \"${code_sha}\"; \
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config ${config} \
  policy.model_name=${model_path} \
  policy.tokenizer.name=${model_path} \
  grpo.max_num_steps=20 \
  grpo.seed=42 \
  grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  checkpointing.enabled=false \
  policy.generation.vllm_cfg.async_engine=false \
  policy.generation.vllm_cfg.enforce_eager=false \
  policy.generation.vllm_cfg.use_tqdm=false \
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \
  logger.log_dir=${run_root}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=sna-pr3294-vllm029-matched-pr-ab \
  logger.wandb.name=${run_name}"

exec "${slurm_bin}/sbatch" \
  "${sbatch_mode[@]}" \
  --export="ALL,PATH=${PATH}" \
  --nodes=16 \
  --gpus-per-node=4 \
  --exclusive \
  --account="${account}" \
  --partition=batch \
  --time=04:00:00 \
  --segment=16 \
  --dependency= \
  --job-name="${run_name}" \
  --output="${run_root}/slurm-%j.out" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"Matched Qwen3-235B latest-main versus PR 3294 A/B"}}' \
  "${repo}/ray.sub"
