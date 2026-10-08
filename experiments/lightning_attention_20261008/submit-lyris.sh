#!/usr/bin/env bash
set -euo pipefail

arm=${1:?Usage: submit-lyris.sh bf16-bf16|bf16-mxfp8|mxfp8-default|mxfp8-option-b flashinfer|triton [test-only]}
backend=${2:?Usage: submit-lyris.sh bf16-bf16|bf16-mxfp8|mxfp8-default|mxfp8-option-b flashinfer|triton [test-only]}
action=${3:-submit}
[[ "$action" == submit || "$action" == test-only ]]

case "$arm" in
  bf16-bf16) config=experiments/lightning_main_20261006/async-bf16.yaml ;;
  bf16-mxfp8) config=experiments/lightning_main_20261006/async-mxfp8.yaml ;;
  mxfp8-default) config=experiments/lightning_pr4353_20261007/async-mxfp8-train.yaml ;;
  mxfp8-option-b) config=experiments/lightning_pr4353_20261007/async-mxfp8-train-option-b.yaml ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
case "$backend" in
  flashinfer) attention_override='' ;;
  triton) attention_override='+policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN' ;;
  *) echo "Unknown attention backend: $backend" >&2; exit 2 ;;
esac

: "${CONTAINER:?Set the immutable vLLM 0.29 nightly image}"
: "${SOURCE_ARCHIVE:?Set the immutable source archive}"
: "${SOURCE_COMMIT:?Set the expected source commit}"
: "${RESULT_ROOT:?Set the shared result directory}"
: "${WANDB_API_KEY:?W&B cloud logging must be enabled}"

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if [[ "$action" == submit ]]; then
  git -C "$repo" -c fetch.recurseSubmodules=false pull --ff-only
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no --ignore-submodules=none)"
test -f "$SOURCE_ARCHIVE"
test -f "$CONTAINER"

account=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
max_steps=${MAX_STEPS:-20}
name="lightning-attn-${arm}-${backend}-${max_steps}step${RUN_SUFFIX:+-${RUN_SUFFIX}}"
run_root="${RESULT_ROOT}/${name}"
local_root="/raid/scratch/${USER}/nr-lightning-attn-${SOURCE_COMMIT:0:10}-${arm}-${backend}"
source_root="${local_root}/source"
te_override=""
if [[ "$arm" == mxfp8-* ]]; then
  te_override="policy.megatron_cfg.te_precision_config_file=${source_root}/experiments/lightning_pr4353_20261007/te-routed-mxfp8.yaml"
fi
model_root="/raid/scratch/${USER}/nr-lightning-model"
hf_source="/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home"
model_cache="models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

if [[ "$action" == submit ]]; then
  mkdir -p "$run_root"
fi
export GPUS_PER_NODE=4 CPUS_PER_WORKER=144 DEDICATED_RAY_HEAD=0
export CONTAINER_REMAP_ROOT=1 BASE_LOG_DIR="$run_root" RAY_TMPDIR=/tmp
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch"
export SETUP_COMMAND="set -euo pipefail
mkdir -p ${model_root}/hf/hub/${model_cache} ${local_root}/uv ${local_root}/vllm ${local_root}/inductor ${local_root}/triton
if [[ ! -f ${source_root}/.source-ready ]]; then
  rm -rf ${source_root}
  mkdir -p ${source_root}
  tar -xf ${SOURCE_ARCHIVE} -C ${source_root}
  for module in Automodel-workspace/Automodel Gym-workspace/Gym Megatron-Bridge-workspace/Megatron-Bridge; do
    test -d /opt/nemo-rl/3rdparty/\${module}
    rmdir ${source_root}/3rdparty/\${module}
    ln -s /opt/nemo-rl/3rdparty/\${module} ${source_root}/3rdparty/\${module}
  done
  touch ${source_root}/.source-ready
fi
(
  flock -x 9
  if [[ ! -f ${model_root}/hf/.lightning-cache-ready ]]; then
    rsync -a --ignore-existing ${hf_source}/hub/${model_cache}/ ${model_root}/hf/hub/${model_cache}/
    touch ${model_root}/hf/.lightning-cache-ready
  fi
) 9>${model_root}/hf/.lightning-cache.lock"
export COMMAND="set -euo pipefail
ulimit -c 0
cd ${source_root}
export PYTHONPATH=${source_root}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=${model_root}/hf HF_HUB_CACHE=${model_root}/hf/hub HUGGINGFACE_HUB_CACHE=${model_root}/hf/hub
export HF_DATASETS_CACHE=${hf_source}/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NRL_MEGATRON_CHECKPOINT_DIR=${RESULT_ROOT}/checkpoints-${SOURCE_COMMIT}
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_FORCE_REBUILD_VENVS=false FLA_TILELANG=0
export UV_CACHE_DIR=${local_root}/uv VLLM_CACHE_ROOT=${local_root}/vllm TORCHINDUCTOR_CACHE_DIR=${local_root}/inductor TRITON_CACHE_DIR=${local_root}/triton
export PYTHONPYCACHEPREFIX=${local_root}/pycache RAY_TMPDIR=/tmp
unset NRL_IGNORE_VERSION_MISMATCH PYTHONOPTIMIZE
/opt/nemo_rl_venv/bin/python tools/config_cli.py expand ${config} >/dev/null
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${config} ${te_override} ${attention_override} grpo.max_num_steps=${max_steps} logger.log_dir=${run_root}/metrics logger.wandb.name=${name}"

args=(--nodes=8 --exclusive --mem=0 --account="$account" --partition=gb200
  --qos=user-restrictions --time=04:00:00 --segment=4
  --job-name="${account}-mxfp8.${name}" --output="${run_root}/slurm-%j.out")
if [[ "$action" == test-only ]]; then
  args+=(--test-only)
fi
printf 'source=%s\ncontainer=%s\nconfig=%s\narm=%s\nbackend=%s\n' \
  "$SOURCE_COMMIT" "$CONTAINER" "$config" "$arm" "$backend"
exec sbatch "${args[@]}" "$repo/ray.sub"
