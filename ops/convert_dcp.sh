#!/bin/bash
# Distributed Super checkpoint conversion.
#
# Usage:
#   ops/convert_dcp.sh <nemo-rl-step-dir> <hf-output-dir>
#
# This launcher uses the repo-local Super converter rather than the generic
# DCP converter, so the Super-specific provider setup remains in repo code.
#SBATCH --account=llmservice_fm_vision
#SBATCH --job-name=nemo-rl-convert-dcp
#SBATCH --partition=batch_long
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00

set -euo pipefail

CKPT_DIR="${1?Usage: ops/convert_dcp.sh <ckpt-dir> <hf-ckpt-dir>}"
HF_CKPT_DIR="${2?Usage: ops/convert_dcp.sh <ckpt-dir> <hf-ckpt-dir>}"

if [[ ! -d $CKPT_DIR ]]; then
    echo "Checkpoint directory $CKPT_DIR does not exist"
    exit 1
fi

WEIGHTS_DIR="$CKPT_DIR/policy/weights"
MCORE_ITER=$(cat "$WEIGHTS_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "0")
WEIGHTS_PATH=$(printf "$WEIGHTS_DIR/iter_%07d" "$MCORE_ITER")

echo "Converting Megatron checkpoint from $WEIGHTS_PATH..."
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/converters" ]]; then
    NEMORL=$(realpath "$SLURM_SUBMIT_DIR")
else
    NEMORL=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
fi
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh}
CONVERTER=${CONVERTER:-$NEMORL/examples/converters/convert_megatron_to_hf_super.py}
DEPS_DIR=${DEPS_DIR:-$NEMORL/.convert_runtime_deps_py312_gpu}
HF_HOME=${HF_HOME:-/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/smohsenitahe/cache/huggingface}
COMMON_MOUNTS=${COMMON_MOUNTS:-/scratch:/scratch,/lustre:/lustre,/home:/home}

PREP_COMMAND="cd $NEMORL && \
mkdir -p $DEPS_DIR && \
export PYTHONPATH=$DEPS_DIR:$NEMORL:$NEMORL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$NEMORL/3rdparty/Megatron-LM-workspace/Megatron-LM:\${PYTHONPATH:-} && \
uv run --extra mcore --no-sync python -m pip install --target $DEPS_DIR --upgrade --no-deps \
transformer-engine==2.14.1 \
transformer-engine-cu12==2.14.1 \
transformer-engine-torch==2.14.1 \
onnxscript onnx_ir onnx \
nvidia-modelopt nvdlfw-inspect pulp && \
uv run --extra mcore --no-sync python -c 'import numpy; import google.protobuf; import transformer_engine.pytorch; import onnxscript; import modelopt; print(\"runtime deps ready; numpy=\" + numpy.__version__)'"

srun -N1 -n1 \
    --container-image=$CONTAINER \
    --container-mounts="$COMMON_MOUNTS" \
    --container-workdir="$NEMORL" \
    --no-container-mount-home \
    bash -lc "$PREP_COMMAND"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT

DIST_COMMAND="cd $NEMORL && \
export MASTER_ADDR=$MASTER_ADDR && \
export MASTER_PORT=$MASTER_PORT && \
export HF_HOME=$HF_HOME && \
export HF_HUB_OFFLINE=0 && \
export TRANSFORMERS_OFFLINE=0 && \
export NRL_IGNORE_VERSION_MISMATCH=true && \
export PYTHONUNBUFFERED=1 && \
export PYTHONPATH=$DEPS_DIR:$NEMORL:$NEMORL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$NEMORL/3rdparty/Megatron-LM-workspace/Megatron-LM:\${PYTHONPATH:-} && \
python -c 'import os, torch; print(\"torchrun launcher visible GPUs=\" + str(torch.cuda.device_count()) + \" CUDA_VISIBLE_DEVICES=\" + str(os.environ.get(\"CUDA_VISIBLE_DEVICES\")), flush=True)' && \
uv run --extra mcore --no-sync torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\${SLURM_PROCID} --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $CONVERTER \
--config=$CKPT_DIR/config.yaml \
--megatron-ckpt-path=$WEIGHTS_PATH \
--hf-ckpt-path=$HF_CKPT_DIR \
--tp=8 --pp=1 --ep=4 --etp=1"

set +e
srun \
    -N "$SLURM_NNODES" \
    -n "$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --cpu-bind=cores \
    --container-image=$CONTAINER \
    --container-mounts="$COMMON_MOUNTS" \
    --container-workdir="$NEMORL" \
    --no-container-mount-home \
    bash -lc "$DIST_COMMAND"
SRUN_EXIT=$?
set -e

if [[ "${SRUN_EXIT}" -eq 0 ]] && \
   [[ -f "${HF_CKPT_DIR}/config.json" ]] && \
   [[ -f "${HF_CKPT_DIR}/model.safetensors.index.json" ]] && \
   ls "${HF_CKPT_DIR}"/*.safetensors >/dev/null 2>&1; then
    N_SAFETENSORS=$(ls "${HF_CKPT_DIR}"/*.safetensors 2>/dev/null | wc -l)
    echo "OK Conversion verified at $HF_CKPT_DIR (${N_SAFETENSORS} safetensors files; srun exit was ${SRUN_EXIT})"
    exit 0
else
    echo "FAIL HF output incomplete at $HF_CKPT_DIR (config.json, model.safetensors.index.json, or safetensors missing; srun exit was ${SRUN_EXIT})"
    exit 1
fi
