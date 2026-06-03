#!/bin/bash

set -euo pipefail

CONTAINER="/lustre/fsw/portfolios/llmservice/users/smohsenitahe/sqsh/super-omni-vllm20-super-vlm2-20260507-0905b74.sqsh"
SLURM_ACCOUNT="nemotron_omni_vision"
SLURM_PARTITION="interactive"
SLURM_TIME="4:00:00"
GPUS_PER_NODE="8"
JOB_NAME="matthieul:interactive"

OVERLAY_SOURCE="${OVERLAY_SOURCE:-/lustre/fsw/portfolios/llmservice/users/matthieul/repos_rl/nemo-rl-super-baseline}"
NRL_NEMO_RL_DIR="${NRL_NEMO_RL_DIR:-${OVERLAY_SOURCE}/nemo_rl}"
NRL_RESULTS_DIR="${NRL_RESULTS_DIR:-${OVERLAY_SOURCE}/results}"
NRL_SCRIPTS_DIR="${NRL_SCRIPTS_DIR:-${OVERLAY_SOURCE}/scripts}"
NRL_CONFIGS_DIR="${NRL_CONFIGS_DIR:-${OVERLAY_SOURCE}/examples/configs}"
NRL_NEMO_GYM_EXAMPLES_DIR="${NRL_NEMO_GYM_EXAMPLES_DIR:-${OVERLAY_SOURCE}/examples/nemo_gym}"
NRL_MEGATRON_LM_DIR="${NRL_MEGATRON_LM_DIR:-${OVERLAY_SOURCE}/3rdparty/Megatron-LM-workspace/Megatron-LM}"
NRL_MEGATRON_BRIDGE_DIR="${NRL_MEGATRON_BRIDGE_DIR:-${OVERLAY_SOURCE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}"
NRL_GYM_DIR="${NRL_GYM_DIR:-${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym}"
NRL_AUTOMODEL_DIR="${NRL_AUTOMODEL_DIR:-${OVERLAY_SOURCE}/3rdparty/Automodel-workspace/Automodel}"
NRL_VLLM_DIR="${NRL_VLLM_DIR:-${OVERLAY_SOURCE}/3rdparty/vllm}"

MOUNTS="/lustre:/lustre"

_maybe_mount() {
  local src="$1" dst="$2" label="$3"
  if [[ -z "${src}" ]]; then
    return
  fi
  if [[ -d "${src}" ]]; then
    MOUNTS="${MOUNTS},${src}:${dst}"
    echo "  Mount: ${label} -> ${dst}"
  else
    echo "  Skip:  ${label} (${src} not found on disk, using container built-in)"
  fi
}

echo ""
echo "Overlay source: ${OVERLAY_SOURCE}"
echo "Overlay mounts:"
_maybe_mount "${NRL_NEMO_RL_DIR}" "/opt/nemo-rl/nemo_rl" "nemo_rl"
_maybe_mount "${NRL_RESULTS_DIR}" "/opt/nemo-rl/results" "results"
_maybe_mount "${NRL_SCRIPTS_DIR}" "/opt/nemo-rl/scripts" "scripts"
_maybe_mount "${NRL_CONFIGS_DIR}" "/opt/nemo-rl/examples/configs" "configs"
_maybe_mount "${NRL_NEMO_GYM_EXAMPLES_DIR}" "/opt/nemo-rl/examples/nemo_gym" "nemo_gym_examples"
_maybe_mount "${NRL_MEGATRON_LM_DIR}" "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" "Megatron-LM"
_maybe_mount "${NRL_MEGATRON_BRIDGE_DIR}" "/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" "Megatron-Bridge"
_maybe_mount "${NRL_GYM_DIR}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym" "NeMo-Gym"
_maybe_mount "${NRL_AUTOMODEL_DIR}" "/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel" "Automodel"
_maybe_mount "${NRL_VLLM_DIR}" "/opt/nemo-rl/3rdparty/vllm" "vLLM"

echo ""
echo "Starting interactive session:"
echo "  container=${CONTAINER}"
echo "  account=${SLURM_ACCOUNT}"
echo "  partition=${SLURM_PARTITION}"
echo "  gpus_per_node=${GPUS_PER_NODE}"
echo "  mounts=${MOUNTS}"

srun -p "${SLURM_PARTITION}" \
     -A "${SLURM_ACCOUNT}" \
     -N 1 \
     --pty \
     --container-image "${CONTAINER}" \
     --no-container-mount-home \
     --container-mounts="${MOUNTS}" \
     --container-workdir=/opt/nemo-rl \
     --gpus-per-node="${GPUS_PER_NODE}" \
     --job-name "${JOB_NAME}" \
     -t "${SLURM_TIME}" \
     bash -l
