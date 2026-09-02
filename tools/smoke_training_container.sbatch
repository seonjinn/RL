#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_training_container
#SBATCH --output=smoke_training_container_%j.log

set -euo pipefail

export PATH="/cm/local/apps/slurm/25.11/bin:/cm/shared/apps/slurm/current/bin:${PATH}"

: "${CONTAINER:?Set CONTAINER to an immutable .sqsh path}"
: "${WORKTREE:?Set WORKTREE to the checked-out NeMo-RL worktree}"

for path in \
  "${WORKTREE}/nemo_rl" \
  "${WORKTREE}/tests" \
  "${WORKTREE}/examples/configs" \
  "${WORKTREE}/3rdparty/Automodel-workspace/Automodel" \
  "${WORKTREE}/3rdparty/Gym-workspace/Gym" \
  "${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"; do
  test -e "${path}"
done

mounts="/lustre:/lustre"
mounts+=",${WORKTREE}/nemo_rl:/opt/nemo-rl/nemo_rl"
mounts+=",${WORKTREE}/tests:/opt/nemo-rl/tests"
mounts+=",${WORKTREE}/examples/configs:/opt/nemo-rl/examples/configs"
mounts+=",${WORKTREE}/3rdparty/Automodel-workspace/Automodel:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel"
mounts+=",${WORKTREE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
mounts+=",${WORKTREE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"

srun \
  --ntasks=1 \
  --gres=gpu:4 \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${mounts}" \
  --container-workdir=/opt/nemo-rl \
  bash -lc '
    set -euo pipefail
    source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env
    export UV_CACHE_DIR="/raid/scratch/${USER}/nemo-rl/${SLURM_JOB_ID}/uv"
    uv run --no-sync python -c "import flashinfer, nemo, nemo.lens, nemo_rl, torch, transformer_engine.pytorch, vllm; from megatron import core; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; print({\"cuda\": True, \"gpus\": 4, \"flashinfer\": getattr(flashinfer, \"__version__\", \"unknown\"), \"nemo\": nemo.__file__, \"nemo_rl\": nemo_rl.__file__, \"vllm\": vllm.__version__})"
    uv run --no-sync python -m pytest -q \
      tests/unit/models/generation/test_vllm_fp8_quantization.py \
      tests/unit/models/generation/test_vllm_refit_loader.py \
      -k "multi_gpu_fp8_patches_before_model_load or expands_prequantized_grouped_experts_for_mxfp8 or rejects_unpaired_prequantized_grouped_mxfp8_entries or refit_loader_cache_records_replays_and_falls_back or refit_loader_cache_preserves_none_return_contract" \
      --vllm-only \
      --maxfail=1
  '
