#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:20:00

# This probe tests the exact launcher environment without permitting uv to
# rewrite the tracked lock. Callers own the cluster-specific GPU request.
set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the staged NeMo-RL nightly image.}"
: "${MOUNTS:?Set MOUNTS for the source filesystem.}"

srun \
  --ntasks=1 \
  --container-image="${CONTAINER}" \
  --container-mounts="${MOUNTS}" \
  --no-container-mount-home \
  --container-workdir="${SLURM_SUBMIT_DIR}" \
  bash -lc '
    set -euo pipefail
    cd "${SLURM_SUBMIT_DIR}"
    if uv lock --check; then
      echo "tracked_uv_lock_check=passed"
    else
      echo "tracked_uv_lock_check=failed"
    fi
    NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore --frozen python -c "import transformer_engine.pytorch; import megatron.core; import mamba_ssm; import nemo_rl; print(\"uv_overlay_imports=passed\")"
  '
