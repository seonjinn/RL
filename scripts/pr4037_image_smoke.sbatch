#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_dlalgo_nemorl-pr4037-image-smoke

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to an immutable squashfs image}"
[[ -f "$CONTAINER" && ! -L "$CONTAINER" ]]

srun --nodes=1 --ntasks=1 \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python \
  -c 'import deep_ep, deep_ep_cpp, hybrid_ep_cpp, torch; print(f"gpu={torch.cuda.get_device_name(0)} deep_ep={deep_ep.__file__} hybrid_ep_cpp={hybrid_ep_cpp.__file__}")'
