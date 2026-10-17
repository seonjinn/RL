#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-pr4759-preflight
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly REPO=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL-r0p8-hybridep-oom-20261014
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs
readonly PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python

test -r "$IMAGE"
test -f "$REPO/examples/configs/recipes/llm/grpo-minimax-m27-dapo-8n8g-automodel-ep64.yaml"
git -C "$REPO" rev-parse HEAD

srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre,/home:/home \
  bash -lc "set -euo pipefail; cd /opt/nemo-rl; export PYTHONPATH=/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel; '$PYTHON' '$REPO/experiments/release_0_8_hybridep_oom_20261014/cw_pr4759_preflight.py'"
