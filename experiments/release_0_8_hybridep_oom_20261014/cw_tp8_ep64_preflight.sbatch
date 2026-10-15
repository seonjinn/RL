#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-tp8ep64-preflight
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly REPO=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL-r0p8-hybridep-oom-20261014
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs

test -r "$IMAGE"
test -f "$REPO/experiments/release_0_8_hybridep_oom_20261014/cw_tp8_ep64_preflight.py"

srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre,/home:/home \
  bash -lc "export PYTHONPATH=/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel; /opt/nemo_rl_venv/bin/python '$REPO/experiments/release_0_8_hybridep_oom_20261014/cw_tp8_ep64_preflight.py'"
