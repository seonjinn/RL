#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=coreai_dlalgo_nemorl.r08-logprob-unit
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/r0p8_hybridep_oom_20261014/slurm/%x-%j.out

set -euo pipefail

readonly REPO=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL-r0p8-hybridep-oom-20261014
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/yukih/enroot-images/gitlab-master.nvidia.com/yukih/docker-images/nemo-rl:r0.8.0-19666f86c.squashfs

test -r "$IMAGE"
test -f "$REPO/tests/unit/models/automodel/test_automodel_train.py"
git -C "$REPO" rev-parse HEAD

srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/lustre:/lustre,/home:/home \
  bash -lc "set -euo pipefail; cd '$REPO'; export PYTHONPATH='$REPO:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel'; /opt/nemo_rl_venv/bin/python -m pytest -q tests/unit/models/automodel/test_automodel_train.py::TestLossPostProcessor::test_local_policy_loss_does_not_save_full_fp32_logprobs"
