#!/usr/bin/env bash
#SBATCH --job-name=sna-q35-posttrained-target-stage
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q35-math-specdec-20260916/stage-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q35-math-specdec-20260916/stage-%j.err
set -euo pipefail
readonly SOURCE=/home/sna/nemorl-q35-math-specdec-20260916
readonly BASE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
test -z "$(git -C "${SOURCE}" status --porcelain=v1 --untracked-files=all)"
git -C "${SOURCE}" rev-parse HEAD >"${BASE}/experiments/q35-math-specdec-20260916/stage-${SLURM_JOB_ID}-source.txt"
srun --container-image="${BASE}/containers/nemo_rl_nightly_20260909_7023221.sqsh" \
  --container-mounts=/home:/home,/lustre:/lustre,/raid:/raid \
  --container-workdir="${SOURCE}" \
  /opt/nemo_rl_venv/bin/python "${SOURCE}/experiments/q35_math_specdec_20260916/stage_target.py"
