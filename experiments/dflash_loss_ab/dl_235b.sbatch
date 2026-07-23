#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=nemotron_sw_post-dflash.dl235b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/logs/dl235b-%j.out
#SBATCH --mem=64G
set -uo pipefail
B=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training
export HF_HOME=$B/hf_home HF_HUB_DISABLE_XET=1
for i in 1 2 3; do
  $B/speculators_venv/bin/python - <<'PY' && break
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-235B-A22B-Thinking-2507", max_workers=4)
print("DL_OK")
PY
  echo "retry $i"; sleep 30
done
IDX=$(find /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/hf_home/hub -path "*Qwen3-235B-A22B-Thinking-2507*/model.safetensors.index.json" | head -1)
if [ -n "$IDX" ]; then
  echo download_complete
else
  echo "DOWNLOAD FAILED after 3 attempts"; exit 1
fi
