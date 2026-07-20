#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=nemotron_sw_post-dflash.235b_screen
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/logs/235b_screen-%j.out

# 235B DFlash drafter pipeline screening: kl_div (4-way winner), UltraChat 10K.
# Node 0: vLLM target server TP4 (full node). Node 1: torchrun train x4 GPUs.
# Hidden states via FileBackend on lustre (cross-node requirement).
set -uo pipefail
B=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training
export HF_HOME=$B/hf_home HF_HUB_OFFLINE=0 TMPDIR=/tmp
export CUDA_HOME=/cm/shared/apps/cuda13.0/toolkit/13.0.2
export PATH=$B/vllm_venv/bin:$CUDA_HOME/bin:$PATH
MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507
LAYER_IDS="1 23 46 68 91"
HS_CACHE=$B/hs_cache_235b
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER_NODE=${NODES[0]}; TRAIN_NODE=${NODES[1]}
SLOG=$B/logs/vllm_server_235b.log
mkdir -p $B/logs $B/output $HS_CACHE
cd $B/speculators

srun --overlap -N1 -n1 -w $SERVER_NODE $B/vllm_venv/bin/python scripts/launch_vllm.py \
  $MODEL --target-layer-ids $LAYER_IDS \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --tensor-parallel-size 4 --no-enable-chunked-prefill \
  --hidden-states-path $HS_CACHE > $SLOG 2>&1 &

for i in $(seq 1 360); do
  sleep 10
  grep -q "Uvicorn running\|Application startup complete" $SLOG && break
  grep -q "Engine core initialization failed" $SLOG && { echo VLLM_SERVER_FAILED; tail -5 $SLOG; exit 1; }
done
if ! grep -q "Uvicorn running\|Application startup complete" $SLOG; then
  echo VLLM_SERVER_TIMEOUT; tail -5 $SLOG; exit 1
fi
echo "server up on $SERVER_NODE $(date +%H:%M:%S)"

echo "===== TRAIN 235B loss=kl_div start $(date +%H:%M:%S) ====="
srun --overlap -N1 -n1 -w $TRAIN_NODE $B/speculators_venv/bin/torchrun --standalone --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path $MODEL \
  --data-path $B/output/data_ultrachat_10k \
  --on-missing generate --on-generate delete \
  --vllm-endpoint http://$SERVER_NODE:8000/v1 \
  --hidden-states-path $HS_CACHE \
  --scheduler-type cosine --draft-vocab-size 32000 --max-anchors 1024 \
  --target-layer-ids $LAYER_IDS --speculator-type dflash --num-layers 5 \
  --loss-fn kl_div --lr 0.0006 --epochs 5 --sliding-window 2048 \
  --draft-hidden-act silu \
  --save-path $B/output/dflash_235bthink_kl_div_10k --run-name dflash235b_kl_div_10k \
  2>&1 | tail -120
echo "===== TRAIN 235B end rc=$? $(date +%H:%M:%S) ====="
echo ALL_DONE
