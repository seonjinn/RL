#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=nemotron_sw_post-dflash.loss_ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/logs/loss_ab-%j.out

# DFlash drafter loss A/B: kl_div (speculators default) vs nla (acceptance-aligned).
# Target: Qwen3-30B-A3B-Thinking-2507, 10K UltraChat, online mode (no hidden-state storage).
# GB200 node: GPUs 0-1 serve target (TP2, hidden-state extraction), GPUs 2-3 train.
set -uo pipefail
B=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training
export HF_HOME=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/hf_home HF_HUB_OFFLINE=0 TMPDIR=/tmp
export CUDA_HOME=/cm/shared/apps/cuda13.0/toolkit/13.0.2
export PATH=/lustre/fsw/portfolios/nemotron/users/sna/dflash_training/vllm_venv/bin:$CUDA_HOME/bin:$PATH
MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
LAYER_IDS="1 12 23 34 45"
mkdir -p $B/logs $B/output

cd $B/speculators

# 1. Data prep (once)
if [ ! -d $B/output/data_ultrachat_10k ]; then
  $B/speculators_venv/bin/python scripts/prepare_data.py \
    --model $MODEL --data ultrachat \
    --output $B/output/data_ultrachat_10k \
    --max-samples 10000 --seq-length 8192
fi

# 2. Launch vLLM target server (GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 $B/vllm_venv/bin/python scripts/launch_vllm.py \
  $MODEL --target-layer-ids $LAYER_IDS \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --tensor-parallel-size 2 --no-enable-chunked-prefill \
  > $B/logs/vllm_server.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 240); do
  sleep 10
  grep -q "Uvicorn running\|Application startup complete" $B/logs/vllm_server.log && break
done
if ! grep -q "Uvicorn running\|Application startup complete" $B/logs/vllm_server.log; then
  echo VLLM_SERVER_FAILED; tail -5 $B/logs/vllm_server.log; exit 1
fi

run_train() {
  loss=$1
  echo "===== TRAIN loss=$loss start $(date +%H:%M:%S) ====="
  CUDA_VISIBLE_DEVICES=2,3 $B/speculators_venv/bin/torchrun --standalone --nproc_per_node 2 \
    scripts/train.py \
    --verifier-name-or-path $MODEL \
    --data-path $B/output/data_ultrachat_10k \
    --on-missing generate --on-generate delete \
    --scheduler-type cosine \
    --draft-vocab-size 32000 \
    --max-anchors 1024 \
    --target-layer-ids $LAYER_IDS \
    --speculator-type dflash \
    --num-layers 5 \
    --loss-fn "$loss" \
    --lr 0.0006 --epochs 5 --sliding-window 2048 \
    --draft-hidden-act silu \
    --save-path $B/output/dflash_30bthink_${loss//[^a-z_]/}_10k --run-name dflash30b_${loss//[^a-z_]/}_10k \
    2>&1 | tail -120
  echo "===== TRAIN loss=$loss end rc=$? $(date +%H:%M:%S) ====="
}

run_train ce
run_train tv

kill $VLLM_PID 2>/dev/null
echo ALL_DONE
