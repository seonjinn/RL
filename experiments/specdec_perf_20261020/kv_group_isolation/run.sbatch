#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --job-name=coreai_dlalgo_nemorl.q35-kvgroup-iso
#SBATCH --output=logs/%j.out

# Submit from this directory: sbatch run.sbatch [lane ...]
# Each lane is a space-separated list of arms that run in order on one GPU (at most 4 lanes).
# BENCH_ARGS passes extra bench.py flags to every arm, for example BENCH_ARGS=--greedy-only.
set -euo pipefail

if (($#)); then
  LANES=("$@")
else
  LANES=("dflash_k3" "dflash_k3_g5" "dspark_k4" "baseline baseline_triton")
fi

EXP=$SLURM_SUBMIT_DIR
REPO=$(git -C "$EXP" rev-parse --show-toplevel)
U=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna
CONTAINER=${CONTAINER:-$U/containers/nemo_rl_nightly.sqsh}
OUT=$EXP/results/run_$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "commit=$(git -C "$REPO" rev-parse HEAD) container=$(readlink -f "$CONTAINER") bench_args=${BENCH_ARGS:-}" | tee "$OUT/metadata.txt"

export HF_HOME=$U/hf_home HF_DATASETS_CACHE=$U/hf_home/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset HF_TOKEN UV_CACHE_DIR

# Same vLLM venv the GRPO generation workers use; compile/JIT caches stay container-local
PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
srun --container-image="$CONTAINER" --container-mounts=/lustre:/lustre --no-container-mount-home \
  bash -c "
set -uo pipefail
C=/tmp/kvgi
export XDG_CACHE_HOME=\$C/cache VLLM_CACHE_ROOT=\$C/cache/vllm TRITON_CACHE_DIR=\$C/cache/triton
export TORCHINDUCTOR_CACHE_DIR=\$C/cache/inductor FLASHINFER_WORKSPACE_BASE=\$C/cache
$PY -c 'import vllm; print(\"vllm\", vllm.__version__)'
run() { CUDA_VISIBLE_DEVICES=\$1 $PY $EXP/bench.py --arm \$2 --out $OUT/\$2.json ${BENCH_ARGS:-} > $OUT/\$2.log 2>&1 || echo \"arm \$2 failed\"; }
gpu=0
for lane in \"\${@}\"; do
  (for arm in \$lane; do run \$gpu \$arm; done) &
  gpu=\$((gpu + 1))
done
wait
" _ "${LANES[@]}"
grep -h '^\[' "$OUT"/*.log