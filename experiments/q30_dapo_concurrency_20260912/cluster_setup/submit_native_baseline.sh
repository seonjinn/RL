#!/usr/bin/env bash
set -euo pipefail
site=${1:?lyris or ptyche}
mode=${2:---plan}
case "$site" in
  lyris) partition=gb200; image_job=3078480 ;;
  ptyche) partition=batch; image_job=2837270 ;;
  *) echo "Unsupported site: $site" >&2; exit 2 ;;
esac
case "$mode" in --plan|--submit) ;; *) exit 2 ;; esac
repo=/home/sna/nemorl-q30-dapo-${site}-20260914
image=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo-rl-20260916/nemo_rl_nightly_20260916_${image_job}.sqsh
target=/lustre/fsw/coreai_dlalgo_llm/users/sna/q30-dapo-parity-20260914/target
run_name=Qwen3-30BA3B-native0916-${site}-Baseline-1step-$(date -u +%Y%m%dT%H%M%SZ)
results=/lustre/fsw/coreai_dlalgo_llm/users/sna/experiments/q30-dapo-${site}-20260914/native-baseline/${run_name}
export CONTAINER="$image" GPUS_PER_NODE=4
export DEDICATED_RAY_HEAD=0
export MOUNTS=/home:/home,/lustre:/lustre,/raid:/raid
export BASE_LOG_DIR="$results" NETRC=/home/sna/.netrc
export HF_HOME=/raid/scratch/sna/nrl-native-20260916/hf
export NRL_MEGATRON_CHECKPOINT_DIR=${results}/megatron-initial-checkpoint
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export XDG_CACHE_HOME=/raid/scratch/sna/nrl-native-20260916/cache
export TRITON_CACHE_DIR=${XDG_CACHE_HOME}/triton
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions
export WANDB_DIR="$results" WANDB_CACHE_DIR=${XDG_CACHE_HOME}/wandb
export WANDB_CONFIG_DIR=${XDG_CACHE_HOME}/wandb-config
export RAY_TMPDIR=/raid/scratch/sna/nrl-native-20260916/${run_name}/ray
export NRL_NATIVE_TMP=/raid/scratch/sna/nrl-native-20260916/${run_name}/tmp
unset PYTHONPATH PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS TMPDIR
export SETUP_COMMAND='set -euo pipefail
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$RAY_TMPDIR" "$NRL_NATIVE_TMP"
test -d "$NRL_MEGATRON_CHECKPOINT_DIR"
test -w "$NRL_MEGATRON_CHECKPOINT_DIR"
touch "$NRL_MEGATRON_CHECKPOINT_DIR/.mount-check-$(hostname)"
test -x /opt/nemo_rl_venv/bin/python
test -x /usr/local/bin/python-MegatronPolicyWorker
test -x /usr/local/bin/python-VllmGenerationWorker'
export COMMAND="set -euo pipefail
cd /opt/nemo-rl
unset PYTHONPATH PYTHONOPTIMIZE NRL_FORCE_REBUILD_VENVS
export TMPDIR=${NRL_NATIVE_TMP}
echo SOURCE_MODE=container-native
git rev-parse HEAD || true
/opt/nemo_rl_venv/bin/python -c 'import sys,nemo_rl; print(sys.executable, nemo_rl.__file__)'
/usr/local/bin/python-MegatronPolicyWorker -c 'import os; from pathlib import Path; from nemo_rl.models.policy.utils import get_megatron_checkpoint_dir; p = Path(get_megatron_checkpoint_dir()); assert str(p) == os.environ[\"NRL_MEGATRON_CHECKPOINT_DIR\"]; markers = list(p.glob(\".mount-check-*\")); assert len(markers) == 4, markers; print(\"SHARED_CHECKPOINT_PREFLIGHT_OK\", p, len(markers))'
exec /opt/nemo_rl_venv/bin/python examples/run_grpo.py \\
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml \\
  grpo.max_num_steps=1 checkpointing.enabled=false \\
  policy.model_name=${target} policy.tokenizer.name=${target} \\
  policy.precision=bfloat16 \\
  policy.generation.vllm_cfg.enforce_eager=false \\
  policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \\
  logger.wandb_enabled=true logger.wandb.project=sna-specdec \\
  logger.wandb.name=${run_name} logger.log_dir=${results}/metrics"
args=(--nodes=4 --ntasks-per-node=1 --exclusive --segment=4
  --account=coreai_dlalgo_llm "--partition=$partition" --time=02:00:00
  "--job-name=coreai_dlalgo_llm-sna.q30-native-baseline"
  "--output=$results/slurm-%j.log")
if [[ "$mode" == --plan ]]; then
  printf 'GPUS_PER_NODE=%s\nCONTAINER=%s\nMOUNTS=%s\n' "$GPUS_PER_NODE" "$CONTAINER" "$MOUNTS"
  printf 'DEDICATED_RAY_HEAD=%s\n' "$DEDICATED_RAY_HEAD"
  printf 'HF_HOME=%s\nNRL_MEGATRON_CHECKPOINT_DIR=%s\n' "$HF_HOME" "$NRL_MEGATRON_CHECKPOINT_DIR"
  printf '%s\n' "$COMMAND"
  printf 'sbatch %s ray.sub\n' "${args[*]}"
  exit 0
fi
cd "$repo"
test -f "${image}.smoke-passed"
test -s "$target/config.json"
test -s "$NETRC"
mkdir -p "$results" "$NRL_MEGATRON_CHECKPOINT_DIR"
git rev-parse HEAD
sbatch --test-only "${args[@]}" ray.sub
sbatch --parsable "${args[@]}" ray.sub
