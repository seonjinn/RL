#!/bin/bash
# Submit from the repository root, with 32 GB200 nodes and segment size 8.
# Required exports: CONTAINER, HF_HOME, BASE_LOG_DIR, AUDIT_ARM.
# Diagnostic run: original recipe, no memory/batch/TP overrides.
set -euo pipefail
: "${CONTAINER:?immutable smoke-tested image required}"
: "${HF_HOME:?existing model cache required}"
: "${BASE_LOG_DIR:?durable experiment directory required}"
: "${SLURM_JOB_ID:?submit through sbatch}"
case "${AUDIT_ARM:?alltoall or hybridep required}" in
  alltoall) DISPATCH_ARGS='policy.megatron_cfg.moe_token_dispatcher_type=alltoall policy.megatron_cfg.moe_flex_dispatcher_backend=deepep' ;;
  hybridep) DISPATCH_ARGS='' ;;
  *) exit 2 ;;
esac
export GPUS_PER_NODE=4 DEDICATED_RAY_HEAD=0
export PYTHONPATH=/opt/nemo-rl
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_HOST_MEMORY_AUDIT=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/raid/scratch/${USER}/oom-audit-${SLURM_JOB_ID}/tmp"
export HF_MODULES_CACHE="/raid/scratch/${USER}/oom-audit-${SLURM_JOB_ID}/hf-modules"
export TRITON_CACHE_DIR="/raid/scratch/${USER}/oom-audit-${SLURM_JOB_ID}/triton"
export TORCH_EXTENSIONS_DIR="/raid/scratch/${USER}/oom-audit-${SLURM_JOB_ID}/torch-extensions"
export XDG_CACHE_HOME="/raid/scratch/${USER}/oom-audit-${SLURM_JOB_ID}/cache"
export MOUNTS="/home:/home,/lustre:/lustre,/raid:/raid,${SLURM_SUBMIT_DIR}:/opt/nemo-rl"
export SETUP_COMMAND='mkdir -p "$TMPDIR" "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"'
export COMMAND="cd /opt/nemo-rl
/opt/nemo_rl_venv/bin/python -c 'import nemo_rl; print(\"EXECUTED_NEMO_RL\", nemo_rl.__file__)'
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \\
 --config examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml \\
 grpo.max_num_steps=20 checkpointing.enabled=false \\
 logger.log_dir=${BASE_LOG_DIR}/metrics-${SLURM_JOB_ID} \\
 logger.wandb.project=sna-hybridep-memory-audit \\
 logger.wandb.name=super-sync-${AUDIT_ARM}-${SLURM_JOB_ID} \\
 ${DISPATCH_ARGS}"
exec bash "${SLURM_SUBMIT_DIR}/ray.sub"
