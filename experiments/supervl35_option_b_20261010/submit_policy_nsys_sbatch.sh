#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NRL_NSYS_WORKER_PATTERNS=megatron_policy_worker
export NRL_NSYS_PROFILE_STEP_RANGE=${NRL_NSYS_PROFILE_STEP_RANGE:-3:4}
export NRL_NSYS_EXTRA_OPTIONS='{"cpuctxsw":"none"}'
export MM_TRAINER_WANDB_PROJECT=${MM_TRAINER_WANDB_PROJECT:-nemo-rl-supervl35-mxfp8-nsys}
export NRL_EXPERIMENT_SOURCE_COMMIT=${NRL_EXPERIMENT_SOURCE_COMMIT:-15045cb9425da21622220c0fa3046169a6be017c}
unset SLURM_AFTEROK_JOB_ID

exec /bin/bash "$script_dir/submit_arm_sbatch.sh" mxfp8-fp8params-on
