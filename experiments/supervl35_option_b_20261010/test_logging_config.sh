#!/bin/bash
set -euo pipefail

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for config in "$launcher_dir"/configs/*.yaml; do
    [[ ${config##*/} == te_precision.yaml ]] && continue
    grep -Fxq '  tensorboard_enabled: true' "$config"
    grep -Fxq '  wandb_enabled: true' "$config"
    # shellcheck disable=SC2016
    grep -Fxq '    project: ${oc.env:MM_TRAINER_WANDB_PROJECT,nemo-rl-supervl35-mxfp8}' "$config"
done

grep -Fq '"logger": {"wandb_enabled": True, "tensorboard_enabled": True}' "$launcher_dir/compose_configs.py"
grep -Fq 'nemo-rl-supervl35-mxfp8' "$launcher_dir/compose_configs.py"

error_log=$(mktemp)
trap 'rm -f "$error_log"' EXIT
if env -u WANDB_API_KEY bash "$launcher_dir/submit_arm_sbatch.sh" bf16-bf16 >"$error_log" 2>&1; then
    echo "Submission without WANDB_API_KEY was accepted" >&2
    exit 1
fi
grep -Fq 'WANDB_API_KEY must be exported' "$error_log"
