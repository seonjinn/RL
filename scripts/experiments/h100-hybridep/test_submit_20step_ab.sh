#!/bin/bash

set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
launcher="$project_root/scripts/experiments/h100-hybridep/submit_20step_ab.sh"
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

capture_file="$temp_dir/sbatch.env"
container="$temp_dir/nightly.sqsh"
hf_home="$temp_dir/hf_home"
run_root="$temp_dir/run"
touch "$container"
mkdir -p "$hf_home"

sbatch() {
  {
    printf 'args=%s\n' "$*"
    printf 'command=%s\n' "$COMMAND"
    printf 'cudnn_home=%s\n' "${CUDNN_HOME-<unset>}"
    printf 'cudnn_path=%s\n' "${CUDNN_PATH-<unset>}"
    printf 'ld_library_path=%s\n' "$LD_LIBRARY_PATH"
    printf 'uv_project_environment=%s\n' "$UV_PROJECT_ENVIRONMENT"
  } >"$capture_file"
}
git() {
  if [[ "${1:-}" == status ]]; then
    return 0
  fi
  /usr/bin/git "$@"
}
export -f sbatch
export -f git
export capture_file

ACCOUNT=coreai_dlalgo_llm \
PARTITION=batch \
CONTAINER="$container" \
HF_HOME="$hf_home" \
RUN_ROOT="$run_root" \
EXPECTED_RL_COMMIT=$(git rev-parse HEAD) \
RUN_NAME=test-nano-hybridep \
CUDNN_HOME=/opt/nemo_rl_venv/stale-cudnn \
CUDNN_PATH=/opt/nemo_rl_venv/stale-cudnn \
TEST_ONLY=1 \
bash "$launcher" nano hybridep

grep -Fq -- '--nodes=2' "$capture_file"
grep -Fq 'examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml' "$capture_file"
grep -Fq 'policy.megatron_cfg.moe_token_dispatcher_type=flex' "$capture_file"
grep -Fq '+policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep' "$capture_file"
grep -Fq '+policy.megatron_cfg.moe_hybridep_num_sms=32' "$capture_file"
grep -Fq 'policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true' "$capture_file"
grep -Fq 'uv sync --locked --no-install-project' "$capture_file"
grep -Fq 'cudnn_home=<unset>' "$capture_file"
grep -Fq 'cudnn_path=<unset>' "$capture_file"
grep -Fq 'nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/nvidia/cudnn/lib' "$capture_file"
