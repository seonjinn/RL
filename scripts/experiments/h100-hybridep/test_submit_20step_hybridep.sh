#!/bin/bash

set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
launcher="$project_root/scripts/experiments/h100-hybridep/submit_20step_hybridep.sh"
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

container="$temp_dir/nightly.sqsh"
hf_home="$temp_dir/hf_home"
touch "$container"
mkdir -p "$hf_home"

sbatch() {
  : "${CAPTURE_FILE:?}"
  printf 'args=%s\n' "$*" >"$CAPTURE_FILE"
  printf 'command=%s\n' "$COMMAND" >>"$CAPTURE_FILE"
}
export -f sbatch

run_case() {
  local model=$1
  local expected_nodes=$2
  local expected_config=$3
  local capture_file="$temp_dir/$model.env"

  CAPTURE_FILE="$capture_file" \
  ACCOUNT=test-account \
  PARTITION=batch \
  CONTAINER="$container" \
  HF_HOME="$hf_home" \
  RUN_ROOT="$temp_dir/$model" \
  EXPECTED_RL_COMMIT=$(git rev-parse HEAD) \
  RUN_NAME="test-$model" \
  TEST_ONLY=1 \
  bash "$launcher" "$model"

  grep -Fq -- "--nodes=$expected_nodes" "$capture_file"
  grep -Fq "$expected_config" "$capture_file"
  grep -Fq 'grpo.max_num_steps=20' "$capture_file"
  grep -Fq -- '--gpus-per-node=8' "$capture_file"
  if grep -Fq -- '--exclusive' "$capture_file"; then
    printf 'Launcher must not request exclusive nodes\n' >&2
    exit 1
  fi
}

run_case \
  qwen30 \
  4 \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
run_case \
  qwen235 \
  16 \
  examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml
run_case \
  nano \
  2 \
  examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml

grep -Fq 'policy.megatron_cfg.moe_token_dispatcher_type=flex' "$temp_dir/nano.env"
grep -Fq 'policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep' "$temp_dir/nano.env"
grep -Fq 'policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=false' "$temp_dir/nano.env"
if grep -Fq 'policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true' "$temp_dir/nano.env"; then
  printf 'Nano PP=2 validation must use MCore variable-input padding\n' >&2
  exit 1
fi
