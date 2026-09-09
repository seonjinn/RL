#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/precision-matrix-submit-test.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

mkdir -p \
  "${TMP_ROOT}/bin" \
  "${TMP_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B" \
  "${TMP_ROOT}/home"
touch "${TMP_ROOT}/container.sqsh" "${TMP_ROOT}/home/.netrc"

cat > "${TMP_ROOT}/bin/sbatch" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@"
EOF
chmod +x "${TMP_ROOT}/bin/sbatch"

output=$(
  PATH="${TMP_ROOT}/bin:${PATH}" \
  ACTION=test-only \
  CLUSTER=oci \
  PARTITION=batch \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-mxfp8 \
  MAX_STEPS=20 \
  AFTEROK_JOB_ID=12345 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  CONTAINER="${TMP_ROOT}/container.sqsh" \
  HF_HOME_SOURCE="${TMP_ROOT}/hf" \
  WANDB_HOME="${TMP_ROOT}/home" \
  RESULT_ROOT="${TMP_ROOT}/results" \
  LOCAL_ROOT="${TMP_ROOT}/local" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -Fx -- '--dependency=afterok:12345' <<<"${output}" >/dev/null
grep -Fx -- 'force_rebuild_venvs=true' <<<"${output}" >/dev/null

render_arm() {
  local arm="$1"

  ACTION=render \
  CLUSTER=lyris \
  MODEL=qwen235 \
  MODE=async \
  ARM="${arm}" \
  SLURM_ACCOUNT=test \
  "${SCRIPT_DIR}/submit.sh"
}

assert_arm() {
  local arm="$1"
  shift
  local output

  output=$(render_arm "${arm}")
  for expected in "$@"; do
    grep -F -- "${expected}" <<<"${output}" >/dev/null
  done
}

assert_arm bf16-bf16 \
  policy.megatron_cfg.fp8_cfg.enabled=false \
  policy.generation.vllm_cfg.precision=bfloat16
assert_arm bf16-mxfp8 \
  policy.megatron_cfg.fp8_cfg.enabled=false \
  policy.generation.vllm_cfg.precision=fp8
assert_arm mxfp8-false-bf16 \
  policy.megatron_cfg.fp8_cfg.fp8_param=false \
  te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed.yaml \
  policy.generation.vllm_cfg.precision=bfloat16
assert_arm mxfp8-false-mxfp8 \
  policy.megatron_cfg.fp8_cfg.fp8_param=false \
  te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed.yaml \
  policy.generation.vllm_cfg.precision=fp8
assert_arm mxfp8-true-mxfp8 \
  policy.megatron_cfg.fp8_cfg.fp8_param=true \
  te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed_fp8param.yaml \
  policy.generation.vllm_cfg.precision=fp8

set +e
unsupported_output=$(render_arm mxfp8-true-bf16 2>&1)
unsupported_rc=$?
set -e

[[ "${unsupported_rc}" -eq 3 ]]
grep -F -- 'native MXFP8 parameter storage cannot refit a BF16 rollout consumer' \
  <<<"${unsupported_output}" >/dev/null

assert_config_line() {
  local file="$1"
  local expected="$2"

  grep -Fx -- "${expected}" "${SCRIPT_DIR}/${file}" >/dev/null
}

# Lightning uses a weighted-squared-ReLU checkpoint. Keep the small matrix
# recipe aligned with its production model settings instead of inheriting the
# Qwen3.5 activation and experimental HybridEP dispatcher.
assert_config_line lightning-sync.yaml '    use_fused_weighted_squared_relu: true'
assert_config_line lightning-sync.yaml '    apply_rope_fusion: true'
assert_config_line lightning-sync.yaml '    moe_token_dispatcher_type: alltoall'
assert_config_line lightning-sync.yaml '      lr: 4.0e-6'
assert_config_line lightning-sync.yaml '      weight_decay: 0.0'
assert_config_line lightning-sync.yaml '  reference_policy_kl_penalty: 0.0'
assert_config_line lightning-sync.yaml '  seq_logprob_error_threshold: 2'

# Async training spans a separate, smaller actor group. It must not inherit the
# colocated Sync HybridEP dispatcher through the local defaults chain.
assert_config_line qwen30-async.yaml '    moe_token_dispatcher_type: alltoall'
assert_config_line qwen30-async.yaml '    moe_flex_dispatcher_backend: null'
assert_config_line qwen30-async.yaml '    moe_hybridep_num_sms: null'
assert_config_line qwen35-async.yaml '    moe_token_dispatcher_type: alltoall'
assert_config_line qwen35-async.yaml '    moe_flex_dispatcher_backend: null'
assert_config_line qwen35-async.yaml '    moe_hybridep_num_sms: null'
