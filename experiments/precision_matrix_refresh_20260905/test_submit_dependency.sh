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
