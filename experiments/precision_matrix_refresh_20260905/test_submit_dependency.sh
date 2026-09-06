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

render_arm() {
  local arm="$1"
  local model="${2:-qwen30}"
  local mode="${3:-async}"
  ACTION=render \
    CLUSTER=oci \
    MODEL="${model}" \
    MODE="${mode}" \
    ARM="${arm}" \
    SLURM_ACCOUNT=test \
    REPO="${REPO}" \
    "${SCRIPT_DIR}/submit.sh"
}

fp8_param_false=$(render_arm mxfp8-param-false)
grep -F -- 'policy.megatron_cfg.fp8_cfg.fp8_param=false' <<<"${fp8_param_false}" >/dev/null
grep -F -- 'te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed_mxfp8.yaml' <<<"${fp8_param_false}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.precision=fp8' <<<"${fp8_param_false}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.refit_prequantize=false' <<<"${fp8_param_false}" >/dev/null
grep -F -- 'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm' <<<"${fp8_param_false}" >/dev/null

fp8_param_false_sync=$(render_arm mxfp8-param-false qwen30 sync)
grep -F -- 'policy.generation.vllm_cfg.refit_prequantize=true' <<<"${fp8_param_false_sync}" >/dev/null

fp8_param_true=$(render_arm mxfp8-param-true)
grep -F -- 'policy.megatron_cfg.fp8_cfg.fp8_param=true' <<<"${fp8_param_true}" >/dev/null
grep -F -- 'te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed_fp8param.yaml' <<<"${fp8_param_true}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.precision=fp8' <<<"${fp8_param_true}" >/dev/null
grep -F -- 'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm' <<<"${fp8_param_true}" >/dev/null

fp8_param_false_bf16=$(render_arm mxfp8-param-false-bf16)
grep -F -- 'policy.megatron_cfg.fp8_cfg.fp8_param=false' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'te_precision_config_file=experiments/precision_matrix_refresh_20260905/te_routed_mxfp8.yaml' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.precision=bfloat16' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.is_mx=false' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.num_first_layers_in_bf16=0' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.num_last_layers_in_bf16=0' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.refit_prequantize=false' <<<"${fp8_param_false_bf16}" >/dev/null
grep -F -- 'policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm' <<<"${fp8_param_false_bf16}" >/dev/null

if render_arm mxfp8-param-true-bf16 >/dev/null 2>&1; then
  echo "native MXFP8 training storage must not be offered with BF16 rollout" >&2
  exit 1
fi

qwen235_bf16=$(render_arm mxfp8-param-false-bf16 qwen235)
grep -F -- 'policy.generation.vllm_kwargs.moe_backend=triton' <<<"${qwen235_bf16}" >/dev/null

for model in qwen30 qwen235 lightning qwen35; do
  sync_output=$(render_arm mxfp8-param-false-bf16 "${model}" sync)
  grep -Fx -- "config=experiments/precision_matrix_refresh_20260905/${model}-sync.yaml" <<<"${sync_output}" >/dev/null
  grep -F -- 'refit_transport: null' "${REPO}/experiments/precision_matrix_refresh_20260905/${model}-sync.yaml" >/dev/null

  async_output=$(render_arm mxfp8-param-false-bf16 "${model}" async)
  grep -Fx -- "config=experiments/precision_matrix_refresh_20260905/${model}-async.yaml" <<<"${async_output}" >/dev/null
  grep -F -- 'refit_transport: nccl_reshard' "${REPO}/experiments/precision_matrix_refresh_20260905/${model}-async.yaml" >/dev/null
done
