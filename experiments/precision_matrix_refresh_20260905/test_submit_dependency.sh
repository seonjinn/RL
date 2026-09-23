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
printf 'COMMAND=%s\n' "${COMMAND:-}"
printf 'SETUP_COMMAND=%s\n' "${SETUP_COMMAND:-}"
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
  RUN_GROUP=dependency-test \
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
grep -F -- "${TMP_ROOT}/results/source-archives/nemo-rl-" <<<"${output}" >/dev/null
grep -F -- "source_payload_sha=$(git -C "${REPO}" rev-parse HEAD)" \
  <<<"${output}" >/dev/null
grep -F -- "export RAY_TMPDIR=${TMP_ROOT}/local/pmx-oci-qwen30-sync-bf16-mxfp8-default-dependency-test/ray;" \
  <<<"${output}" >/dev/null
if grep -F -- '${SLURM_JOB_ID}' <<<"${output}" >/dev/null; then
  echo "Rendered setup and driver commands must not depend on SLURM_JOB_ID" >&2
  exit 1
fi

grep -F -- 'export PATH="${SLURM_COMMAND_PATH}:${PATH}"' "${REPO}/ray.sub" >/dev/null

qwen35_output=$(
  ACTION=render \
  CLUSTER=oci \
  MODEL=qwen35 \
  MODE=sync \
  ARM=bf16-bf16 \
  MAX_STEPS=20 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- 'grpo.num_prompts_per_step=128' <<<"${qwen35_output}" >/dev/null
grep -F -- 'grpo.num_generations_per_prompt=16' <<<"${qwen35_output}" >/dev/null
grep -F -- 'policy.train_global_batch_size=2048' <<<"${qwen35_output}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.quantization_ignore_patterns=\[\]' \
  <<<"${qwen35_output}" >/dev/null

if QWEN35_TRAIN_GLOBAL_BATCH_SIZE=1024 \
  ACTION=render \
  CLUSTER=oci \
  MODEL=qwen35 \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh" >/dev/null 2>&1; then
  echo "Qwen3.5 performance recipes must reject a non-standard GBS" >&2
  exit 1
fi

super_output=$(
  ACTION=render \
  CLUSTER=oci \
  MODEL=super \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  SUPER_GPU_MEMORY_UTILIZATION=0.6 \
  MAX_STEPS=20 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- 'policy.generation.vllm_cfg.gpu_memory_utilization=0.6' \
  <<<"${super_output}" >/dev/null

qwen235_memory_output=$(
  ACTION=render \
  CLUSTER=lyris \
  MODEL=qwen235 \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  GPU_MEMORY_UTILIZATION=0.65 \
  MAX_STEPS=20 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- 'gpu_memory_utilization=0.65' <<<"${qwen235_memory_output}" >/dev/null
grep -F -- 'policy.generation.vllm_cfg.gpu_memory_utilization=0.65' \
  <<<"${qwen235_memory_output}" >/dev/null

# Keep the upstream Qwen3-235B performance workload and topology intact.
grep -F -- 'defaults: ../../examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml' \
  "${SCRIPT_DIR}/qwen235-performance-sync.yaml" >/dev/null
grep -F -- 'reuse_optimizer_cpu_buffers_for_refit: true' \
  "${SCRIPT_DIR}/qwen235-performance-sync.yaml" >/dev/null

mkdir -p "${TMP_ROOT}/direct-model"
direct_model_output=$(
  PATH="${TMP_ROOT}/bin:${PATH}" \
  ACTION=test-only \
  CLUSTER=oci \
  PARTITION=batch \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  MODEL_SNAPSHOT_OVERRIDE="${TMP_ROOT}/direct-model" \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  CONTAINER="${TMP_ROOT}/container.sqsh" \
  HF_HOME_SOURCE="${TMP_ROOT}/hf" \
  WANDB_HOME="${TMP_ROOT}/home" \
  RESULT_ROOT="${TMP_ROOT}/results" \
  LOCAL_ROOT="${TMP_ROOT}/local" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- "policy.model_name=${TMP_ROOT}/direct-model" \
  <<<"${direct_model_output}" >/dev/null

touch "${TMP_ROOT}/existing-source.tar"
source_archive_sha256=$(sha256sum "${TMP_ROOT}/existing-source.tar" | cut -d ' ' -f 1)
source_archive_output=$(
  PATH="${TMP_ROOT}/bin:${PATH}" \
  ACTION=test-only \
  CLUSTER=oci \
  PARTITION=batch \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-bf16 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  CONTAINER="${TMP_ROOT}/container.sqsh" \
  HF_HOME_SOURCE="${TMP_ROOT}/hf" \
  WANDB_HOME="${TMP_ROOT}/home" \
  RESULT_ROOT="${TMP_ROOT}/results" \
  LOCAL_ROOT="${TMP_ROOT}/local" \
  SOURCE_ARCHIVE_OVERRIDE="${TMP_ROOT}/existing-source.tar" \
  SOURCE_ARCHIVE_SHA256="${source_archive_sha256}" \
  SOURCE_PAYLOAD_SHA=test-source \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- "source_archive_override=${TMP_ROOT}/existing-source.tar" \
  <<<"${source_archive_output}" >/dev/null
grep -F -- "source_archive_sha256=${source_archive_sha256}" \
  <<<"${source_archive_output}" >/dev/null
grep -F -- 'source_payload_sha=test-source' \
  <<<"${source_archive_output}" >/dev/null
grep -F -- "tar -xf ${TMP_ROOT}/existing-source.tar" \
  <<<"${source_archive_output}" >/dev/null

if ACTION=render \
  CLUSTER=oci \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-bf16 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  SOURCE_ARCHIVE_OVERRIDE="${TMP_ROOT}/existing-source.tar" \
  SOURCE_ARCHIVE_SHA256="${source_archive_sha256}" \
  "${SCRIPT_DIR}/submit.sh" >/dev/null 2>&1; then
  echo "Source archive overrides must require an explicit payload commit" >&2
  exit 1
fi
