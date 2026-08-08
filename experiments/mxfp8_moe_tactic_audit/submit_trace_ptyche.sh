#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
source "${SCRIPT_DIR}/provenance.sh"

ACTION=${ACTION:-dry-run}
case "${ACTION}" in
    test-only|dry-run|submit) ;;
    *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;;
esac

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-cb7dc7d7e560c0b95055772f1ee4d3a31a605edc}
MODEL=Qwen/Qwen3-30B-A3B
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
if [[ -n "${RUN_ID:-}" ]]; then RUN_ID=${RUN_ID}; elif [[ "${ACTION}" == submit ]]; then RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$; else RUN_ID=dry-run; fi
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/trace/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
WANDB_ENABLED=${WANDB_ENABLED:-false}
case "${WANDB_ENABLED}" in
    true|false) ;;
    *) echo "WANDB_ENABLED must be true or false" >&2; exit 2 ;;
esac

if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi

MODEL_SNAPSHOT=dry-run-not-validated
MODEL_REVISION=dry-run-not-validated
if [[ "${ACTION}" != dry-run ]]; then
    IFS=$'\t' read -r MODEL_SNAPSHOT MODEL_REVISION < <(
        audit_resolve_model_snapshot "${HF_MODEL_CACHE_DIR}" 16
    )
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
fi
if [[ "${ACTION}" == submit ]]; then
    [[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 1; }
fi

TRACE_DIR=${RUN_ROOT}/trace
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${NEMO_RL_COMMIT}" ]]
[[ "\${runtime_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]]
export VLLM_MXFP8_AUDIT_SOURCE_ROOT=${CUSTOM_VLLM_ROOT}
PYTHON_OVERLAY=${RUN_ROOT}/python-overlay
mkdir -p \${PYTHON_OVERLAY}
cat > \${PYTHON_OVERLAY}/sitecustomize.py <<'PY'
import os
from pathlib import Path

try:
    import vllm
except ModuleNotFoundError:
    # The NeMo-RL driver does not install the vLLM extra. Ray's vLLM actors do.
    pass
else:
    source = Path(os.environ["VLLM_MXFP8_AUDIT_SOURCE_ROOT"]).resolve() / "vllm"
    if not source.is_dir():
        raise RuntimeError(f"missing custom vLLM source: {source}")
    vllm.__path__.insert(0, str(source))
PY
export PYTHONPATH=\${PYTHON_OVERLAY}:\${PYTHONPATH:-}
for audit_module in \
  ${CUSTOM_VLLM_ROOT}/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py \
  ${CUSTOM_VLLM_ROOT}/vllm/model_executor/layers/fused_moe/experts/trtllm_moe_trace.py; do
  [[ -s "\${audit_module}" ]] || {
    echo "missing custom vLLM audit module: \${audit_module}" >&2
    exit 1
  }
done
unset VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
export VLLM_MXFP8_MOE_TRACE_DIR=${TRACE_DIR}
export VLLM_MXFP8_MOE_MODEL_REVISION=${MODEL_REVISION}
export VLLM_MXFP8_MOE_RUNTIME_FINGERPRINT=${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT}
export VLLM_MXFP8_MOE_DP_SIZE=16
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p ${TRACE_DIR}
python examples/run_grpo.py \\
  --config ${CONFIG} \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=4 \\
  cluster.segment_size=4 \\
  policy.model_name=${MODEL_SNAPSHOT} \\
  policy.generation.vllm_cfg.enforce_eager=true \\
  ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm \\
  grpo.max_num_steps=2 \\
  grpo.val_at_start=false \\
  checkpointing.enabled=false \\
  checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints \\
  logger.log_dir=${RUN_ROOT}/logs \\
  logger.wandb_enabled=${WANDB_ENABLED}
find ${TRACE_DIR} -type f -name '*.jsonl' -size +0c -print -quit | grep -q .
python - ${TRACE_DIR} ${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT} <<'PY'
import json
import re
import sys
from pathlib import Path

trace_dir = Path(sys.argv[1])
expected_fingerprint = sys.argv[2]
expected_ranks = set(range(16))
observed_ranks = set()
for trace_path in trace_dir.glob("moe-routing-rank*-pid*.jsonl"):
    match = re.fullmatch(r"moe-routing-rank(\d+)-pid\d+\.jsonl", trace_path.name)
    if match is None:
        raise RuntimeError(f"unexpected trace filename: {trace_path.name}")
    rank = int(match.group(1))
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty trace file: {trace_path}")
    for row in rows:
        if row["runtime_fingerprint"] != expected_fingerprint:
            raise RuntimeError(f"runtime fingerprint mismatch in {trace_path}")
        if row["cuda_graph_state"] != "trace-eager" or row["dp_size"] != 16:
            raise RuntimeError(f"unexpected trace execution state in {trace_path}")
    observed_ranks.add(rank)
if observed_ranks != expected_ranks:
    raise RuntimeError(
        f"incomplete trace rank coverage: expected={expected_ranks}, observed={observed_ranks}"
    )
PY
touch ${RUN_ROOT}/trace_complete
EOF
)

SBATCH_ARGS=(
    --nodes=4
    --exclusive
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --segment=4
    --time="${WALLTIME}"
    --job-name="coreai_dlalgo_llm-mxmoe.trace-${RUN_ID}"
    --output="${RUN_ROOT}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

printf 'action=%s\n' "${ACTION}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'trace_is_metadata_only=true\n'
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
printf '%s\n' "${COMMAND}"

case "${ACTION}" in
    dry-run) ;;
    test-only)
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
    submit)
        audit_write_manifest "${RUN_ROOT}" trace "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" \
            "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" "${CONFIG}" "${MODEL_SNAPSHOT}" \
            - "${SCRIPT_DIR}" "${SCRIPT_DIR}/submit_trace_ptyche.sh" \
            "${SCRIPT_DIR}/provenance.sh"
        CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 \
            BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
        ;;
esac
