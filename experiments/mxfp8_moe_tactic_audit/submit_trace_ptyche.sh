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

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-1de469ba64891f13c871ab008b42e7fdb970a817}
MODEL=Qwen/Qwen3-30B-A3B
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
if [[ -n "${RUN_ID:-}" ]]; then RUN_ID=${RUN_ID}; elif [[ "${ACTION}" == submit ]]; then RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$; else RUN_ID=dry-run; fi
RUN_ROOT=${RUN_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/trace/${RUN_ID}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
VLLM_ENVIRONMENT_ROOT=${VLLM_ENVIRONMENT_ROOT:-}
DRIVER_VENV=${VLLM_ENVIRONMENT_ROOT:+${VLLM_ENVIRONMENT_ROOT}/vllm-canonical}
HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
HF_HOME=${HF_HOME:-${WORK_ROOT}/hf}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
WALLTIME=${WALLTIME:-05:00:00}
WANDB_ENABLED=${WANDB_ENABLED:-false}
TRACE_WARMUP_CALLS=${TRACE_WARMUP_CALLS:-192}
TRACE_INTERVAL=${TRACE_INTERVAL:-127}
TRACE_MAX_SAMPLES=${TRACE_MAX_SAMPLES:-512}
AUTOTUNE_CACHE_CAPTURE_ROOT=${AUTOTUNE_CACHE_CAPTURE_ROOT:-}
case "${WANDB_ENABLED}" in
    true|false) ;;
    *) echo "WANDB_ENABLED must be true or false" >&2; exit 2 ;;
esac
[[ "${TRACE_WARMUP_CALLS}" =~ ^[0-9]+$ ]] || {
    echo "TRACE_WARMUP_CALLS must be a nonnegative integer" >&2
    exit 2
}
[[ "${TRACE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRACE_INTERVAL must be a positive integer" >&2
    exit 2
}
[[ "${TRACE_MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRACE_MAX_SAMPLES must be a positive integer" >&2
    exit 2
}

if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi

MODEL_SNAPSHOT=dry-run-not-validated
MODEL_REVISION=dry-run-not-validated
if [[ "${ACTION}" != dry-run ]]; then
    [[ -n "${VLLM_ENVIRONMENT_ROOT}" ]] || {
        echo "VLLM_ENVIRONMENT_ROOT must name a prepared vLLM environment" >&2
        exit 1
    }
    [[ -L "${DRIVER_VENV}/bin/python" || -x "${DRIVER_VENV}/bin/python" ]] || {
        echo "Prepared vLLM environment is missing: ${DRIVER_VENV}" >&2
        exit 1
    }
    [[ -f "${VLLM_ENVIRONMENT_ROOT}/READY" ]] || {
        echo "Prepared vLLM environment marker is missing: ${VLLM_ENVIRONMENT_ROOT}/READY" >&2
        exit 1
    }
    IFS=$'\t' read -r MODEL_SNAPSHOT MODEL_REVISION < <(
        audit_resolve_model_snapshot "${HF_MODEL_CACHE_DIR}" 16
    )
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
    audit_require_nonempty_dir "${HF_DATASETS_CACHE}/nvidia___open_math_instruct-2"
fi
if [[ "${ACTION}" == submit ]]; then
    [[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 1; }
    if [[ -n "${AUTOTUNE_CACHE_CAPTURE_ROOT}" ]]; then
        [[ ! -e "${AUTOTUNE_CACHE_CAPTURE_ROOT}" ]] || {
            echo "Autotune cache capture root already exists: ${AUTOTUNE_CACHE_CAPTURE_ROOT}" >&2
            exit 1
        }
    fi
fi

TRACE_DIR=${RUN_ROOT}/trace
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
CACHE_SETUP_COMMAND='unset VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'
CACHE_FINALIZE_COMMAND=''
if [[ -n "${AUTOTUNE_CACHE_CAPTURE_ROOT}" ]]; then
    CACHE_RAW_ROOT=${AUTOTUNE_CACHE_CAPTURE_ROOT}/raw
    CACHE_OUTPUT=${AUTOTUNE_CACHE_CAPTURE_ROOT}/autotune_configs.json
    CACHE_SETUP_COMMAND="mkdir -p ${CACHE_RAW_ROOT}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${CACHE_RAW_ROOT}"
    CACHE_FINALIZE_COMMAND=$(cat <<EOF
mapfile -d '' generated_caches < <(find ${CACHE_RAW_ROOT} -type f -name autotune_configs.json -print0)
[[ "\${#generated_caches[@]}" -eq 1 ]] || {
  echo "expected exactly one generated FlashInfer autotune cache, found \${#generated_caches[@]}" >&2
  exit 1
}
${DRIVER_VENV}/bin/python - "\${generated_caches[0]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="ascii") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict) or not any(key != "_metadata" for key in payload):
    raise RuntimeError(f"invalid or empty FlashInfer autotune cache: {path}")
PY
install -m 0444 "\${generated_caches[0]}" ${CACHE_OUTPUT}
printf 'captured_cache=%s\n' ${CACHE_OUTPUT}
EOF
)
fi
COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
runtime_nemo_rl_commit=\$(git rev-parse HEAD)
runtime_vllm_commit=\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)
[[ "\${runtime_nemo_rl_commit}" == "${NEMO_RL_COMMIT}" ]]
[[ "\${runtime_vllm_commit}" == "${EXPECTED_VLLM_COMMIT}" ]]
export VLLM_MXFP8_AUDIT_SOURCE_ROOT=${CUSTOM_VLLM_ROOT}
export NEMO_RL_VENV_DIR=${VLLM_ENVIRONMENT_ROOT}
export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}
export VIRTUAL_ENV=${DRIVER_VENV}
export PATH=${DRIVER_VENV}/bin:\${PATH}
PYTHON_OVERLAY=${RUN_ROOT}/python-overlay
mkdir -p \${PYTHON_OVERLAY}
cat > \${PYTHON_OVERLAY}/sitecustomize.py <<'PY'
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

root = Path(os.environ["VLLM_MXFP8_AUDIT_SOURCE_ROOT"]).resolve() / "vllm"
module_root = root / "model_executor/layers/fused_moe/experts"
audit_modules = {
    "vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe": (
        module_root / "trtllm_fp8_moe.py"
    ),
    "vllm.model_executor.layers.fused_moe.experts.trtllm_moe_trace": (
        module_root / "trtllm_moe_trace.py"
    ),
}


class AuditModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = audit_modules.get(fullname)
        if source is None:
            return None
        if not source.is_file():
            raise ImportError(f"missing custom vLLM audit module: {source}")
        return importlib.util.spec_from_file_location(fullname, source)


sys.meta_path.insert(0, AuditModuleFinder())
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
${CACHE_SETUP_COMMAND}
export VLLM_MXFP8_MOE_TRACE_DIR=${TRACE_DIR}
export VLLM_MXFP8_MOE_TRACE_WARMUP_CALLS=${TRACE_WARMUP_CALLS}
export VLLM_MXFP8_MOE_TRACE_INTERVAL=${TRACE_INTERVAL}
export VLLM_MXFP8_MOE_TRACE_MAX_SAMPLES=${TRACE_MAX_SAMPLES}
export VLLM_MXFP8_MOE_MODEL_REVISION=${MODEL_REVISION}
export VLLM_MXFP8_MOE_RUNTIME_FINGERPRINT=${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT}
export VLLM_MXFP8_MOE_DP_SIZE=16
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p ${TRACE_DIR}
[[ -x ${DRIVER_VENV}/bin/python ]] || {
  echo "Prepared vLLM environment is not executable in the container: ${DRIVER_VENV}" >&2
  exit 1
}
container_ray_version=\$(/opt/nemo_rl_venv/bin/python -c 'import ray; print(ray.__version__)')
prepared_ray_version=\$(${DRIVER_VENV}/bin/python -c 'import ray; print(ray.__version__)')
[[ "\${container_ray_version}" == "\${prepared_ray_version}" ]] || {
  echo "Ray version mismatch before trace launch: container=\${container_ray_version}, prepared=\${prepared_ray_version}" >&2
  exit 1
}
${DRIVER_VENV}/bin/python - <<'PY'
import flashinfer
import vllm
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

if vllm.__version__ != "0.25.1":
    raise RuntimeError(f"Expected vLLM 0.25.1, found {vllm.__version__}")
if flashinfer.__version__ != "0.6.13":
    raise RuntimeError(f"Expected FlashInfer 0.6.13, found {flashinfer.__version__}")
print(f"vLLM={vllm.__version__} FlashInfer={flashinfer.__version__}")
print(f"MoE API={RoutedExperts.__name__}/{MoERunner.__name__}")
PY
${DRIVER_VENV}/bin/python examples/run_grpo.py \\
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
${DRIVER_VENV}/bin/python - ${TRACE_DIR} ${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT} <<'PY'
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
${CACHE_FINALIZE_COMMAND}
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
printf 'trace_warmup_calls=%s\n' "${TRACE_WARMUP_CALLS}"
printf 'trace_interval=%s\n' "${TRACE_INTERVAL}"
printf 'trace_max_samples_per_process=%s\n' "${TRACE_MAX_SAMPLES}"
if [[ -n "${AUTOTUNE_CACHE_CAPTURE_ROOT}" ]]; then
    printf 'autotune_cache_capture_root=%s\n' "${AUTOTUNE_CACHE_CAPTURE_ROOT}"
else
    printf 'autotune_cache_capture_root=disabled\n'
fi
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
