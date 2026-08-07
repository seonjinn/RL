#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR_OVERRIDE:-$(realpath "${SCRIPT_DIR}/../..")}
source "${SCRIPT_DIR}/provenance.sh"

ACTION=${ACTION:-dry-run}
VALIDATION_MODE=${VALIDATION_MODE:-run}
COMPARE_ACTION=${COMPARE_ACTION:-dry-run}
ARM=${ARM:-candidate}
MAX_STEPS=${MAX_STEPS:-2}
case "${ACTION}" in test-only|dry-run|submit) ;; *) echo "Unsupported ACTION: ${ACTION}" >&2; exit 2 ;; esac
case "${VALIDATION_MODE}" in run|compare) ;; *) echo "VALIDATION_MODE must be run or compare" >&2; exit 2 ;; esac
case "${COMPARE_ACTION}" in dry-run|run) ;; *) echo "COMPARE_ACTION must be dry-run or run" >&2; exit 2 ;; esac
case "${ARM}" in stock|candidate) ;; *) echo "ARM must be stock or candidate" >&2; exit 2 ;; esac
case "${MAX_STEPS}" in 2|8) ;; *) echo "MAX_STEPS must be 2 or 8" >&2; exit 2 ;; esac

EXPECTED_VLLM_COMMIT=${EXPECTED_VLLM_COMMIT:-cb7dc7d7e560c0b95055772f1ee4d3a31a605edc}
MODEL=Qwen/Qwen3-30B-A3B
CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
WORK_ROOT=${WORK_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/sna}
if [[ -n "${RUN_ID:-}" ]]; then
    RUN_ID=${RUN_ID}
elif [[ "${ACTION}" == submit ]]; then
    RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
else
    RUN_ID=dry-run
fi
RUN_BASE=${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/validation/${ARM}/${RUN_ID}
RUN_ROOT=${RUN_ROOT:-${RUN_BASE}/steps-${MAX_STEPS}}
CONTAINER=${CONTAINER:-${WORK_ROOT}/containers/nemo_rl_nightly_20260711_vllm025_ffmpeg_20260713_1218.sqsh}
CUSTOM_VLLM_ROOT=${CUSTOM_VLLM_ROOT:-${REPO_DIR}/3rdparty/vllm}
HF_MODEL_CACHE_DIR=${HF_MODEL_CACHE_DIR:-${WORK_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B}
STOCK_CACHE_ROOT=${STOCK_CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/cache/stock}
CANDIDATE_CACHE_ROOT=${CANDIDATE_CACHE_ROOT:-${WORK_ROOT}/.cache/mxfp8-moe-tactic-audit/cache/candidate}
ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-}
DRIVER_VENV=${DRIVER_VENV:-${WORK_ROOT}/.cache/nemo-rl-vllm0251-worker-venvs/vllm-canonical}
GSM8K_EVALUATOR=${GSM8K_EVALUATOR:-${WORK_ROOT}/vllm-benchmark/experiments/eval/gsm8k_vllm_eval.py}
GSM8K_DATASET=${GSM8K_DATASET:-${WORK_ROOT}/vllm-benchmark/experiments/eval/data/gsm8k_test_openai_1319.jsonl}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-18000}
case "${ARM}" in stock) CACHE_ROOT=${STOCK_CACHE_ROOT} ;; candidate) CACHE_ROOT=${CANDIDATE_CACHE_ROOT} ;; esac
CACHE_FILE=${CACHE_ROOT}/autotune_configs.json
CACHE_MANIFEST=${CACHE_MANIFEST:-${CANDIDATE_CACHE_ROOT}/cache_manifest.json}

if [[ "${VALIDATION_MODE}" == compare ]]; then
    COMPARE_ROOT=${COMPARE_ROOT:-${WORK_ROOT}/experiments/mxfp8-moe-tactic-audit/validation}
    COMPARE_RUN_ID=${COMPARE_RUN_ID:-${RUN_ID}}
    [[ -n "${COMPARE_RUN_ID}" && "${COMPARE_RUN_ID}" != *,* ]] || { echo "COMPARE_RUN_ID must name exactly one run pair" >&2; exit 2; }
    COMMAND="mkdir -p ${COMPARE_ROOT}
python ${SCRIPT_DIR}/validate_correctness.py generation --stock ${COMPARE_ROOT}/stock/${COMPARE_RUN_ID}/steps-8/generation.jsonl --candidate ${COMPARE_ROOT}/candidate/${COMPARE_RUN_ID}/steps-8/generation.jsonl > ${COMPARE_ROOT}/deterministic_generation_comparison.json
python ${SCRIPT_DIR}/compare_gsm8k.py --stock ${COMPARE_ROOT}/stock/${COMPARE_RUN_ID}/steps-8/gsm8k --candidate ${COMPARE_ROOT}/candidate/${COMPARE_RUN_ID}/steps-8/gsm8k > ${COMPARE_ROOT}/gsm8k_comparison.json"
    printf 'validation_mode=compare\n%s\n' "${COMMAND}"
    [[ "${ACTION}" == dry-run ]] || {
        echo "VALIDATION_MODE=compare is local; ACTION must be dry-run" >&2
        exit 2
    }
    case "${COMPARE_ACTION}" in
        dry-run) ;;
        run) eval "${COMMAND}" ;;
    esac
    exit 0
fi

MODEL_SNAPSHOT=dry-run-not-validated
MODEL_REVISION=dry-run-not-validated
if [[ "${ACTION}" == submit ]]; then
    audit_prepare_submit "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}"
fi
if [[ "${ACTION}" != dry-run ]]; then
    IFS=$'\t' read -r MODEL_SNAPSHOT MODEL_REVISION < <(audit_resolve_model_snapshot "${HF_MODEL_CACHE_DIR}" 16)
    audit_require_nonempty_dir "${CACHE_ROOT}"
    [[ -s "${CACHE_FILE}" ]] || { echo "Missing runtime tactic cache: ${CACHE_FILE}" >&2; exit 1; }
    [[ -f "${CACHE_MANIFEST}" ]] || { echo "Missing qualification cache manifest: ${CACHE_MANIFEST}" >&2; exit 1; }
    [[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 1; }
    [[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 1; }
fi
NEMO_RL_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD)
CACHE_SHA256=dry-run-not-validated
MODEL_SHA256=dry-run-not-validated
RECIPE_SHA256=dry-run-not-validated
SCRIPTS_SHA256=dry-run-not-validated
EXECUTION_INPUTS_SHA256=dry-run-not-validated
VALIDATION_EXECUTION_INPUTS=(
    "${SCRIPT_DIR}"
    "${SCRIPT_DIR}/submit_validation_ptyche.sh"
    "${SCRIPT_DIR}/provenance.sh"
    "${SCRIPT_DIR}/validate_correctness.py"
    "${SCRIPT_DIR}/compare_gsm8k.py"
    "${GSM8K_EVALUATOR}"
)
if [[ "${ACTION}" != dry-run ]]; then
    CACHE_SHA256=$(audit_sha256_path "${CACHE_FILE}")
    MODEL_SHA256=$(audit_sha256_path "${MODEL_SNAPSHOT}")
    RECIPE_SHA256=$(audit_sha256_path "${REPO_DIR}/${CONFIG}")
    SCRIPTS_SHA256=$(audit_scripts_sha256 "${SCRIPT_DIR}")
    EXECUTION_INPUTS_SHA256=$(audit_execution_inputs_sha256 "${VALIDATION_EXECUTION_INPUTS[@]}")
fi
SMOKE_MANIFEST=${SMOKE_MANIFEST:-${RUN_BASE}/steps-2/run_manifest.json}
SMOKE_MARKER=${SMOKE_MARKER:-${RUN_BASE}/smoke-${ARM}-${CACHE_SHA256}.json}
if [[ "${MAX_STEPS}" == 8 && "${ACTION}" != dry-run ]]; then
    [[ -f "${SMOKE_MANIFEST}" && -f "${SMOKE_MARKER}" ]] || { echo "Missing arm-specific two-step smoke marker" >&2; exit 1; }
    audit_assert_smoke_manifest_matches "${SMOKE_MANIFEST}" "${NEMO_RL_COMMIT}" \
        "${EXPECTED_VLLM_COMMIT}" "${RECIPE_SHA256}" "${MODEL_SHA256}" \
        "${CACHE_SHA256}" "${SCRIPTS_SHA256}" "${EXECUTION_INPUTS_SHA256}"
    grep -Fq "\"cache_sha256\": \"${CACHE_SHA256}\"" "${SMOKE_MARKER}" || { echo "Stale smoke marker cache" >&2; exit 1; }
    grep -Fq "\"model_snapshot_sha256\": \"${MODEL_SHA256}\"" "${SMOKE_MARKER}" || { echo "Stale smoke marker model" >&2; exit 1; }
    grep -Fq "\"smoke_manifest_sha256\": \"$(audit_sha256_path "${SMOKE_MANIFEST}")\"" "${SMOKE_MARKER}" || { echo "Stale smoke marker manifest" >&2; exit 1; }
fi

POST_RUN=''
if [[ "${MAX_STEPS}" == 8 ]]; then
    POST_RUN=$(cat <<EOF
server_pid=
cleanup_server() { [[ -z "\${server_pid}" ]] || { kill "\${server_pid}" 2>/dev/null || true; wait "\${server_pid}" 2>/dev/null || true; }; }
trap cleanup_server EXIT INT TERM
${DRIVER_VENV}/bin/vllm serve ${MODEL_SNAPSHOT} --tokenizer ${MODEL_SNAPSHOT} --return-tokens-as-token-ids --host 127.0.0.1 --port ${VLLM_SERVER_PORT} > ${RUN_ROOT}/vllm-server.log 2>&1 &
server_pid=\$!
for _ in {1..120}; do curl -fsS http://127.0.0.1:${VLLM_SERVER_PORT}/health >/dev/null && break; sleep 1; done
curl -fsS http://127.0.0.1:${VLLM_SERVER_PORT}/health >/dev/null
${DRIVER_VENV}/bin/python - ${RUN_ROOT}/generation.jsonl ${MODEL_REVISION} ${NEMO_RL_COMMIT}-${EXPECTED_VLLM_COMMIT} <<'PY'
import hashlib, json, sys
from urllib.request import Request, urlopen
out, revision, runtime = sys.argv[1:]
rows = []
for identifier, prompt in (("fixed-0", "What is 1 plus 1?"), ("fixed-1", "Return the word stable.")):
    payload = {"model": "${MODEL_SNAPSHOT}", "prompt": prompt, "temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 20260807, "logprobs": 1, "return_token_ids": True, "return_tokens_as_token_ids": True}
    request = Request("http://127.0.0.1:${VLLM_SERVER_PORT}/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response: choice = json.load(response)["choices"][0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(isinstance(token_id, int) for token_id in token_ids):
        raise RuntimeError("vLLM did not return generated token IDs")
    rows.append({"id": identifier, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "token_ids": token_ids, "provenance": {"model_revision": revision, "tokenizer_revision": revision, "runtime_fingerprint": runtime, "decoding": {"mode": "greedy", "temperature": 0, "top_p": 1, "seed": 20260807, "max_tokens": 32}}})
with open(out, "w", encoding="ascii") as handle:
    for row in rows: handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\\n")
PY
${DRIVER_VENV}/bin/python ${GSM8K_EVALUATOR} --endpoint http://127.0.0.1:${VLLM_SERVER_PORT} --model ${MODEL} --dataset ${GSM8K_DATASET} --limit 1319 --seed 20260807 --concurrency 1 --output-dir ${RUN_ROOT}/gsm8k --provenance-json ${RUN_ROOT}/run_manifest.json
EOF
)
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO_DIR}
source ${CUSTOM_VLLM_ROOT}/nemo-rl.env
[[ "\$(git rev-parse HEAD)" == "${NEMO_RL_COMMIT}" ]]
[[ "\$(git -C ${CUSTOM_VLLM_ROOT} rev-parse HEAD)" == "${EXPECTED_VLLM_COMMIT}" ]]
export HF_HOME=${WORK_ROOT}/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${CACHE_ROOT}
export MXFP8_MOE_CUDA_GRAPH_REPLAY=required
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_EXPERT_PARALLEL_SIZE=1
mkdir -p ${RUN_ROOT}
python examples/run_grpo.py --config ${CONFIG} cluster.num_nodes=4 cluster.gpus_per_node=4 cluster.segment_size=4 policy.model_name=${MODEL_SNAPSHOT} policy.generation.vllm_cfg.enforce_eager=false ++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm grpo.max_num_steps=${MAX_STEPS} grpo.val_at_start=false checkpointing.enabled=false checkpointing.checkpoint_dir=${RUN_ROOT}/checkpoints logger.log_dir=${RUN_ROOT}/logs logger.wandb_enabled=false
RUNTIME_FINGERPRINTS_JSON=\$(${DRIVER_VENV}/bin/python -m experiments.mxfp8_moe_tactic_audit.observe_runtime --nemo-rl-root ${REPO_DIR} --vllm-root ${CUSTOM_VLLM_ROOT} --model-snapshot ${MODEL_SNAPSHOT} --container ${CONTAINER} --cache-root ${CACHE_ROOT})
if [[ ${MAX_STEPS} -eq 8 ]]; then
  ${DRIVER_VENV}/bin/python -m experiments.mxfp8_moe_tactic_audit.collect_results --write-run-evidence --run-root ${RUN_ROOT} --arm ${ARM} --run-id ${RUN_ID} --metadata-json '{"batch":"64 prompts x 32 generations","generation_settings":"max_total_sequence_length=4096; enforce_eager=false; CUDA Graph replay required","run_id":"${RUN_ID}","run_kind":"validation","topology":"4 nodes x 4 GPUs"}' --runtime-fingerprints-json "\${RUNTIME_FINGERPRINTS_JSON}"
fi
if [[ ${MAX_STEPS} -eq 2 ]]; then printf '{"arm":"${ARM}","cache_sha256":"${CACHE_SHA256}","execution_inputs_sha256":"${EXECUTION_INPUTS_SHA256}","model_snapshot_sha256":"${MODEL_SHA256}","nemo_rl_commit":"${NEMO_RL_COMMIT}","recipe_sha256":"${RECIPE_SHA256}","scripts_sha256":"${SCRIPTS_SHA256}","smoke_manifest_sha256":"%s","vllm_commit":"${EXPECTED_VLLM_COMMIT}"}\\n' "\$(shasum -a 256 ${RUN_ROOT}/run_manifest.json | awk '{print \$1}')" > ${SMOKE_MARKER}; fi
${POST_RUN}
EOF
)

SBATCH_ARGS=(--nodes=4 --exclusive --constraint=GB200 --account="${ACCOUNT}" --partition="${PARTITION}" --segment=4 --time=05:00:00 --job-name="mx-moe-${ARM}-${MAX_STEPS}s-${RUN_ID}" --output="${RUN_ROOT}/slurm-%j.out")
[[ -z "${QOS}" ]] || SBATCH_ARGS+=(--qos="${QOS}")
printf 'action=%s\narm=%s\nrun_root=%s\ncache_root=%s\n' "${ACTION}" "${ARM}" "${RUN_ROOT}" "${CACHE_ROOT}"
printf 'sbatch_args='; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n%s\n' "${COMMAND}"
case "${ACTION}" in
  dry-run) ;;
  test-only) CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 BASE_LOG_DIR="${RUN_ROOT}" sbatch --test-only "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub" ;;
  submit)
    audit_write_manifest "${RUN_ROOT}" validation "${REPO_DIR}" "${CUSTOM_VLLM_ROOT}" "${EXPECTED_VLLM_COMMIT}" "${CONTAINER}" "${CONFIG}" "${MODEL_SNAPSHOT}" "${CACHE_FILE}" "${VALIDATION_EXECUTION_INPUTS[0]}" "${VALIDATION_EXECUTION_INPUTS[@]:1}"
    CONTAINER=${CONTAINER} MOUNTS=/lustre:/lustre COMMAND="${COMMAND}" GPUS_PER_NODE=4 BASE_LOG_DIR="${RUN_ROOT}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub" ;;
esac
