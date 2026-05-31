#!/bin/bash
#SBATCH --job-name=q235b-vllm-runtime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3}"
REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/reports}"
CONTAINER="${CONTAINER:?set CONTAINER}"
MOUNTS="${MOUNTS:-/lustre:/lustre,$REPO_ROOT:$REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
VLLM_SITE="${VLLM_SITE-$ARTIFACT_ROOT/python_site/vllm_0_10_2_cu129_torch28nv_source_py312}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
HF_HOME_PATH="${HF_HOME_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_runtime_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_runtime_probe.md}"
PROBE_PAYLOAD="${PROBE_PAYLOAD:-$REPORT_DIR/vllm_runtime_probe_payload.py}"
STRICT_PIP_CHECK="${STRICT_PIP_CHECK:-true}"

mkdir -p logs "$REPORT_DIR"

cat > "$PROBE_PAYLOAD" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

json_out = Path(os.environ["JSON_OUT"])
markdown_out = Path(os.environ["MARKDOWN_OUT"])
model_name = os.environ["MODEL_NAME"]
site = os.environ["VLLM_SITE"]
strict_pip_check = os.environ.get("STRICT_PIP_CHECK", "true").lower() in {"1", "true", "yes"}

checks = []


def record(name, fn):
    try:
        value = fn()
        checks.append({"name": name, "status": "pass", "detail": value})
    except Exception as exc:
        checks.append({
            "name": name,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })


record("torch", lambda: __import__("torch").__version__)
record("vllm", lambda: __import__("vllm").__version__)
record("vllm._C", lambda: __import__("vllm._C")._C.__file__)
record(
    "CompilationConfig",
    lambda: getattr(__import__("vllm.config", fromlist=["CompilationConfig"]), "CompilationConfig").__name__,
)
record(
    "AsyncLLM",
    lambda: getattr(__import__("vllm.v1.engine.async_llm", fromlist=["AsyncLLM"]), "AsyncLLM").__name__,
)
record(
    "Qwen3MoeForCausalLM",
    lambda: getattr(
        __import__("vllm.model_executor.models.qwen3_moe", fromlist=["Qwen3MoeForCausalLM"]),
        "Qwen3MoeForCausalLM",
    ).__name__,
)


def engine_config_probe():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

    args = AsyncEngineArgs(
        model=model_name,
        tokenizer=model_name,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        gpu_memory_utilization=0.1,
        enforce_eager=True,
        disable_log_stats=True,
        skip_tokenizer_init=False,
    )
    cfg = args.create_engine_config(UsageContext.LLM_CLASS)
    return {
        "model": cfg.model_config.model,
        "runner_type": type(cfg).__name__,
        "model_config": type(cfg.model_config).__name__,
    }


record("AsyncEngineArgs.create_engine_config", engine_config_probe)


def pip_check_probe():
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-2000:])
    return proc.stdout.strip() or "No broken requirements found."


if strict_pip_check:
    record("pip check", pip_check_probe)
else:
    record("pip check skipped", lambda: "STRICT_PIP_CHECK=false")

payload = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
    "site": site,
    "model_name": model_name,
    "python": sys.executable,
    "checks": checks,
}
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# vLLM Runtime Probe",
    "",
    f"Overall: **{payload['overall_status'].upper()}**",
    "",
    f"Site: `{site}`",
    f"Model: `{model_name}`",
    "",
    "| check | status | detail |",
    "| --- | --- | --- |",
]
for item in checks:
    detail = item.get("detail")
    if detail is None:
        detail = f"{item.get('error_type')}: {item.get('error')}"
    lines.append(f"| `{item['name']}` | {item['status']} | `{str(detail)[:260]}` |")
markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(markdown_out.read_text(encoding="utf-8"))
raise SystemExit(0 if payload["overall_status"] == "pass" else 1)
PY

SRUN_CONTAINER_ARGS=(
  --container-image="$CONTAINER"
  --container-mounts="$MOUNTS"
  --container-workdir="$REPO_ROOT"
  --no-container-mount-home
)

if sinfo -p "$SLURM_JOB_PARTITION" -h -o "%G" | grep -q "gpu:"; then
  SRUN_CONTAINER_ARGS+=(--gres="gpu:$GPUS_PER_NODE")
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 env \
  PYTHONPATH="${VLLM_SITE:+$VLLM_SITE:}${PYTHONPATH:-}" \
  VLLM_SITE="$VLLM_SITE" \
  STRICT_PIP_CHECK="$STRICT_PIP_CHECK" \
  MODEL_NAME="$MODEL_NAME" \
  JSON_OUT="$JSON_OUT" \
  MARKDOWN_OUT="$MARKDOWN_OUT" \
  HF_HOME="$HF_HOME_PATH" \
  HF_DATASETS_CACHE="$HF_HOME_PATH/cache" \
  "$PYTHON_BIN" "$PROBE_PAYLOAD"
