#!/bin/bash
#SBATCH --job-name=q235b-vllm-abi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:20:00
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
VLLM_SITE_CANDIDATES="${VLLM_SITE_CANDIDATES:-$ARTIFACT_ROOT/python_site/vllm_0_10_2_nodeps_py312 $ARTIFACT_ROOT/python_site/vllm_0_11_2_nodeps_py312 $ARTIFACT_ROOT/python_site/vllm_0_13_0_nodeps_py312}"
JSON_OUT="${JSON_OUT:-$REPORT_DIR/vllm_native_abi_probe.json}"
MARKDOWN_OUT="${MARKDOWN_OUT:-$REPORT_DIR/vllm_native_abi_probe.md}"
PROBE_PAYLOAD="${PROBE_PAYLOAD:-$REPORT_DIR/vllm_native_abi_probe_payload.py}"

mkdir -p logs "$REPORT_DIR"

SRUN_CONTAINER_ARGS=(
  --container-image="$CONTAINER"
  --container-mounts="$MOUNTS"
  --container-workdir="$REPO_ROOT"
  --no-container-mount-home
)

cat > "$PROBE_PAYLOAD" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sites = [part for part in os.environ["VLLM_SITE_CANDIDATES"].split() if part]
json_out = Path(os.environ["JSON_OUT"])
markdown_out = Path(os.environ["MARKDOWN_OUT"])
probe = r"""
import json
import sys
out = {"python": sys.executable}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_cuda"] = getattr(torch.version, "cuda", None)
    out["torch_file"] = torch.__file__
except Exception as exc:
    out["torch_error"] = repr(exc)
try:
    import vllm
    out["vllm_version"] = getattr(vllm, "__version__", None)
    out["vllm_file"] = vllm.__file__
except Exception as exc:
    out["vllm_import_error"] = f"{type(exc).__name__}: {exc}"
try:
    import vllm._C
    out["vllm_c_file"] = vllm._C.__file__
    out["vllm_c_ok"] = True
except Exception as exc:
    out["vllm_c_ok"] = False
    out["vllm_c_error"] = f"{type(exc).__name__}: {exc}"
try:
    from vllm.config import CompilationConfig
    out["compilation_config_ok"] = True
except Exception as exc:
    out["compilation_config_ok"] = False
    out["compilation_config_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out, sort_keys=True))
sys.exit(0 if out.get("vllm_c_ok") and out.get("compilation_config_ok") else 1)
"""

results = []
for site in sites:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{site}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    parsed = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    results.append(
        {
            "site": site,
            "exists": Path(site).exists(),
            "returncode": proc.returncode,
            "parsed": parsed,
            "output_tail": proc.stdout[-4000:],
        }
    )

payload = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "overall_status": "pass" if any(r["returncode"] == 0 for r in results) else "fail",
    "results": results,
}
json_out.parent.mkdir(parents=True, exist_ok=True)
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# vLLM Native ABI Probe",
    "",
    f"Overall: **{payload['overall_status'].upper()}**",
    "",
    "| status | site | vLLM | torch | error |",
    "| --- | --- | --- | --- | --- |",
]
for item in results:
    parsed = item.get("parsed") or {}
    status = "pass" if item["returncode"] == 0 else "fail"
    error = parsed.get("vllm_c_error") or parsed.get("compilation_config_error") or parsed.get("vllm_import_error") or ""
    lines.append(
        f"| {status} | `{item['site']}` | `{parsed.get('vllm_version')}` | "
        f"`{parsed.get('torch_version')}` | `{str(error)[:220]}` |"
    )
markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(markdown_out.read_text(encoding="utf-8"))
raise SystemExit(0 if payload["overall_status"] == "pass" else 1)
PY

if sinfo -p "$SLURM_JOB_PARTITION" -h -o "%G" | grep -q "gpu:"; then
  SRUN_CONTAINER_ARGS+=(--gres="gpu:$GPUS_PER_NODE")
fi

srun "${SRUN_CONTAINER_ARGS[@]}" --ntasks=1 env \
  VLLM_SITE_CANDIDATES="$VLLM_SITE_CANDIDATES" \
  JSON_OUT="$JSON_OUT" \
  MARKDOWN_OUT="$MARKDOWN_OUT" \
  "$PYTHON_BIN" "$PROBE_PAYLOAD"
