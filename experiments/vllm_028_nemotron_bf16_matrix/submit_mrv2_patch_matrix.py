#!/usr/bin/env python3
"""Render patched-MRV2 baseline and DynamicMTP benchmark cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PATCHED_IMAGE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
)
RAY_BUNDLE = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-ray-2.48.0-aarch64.tar.gz"
)
RESULT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/"
    "vllm028-nemotron-bf16-mrv2-patched-matrix"
)
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 128, 512)
SHAPES = (
    ("isl1k_osl10k", 1000, 10000),
    ("isl10k_osl1k", 10000, 1000),
)
METHODS = ("baseline", "mtp_dynamic_max_k5")
DYNAMIC_SCHEDULE = "1:4:5,5:16:3,17:64:2,65:128:1,129:512:0"
DYNAMIC_TABLE = [[1, 4, 5], [5, 16, 3], [17, 64, 2], [65, 128, 1], [129, 512, 0]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported patch manifest schema")
    if payload.get("vllm_version") != "0.28.0":
        raise ValueError("patch manifest must target vLLM 0.28.0")
    base_commit = payload.get("base_commit")
    patches = payload.get("patches")
    if not isinstance(base_commit, str) or len(base_commit) != 40:
        raise ValueError("patch manifest must pin a full base commit")
    if not isinstance(patches, list) or not patches:
        raise ValueError("patch manifest must contain a nonempty patch stack")
    for patch in patches:
        if not isinstance(patch, dict) or not all(
            isinstance(patch.get(field), str) and patch[field]
            for field in ("file", "commit", "sha256", "source")
        ):
            raise ValueError("patch manifest contains incomplete provenance")
    return payload


def _topology(model_key: str) -> dict[str, Any]:
    if model_key == "super":
        return {"nodes": 1, "tp": 2, "expert": False, "ray": False}
    if model_key == "ultra":
        return {"nodes": 2, "tp": 8, "expert": True, "ray": True}
    raise ValueError(f"unsupported model_key={model_key!r}")


def _speculative_config(method_key: str) -> dict[str, Any] | None:
    if method_key == "baseline":
        return None
    if method_key == "mtp_dynamic_max_k5":
        return {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": DYNAMIC_TABLE,
        }
    raise ValueError(f"unsupported method_key={method_key!r}")


def build_matrix_rows(*, manifest_path: Path) -> list[dict[str, Any]]:
    """Build the complete 64-cell patched MRV2 comparison matrix."""
    manifest = _load_manifest(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    rows: list[dict[str, Any]] = []
    for model_key in ("super", "ultra"):
        for shape_key, isl, osl in SHAPES:
            for method_key in METHODS:
                for batch_size in BATCH_SIZES:
                    rows.append(
                        {
                            "model_key": model_key,
                            "shape_key": shape_key,
                            "isl": isl,
                            "osl": osl,
                            "method_key": method_key,
                            "batch_size": batch_size,
                            "runner_key": "mrv2",
                            "cudagraph_mode": "FULL_AND_PIECEWISE",
                            "dtype": "bfloat16",
                            "kv_cache_dtype": "fp8",
                            "speculative_config": _speculative_config(method_key),
                            "patch_id": manifest_sha256[:12],
                            "patchset_manifest_sha256": manifest_sha256,
                            "vllm_base_commit": manifest["base_commit"],
                        }
                    )
    return rows


def select_matrix_rows(
    rows: list[dict[str, Any]], *, gate: str
) -> list[dict[str, Any]]:
    """Select the 16-cell edge gate, remaining 48 cells, or all 64 cells."""
    if gate == "edge":
        return [row for row in rows if row["batch_size"] in {1, 512}]
    if gate == "remaining":
        return [row for row in rows if row["batch_size"] not in {1, 512}]
    if gate == "full":
        return list(rows)
    raise ValueError(f"unsupported gate={gate!r}")


def result_dir_for_row(row: dict[str, Any]) -> Path:
    """Return the collision-free durable result directory for one cell."""
    return (
        RESULT_ROOT
        / str(row["patch_id"])
        / str(row["model_key"])
        / str(row["shape_key"])
        / str(row["method_key"])
        / f"bs{row['batch_size']}"
    )


def render_cell_sbatch(
    row: dict[str, Any],
    *,
    experiment_dir: Path,
    result_dir: Path,
    manifest: dict[str, Any],
) -> str:
    """Render exactly one benchmark cell without invoking the scheduler."""
    manifest_path = experiment_dir / "mrv2_patch_manifest.json"
    committed_manifest = _load_manifest(manifest_path)
    if manifest != committed_manifest:
        raise ValueError("provided manifest does not match the committed manifest file")
    manifest_sha256 = _sha256(manifest_path)
    if row.get("patch_id") != manifest_sha256[:12] or (
        row.get("patchset_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("matrix row does not match the committed patch manifest")
    if row.get("vllm_base_commit") != manifest["base_commit"]:
        raise ValueError("matrix row base commit does not match the committed manifest")
    if row.get("runner_key") != "mrv2" or (
        row.get("cudagraph_mode") != "FULL_AND_PIECEWISE"
    ):
        raise ValueError("patched matrix requires MRV2 with FULL_AND_PIECEWISE")
    if row.get("dtype") != "bfloat16" or row.get("kv_cache_dtype") != "fp8":
        raise ValueError("patched matrix requires BF16 weights and FP8 KV cache")

    model_key = str(row["model_key"])
    topology = _topology(model_key)
    nodes = int(topology["nodes"])
    tp = int(topology["tp"])
    method_key = str(row["method_key"])
    config = _speculative_config(method_key)
    if row.get("speculative_config") != config:
        raise ValueError("matrix row speculative config does not match its method")
    spec_json = json.dumps(config, separators=(",", ":"))
    expert_arg = " --enable-expert-parallel" if topology["expert"] else ""
    distributed_arg = " --distributed-executor-backend ray" if topology["ray"] else ""
    ray_prefix = "/workspace/exp/run_multinode_ray.sh " if topology["ray"] else ""
    ray_declaration = (
        'readonly RAY_SITE_PACKAGES="${NODE_LOCAL_CACHE_ROOT}/ray-sidecar"\n'
        if topology["ray"]
        else ""
    )
    ray_mount = ",${RAY_SITE_PACKAGES}:/opt/ray-sidecar" if topology["ray"] else ""
    ray_args = (
        '--ray-version 2.48.0 --ray-bundle "${RAY_BUNDLE_PATH}" '
        '--ray-bundle-sha256 "${RAY_BUNDLE_SHA256}" '
        if topology["ray"]
        else ""
    )
    ray_import = (
        "import ray\nassert ray.__version__ == '2.48.0'\n" if topology["ray"] else ""
    )
    dynamic_schedule_comment = (
        f"# Dynamic schedule: {DYNAMIC_SCHEDULE}\n"
        if method_key == "mtp_dynamic_max_k5"
        else ""
    )
    benchmark_command = (
        f"{ray_prefix}python3 /workspace/exp/benchmark.py "
        f"--model-key {model_key} --runner-key mrv2 --method-key {method_key} "
        f"--tensor-parallel-size {tp}{expert_arg}{distributed_arg} "
        f"--isl {row['isl']} --osl {row['osl']} --batch-size {row['batch_size']} "
        "--max-num-seqs 512 --max-num-batched-tokens 32768 "
        "--no-enable-prefix-caching --enable-chunked-prefill --ignore-eos "
        f"--speculative-config-json {shlex.quote(spec_json)} "
        '--container-digest "sha256:${CONTAINER_ARTIFACT_SHA256}" '
        '--container-artifact "${CONTAINER_IMAGE}" '
        '--container-artifact-sha256 "${CONTAINER_ARTIFACT_SHA256}" '
        '--vllm-base-commit "${EXPECTED_BASE_COMMIT}" '
        '--patchset-manifest-sha256 "${PATCHSET_MANIFEST_SHA256}" '
        '--harness-commit "${HARNESS_COMMIT}" '
        '--harness-manifest-sha256 "${HARNESS_MANIFEST_SHA256}" '
        f"{ray_args}"
        '--output "${RESULT_RUN_DIR}/result.json"'
    )
    container_command = (
        "python3 - <<'PY'\n"
        "from importlib.metadata import version\n"
        "assert version('vllm') == '0.28.0'\n"
        f"{ray_import}"
        "PY\n"
        f"{benchmark_command}"
    )

    ray_setup = ""
    if topology["ray"]:
        ray_setup = f"""
readonly STABLE_RAY_BUNDLE={RAY_BUNDLE}
if [[ ! -r "${{STABLE_RAY_BUNDLE}}" ]]; then
  echo "Missing Ray bundle: ${{STABLE_RAY_BUNDLE}}" >&2
  exit 1
fi
RAY_BUNDLE_PATH=$(readlink -f "${{STABLE_RAY_BUNDLE}}")
readonly RAY_BUNDLE_PATH
readonly RAY_METADATA="${{RAY_BUNDLE_PATH}}.metadata.json"
RAY_BUNDLE_SHA256=$(python3 - "${{RAY_METADATA}}" "${{RAY_BUNDLE_PATH}}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = metadata["ray_bundle_sha256"]
actual = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if actual != expected or metadata.get("ray_version") != "2.48.0":
    raise SystemExit("Ray bundle provenance mismatch")
print(actual)
PY
)
readonly RAY_BUNDLE_SHA256
export RAY_BUNDLE_PATH RAY_BUNDLE_SHA256
srun \
  --nodes={nodes} \
  --ntasks={nodes} \
  --ntasks-per-node=1 \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{RAY_BUNDLE_PATH}}:/opt/ray-bundle.tar.gz,/raid/scratch:/raid/scratch" \
  --no-container-mount-home \
  --container-remap-root \
  bash -lc "mkdir -p '${{RAY_SITE_PACKAGES}}' && tar -xzf /opt/ray-bundle.tar.gz -C '${{RAY_SITE_PACKAGES}}' && PYTHONPATH='${{RAY_SITE_PACKAGES}}' python3 -c 'import ray; assert ray.__version__ == \\\"2.48.0\\\"'"
export PYTHONPATH=/opt/ray-sidecar
export HEAD_NODE=$(scontrol show hostnames "${{SLURM_JOB_NODELIST}}" | head -n 1)
export HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="${{HEAD_NODE}}" hostname --ip-address | awk '{{print $1}}')
export RAY_PORT=$((24000 + SLURM_JOB_ID % 1000))
export RAY_SYNC_DIR="${{RESULT_RUN_DIR}}/ray-sync"
"""

    legacy_dir = experiment_dir.parent / "vllm_024_dynamicsd"
    repo_root = experiment_dir.parents[1]
    return f"""#!/usr/bin/env bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb200
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --segment={nodes}
#SBATCH --time=05:00:00
#SBATCH --job-name=coreai_dlalgo_llm-v028.{model_key}-mrv2-{row["shape_key"]}-{method_key}-bs{row["batch_size"]}
#SBATCH --output={experiment_dir}/logs/slurm-%j.out

set -euo pipefail
# Runner: MRV2; CUDA graph: FULL_AND_PIECEWISE; precision: BF16 weights; FP8 KV cache
{dynamic_schedule_comment}export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DISABLE_USAGE_STATS=1
export NODE_LOCAL_CACHE_ROOT=/raid/scratch/${{USER}}/vllm028_mrv2_matrix_${{SLURM_JOB_ID}}
export XDG_CACHE_HOME=${{NODE_LOCAL_CACHE_ROOT}}/xdg
export VLLM_CACHE_ROOT=${{NODE_LOCAL_CACHE_ROOT}}/vllm
export TORCHINDUCTOR_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/torchinductor
export TRITON_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/triton
export CUDA_CACHE_PATH=${{NODE_LOCAL_CACHE_ROOT}}/cuda
readonly RESULT_CELL_DIR={shlex.quote(str(result_dir))}
readonly RESULT_RUN_DIR="${{RESULT_CELL_DIR}}/job-${{SLURM_JOB_ID}}"
if [[ -e "${{RESULT_RUN_DIR}}" ]]; then
  echo "Refusing to overwrite existing result run: ${{RESULT_RUN_DIR}}" >&2
  exit 1
fi
mkdir -p "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}" "${{TORCHINDUCTOR_CACHE_DIR}}" "${{TRITON_CACHE_DIR}}" "${{CUDA_CACHE_PATH}}" "${{RESULT_RUN_DIR}}"
readonly STABLE_CONTAINER_IMAGE={PATCHED_IMAGE}
readonly EXPECTED_BASE_COMMIT={manifest["base_commit"]}
readonly EXPECTED_PATCHSET_MANIFEST_SHA256={manifest_sha256}
readonly REPO_ROOT={shlex.quote(str(repo_root))}
readonly SOURCE_SNAPSHOT="${{RESULT_RUN_DIR}}/source"
mkdir -p "${{SOURCE_SNAPSHOT}}/exp" "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd"
install -m 0644 \
  {experiment_dir}/benchmark.py \
  {experiment_dir}/results.py \
  {experiment_dir}/mrv2_patch_manifest.json \
  {experiment_dir}/submit_mrv2_patch_matrix.py \
  "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0755 {experiment_dir}/run_multinode_ray.sh "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0644 {legacy_dir}/benchmark.py "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd/"
(
  cd "${{SOURCE_SNAPSHOT}}"
  sha256sum \
    exp/benchmark.py \
    exp/results.py \
    exp/mrv2_patch_manifest.json \
    exp/submit_mrv2_patch_matrix.py \
    exp/run_multinode_ray.sh \
    vllm_024_dynamicsd/benchmark.py > manifest.sha256
)
HARNESS_COMMIT=$(git -C "${{REPO_ROOT}}" rev-parse HEAD)
readonly HARNESS_COMMIT
HARNESS_MANIFEST_SHA256=$(sha256sum "${{SOURCE_SNAPSHOT}}/manifest.sha256" | awk '{{print $1}}')
readonly HARNESS_MANIFEST_SHA256
PATCHSET_MANIFEST_SHA256=$(sha256sum "${{SOURCE_SNAPSHOT}}/exp/mrv2_patch_manifest.json" | awk '{{print $1}}')
readonly PATCHSET_MANIFEST_SHA256
if [[ "${{PATCHSET_MANIFEST_SHA256}}" != "${{EXPECTED_PATCHSET_MANIFEST_SHA256}}" ]]; then
  echo "patchset manifest changed after rendering" >&2
  exit 1
fi
{ray_declaration}readonly CONTAINER_MOUNTS="${{SOURCE_SNAPSHOT}}/exp:/workspace/exp,${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd:/workspace/vllm_024_dynamicsd,/lustre:/lustre,/raid/scratch:/raid/scratch{ray_mount}"
export HARNESS_COMMIT HARNESS_MANIFEST_SHA256 PATCHSET_MANIFEST_SHA256

if [[ ! -r "${{STABLE_CONTAINER_IMAGE}}" ]]; then
  echo "Missing patched vLLM container: ${{STABLE_CONTAINER_IMAGE}}" >&2
  exit 1
fi
CONTAINER_IMAGE=$(readlink -f "${{STABLE_CONTAINER_IMAGE}}")
readonly CONTAINER_IMAGE
readonly CONTAINER_METADATA="${{CONTAINER_IMAGE}}.metadata.json"
CONTAINER_ARTIFACT_SHA256=$(python3 - "${{CONTAINER_METADATA}}" "${{CONTAINER_IMAGE}}" "${{EXPECTED_BASE_COMMIT}}" "${{EXPECTED_PATCHSET_MANIFEST_SHA256}}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metadata.get("vllm_base_commit") != sys.argv[3]:
    raise SystemExit("patched container base commit mismatch")
if metadata.get("patchset_manifest_sha256") != sys.argv[4]:
    raise SystemExit("patched container patchset mismatch")
expected = metadata.get("artifact_sha256")
actual = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if not isinstance(expected, str) or actual != expected:
    raise SystemExit("patched container artifact digest mismatch")
print(actual)
PY
)
readonly CONTAINER_ARTIFACT_SHA256
export CONTAINER_IMAGE CONTAINER_ARTIFACT_SHA256

read -r -d '' CONTAINER_COMMAND <<'CONTAINER_SCRIPT' || true
{container_command}
CONTAINER_SCRIPT
{ray_setup}
readonly BENCHMARK_LOG="${{RESULT_RUN_DIR}}/benchmark.log"
srun \
  --nodes={nodes} \
  --ntasks={nodes} \
  --ntasks-per-node=1 \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{CONTAINER_MOUNTS}}" \
  --no-container-mount-home \
  --container-remap-root \
  bash -lc "${{CONTAINER_COMMAND}}" 2>&1 | tee "${{BENCHMARK_LOG}}"
python3 - "${{BENCHMARK_LOG}}" "${{RESULT_RUN_DIR}}/cuda_graph_evidence.json" {shlex.quote(method_key)} <<'PY'
import json
import re
import sys
from pathlib import Path

log_text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
completed = []
for line in log_text.splitlines():
    if "Capturing" not in line or "CUDA graphs" not in line:
        continue
    match = re.search(r"([0-9]+)\\s*/\\s*([0-9]+)", line)
    if match and int(match.group(1)) == int(match.group(2)) and int(match.group(2)) > 0:
        completed.append({{"line": line, "count": int(match.group(2))}})
piecewise = [row for row in completed if "PIECEWISE" in row["line"]]
full = [
    row
    for row in completed
    if "FULL" in row["line"] and "PIECEWISE" not in row["line"]
]
decode = [
    row
    for row in completed
    if "decode CUDA graphs" in row["line"] and "prefill-decode" not in row["line"]
]
method = sys.argv[3]
required_prefill_captures = 1 if method == "baseline" else 2
if len(piecewise) < required_prefill_captures or len(full) < required_prefill_captures:
    raise SystemExit("incomplete target CUDA Graph capture evidence")
if method != "baseline" and not decode:
    raise SystemExit("incomplete drafter decode FULL CUDA Graph capture evidence")
evidence = {{
    "method_key": method,
    "piecewise_completed": piecewise,
    "full_completed": full,
    "drafter_decode_completed": decode,
}}
Path(sys.argv[2]).write_text(json.dumps(evidence, indent=2) + "\\n", encoding="utf-8")
PY
"""


def _script_name(row: dict[str, Any]) -> str:
    return (
        f"{row['model_key']}-{row['shape_key']}-{row['method_key']}-"
        f"bs{row['batch_size']}.sbatch"
    )


def render_matrix(*, gate: str, output_dir: Path) -> list[Path]:
    """Render a selected matrix to disk and return the created script paths."""
    manifest_path = PACKAGE_ROOT / "mrv2_patch_manifest.json"
    manifest = _load_manifest(manifest_path)
    rows = select_matrix_rows(
        build_matrix_rows(manifest_path=manifest_path),
        gate=gate,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(output_dir.glob("*.sbatch"))
    if stale:
        raise FileExistsError(
            f"refusing to render over {len(stale)} stale sbatch file(s) in {output_dir}"
        )
    rendered: list[Path] = []
    for row in rows:
        destination = output_dir / _script_name(row)
        destination.write_text(
            render_cell_sbatch(
                row,
                experiment_dir=PACKAGE_ROOT,
                result_dir=result_dir_for_row(row),
                manifest=manifest,
            ),
            encoding="utf-8",
        )
        rendered.append(destination)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("edge", "remaining", "full"), default="edge")
    parser.add_argument("--output-dir", type=Path)
    parsed = parser.parse_args()
    output_dir = (
        parsed.output_dir or PACKAGE_ROOT / "logs" / "rendered-patched-mrv2-matrix"
    )
    for path in render_matrix(gate=parsed.gate, output_dir=output_dir):
        print(path)


if __name__ == "__main__":
    main()
