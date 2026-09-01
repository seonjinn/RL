#!/usr/bin/env python3
"""Render auditable Super and Ultra patched-MRV2 Dynamic-K canaries."""

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
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "vllm-ray-2.48.0-aarch64.tar.gz"
)
RESULT_ROOT = Path(
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/"
    "vllm028-nemotron-bf16-mrv2-patched-canary"
)
CANARY_BATCH_SIZES = (1, 2, 4, 8, 16)
CANARY_DYNAMIC_SCHEDULE = "1:1:5,2:2:3,3:4:2,5:8:1,9:512:0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology(model_key: str) -> dict[str, Any]:
    if model_key == "super":
        return {"nodes": 1, "tp": 2, "expert": False, "ray": False}
    if model_key == "ultra":
        return {"nodes": 2, "tp": 8, "expert": True, "ray": True}
    raise ValueError(f"unsupported model_key={model_key!r}")


def render_canary_sbatch(
    *,
    model_key: str,
    experiment_dir: Path,
    result_dir: Path,
    manifest: dict[str, Any],
) -> str:
    topology = _topology(model_key)
    nodes = int(topology["nodes"])
    tp = int(topology["tp"])
    manifest_path = experiment_dir / "mrv2_patch_manifest.json"
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("provided manifest does not match the committed manifest file")
    manifest_sha256 = _sha256(manifest_path)
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
    ray_import = "import ray\nassert ray.__version__ == '2.48.0'\n" if topology["ray"] else ""
    benchmark_command = (
        f"{ray_prefix}python3 /workspace/exp/benchmark_mrv2_patch_canary.py "
        f"--model-key {model_key} --tensor-parallel-size {tp}"
        f"{expert_arg}{distributed_arg} "
        "--isl 1000 --osl 128 "
        "--batch-sizes 1 2 4 8 16 "
        f"--dynamic-schedule {CANARY_DYNAMIC_SCHEDULE} "
        '--patchset-manifest-sha256 "${PATCHSET_MANIFEST_SHA256}" '
        '--container-artifact "${CONTAINER_IMAGE}" '
        '--container-artifact-sha256 "${CONTAINER_ARTIFACT_SHA256}" '
        '--harness-commit "${HARNESS_COMMIT}" '
        '--harness-manifest-sha256 "${HARNESS_MANIFEST_SHA256}" '
        f"{ray_args}"
        f"--output {shlex.quote(str(result_dir / 'result.json'))}"
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
export RAY_SYNC_DIR="{result_dir}/ray-sync-${{SLURM_JOB_ID}}"
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
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_dlalgo_llm-v028.{model_key}-mrv2-dynamick-core-canary
#SBATCH --output={experiment_dir}/logs/slurm-%j.out

set -euo pipefail
# Runner gate: MRV2; CUDA graph gate: FULL_AND_PIECEWISE
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DISABLE_USAGE_STATS=1
export NODE_LOCAL_CACHE_ROOT=/raid/scratch/${{USER}}/vllm028_mrv2_canary_${{SLURM_JOB_ID}}
export XDG_CACHE_HOME=${{NODE_LOCAL_CACHE_ROOT}}/xdg
export VLLM_CACHE_ROOT=${{NODE_LOCAL_CACHE_ROOT}}/vllm
export TORCHINDUCTOR_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/torchinductor
export TRITON_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/triton
export CUDA_CACHE_PATH=${{NODE_LOCAL_CACHE_ROOT}}/cuda
mkdir -p "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}" "${{TORCHINDUCTOR_CACHE_DIR}}" "${{TRITON_CACHE_DIR}}" "${{CUDA_CACHE_PATH}}" {shlex.quote(str(result_dir))}
readonly STABLE_CONTAINER_IMAGE={PATCHED_IMAGE}
readonly EXPECTED_BASE_COMMIT={manifest['base_commit']}
readonly EXPECTED_PATCHSET_MANIFEST_SHA256={manifest_sha256}
readonly REPO_ROOT={shlex.quote(str(repo_root))}
readonly SOURCE_SNAPSHOT="{result_dir}/source-${{SLURM_JOB_ID}}"
mkdir -p "${{SOURCE_SNAPSHOT}}/exp" "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd"
install -m 0644 \
  {experiment_dir}/benchmark.py \
  {experiment_dir}/benchmark_mrv2_patch_canary.py \
  {experiment_dir}/mrv2_patch_manifest.json \
  {experiment_dir}/mrv2_patchset.py \
  "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0755 {experiment_dir}/run_multinode_ray.sh "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0644 {legacy_dir}/benchmark.py "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd/"
(
  cd "${{SOURCE_SNAPSHOT}}"
  sha256sum \
    exp/benchmark.py \
    exp/benchmark_mrv2_patch_canary.py \
    exp/mrv2_patch_manifest.json \
    exp/mrv2_patchset.py \
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
  echo "Patchset manifest changed after rendering" >&2
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
CONTAINER_ARTIFACT_SHA256=$(python3 - "${{CONTAINER_METADATA}}" "${{EXPECTED_BASE_COMMIT}}" "${{EXPECTED_PATCHSET_MANIFEST_SHA256}}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metadata.get("vllm_base_commit") != sys.argv[2]:
    raise SystemExit("patched container base commit mismatch")
if metadata.get("patchset_manifest_sha256") != sys.argv[3]:
    raise SystemExit("patched container patchset mismatch")
print(metadata["artifact_sha256"])
PY
)
readonly CONTAINER_ARTIFACT_SHA256
export CONTAINER_IMAGE CONTAINER_ARTIFACT_SHA256

read -r -d '' CONTAINER_COMMAND <<'CONTAINER_SCRIPT' || true
{container_command}
CONTAINER_SCRIPT
{ray_setup}
srun \
  --nodes={nodes} \
  --ntasks={nodes} \
  --ntasks-per-node=1 \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{CONTAINER_MOUNTS}}" \
  --no-container-mount-home \
  --container-remap-root \
  bash -lc "${{CONTAINER_COMMAND}}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("super", "ultra", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parsed = parser.parse_args()
    manifest_path = PACKAGE_ROOT / "mrv2_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = ("super", "ultra") if parsed.model == "all" else (parsed.model,)
    output_dir = parsed.output_dir or PACKAGE_ROOT / "logs" / "rendered-patched-mrv2"
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_id = _sha256(manifest_path)[:12]
    for model_key in models:
        result_dir = RESULT_ROOT / patch_id / model_key / "isl1000_osl128"
        script = render_canary_sbatch(
            model_key=model_key,
            experiment_dir=PACKAGE_ROOT,
            result_dir=result_dir,
            manifest=manifest,
        )
        destination = output_dir / f"{model_key}-mrv2-dynamick-core-canary.sbatch"
        destination.write_text(script, encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
