#!/usr/bin/env python3
"""Build and execute validated vLLM 0.28 benchmark submission plans."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any


_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-f]{64}$")


def _method_config(method_key: str, schedule: list[dict[str, int]]) -> dict[str, Any] | None:
    if method_key == "baseline":
        return None
    if method_key.startswith("mtp_static_k"):
        return {
            "method": "mtp",
            "num_speculative_tokens": int(method_key.removeprefix("mtp_static_k")),
        }
    if method_key == "mtp_dynamic_max_k5":
        return {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [
                [row["start"], row["end"], row["k"]] for row in schedule
            ],
        }
    raise ValueError(f"unsupported method_key={method_key!r}")


def build_submission_plan(
    contract: dict[str, Any],
    *,
    data_parallel_size: int,
    container_image: str,
) -> list[dict[str, Any]]:
    """Expand the experiment contract after enforcing unsafe-setting gates."""
    if data_parallel_size != 1:
        raise ValueError("DynamicSD requires data_parallel_size=1")
    if not _DIGEST_PATTERN.search(container_image):
        raise ValueError("container image must be pinned by a sha256 digest")
    release = contract["runtime_release"]
    container_digest = container_image.split("@", 1)[1]
    runtime = {
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 512,
        "ignore_eos": True,
    }
    provenance = {
        "vllm_version": release["vllm_version"],
        "vllm_branch": release["vllm_branch"],
        "vllm_commit": release["vllm_commit"],
        "container_digest": container_digest,
    }
    plan: list[dict[str, Any]] = []
    for model in contract["models"]:
        topology = dict(model["runtime_topology"])
        if topology["data_parallel_size"] != data_parallel_size:
            raise ValueError("model topology data_parallel_size does not match plan")
        for shape in contract["shapes"]:
            for method_key in contract["method_order"]:
                for runner_gate_key, cudagraph_mode in contract["cuda_graph_modes"].items():
                    runner_key = runner_gate_key.split("_", 1)[0]
                    for batch_size in contract["batch_sizes"]:
                        plan.append(
                            {
                                "model": dict(model),
                                "model_key": model["key"],
                                "shape": dict(shape),
                                "batch_size": batch_size,
                                "method_key": method_key,
                                "speculative_config": _method_config(
                                    method_key, contract["dynamic_schedule"]
                                ),
                                "runner_key": runner_key,
                                "runner_gate_key": runner_gate_key,
                                "cudagraph_mode": cudagraph_mode,
                                "runtime": dict(runtime),
                                "runtime_provenance": dict(provenance),
                                "container_image": container_image,
                            }
                        )
    return plan


def execute_submission_plan(
    plan: list[dict[str, Any]],
    *,
    dry_run: bool,
    sbatch_runner: Callable[..., Any],
) -> list[str]:
    """Render plan rows and submit only when explicitly not in dry-run mode."""
    rendered = [json.dumps(row, sort_keys=True) for row in plan]
    if dry_run:
        return rendered
    for row in rendered:
        sbatch_runner(row)
    return rendered


def select_smoke_plan_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select runner gates at low, medium, and threshold-adjacent concurrency."""
    methods = {"baseline", "mtp_static_k5", "mtp_dynamic_max_k5"}
    batch_sizes = {1, 32, 128}
    return [
        row
        for row in plan
        if row.get("method_key") in methods and row.get("batch_size") in batch_sizes
    ]


def render_sbatch(
    plan_row: dict[str, Any],
    *,
    experiment_dir: Path,
    result_dir: Path,
) -> str:
    """Render one auditable SLURM script without invoking the scheduler."""
    model_key = str(plan_row["model_key"])
    topology = {
        "super": {"nodes": 1, "tp": 2, "expert": False},
        "ultra": {"nodes": 2, "tp": 8, "expert": True},
    }.get(model_key)
    if topology is None:
        raise ValueError(f"unsupported model_key={model_key!r}")
    runner_key = str(plan_row["runner_key"])
    if runner_key not in {"mrv1", "mrv2"}:
        raise ValueError(f"unsupported runner_key={runner_key!r}")
    runtime = plan_row["runtime"]
    provenance = plan_row["runtime_provenance"]
    shape = plan_row["shape"]
    speculative_config = plan_row.get("speculative_config")
    spec_json = json.dumps(speculative_config, separators=(",", ":"))
    expert_arg = " --enable-expert-parallel" if topology["expert"] else ""
    distributed_arg = (
        " --distributed-executor-backend ray" if model_key == "ultra" else ""
    )
    prefix_arg = (
        "--enable-prefix-caching"
        if runtime["enable_prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    chunked_arg = (
        "--enable-chunked-prefill"
        if runtime["enable_chunked_prefill"]
        else "--no-enable-chunked-prefill"
    )
    nodes = int(topology["nodes"])
    tp = int(topology["tp"])
    runner_flag = 1 if runner_key == "mrv2" else 0
    runner_gate_key = str(plan_row.get("runner_gate_key", runner_key))
    ray_runtime_args = ""
    ray_version_check = ""
    ray_declarations = ""
    ray_mount = ""
    if model_key == "ultra":
        ray_runtime_args = (
            '--ray-version 2.48.0 --ray-bundle "${RAY_BUNDLE}" '
            '--ray-bundle-sha256 "${RAY_BUNDLE_SHA256}" '
        )
        ray_version_check = "import ray\nassert ray.__version__ == '2.48.0'\n"
        ray_declarations = (
            'readonly RAY_SITE_PACKAGES="${NODE_LOCAL_CACHE_ROOT}/ray-sidecar"\n'
        )
        ray_mount = ',${RAY_SITE_PACKAGES}:/opt/ray-sidecar'
    stable_container_image = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
        "vllm-openai-v0.28.0-aarch64-ubuntu2404.sqsh"
    )
    legacy_dir = experiment_dir.parent / "vllm_024_dynamicsd"
    repo_root = experiment_dir.parents[1]
    benchmark_command = (
        "python3 /workspace/exp/benchmark.py "
        f"--model-key {model_key} "
        f"--runner-key {runner_key} "
        f"--method-key {plan_row['method_key']} "
        f"--tensor-parallel-size {tp}{expert_arg}{distributed_arg} "
        f"--isl {shape['isl']} "
        f"--osl {shape['osl']} "
        f"--batch-size {plan_row['batch_size']} "
        f"--max-num-seqs {runtime['max_num_seqs']} "
        f"--max-num-batched-tokens {runtime['max_num_batched_tokens']} "
        f"{prefix_arg} {chunked_arg} --ignore-eos "
        f"--speculative-config-json {shlex.quote(spec_json)} "
        f"--container-digest {shlex.quote(provenance['container_digest'])} "
        '--container-artifact "${CONTAINER_IMAGE}" '
        '--container-artifact-sha256 "${CONTAINER_ARTIFACT_SHA256}" '
        '--harness-commit "${HARNESS_COMMIT}" '
        '--harness-manifest-sha256 "${HARNESS_MANIFEST_SHA256}" '
        f"{ray_runtime_args}"
        f"--output {result_dir}/result.json"
    )
    if model_key == "ultra":
        benchmark_command = f"/workspace/exp/run_multinode_ray.sh {benchmark_command}"
    container_command = (
        "python3 - <<'PY'\n"
        "from importlib.metadata import version\n"
        f"assert version('vllm') == '{provenance['vllm_version']}'\n"
        f"{ray_version_check}"
        "PY\n"
        f"{benchmark_command}"
    )
    ray_setup = ""
    if model_key == "ultra":
        ray_setup = f"""
readonly STABLE_RAY_BUNDLE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/vllm-ray-2.48.0-aarch64.tar.gz
if [[ ! -r "${{STABLE_RAY_BUNDLE}}" ]]; then
  echo "Missing staged Ray 2.48.0 bundle: ${{STABLE_RAY_BUNDLE}}" >&2
  exit 1
fi
RAY_BUNDLE=$(readlink -f "${{STABLE_RAY_BUNDLE}}")
readonly RAY_BUNDLE
readonly RAY_METADATA="${{RAY_BUNDLE}}.metadata.json"
RAY_BUNDLE_SHA256=$(python3 - "${{RAY_METADATA}}" "${{RAY_BUNDLE}}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metadata.get("ray_version") != "2.48.0":
    raise SystemExit("staged Ray version does not match 2.48.0")
expected = metadata["ray_bundle_sha256"]
digest = hashlib.sha256()
with Path(sys.argv[2]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit("staged Ray bundle SHA256 does not match metadata")
print(actual)
PY
)
readonly RAY_BUNDLE_SHA256
export RAY_BUNDLE RAY_BUNDLE_SHA256
srun \
  --nodes={nodes} \
  --ntasks={nodes} \
  --ntasks-per-node=1 \
  --container-image="${{CONTAINER_IMAGE}}" \
  --container-mounts="${{RAY_BUNDLE}}:/opt/ray-bundle.tar.gz,/raid/scratch:/raid/scratch" \
  --no-container-mount-home \
  --container-remap-root \
  bash -lc "mkdir -p '${{RAY_SITE_PACKAGES}}' && tar -xzf /opt/ray-bundle.tar.gz -C '${{RAY_SITE_PACKAGES}}' && PYTHONPATH='${{RAY_SITE_PACKAGES}}' python3 -c 'import ray; assert ray.__version__ == \"2.48.0\"'"
export PYTHONPATH=/opt/ray-sidecar
export HEAD_NODE=$(scontrol show hostnames "${{SLURM_JOB_NODELIST}}" | head -n 1)
export HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="${{HEAD_NODE}}" hostname --ip-address | awk '{{print $1}}')
export RAY_PORT=$((24000 + SLURM_JOB_ID % 1000))
export RAY_SYNC_DIR="{result_dir}/ray-sync-${{SLURM_JOB_ID}}"
"""
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
#SBATCH --job-name=v028-{model_key}-{runner_key}-{plan_row['method_key']}-bs{plan_row['batch_size']}
#SBATCH --output={experiment_dir}/logs/slurm-%j.out

set -euo pipefail
# Runner gate: {runner_gate_key}
export VLLM_USE_V2_MODEL_RUNNER={runner_flag}
export VLLM_DISABLE_USAGE_STATS=1
export NODE_LOCAL_CACHE_ROOT=/raid/scratch/${{USER}}/vllm028_${{SLURM_JOB_ID}}
export XDG_CACHE_HOME=${{NODE_LOCAL_CACHE_ROOT}}/xdg
export VLLM_CACHE_ROOT=${{NODE_LOCAL_CACHE_ROOT}}/vllm
export TORCHINDUCTOR_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/torchinductor
export TRITON_CACHE_DIR=${{NODE_LOCAL_CACHE_ROOT}}/triton
export CUDA_CACHE_PATH=${{NODE_LOCAL_CACHE_ROOT}}/cuda
mkdir -p "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}" "${{TORCHINDUCTOR_CACHE_DIR}}" "${{TRITON_CACHE_DIR}}" "${{CUDA_CACHE_PATH}}" "{result_dir}"
readonly STABLE_CONTAINER_IMAGE={stable_container_image}
readonly EXPECTED_SOURCE_DIGEST={shlex.quote(provenance['container_digest'])}
readonly REPO_ROOT={shlex.quote(str(repo_root))}
readonly SOURCE_SNAPSHOT="{result_dir}/source-${{SLURM_JOB_ID}}"
mkdir -p "${{SOURCE_SNAPSHOT}}/exp" "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd"
install -m 0644 \
  {experiment_dir}/benchmark.py \
  {experiment_dir}/contract.py \
  {experiment_dir}/launcher.py \
  {experiment_dir}/results.py \
  "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0755 {experiment_dir}/run_multinode_ray.sh "${{SOURCE_SNAPSHOT}}/exp/"
install -m 0644 {legacy_dir}/benchmark.py "${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd/"
(
  cd "${{SOURCE_SNAPSHOT}}"
  sha256sum \
    exp/benchmark.py \
    exp/contract.py \
    exp/launcher.py \
    exp/results.py \
    exp/run_multinode_ray.sh \
    vllm_024_dynamicsd/benchmark.py > manifest.sha256
)
HARNESS_COMMIT=$(git -C "${{REPO_ROOT}}" rev-parse HEAD)
readonly HARNESS_COMMIT
HARNESS_MANIFEST_SHA256=$(sha256sum "${{SOURCE_SNAPSHOT}}/manifest.sha256" | awk '{{print $1}}')
readonly HARNESS_MANIFEST_SHA256
{ray_declarations}readonly CONTAINER_MOUNTS="${{SOURCE_SNAPSHOT}}/exp:/workspace/exp,${{SOURCE_SNAPSHOT}}/vllm_024_dynamicsd:/workspace/vllm_024_dynamicsd,/lustre:/lustre,/raid/scratch:/raid/scratch{ray_mount}"
export HARNESS_COMMIT HARNESS_MANIFEST_SHA256

if [[ ! -r "${{STABLE_CONTAINER_IMAGE}}" ]]; then
  echo "Missing staged vLLM container: ${{STABLE_CONTAINER_IMAGE}}" >&2
  exit 1
fi
CONTAINER_IMAGE=$(readlink -f "${{STABLE_CONTAINER_IMAGE}}")
readonly CONTAINER_IMAGE
readonly CONTAINER_METADATA="${{CONTAINER_IMAGE}}.metadata.json"
CONTAINER_ARTIFACT_SHA256=$(python3 - "${{CONTAINER_METADATA}}" "${{EXPECTED_SOURCE_DIGEST}}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metadata.get("source_digest") != sys.argv[2]:
    raise SystemExit("staged container source digest does not match the benchmark contract")
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
