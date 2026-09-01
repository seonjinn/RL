#!/usr/bin/env python3
"""Pure renderer and explicit dispatcher for Q30 Lyris jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .contract import (
    ExperimentContract,
    MethodPlan,
    build_barrier_rows,
    build_calibration_rows,
)
from .live_runner import build_speculative_config


PATCHSET_SHA256 = "238e2ffcc14d2fb2f0fc07c419004820efa9a3cfd284c3a41515e84e39aecc25"
CONTAINER_SHA256 = "5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066"
UNSUPPORTED_REASON = (
    "exact selected-K and physical-width profiler integration unavailable"
)
DispatchMode = Literal["render", "test-only", "submit"]


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    cluster: str
    ssh_host: str
    user: str
    remote_cwd: str
    partition: str
    account: str
    container_image: str
    container_verification_receipt: str
    result_root: str
    prompt_jsonl: str
    gpus_per_node: int

    def __post_init__(self) -> None:
        if self.cluster != "lyris" or self.partition != "gb200":
            raise ValueError("Q30 renderer is pinned to the Lyris gb200 partition")
        if self.account != "coreai_dlalgo_llm":
            raise ValueError("Q30 renderer account drift")
        if not self.remote_cwd.startswith("/home/"):
            raise ValueError("remote_cwd must be under /home")
        for value in (
            self.container_image,
            self.container_verification_receipt,
            self.result_root,
            self.prompt_jsonl,
        ):
            if not value.startswith("/lustre/"):
                raise ValueError("durable inputs and outputs must be under /lustre")
            if any(character.isspace() for character in value):
                raise ValueError("cluster paths must not contain whitespace")
        if self.gpus_per_node != 4:
            raise ValueError("Lyris Q30 nodes must expose four GPUs")


@dataclass(frozen=True, slots=True)
class JobSpec:
    key: str
    plan: MethodPlan
    nodes: int
    gpus_per_node: int
    worker_count: int
    result_subdir: str
    schedule: Sequence[Sequence[int]] | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9_]+", self.key) is None:
            raise ValueError(
                "key must contain only lowercase letters, digits, underscores"
            )
        if re.fullmatch(r"[a-z0-9_]+(?:/[a-z0-9_-]+)*", self.result_subdir) is None:
            raise ValueError("result_subdir must be a safe relative path")


def load_cluster_config(path: Path) -> ClusterConfig:
    """Load the deliberately scalar cluster configuration without PyYAML."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid scalar YAML at line {line_number}")
        values[key.strip()] = value.strip()
    required = {field for field in ClusterConfig.__dataclass_fields__}
    if set(values) != required:
        raise ValueError(f"cluster config keys must be exactly {sorted(required)}")
    return ClusterConfig(
        cluster=values["cluster"],
        ssh_host=values["ssh_host"],
        user=values["user"],
        remote_cwd=values["remote_cwd"],
        partition=values["partition"],
        account=values["account"],
        container_image=values["container_image"],
        container_verification_receipt=values["container_verification_receipt"],
        result_root=values["result_root"],
        prompt_jsonl=values["prompt_jsonl"],
        gpus_per_node=int(values["gpus_per_node"]),
    )


def render_adaptive_overlay(source: Path, overlay: Path) -> None:
    """Copy a DSpark checkpoint and enable adaptive verification only in the copy."""
    if overlay.exists():
        raise FileExistsError(f"refusing to overwrite adaptive overlay: {overlay}")
    shutil.copytree(source, overlay)
    config_path = overlay / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DSpark config.json must contain an object")
    payload["enable_confidence_head"] = True
    payload["confidence_head_with_markov"] = True
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _compact_json(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _shell_quote(value: str) -> str:
    """Return an always-visible POSIX single-quoted scalar."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _runtime_command(spec: JobSpec, drafter_path: str | None) -> str:
    contract = ExperimentContract()
    request_count = (
        spec.plan.batch_size
        if spec.plan.stage == "calibration"
        else contract.requests_per_engine
    )
    if request_count is None:
        raise ValueError("request count is unavailable")
    evidence_status = (
        "baseline_no_speculation"
        if spec.plan.drafter is None
        else "aggregate_fixed_k_counters_only"
        if spec.plan.method == "fixed"
        and spec.plan.verifier_k is not None
        and spec.plan.verifier_k > 0
        else "selected_k_and_physical_trace_unavailable"
    )
    speculative = build_speculative_config(
        spec.plan,
        spec.schedule,
        drafter_path=drafter_path,
    )
    args = [
        "python3",
        "-m",
        "experiments.vllm_028_q30_sync_dynamicsd.live_runner",
        "--worker-index",
        '"${SLURM_PROCID}"',
        "--external-engine-count",
        str(spec.worker_count),
        "--requests-per-engine",
        str(request_count),
        "--max-tokens",
        str(contract.max_tokens),
        "--temperature",
        str(contract.temperature),
        "--top-p",
        str(contract.top_p),
        "--data-parallel-size",
        str(contract.data_parallel_size),
        "--prompt-jsonl",
        '"${PROMPT_JSONL}"',
        "--result-dir",
        '"${WORKER_RESULT_DIR}"',
        "--cuda-graph-mode",
        contract.cuda_graph_mode,
        "--moe-backend",
        "flashinfer_trtllm",
        "--target-path",
        '"${TARGET_PATH}"',
        "--plan-json",
        shlex.quote(
            _compact_json(
                {
                    "key": spec.plan.key,
                    "stage": spec.plan.stage,
                    "drafter": spec.plan.drafter,
                    "method": spec.plan.method,
                    "controller": spec.plan.controller,
                    "batch_size": spec.plan.batch_size,
                    "verifier_k": spec.plan.verifier_k,
                    "physical_block_size": spec.plan.physical_block_size,
                }
            )
        ),
        "--unsupported-receipt",
        '"${WORKER_RESULT_DIR}/unsupported-receipt.json"',
        "--reason",
        shlex.quote(UNSUPPORTED_REASON),
        "--evidence-status",
        evidence_status,
        "--runtime-drafter-path",
        '"${RUNTIME_DRAFTER_PATH}"',
        "--runtime-drafter-config-sha256",
        '"${RUNTIME_DRAFTER_CONFIG_SHA256}"',
        "--source-drafter-config-sha256",
        '"${DRAFTER_CONFIG_SHA256}"',
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-batched-tokens",
        "32768",
        "--disable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-model-len",
        "4096",
    ]
    if speculative is not None:
        escaped = _compact_json(speculative).replace('"', '\\"')
        args.extend(("--speculative-config-json", f'"{escaped}"'))
    command = " ".join(args)
    if evidence_status == "selected_k_and_physical_trace_unavailable":
        return command + "; status=$?; [[ $status -eq 2 ]] || exit $status; exit 2"
    return command + "; status=$?; [[ $status -eq 0 ]] || exit $status"


def render_job_sbatch(
    spec: JobSpec,
    *,
    cluster: ClusterConfig,
    source_commit: str,
) -> str:
    """Return one self-validating sbatch script without contacting Slurm."""
    if len(source_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if spec.worker_count != spec.nodes * spec.gpus_per_node:
        raise ValueError("worker_count must equal nodes * gpus_per_node")
    if spec.gpus_per_node not in (1, cluster.gpus_per_node):
        raise ValueError("job must request either one canary GPU or four GPUs per node")
    contract = ExperimentContract()
    request_count = (
        spec.plan.batch_size
        if spec.plan.stage == "calibration"
        else contract.requests_per_engine
    )
    if request_count is None:
        raise ValueError("request count is unavailable")
    target = contract.target_path
    drafter = (
        "" if spec.plan.drafter is None else contract.drafter_paths[spec.plan.drafter]
    )
    adaptive_setup = ""
    runtime_drafter = drafter
    runtime_drafter_declaration = (
        f"readonly RUNTIME_DRAFTER_PATH={_shell_quote(drafter)}"
    )
    if spec.plan.method == "adaptive":
        runtime_drafter = (
            "/raid/scratch/${USER}/q30-vllm028-${SLURM_JOB_ID}/dspark-adaptive-overlay"
        )
        runtime_drafter_declaration = (
            'readonly RUNTIME_DRAFTER_PATH="${NODE_LOCAL_ROOT}/dspark-adaptive-overlay"'
        )
        adaptive_setup = f"""
# DSpark adaptive overlay: copy the checkpoint, never mutate the source checkpoint.
srun --nodes={spec.nodes} --ntasks={spec.nodes} --ntasks-per-node=1 --export=ALL,NODE_LOCAL_ROOT bash -lc '
set -euo pipefail
readonly ADAPTIVE_OVERLAY="${{NODE_LOCAL_ROOT}}/dspark-adaptive-overlay"
[[ ! -e "${{ADAPTIVE_OVERLAY}}" ]] || {{ echo "Refusing to overwrite DSpark adaptive overlay" >&2; exit 1; }}
cp -a {_shell_quote(drafter)} "${{ADAPTIVE_OVERLAY}}"
python3 - "${{ADAPTIVE_OVERLAY}}/config.json" <<'"'"'PY'"'"'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
config["enable_confidence_head"] = True
config["confidence_head_with_markov"] = True
path.write_text(json.dumps(config, indent=2) + "\\n", encoding="utf-8")
PY
'
"""
    command = _runtime_command(spec, runtime_drafter)
    rendered_speculative = build_speculative_config(
        spec.plan,
        spec.schedule,
        drafter_path=runtime_drafter,
    )
    speculative_receipt = _compact_json(rendered_speculative)
    tasks_per_node = spec.gpus_per_node
    result_cell = f"{cluster.result_root}/{spec.result_subdir}"
    shell = _shell_quote
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=coreai_dlalgo_llm-q30dyn.{spec.key}
#SBATCH --account={cluster.account}
#SBATCH --partition={cluster.partition}
#SBATCH --nodes={spec.nodes}
#SBATCH --exclusive
#SBATCH --segment={spec.nodes}
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output={shell(cluster.result_root + "/slurm-%x-%j.out")}
# speculative_config_json={speculative_receipt}

set -euo pipefail
readonly EXPECTED_SOURCE_COMMIT={source_commit}
readonly EXPECTED_VLLM_COMMIT={contract.vllm_commit}
readonly EXPECTED_PATCHSET_MANIFEST_SHA256={PATCHSET_SHA256}
readonly EXPECTED_CONTAINER_ARTIFACT_SHA256={CONTAINER_SHA256}
readonly REPO_ROOT={shell(cluster.remote_cwd)}
readonly STABLE_CONTAINER_IMAGE={shell(cluster.container_image)}
readonly CONTAINER_VERIFICATION_RECEIPT={shell(cluster.container_verification_receipt)}
readonly CONTAINER_METADATA="${{STABLE_CONTAINER_IMAGE}}.metadata.json"
readonly TARGET_PATH={shell(target)}
readonly DRAFTER_PATH={shell(drafter)}
readonly PROMPT_JSONL={shell(cluster.prompt_jsonl)}
readonly NODE_LOCAL_ROOT="/raid/scratch/${{USER}}/q30-vllm028-${{SLURM_JOB_ID}}"
readonly RESULT_CELL_DIR={shell(result_cell)}
readonly RESULT_RUN_DIR="${{RESULT_CELL_DIR}}/job-${{SLURM_JOB_ID}}"

[[ "$(git -C "${{REPO_ROOT}}" rev-parse HEAD)" == "${{EXPECTED_SOURCE_COMMIT}}" ]] || {{ echo "source commit mismatch" >&2; exit 1; }}
if [[ -n "$(git -C "${{REPO_ROOT}}" status --porcelain --untracked-files=all)" ]]; then
  echo "source worktree is not clean" >&2
  exit 1
fi
[[ -r "${{STABLE_CONTAINER_IMAGE}}" && -r "${{CONTAINER_METADATA}}" && -r "${{CONTAINER_VERIFICATION_RECEIPT}}" ]] || {{ echo "missing authenticated container receipt" >&2; exit 1; }}
python3 - "${{CONTAINER_METADATA}}" "${{STABLE_CONTAINER_IMAGE}}" "${{CONTAINER_VERIFICATION_RECEIPT}}" "${{EXPECTED_VLLM_COMMIT}}" "${{EXPECTED_PATCHSET_MANIFEST_SHA256}}" "${{EXPECTED_CONTAINER_ARTIFACT_SHA256}}" <<'PY'
import json
import sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
container_stat = Path(sys.argv[2]).stat()
if metadata.get("vllm_base_commit") != sys.argv[4]:
    raise SystemExit("vLLM base commit mismatch")
if metadata.get("patchset_manifest_sha256") != sys.argv[5]:
    raise SystemExit("patchset_manifest_sha256 mismatch")
if metadata.get("artifact_sha256") != sys.argv[6]:
    raise SystemExit("artifact_sha256 metadata mismatch")
expected_receipt = {{
    "container_path": str(Path(sys.argv[2]).resolve()),
    "artifact_sha256": sys.argv[6],
    "container_size_bytes": container_stat.st_size,
    "container_mtime_ns": container_stat.st_mtime_ns,
}}
if any(receipt.get(key) != value for key, value in expected_receipt.items()):
    raise SystemExit("container verification receipt mismatch")
PY
[[ -r "${{TARGET_PATH}}/config.json" && -r "${{PROMPT_JSONL}}" ]] || {{ echo "missing model or prompts" >&2; exit 1; }}
TARGET_CONFIG_SHA256=$(sha256sum "${{TARGET_PATH}}/config.json" | awk '{{print $1}}')
readonly TARGET_CONFIG_SHA256
DRAFTER_CONFIG_SHA256=""
if [[ -n "${{DRAFTER_PATH}}" ]]; then
  DRAFTER_CONFIG_SHA256=$(sha256sum "${{DRAFTER_PATH}}/config.json" | awk '{{print $1}}')
fi
readonly DRAFTER_CONFIG_SHA256
PROMPT_SOURCE_SHA256=$(sha256sum "${{PROMPT_JSONL}}" | awk '{{print $1}}')
readonly PROMPT_SOURCE_SHA256
export EXPECTED_SOURCE_COMMIT EXPECTED_VLLM_COMMIT EXPECTED_PATCHSET_MANIFEST_SHA256
export EXPECTED_CONTAINER_ARTIFACT_SHA256 TARGET_PATH TARGET_CONFIG_SHA256
export DRAFTER_PATH DRAFTER_CONFIG_SHA256 PROMPT_JSONL PROMPT_SOURCE_SHA256
export CONTAINER_VERIFICATION_RECEIPT
if [[ -e "${{RESULT_RUN_DIR}}" ]]; then
  echo "Refusing to overwrite existing result run: ${{RESULT_RUN_DIR}}" >&2
  exit 1
fi
mkdir -p "${{NODE_LOCAL_ROOT}}" "${{RESULT_RUN_DIR}}"
python3 - "${{RESULT_RUN_DIR}}/runtime-provenance.json" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {{
    "schema_version": 1,
    "source_commit": os.environ["EXPECTED_SOURCE_COMMIT"],
    "vllm_commit": os.environ["EXPECTED_VLLM_COMMIT"],
    "patchset_manifest_sha256": os.environ["EXPECTED_PATCHSET_MANIFEST_SHA256"],
    "container_artifact_sha256": os.environ["EXPECTED_CONTAINER_ARTIFACT_SHA256"],
    "container_verification_receipt": os.environ["CONTAINER_VERIFICATION_RECEIPT"],
    "target_path": os.environ["TARGET_PATH"],
    "target_config_sha256": os.environ["TARGET_CONFIG_SHA256"],
    "drafter_path": os.environ["DRAFTER_PATH"] or None,
    "source_drafter_config_sha256": os.environ["DRAFTER_CONFIG_SHA256"] or None,
    "prompt_jsonl": os.environ["PROMPT_JSONL"],
    "prompt_source_sha256": os.environ["PROMPT_SOURCE_SHA256"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "nodes": {spec.nodes},
    "gpus_per_node": {spec.gpus_per_node},
    "external_engine_count": {spec.worker_count},
    "requests_per_engine": {request_count},
    "actual_request_count": {request_count},
    "max_tokens": {contract.max_tokens},
    "temperature": {contract.temperature},
    "top_p": {contract.top_p},
    "natural_eos": True,
    "cuda_graph_mode": "{contract.cuda_graph_mode}",
    "moe_backend": "flashinfer_trtllm",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 32768,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "max_model_len": 4096,
}}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
temporary.replace(path)
PY
export HF_HOME="${{NODE_LOCAL_ROOT}}/hf" XDG_CACHE_HOME="${{NODE_LOCAL_ROOT}}/xdg"
export VLLM_CACHE_ROOT="${{NODE_LOCAL_ROOT}}/vllm" TORCHINDUCTOR_CACHE_DIR="${{NODE_LOCAL_ROOT}}/torchinductor"
export TRITON_CACHE_DIR="${{NODE_LOCAL_ROOT}}/triton" CUDA_CACHE_PATH="${{NODE_LOCAL_ROOT}}/cuda"
export VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_MOE_BACKEND=flashinfer_trtllm
export RESULT_RUN_DIR TARGET_PATH PROMPT_JSONL NODE_LOCAL_ROOT
{adaptive_setup}
readonly CONTAINER_MOUNTS="${{REPO_ROOT}}:/workspace/repo,/lustre:/lustre,/raid/scratch:/raid/scratch"
read -r -d '' CONTAINER_COMMAND <<'CONTAINER_SCRIPT' || true
    set -euo pipefail
    cd /workspace/repo
    export PYTHONPATH=/workspace/repo
    readonly WORKER_RESULT_DIR="${{RESULT_RUN_DIR}}/worker-${{SLURM_PROCID:-0}}"
    if [[ -e "${{WORKER_RESULT_DIR}}" ]]; then echo "Refusing to overwrite ${{WORKER_RESULT_DIR}}" >&2; exit 1; fi
    mkdir -p "${{WORKER_RESULT_DIR}}"
    {runtime_drafter_declaration}
    RUNTIME_DRAFTER_CONFIG_SHA256=""
    if [[ -n "${{RUNTIME_DRAFTER_PATH}}" ]]; then
      RUNTIME_DRAFTER_CONFIG_SHA256=$(sha256sum "${{RUNTIME_DRAFTER_PATH}}/config.json" | awk '{{print $1}}')
    fi
    readonly RUNTIME_DRAFTER_CONFIG_SHA256
    set +e
    {command}
CONTAINER_SCRIPT
srun --nodes={spec.nodes} --ntasks={spec.worker_count} --ntasks-per-node={tasks_per_node} --gpus-per-task=1 \
  --container-image="${{STABLE_CONTAINER_IMAGE}}" --container-mounts="${{CONTAINER_MOUNTS}}" \
  --no-container-mount-home --container-remap-root bash -lc "${{CONTAINER_COMMAND}}"
"""


def render_stage(
    *,
    stage: Literal["canary", "calibration", "barrier"],
    output_dir: Path,
    cluster: ClusterConfig,
    source_commit: str,
    schedules: Mapping[str, Sequence[Sequence[int]]] | None = None,
    best_fixed_k: Mapping[str, int] | None = None,
    include_dspark_adaptive: bool = False,
) -> list[Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to render into non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "canary":
        if schedules is None:
            raise ValueError("canary rendering requires explicit DynamicSD schedules")
        barrier_by_key = {row.key: row for row in build_barrier_rows()}
        calibration = build_calibration_rows()
        canary_rows = (
            barrier_by_key["target_only"],
            barrier_by_key["dflash_dynamicsd"],
            barrier_by_key["dspark_dynamicsd"],
            next(
                row
                for row in calibration
                if row.drafter == "dflash"
                and row.batch_size == 1
                and row.verifier_k == 0
            ),
            next(
                row
                for row in calibration
                if row.drafter == "dspark"
                and row.batch_size == 1
                and row.verifier_k == 0
            ),
            build_barrier_rows(include_dspark_adaptive=True)[-1],
        )
        specs = [
            JobSpec(
                f"canary_{row.key}",
                row,
                1,
                1,
                1,
                f"canary/{row.key}",
                schedules[row.drafter]
                if row.method == "dynamic" and row.drafter
                else None,
            )
            for row in canary_rows
        ]
    elif stage == "calibration":
        rows = build_calibration_rows()
        specs = [
            JobSpec(row.key, row, 1, 1, 1, f"calibration/{row.key}") for row in rows
        ]
    elif stage == "barrier":
        if schedules is None or best_fixed_k is None:
            raise ValueError(
                "barrier rendering requires calibrated schedules and fixed K"
            )
        typed_fixed = {key: int(value) for key, value in best_fixed_k.items()}
        rows = build_barrier_rows(
            best_fixed_k=typed_fixed,  # type: ignore[arg-type]
            include_dspark_adaptive=include_dspark_adaptive,
        )
        specs = [
            JobSpec(
                row.key,
                row,
                4,
                4,
                16,
                f"barrier/{row.key}",
                None if row.method != "dynamic" else schedules[row.drafter or ""],
            )
            for row in rows
        ]
    else:
        raise ValueError(f"unsupported stage: {stage}")
    rendered: list[Path] = []
    for spec in specs:
        path = output_dir / f"{spec.key}.sbatch"
        if path.exists():
            raise FileExistsError(f"refusing to render existing script: {path}")
        path.write_text(
            render_job_sbatch(spec, cluster=cluster, source_commit=source_commit),
            encoding="utf-8",
        )
        path.chmod(0o750)
        rendered.append(path)
    return rendered


Runner = Callable[..., subprocess.CompletedProcess[str]]
AcceptedCallback = Callable[[str, Path], None]


class SubmissionDispatchError(RuntimeError):
    """Submission failed after zero or more job IDs were durably accepted."""

    def __init__(
        self,
        message: str,
        *,
        accepted_job_ids: Sequence[str],
        receipt_path: Path | None,
    ) -> None:
        super().__init__(message)
        self.accepted_job_ids = tuple(accepted_job_ids)
        self.receipt_path = receipt_path


def _append_submission_receipt(path: Path, *, job_id: str, script: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"job_id": job_id, "script": str(script)}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def dispatch_scripts(
    scripts: Sequence[Path],
    *,
    mode: DispatchMode,
    runner: Runner = subprocess.run,
    receipt_path: Path | None = None,
    on_accepted: AcceptedCallback | None = None,
) -> list[str]:
    if mode == "render":
        return []
    if mode not in ("test-only", "submit"):
        raise ValueError(f"unsupported dispatch mode: {mode}")
    durable_receipt = receipt_path
    if mode == "submit" and durable_receipt is None:
        if not scripts:
            raise ValueError("submit mode requires at least one script")
        durable_receipt = scripts[0].parent / "submission-receipt.jsonl"
    if durable_receipt is not None and durable_receipt.exists():
        raise FileExistsError(
            f"refusing to append to existing receipt: {durable_receipt}"
        )
    job_ids: list[str] = []
    for script in scripts:
        args = ["sbatch"]
        if mode == "test-only":
            args.append("--test-only")
        args.extend(("--parsable", str(script)))
        try:
            completed = runner(args, check=True, capture_output=True, text=True)
            if mode == "test-only":
                continue
            job_id = completed.stdout.strip().split(";", 1)[0]
            if not job_id:
                raise ValueError("sbatch returned an empty job ID")
            if durable_receipt is not None:
                _append_submission_receipt(
                    durable_receipt,
                    job_id=job_id,
                    script=script,
                )
            job_ids.append(job_id)
            if on_accepted is not None:
                on_accepted(job_id, script)
        except Exception as exc:
            raise SubmissionDispatchError(
                f"dispatch failed for {script}",
                accepted_job_ids=job_ids,
                receipt_path=durable_receipt,
            ) from exc
    return job_ids


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("canary", "calibration", "barrier"))
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parsed = parser.parse_args()
    if parsed.stage != "calibration":
        parser.error(
            "canary/barrier CLI rendering requires programmatic schedule inputs"
        )
    scripts = render_stage(
        stage=parsed.stage,
        output_dir=parsed.output_dir,
        cluster=load_cluster_config(parsed.cluster_config),
        source_commit=parsed.source_commit,
    )
    selected_mode: DispatchMode = (
        "submit" if parsed.submit else "test-only" if parsed.test_only else "render"
    )
    for job_id in dispatch_scripts(scripts, mode=selected_mode):
        print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
