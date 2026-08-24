#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


MAX_SMOKE_REQUESTS = 20
_READINESS_TIMEOUT_SECONDS = 900.0
_REQUEST_TIMEOUT_SECONDS = 600.0
_PROMPTS = (
    "A box has 12 pencils. Mia adds 7 and gives away 5. How many remain?",
    "A train travels 45 miles in one hour and 60 miles in the next. How far total?",
    "There are 8 bags with 6 oranges each. Four oranges are eaten. How many remain?",
    "Sam saves $9 each week for 5 weeks, then spends $12. How much is left?",
    "A class has 14 girls and 11 boys. Three students leave. How many remain?",
)


class TokenCounts(TypedDict):
    prompt: int
    completion: int
    total: int


@dataclass(frozen=True, slots=True)
class EngineConfig:
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    served_model_name: str
    runner: str


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    dataset: str
    requests: int
    concurrency: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    source: Path
    sha256: str
    schema_version: int
    name: str
    mode: str
    model: str
    model_revision: str
    engine: EngineConfig
    workload: WorkloadConfig
    speculative_config: dict[str, object] | None
    draft_refit: bool
    online_draft_training: bool

    @property
    def arm(self) -> str:
        return "dflash2" if self.speculative_config is not None else "baseline"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _required(
    values: Mapping[str, object], name: str, expected_type: type[object]
) -> object:
    value = values.get(name)
    if isinstance(value, bool) and expected_type is int:
        raise ValueError(f"{name} must be {expected_type.__name__}")
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be {expected_type.__name__}")
    return value


def validate_request_count(request_count: int) -> int:
    if isinstance(request_count, bool) or not 1 <= request_count <= MAX_SMOKE_REQUESTS:
        raise ValueError("request_count must be between 1 and 20")
    return request_count


def load_manifest(path: Path) -> ExperimentManifest:
    raw_bytes = path.read_bytes()
    values = _mapping(yaml.safe_load(raw_bytes), name="manifest")
    engine_values = _mapping(values.get("engine"), name="engine")
    workload_values = _mapping(values.get("workload"), name="workload")
    request_count = validate_request_count(
        cast(int, _required(workload_values, "requests", int))
    )
    concurrency = cast(int, _required(workload_values, "concurrency", int))
    if not 1 <= concurrency <= request_count:
        raise ValueError("workload.concurrency must be between 1 and requests")

    speculative_value = values.get("speculative_config")
    speculative_config: dict[str, object] | None = None
    if speculative_value is not None:
        speculative_config = dict(
            _mapping(speculative_value, name="speculative_config")
        )
        expected_speculative_config = {
            "method": "dflash",
            "model": "incoai/Qwen3.8-27B-DFlash2",
            "num_speculative_tokens": 7,
            "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
        }
        if speculative_config != expected_speculative_config:
            raise ValueError(
                "speculative_config must match the pinned DFlash2 static contract"
            )

    manifest = ExperimentManifest(
        source=path.resolve(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=cast(int, _required(values, "schema_version", int)),
        name=cast(str, _required(values, "name", str)),
        mode=cast(str, _required(values, "mode", str)),
        model=cast(str, _required(values, "model", str)),
        model_revision=cast(str, _required(values, "model_revision", str)),
        engine=EngineConfig(
            tensor_parallel_size=cast(
                int, _required(engine_values, "tensor_parallel_size", int)
            ),
            dtype=cast(str, _required(engine_values, "dtype", str)),
            max_model_len=cast(int, _required(engine_values, "max_model_len", int)),
            gpu_memory_utilization=cast(
                float, _required(engine_values, "gpu_memory_utilization", float)
            ),
            served_model_name=cast(
                str, _required(engine_values, "served_model_name", str)
            ),
            runner=cast(str, _required(engine_values, "runner", str)),
        ),
        workload=WorkloadConfig(
            dataset=cast(str, _required(workload_values, "dataset", str)),
            requests=request_count,
            concurrency=concurrency,
            max_new_tokens=cast(int, _required(workload_values, "max_new_tokens", int)),
            temperature=float(
                cast(float, _required(workload_values, "temperature", float))
            ),
            top_p=float(cast(float, _required(workload_values, "top_p", float))),
            top_k=cast(int, _required(workload_values, "top_k", int)),
            reasoning_effort=cast(
                str, _required(workload_values, "reasoning_effort", str)
            ),
        ),
        speculative_config=speculative_config,
        draft_refit=cast(bool, _required(values, "draft_refit", bool)),
        online_draft_training=cast(
            bool, _required(values, "online_draft_training", bool)
        ),
    )
    if manifest.schema_version != 1 or manifest.mode != "static_rollout":
        raise ValueError("launcher supports only schema_version=1 static_rollout")
    if manifest.model != "Qwen/Qwen3.8-27B" or manifest.model_revision != (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    ):
        raise ValueError(
            "target model and revision must match the pinned smoke contract"
        )
    if manifest.engine.runner != "v2":
        raise ValueError("DFlash2 static smoke requires engine.runner=v2")
    if manifest.draft_refit or manifest.online_draft_training:
        raise ValueError(
            "standalone static smoke forbids draft refit and online training"
        )
    return manifest


def build_server_command(manifest: ExperimentManifest, *, port: int) -> list[str]:
    command = [
        "vllm",
        "serve",
        manifest.model,
        "--revision",
        manifest.model_revision,
        "--served-model-name",
        manifest.engine.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(manifest.engine.tensor_parallel_size),
        "--dtype",
        manifest.engine.dtype,
        "--max-model-len",
        str(manifest.engine.max_model_len),
        "--gpu-memory-utilization",
        str(manifest.engine.gpu_memory_utilization),
    ]
    if manifest.speculative_config is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    manifest.speculative_config,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ]
        )
    return command


def manifest_metadata(manifest: ExperimentManifest) -> dict[str, object]:
    """Return the complete manifest with filesystem values normalized for JSON."""
    values = asdict(manifest)
    values["source"] = str(manifest.source)
    values["arm"] = manifest.arm
    return values


def _post_chat_completion(
    manifest: ExperimentManifest,
    *,
    base_url: str,
    request_index: int,
) -> dict[str, object]:
    payload = {
        "model": manifest.engine.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": _PROMPTS[request_index % len(_PROMPTS)],
            }
        ],
        "max_tokens": manifest.workload.max_new_tokens,
        "temperature": manifest.workload.temperature,
        "top_p": manifest.workload.top_p,
        "top_k": manifest.workload.top_k,
        "seed": 1000 + request_index,
        "chat_template_kwargs": {
            "reasoning_effort": manifest.workload.reasoning_effort
        },
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            response_values = _mapping(
                json.loads(response.read()), name="chat completion"
            )
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"chat request {request_index} failed: {exc.code} {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"chat request {request_index} failed: {exc.reason}"
        ) from exc
    elapsed = time.perf_counter() - started
    usage = _mapping(response_values.get("usage"), name="usage")
    choices = response_values.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("chat completion must contain exactly one choice")
    choice = _mapping(choices[0], name="choice")
    return {
        "index": request_index,
        "seed": 1000 + request_index,
        "latency_seconds": elapsed,
        "prompt_tokens": cast(int, _required(usage, "prompt_tokens", int)),
        "completion_tokens": cast(int, _required(usage, "completion_tokens", int)),
        "finish_reason": cast(str, _required(choice, "finish_reason", str)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def benchmark_server(
    manifest: ExperimentManifest,
    *,
    base_url: str,
    request_count: int,
) -> dict[str, object]:
    request_count = validate_request_count(request_count)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=manifest.workload.concurrency) as executor:
        requests = list(
            executor.map(
                lambda request_index: _post_chat_completion(
                    manifest,
                    base_url=base_url,
                    request_index=request_index,
                ),
                range(request_count),
            )
        )
    elapsed = time.perf_counter() - started
    prompt_tokens = sum(cast(int, result["prompt_tokens"]) for result in requests)
    completion_tokens = sum(
        cast(int, result["completion_tokens"]) for result in requests
    )
    latencies = [cast(float, result["latency_seconds"]) for result in requests]
    return {
        "schema_version": 1,
        "status": "passed",
        "experiment": manifest.name,
        "arm": manifest.arm,
        "model": manifest.model,
        "model_revision": manifest.model_revision,
        "manifest_sha256": manifest.sha256,
        "speculative_config": manifest.speculative_config,
        "requested_requests": request_count,
        "completed_requests": len(requests),
        "failed_requests": 0,
        "concurrency": manifest.workload.concurrency,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": completion_tokens / elapsed,
        "tokens": TokenCounts(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens,
        ),
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "requests": requests,
    }


def _runtime_fingerprint(experiment_root: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(experiment_root / "preflight.py"), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(_mapping(json.loads(result.stdout), name="runtime fingerprint"))


def _wait_for_server(
    base_url: str,
    server: subprocess.Popen[str] | subprocess.Popen[bytes],
) -> None:
    deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before readiness: {server.returncode}"
            )
        try:
            with urlopen(f"{base_url}/health", timeout=2.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(2.0)
    raise TimeoutError("vLLM server did not become healthy within 900 seconds")


def _stop_server(
    server: subprocess.Popen[str] | subprocess.Popen[bytes],
) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def run_experiment(
    manifest: ExperimentManifest,
    *,
    output_dir: Path,
    port: int,
    request_count: int,
) -> dict[str, object]:
    request_count = validate_request_count(request_count)
    output_dir.mkdir(parents=True, exist_ok=False)
    experiment_root = Path(__file__).resolve().parent
    command = build_server_command(manifest, port=port)
    started_at = datetime.now(timezone.utc)
    execution = {
        "container_image": os.environ.get("NRL_CONTAINER_IMAGE"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    runtime: dict[str, object] | None = None
    server: subprocess.Popen[str] | subprocess.Popen[bytes] | None = None
    try:
        runtime = _runtime_fingerprint(experiment_root)
        with (output_dir / "server.log").open("wb") as server_log:
            server = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base_url = f"http://127.0.0.1:{port}"
            _wait_for_server(base_url, server)
            summary = benchmark_server(
                manifest,
                base_url=base_url,
                request_count=request_count,
            )
            summary.update(
                {
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "runtime": runtime,
                    "server_command": command,
                    "manifest": manifest_metadata(manifest),
                    "execution": execution,
                }
            )
    except Exception as exc:
        summary = {
            "schema_version": 1,
            "status": "failed",
            "experiment": manifest.name,
            "arm": manifest.arm,
            "requested_requests": request_count,
            "completed_requests": 0,
            "failed_requests": request_count,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
            "server_command": command,
            "manifest": manifest_metadata(manifest),
            "execution": execution,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _write_json(output_dir / "summary.json", summary)
        raise
    finally:
        if server is not None:
            _stop_server(server)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    request_count = (
        manifest.workload.requests
        if args.request_count is None
        else validate_request_count(args.request_count)
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "arm": manifest.arm,
                    "request_count": request_count,
                    "server_command": build_server_command(manifest, port=args.port),
                },
                sort_keys=True,
            )
        )
        return
    run_experiment(
        manifest,
        output_dir=args.output_dir,
        port=args.port,
        request_count=request_count,
    )


if __name__ == "__main__":
    main()
