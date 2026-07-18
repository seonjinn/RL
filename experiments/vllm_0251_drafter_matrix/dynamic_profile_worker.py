# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collect and assemble matched vLLM 0.25.1 DynamicSD profile points."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol, Sequence


G_BATCH_SIZES = (1, 4, 16, 32, 64, 128, 192, 256)
G_K_VALUES = tuple(range(6))
G_NUM_BATCHES = 20
G_OUTPUT_LEN = 256
G_ACCEPTANCE_PROMPTS = 1024
G_DATASET_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
G_PYTHON_BIN = sys.executable
G_VLLM_BIN = str(Path(G_PYTHON_BIN).with_name("vllm"))


class ChatTokenizer(Protocol):
    """Small tokenizer surface required by the prompt exporter."""

    def apply_chat_template(self, messages: object, **kwargs: object) -> str: ...


def validate_batch_sizes(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    """Require a nonempty canonical batch-size subset supported by vLLM."""
    values = tuple(batch_sizes)
    if (
        not values
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > 256
            for value in values
        )
        or tuple(sorted(values)) != values
        or len(set(values)) != len(values)
    ):
        raise ValueError("batch sizes must be sorted unique integers in 1..256")
    return values


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def render_math_prompt(
    problem: str,
    prompt_template: str,
    tokenizer: ChatTokenizer,
) -> str:
    """Render one prompt exactly as ``math_hf_data_processor`` does."""
    content = prompt_template.format(problem) if prompt_template else problem
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )


def build_server_command(
    k: int,
    target_snapshot: Path,
    drafter_snapshot: Path | None,
    port: int,
    *,
    max_k: int = 5,
    profile_max_batch_size: int | None = None,
    served_model_name: str = "qwen32-profile",
    target_tensor_parallel_size: int = 2,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.6,
    enable_prefix_caching: bool = False,
    distributed_executor_backend: str | None = None,
    moe_backend: str | None = None,
    cudagraph_capture_sizes: tuple[int, ...] = (),
) -> tuple[str, ...]:
    """Build one matched target server command for a fixed K."""
    if max_k not in G_K_VALUES or k not in tuple(range(max_k + 1)):
        raise ValueError(f"K must be one of {G_K_VALUES}")
    if drafter_snapshot is None:
        raise ValueError("DynamicSD profiling requires the exact drafter snapshot")
    if k == 0 and profile_max_batch_size is None:
        raise ValueError("K0 profiling requires the active profile batch endpoint")
    compilation_config: dict[str, object] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if cudagraph_capture_sizes:
        compilation_config["cudagraph_capture_sizes"] = list(cudagraph_capture_sizes)
    compilation = json.dumps(compilation_config, separators=(",", ":"))
    speculative_config: dict[str, object] = {
        "method": "eagle3",
        "model": str(drafter_snapshot),
        "num_speculative_tokens": k if k > 0 else max_k,
        "draft_tensor_parallel_size": 1,
    }
    if k == 0:
        assert profile_max_batch_size is not None
        speculative_config["num_speculative_tokens_per_batch_size"] = [
            [1, profile_max_batch_size, 0]
        ]
    speculative = (
        "--speculative-config",
        json.dumps(
            speculative_config,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    distributed_backend = (
        ("--distributed-executor-backend", distributed_executor_backend)
        if distributed_executor_backend is not None
        else ()
    )
    moe = ("--moe-backend", moe_backend) if moe_backend is not None else ()
    visible_devices = (
        () if distributed_executor_backend == "ray" else ("CUDA_VISIBLE_DEVICES=0,1",)
    )
    return (
        "env",
        *visible_devices,
        "VLLM_USE_V2_MODEL_RUNNER=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        G_VLLM_BIN,
        "serve",
        str(target_snapshot),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(target_tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enable-chunked-prefill",
        "--enable-prefix-caching"
        if enable_prefix_caching
        else "--no-enable-prefix-caching",
        "--compilation-config",
        compilation,
        *distributed_backend,
        *moe,
        *speculative,
    )


def build_benchmark_command(
    *,
    batch_size: int,
    prompt_file: Path,
    tokenizer_snapshot: Path,
    result_dir: Path,
    port: int,
    served_model_name: str = "qwen32-profile",
) -> tuple[str, ...]:
    """Build an upstream-style twenty-batch serving benchmark."""
    validate_batch_sizes((batch_size,))
    return (
        G_VLLM_BIN,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        served_model_name,
        "--tokenizer",
        str(tokenizer_snapshot),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(prompt_file),
        "--skip-chat-template",
        "--output-len",
        str(G_OUTPUT_LEN),
        "--num-prompts",
        str(batch_size * G_NUM_BATCHES),
        "--max-concurrency",
        str(batch_size),
        "--request-rate",
        "inf",
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--seed",
        "42",
        "--num-warmups",
        str(min(batch_size, 32)),
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "result.json",
    )


def _ensure_prompt_file(
    root: Path,
    target_snapshot: Path,
    prompt_template_path: Path,
    *,
    num_prompts: int,
) -> Path:
    prompt_path = root / "prompts.jsonl"
    meta_path = root / "prompts.meta.json"
    lock_path = root / ".prompts.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if prompt_path.is_file() and meta_path.is_file():
            return prompt_path

        prompt_template = prompt_template_path.read_text(encoding="utf-8")
        datasets_module: Any = importlib.import_module("datasets")
        transformers_module: Any = importlib.import_module("transformers")
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            target_snapshot, local_files_only=True, trust_remote_code=True
        )
        dataset = datasets_module.load_dataset(
            "nvidia/OpenMathInstruct-2",
            revision=G_DATASET_REVISION,
            split="train_1M",
        ).train_test_split(test_size=0.05, seed=42)["train"]
        dataset = dataset.shuffle(seed=42).select(range(num_prompts))
        lines = []
        for datum in dataset:
            problem = datum["problem"]
            if not isinstance(problem, str):
                raise ValueError("OpenMathInstruct-2 problem must be a string")
            lines.append(
                json.dumps(
                    {
                        "prompt": render_math_prompt(
                            problem, prompt_template, tokenizer
                        ),
                        "output_tokens": G_OUTPUT_LEN,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        content = "\n".join(lines) + "\n"
        _atomic_write(prompt_path, content)
        _atomic_write(
            meta_path,
            json.dumps(
                {
                    "dataset_name": "OpenMathInstruct-2",
                    "dataset_repo": "nvidia/OpenMathInstruct-2",
                    "dataset_revision": G_DATASET_REVISION,
                    "dataset_split": "train_1M",
                    "dataset_seed": 42,
                    "num_prompts": num_prompts,
                    "prompt_template_sha256": hashlib.sha256(
                        prompt_template.encode("utf-8")
                    ).hexdigest(),
                    "prompt_file_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    return prompt_path


def _wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 1800
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("vLLM server did not become healthy within 30 minutes")


def _runtime_vllm_version() -> str:
    result = subprocess.run(
        (G_PYTHON_BIN, "-c", "import vllm; print(vllm.__version__)"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_benchmark_result(path: Path, batch_size: int, version: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = batch_size * G_NUM_BATCHES
    if payload.get("completed") != expected or payload.get("failed") not in {0, None}:
        raise ValueError(f"incomplete benchmark result for batch size {batch_size}")
    median_itl = payload.get("median_itl_ms")
    if (
        not isinstance(median_itl, (int, float))
        or isinstance(median_itl, bool)
        or not math.isfinite(float(median_itl))
        or float(median_itl) <= 0.0
    ):
        raise ValueError(f"invalid median_itl_ms for batch size {batch_size}")
    payload["vllm_version"] = version
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_fixed_k(
    root: Path,
    k: int,
    target_snapshot: Path,
    drafter_snapshot: Path,
    prompt_template_path: Path,
    port: int,
    *,
    batch_sizes: tuple[int, ...] = G_BATCH_SIZES,
    model_key: str = "qwen32",
    max_k: int = 5,
    profile_max_batch_size: int = 256,
    served_model_name: str = "qwen32-profile",
    target_tensor_parallel_size: int = 2,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.6,
    enable_prefix_caching: bool = False,
    distributed_executor_backend: str | None = None,
    moe_backend: str | None = None,
    cudagraph_capture_sizes: tuple[int, ...] = (),
) -> None:
    """Run selected batch-size points for one K and preserve completed cells."""
    batch_sizes = validate_batch_sizes(batch_sizes)
    if max_k not in G_K_VALUES or k not in tuple(range(max_k + 1)):
        raise ValueError(f"K must be in 0..{max_k}, got {k}")
    prompt_file = _ensure_prompt_file(
        root,
        target_snapshot,
        prompt_template_path,
        num_prompts=max(batch_sizes) * G_NUM_BATCHES,
    )
    output_dir = root / f"k-{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    version = _runtime_vllm_version()
    if version != "0.25.1":
        raise RuntimeError(f"profile requires vLLM 0.25.1, found {version}")
    server_log_name = (
        "server.log"
        if batch_sizes == G_BATCH_SIZES
        else f"server-bs-{'-'.join(str(size) for size in batch_sizes)}.log"
    )
    server_log = (output_dir / server_log_name).open("w", encoding="utf-8")
    process = subprocess.Popen(
        build_server_command(
            k,
            target_snapshot,
            drafter_snapshot,
            port,
            max_k=max_k,
            profile_max_batch_size=profile_max_batch_size,
            served_model_name=served_model_name,
            target_tensor_parallel_size=target_tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            distributed_executor_backend=distributed_executor_backend,
            moe_backend=moe_backend,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
        ),
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_server(port, process)
        for batch_size in batch_sizes:
            result_dir = output_dir / f"bs-{batch_size}"
            result_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                build_benchmark_command(
                    batch_size=batch_size,
                    prompt_file=prompt_file,
                    tokenizer_snapshot=target_snapshot,
                    result_dir=result_dir,
                    port=port,
                    served_model_name=served_model_name,
                ),
                check=True,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
            _validate_benchmark_result(result_dir / "result.json", batch_size, version)
    finally:
        _terminate_process_group(process)
        server_log.close()

    if k == max_k and (
        batch_sizes == G_BATCH_SIZES or batch_sizes[-1] == profile_max_batch_size
    ):
        subprocess.run(
            (
                G_PYTHON_BIN,
                str(Path(__file__).resolve()),
                "acceptance",
                "--root",
                str(root),
                "--target-snapshot",
                str(target_snapshot),
                "--drafter-snapshot",
                str(drafter_snapshot),
                "--max-k",
                str(max_k),
                "--target-tp",
                str(target_tensor_parallel_size),
                "--max-model-len",
                str(max_model_len),
                "--max-num-seqs",
                str(max_num_seqs),
                "--profile-max-batch-size",
                str(profile_max_batch_size),
                "--max-num-batched-tokens",
                str(max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--enable-prefix-caching"
                if enable_prefix_caching
                else "--no-enable-prefix-caching",
                *(
                    ("--distributed-executor-backend", distributed_executor_backend)
                    if distributed_executor_backend is not None
                    else ()
                ),
                *(("--moe-backend", moe_backend) if moe_backend is not None else ()),
                *(
                    (
                        "--cudagraph-capture-sizes",
                        *(str(size) for size in cudagraph_capture_sizes),
                    )
                    if cudagraph_capture_sizes
                    else ()
                ),
            ),
            check=True,
            env={
                **os.environ,
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )


def collect_acceptance(
    root: Path,
    target_snapshot: Path,
    drafter_snapshot: Path,
    *,
    max_k: int = 5,
    profile_max_batch_size: int = 256,
    target_tensor_parallel_size: int = 2,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.6,
    enable_prefix_caching: bool = False,
    distributed_executor_backend: str | None = None,
    moe_backend: str | None = None,
    cudagraph_capture_sizes: tuple[int, ...] = (),
) -> None:
    """Collect max-K position-level acceptance on deterministic Math prompts."""
    if max_k not in G_K_VALUES or max_k == 0:
        raise ValueError("acceptance profiling requires max K in 1..5")
    prompt_file = root / "prompts.jsonl"
    raw_prompts = [
        json.loads(line)["prompt"]
        for line in prompt_file.read_text(encoding="utf-8").splitlines()[
            :G_ACCEPTANCE_PROMPTS
        ]
    ]
    transformers_module: Any = importlib.import_module("transformers")
    vllm_module: Any = importlib.import_module("vllm")
    metrics_module: Any = importlib.import_module("vllm.v1.metrics.reader")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        target_snapshot, local_files_only=True, trust_remote_code=True
    )
    prompts = [
        {"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}
        for prompt in raw_prompts
    ]
    compilation_config: dict[str, object] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if cudagraph_capture_sizes:
        compilation_config["cudagraph_capture_sizes"] = list(cudagraph_capture_sizes)
    llm_kwargs: dict[str, object] = {}
    if distributed_executor_backend is not None:
        llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
    if moe_backend is not None:
        llm_kwargs["moe_backend"] = moe_backend
    llm = vllm_module.LLM(
        model=str(target_snapshot),
        tokenizer=str(target_snapshot),
        tensor_parallel_size=target_tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=False,
        compilation_config=compilation_config,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": str(drafter_snapshot),
            "num_speculative_tokens": max_k,
            "draft_tensor_parallel_size": 1,
        },
        **llm_kwargs,
    )
    llm.generate(
        prompts,
        vllm_module.SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=G_OUTPUT_LEN,
            seed=42,
        ),
        use_tqdm=False,
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * max_k
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            if not isinstance(metric, metrics_module.Counter):
                raise TypeError("num_drafts metric has an unexpected type")
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            if not isinstance(metric, metrics_module.Counter):
                raise TypeError("num_draft_tokens metric has an unexpected type")
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            if not isinstance(metric, metrics_module.Counter):
                raise TypeError("num_accepted_tokens metric has an unexpected type")
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            if not isinstance(metric, metrics_module.Vector):
                raise TypeError("position acceptance metric has an unexpected type")
            for position, value in enumerate(metric.values[:max_k]):
                acceptance_counts[position] += value
    if num_drafts <= 0:
        raise RuntimeError(f"K{max_k} acceptance profile emitted no draft events")
    payload = {
        "num_prompts": len(prompts),
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate_per_pos": [count / num_drafts for count in acceptance_counts],
    }
    _atomic_write(
        root / "acceptance.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def assemble_profile(
    root: Path,
    *,
    model_key: str = "qwen32",
    target_revision: str,
    drafter_revision: str,
    batch_sizes: tuple[int, ...] = G_BATCH_SIZES,
    k_values: tuple[int, ...] = G_K_VALUES,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 256,
    profile_max_batch_size: int = 256,
    target_tensor_parallel_size: int = 2,
    enable_prefix_caching: bool = False,
    moe_backend: str | None = None,
    cudagraph_capture_sizes: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Assemble all complete cells into the calibrator's strict raw schema."""
    if batch_sizes[-1] != profile_max_batch_size:
        raise ValueError("profile batch-size grid must end at profile_max_batch_size")
    if profile_max_batch_size > max_num_seqs:
        raise ValueError("profile_max_batch_size must not exceed max_num_seqs")
    if max_num_batched_tokens < max_num_seqs:
        raise ValueError("max_num_batched_tokens must not be smaller than max_num_seqs")
    if cudagraph_capture_sizes:
        required_endpoint_shapes = {
            profile_max_batch_size * (k + 1) for k in k_values[1:]
        }
        missing_shapes = sorted(required_endpoint_shapes - set(cudagraph_capture_sizes))
        if missing_shapes:
            raise ValueError(
                "CUDA Graph capture sizes do not cover profile endpoint shapes: "
                f"{missing_shapes}"
            )
    acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    rates = acceptance.get("acceptance_rate_per_pos")
    if not isinstance(rates, list) or len(rates) != k_values[-1]:
        raise ValueError(f"acceptance profile must contain {k_values[-1]} positions")
    prompt_meta = json.loads((root / "prompts.meta.json").read_text(encoding="utf-8"))
    if prompt_meta.get("dataset_revision") != G_DATASET_REVISION:
        raise ValueError("prompt data must use the pinned OpenMathInstruct-2 revision")
    rows: list[dict[str, Any]] = []
    for k in k_values:
        for batch_size in batch_sizes:
            result_path = root / f"k-{k}" / f"bs-{batch_size}" / "result.json"
            if not result_path.is_file():
                raise ValueError(f"missing profile result: K{k}, BS{batch_size}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            completed = result.get("completed")
            if completed != batch_size * G_NUM_BATCHES:
                raise ValueError(f"incomplete profile result: K{k}, BS{batch_size}")
            if result.get("vllm_version") != "0.25.1":
                raise ValueError(f"wrong vLLM profile result: K{k}, BS{batch_size}")
            rows.append(
                {
                    "batch_size": batch_size,
                    "k": k,
                    "median_itl_ms": result["median_itl_ms"],
                    "completed_batches": completed // batch_size,
                }
            )
    return {
        "schema_version": 2,
        "calibration_status": "complete",
        "model_key": model_key,
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "runtime_vllm": "0.25.1",
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
        "dataset_name": prompt_meta["dataset_name"],
        "dataset_revision": prompt_meta["dataset_revision"],
        "prompt_template_sha256": prompt_meta["prompt_template_sha256"],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "profile_max_batch_size": profile_max_batch_size,
        "enable_prefix_caching": enable_prefix_caching,
        "moe_backend": moe_backend,
        "cudagraph_capture_sizes": list(cudagraph_capture_sizes),
        "target_tensor_parallel_size": target_tensor_parallel_size,
        "draft_tensor_parallel_size": 1,
        "num_batches_per_point": G_NUM_BATCHES,
        "batch_sizes": list(batch_sizes),
        "k_values": list(k_values),
        "acceptance_rate_per_pos": rates,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-k")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--k", type=int, choices=G_K_VALUES, required=True)
    run.add_argument("--target-snapshot", type=Path, required=True)
    run.add_argument("--drafter-snapshot", type=Path, required=True)
    run.add_argument("--prompt-template", type=Path, required=True)
    run.add_argument("--port", type=int, default=8100)
    run.add_argument("--batch-sizes", nargs="+", type=int, default=G_BATCH_SIZES)
    run.add_argument("--model-key", default="qwen32")
    run.add_argument("--max-k", type=int, choices=G_K_VALUES, default=5)
    run.add_argument("--profile-max-batch-size", type=int, default=256)
    run.add_argument("--served-model-name", default="qwen32-profile")
    run.add_argument("--target-tp", type=int, default=2)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--max-num-seqs", type=int, default=256)
    run.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    run.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument("--distributed-executor-backend")
    run.add_argument("--moe-backend")
    run.add_argument("--cudagraph-capture-sizes", nargs="+", type=int, default=())

    acceptance = subparsers.add_parser("acceptance")
    acceptance.add_argument("--root", type=Path, required=True)
    acceptance.add_argument("--target-snapshot", type=Path, required=True)
    acceptance.add_argument("--drafter-snapshot", type=Path, required=True)
    acceptance.add_argument("--max-k", type=int, choices=G_K_VALUES, default=5)
    acceptance.add_argument("--profile-max-batch-size", type=int, default=256)
    acceptance.add_argument("--target-tp", type=int, default=2)
    acceptance.add_argument("--max-model-len", type=int, default=4096)
    acceptance.add_argument("--max-num-seqs", type=int, default=256)
    acceptance.add_argument("--max-num-batched-tokens", type=int, default=16384)
    acceptance.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    acceptance.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    acceptance.add_argument("--distributed-executor-backend")
    acceptance.add_argument("--moe-backend")
    acceptance.add_argument(
        "--cudagraph-capture-sizes", nargs="+", type=int, default=()
    )

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--root", type=Path, required=True)
    assemble.add_argument("--target-revision", required=True)
    assemble.add_argument("--drafter-revision", required=True)
    assemble.add_argument("--model-key", default="qwen32")
    assemble.add_argument("--max-k", type=int, choices=G_K_VALUES, default=5)
    assemble.add_argument("--profile-max-batch-size", type=int, default=256)
    assemble.add_argument("--batch-sizes", nargs="+", type=int, default=G_BATCH_SIZES)
    assemble.add_argument("--target-tp", type=int, default=2)
    assemble.add_argument("--max-model-len", type=int, default=4096)
    assemble.add_argument("--max-num-seqs", type=int, default=256)
    assemble.add_argument("--max-num-batched-tokens", type=int, default=16384)
    assemble.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    assemble.add_argument("--moe-backend")
    assemble.add_argument("--cudagraph-capture-sizes", nargs="+", type=int, default=())
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-k":
        try:
            batch_sizes = validate_batch_sizes(args.batch_sizes)
        except ValueError as error:
            parser.error(str(error))
        run_fixed_k(
            args.root,
            args.k,
            args.target_snapshot,
            args.drafter_snapshot,
            args.prompt_template,
            args.port,
            batch_sizes=batch_sizes,
            model_key=args.model_key,
            max_k=args.max_k,
            profile_max_batch_size=args.profile_max_batch_size,
            served_model_name=args.served_model_name,
            target_tensor_parallel_size=args.target_tp,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            distributed_executor_backend=args.distributed_executor_backend,
            moe_backend=args.moe_backend,
            cudagraph_capture_sizes=tuple(args.cudagraph_capture_sizes),
        )
        return
    if args.command == "acceptance":
        collect_acceptance(
            args.root,
            args.target_snapshot,
            args.drafter_snapshot,
            max_k=args.max_k,
            profile_max_batch_size=args.profile_max_batch_size,
            target_tensor_parallel_size=args.target_tp,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            distributed_executor_backend=args.distributed_executor_backend,
            moe_backend=args.moe_backend,
            cudagraph_capture_sizes=tuple(args.cudagraph_capture_sizes),
        )
        return
    payload = assemble_profile(
        args.root,
        model_key=args.model_key,
        target_revision=args.target_revision,
        drafter_revision=args.drafter_revision,
        batch_sizes=validate_batch_sizes(args.batch_sizes),
        k_values=tuple(range(args.max_k + 1)),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        profile_max_batch_size=args.profile_max_batch_size,
        target_tensor_parallel_size=args.target_tp,
        enable_prefix_caching=args.enable_prefix_caching,
        moe_backend=args.moe_backend,
        cudagraph_capture_sizes=tuple(args.cudagraph_capture_sizes),
    )
    _atomic_write(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
