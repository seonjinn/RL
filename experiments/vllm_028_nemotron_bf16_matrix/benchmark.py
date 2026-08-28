#!/usr/bin/env python3
"""vLLM 0.28 benchmark result helpers and executable entry point."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


_EXPERIMENT_ROOT = Path(__file__).resolve().parent
_LEGACY_DRIVER = _EXPERIMENT_ROOT.parent / "vllm_024_dynamicsd" / "benchmark.py"
_DYNAMIC_SCHEDULE = "1:4:5,5:16:3,17:64:2,65:128:1,129:512:0"
_CHECKPOINTS = {
    "super": (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/"
        "snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
    ),
    "ultra": (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/"
        "snapshots/624ba927cfbef0427354998700de3d51173c8c04"
    ),
}


def resolve_k_provenance(method_key: str, batch_size: int) -> dict[str, Any]:
    """Describe K without confusing offered and actively scheduled batches."""
    if method_key == "baseline":
        return {"effective_k": 0, "k_selection_basis": "baseline"}
    if method_key.startswith("mtp_static_k"):
        return {
            "effective_k": int(method_key.removeprefix("mtp_static_k")),
            "k_selection_basis": "static",
        }
    if method_key != "mtp_dynamic_max_k5":
        raise ValueError(f"unsupported method_key={method_key!r}")
    for start, end, k in (
        (1, 4, 5),
        (5, 16, 3),
        (17, 64, 2),
        (65, 128, 1),
        (129, 512, 0),
    ):
        if start <= batch_size <= end:
            return {
                "effective_k": None,
                "requested_batch_schedule_k": k,
                "k_selection_basis": "active_scheduled_batch",
            }
    raise ValueError(f"batch_size={batch_size} is outside the DynamicSD schedule")


def _validate_result_contract(payload: dict[str, Any]) -> None:
    results_path = _EXPERIMENT_ROOT / "results.py"
    spec = importlib.util.spec_from_file_location("vllm028_results", results_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load result validator from {results_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_result_payload(payload)


def _append_common_runtime(argv: list[str], args: dict[str, Any]) -> None:
    isl = int(args["isl"])
    osl = int(args["osl"])
    argv.extend(
        [
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            str(isl + osl + 256),
            "--max-num-seqs",
            str(args["max_num_seqs"]),
            "--max-num-batched-tokens",
            "32768",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--isl",
            str(isl),
            "--osl",
            str(osl),
            "--batch-sizes",
            *(str(value) for value in args["batch_sizes"]),
            "--temperature",
            "1.0",
            "--top-p",
            "1.0",
            "--seed",
            "42",
            "--warmup-repeats",
            "1",
            "--measure-repeats",
            "1",
            "--output",
            str(args["result_path"]),
        ]
    )


def _validate_speculative_config(args: dict[str, Any]) -> None:
    method_key = str(args["method_key"])
    config = args.get("speculative_config")
    if method_key == "baseline":
        if config is not None:
            raise ValueError("baseline speculative config must be null")
        return
    if config is None:
        return
    if config.get("method") != "mtp":
        raise ValueError("speculative config method must be mtp")
    if method_key.startswith("mtp_static_k"):
        expected_k = int(method_key.removeprefix("mtp_static_k"))
        if config.get("num_speculative_tokens") != expected_k or (
            "num_speculative_tokens_per_batch_size" in config
        ):
            raise ValueError("static speculative config is inconsistent with method")
        return
    expected_table = [
        [1, 4, 5],
        [5, 16, 3],
        [17, 64, 2],
        [65, 128, 1],
        [129, 512, 0],
    ]
    if method_key != "mtp_dynamic_max_k5" or (
        config.get("num_speculative_tokens") != 5
        or config.get("num_speculative_tokens_per_batch_size") != expected_table
    ):
        raise ValueError("dynamic speculative config is inconsistent with method")


def build_legacy_benchmark_argv(args: dict[str, Any]) -> list[str]:
    """Map the reviewed v0.28 contract to the proven standalone driver."""
    _validate_speculative_config(args)
    model_key = str(args["model_key"])
    runner_key = str(args["runner_key"])
    if model_key not in _CHECKPOINTS:
        raise ValueError(f"unsupported model_key={model_key!r}")
    if runner_key not in {"mrv1", "mrv2"}:
        raise ValueError(f"unsupported runner_key={runner_key!r}")
    tp = 2 if model_key == "super" else 8
    cudagraph_mode = "PIECEWISE" if runner_key == "mrv1" else "FULL_AND_PIECEWISE"
    argv = [
        "python3",
        str(_LEGACY_DRIVER),
        "--model",
        _CHECKPOINTS[model_key],
        "--tensor-parallel-size",
        str(tp),
        "--pipeline-parallel-size",
        "1",
        "--throughput-gpu-count",
        str(tp),
        "--cudagraph-mode",
        cudagraph_mode,
        "--moe-backend",
        "flashinfer_trtllm",
        "--mamba-backend",
        "flashinfer",
    ]
    if model_key == "super":
        argv.extend(
            [
                "--model-loader-num-threads",
                "48",
                "--mamba-ssm-cache-dtype",
                "float32",
            ]
        )
    else:
        argv.extend(
            [
                "--distributed-executor-backend",
                "ray",
                "--distributed-timeout-seconds",
                "3600",
                "--enable-expert-parallel",
                "--model-loader-num-threads",
                "96",
                "--disable-fuse-allreduce-rms",
                "--mamba-ssm-cache-dtype",
                "float16",
                "--enable-mamba-cache-stochastic-rounding",
                "--mamba-cache-philox-rounds",
                "5",
            ]
        )
    method_key = str(args["method_key"])
    if method_key == "baseline":
        argv.extend(["--mode", "baseline"])
    elif method_key.startswith("mtp_static_k"):
        argv.extend(
            [
                "--mode",
                "mtp_static",
                "--static-k",
                method_key.removeprefix("mtp_static_k"),
            ]
        )
    elif method_key == "mtp_dynamic_max_k5":
        argv.extend(
            [
                "--mode",
                "mtp_dynamic",
                "--dynamic-schedule",
                _DYNAMIC_SCHEDULE,
            ]
        )
    else:
        raise ValueError(f"unsupported method_key={method_key!r}")
    _append_common_runtime(argv, args)
    return argv


def enrich_result_payload(
    payload: dict[str, Any],
    *,
    osl: int,
    repeats: int,
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Add strict token-completion fields and immutable runtime provenance."""
    if payload.get("status") != "complete":
        raise ValueError("benchmark payload must be complete")
    enriched = copy.deepcopy(payload)
    rows = enriched.get("rows")
    if not isinstance(rows, list):
        rows = enriched.get("results")
    if not isinstance(rows, list):
        raise ValueError("benchmark payload must contain rows or results")
    for row in rows:
        batch_size = int(row.get("batch_size", row.get("bs", 0)))
        actual = int(row["output_tokens"])
        expected = batch_size * osl * repeats
        row["expected_output_tokens"] = expected
        row["actual_output_tokens"] = actual
        row["tokens_ok"] = actual == expected
        if not row["tokens_ok"]:
            raise ValueError(
                f"output tokens mismatch: actual={actual}, expected={expected}"
            )
    enriched["runtime_provenance"] = copy.deepcopy(runtime_provenance)
    return enriched


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("super", "ultra"), required=True)
    parser.add_argument("--runner-key", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--distributed-executor-backend", choices=("ray",))
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--speculative-config-json", required=True)
    parser.add_argument("--container-digest", default="")
    parser.add_argument("--container-artifact", default="")
    parser.add_argument("--container-artifact-sha256", default="")
    parser.add_argument("--vllm-base-commit", default="")
    parser.add_argument("--patched-vllm-head", default="")
    parser.add_argument("--patchset-manifest-sha256", default="")
    parser.add_argument("--harness-commit", default="")
    parser.add_argument("--harness-manifest-sha256", default="")
    parser.add_argument("--ray-version", default="")
    parser.add_argument("--ray-bundle", default="")
    parser.add_argument("--ray-bundle-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parsed = _build_parser().parse_args()
    if not parsed.ignore_eos:
        raise ValueError("exact-work benchmark requires --ignore-eos")
    config = json.loads(parsed.speculative_config_json)
    expected_tp = 2 if parsed.model_key == "super" else 8
    if parsed.tensor_parallel_size != expected_tp:
        raise ValueError("tensor parallel size does not match pinned topology")
    if parsed.enable_expert_parallel != (parsed.model_key == "ultra"):
        raise ValueError("expert parallel setting does not match pinned topology")
    if (parsed.distributed_executor_backend == "ray") != (parsed.model_key == "ultra"):
        raise ValueError("distributed executor backend does not match pinned topology")
    raw_output = parsed.output.with_name(f"{parsed.output.stem}.raw.json")
    temporary_output = parsed.output.with_name(
        f"{parsed.output.name}.tmp.{os.getpid()}"
    )
    args = {
        "model_key": parsed.model_key,
        "runner_key": parsed.runner_key,
        "method_key": parsed.method_key,
        "isl": parsed.isl,
        "osl": parsed.osl,
        "batch_sizes": [parsed.batch_size],
        "max_num_seqs": parsed.max_num_seqs,
        "speculative_config": config,
        "result_path": str(raw_output),
    }
    subprocess.run(build_legacy_benchmark_argv(args), check=True)
    payload = json.loads(raw_output.read_text(encoding="utf-8"))
    enriched = enrich_result_payload(
        payload,
        osl=parsed.osl,
        repeats=1,
        runtime_provenance={
            "vllm_version": "0.28.0",
            "vllm_branch": "release",
            "vllm_commit": "2cf0a69",
            "container_digest": parsed.container_digest,
            "container_artifact": parsed.container_artifact,
            "container_artifact_sha256": parsed.container_artifact_sha256,
            "vllm_base_commit": parsed.vllm_base_commit or None,
            "patched_vllm_head": parsed.patched_vllm_head or None,
            "patchset_manifest_sha256": parsed.patchset_manifest_sha256 or None,
            "harness_commit": parsed.harness_commit,
            "harness_manifest_sha256": parsed.harness_manifest_sha256,
            "ray_version": parsed.ray_version or None,
            "ray_bundle": parsed.ray_bundle or None,
            "ray_bundle_sha256": parsed.ray_bundle_sha256 or None,
        },
    )
    rows = enriched.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("one benchmark invocation must produce exactly one result row")
    enriched["config"].update(
        {
            "model_key": parsed.model_key,
            "method_key": parsed.method_key,
            "runner_key": parsed.runner_key,
            "isl": parsed.isl,
            "osl": parsed.osl,
            "batch_size": parsed.batch_size,
            **resolve_k_provenance(parsed.method_key, parsed.batch_size),
        }
    )
    enriched["summary"] = {
        "tokens_ok": bool(rows[0]["tokens_ok"]),
        "spec_decode_metrics": rows[0].get("spec_decode_metrics", {}),
    }
    if parsed.method_key != "baseline":
        _validate_result_contract(enriched)
    temporary_output.write_text(
        json.dumps(enriched, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(parsed.output)


if __name__ == "__main__":
    main()
