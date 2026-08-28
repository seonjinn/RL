#!/usr/bin/env python3
"""Build validated CSV, JSON, and HTML for the patched MRV2 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple


PACKAGE_ROOT = Path(__file__).resolve().parent
PATCHSET_SHA256 = "238e2ffcc14d2fb2f0fc07c419004820efa9a3cfd284c3a41515e84e39aecc25"
VLLM_BASE_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
PATCHED_VLLM_HEAD = "bf0719d13bbc74da8af86d2b326f7b16ba4f7462"
CONTAINER_SHA256 = "5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066"
FIXED_K_CONTAINER_DIGEST = (
    "sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
)
MRV1_DYNAMIC_HARNESS_PAIR_EXCEPTIONS = frozenset(
    {
        (
            model,
            10000,
            1000,
            512,
            "3fb707333928a8841713a19ae0917e26cf59a0a0",
            "f0dd8af3110820c708de2ce7b0720970f4c8ef8c",
        )
        for model in ("super", "ultra")
    }
)
OLD_HARNESS = (
    "85cfe1a8a2f7c97684e65e5428a679cab7058142",
    "86012387865c84b68150f0aa2e758dbb7b5ccc75be07ec15dc1c9663d6b87147",
)
ACTIVE_BATCH_HARNESS = (
    "f3a842c0ad3c91f97e37b4f58ef84bd9148e281a",
    "d182f2e26920c6c68e1961d13cbfec60340cf4e3394f65ac3d560589146fc873",
)
DYNAMIC_SCHEDULE = ((1, 4, 5), (5, 16, 3), (17, 64, 2), (65, 128, 1), (129, 512, 0))
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 128, 512)
SHAPES = (("isl1k_osl10k", 1000, 10000), ("isl10k_osl1k", 10000, 1000))
METHODS = ("baseline", "mtp_dynamic_max_k5")
FIXED_K_METHODS = tuple(f"mtp_static_k{k}" for k in range(1, 6))
SCHEDULER_CODE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{VLLM_BASE_COMMIT}/vllm/v1/core/sched/scheduler.py#L1254"
)
SCHEDULE_LOOKUP_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{VLLM_BASE_COMMIT}/vllm/v1/spec_decode/dynamic/utils.py#L77"
)


class ResultKey(NamedTuple):
    model: str
    isl: int
    osl: int
    method: str
    batch_size: int


def expected_keys() -> list[ResultKey]:
    return [
        ResultKey(model, isl, osl, method, batch_size)
        for model in ("super", "ultra")
        for _, isl, osl in SHAPES
        for batch_size in BATCH_SIZES
        for method in METHODS
    ]


def _shape_key(isl: int, osl: int) -> str:
    for key, expected_isl, expected_osl in SHAPES:
        if (isl, osl) == (expected_isl, expected_osl):
            return key
    raise ValueError(f"unsupported ISL/OSL shape: {isl}/{osl}")


def _requested_k(batch_size: int) -> int:
    return next(k for start, end, k in DYNAMIC_SCHEDULE if start <= batch_size <= end)


def load_cells(
    result_root: Path, keys: Iterable[ResultKey]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load one canonical result and graph-evidence object for every key."""
    cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key in keys:
        leaf = (
            result_root
            / key.model
            / _shape_key(key.isl, key.osl)
            / key.method
            / f"bs{key.batch_size}"
        )
        results = sorted(leaf.glob("job-*/result.json"))
        if len(results) != 1:
            raise ValueError(
                f"expected exactly one canonical result for {key}, found {len(results)}"
            )
        result_path = results[0]
        evidence_path = result_path.with_name("cuda_graph_evidence.json")
        if not evidence_path.is_file():
            raise ValueError(f"missing CUDA Graph evidence: {evidence_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        observed = ResultKey(
            str(config.get("model_key")),
            int(config.get("isl", -1)),
            int(config.get("osl", -1)),
            str(config.get("method_key")),
            int(config.get("batch_size", -1)),
        )
        if observed != key:
            raise ValueError(
                f"result identity mismatch: expected {key}, got {observed}"
            )
        job_id = str(
            payload.get("runtime", {}).get("environment", {}).get("SLURM_JOB_ID", "")
        )
        if result_path.parent.name != f"job-{job_id}":
            raise ValueError("result path is not bound to its SLURM job ID")
        cells.append((payload, evidence))
    return cells


def _completed_capture(rows: Any, *, required_text: tuple[str, ...]) -> bool:
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        line = row.get("line")
        count = row.get("count")
        if (
            isinstance(line, str)
            and isinstance(count, int)
            and count > 0
            and all(fragment in line for fragment in required_text)
            and f"{count}/{count}" in line
        ):
            return True
    return False


def _validate_graph(method: str, evidence: dict[str, Any]) -> tuple[bool, bool]:
    if evidence.get("method_key") != method:
        raise ValueError("CUDA Graph evidence method mismatch")
    piecewise = evidence.get("piecewise_completed")
    full = evidence.get("full_completed")
    decode = evidence.get("drafter_decode_completed")
    target_piecewise = _completed_capture(
        piecewise, required_text=("Capturing CUDA graphs (PIECEWISE)",)
    )
    target_full = _completed_capture(
        full, required_text=("Capturing CUDA graphs (FULL)",)
    )
    if not target_piecewise or not target_full:
        raise ValueError("missing completed target PIECEWISE/FULL CUDA Graph capture")
    if method == "baseline":
        return True, False
    drafter_prefill_piecewise = _completed_capture(
        piecewise,
        required_text=("Capturing prefill CUDA graphs (PIECEWISE)",),
    )
    if not drafter_prefill_piecewise:
        raise ValueError(
            "missing completed drafter prefill PIECEWISE CUDA Graph capture"
        )
    drafter_prefill_full = _completed_capture(
        full, required_text=("Capturing prefill CUDA graphs (FULL)",)
    )
    if not drafter_prefill_full:
        raise ValueError("missing completed drafter prefill FULL CUDA Graph capture")
    drafter_verified = _completed_capture(
        decode, required_text=("Capturing decode CUDA graphs (FULL)",)
    )
    if not drafter_verified:
        raise ValueError("missing completed drafter decode FULL CUDA Graph capture")
    return True, drafter_verified


def _validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("result is not a complete schema-version-1 payload")
    config = payload.get("config")
    results = payload.get("results")
    if (
        not isinstance(config, dict)
        or not isinstance(results, list)
        or len(results) != 1
    ):
        raise ValueError("result must contain one config and one measured row")
    row = results[0]
    batch_size = int(config.get("batch_size", -1))
    expected_tokens = batch_size * int(config.get("osl", -1))
    if (
        row.get("tokens_ok") is not True
        or int(row.get("expected_output_tokens", -1)) != expected_tokens
        or int(row.get("actual_output_tokens", -1)) != expected_tokens
    ):
        raise ValueError("result failed exact-token validation")
    if (
        config.get("runner_key") != "mrv2"
        or config.get("cudagraph_mode") != "FULL_AND_PIECEWISE"
        or config.get("enforce_eager") is not False
        or config.get("dtype") != "bfloat16"
        or config.get("kv_cache_dtype") != "fp8"
    ):
        raise ValueError("runner, CUDA Graph, or precision configuration drift")
    fixed_runtime = {
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "warmup_repeats": 1,
        "measure_repeats": 1,
        "max_num_seqs": 512,
        "max_num_batched_tokens": 32768,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.9,
        "disable_custom_all_reduce": False,
        "attention_backend": "auto",
        "moe_backend": "flashinfer_trtllm",
        "mamba_backend": "flashinfer",
    }
    if any(config.get(key) != value for key, value in fixed_runtime.items()):
        raise ValueError("performance-affecting runtime configuration drift")
    method = str(config.get("method_key"))
    if method == "baseline":
        if (
            config.get("speculative_config") is not None
            or config.get("effective_k") != 0
        ):
            raise ValueError("baseline speculative configuration drift")
    elif method == "mtp_dynamic_max_k5":
        expected_config = {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [
                list(item) for item in DYNAMIC_SCHEDULE
            ],
        }
        if (
            config.get("speculative_config") != expected_config
            or config.get("effective_k") is not None
            or config.get("requested_batch_schedule_k") != _requested_k(batch_size)
            or config.get("k_selection_basis") != "active_scheduled_batch"
        ):
            raise ValueError("DynamicMTP schedule or K-selection provenance drift")
    else:
        raise ValueError(f"unsupported method: {method}")
    model = str(config.get("model_key"))
    expected_model = {
        "super": (
            2,
            False,
            "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "d51eab0d1f979ebc26b546e634a04f450d99158e",
        ),
        "ultra": (
            8,
            True,
            "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            "624ba927cfbef0427354998700de3d51173c8c04",
        ),
    }.get(model)
    if expected_model is None:
        raise ValueError(f"unsupported model: {model}")
    tp, expert_parallel, checkpoint, revision = expected_model
    model_path = str(config.get("model", ""))
    if checkpoint not in model_path or not model_path.endswith(
        f"/snapshots/{revision}"
    ):
        raise ValueError("checkpoint revision drift")
    if (
        int(config.get("tensor_parallel_size", -1)) != tp
        or int(config.get("pipeline_parallel_size", -1)) != 1
        or int(config.get("engine_gpus", -1)) != tp
        or int(config.get("total_gpus", -1)) != tp
        or bool(config.get("enable_expert_parallel")) is not expert_parallel
    ):
        raise ValueError("parallelism configuration drift")
    runtime = payload.get("runtime", {})
    provenance = payload.get("runtime_provenance", {})
    if runtime.get("vllm_version") != "0.28.0":
        raise ValueError("vLLM version drift")
    expected_provenance = {
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_vllm_head": PATCHED_VLLM_HEAD,
        "patchset_manifest_sha256": PATCHSET_SHA256,
        "container_artifact_sha256": CONTAINER_SHA256,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("patched runtime provenance drift")
    expected_ray = "2.48.0" if model == "ultra" else None
    if provenance.get("ray_version") != expected_ray:
        raise ValueError("Ray runtime provenance drift")
    harness = (
        provenance.get("harness_commit"),
        provenance.get("harness_manifest_sha256"),
    )
    if harness not in {OLD_HARNESS, ACTIVE_BATCH_HARNESS}:
        raise ValueError("unrecognized harness provenance")
    job_id = str(runtime.get("environment", {}).get("SLURM_JOB_ID", ""))
    if not job_id:
        raise ValueError("missing SLURM job ID")
    return config, row


def _comparison_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    config = payload["config"]
    provenance = payload["runtime_provenance"]
    method_fields = {
        "effective_k",
        "k_selection_basis",
        "method_key",
        "mode",
        "requested_batch_schedule_k",
        "speculative_config",
    }
    matched_config = {
        key: value for key, value in config.items() if key not in method_fields
    }
    return (
        json.dumps(matched_config, separators=(",", ":"), sort_keys=True),
        str(provenance["vllm_base_commit"]),
        str(provenance["patched_vllm_head"]),
        str(provenance["patchset_manifest_sha256"]),
        str(provenance["container_artifact_sha256"]),
        provenance.get("ray_version"),
        provenance.get("ray_bundle_sha256"),
    )


def _harness_provenance(payload: dict[str, Any]) -> tuple[Any, Any]:
    provenance = payload["runtime_provenance"]
    return (
        provenance.get("harness_commit"),
        provenance.get("harness_manifest_sha256"),
    )


def _validate_harness_pair(baseline: dict[str, Any], dynamic: dict[str, Any]) -> bool:
    baseline_harness = _harness_provenance(baseline)
    dynamic_harness = _harness_provenance(dynamic)
    if baseline_harness == dynamic_harness:
        return False
    config = dynamic["config"]
    allowed_validation_only_rerun = (
        int(config["batch_size"]) == 512
        and baseline_harness == OLD_HARNESS
        and dynamic_harness == ACTIVE_BATCH_HARNESS
    )
    if not allowed_validation_only_rerun:
        raise ValueError("baseline and DynamicMTP harness provenance mismatch")
    return True


def normalize_rows(
    cells: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Validate cells, join exact baselines, and calculate report metrics."""
    validated: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_jobs: set[str] = set()
    for payload, evidence in cells:
        config, row = _validate_payload(payload)
        job_id = str(payload["runtime"]["environment"]["SLURM_JOB_ID"])
        if job_id in seen_jobs:
            raise ValueError(f"duplicate SLURM job ID: {job_id}")
        seen_jobs.add(job_id)
        _validate_graph(str(config["method_key"]), evidence)
        if config["method_key"] == "baseline":
            baselines[_comparison_signature(payload)] = payload
        validated.append((payload, evidence, config, row))

    normalized: list[dict[str, Any]] = []
    for payload, evidence, config, row in validated:
        signature = _comparison_signature(payload)
        baseline_payload = baselines.get(signature)
        if baseline_payload is None:
            raise ValueError("missing matched baseline for exact runtime signature")
        baseline = float(baseline_payload["results"][0]["output_tok_s_per_gpu"])
        method = str(config["method_key"])
        metrics = row.get("spec_decode_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        dynamic = method == "mtp_dynamic_max_k5"
        harness_pair_exception = (
            _validate_harness_pair(baseline_payload, payload) if dynamic else False
        )
        drafts = float(metrics.get("num_drafts", 0.0))
        draft_tokens = float(metrics.get("num_draft_tokens", 0.0))
        if dynamic and drafts <= 0:
            raise ValueError("DynamicMTP row has no draft iterations")
        graph_verified, drafter_verified = _validate_graph(method, evidence)
        provenance = payload["runtime_provenance"]
        runtime = payload["runtime"]
        tok_s_gpu = float(row["output_tok_s_per_gpu"])
        normalized.append(
            {
                "model": str(config["model_key"]),
                "isl": int(config["isl"]),
                "osl": int(config["osl"]),
                "concurrency": int(config["batch_size"]),
                "method": method,
                "requested_batch_schedule_k": _requested_k(int(config["batch_size"]))
                if dynamic
                else 0,
                "k_selection_basis": "active_scheduled_batch"
                if dynamic
                else "baseline",
                "mean_draft_width": draft_tokens / drafts if dynamic else None,
                "output_tok_s": float(row["output_tok_s"]),
                "tok_s_gpu": tok_s_gpu,
                "baseline_tok_s_gpu": baseline,
                "throughput_speedup": tok_s_gpu / baseline,
                "latency_s": float(row["latency_s"]),
                "acceptance_rate": float(metrics["acceptance_rate"])
                if dynamic
                else None,
                "mean_accepted_length": float(metrics["mean_acceptance_length"])
                if dynamic
                else None,
                "num_drafts": drafts if dynamic else None,
                "num_draft_tokens": draft_tokens if dynamic else None,
                "num_accepted_tokens": float(metrics["num_accepted_tokens"])
                if dynamic
                else None,
                "actual_output_tokens": int(row["actual_output_tokens"]),
                "expected_output_tokens": int(row["expected_output_tokens"]),
                "tokens_ok": True,
                "weights": "BF16",
                "kv_cache": "FP8",
                "tp": int(config["tensor_parallel_size"]),
                "pp": int(config["pipeline_parallel_size"]),
                "dp": 1,
                "expert_parallel": bool(config["enable_expert_parallel"]),
                "nodes": 1 if config["model_key"] == "super" else 2,
                "runner": "MRV2",
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cuda_graph_verified": graph_verified,
                "drafter_decode_full_verified": drafter_verified if dynamic else None,
                "piecewise_completed_entries": len(evidence["piecewise_completed"]),
                "full_completed_entries": len(evidence["full_completed"]),
                "drafter_decode_completed_entries": len(
                    evidence["drafter_decode_completed"]
                ),
                "temperature": float(config["temperature"]),
                "top_p": float(config["top_p"]),
                "warmup_repeats": int(config["warmup_repeats"]),
                "measure_repeats": int(config["measure_repeats"]),
                "max_num_seqs": int(config["max_num_seqs"]),
                "max_num_batched_tokens": int(config["max_num_batched_tokens"]),
                "prefix_caching": bool(config["enable_prefix_caching"]),
                "chunked_prefill": bool(config["enable_chunked_prefill"]),
                "job_id": str(runtime["environment"]["SLURM_JOB_ID"]),
                "vllm_version": str(runtime["vllm_version"]),
                "vllm_base_commit": str(provenance["vllm_base_commit"]),
                "patched_vllm_head": str(provenance["patched_vllm_head"]),
                "patchset_sha256": str(provenance["patchset_manifest_sha256"]),
                "container_sha256": str(provenance["container_artifact_sha256"]),
                "ray_version": provenance.get("ray_version"),
                "harness_commit": str(provenance["harness_commit"]),
                "harness_manifest_sha256": str(provenance["harness_manifest_sha256"]),
                "baseline_harness_commit": str(
                    baseline_payload["runtime_provenance"]["harness_commit"]
                ),
                "harness_pair_exception": harness_pair_exception,
                "result_quality": "preliminary_single_repeat",
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["model"],
            item["isl"],
            item["osl"],
            item["concurrency"],
            item["method"],
        ),
    )


def build_source_manifest(
    result_root: Path, keys: Iterable[ResultKey]
) -> dict[str, Any]:
    """Hash the curated result/evidence pair for every canonical matrix cell."""
    files: list[dict[str, Any]] = []
    for key in keys:
        leaf = (
            result_root
            / key.model
            / _shape_key(key.isl, key.osl)
            / key.method
            / f"bs{key.batch_size}"
        )
        results = sorted(leaf.glob("job-*/result.json"))
        if len(results) != 1:
            raise ValueError(f"manifest expected one result for {key}")
        result_path = results[0]
        evidence_path = result_path.with_name("cuda_graph_evidence.json")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        job_id = str(payload["runtime"]["environment"]["SLURM_JOB_ID"])
        for kind, path in (
            ("result", result_path),
            ("cuda_graph_evidence", evidence_path),
        ):
            files.append(
                {
                    "key": list(key),
                    "job_id": job_id,
                    "kind": kind,
                    "path": path.relative_to(result_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {
        "schema_version": 1,
        "matrix_cells": 64,
        "files": files,
    }


def _csv_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _is_full_git_commit(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(
        character in "0123456789abcdef" for character in text
    )


def relative_href(output_html: Path, target: Path) -> str:
    """Return a portable link from the report location to a generated/input artifact."""
    return Path(os.path.relpath(target, start=output_html.parent)).as_posix()


def _validate_mrv1_row(row: dict[str, Any]) -> tuple[str, int, int, int]:
    model = str(row["model"])
    isl = int(row["isl"])
    osl = int(row["osl"])
    concurrency = int(row["batch_size"])
    expected_tokens = concurrency * osl
    expected_tp = 2 if model == "super" else 8
    expected_ep = model == "ultra"
    if (
        not _csv_true(row.get("tokens_ok"))
        or int(row.get("actual_output_tokens", -1)) != expected_tokens
        or int(row.get("expected_output_tokens", -1)) != expected_tokens
        or str(row.get("weight_dtype")) != "bfloat16"
        or str(row.get("kv_cache_dtype")) != "fp8"
        or str(row.get("runner")) != "mrv1"
        or str(row.get("vllm_version")) != "0.28.0"
        or str(row.get("cudagraph_mode")) != "PIECEWISE"
        or _csv_true(row.get("enforce_eager"))
        or not _csv_true(row.get("cuda_graph_verified"))
    ):
        raise ValueError(
            "MRV1 row failed runtime, exact output tokens, or graph validation"
        )
    if (
        int(row.get("tensor_parallel_size", -1)) != expected_tp
        or _csv_true(row.get("expert_parallel")) != expected_ep
        or str(row.get("vllm_commit")) != VLLM_BASE_COMMIT[:7]
        or str(row.get("container_digest")) != FIXED_K_CONTAINER_DIGEST
        or not _is_full_git_commit(row.get("harness_commit"))
        or int(row.get("cuda_graph_capture_completed", -1)) != 83
        or int(row.get("cuda_graph_capture_total", -1)) != 83
    ):
        raise ValueError(
            "MRV1 row failed topology, provenance, or graph capture validation"
        )
    return model, isl, osl, concurrency


def normalize_fixed_k_rows(
    raw_rows: Iterable[dict[str, Any]], *, require_complete_matrix: bool = True
) -> list[dict[str, Any]]:
    """Validate the MRV1 fixed-K ladder and recompute matched speedups."""
    relevant = [
        row
        for row in raw_rows
        if str(row.get("method")) in {"baseline", *FIXED_K_METHODS}
        and str(row.get("model")) in {"super", "ultra"}
        and (int(row.get("isl", -1)), int(row.get("osl", -1)))
        in {(1000, 10000), (10000, 1000)}
        and int(row.get("batch_size", -1)) in BATCH_SIZES
    ]
    baselines: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    static_rows: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for row in relevant:
        model, isl, osl, concurrency = _validate_mrv1_row(row)
        base_key = (model, isl, osl, concurrency)
        method = str(row["method"])
        if method == "baseline":
            if base_key in baselines:
                raise ValueError(f"duplicate fixed-K baseline: {base_key}")
            baselines[base_key] = row
            continue
        k = int(method.removeprefix("mtp_static_k"))
        key = (*base_key, k)
        if key in static_rows:
            raise ValueError(f"duplicate fixed-K row: {key}")
        static_rows[key] = row

    if require_complete_matrix and (len(baselines) != 32 or len(static_rows) != 160):
        raise ValueError("fixed-K matrix must contain 32 baselines and 160 K1-K5 rows")

    normalized: list[dict[str, Any]] = []
    for key, row in static_rows.items():
        model, isl, osl, concurrency, k = key
        baseline_row = baselines.get((model, isl, osl, concurrency))
        if baseline_row is None:
            raise ValueError(f"missing matched fixed-K baseline: {key[:-1]}")
        tok_s_gpu = float(row["output_tok_s_per_gpu"])
        baseline_tok_s_gpu = float(baseline_row["output_tok_s_per_gpu"])
        speedup = tok_s_gpu / baseline_tok_s_gpu
        reported = float(row["speedup_vs_baseline"])
        if not math.isclose(speedup, reported, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"fixed-K speedup drift: {key}")
        normalized.append(
            {
                "model": model,
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "k": k,
                "tok_s_gpu": tok_s_gpu,
                "baseline_tok_s_gpu": baseline_tok_s_gpu,
                "throughput_speedup": speedup,
                "acceptance_rate": (
                    float(row["acceptance_rate"])
                    if str(row.get("acceptance_rate", ""))
                    else None
                ),
                "mean_accepted_length": (
                    float(row["mean_acceptance_length"])
                    if str(row.get("mean_acceptance_length", ""))
                    else None
                ),
                "job_id": str(row.get("job_id", "")),
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            row["model"],
            row["isl"],
            row["osl"],
            row["concurrency"],
            row["k"],
        ),
    )


def normalize_mrv1_dynamic_rows(
    raw_rows: Iterable[dict[str, Any]], *, require_complete_matrix: bool = True
) -> list[dict[str, Any]]:
    """Validate MRV1 DynamicSD rows and recompute their matched speedups."""
    relevant = [
        row
        for row in raw_rows
        if str(row.get("method")) in {"baseline", "mtp_dynamic_max_k5"}
        and str(row.get("model")) in {"super", "ultra"}
        and (int(row.get("isl", -1)), int(row.get("osl", -1)))
        in {(1000, 10000), (10000, 1000)}
        and int(row.get("batch_size", -1)) in BATCH_SIZES
    ]
    baselines: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    dynamic_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in relevant:
        model, isl, osl, concurrency = _validate_mrv1_row(row)
        key = (model, isl, osl, concurrency)
        if str(row["method"]) == "baseline":
            if key in baselines:
                raise ValueError(f"duplicate MRV1 DynamicSD baseline: {key}")
            baselines[key] = row
            continue
        if (
            int(row.get("requested_batch_schedule_k", -1)) != _requested_k(concurrency)
            or str(row.get("k_selection_basis")) != "active_scheduled_batch"
        ):
            raise ValueError(f"MRV1 DynamicSD schedule drift: {key}")
        if key in dynamic_rows:
            raise ValueError(f"duplicate MRV1 DynamicSD row: {key}")
        dynamic_rows[key] = row

    if require_complete_matrix and (len(baselines) != 32 or len(dynamic_rows) != 32):
        raise ValueError("MRV1 matrix must contain 32 baselines and 32 DynamicSD rows")

    normalized: list[dict[str, Any]] = []
    for key, row in dynamic_rows.items():
        baseline_row = baselines.get(key)
        if baseline_row is None:
            raise ValueError(f"missing matched MRV1 DynamicSD baseline: {key}")
        baseline_harness = str(baseline_row["harness_commit"])
        dynamic_harness = str(row["harness_commit"])
        harness_pair = (*key, baseline_harness, dynamic_harness)
        if (
            baseline_harness != dynamic_harness
            and harness_pair not in MRV1_DYNAMIC_HARNESS_PAIR_EXCEPTIONS
        ):
            raise ValueError(f"MRV1 DynamicSD harness provenance mismatch: {key}")
        tok_s_gpu = float(row["output_tok_s_per_gpu"])
        baseline_tok_s_gpu = float(baseline_row["output_tok_s_per_gpu"])
        speedup = tok_s_gpu / baseline_tok_s_gpu
        if not math.isclose(
            speedup,
            float(row["speedup_vs_baseline"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"MRV1 DynamicSD speedup drift: {key}")
        normalized.append(
            {
                "model": key[0],
                "isl": key[1],
                "osl": key[2],
                "concurrency": key[3],
                "tok_s_gpu": tok_s_gpu,
                "baseline_tok_s_gpu": baseline_tok_s_gpu,
                "throughput_speedup": speedup,
                "acceptance_rate": float(row["acceptance_rate"]),
                "mean_accepted_length": float(row["mean_acceptance_length"]),
                "job_id": str(row.get("job_id", "")),
            }
        )
    return sorted(
        normalized,
        key=lambda row: (row["model"], row["isl"], row["osl"], row["concurrency"]),
    )


def load_fixed_k_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return normalize_fixed_k_rows(csv.DictReader(stream))


def load_mrv1_dynamic_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return normalize_mrv1_dynamic_rows(csv.DictReader(stream))


def best_fixed_k(rows: Iterable[dict[str, Any]]) -> tuple[int, float, float]:
    candidates = list(rows)
    if not candidates:
        raise ValueError("no fixed-K candidates")
    best = max(candidates, key=lambda row: float(row["throughput_speedup"]))
    return int(best["k"]), float(best["throughput_speedup"]), float(best["tok_s_gpu"])


def render_html(
    rows: list[dict[str, Any]],
    *,
    fixed_k_rows: list[dict[str, Any]],
    mrv1_dynamic_rows: list[dict[str, Any]],
    csv_href: str,
    fixed_k_csv_href: str,
    source_manifest_href: str,
) -> str:
    dynamic_rows = [row for row in rows if row["method"] == "mtp_dynamic_max_k5"]
    wins = sum(float(row["throughput_speedup"]) > 1.0 for row in dynamic_rows)
    best = max(dynamic_rows, key=lambda row: float(row["throughput_speedup"]))
    worst = min(dynamic_rows, key=lambda row: float(row["throughput_speedup"]))
    baseline_by_key = {
        (row["model"], row["isl"], row["osl"], row["concurrency"]): row
        for row in rows
        if row["method"] == "baseline"
    }
    mrv1_dynamic_by_key = {
        (row["model"], row["isl"], row["osl"], row["concurrency"]): row
        for row in mrv1_dynamic_rows
    }
    shape_groups = [
        (model, isl, osl)
        for model in ("super", "ultra")
        for isl, osl in ((1000, 10000), (10000, 1000))
    ]

    dynamic_summary_rows = []
    fixed_tables = []
    for model, isl, osl in shape_groups:
        per_shape_dynamic = {
            int(row["concurrency"]): row
            for row in dynamic_rows
            if (row["model"], row["isl"], row["osl"]) == (model, isl, osl)
        }
        if per_shape_dynamic:
            speedup_cells = "".join(
                f'<td class="{"win" if float(per_shape_dynamic[c]["throughput_speedup"]) > 1 else "loss"}">'
                f"{float(per_shape_dynamic[c]['throughput_speedup']):.3f}×</td>"
                for c in BATCH_SIZES
                if c in per_shape_dynamic
            )
            dynamic_summary_rows.append(
                f"<tr><td>{model.title()}</td><td>{isl}/{osl}</td>{speedup_cells}</tr>"
            )

        grouped_fixed = {
            int(c): [
                row
                for row in fixed_k_rows
                if (row["model"], row["isl"], row["osl"], row["concurrency"])
                == (model, isl, osl, c)
            ]
            for c in BATCH_SIZES
        }
        if not any(grouped_fixed.values()):
            continue
        fixed_body = []
        for concurrency, candidates in grouped_fixed.items():
            if not candidates:
                continue
            by_k = {int(row["k"]): row for row in candidates}
            k, speedup, _ = best_fixed_k(candidates)
            comparison_cells = []
            for comparison in (
                mrv1_dynamic_by_key.get((model, isl, osl, concurrency)),
                per_shape_dynamic.get(concurrency),
            ):
                if comparison is None:
                    comparison_cells.append('<td class="muted">N/A</td>')
                    continue
                comparison_speedup = float(comparison["throughput_speedup"])
                comparison_class = "win" if comparison_speedup > 1 else "loss"
                comparison_cells.append(
                    f'<td class="{comparison_class}">{comparison_speedup:.2f}×</td>'
                )
            fixed_body.append(
                "<tr>"
                f"<td>{concurrency}</td>"
                + "".join(
                    f'<td class="{"win" if float(by_k[candidate_k]["throughput_speedup"]) > 1 else "loss"}">'
                    f"{float(by_k[candidate_k]['throughput_speedup']):.2f}×</td>"
                    for candidate_k in range(1, 6)
                )
                + "".join(comparison_cells)
                + f"<td><strong>K{k} ({speedup:.2f}×)</strong></td></tr>"
            )
        fixed_tables.append(
            f"<details><summary>{model.title()} · {isl}/{osl}</summary>"
            '<div class="scroll"><table><thead><tr><th>C</th><th>K1</th><th>K2</th>'
            "<th>K3</th><th>K4</th><th>K5</th><th>DynamicSD (MRV1)</th>"
            "<th>DynamicSD (MRV2)</th><th>Best fixed</th></tr></thead>"
            f"<tbody>{''.join(fixed_body)}</tbody></table></div></details>"
        )

    compact_pairs = []
    for dynamic in dynamic_rows:
        key = (
            dynamic["model"],
            dynamic["isl"],
            dynamic["osl"],
            dynamic["concurrency"],
        )
        baseline = baseline_by_key[key]
        compact_pairs.append(
            [
                dynamic["model"],
                f"{dynamic['isl']}/{dynamic['osl']}",
                dynamic["concurrency"],
                round(float(baseline["tok_s_gpu"]), 3),
                round(float(dynamic["tok_s_gpu"]), 3),
                round(float(dynamic["throughput_speedup"]), 3),
                dynamic["requested_batch_schedule_k"],
                round(float(dynamic["mean_draft_width"]), 3),
                round(float(dynamic["acceptance_rate"]), 4),
                round(float(dynamic["mean_accepted_length"]), 3),
                baseline["job_id"],
                dynamic["job_id"],
            ]
        )
    embedded = json.dumps(compact_pairs, separators=(",", ":")).replace("</", "<\\/")
    github_schedule = html.escape(SCHEDULER_CODE_URL, quote=True)
    github_lookup = html.escape(SCHEDULE_LOOKUP_URL, quote=True)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron-3 BF16 DynamicSD schedule와 MTP 성능</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#9da7b3;--line:#30363d;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--amber:#d29922;--gray:#6e7681;--code:#21262d}}
@media(prefers-color-scheme:light){{:root{{--bg:#f6f8fa;--panel:#fff;--ink:#1f2328;--muted:#59636e;--line:#d0d7de;--green:#1a7f37;--red:#cf222e;--blue:#0969da;--amber:#9a6700;--gray:#6e7781;--code:#eff2f5}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:28px 18px 56px}}h1{{font-size:30px;line-height:1.2;margin:0 0 8px}}h2{{font-size:21px;margin:0 0 12px}}h3{{font-size:16px;margin:18px 0 8px}}p{{margin:8px 0;color:var(--muted)}}a{{color:var(--blue)}}code{{background:var(--code);padding:2px 5px;border-radius:5px}}section,.card,details{{background:var(--panel);border:1px solid var(--line);border-radius:10px}}section{{padding:18px;margin-top:16px}}.sub{{font-size:12px}}.lead{{font-size:15px;color:var(--ink)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}}.card{{padding:11px}}.label{{font-size:11px;color:var(--muted);text-transform:uppercase}}.metric{{font-size:20px;font-weight:750}}.win{{color:var(--green)}}.loss{{color:var(--red)}}.warn{{border-left:4px solid var(--amber);padding:10px 12px;background:color-mix(in srgb,var(--amber) 9%,var(--panel))}}.hl{{border-left:4px solid var(--green);padding:11px 13px;background:color-mix(in srgb,var(--green) 8%,var(--panel));margin-top:14px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:6px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:var(--code)}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}details{{padding:9px 11px;margin:8px 0}}summary{{cursor:pointer;font-weight:700}}.controls{{display:flex;gap:8px;margin:8px 0}}select{{background:var(--panel);color:var(--ink);border:1px solid var(--line);padding:6px;border-radius:6px}}svg{{width:100%;height:auto;margin:12px 0}}svg text{{fill:var(--ink);font-family:system-ui,sans-serif}}svg .muted{{fill:var(--muted)}}svg .code-link{{fill:var(--blue);text-decoration:underline}}.config{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.config p{{margin:4px 0}}@media(max-width:760px){{.cards,.config{{grid-template-columns:1fr 1fr}}h1{{font-size:25px}}}}
</style></head><body><main>
<h1>DynamicSD는 어떻게 K를 고르고, 언제 MTP가 빨라졌나?</h1>
<p class="sub">vLLM <code>0.28.0</code> base <code>{VLLM_BASE_COMMIT[:12]}</code>, patched MRV2 <code>{PATCHED_VLLM_HEAD[:12]}</code>. Code: <a href="{github_schedule}">scheduler.py@{VLLM_BASE_COMMIT[:7]} L1254</a> · Evidence: 2026-08-28 canonical matrix.</p>
<p class="lead">Dynamic speculative decoding(DynamicSD)은 매 scheduler step마다 speculative depth K를 다시 고른다. 입력은 사용자가 보낸 전체 concurrency가 아니라 그 step에 실제로 schedule된 active batch 크기다. K는 global max-K 5를 넘지 않는다. 이 페이지는 설정, fixed-K 기준선, matched MRV2 성능을 한 cohort씩 분리해 보여준다.</p>

<section><h2>1. DynamicSD schedule과 실행 configuration</h2>
<svg viewBox="0 0 960 270" role="img" aria-label="Offered concurrency와 active scheduled batch가 Dynamic K lookup을 거쳐 MTP draft와 target verification으로 이어지는 흐름">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--blue)"/></marker></defs>
<rect x="20" y="35" width="180" height="72" rx="9" fill="var(--code)" stroke="var(--line)"/><text x="110" y="64" text-anchor="middle" font-weight="700">Offered concurrency</text><text x="110" y="87" text-anchor="middle" class="muted">C = 1…512</text>
<path d="M200 71H278" stroke="var(--blue)" stroke-width="2" marker-end="url(#arr)"/><text x="239" y="57" text-anchor="middle" class="muted">admit</text>
<rect x="280" y="35" width="190" height="72" rx="9" fill="var(--code)" stroke="var(--blue)"/><text x="375" y="64" text-anchor="middle" font-weight="700">Active scheduled batch</text><text x="375" y="87" text-anchor="middle" class="muted">N = len(scheduled requests)</text>
<path d="M470 71H548" stroke="var(--blue)" stroke-width="2" marker-end="url(#arr)"/><a href="{github_schedule}"><text x="509" y="54" text-anchor="middle" class="code-link">lookup[N]</text></a>
<rect x="550" y="35" width="170" height="72" rx="9" fill="var(--code)" stroke="var(--green)"/><text x="635" y="64" text-anchor="middle" font-weight="700">Selected K</text><text x="635" y="87" text-anchor="middle" class="muted">min(schedule K, max-K 5)</text>
<path d="M720 71H798" stroke="var(--blue)" stroke-width="2" marker-end="url(#arr)"/><text x="759" y="54" text-anchor="middle" class="muted">execute</text>
<rect x="800" y="24" width="140" height="94" rx="9" fill="var(--code)" stroke="var(--line)"/><text x="870" y="55" text-anchor="middle" font-weight="700">MTP draft K</text><text x="870" y="77" text-anchor="middle" class="muted">target verify</text><text x="870" y="99" text-anchor="middle" class="muted">accept / reject</text>
<text x="20" y="152" font-weight="700">Configured active-batch lookup</text>
<a href="{github_lookup}"><text x="265" y="152" class="code-link">dense lookup construction</text></a>
<g font-size="13"><rect x="20" y="168" width="180" height="48" rx="7" fill="var(--green)" opacity=".22"/><text x="110" y="188" text-anchor="middle">N 1–4</text><text x="110" y="207" text-anchor="middle" font-weight="700">K5</text><rect x="207" y="168" width="180" height="48" rx="7" fill="var(--green)" opacity=".18"/><text x="297" y="188" text-anchor="middle">N 5–16</text><text x="297" y="207" text-anchor="middle" font-weight="700">K3</text><rect x="394" y="168" width="180" height="48" rx="7" fill="var(--blue)" opacity=".18"/><text x="484" y="188" text-anchor="middle">N 17–64</text><text x="484" y="207" text-anchor="middle" font-weight="700">K2</text><rect x="581" y="168" width="180" height="48" rx="7" fill="var(--amber)" opacity=".20"/><text x="671" y="188" text-anchor="middle">N 65–128</text><text x="671" y="207" text-anchor="middle" font-weight="700">K1</text><rect x="768" y="168" width="172" height="48" rx="7" fill="var(--gray)" opacity=".24"/><text x="854" y="188" text-anchor="middle">N 129–512</text><text x="854" y="207" text-anchor="middle" font-weight="700">K0</text></g>
<text x="20" y="250" class="muted">Worked example: offered C=512라도 한 step의 active N=73이면 lookup[73]=K1이다.</text>
</svg>
<p><code>{{"method":"mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,16,3],[17,64,2],[65,128,1],[129,512,0]]}}</code></p>
<div class="config"><div class="card"><strong>Workload</strong><p>ISL/OSL 1000/10000, 10000/1000</p><p>C 1,2,4,8,16,32,128,512</p><p>temperature/top-p 1.0/1.0, seed 42</p></div><div class="card"><strong>Runtime</strong><p>BF16 weights · FP8 KV · chunked prefill on</p><p>prefix caching off · max seqs 512</p><p>max batched tokens 32768 · repeat 1+1</p></div><div class="card"><strong>Parallelism</strong><p>Super: TP2/PP1/DP1, 2 GPUs</p><p>Ultra: TP8/PP1/DP1/EP, 8 GPUs</p><p>Ultra Ray 2.48.0</p></div><div class="card"><strong>CUDA Graph</strong><p>MRV2 <code>FULL_AND_PIECEWISE</code></p><p><code>enforce_eager=false</code></p><p>target + drafter prefill/decode verified</p></div></div>
<details><summary>Checkpoint와 patch provenance</summary><p>Super BF16 revision <code>d51eab0d1f97</code>; Ultra BF16 revision <code>624ba927cfbe</code>. 둘 다 <code>num_nextn_predict_layers=1</code>이며 K2–K5는 같은 MTP block을 반복 사용한다.</p><p>Native DynamicSD: <a href="https://github.com/vllm-project/vllm/pull/32374">#32374</a>; MRV2 full graph: <a href="https://github.com/vllm-project/vllm/pull/45953">#45953</a>. Patchset <code>{PATCHSET_SHA256[:12]}</code>: <a href="https://github.com/vllm-project/vllm/pull/49652">#49652</a>, <a href="https://github.com/vllm-project/vllm/pull/51575">#51575</a>, <a href="https://github.com/vllm-project/vllm/pull/52548">#52548</a>와 Mamba reduced-query guard. Optional <a href="https://github.com/vllm-project/vllm/pull/53426">#53426</a>은 제외했다. Container SHA256 <code>{CONTAINER_SHA256[:16]}…</code>.</p></details>
<div class="warn"><strong>C=512 proves positive-K drafting occurred, not which K dominated.</strong> Offered concurrency는 lookup key가 아니다. Positive mean draft width는 batch drain 중 positive-K drafting이 있었다는 뜻이며, 특정 K가 대부분이었다는 뜻은 아니다.</div></section>

<section><h2>2. Does it work? — matched 결과와 fixed-K 기준</h2>
<div class="cards"><div class="card"><div class="label">MRV2 jobs</div><div class="metric win">64/64</div></div><div class="card"><div class="label">Exact tokens + graphs</div><div class="metric win">64/64</div></div><div class="card"><div class="label">Dynamic wins</div><div class="metric">{wins}/32</div></div><div class="card"><div class="label">Best / worst</div><div class="metric">{best["throughput_speedup"]:.2f}× / {worst["throughput_speedup"]:.2f}×</div></div></div>
<h3>Patched MRV2 DynamicMTP speedup</h3><div class="scroll"><table><thead><tr><th>Model</th><th>ISL/OSL</th>{"".join(f"<th>C{c}</th>" for c in BATCH_SIZES)}</tr></thead><tbody>{"".join(dynamic_summary_rows)}</tbody></table></div>
<p>C1–32는 24/24 승리했다. C128/512는 0/8 승리했다. Speedup은 각 행의 Dynamic tok/s/GPU를 완전히 일치하는 K0 baseline tok/s/GPU로 나눈 값이다.</p>
<h3>MRV1 fixed-K1–K5 전체 ladder + DynamicSD</h3><p><code>DynamicSD (MRV1)</code>은 K1–K5와 동일한 MRV1/PIECEWISE baseline을 사용하므로 직접 비교할 수 있다. 단, 10K/1K C512의 Super/Ultra 두 pair는 corrected Dynamic harness <code>f0dd8af</code>와 이전 baseline harness <code>3fb7073</code>를 결합한 명시적 provenance 예외다. <code>DynamicSD (MRV2)</code>는 최신 patched FULL_AND_PIECEWISE 결과를 같은 설정 옆에 표시한 참고 열이며, MRV1 값과 직접 순위를 매기지 않는다.</p>{"".join(fixed_tables)}
<details><summary>MRV2 전체 32 matched pairs / 64 jobs</summary><div class="controls"><select id="model"><option value="">All models</option><option value="super">Super</option><option value="ultra">Ultra</option></select><select id="shape"><option value="">All shapes</option><option value="1000/10000">1K/10K</option><option value="10000/1000">10K/1K</option></select></div><div class="scroll"><table id="results"><thead><tr><th>Model</th><th>ISL/OSL</th><th>C</th><th>Base tok/s/GPU</th><th>Dynamic</th><th>Speedup</th><th>Offered-C K</th><th>Mean width</th><th>Acceptance</th><th>Mean accepted</th><th>Base job</th><th>Dynamic job</th></tr></thead><tbody></tbody></table></div></details>
<p><a href="{html.escape(csv_href)}">MRV2 canonical CSV</a> · <a href="{html.escape(fixed_k_csv_href)}">MRV1 fixed-K canonical CSV</a> · <a href="{html.escape(source_manifest_href)}">MRV2 SHA256 manifest</a></p>
<p class="warn"><strong>Evidence limit:</strong> 모든 행은 single measured repeat다. 정확한 token 수와 CUDA Graph 실행은 증명하지만 분산이나 품질 동등성은 증명하지 않는다. 과거 unmatched canary의 <code>N/A</code>는 speedup 부재가 아니라 matched baseline 부재다.</p></section>

<section><h2>3. 무엇을 바꿔야 하나?</h2>
<p>High concurrency에서는 baseline batching 효율이 커지고 실제 mean draft width는 C32의 약 2.0에서 C128의 1.20–1.40, C512의 1.01–1.10으로 내려갔다. Acceptance가 높아도 제안량이 작으면 절약한 target work가 MTP 전체 추가 work를 상쇄하지 못한다. 측정은 총 overhead를 보여주지만 drafter, verification, synchronization, scheduling 비용을 개별 분해하지는 않는다.</p>
<div class="scroll"><table><thead><tr><th>Model / shape</th><th>Fixed-K가 제안하는 후보</th><th>C512 판단</th></tr></thead><tbody><tr><td>Super 1K/10K</td><td>K5 → K3 → K1</td><td>K3 1.05×, repeat 필요</td></tr><tr><td>Super 10K/1K</td><td>K5 → K4 → K3 → K1</td><td>K1 1.03×, repeat 필요</td></tr><tr><td>Ultra 1K/10K</td><td>K4 → K2</td><td>K2 1.16×</td></tr><tr><td>Ultra 10K/1K</td><td>K4/K5 → K4 → K1 → K0</td><td>모든 K1–K5 &lt;1×</td></tr></tbody></table></div>
<div class="hl"><strong>Next step:</strong> 하나의 공통 schedule 대신 모델·ISL/OSL별 active-batch schedule을 만들고, 경계 C32/64/128/512를 3회 이상 반복한다. C128 이상 K0 후보는 별도 matched MRV2 cohort로 검증한다.</div></section>
<script id="data" type="application/json">{embedded}</script><script>const d=JSON.parse(document.querySelector('#data').textContent),b=document.querySelector('#results tbody'),mf=document.querySelector('#model'),sf=document.querySelector('#shape');function draw(){{b.textContent='';for(const r of d){{if((mf.value&&r[0]!==mf.value)||(sf.value&&r[1]!==sf.value))continue;const tr=document.createElement('tr');tr.innerHTML=`<td>${{r[0][0].toUpperCase()+r[0].slice(1)}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td><td>${{r[3]}}</td><td>${{r[4]}}</td><td class="${{r[5]>1?'win':'loss'}}">${{r[5]}}×</td><td>K${{r[6]}}</td><td>${{r[7]}}</td><td>${{(r[8]*100).toFixed(2)}}%</td><td>${{r[9]}}</td><td>${{r[10]}}</td><td>${{r[11]}}</td>`;b.appendChild(tr);}}}}mf.onchange=sf.onchange=draw;draw();</script>
</main></body></html>"""


def build_report(
    *,
    result_root: Path,
    fixed_k_csv: Path,
    output_json: Path,
    output_csv: Path,
    output_html: Path,
    output_source_manifest: Path,
) -> list[dict[str, Any]]:
    rows = normalize_rows(load_cells(result_root, expected_keys()))
    fixed_k_rows = load_fixed_k_rows(fixed_k_csv)
    mrv1_dynamic_rows = load_mrv1_dynamic_rows(fixed_k_csv)
    if len(rows) != 64 or sum(bool(row["cuda_graph_verified"]) for row in rows) != 64:
        raise ValueError("full matrix did not validate as 64 CUDA-Graph-backed rows")
    for path in (output_json, output_csv, output_html, output_source_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output_html.write_text(
        render_html(
            rows,
            fixed_k_rows=fixed_k_rows,
            mrv1_dynamic_rows=mrv1_dynamic_rows,
            csv_href=relative_href(output_html, output_csv),
            fixed_k_csv_href=relative_href(output_html, fixed_k_csv),
            source_manifest_href=relative_href(output_html, output_source_manifest),
        ),
        encoding="utf-8",
    )
    output_source_manifest.write_text(
        json.dumps(
            build_source_manifest(result_root, expected_keys()),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--fixed-k-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-source-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = build_report(
        result_root=args.result_root,
        fixed_k_csv=args.fixed_k_csv,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_html=args.output_html,
        output_source_manifest=args.output_source_manifest,
    )
    dynamic = [row for row in rows if row["method"] == "mtp_dynamic_max_k5"]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "cuda_graph_verified": sum(
                    bool(row["cuda_graph_verified"]) for row in rows
                ),
                "dynamic_wins": sum(
                    float(row["throughput_speedup"]) > 1.0 for row in dynamic
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
