#!/usr/bin/env python3
"""Build validated CSV, JSON, and HTML for the patched MRV2 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple


PACKAGE_ROOT = Path(__file__).resolve().parent
PATCHSET_SHA256 = "238e2ffcc14d2fb2f0fc07c419004820efa9a3cfd284c3a41515e84e39aecc25"
VLLM_BASE_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
PATCHED_VLLM_HEAD = "bf0719d13bbc74da8af86d2b326f7b16ba4f7462"
CONTAINER_SHA256 = "5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066"
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
            raise ValueError(f"result identity mismatch: expected {key}, got {observed}")
        job_id = str(payload.get("runtime", {}).get("environment", {}).get("SLURM_JOB_ID", ""))
        if result_path.parent.name != f"job-{job_id}":
            raise ValueError("result path is not bound to its SLURM job ID")
        cells.append((payload, evidence))
    return cells


def _completed_capture(
    rows: Any, *, required_text: tuple[str, ...]
) -> bool:
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
        raise ValueError("missing completed drafter prefill PIECEWISE CUDA Graph capture")
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
    if not isinstance(config, dict) or not isinstance(results, list) or len(results) != 1:
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
        if config.get("speculative_config") is not None or config.get("effective_k") != 0:
            raise ValueError("baseline speculative configuration drift")
    elif method == "mtp_dynamic_max_k5":
        expected_config = {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [list(item) for item in DYNAMIC_SCHEDULE],
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
        "super": (2, False, "NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "d51eab0d1f979ebc26b546e634a04f450d99158e"),
        "ultra": (8, True, "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "624ba927cfbef0427354998700de3d51173c8c04"),
    }.get(model)
    if expected_model is None:
        raise ValueError(f"unsupported model: {model}")
    tp, expert_parallel, checkpoint, revision = expected_model
    model_path = str(config.get("model", ""))
    if checkpoint not in model_path or not model_path.endswith(f"/snapshots/{revision}"):
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


def _validate_harness_pair(
    baseline: dict[str, Any], dynamic: dict[str, Any]
) -> bool:
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
    validated: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
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
                "requested_batch_schedule_k": _requested_k(int(config["batch_size"])) if dynamic else 0,
                "k_selection_basis": "active_scheduled_batch" if dynamic else "baseline",
                "mean_draft_width": draft_tokens / drafts if dynamic else None,
                "output_tok_s": float(row["output_tok_s"]),
                "tok_s_gpu": tok_s_gpu,
                "baseline_tok_s_gpu": baseline,
                "throughput_speedup": tok_s_gpu / baseline,
                "latency_s": float(row["latency_s"]),
                "acceptance_rate": float(metrics["acceptance_rate"]) if dynamic else None,
                "mean_accepted_length": float(metrics["mean_acceptance_length"]) if dynamic else None,
                "num_drafts": drafts if dynamic else None,
                "num_draft_tokens": draft_tokens if dynamic else None,
                "num_accepted_tokens": float(metrics["num_accepted_tokens"]) if dynamic else None,
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
                "drafter_decode_completed_entries": len(evidence["drafter_decode_completed"]),
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
                "harness_manifest_sha256": str(
                    provenance["harness_manifest_sha256"]
                ),
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
        for kind, path in (("result", result_path), ("cuda_graph_evidence", evidence_path)):
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


def _display_method(method: str) -> str:
    return "K0 baseline" if method == "baseline" else "DynamicMTP"


def render_html(rows: list[dict[str, Any]], *, csv_href: str) -> str:
    dynamic_rows = [row for row in rows if row["method"] == "mtp_dynamic_max_k5"]
    wins = sum(float(row["throughput_speedup"]) > 1.0 for row in dynamic_rows)
    best = max(dynamic_rows, key=lambda row: float(row["throughput_speedup"]))
    worst = min(dynamic_rows, key=lambda row: float(row["throughput_speedup"]))

    def render_row(row: dict[str, Any]) -> str:
        acceptance = row["acceptance_rate"]
        accepted_length = row["mean_accepted_length"]
        draft_width = row["mean_draft_width"]
        return (
            "<tr "
            f'data-model="{html.escape(str(row["model"]))}" '
            f'data-shape="{row["isl"]}/{row["osl"]}" '
            f'data-method="{html.escape(str(row["method"]))}">'
            f"<td>{html.escape(str(row['model']).title())}</td>"
            f"<td>{row['isl']}/{row['osl']}</td>"
            f"<td>{row['concurrency']}</td>"
            f"<td>{_display_method(str(row['method']))}</td>"
            f"<td>{row['requested_batch_schedule_k']}</td>"
            f"<td>{'-' if draft_width is None else f'{float(draft_width):.3f}'}</td>"
            f"<td>{row['tok_s_gpu']:.3f}</td>"
            f"<td>{row['throughput_speedup']:.3f}×</td>"
            f"<td>{'-' if acceptance is None else f'{float(acceptance):.2%}'}</td>"
            f"<td>{'-' if accepted_length is None else f'{float(accepted_length):.3f}'}</td>"
            f"<td>{row['actual_output_tokens']:,}</td>"
            f"<td>{'yes' if row['cuda_graph_verified'] else 'no'}</td>"
            f"<td><code>{row['job_id']}</code></td>"
            "</tr>"
        )

    table_rows = "".join(render_row(row) for row in rows)
    schedule_rows = "".join(
        f"<tr><td>{start}–{end}</td><td>K{k}</td></tr>"
        for start, end, k in DYNAMIC_SCHEDULE
    )
    embedded = json.dumps(rows, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vLLM 0.28 Nemotron-3 BF16 patched MRV2 DynamicMTP matrix</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#5d687d;--line:#d9dfeb;--green:#4b7f00;--blue:#2563a9;--amber:#925d00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:30px 18px 60px}}h1{{margin:0;font-size:30px}}h2{{margin-top:0}}p{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}.card,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:18px}}.label{{font-size:12px;text-transform:uppercase;color:var(--muted);font-weight:700}}.metric{{font-size:24px;font-weight:800}}.ok{{color:var(--green)}}.warn{{border-left:4px solid var(--amber)}}code{{font-size:.9em}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:1100px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#edf2f8;position:sticky;top:0}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(4),td:nth-child(4){{text-align:left}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}}select{{padding:8px;border:1px solid var(--line);border-radius:7px;background:#fff}}a{{color:var(--blue)}}@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>vLLM 0.28 · Nemotron-3 BF16 · patched MRV2 DynamicMTP</h1>
<p>GB200 standalone benchmark, preliminary single repeat. Every DynamicMTP row is compared only with the exact model, ISL/OSL, concurrency, precision, and topology-matched K0 baseline.</p>
<div class="grid"><div class="card"><div class="label">Canonical cells</div><div class="metric ok">64/64</div></div><div class="card"><div class="label">Exact-token + graphs</div><div class="metric ok">64/64</div></div><div class="card"><div class="label">Dynamic wins</div><div class="metric">{wins}/32</div></div><div class="card"><div class="label">Best speedup</div><div class="metric">{best['throughput_speedup']:.2f}×</div></div></div>
<section><h2>Key result</h2><p>Best: {best['model'].title()} {best['isl']}/{best['osl']} C={best['concurrency']} at {best['throughput_speedup']:.3f}×. Worst: {worst['model'].title()} {worst['isl']}/{worst['osl']} C={worst['concurrency']} at {worst['throughput_speedup']:.3f}×. DynamicMTP helps low-to-mid concurrency but the current schedule loses throughput at the high-concurrency edge.</p></section>
<section><h2>Dynamic K configuration</h2><p><code>{{"method":"mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,16,3],[17,64,2],[65,128,1],[129,512,0]]}}</code></p><p>K is selected from the <strong>active scheduled batch</strong> on every scheduler step, not from offered offline concurrency. Therefore C=512 can still execute K1 while requests drain; the actual mean draft width is reported separately.</p><table style="min-width:0;max-width:420px"><thead><tr><th>Active batch</th><th>K</th></tr></thead><tbody>{schedule_rows}</tbody></table></section>
<section><h2>Configuration and provenance</h2><p>BF16 checkpoint weights, FP8 KV cache, MRV2, <code>FULL_AND_PIECEWISE</code>, <code>enforce_eager=False</code>, max-num-seqs 512, max-num-batched-tokens 32768, chunked prefill on, prefix caching off, temperature/top-p 1.0/1.0, warmup 1 and measured repeat 1. Super: TP2/PP1/DP1, one node. Ultra: TP8/PP1/DP1 with expert parallel and Ray 2.48, two nodes.</p><p>vLLM base <code>{VLLM_BASE_COMMIT}</code>; patched head <code>{PATCHED_VLLM_HEAD}</code>; patchset <code>{PATCHSET_SHA256}</code>.</p></section>
<section class="warn"><h2>Interpretation</h2><p>Acceptance rate is accepted draft tokens divided by actually proposed draft tokens. Mean accepted length is vLLM's speculative iteration length metric. Mean draft width is actual draft tokens divided by draft iterations and exposes Dynamic K behavior directly. This is a single-repeat sweep; repeat runs are still required for publication-grade variance.</p><p>The four C=512 DynamicMTP rows use harness <code>f3a842c0</code> while their matched baselines use <code>85cfe1a8</code>. The later commit changes only active-batch result validation; the complete benchmark configuration, patched runtime, container, checkpoint, and CUDA Graph signature is identical. This one validation-only exception is explicitly gated and recorded per row.</p></section>
<section><h2>Complete matrix</h2><div class="controls"><select id="model"><option value="">All models</option><option value="super">Super</option><option value="ultra">Ultra</option></select><select id="shape"><option value="">All shapes</option><option value="1000/10000">1K/10K</option><option value="10000/1000">10K/1K</option></select><select id="method"><option value="">All methods</option><option value="baseline">K0 baseline</option><option value="mtp_dynamic_max_k5">DynamicMTP</option></select></div><div class="scroll"><table id="results"><thead><tr><th>Model</th><th>ISL/OSL</th><th>Concurrency</th><th>Method</th><th>Offered C K</th><th>Mean draft width</th><th>tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accepted length</th><th>Tokens</th><th>CUDA Graph</th><th>Job</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section><h2>Downloads</h2><p><a href="{html.escape(csv_href)}">Canonical full-matrix CSV</a> · <a href="../data/vllm028_nemotron3_bf16_mrv2_patched_matrix_20260828/source_manifest.json">SHA256 source manifest</a></p></section>
<script id="data" type="application/json">{embedded}</script><script>const rs=[...document.querySelectorAll('#results tbody tr')],modelFilter=document.querySelector('#model'),shapeFilter=document.querySelector('#shape'),methodFilter=document.querySelector('#method');function f(){{const m=modelFilter.value,s=shapeFilter.value,k=methodFilter.value;for(const r of rs)r.hidden=!!((m&&r.dataset.model!==m)||(s&&r.dataset.shape!==s)||(k&&r.dataset.method!==k));}}for(const x of document.querySelectorAll('select'))x.addEventListener('change',f);</script>
</main></body></html>"""


def build_report(
    *,
    result_root: Path,
    output_json: Path,
    output_csv: Path,
    output_html: Path,
    output_source_manifest: Path,
) -> list[dict[str, Any]]:
    rows = normalize_rows(load_cells(result_root, expected_keys()))
    if len(rows) != 64 or sum(bool(row["cuda_graph_verified"]) for row in rows) != 64:
        raise ValueError("full matrix did not validate as 64 CUDA-Graph-backed rows")
    for path in (output_json, output_csv, output_html, output_source_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output_html.write_text(
        render_html(rows, csv_href="../data/vllm028_nemotron3_bf16_mrv2_patched_matrix_20260828/results.csv"),
        encoding="utf-8",
    )
    output_source_manifest.write_text(
        json.dumps(build_source_manifest(result_root, expected_keys()), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-source-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = build_report(
        result_root=args.result_root,
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
                "cuda_graph_verified": sum(bool(row["cuda_graph_verified"]) for row in rows),
                "dynamic_wins": sum(float(row["throughput_speedup"]) > 1.0 for row in dynamic),
            }
        )
    )


if __name__ == "__main__":
    main()
