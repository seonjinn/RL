#!/usr/bin/env python3
"""Validate and normalize the vLLM 0.28 Nemotron BF16 MRV1 cohort."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, NamedTuple


class ResultKey(NamedTuple):
    model_key: str
    isl: int
    osl: int
    runner_key: str
    method_key: str
    batch_size: int


def expected_mrv1_keys(contract: dict[str, Any]) -> list[ResultKey]:
    """Return the deterministic canonical MRV1 result key order."""
    return [
        ResultKey(
            model_key=str(model["key"]),
            isl=int(shape["isl"]),
            osl=int(shape["osl"]),
            runner_key="mrv1",
            method_key=str(method_key),
            batch_size=int(batch_size),
        )
        for model in contract["models"]
        for shape in contract["shapes"]
        for method_key in contract["method_order"]
        for batch_size in contract["batch_sizes"]
    ]


def result_path(result_root: Path, key: ResultKey) -> Path:
    return (
        result_root
        / key.model_key
        / f"isl{key.isl}_osl{key.osl}"
        / key.runner_key
        / key.method_key
        / f"bs{key.batch_size}"
        / "result.json"
    )


def load_payloads(result_root: Path, keys: list[ResultKey]) -> list[dict[str, Any]]:
    """Load only the expected canonical hierarchy and reject identity drift."""
    payloads: list[dict[str, Any]] = []
    for key in keys:
        path = result_path(result_root, key)
        if not path.is_file():
            raise ValueError(f"missing canonical result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        observed = ResultKey(
            str(config.get("model_key")),
            int(config.get("isl", -1)),
            int(config.get("osl", -1)),
            str(config.get("runner_key")),
            str(config.get("method_key")),
            int(config.get("batch_size", -1)),
        )
        if observed != key:
            raise ValueError(
                f"result identity mismatch at {path}: expected {key}, got {observed}"
            )
        payloads.append(payload)
    return payloads


def parse_cuda_graph_evidence(log_text: str) -> dict[str, Any]:
    """Extract observed PIECEWISE CUDA Graph capture evidence from one job log."""
    capture_matches = re.findall(
        r"Capturing CUDA graphs[^\r\n]*PIECEWISE[^\r\n]*?(\d+)\s*/\s*(\d+)",
        log_text,
    )
    completed, total = (0, 0)
    if capture_matches:
        completed, total = (int(value) for value in capture_matches[-1])
    profiled_match = re.search(r"PIECEWISE\s*=\s*(\d+)", log_text)
    profiled_total = int(profiled_match.group(1)) if profiled_match else 0
    memory_match = re.search(
        r"CUDA graph pool memory:\s*([0-9.]+)\s*GiB\s*\(actual\)",
        log_text,
    )
    piecewise = "PIECEWISE" in log_text
    eager_disabled = "enforce_eager=False" in log_text
    verified = (
        piecewise
        and eager_disabled
        and completed == total == profiled_total > 0
    )
    return {
        "verified": verified,
        "mode": "PIECEWISE" if piecewise else None,
        "enforce_eager": False if eager_disabled else None,
        "capture_completed": completed,
        "capture_total": total,
        "profiled_total": profiled_total,
        "pool_memory_gib": float(memory_match.group(1)) if memory_match else None,
        "allocator_oom_warning": "CUDACachingAllocator" in log_text
        and "OOM" in log_text,
    }


def _identity(payload: dict[str, Any]) -> tuple[str, int, int, str, int]:
    config = payload["config"]
    return (
        str(config["model_key"]),
        int(config["isl"]),
        int(config["osl"]),
        str(config["runner_key"]),
        int(config["batch_size"]),
    )


def _validate_payload_provenance(payload: dict[str, Any]) -> str:
    """Reject method, batch, runtime, and topology drift from the contract."""
    config = payload["config"]
    row = payload["results"][0]
    method = str(config.get("method_key", ""))
    batch_size = int(config.get("batch_size", -1))
    if int(row.get("bs", row.get("batch_size", -1))) != batch_size:
        raise ValueError("result row batch size does not match config batch size")

    speculative_config = config.get("speculative_config")
    if method == "baseline":
        if (
            speculative_config is not None
            or config.get("effective_k") != 0
            or config.get("k_selection_basis") not in (None, "baseline")
        ):
            raise ValueError("baseline method provenance is inconsistent")
    elif method.startswith("mtp_static_k"):
        expected_k = int(method.removeprefix("mtp_static_k"))
        if (
            speculative_config
            != {"method": "mtp", "num_speculative_tokens": expected_k}
            or config.get("effective_k") != expected_k
            or config.get("k_selection_basis") not in (None, "static")
        ):
            raise ValueError(f"static method K{expected_k} provenance is inconsistent")
    elif method == "mtp_dynamic_max_k5":
        schedule = [
            [1, 4, 5],
            [5, 16, 3],
            [17, 64, 2],
            [65, 128, 1],
            [129, 512, 0],
        ]
        expected_requested_k = next(
            k for start, end, k in schedule if start <= batch_size <= end
        )
        expected_speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": schedule,
        }
        current_provenance = (
            config.get("effective_k") is None
            and config.get("requested_batch_schedule_k") == expected_requested_k
            and config.get("k_selection_basis") == "active_scheduled_batch"
        )
        legacy_provenance = (
            config.get("effective_k") == expected_requested_k
            and config.get("requested_batch_schedule_k") is None
            and config.get("k_selection_basis") is None
        )
        if speculative_config != expected_speculative_config or not (
            current_provenance or legacy_provenance
        ):
            raise ValueError("dynamic method schedule or K provenance is inconsistent")
    else:
        raise ValueError(f"unsupported method provenance: {method!r}")

    model = str(config.get("model_key", ""))
    expected_models = {
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
    }
    expected_model = expected_models.get(model)
    if expected_model is None:
        raise ValueError(f"unknown checkpoint model key: {model!r}")
    expected_tp, expected_ep, checkpoint_name, checkpoint_revision = expected_model
    model_path = str(config.get("model", ""))
    if checkpoint_name not in model_path or not model_path.endswith(
        f"/snapshots/{checkpoint_revision}"
    ):
        raise ValueError(f"checkpoint revision drift for model={model!r}")
    expected_topology = (expected_tp, expected_ep)
    if (
        int(config.get("tensor_parallel_size", -1)),
        bool(config.get("enable_expert_parallel", False)),
    ) != expected_topology:
        raise ValueError(f"runtime topology drift for model={model!r}")
    if (
        int(config.get("pipeline_parallel_size", -1)) != 1
        or int(config.get("engine_gpus", -1)) != expected_tp
        or int(config.get("total_gpus", -1)) != expected_tp
    ):
        raise ValueError("runtime PP/DP/total-GPU topology drift")
    if (
        config.get("dtype") != "bfloat16"
        or config.get("kv_cache_dtype") != "fp8"
        or config.get("runner_key") != "mrv1"
    ):
        raise ValueError("runtime precision or runner drift")

    runtime = payload.get("runtime", {})
    provenance = payload.get("runtime_provenance", {})
    if (
        runtime.get("vllm_version") != "0.28.0"
        or provenance.get("vllm_commit") != "2cf0a69"
    ):
        raise ValueError("runtime vLLM release or commit drift")
    if provenance.get("container_digest") != (
        "sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4"
    ):
        raise ValueError("container digest drift")
    job_id = str(runtime.get("environment", {}).get("SLURM_JOB_ID", ""))
    if not job_id:
        raise ValueError("empty SLURM job ID")
    return job_id


def normalize_rows(
    payloads: list[dict[str, Any]],
    *,
    logs_by_job_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate payloads, join matched baselines, and return flat report rows."""
    baseline_tps: dict[tuple[str, int, int, str, int], float] = {}
    seen_job_ids: set[str] = set()
    for payload in payloads:
        config = payload.get("config")
        results = payload.get("results")
        if not isinstance(config, dict) or not isinstance(results, list) or len(results) != 1:
            raise ValueError("result must contain one config and one result row")
        job_id = _validate_payload_provenance(payload)
        if job_id in seen_job_ids:
            raise ValueError(f"duplicate SLURM job ID: {job_id}")
        seen_job_ids.add(job_id)
        row = results[0]
        if config.get("method_key") == "baseline":
            baseline_tps[_identity(payload)] = float(row["output_tok_s_per_gpu"])

    normalized: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("schema_version") != 1 or payload.get("status") != "complete":
            raise ValueError("result payload is not complete schema version 1")
        config = payload["config"]
        row = payload["results"][0]
        expected_tokens = int(config["batch_size"]) * int(config["osl"])
        if (
            row.get("tokens_ok") is not True
            or int(row.get("actual_output_tokens", -1)) != expected_tokens
            or int(row.get("expected_output_tokens", -1)) != expected_tokens
        ):
            raise ValueError("result failed exact output-token validation")
        if config.get("cudagraph_mode") != "PIECEWISE" or config.get("enforce_eager") is not False:
            raise ValueError("result did not request PIECEWISE CUDA Graph execution")
        runtime = payload.get("runtime", {})
        job_id = str(runtime.get("environment", {}).get("SLURM_JOB_ID", ""))
        graph = parse_cuda_graph_evidence(logs_by_job_id.get(job_id, ""))
        metrics = row.get("spec_decode_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        baseline = baseline_tps.get(_identity(payload))
        if baseline is None:
            raise ValueError(f"missing matched baseline for {_identity(payload)}")
        tok_s_per_gpu = float(row["output_tok_s_per_gpu"])
        provenance = payload.get("runtime_provenance", {})
        method = str(config["method_key"])
        canonical_effective_k = config.get("effective_k")
        canonical_requested_k = config.get("requested_batch_schedule_k")
        canonical_basis = str(config.get("k_selection_basis", ""))
        if method == "baseline":
            canonical_basis = "baseline"
        elif method.startswith("mtp_static_k"):
            canonical_basis = "static"
        else:
            canonical_effective_k = None
            canonical_requested_k = next(
                item[2]
                for item in config["speculative_config"][
                    "num_speculative_tokens_per_batch_size"
                ]
                if item[0] <= int(config["batch_size"]) <= item[1]
            )
            canonical_basis = "active_scheduled_batch"
        normalized.append(
            {
                "model": str(config["model_key"]),
                "isl": int(config["isl"]),
                "osl": int(config["osl"]),
                "runner": str(config["runner_key"]),
                "method": method,
                "batch_size": int(config["batch_size"]),
                "effective_k": canonical_effective_k,
                "requested_batch_schedule_k": canonical_requested_k,
                "k_selection_basis": canonical_basis,
                "weight_dtype": str(config.get("dtype", "")),
                "kv_cache_dtype": str(config.get("kv_cache_dtype", "")),
                "tensor_parallel_size": int(config.get("tensor_parallel_size", 0)),
                "expert_parallel": bool(config.get("enable_expert_parallel", False)),
                "output_tok_s": float(row["output_tok_s"]),
                "output_tok_s_per_gpu": tok_s_per_gpu,
                "baseline_tok_s_per_gpu": baseline,
                "speedup_vs_baseline": tok_s_per_gpu / baseline,
                "latency_s": float(row["latency_s"]),
                "acceptance_rate": metrics.get("acceptance_rate"),
                "mean_acceptance_length": metrics.get("mean_acceptance_length"),
                "num_drafts": metrics.get("num_drafts"),
                "num_draft_tokens": metrics.get("num_draft_tokens"),
                "num_accepted_tokens": metrics.get("num_accepted_tokens"),
                "actual_output_tokens": int(row["actual_output_tokens"]),
                "expected_output_tokens": int(row["expected_output_tokens"]),
                "tokens_ok": True,
                "job_id": job_id,
                "vllm_version": str(runtime.get("vllm_version", "")),
                "vllm_commit": str(provenance.get("vllm_commit", "")),
                "container_digest": str(provenance.get("container_digest", "")),
                "harness_commit": str(provenance.get("harness_commit", "")),
                "cudagraph_mode": str(config["cudagraph_mode"]),
                "enforce_eager": bool(config["enforce_eager"]),
                "cuda_graph_verified": bool(graph["verified"]),
                "cuda_graph_capture_completed": int(graph["capture_completed"]),
                "cuda_graph_capture_total": int(graph["capture_total"]),
                "cuda_graph_pool_memory_gib": graph["pool_memory_gib"],
                "allocator_oom_warning": bool(graph["allocator_oom_warning"]),
            }
        )
    return normalized


def _display_method(method: str) -> str:
    if method == "baseline":
        return "K0 baseline"
    if method == "mtp_dynamic_max_k5":
        return "DynamicMTP"
    return method.replace("mtp_static_k", "Static K")


def render_html(
    rows: list[dict[str, Any]],
    *,
    dynamic_schedule: list[dict[str, int]],
) -> str:
    """Render a self-contained, filterable experiment page."""
    graph_verified = sum(bool(row["cuda_graph_verified"]) for row in rows)
    exact_tokens = sum(bool(row["tokens_ok"]) for row in rows)
    schedule_rows = "".join(
        "<tr>"
        f"<td>{item['start']}–{item['end']}</td>"
        f"<td>K={item['k']}</td>"
        "</tr>"
        for item in dynamic_schedule
    )
    best_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["model"], row["isl"], row["osl"], row["batch_size"]), []
        ).append(row)
    for group in groups.values():
        best_rows.append(max(group, key=lambda item: item["speedup_vs_baseline"]))
    best_rows.sort(
        key=lambda item: (
            item["model"],
            item["isl"],
            item["osl"],
            item["batch_size"],
        )
    )

    def table_body(items: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for row in items:
            acceptance = row["acceptance_rate"]
            mean_length = row["mean_acceptance_length"]
            graph_capture = "not verified"
            if row["cuda_graph_verified"]:
                graph_capture = (
                    f"{row['cuda_graph_capture_completed']}/"
                    f"{row['cuda_graph_capture_total']}"
                )
            rendered.append(
                "<tr "
                f"data-model=\"{html.escape(str(row['model']))}\" "
                f"data-shape=\"{row['isl']}/{row['osl']}\" "
                f"data-method=\"{html.escape(str(row['method']))}\">"
                f"<td>{html.escape(str(row['model']).title())}</td>"
                f"<td>{row['isl']}/{row['osl']}</td>"
                f"<td>{row['batch_size']}</td>"
                f"<td>{html.escape(_display_method(str(row['method'])))}</td>"
                f"<td>{row['output_tok_s_per_gpu']:.4f}</td>"
                f"<td>{row['speedup_vs_baseline']:.4f}×</td>"
                f"<td>{'-' if acceptance is None else f'{float(acceptance):.2%}'}</td>"
                f"<td>{'-' if mean_length is None else f'{float(mean_length):.4f}'}</td>"
                f"<td>{graph_capture}</td>"
                f"<td><code>{html.escape(str(row['job_id']))}</code></td>"
                "</tr>"
            )
        return "".join(rendered)

    embedded = json.dumps(rows, separators=(",", ":"), sort_keys=True).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM 0.28 Nemotron-3 BF16 MTP / DynamicSD</title>
<style>
:root{{--bg:#f5f7fa;--panel:#fff;--ink:#111827;--muted:#5f6b7a;--line:#d9dee7;--green:#14804a;--blue:#1d5fbf;--amber:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1400px;margin:auto;padding:28px 18px 48px}}h1{{font-size:30px;margin:0}}h2{{margin:30px 0 10px}}p{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px}}
.label{{font-size:12px;text-transform:uppercase;color:var(--muted);font-weight:700}}.metric{{font-size:25px;font-weight:800;margin-top:4px}}.ok{{color:var(--green)}}
code{{background:#eef1f6;border:1px solid var(--line);padding:1px 4px;border-radius:4px}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:9px;background:#fff}}
table{{border-collapse:collapse;width:100%;min-width:980px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#eef1f6;position:sticky;top:0}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(4),td:nth-child(4){{text-align:left}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}select{{padding:8px;border:1px solid var(--line);border-radius:7px;background:#fff}}.note{{border-left:4px solid var(--blue)}}
.warn{{border-left:4px solid var(--amber)}}a{{color:var(--blue)}}@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>vLLM 0.28 Nemotron-3 BF16 MTP / DynamicSD</h1>
<p>Standalone GB200 evaluation for Super 120B-A12B BF16 and Ultra 550B-A55B BF16. Throughput speedups use an exact model/shape/concurrency matched baseline.</p>
<div class="grid">
<div class="card"><div class="label">Canonical rows</div><div class="metric">{len(rows)}</div></div>
<div class="card"><div class="label">Exact-token rows</div><div class="metric ok">{exact_tokens}/{len(rows)}</div></div>
<div class="card"><div class="label">Observed CUDA Graph</div><div class="metric {'ok' if graph_verified == len(rows) else ''}">{graph_verified}/{len(rows)}</div></div>
<div class="card"><div class="label">Static sweep</div><div class="metric">K0–K5</div></div>
</div>
<section class="panel note"><h2>What K means for these checkpoints</h2>
<p>Both BF16 checkpoints declare <code>num_nextn_predict_layers=1</code>. K=2–5 therefore reuse the same native one-layer MTP block for multiple forwards; they do not select 2–5 distinct MTP heads. vLLM accepts positive K values divisible by the native depth. K1–K5 is the chosen practical sweep, not every theoretically accepted positive integer.</p></section>
<section><h2>DynamicMTP configuration</h2><div class="panel"><p><code>{{"method":"mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,16,3],[17,64,2],[65,128,1],[129,512,0]]}}</code></p>
<p>K is chosen every scheduler step from the <strong>active scheduled batch</strong>, not the offered offline concurrency. DP is fixed at 1; MRV1 forces PIECEWISE CUDA Graphs for reliability.</p>
<table style="max-width:420px;min-width:0"><thead><tr><th>Active batch</th><th>Selected K</th></tr></thead><tbody>{schedule_rows}</tbody></table></div></section>
<section><h2>Best method by model, shape, and concurrency</h2><div class="scroll"><table><thead><tr><th>Model</th><th>ISL/OSL</th><th>Concurrency</th><th>Best method</th><th>tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accepted length</th><th>CUDA Graph</th><th>Job</th></tr></thead><tbody>{table_body(best_rows)}</tbody></table></div></section>
<section><h2>Complete result table</h2><div class="controls"><select id="model"><option value="">All models</option><option value="super">Super</option><option value="ultra">Ultra</option></select><select id="shape"><option value="">All shapes</option><option value="1000/10000">1K/10K</option><option value="10000/1000">10K/1K</option></select><select id="method"><option value="">All methods</option>{''.join(f'<option value="{html.escape(method)}">{html.escape(_display_method(method))}</option>' for method in sorted({str(row['method']) for row in rows}))}</select></div>
<div class="scroll"><table id="results"><thead><tr><th>Model</th><th>ISL/OSL</th><th>Concurrency</th><th>Method</th><th>tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accepted length</th><th>CUDA Graph</th><th>Job</th></tr></thead><tbody>{table_body(rows)}</tbody></table></div></section>
<section class="panel warn"><h2>Runtime interpretation</h2><p>Configuration alone is not counted as CUDA Graph proof. A row is verified only when its job log records <code>enforce_eager=False</code>, PIECEWISE profiling, and completed capture. DynamicMTP acceptance is aggregated across the K values selected while the active batch drains.</p></section>
<section><h2>Downloads and provenance</h2><p><a href="../data/vllm028_nemotron3_bf16_dynamicsd_20260827/results.csv">Canonical CSV</a></p><p>vLLM 0.28.0 commit <code>2cf0a69</code>; BF16 weights, FP8 KV cache; Super TP2/DP1, Ultra TP8/DP1/EP across two nodes.</p></section>
<script id="data" type="application/json">{embedded}</script>
<script>const controls=[...document.querySelectorAll('select')];const rows=[...document.querySelectorAll('#results tbody tr')];function filter(){{const m=document.querySelector('#model').value,s=document.querySelector('#shape').value,k=document.querySelector('#method').value;for(const r of rows)r.hidden=!!((m&&r.dataset.model!==m)||(s&&r.dataset.shape!==s)||(k&&r.dataset.method!==k));}}controls.forEach(x=>x.addEventListener('change',filter));</script>
</main></body></html>"""


def build_artifacts(
    *,
    result_root: Path,
    log_dir: Path,
    output_json: Path,
    output_csv: Path,
    output_html: Path,
    keys: list[ResultKey],
    dynamic_schedule: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Build validated normalized JSON, CSV, and self-contained HTML artifacts."""
    payloads = load_payloads(result_root, keys)
    logs_by_job_id: dict[str, str] = {}
    for payload in payloads:
        job_id = str(
            payload.get("runtime", {}).get("environment", {}).get("SLURM_JOB_ID", "")
        )
        log_path = log_dir / f"slurm-{job_id}.out"
        if not log_path.is_file():
            raise ValueError(f"missing CUDA Graph evidence log: {log_path}")
        logs_by_job_id[job_id] = log_path.read_text(encoding="utf-8", errors="replace")
    rows = normalize_rows(payloads, logs_by_job_id=logs_by_job_id)
    unverified = [row["job_id"] for row in rows if not row["cuda_graph_verified"]]
    if unverified:
        raise ValueError(f"CUDA Graph capture is not verified for jobs: {unverified}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    output_html.write_text(
        render_html(rows, dynamic_schedule=dynamic_schedule),
        encoding="utf-8",
    )
    return rows


def _load_contract() -> dict[str, Any]:
    contract_path = Path(__file__).with_name("contract.py")
    spec = importlib.util.spec_from_file_location("vllm028_report_contract", contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load contract: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_contract_matrix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    contract = _load_contract()
    rows = build_artifacts(
        result_root=args.result_root,
        log_dir=args.log_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_html=args.output_html,
        keys=expected_mrv1_keys(contract),
        dynamic_schedule=contract["dynamic_schedule"],
    )
    print(json.dumps({"rows": len(rows), "cuda_graph_verified": len(rows)}))


if __name__ == "__main__":
    main()
