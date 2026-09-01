#!/usr/bin/env python3
"""Build the standalone patched-MRV2 DynamicMTP canary CSV and HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_BATCH_K = ((1, 5), (2, 3), (4, 2), (8, 1), (16, 0))
EXPECTED_PATCHSET = (
    "238e2ffcc14d2fb2f0fc07c419004820efa9a3cfd284c3a41515e84e39aecc25"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _validate_graph_evidence(evidence: dict[str, Any]) -> None:
    expected = {
        "target_piecewise": [83, 83],
        "target_full": [375, 375],
        "drafter_prefill_piecewise": [83, 83],
        "drafter_prefill_full": [375, 375],
        "drafter_decode_full": [51, 51],
    }
    for model in ("super", "ultra"):
        observed = evidence.get(model)
        if not isinstance(observed, dict) or not observed.get("job_id"):
            raise ValueError(f"missing CUDA Graph evidence for {model}")
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(f"incomplete {model} CUDA Graph evidence: {key}")


def load_report_data(
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence = _load_json(artifact_root / "cuda_graph_evidence.json")
    _validate_graph_evidence(evidence)
    rows: list[dict[str, Any]] = []
    runtimes: dict[str, Any] = {}
    for model in ("super", "ultra"):
        payload = _load_json(artifact_root / model / "result.json")
        config = payload.get("config")
        runtime = payload.get("runtime_provenance")
        result_rows = payload.get("results")
        if payload.get("status") != "complete":
            raise ValueError(f"{model} canary is not complete")
        if not isinstance(config, dict) or not isinstance(runtime, dict):
            raise ValueError(f"{model} canary lacks config or runtime provenance")
        if not isinstance(result_rows, list) or len(result_rows) != len(EXPECTED_BATCH_K):
            raise ValueError(f"{model} canary does not contain five rows")
        if (
            config.get("model_key") != model
            or config.get("runner_key") != "mrv2"
            or config.get("cudagraph_mode") != "FULL_AND_PIECEWISE"
            or config.get("isl") != 1000
            or config.get("osl") != 128
            or config.get("batch_sizes") != [1, 2, 4, 8, 16]
        ):
            raise ValueError(f"{model} canary configuration drift")
        if (
            runtime.get("vllm_version") != "0.28.0"
            or runtime.get("vllm_base_commit")
            != "2cf0a6915ce544dc493a0990f2ea38d81601128a"
            or runtime.get("patchset_manifest_sha256") != EXPECTED_PATCHSET
        ):
            raise ValueError(f"{model} runtime provenance drift")
        runtimes[model] = runtime
        for result, (batch_size, selected_k) in zip(
            result_rows, EXPECTED_BATCH_K, strict=True
        ):
            metrics = result.get("spec_decode_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"{model} BS{batch_size} lacks SpecDec metrics")
            if (
                result.get("bs") != batch_size
                or result.get("expected_dynamic_k") != selected_k
                or result.get("expected_output_tokens") != batch_size * 128
                or result.get("actual_output_tokens") != batch_size * 128
                or result.get("tokens_ok") is not True
            ):
                raise ValueError(f"{model} BS{batch_size} correctness drift")
            graph = evidence[model]
            rows.append(
                {
                    "model": model,
                    "batch_size": batch_size,
                    "selected_k": selected_k,
                    "isl": 1000,
                    "osl": 128,
                    "tok_s_gpu": float(result["output_tok_s_per_gpu"]),
                    "speedup": None,
                    "matched_baseline_available": False,
                    "acceptance_rate": (
                        float(metrics["acceptance_rate"]) if selected_k else None
                    ),
                    "mean_accepted_length": (
                        float(metrics["mean_acceptance_length"])
                        if selected_k
                        else None
                    ),
                    "expected_output_tokens": batch_size * 128,
                    "actual_output_tokens": batch_size * 128,
                    "tokens_ok": True,
                    "draft_counter_async_skew_tokens": float(
                        result["draft_counter_async_skew_tokens"]
                    ),
                    "draft_counter_residual_at_k0": bool(
                        result["draft_counter_residual_at_k0"]
                    ),
                    "target_piecewise_capture": "/".join(
                        str(value) for value in graph["target_piecewise"]
                    ),
                    "target_full_capture": "/".join(
                        str(value) for value in graph["target_full"]
                    ),
                    "drafter_prefill_piecewise_capture": "/".join(
                        str(value) for value in graph["drafter_prefill_piecewise"]
                    ),
                    "drafter_prefill_full_capture": "/".join(
                        str(value) for value in graph["drafter_prefill_full"]
                    ),
                    "drafter_decode_full_capture": "/".join(
                        str(value) for value in graph["drafter_decode_full"]
                    ),
                    "job_id": str(graph["job_id"]),
                    "topology": (
                        "TP2/DP1/1-node"
                        if model == "super"
                        else "TP8/DP1/EP/Ray/2-node"
                    ),
                    "weights": "BF16",
                    "kv_cache": "FP8",
                    "vllm_version": "0.28.0",
                    "patchset_id": EXPECTED_PATCHSET[:12],
                    "container_artifact_sha256": str(
                        runtime["container_artifact_sha256"]
                    ),
                }
            )
    return rows, {
        "cuda_graph_evidence": evidence,
        "runtime_provenance": runtimes,
        "patchset_id": EXPECTED_PATCHSET[:12],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _metric(value: float | int | None, *, percentage: bool = False) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:.2%}" if percentage else f"{number:.4f}"


def render_html(
    rows: list[dict[str, Any]], provenance: dict[str, Any], *, csv_href: str
) -> str:
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['model']).title())}</td>"
        f"<td>{row['batch_size']}</td>"
        f"<td>{row['selected_k']}</td>"
        f"<td>{row['tok_s_gpu']:.2f}</td>"
        "<td>N/A</td>"
        f"<td>{_metric(row['acceptance_rate'], percentage=True)}</td>"
        f"<td>{_metric(row['mean_accepted_length'])}</td>"
        f"<td>{row['actual_output_tokens']}/{row['expected_output_tokens']}</td>"
        f"<td>{row['draft_counter_async_skew_tokens']:.0f}</td>"
        f"<td>{row['target_full_capture']}</td>"
        f"<td>{row['drafter_decode_full_capture']}</td>"
        f"<td><code>{row['job_id']}</code></td>"
        "</tr>"
        for row in rows
    )
    graph_rows = "".join(
        "<tr>"
        f"<td>{model.title()}</td>"
        f"<td><code>{row['job_id']}</code></td>"
        f"<td>{row['target_piecewise'][0]}/{row['target_piecewise'][1]}</td>"
        f"<td>{row['target_full'][0]}/{row['target_full'][1]}</td>"
        f"<td>{row['drafter_prefill_piecewise'][0]}/{row['drafter_prefill_piecewise'][1]}</td>"
        f"<td>{row['drafter_prefill_full'][0]}/{row['drafter_prefill_full'][1]}</td>"
        f"<td>{row['drafter_decode_full'][0]}/{row['drafter_decode_full'][1]}</td>"
        "</tr>"
        for model, row in provenance["cuda_graph_evidence"].items()
    )
    patch_id = html.escape(str(provenance["patchset_id"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vLLM 0.28 Nemotron-3 BF16 patched MRV2 Dynamic-K canary</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#5d687d;--line:#d9dfeb;--accent:#5b8c00;--warn:#8a5700}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:32px 20px 64px}}h1{{margin:0 0 8px;font-size:30px}}h2{{margin-top:0}}section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:18px}}.muted{{color:var(--muted)}}.ok{{color:var(--accent);font-weight:700}}.warn{{color:var(--warn)}}code{{font-size:.9em}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#eef2f7}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#111827;color:#e8edf7;padding:14px;border-radius:8px}}
</style></head><body><main>
<h1>vLLM 0.28 Nemotron-3 BF16 patched MRV2 Dynamic-K canary</h1>
<p class="muted">Preliminary single-repeat standalone benchmark · ISL/OSL 1000/128 · patchset <code>{patch_id}</code></p>
<section><h2>Status</h2><p class="ok">Super and Ultra completed with SLURM exit code 0:0, exact-token generation, and all target/drafter CUDA Graph captures.</p><p>No matched MRV2 K0 baseline was run in this canary, so throughput speedup is intentionally N/A. BS4 includes first-use JIT latency and all throughput rows require representative repeats before performance publication.</p></section>
<section><h2>DynamicMTP configuration</h2><pre>{{
  "method": "mtp",
  "num_speculative_tokens": 5,
  "num_speculative_tokens_per_batch_size": [[1,1,5],[2,2,3],[3,4,2],[5,8,1],[9,512,0]],
  "runner": "MRV2",
  "cudagraph_mode": "FULL_AND_PIECEWISE"
}}</pre><p>Super: BF16/FP8, TP2/DP1, one GB200 node. Ultra: BF16/FP8, TP8/DP1/EP/Ray, two GB200 nodes.</p></section>
<section><h2>Results</h2><div class="scroll"><table><thead><tr><th>Model</th><th>BS</th><th>K</th><th>tok/s/GPU</th><th>Speedup</th><th>Acceptance</th><th>Mean accepted length</th><th>Tokens</th><th>Counter skew</th><th>Target FULL</th><th>Drafter decode FULL</th><th>Job</th></tr></thead><tbody>{table_rows}</tbody></table></div><p class="warn">K0 acceptance is N/A. Prometheus SpecDec counters update asynchronously; the validator permits at most one offered batch at max-K width, records the skew per row, and still rejects substantive max-K work hidden under reduced K. Publication sweeps use one fresh engine per workload cell.</p></section>
<section><h2>CUDA Graph evidence</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Job</th><th>Target PIECEWISE</th><th>Target FULL</th><th>Drafter prefill PIECEWISE</th><th>Drafter prefill FULL</th><th>Drafter decode FULL</th></tr></thead><tbody>{graph_rows}</tbody></table></div></section>
<section><h2>Patch stack and known issues</h2><p>vLLM 0.28.0 base <code>2cf0a6915ce544dc493a0990f2ea38d81601128a</code>. Required stack: #49652 fixes DynamicSD draft-decode graph capture, <strong>#51575</strong> propagates scheduler-selected K and handles K0, #52548 reduces physical draft work for positive K, and a local Mamba guard patch permits reduced query lengths. Optional #53426 was excluded because skipping the K0 sync forward needs separate Nemotron state/acceptance validation.</p><p>The patched image SHA256 is <code>5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066</code>.</p></section>
<section><h2>Downloads</h2><p><a href="{html.escape(csv_href)}">Canonical MRV2 canary CSV</a></p></section>
</main></body></html>"""


def build_report(
    *, artifact_root: Path, output_csv: Path, output_html: Path
) -> list[dict[str, Any]]:
    rows, provenance = load_report_data(artifact_root)
    _write_csv(output_csv, rows)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    csv_href = "../data/vllm028_nemotron3_bf16_mrv2_patched_canary_20260828/results.csv"
    output_html.write_text(
        render_html(rows, provenance, csv_href=csv_href), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "mrv2_patch_canary",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    rows = build_report(
        artifact_root=args.artifact_root,
        output_csv=args.output_csv,
        output_html=args.output_html,
    )
    print(json.dumps({"rows": len(rows), "patchset_id": EXPECTED_PATCHSET[:12]}))


if __name__ == "__main__":
    main()
