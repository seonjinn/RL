"""Render matched 1K/1K active-concurrency results after validation."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vllm029_active_trend_results.json"
PNG = ROOT / "vllm029_active_trend.png"
PDF = ROOT / "vllm029_active_trend.pdf"
HTML = ROOT / "vllm029-active-trend.html"
BF16 = "#47545c"
MXFP8 = "#157f86"


def paired_records(data: dict) -> list[tuple[int, dict, dict]]:
    records = {(record["arm"], record["concurrency"]): record for record in data["records"]}
    pairs = []
    for concurrency in (64, 128, 256, 512, 1024):
        bf16 = records.get(("bf16", concurrency))
        mxfp8 = records.get(("moe", concurrency))
        if bf16 is not None and mxfp8 is not None:
            pairs.append((concurrency, bf16, mxfp8))
    return pairs


def make_figure(pairs: list[tuple[int, dict, dict]]) -> None:
    valid = [(c, bf16, mxfp8) for c, bf16, mxfp8 in pairs if bf16["reached_target"] and mxfp8["reached_target"]]
    if not valid:
        raise ValueError("No matched pair reached its configured active concurrency")
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=(7.5, 5.3), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for arm, color, label in (("bf16", BF16, "BF16"), ("moe", MXFP8, "Routed-expert MXFP8")):
        xs = [c for c, _, _ in valid]
        records = [bf16 if arm == "bf16" else mxfp8 for _, bf16, mxfp8 in valid]
        ys = [record["mean_output_tok_s_gpu"] for record in records]
        ax.plot(xs, ys, color=color, marker="o", markersize=6, linewidth=2.2, label=label, zorder=3)
        for x, record in zip(xs, records, strict=True):
            values = record["output_tok_s_gpu"]
            ax.vlines(x, min(values), max(values), color=color, alpha=0.65, linewidth=2.3, zorder=2)
            ax.scatter([x] * len(values), values, color=color, s=13, alpha=0.55, zorder=4)
    ratios = [mxfp8["mean_output_tok_s_gpu"] / bf16["mean_output_tok_s_gpu"] for _, bf16, mxfp8 in valid]
    ratio_ax.plot([c for c, _, _ in valid], ratios, color=MXFP8, marker="o", markersize=6, linewidth=2.2)
    ratio_ax.axhline(1, color="#87949a", linewidth=1.1, linestyle="--")
    ax.set_ylabel("Output tokens/s/GPU")
    ratio_ax.set_ylabel("MXFP8 / BF16")
    ratio_ax.set_xlabel("Observed active concurrency target")
    for axis in (ax, ratio_ax):
        axis.set_xscale("log", base=2)
        axis.grid(axis="y", color="#dce4e5", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    ratio_ax.set_xticks([c for c, _, _ in valid], [str(c) for c, _, _ in valid])
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    fig.suptitle("SuperVL3.5 · vLLM 0.29 active-concurrency trend", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=600, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)


def make_html(data: dict, pairs: list[tuple[int, dict, dict]]) -> None:
    rows = []
    for concurrency, bf16, mxfp8 in pairs:
        ratio = mxfp8["mean_output_tok_s_gpu"] / bf16["mean_output_tok_s_gpu"]
        reached = bf16["reached_target"] and mxfp8["reached_target"]
        rows.append(
            "<tr>"
            f"<td>{concurrency}</td><td>{bf16['max_running']} / {mxfp8['max_running']}</td>"
            f"<td>{bf16['mean_output_tok_s_gpu']:.1f}</td><td>{mxfp8['mean_output_tok_s_gpu']:.1f}</td>"
            f"<td>{ratio:.3f}x</td><td>{'yes' if reached else 'no'}</td>"
            f"<td>{bf16['cudagraph_capture_max']} / {mxfp8['cudagraph_capture_max']}</td>"
            f"<td>{html.escape(bf16['job_id'])} / {html.escape(mxfp8['job_id'])}</td>"
            "</tr>"
        )
    status = "Complete" if len(pairs) == 5 and not data["pending_jobs"] else "Partial"
    source_commit = html.escape(data["manifest"]["source_commit"])
    pending = ", ".join(data["pending_jobs"]) or "none"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperVL3.5 active-concurrency trend</title>
<style>body{{font:15px/1.5 Arial,sans-serif;color:#1d2c30;margin:0}}main{{max-width:940px;margin:auto;padding:28px 20px 56px}}h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:19px;margin-top:28px}}.meta{{color:#52656a}}img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:8px;border-bottom:1px solid #dbe3e4;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}code{{font-family:Menlo,monospace;font-size:13px}}a{{color:#126779}}</style>
</head><body><main>
<h1>SuperVL3.5: BF16 vs routed-expert MXFP8</h1>
<p class="meta">{status} · OCI-HSG GB200 · vLLM 0.29 · TP4/DP1/EP4 · 1,024 input / 1,024 output tokens</p>
<p>Each arm used the same BF16 checkpoint, synthetic prompt, 2,048 requests per trial and two timed trials. MXFP8 quantizes routed experts during model loading, outside the timed generation call. Lines show the two-trial mean output tokens/s/GPU; vertical spans and faint dots show the two individual trials. Points appear only when both logs show the configured active concurrency.</p>
<img src="vllm029_active_trend.png" alt="Output throughput and MXFP8/BF16 ratio by observed active concurrency">
<p><a href="vllm029_active_trend.pdf">PDF figure</a> · <a href="vllm029_active_trend_results.json">Validated JSON</a> · <a href="vllm029_active_trend_jobs.json">Job manifest</a></p>
<h2>Measured pairs</h2><div class="scroll"><table><thead><tr><th>Target C</th><th>Max Running BF16 / MXFP8</th><th>BF16 tok/s/GPU</th><th>MXFP8 tok/s/GPU</th><th>MXFP8 / BF16</th><th>Reached C</th><th>CUDA Graph max BF16 / MXFP8</th><th>Job IDs</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Pending jobs: {html.escape(pending)}. These are repeated measurements, not confidence intervals. The figure does not include model loading, online conversion, warmup, refit, or RL/Gym latency.</p>
<h2>Configuration</h2><p>One 4-GPU GB200 node per arm. FlashInfer attention, FlashInfer TRTLLM MoE, FlashInfer CuteDSL linear backend, chunked prefill and prefix caching. <code>max_num_seqs</code> equals each target C; <code>max_num_batched_tokens=65536</code>. vLLM PIECEWISE CUDA graphs and <code>enforce_eager=false</code>; capture sizes use vLLM 0.29 Blackwell defaults and are validated from each job log. The MXFP8 scope retains layer 0 and layers 80-87 in BF16, and excludes shared experts, vision/audio, MTP, router, and non-expert projections.</p>
<p>Source commit: <code>{source_commit}</code>. JSON validation checks vLLM version, model and prompt identity, backend, cache settings, scope, exactly 2,048 full-length outputs per trial, normal exit, and vLLM scheduler Running counts.</p>
</main></body></html>"""
    HTML.write_text(document, encoding="utf-8")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    pairs = paired_records(data)
    make_figure(pairs)
    make_html(data, pairs)
    print(f"Rendered {len(pairs)} matched pairs to {PNG} and {HTML}")


if __name__ == "__main__":
    main()
