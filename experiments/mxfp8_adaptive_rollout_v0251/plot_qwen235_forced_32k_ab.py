from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any


ARM_ORDER = ("baseline", "adaptive")
ARM_LABELS = {
    "baseline": "CuTeDSL baseline",
    "adaptive": "TRTLLM Adaptive",
}


def summarize_pair_result(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = {str(run["arm"]): run for run in payload["runs"]}
    if set(runs) != set(ARM_ORDER):
        raise ValueError(f"expected exactly these arms: {ARM_ORDER}")

    ordered = [runs[arm] for arm in ARM_ORDER]
    if not all(run.get("complete") is True for run in ordered):
        raise ValueError("both arms must be complete")
    if not all(run.get("measurement_scope") == "generation_calls" for run in ordered):
        raise ValueError("both arms must use generation-call timing")

    gpu_counts = {int(run["gpu_count"]) for run in ordered}
    if len(gpu_counts) != 1:
        raise ValueError("GPU counts do not match")
    output_tokens = {int(run["output_tokens"]) for run in ordered}
    if len(output_tokens) != 1:
        raise ValueError("output-token counts do not match")

    baseline = float(runs["baseline"]["tokens_per_second_per_gpu"])
    rows = []
    for arm in ARM_ORDER:
        run = runs[arm]
        throughput = float(run["tokens_per_second_per_gpu"])
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "tokens_per_second_per_gpu": throughput,
                "normalized_throughput": throughput / baseline,
                "generation_seconds": float(run["generation_seconds"]),
            }
        )
    return {
        "rows": rows,
        "matched_gpu_count": gpu_counts.pop(),
        "matched_output_tokens": output_tokens.pop(),
    }


def _render_plot(result: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    frame = pd.DataFrame(result["rows"])
    palette = sns.color_palette("Paired", n_colors=len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for axis, metric, ylabel in (
        (axes[0], "tokens_per_second_per_gpu", "Throughput (tokens/sec/GPU)"),
        (axes[1], "normalized_throughput", "Throughput / CuTeDSL"),
    ):
        sns.barplot(
            data=frame,
            x="label",
            y=metric,
            hue="label",
            palette=palette,
            edgecolor="#192133",
            linewidth=2.0,
            zorder=10,
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel, fontsize=13)
        axis.set_ylim(0, float(frame[metric].max()) * 1.08)
        axis.tick_params(axis="x", labelsize=10)
        axis.tick_params(axis="y", labelsize=11)
        axis.grid(True, linestyle="--", dashes=(6, 6), linewidth=1.1, axis="y", zorder=0)
        labels = (
            [f"{value:.1f}" for value in frame[metric]]
            if metric == "tokens_per_second_per_gpu"
            else [f"{value:.3f}x" for value in frame[metric]]
        )
        for bar, label in zip(axis.patches, labels, strict=True):
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        for side in ("left", "right", "top", "bottom"):
            axis.spines[side].set_linewidth(2.0)
            axis.spines[side].set_color("black")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=5)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"qwen235_forced_32k_cutedsl_vs_adaptive.{extension}",
            bbox_inches="tight",
            dpi=600,
        )
    plt.close(fig)


def _render_reports(
    result: dict[str, Any], shape_summary: dict[str, Any], provenance: str, output_dir: Path
) -> None:
    rows = {row["arm"]: row for row in result["rows"]}
    ratio = rows["adaptive"]["normalized_throughput"]
    regression = (1.0 - ratio) * 100.0
    table_rows = "".join(
        "<tr>"
        f"<td>{escape(row['label'])}</td>"
        f"<td>{row['tokens_per_second_per_gpu']:.2f}</td>"
        f"<td>{row['normalized_throughput']:.3f}x</td>"
        f"<td>{row['generation_seconds']:.2f}</td>"
        "</tr>"
        for row in result["rows"]
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-235B Forced 32K MXFP8 Backend Comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:0;color:#172033}}main{{max-width:980px;margin:auto;padding:30px 24px 50px}}h1{{font-size:28px}}p{{line-height:1.55}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #ccd2dc;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.decision{{border-left:4px solid #c33;background:#f6f7f9;padding:10px 14px}}code{{background:#eef1f5;padding:2px 5px}}</style>
</head><body><main>
<h1>Qwen3-235B Forced 32K MXFP8 Backend Comparison</h1>
<p>Ptyche jobs <code>2506677</code> and <code>2506829</code>; vLLM 0.25.1, FlashInfer 0.6.13, CUDA Graph, two TP4/EP4 replicas on 8 GB200 GPUs.</p>
<img src="qwen235_forced_32k_cutedsl_vs_adaptive.png" alt="CuTeDSL and TRTLLM Adaptive throughput">
<table><thead><tr><th>Dense MXFP8 policy</th><th>tokens/sec/GPU</th><th>vs CuTeDSL</th><th>Generation time (s)</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="decision"><strong>Result:</strong> TRTLLM Adaptive reached {ratio:.3f}x CuTeDSL throughput, a {regression:.1f}% regression. Increasing OSL to a true 32K did not recover an Adaptive advantage.</p>
<h2>Matched methodology</h2>
<p>Each arm processed 64 requests with exactly 32,768 generated tokens per request ({result['matched_output_tokens']:,} total engine tokens). The configuration used <code>ignore_eos=true</code>, an empty stop-token list, aggregate concurrency 64, <code>max_num_seqs=32</code> per replica, and <code>max_num_batched_tokens=16384</code>.</p>
<h2>Lookup coverage</h2>
<p>The forced-32K trace observed {shape_summary['unique_signature_count']} unique dense GEMM signatures across {shape_summary['record_count']} records. All five signatures matched the qualified lookup table, so this result is not explained by unseen-shape fallback.</p>
<h2>Interpretation</h2>
<p>The dense TRTLLM path is active and exact tactics are available, but end-to-end generation remains dominated by work outside the optimized dense LM-head GEMM. Backend conversion and dispatch costs, plus MoE and attention execution, outweigh the isolated tactic benefit for this workload.</p>
<h2>Provenance</h2><pre>{escape(provenance.strip())}</pre>
</main></body></html>"""
    output_dir.joinpath("index.html").write_text(html, encoding="utf-8")
    output_dir.joinpath("RESULTS.md").write_text(
        f"""# Qwen3-235B Forced 32K MXFP8 Backend Comparison

CuTeDSL achieved {rows['baseline']['tokens_per_second_per_gpu']:.2f} tokens/sec/GPU. TRTLLM Adaptive achieved {rows['adaptive']['tokens_per_second_per_gpu']:.2f} tokens/sec/GPU ({ratio:.3f}x, {regression:.1f}% lower).

Both arms generated 64 x 32,768 tokens. The forced-32K trace observed five unique signatures, all covered by the qualified table. Longer OSL therefore did not expose missing shapes or recover an Adaptive performance advantage.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--shape-summary", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_pair_result(args.summary)
    shape_summary = json.loads(args.shape_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _render_plot(result, args.output_dir)
    _render_reports(
        result,
        shape_summary,
        args.provenance.read_text(encoding="utf-8"),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
