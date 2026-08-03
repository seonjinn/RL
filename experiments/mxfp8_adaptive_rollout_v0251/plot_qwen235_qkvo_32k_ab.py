from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any


ARM_ORDER = ("baseline", "adaptive")
ARM_LABELS = {
    "baseline": "QKVO MXFP8 CuTeDSL",
    "adaptive": "QKV TRTLLM Adaptive",
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


def load_correctness_gate(gate_path: Path) -> dict[str, Any]:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("correctness gate did not pass")
    paired = payload.get("paired")
    if not isinstance(paired, dict):
        raise ValueError("correctness gate has no paired result")
    return {
        "matched_examples": int(payload["row_count"]),
        "baseline_accuracy": float(payload["baseline_accuracy"]),
        "adaptive_accuracy": float(payload["adaptive_accuracy"]),
        "absolute_accuracy_delta": float(payload["absolute_accuracy_delta"]),
        "adaptive_gains": int(paired["adaptive_gains"]),
        "adaptive_losses": int(paired["adaptive_losses"]),
        "one_sided_p_value": float(paired["one_sided_p_value"]),
        "passed": True,
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
        axis.set_ylim(0, float(frame[metric].max()) * 1.1)
        axis.tick_params(axis="x", labelsize=9, rotation=8)
        axis.tick_params(axis="y", labelsize=11)
        axis.grid(
            True,
            linestyle="--",
            dashes=(6, 6),
            linewidth=1.1,
            axis="y",
            zorder=0,
        )
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
            output_dir / f"qwen235_qkvo_32k_cutedsl_vs_adaptive.{extension}",
            bbox_inches="tight",
            dpi=600,
        )
    plt.close(fig)


def _correctness_html(correctness: dict[str, Any] | None) -> str:
    if correctness is None:
        return "<p>Matched GSM8K correctness evaluation is still in progress.</p>"
    return (
        "<p><strong>Passed:</strong> "
        f"{correctness['matched_examples']:,} matched GSM8K examples; "
        f"baseline accuracy {correctness['baseline_accuracy']:.4f}, "
        f"adaptive accuracy {correctness['adaptive_accuracy']:.4f}, "
        f"one-sided paired exact p={correctness['one_sided_p_value']:.4g}.</p>"
    )


def _render_reports(
    result: dict[str, Any],
    audit: dict[str, Any],
    correctness: dict[str, Any] | None,
    provenance: str,
    output_dir: Path,
) -> None:
    rows = {row["arm"]: row for row in result["rows"]}
    ratio = rows["adaptive"]["normalized_throughput"]
    delta = (ratio - 1.0) * 100.0
    table_rows = "".join(
        "<tr>"
        f"<td>{escape(row['label'])}</td>"
        f"<td>{row['tokens_per_second_per_gpu']:.2f}</td>"
        f"<td>{row['normalized_throughput']:.3f}x</td>"
        f"<td>{row['generation_seconds']:.2f}</td>"
        "</tr>"
        for row in result["rows"]
    )
    qualified = int(audit["qualified_shape_count"])
    observed = int(audit["observed_shape_count"])
    decision_class = "gain" if delta >= 0 else "regression"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-235B QKVO MXFP8 Adaptive Validation</title>
<style>body{{font-family:Arial,sans-serif;margin:0;color:#172033}}main{{max-width:980px;margin:auto;padding:30px 24px 50px}}h1{{font-size:28px}}p{{line-height:1.55}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #ccd2dc;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.decision{{border-left:4px solid #356f3b;background:#f6f7f9;padding:10px 14px}}code{{background:#eef1f5;padding:2px 5px}}pre{{white-space:pre-wrap}}</style>
</head><body><main>
<h1>Qwen3-235B QKVO MXFP8 Adaptive Validation</h1>
<p>vLLM 0.25.1, FlashInfer 0.6.13, CUDA Graph, TP4/EP4 on 8 GB200 GPUs. Both arms use MXFP8 for MoE experts and QKVO projections while leaving the router gate and LM head unquantized.</p>
<img src="qwen235_qkvo_32k_cutedsl_vs_adaptive.png" alt="QKVO CuTeDSL and TRTLLM Adaptive throughput">
<table><thead><tr><th>QKVO MXFP8 policy</th><th>tokens/sec/GPU</th><th>vs CuTeDSL</th><th>Generation time (s)</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="decision"><strong>Result:</strong> QKV TRTLLM Adaptive reached {ratio:.3f}x the matched QKVO CuTeDSL baseline ({delta:+.1f}%, {decision_class}).</p>
<h2>Matched methodology</h2>
<p>Each arm processed the same {result['matched_output_tokens']:,} output tokens on {result['matched_gpu_count']} GPUs. The run used 64 requests, 32,768 output tokens per request, <code>ignore_eos=true</code>, aggregate concurrency 64, <code>max_num_seqs=32</code> per replica, and <code>max_num_batched_tokens=16384</code>.</p>
<h2>Lookup qualification</h2>
<p>The serving trace observed {observed} QKVO GEMM signatures. Offline shmoo qualified {qualified}; the complete-family policy enabled the QKV family only. O-projection shapes fail closed to CuTeDSL because two low-M signatures did not satisfy the repeatability threshold.</p>
<h2>Correctness</h2>{_correctness_html(correctness)}
<h2>Provenance</h2><pre>{escape(provenance.strip())}</pre>
</main></body></html>"""
    output_dir.joinpath("index.html").write_text(html, encoding="utf-8")

    correctness_line = (
        "GSM8K correctness is still in progress."
        if correctness is None
        else (
            f"The matched {correctness['matched_examples']:,}-example GSM8K gate passed "
            f"(baseline {correctness['baseline_accuracy']:.4f}, adaptive "
            f"{correctness['adaptive_accuracy']:.4f}, one-sided p="
            f"{correctness['one_sided_p_value']:.4g})."
        )
    )
    output_dir.joinpath("RESULTS.md").write_text(
        f"""# Qwen3-235B QKVO MXFP8 Adaptive Validation

QKV TRTLLM Adaptive achieved {rows['adaptive']['tokens_per_second_per_gpu']:.2f} tokens/sec/GPU versus {rows['baseline']['tokens_per_second_per_gpu']:.2f} for the matched QKVO CuTeDSL baseline ({ratio:.3f}x, {delta:+.1f}%).

The trace observed {observed} QKVO GEMM signatures and the shmoo qualified {qualified}. The fail-closed complete-family policy enabled QKV only; O projection remained on CuTeDSL. {correctness_line}
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--correctness-gate", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = summarize_pair_result(args.summary)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    correctness = (
        load_correctness_gate(args.correctness_gate)
        if args.correctness_gate is not None
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("aggregate.json").write_text(
        json.dumps(
            {"performance": result, "correctness": correctness},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _render_plot(result, args.output_dir)
    _render_reports(result, audit, correctness, args.provenance.read_text(), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
