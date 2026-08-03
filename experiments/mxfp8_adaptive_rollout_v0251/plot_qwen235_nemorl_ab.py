from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any


ARM_ORDER = ("baseline", "trtllm_default", "adaptive")
ARM_LABELS = {
    "baseline": "CuTeDSL",
    "trtllm_default": "TRTLLM default",
    "adaptive": "Complete-table adaptive",
}


def summarize_three_arm_result(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = {str(run["arm"]): run for run in payload["runs"]}
    if set(runs) != set(ARM_ORDER):
        raise ValueError(f"expected exactly these arms: {ARM_ORDER}")

    ordered = [runs[arm] for arm in ARM_ORDER]
    if not all(run.get("complete") is True for run in ordered):
        raise ValueError("all arms must be complete")
    if not all(run.get("measurement_scope") == "generation_calls" for run in ordered):
        raise ValueError("all arms must use generation-call timing")

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
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    sns.barplot(
        data=frame,
        x="label",
        y="tokens_per_second_per_gpu",
        hue="label",
        palette=palette,
        edgecolor="#192133",
        linewidth=2.0,
        zorder=10,
        legend=False,
        ax=axes[0],
    )
    sns.barplot(
        data=frame,
        x="label",
        y="normalized_throughput",
        hue="label",
        palette=palette,
        edgecolor="#192133",
        linewidth=2.0,
        zorder=10,
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(1.0, linestyle="--", linewidth=1.1, color="black", zorder=5)

    axes[0].set_ylabel("Throughput (tokens/sec/GPU)", fontsize=13)
    axes[1].set_ylabel("Throughput / CuTeDSL", fontsize=13)
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelsize=10, rotation=12)
        axis.tick_params(axis="y", labelsize=11)
        axis.grid(
            True,
            linestyle="--",
            dashes=(6, 6),
            linewidth=1.1,
            axis="y",
            zorder=0,
        )
        for side in ("left", "right", "top", "bottom"):
            axis.spines[side].set_linewidth(2.0)
            axis.spines[side].set_color("black")

    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"qwen235_nemorl_three_arm_throughput.{extension}",
            bbox_inches="tight",
            dpi=600,
        )
    plt.close(fig)


def _render_text_reports(
    result: dict[str, Any],
    provenance: str,
    job_id: str,
    output_dir: Path,
) -> None:
    rows = result["rows"]
    by_arm = {row["arm"]: row for row in rows}
    adaptive_ratio = by_arm["adaptive"]["normalized_throughput"]
    default_ratio = by_arm["trtllm_default"]["normalized_throughput"]
    adaptive_vs_default = (
        by_arm["adaptive"]["tokens_per_second_per_gpu"]
        / by_arm["trtllm_default"]["tokens_per_second_per_gpu"]
    )
    decision = (
        "Do not enable the complete-table adaptive policy for this workload."
        if adaptive_ratio < 1.0
        else "Advance the adaptive policy to the matched correctness gate."
    )
    markdown_rows = "\n".join(
        f"| {row['label']} | {row['tokens_per_second_per_gpu']:.2f} | "
        f"{row['normalized_throughput']:.3f}x | {row['generation_seconds']:.3f} |"
        for row in rows
    )
    results_md = f"""# Qwen3-235B MXFP8 Adaptive Canary

Ptyche job `{job_id}` used vLLM 0.25.1, FlashInfer 0.6.13, two TP4/EP4
rollout replicas, and CUDA Graph execution. Every arm generated
{result['matched_output_tokens']:,} output tokens on {result['matched_gpu_count']} GPUs.

| Backend policy | tokens/sec/GPU | vs CuTeDSL | Generation time (s) |
|---|---:|---:|---:|
{markdown_rows}

Complete-table adaptive was `{adaptive_ratio:.3f}x` versus CuTeDSL and
`{adaptive_vs_default:.3f}x` versus TRTLLM default. TRTLLM default was
`{default_ratio:.3f}x` versus CuTeDSL.

**Decision:** {decision}

The offline table covered all five observed signatures and passed numerical and
CUDA Graph microbenchmark gates. This single matched run does not establish a
production gain, and the GSM8K promotion gate was therefore not executed.

## Provenance

```text
{provenance.strip()}
```
"""
    output_dir.joinpath("RESULTS.md").write_text(results_md, encoding="utf-8")

    table_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['label']))}</td>"
        f"<td>{row['tokens_per_second_per_gpu']:.2f}</td>"
        f"<td>{row['normalized_throughput']:.3f}x</td>"
        f"<td>{row['generation_seconds']:.3f}</td>"
        "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-235B MXFP8 Adaptive Canary</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;color:#172033;background:#fff}}main{{max-width:1050px;margin:auto;padding:32px 24px 56px}}
h1{{font-size:30px;margin:0 0 8px}}p{{line-height:1.55}}img{{width:100%;height:auto;margin:18px 0 24px}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border-bottom:1px solid #ccd2dc;padding:10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.decision{{border-left:4px solid #d44;padding:10px 14px;background:#f7f8fa}}code{{background:#f1f3f6;padding:2px 5px}}
</style></head><body><main>
<h1>Qwen3-235B MXFP8 Adaptive Canary</h1>
<p>Ptyche job <code>{escape(job_id)}</code>, vLLM 0.25.1, FlashInfer 0.6.13, CUDA Graph, two TP4/EP4 replicas.</p>
<img src="qwen235_nemorl_three_arm_throughput.png" alt="Absolute and normalized generation throughput">
<table><thead><tr><th>Backend policy</th><th>tokens/sec/GPU</th><th>vs CuTeDSL</th><th>Generation time (s)</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="decision"><strong>Decision:</strong> {escape(decision)}</p>
<p>The complete offline table covered all five observed GEMM signatures and passed numerical and CUDA Graph microbenchmark gates. Since this matched end-to-end run did not improve on CuTeDSL, GSM8K was not promoted for this candidate.</p>
</main></body></html>
"""
    output_dir.joinpath("index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = summarize_three_arm_result(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = args.provenance.read_text(encoding="utf-8")
    _render_plot(result, args.output_dir)
    _render_text_reports(result, provenance, args.job_id, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
