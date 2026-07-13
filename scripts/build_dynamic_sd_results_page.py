#!/usr/bin/env python3
"""Build docs/dynamic_sd_sync_rollout_results_latest.html from experiment report assets.

Copies plots/data from experiments/dynamic_sd_sync_rollout/report/ into
docs/dynamic_sd_plots/ and docs/dynamic_sd_data/, then renders a self-contained
HTML page (seaborn PNG charts + summary tables). Rerun after every harvest.
"""

from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dynamic_sd_sync_rollout"
REPORT_DATA = EXP / "report" / "data"
REPORT_PLOTS = EXP / "report" / "plots"
TABLES_DIR = EXP / "tables"
DOCS = ROOT / "docs"
PLOTS_OUT = DOCS / "dynamic_sd_plots"
DATA_OUT = DOCS / "dynamic_sd_data"
PAGE = DOCS / "dynamic_sd_sync_rollout_results_latest.html"

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px auto;
       max-width: 1180px; color: #192133; padding: 0 16px; }
h1 { font-size: 26px; } h2 { font-size: 20px; margin-top: 34px;
     border-bottom: 2px solid #192133; padding-bottom: 4px; }
p.note { color: #444; }
table { border-collapse: collapse; margin: 12px 0; font-size: 14px; }
th, td { border: 1px solid #8a93a6; padding: 5px 10px; text-align: right; }
th { background: #eef1f6; } td:first-child, th:first-child { text-align: left; }
.chart-card { display: inline-block; vertical-align: top; margin: 6px 14px 6px 0; }
.chart-card img { max-width: 450px; width: 100%; border: 1px solid #d4d9e2; }
.chart-wide { display: block; margin: 8px 0; }
.chart-wide img { max-width: 1100px; width: 100%; border: 1px solid #d4d9e2; }
code { background: #eef1f6; padding: 1px 5px; }
.tag { display: inline-block; background: #eef1f6; border: 1px solid #8a93a6;
       padding: 1px 8px; margin-right: 6px; font-size: 12px; }
"""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value or "-"


def img_cards(names: list[str], wide: bool = False) -> str:
    cls = "chart-wide" if wide else "chart-card"
    cards = []
    for name in names:
        cards.append(
            f'<div class="{cls}"><img src="dynamic_sd_plots/{name}" '
            f'alt="{html.escape(name)}"></div>'
        )
    return "\n".join(cards)


def render_markdown_lite(md_path: Path) -> str:
    """Minimal markdown -> HTML for the patch ledger (headers, pipe tables,
    fenced code, paragraphs)."""
    if not md_path.exists():
        return ""
    out: list[str] = []
    table: list[str] = []
    code: list[str] | None = None
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p class='note'>" + html.escape(" ".join(para)) + "</p>")
            para.clear()

    def flush_table() -> None:
        if not table:
            return
        rows = [
            [c.strip() for c in line.strip().strip("|").split("|")]
            for line in table
            if not set(line.replace("|", "").strip()) <= {"-", " ", ":"}
        ]
        out.append("<table>")
        for idx, cells in enumerate(rows):
            tag = "th" if idx == 0 else "td"
            out.append(
                "<tr>"
                + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells)
                + "</tr>"
            )
        out.append("</table>")
        table.clear()

    for line in md_path.read_text(encoding="utf-8").splitlines():
        if code is not None:
            if line.startswith("```"):
                out.append(
                    "<pre style='background:#eef1f6;padding:8px;font-size:12px;"
                    "overflow-x:auto'>" + html.escape("\n".join(code)) + "</pre>"
                )
                code = None
            else:
                code.append(line)
            continue
        if line.startswith("```"):
            flush_para()
            flush_table()
            code = []
        elif line.startswith("|"):
            flush_para()
            table.append(line)
        elif line.startswith("# "):
            flush_para()
            flush_table()
        elif line.startswith("## "):
            flush_para()
            flush_table()
            out.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("### "):
            flush_para()
            flush_table()
            out.append(f"<h4>{html.escape(line[4:])}</h4>")
        elif line.strip() in ("", "---"):
            flush_para()
            flush_table()
        else:
            flush_table()
            para.append(line.strip())
    flush_para()
    flush_table()
    return "\n".join(out)


def tables_section() -> str:
    rows = []
    for spec_path in sorted(TABLES_DIR.glob("*_dynamic_spec.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        ranges = spec.get("num_speculative_tokens_per_batch_size", [])
        pretty = ", ".join(f"BS {lo}-{hi} &rarr; K={k}" for lo, hi, k in ranges)
        rows.append(
            f"<tr><td>{html.escape(spec_path.stem.replace('_dynamic_spec', ''))}</td>"
            f"<td style='text-align:left'>{pretty}</td></tr>"
        )
    if not rows:
        return "<p class='note'>No derived tables yet.</p>"
    return (
        "<table><tr><th>Setting</th><th>Derived batch-size &rarr; K schedule "
        "(argmax measured tok/s)</th></tr>" + "".join(rows) + "</table>"
    )


def rollout_table() -> str:
    rows = read_csv_rows(REPORT_DATA / "rollout_summary.csv")
    if not rows:
        return "<p class='note'>Rollout results pending.</p>"
    baselines = {
        (r["model"], r["bench"]): float(r["mean_step_wall_s"])
        for r in rows
        if r["variant"] == "baseline"
    }
    out = [
        "<table><tr><th>Model</th><th>Bench</th><th>Variant</th>"
        "<th>Mean step wall (s)</th><th>Mean gen tok/s</th>"
        "<th>Tokens/s/GPU</th><th>Speedup vs baseline</th></tr>"
    ]
    for r in sorted(rows, key=lambda x: (x["model"], x["bench"], x["variant"])):
        base = baselines.get((r["model"], r["bench"]))
        speedup = f"{base / float(r['mean_step_wall_s']):.3f}x" if base else "-"
        out.append(
            f"<tr><td>{html.escape(r['model'])}</td><td>{html.escape(r['bench'])}</td>"
            f"<td>{html.escape(r['variant'])}</td><td>{fmt(r['mean_step_wall_s'])}</td>"
            f"<td>{fmt(r['mean_output_tok_s'])}</td>"
            f"<td>{fmt(r.get('mean_output_tok_s_per_gpu', ''))}</td><td>{speedup}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)



def rollout_table_rows(model_filter: str | None = None) -> str:
    rows = read_csv_rows(REPORT_DATA / "rollout_summary.csv")
    if model_filter is not None:
        rows = [r for r in rows if slug_model(r["model"]) == model_filter]
    if not rows:
        return ""
    baselines = {
        (r["model"], r["bench"]): float(r["mean_step_wall_s"])
        for r in rows
        if r["variant"] == "baseline"
    }
    out = [
        "<table><tr><th>Bench</th><th>Variant</th>"
        "<th>Mean step wall (s)</th><th>Tokens/s/GPU</th><th>Speedup</th></tr>"
    ]
    for r in sorted(rows, key=lambda x: (x["bench"], x["variant"])):
        base = baselines.get((r["model"], r["bench"]))
        speedup = f"{base / float(r['mean_step_wall_s']):.3f}x" if base else "-"
        out.append(
            f"<tr><td>{html.escape(r['bench'])}</td>"
            f"<td>{html.escape(r['variant'])}</td><td>{fmt(r['mean_step_wall_s'])}</td>"
            f"<td>{fmt(r.get('mean_output_tok_s_per_gpu', ''))}</td><td>{speedup}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def slug_model(label: str) -> str:
    return (
        label.lower().replace(" ", "_").replace("/", "_")
        .replace("(", "").replace(")", "")
    )


def build() -> None:
    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    plot_names = []
    for png in sorted(REPORT_PLOTS.glob("*.png")):
        shutil.copy2(png, PLOTS_OUT / png.name)
        plot_names.append(png.name)
    for csv_file in sorted(REPORT_DATA.glob("*.csv")):
        shutil.copy2(csv_file, DATA_OUT / csv_file.name)

    accept_imgs = [n for n in plot_names if n.startswith("profile_acceptance")]
    rollout_imgs = [n for n in plot_names if n.startswith("rollout_")]

    model_slugs = [
        (
            "qwen3-30b-a3b_40k",
            "Qwen3-30B-A3B &mdash; 40K long-tail (TP2, 32K max_tokens)",
        ),
        ("qwen3-30b-a3b", "Qwen3-30B-A3B (TP1)"),
        ("qwen3-32b", "Qwen3-32B (TP2)"),
        ("qwen3-235b-a22b", "Qwen3-235B-A22B (TP4)"),
        (
            "nemotron3-super-120b_fp8",
            "Nemotron3-Super-120B FP8 &mdash; in-checkpoint MTP (TP4)",
        ),
        (
            "nemotron3-ultra-550b_nvfp4",
            "Nemotron3-Ultra-550B NVFP4 &mdash; in-checkpoint MTP (TP4)",
        ),
    ]

    def model_of(name: str) -> str:
        for slug_key, _ in model_slugs:
            if slug_key in name:
                return slug_key
        return "other"

    model_sections = []
    claimed: set[str] = set()
    for slug_key, title in model_slugs:
        grids = [
            n
            for n in plot_names
            if n.startswith("profile_tok_s_") and model_of(n) == slug_key
        ]
        speedups = [
            n
            for n in plot_names
            if n.startswith("profile_speedup_per_gpu_") and model_of(n) == slug_key
        ]
        drains = [
            n for n in plot_names if n.startswith("drain_") and model_of(n) == slug_key
        ]
        rollouts = [
            n for n in plot_names
            if n.startswith("rollout_") and model_of(n) == slug_key
        ]
        claimed.update(grids + speedups + drains + rollouts)
        if not (grids or speedups or drains or rollouts):
            continue
        parts = [f"<h2>{title}</h2>"]
        if rollouts:
            parts.append("<h3>Sync rollout: baseline vs fixed-K vs DynamicSD</h3>")
            parts.append(img_cards(rollouts))
            raw = rollout_table_rows(slug_key)
            if raw:
                parts.append(
                    "<details><summary>Raw rollout numbers for this model"
                    "</summary>" + raw + "</details>"
                )
        if grids:
            parts.append("<h3>Tokens/s across batch size &times; K</h3>")
            parts.append(img_cards(grids))
        if speedups:
            parts.append("<h3>Tokens/s/GPU speedup vs no-SD (dashed = break-even)</h3>")
            parts.append(img_cards(speedups))
        if drains:
            parts.append(
                "<h3>Rollout drain curves (sequences in flight over time)</h3>"
            )
            parts.append(img_cards(drains))
        model_sections.append("\n".join(parts))
    per_model_html = "\n".join(model_sections)

    data_links = " ".join(
        f'<a class="tag" href="dynamic_sd_data/{f.name}">{f.name}</a>'
        for f in sorted(DATA_OUT.glob("*.csv"))
    )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>DynamicSD under Synchronous RL Rollout (vLLM 0.24)</title>
<style>{CSS}</style></head><body>
<h1>DynamicSD under Synchronous RL Rollout (vLLM 0.24)</h1>
<p class="note">Qwen3-30B-A3B / Qwen3-32B / Qwen3-235B-A22B with RedHatAI EAGLE3
Thinking speculators on Lyris GB200 (vLLM 0.24.0, temperature 1.0, top_p 1.0,
seed 42). Rollout shapes mirror one vLLM DP-worker shard of the NeMo-RL GB200
SyncRL performance recipes (<code>*4g.yaml</code>): N prompts &times; 32
generations per step with barrier semantics. DynamicSD =
<code>speculative_config.num_speculative_tokens_per_batch_size</code>, the
batch-size&rarr;K schedule derived from the Phase-1 grid below.</p>

<h2>Cross-model summary &mdash; synchronous rollout: baseline vs fixed-K vs DynamicSD</h2>
<p class="note">Per-engine sync rollout (N&times;32 generations, barrier per
step). Charts and table are per model &times; benchmark; tokens/s/GPU
normalizes TP differences (TP1/2/4).</p>
<details><summary><b>Full cross-model summary table (click to expand)</b></summary>
{rollout_table()}
</details>
<p class="note"><b>Capture-cliff lesson:</b> the first dynamic tables carried
K=5 into BS 86-127 where bs&times;(K+1) &gt; 512 exceeds the cudagraph capture
budget, forcing eager-mode decode (openmath dynamic 37.4s vs fixed-K3 25.4s).
Tables are now derived with an analytic bs&times;(K+1) &le; capture-budget cap;
capture-aware dynamic recovers to within ~5% of fixed-K3 on 30B-A3B
(1.87-1.90x vs 2.00x). In this 4K-generation regime most wall time sits at
high concurrency where K=3 is already optimal, so fixed-K3 keeps a small edge -
the settings where the derived schedule turns speculation off at high BS
(Qwen3-32B SWE, Qwen3-235B) and the 32K long-tail preset are where DynamicSD
is expected to pull ahead.</p>
<p class="note"><b>Deeper K is not the memory-bound answer here:</b> K=7
raises acceptance length to 4.11 (vs 3.73 at K=5) but per-position acceptance
decay means tokens/s never beats K=5, and at BS=1 plain K=3 is fastest
(607 vs 590/556 tok/s). The derived schedules therefore never select K&gt;5.</p>

<h2>Acceptance length by model (temperature 1.0)</h2>
{img_cards(accept_imgs)}

<h2>Derived DynamicSD schedules</h2>
{tables_section()}

<h2>Draft sampling: greedy vs probabilistic</h2>
<p class="note">vLLM 0.24's <code>draft_sample_method="probabilistic"</code>
(stochastic drafter sampling with cached draft logits for exact rejection
sampling) showed no acceptance-length gain over greedy drafting on
Qwen3-30B-A3B/openmath (AL 2.99 vs 3.01 at K=3) and 3-10% lower tok/s from the
logits-caching overhead, so the main matrix uses greedy drafting.</p>

<h2>vLLM patch &amp; change tracking (0.24 vs 0.25, per-change perf impact)</h2>
{render_markdown_lite(EXP / "PATCH_LEDGER.md")}

<h1>Per-model results</h1>
<p class="note">Profiling grids use fixed-length generation (ignore_eos); each
K is a separate engine, K=0 disables speculation. The K=5 collapse at BS=128 is
the cudagraph-capture cliff (128&times;6 = 768 tokens/step &gt; default 512 max
capture size) - exactly the regime cost DynamicSD avoids. Drain curves show
sequences still in flight within a rollout step; the long low-concurrency tail
is where DynamicSD raises K beyond the fixed-K compromise.</p>
{per_model_html}

<h2>Data</h2>
<p>{data_links}</p>
<p class="note">Source: <code>experiments/dynamic_sd_sync_rollout/</code>
(harness, tables, report data). Jobs run in
<code>vllm-benchmark/dynamic_sd_runs/</code> on Lyris.</p>
</body></html>
"""
    PAGE.write_text(page, encoding="utf-8")
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    build()
