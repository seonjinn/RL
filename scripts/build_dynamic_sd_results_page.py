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
.chart-card { margin: 14px 0; }
.chart-card img { max-width: 100%; border: 1px solid #d4d9e2; }
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


def img_cards(names: list[str]) -> str:
    cards = []
    for name in names:
        cards.append(
            f'<div class="chart-card"><img src="dynamic_sd_plots/{name}" '
            f'alt="{html.escape(name)}"></div>'
        )
    return "\n".join(cards)


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
        "<th>Mean step wall (s)</th><th>Mean gen tok/s</th><th>Speedup vs baseline</th></tr>"
    ]
    for r in sorted(rows, key=lambda x: (x["model"], x["bench"], x["variant"])):
        base = baselines.get((r["model"], r["bench"]))
        speedup = f"{base / float(r['mean_step_wall_s']):.3f}x" if base else "-"
        out.append(
            f"<tr><td>{html.escape(r['model'])}</td><td>{html.escape(r['bench'])}</td>"
            f"<td>{html.escape(r['variant'])}</td><td>{fmt(r['mean_step_wall_s'])}</td>"
            f"<td>{fmt(r['mean_output_tok_s'])}</td><td>{speedup}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def build() -> None:
    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    plot_names = []
    for png in sorted(REPORT_PLOTS.glob("*.png")):
        shutil.copy2(png, PLOTS_OUT / png.name)
        plot_names.append(png.name)
    for csv_file in sorted(REPORT_DATA.glob("*.csv")):
        shutil.copy2(csv_file, DATA_OUT / csv_file.name)

    profile_imgs = [n for n in plot_names if n.startswith("profile_tok_s_")]
    accept_imgs = [n for n in plot_names if n.startswith("profile_acceptance")]
    rollout_imgs = [n for n in plot_names if n.startswith("rollout_")]
    drain_imgs = [n for n in plot_names if n.startswith("drain_")]

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

<h2>Phase 1 &mdash; Profiling grid: tokens/s across batch size &times; K</h2>
<p class="note">Fixed-length generation (ignore_eos), each K is a separate
engine; K=0 disables speculation. The K=5 collapse at BS=128 is the
cudagraph-capture cliff (128&times;6 = 768 tokens/step &gt; default 512 max
capture size): exactly the regime cost DynamicSD avoids.</p>
{img_cards(profile_imgs)}

<h2>Acceptance length (temperature 1.0)</h2>
{img_cards(accept_imgs)}

<h2>Derived DynamicSD schedules</h2>
{tables_section()}

<h2>Draft sampling: greedy vs probabilistic</h2>
<p class="note">vLLM 0.24's <code>draft_sample_method="probabilistic"</code>
(stochastic drafter sampling with cached draft logits for exact rejection
sampling) showed no acceptance-length gain over greedy drafting on
Qwen3-30B-A3B/openmath (AL 2.99 vs 3.01 at K=3) and 3-10% lower tok/s from the
logits-caching overhead, so the main matrix uses greedy drafting.</p>

<h2>Phase 3 &mdash; Synchronous rollout: baseline vs fixed-K vs DynamicSD</h2>
{rollout_table()}
{img_cards(rollout_imgs)}

<h2>Rollout drain curves</h2>
<p class="note">Sequences still in flight over time within a rollout step
(steps &gt; 0). The long right tail at low concurrency is where DynamicSD can
raise K beyond the fixed-K compromise.</p>
{img_cards(drain_imgs)}

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
