#!/usr/bin/env python3
"""Build the GitLab Pages landing page for SpecDec RL benchmark results."""

from __future__ import annotations

import html
import math
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PUBLIC = ROOT / "public"
REPORTS = PUBLIC / "reports"
DATA = PUBLIC / "data"
ARCHIVE = PUBLIC / "archive"

MODELS = ["Qwen3-30B-A3B", "Qwen3-32B", "Qwen3-235B-A22B"]


def esc(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return html.escape(text)


def as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    num = as_float(value)
    if math.isnan(num):
        return "n/a"
    return f"{num:.{digits}f}{suffix}"


def fmt_int(value: object) -> str:
    num = as_float(value)
    if math.isnan(num):
        return "n/a"
    return str(int(round(num)))


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    if dst.suffix in {".csv", ".html", ".json", ".txt"}:
        raw = dst.read_bytes()
        dst.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return dst


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def vllm_best_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rel in [
        "vllm_standalone_all_batches_combined_20260619.csv",
        "vllm_standalone_added_results_latest.csv",
    ]:
        path = DOCS / rel
        df = load_csv(path)
        if df.empty:
            continue
        if "valid_result" in df.columns:
            df = df[df["valid_result"].astype(str).str.lower().eq("true")]
        df["source_file"] = f"docs/{rel}"
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows = rows[rows["model"].isin(MODELS)]
    rows = rows[rows["method"].astype(str) != "baseline"]
    for column in [
        "temperature",
        "top_p",
        "batch_size",
        "isl",
        "osl",
        "tok_s_gpu",
        "speedup",
        "acceptance_pct",
        "mean_accept_len",
    ]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["speedup"])
    if rows.empty:
        return rows
    idx = rows.groupby(["model", "domain", "temperature"], dropna=False)["speedup"].idxmax()
    keep = [
        "domain",
        "model",
        "temperature",
        "method",
        "batch_size",
        "isl",
        "osl",
        "tok_s_gpu",
        "speedup",
        "acceptance_pct",
        "mean_accept_len",
        "source_file",
    ]
    return rows.loc[idx, keep].sort_values(["model", "domain", "temperature"])


def nemorl_best_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    perf = load_csv(DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv")
    if not perf.empty:
        perf = perf[perf["model"].isin(["Qwen3-30B-A3B", "Qwen3-32B"])]
        perf = perf[~perf["method"].astype(str).str.contains("baseline", case=False, na=False)]
        for column in [
            "generation_throughput_speedup",
            "generation_time_speedup",
            "e2e_throughput_speedup",
            "e2e_step_time_speedup",
            "acceptance_pct",
            "mean_accept_len",
            "generation_throughput_tok_s_gpu",
            "generation_time_s",
            "e2e_step_time_s",
        ]:
            perf[column] = pd.to_numeric(perf[column], errors="coerce")
        for _, row in perf.dropna(subset=["generation_throughput_speedup"]).iterrows():
            rows.append(
                {
                    "source_file": "docs/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
                    "job_id": row["job_id"],
                    "model": row["model"],
                    "mode": row["mode"],
                    "method": row["method"],
                    "completed": row["completed_last_step"],
                    "max_osl": row["max_osl"],
                    "gen_tps": row["generation_throughput_tok_s_gpu"],
                    "gen_tps_speedup": row["generation_throughput_speedup"],
                    "gen_time_speedup": row["generation_time_speedup"],
                    "e2e_tps_speedup": row["e2e_throughput_speedup"],
                    "e2e_step_speedup": row["e2e_step_time_speedup"],
                    "acceptance_pct": row["acceptance_pct"],
                    "mean_accept_len": row["mean_accept_len"],
                }
            )

    q235 = load_csv(DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv")
    if not q235.empty:
        q235 = q235[q235["model_name"].eq("Qwen3-235B-A22B")]
        q235 = q235[~q235["method_k"].astype(str).str.contains("baseline", case=False, na=False)]
        for column in [
            "gen_tps_speedup",
            "generation_time_speedup",
            "e2e_tps_speedup",
            "e2e_step_time_speedup",
            "generation_worker_tokens_per_sec_per_gpu_mean",
            "vllm_token_acceptance_pct",
            "vllm_acceptance_length_mean_weighted_mean",
        ]:
            q235[column] = pd.to_numeric(q235[column], errors="coerce")
        for _, row in q235.dropna(subset=["gen_tps_speedup"]).iterrows():
            rows.append(
                {
                    "source_file": "docs/lyris_qwen235b_pr2879_live_enriched_20260621.csv",
                    "job_id": row["job_id"],
                    "model": row["model_name"],
                    "mode": row["mode"],
                    "method": row["method_k"],
                    "completed": row["completed_last_step"],
                    "max_osl": row["max_new_tokens"],
                    "gen_tps": row["generation_worker_tokens_per_sec_per_gpu_mean"],
                    "gen_tps_speedup": row["gen_tps_speedup"],
                    "gen_time_speedup": row["generation_time_speedup"],
                    "e2e_tps_speedup": row["e2e_tps_speedup"],
                    "e2e_step_speedup": row["e2e_step_time_speedup"],
                    "acceptance_pct": row.get("vllm_token_acceptance_pct", math.nan),
                    "mean_accept_len": row.get("vllm_acceptance_length_mean_weighted_mean", math.nan),
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    idx = df.groupby("model")["gen_tps_speedup"].idxmax()
    return df.loc[idx].sort_values("model")


def rows_to_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<div class=\"table-scroll\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def vllm_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class=\"muted\">No vLLM rows are available in the local CSV artifacts.</p>"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                esc(row["domain"]),
                esc(row["model"]),
                fmt(row["temperature"], 1),
                esc(row["method"]),
                fmt_int(row["batch_size"]),
                f"{fmt_int(row['isl'])}/{fmt_int(row['osl'])}",
                fmt(row["tok_s_gpu"], 2),
                fmt(row["speedup"], 2, "x"),
                fmt(row["acceptance_pct"], 1, "%"),
                fmt(row["mean_accept_len"], 2),
                f"<code>{esc(row['source_file'])}</code>",
            ]
        )
    return rows_to_table(
        [
            "Domain",
            "Model",
            "Temp",
            "Method",
            "BS",
            "ISL/OSL",
            "tok/s/GPU",
            "Speedup",
            "Acceptance",
            "Mean accept len",
            "Source",
        ],
        rows,
    )


def nemorl_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class=\"muted\">No NeMo-RL speedup rows are available in the local CSV artifacts.</p>"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                f"<code>{esc(row['job_id'])}</code>",
                esc(row["model"]),
                esc(row["mode"]),
                esc(row["method"]),
                esc(row["completed"]),
                fmt_int(row["max_osl"]),
                fmt(row["gen_tps"], 2),
                fmt(row["gen_tps_speedup"], 2, "x"),
                fmt(row["gen_time_speedup"], 2, "x"),
                fmt(row["e2e_tps_speedup"], 2, "x"),
                fmt(row["e2e_step_speedup"], 2, "x"),
                fmt(row["acceptance_pct"], 1, "%"),
                fmt(row["mean_accept_len"], 2),
                f"<code>{esc(row['source_file'])}</code>",
            ]
        )
    return rows_to_table(
        [
            "Job",
            "Model",
            "Mode",
            "Method",
            "Completed",
            "Max OSL",
            "Gen tok/s/GPU",
            "Gen TPS speedup",
            "Gen time speedup",
            "E2E TPS speedup",
            "E2E step speedup",
            "Acceptance",
            "Mean accept len",
            "Source",
        ],
        rows,
    )


def read_job_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build() -> None:
    PUBLIC.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    report_files = [
        DOCS / "vllm_standalone_results_latest.html",
        DOCS / "lyris_nemorl_perfcfg_specdec_live_status_latest.html",
        DOCS / "specdec_clean_benchmark_results_20260617.html",
    ]
    data_files = [
        DOCS / "vllm_standalone_added_results_latest.csv",
        DOCS / "vllm_standalone_all_batches_combined_20260619.csv",
        DOCS / "lyris_qwen235b_pr2879_live_enriched_20260621.csv",
        DOCS / "lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv",
        DOCS / "nemorl_clean_results_20260617.csv",
        DOCS / "lyris_angelslim_checkpoint_prewarm_summary_20260622.json",
        ROOT / "latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt",
    ]
    archive_files = [
        ROOT / "experiments/eagle3_qwen3_235b/specdec_math_progress_report.html",
    ]
    for src in report_files:
        copy_if_exists(src, REPORTS)
    for src in data_files:
        copy_if_exists(src, DATA)
    for src in archive_files:
        copy_if_exists(src, ARCHIVE)

    vllm = vllm_best_rows()
    nemorl = nemorl_best_rows()
    job = read_job_file(ROOT / "latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt")
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    job_id = job.get("job_id", "pending")
    job_status = job.get("status", "submitted")
    sacct_state = job.get("sacct_state", "")
    logs_dir = job.get("logs_dir", "")
    summary_json = job.get("summary_json", "")
    model_ids = job.get("model_ids", "")
    status_class = "ok" if sacct_state == "COMPLETED" else "warn"
    status_label = sacct_state or job_status

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>SpecDec RL Benchmark Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #5d6675;
      --line: #d7dce5;
      --blue: #1d5fbf;
      --green: #0c7a4b;
      --amber: #986100;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 26px 18px 44px; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 10px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: var(--muted); margin: 7px 0 0; }}
    code {{ background: #eef1f6; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; font-size: 12px; }}
    a {{ color: var(--blue); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: .04em; }}
    .metric {{ font-size: 24px; font-weight: 780; margin-top: 6px; }}
    .muted {{ color: var(--muted); }}
    .pill {{ display: inline-flex; align-items: center; min-height: 26px; padding: 2px 9px; border-radius: 999px; background: #eef1f6; border: 1px solid var(--line); font-weight: 700; font-size: 12px; }}
    .pill.ok {{ color: var(--green); }}
    .pill.warn {{ color: var(--amber); }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .links a {{ display: inline-flex; align-items: center; min-height: 34px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 7px 10px; text-decoration: none; font-weight: 700; }}
    .table-scroll {{ width: 100%; overflow-x: auto; margin-top: 10px; }}
    table {{ min-width: 1060px; width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef1f6; }}
    tr:last-child td {{ border-bottom: 0; }}
    .note {{ border-left: 4px solid var(--blue); background: var(--panel); border-radius: 8px; padding: 13px 14px; border-top: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); margin-top: 14px; }}
    @media (max-width: 840px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 18px 12px 32px; }} }}
  </style>
</head>
<body>
<main>
  <h1>SpecDec RL Benchmark Dashboard</h1>
  <p>Updated {esc(generated_at)}. This Pages entry point mirrors the latest local vLLM standalone and NeMo-RL benchmark artifacts for Qwen3 speculative decoding.</p>

  <div class=\"grid\">
    <div class=\"card\"><div class=\"label\">vLLM scope</div><div class=\"metric\">Math + SWE</div><p>Batch sweeps and temp 0/1 comparisons with ISL/OSL shown in the result tables.</p></div>
    <div class=\"card\"><div class=\"label\">NeMo-RL scope</div><div class=\"metric\">Perf recipe</div><p>Qwen30/32 use recipe OSL4096; Qwen235B PR2879 rows use recipe OSL8192.</p></div>
    <div class=\"card\"><div class=\"label\">AngelSlim staging</div><div class=\"metric\"><code>{esc(job_id)}</code></div><p>HF download job state: <span class=\"pill {status_class}\">{esc(status_label)}</span></p></div>
  </div>

  <section>
    <h2>Report Links</h2>
    <div class=\"links\">
      <a href=\"reports/vllm_standalone_results_latest.html\">vLLM standalone latest</a>
      <a href=\"reports/lyris_nemorl_perfcfg_specdec_live_status_latest.html\">NeMo-RL latest</a>
      <a href=\"reports/specdec_clean_benchmark_results_20260617.html\">Combined clean report</a>
      <a href=\"archive/specdec_math_progress_report.html\">Archive: old Eagle3 report</a>
    </div>
  </section>

  <section>
    <h2>Best vLLM Standalone Rows</h2>
    <p>Best rows are selected by matched baseline speedup for each model, domain, and temperature.</p>
    {vllm_table(vllm)}
  </section>

  <section>
    <h2>Best NeMo-RL Rows</h2>
    <p>Best rows are selected by generation throughput speedup against the matched baseline. E2E throughput and step-time speedup are shown separately.</p>
    {nemorl_table(nemorl)}
  </section>

  <section>
    <h2>DFlare and AngelSlim Status</h2>
    <div class=\"note\">
      <span class=\"pill {status_class}\">{esc(status_label)}</span>
      <p>DFlare public checkpoints found so far are <code>AngelSlim/Qwen3-4b-dflare</code>, <code>AngelSlim/Qwen3-8b-dflare</code>, and <code>AngelSlim/Gpt-oss-20b-dflare</code>. Current vLLM/NeMo-RL result pages do not yet include a direct DFlare row because the local generation path is vLLM SpecDec, while DFlare is exposed through AngelSlim's standalone tooling. The submitted HF staging job also downloads AngelSlim Eagle3 drafters for Qwen3-A3B, Qwen3-32B, Qwen3-8B, Qwen3-14B, and Qwen3-4B.</p>
      <p>Models requested in staging job: <code>{esc(model_ids)}</code></p>
      <p>Logs: <code>{esc(logs_dir)}</code></p>
      <p>Summary JSON: <code>{esc(summary_json)}</code></p>
    </div>
  </section>

  <section>
    <h2>Data Artifacts</h2>
    <div class=\"links\">
      <a href=\"data/vllm_standalone_added_results_latest.csv\">vLLM added CSV</a>
      <a href=\"data/vllm_standalone_all_batches_combined_20260619.csv\">vLLM all-batch CSV</a>
      <a href=\"data/lyris_qwen235b_pr2879_live_enriched_20260621.csv\">Qwen235B NeMo-RL CSV</a>
      <a href=\"data/lyris_nemorl_perfcfg_step20_live_speedups_20260618.csv\">Qwen30/32 NeMo-RL CSV</a>
      <a href=\"data/lyris_angelslim_checkpoint_prewarm_summary_20260622.json\">AngelSlim prewarm summary</a>
      <a href=\"data/latest_lyris_angelslim_checkpoint_prewarm_20260622_jobs.txt\">AngelSlim job record</a>
    </div>
  </section>
</main>
</body>
</html>
"""
    (PUBLIC / "index.html").write_text(html_text)
    print(PUBLIC / "index.html")


if __name__ == "__main__":
    build()
