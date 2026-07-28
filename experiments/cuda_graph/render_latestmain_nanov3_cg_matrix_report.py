#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render the latest-main NanoV3 CUDA Graph matrix's safe static status page."""

import argparse
import csv
import html
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


SMOKE_CSV = "latestmain_nanov3_cg_matrix_smoke.csv"
RUNS_CSV = "latestmain_nanov3_cg_matrix_runs.csv"
PERFORMANCE_CSV = "latestmain_nanov3_cg_matrix_performance.csv"


def parse_args() -> argparse.Namespace:
    """Return command-line paths for the result CSVs and static report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/cuda_graph/results"),
        help="Directory containing the small result-index CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/cuda_graph/results/latestmain_nanov3_cg_matrix_report.html"
        ),
        help="Path of the self-contained static HTML report.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one optional CSV result index without failing before the first run."""
    if not path.is_file():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def escape(value: object) -> str:
    """Escape every CSV field before it becomes HTML content or an attribute."""
    return html.escape(str(value), quote=True)


def safe_href(value: str) -> str | None:
    """Permit ordinary HTTPS and repository-relative evidence links only."""
    parsed = urlparse(value)
    if parsed.scheme in {"https", "http"}:
        return value
    if not parsed.scheme and not value.startswith("//"):
        return value
    return None


def cell(value: str) -> str:
    """Render a CSV field as escaped text."""
    return f"<td>{escape(value)}</td>"


def link_cell(label: str, value: str) -> str:
    """Render a safe link, or escaped plain text for an invalid link target."""
    href = safe_href(value)
    if not href:
        return cell(value)
    return f'<td><a href="{escape(href)}">{escape(label)}</a></td>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a compact table or an explicit no-data message."""
    if not rows:
        return '<p class="empty">No result CSV is available yet.</p>'
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = "".join(f"<tr>{''.join(row)}</tr>" for row in rows)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>"
    )


def provenance_rows(smoke_rows: list[dict[str, str]]) -> list[list[str]]:
    """Keep source, container, and committed script provenance visible."""
    fields = ("cluster", "source_sha", "container", "script")
    seen: set[tuple[str, ...]] = set()
    rows: list[list[str]] = []
    for smoke_row in smoke_rows:
        values = tuple(smoke_row.get(field, "") for field in fields)
        if values in seen:
            continue
        seen.add(values)
        cluster, source_sha, container, script = values
        rows.append(
            [
                cell(cluster),
                cell(source_sha),
                cell(container),
                link_cell(script, script) if script else cell(""),
            ]
        )
    return rows


def job_rows(smoke_rows: list[dict[str, str]]) -> list[list[str]]:
    """Render current Slurm status and evidence links from the smoke index."""
    rows: list[list[str]] = []
    for smoke_row in smoke_rows:
        script = smoke_row.get("script", "")
        log_link = smoke_row.get("log_link", "")
        wandb_link = smoke_row.get("wandb_link", "")
        rows.append(
            [
                cell(smoke_row.get("cluster", "")),
                cell(smoke_row.get("scope", "")),
                cell(smoke_row.get("job_id", "")),
                cell(smoke_row.get("state", "")),
                cell(smoke_row.get("reason", "")),
                link_cell(script, script) if script else cell(""),
                link_cell("log", log_link) if log_link else cell(""),
                link_cell("W&B", wandb_link) if wandb_link else cell(""),
            ]
        )
    return rows


def performance_rows(performance_rows_: list[dict[str, str]]) -> list[list[str]]:
    """Render only the performance measures needed for matrix comparisons."""
    columns = (
        "scope",
        "timing/train/total_step_time",
        "performance/tokens_per_sec_per_gpu",
        "timing/train/generation",
        "timing/train/policy_training",
        "timing/train/policy_and_reference_logprobs",
    )
    return [
        [cell(performance_row.get(column, "")) for column in columns]
        for performance_row in performance_rows_
    ]


def convergence_rows(run_rows: list[dict[str, str]]) -> list[list[str]]:
    """Render short-horizon accuracy/convergence deltas without raw run logs."""
    columns = (
        "scope",
        "accuracy_metric",
        "baseline_value",
        "candidate_value",
        "delta",
    )
    return [
        [cell(run_row.get(column, "")) for column in columns] for run_row in run_rows
    ]


def render_html(
    *,
    smoke_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
    performance_rows_: list[dict[str, str]],
) -> str:
    """Build the self-contained report from small, sanitized result indexes."""
    updated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    provenance = table(
        ["Cluster", "Source SHA", "Container", "Submission script"],
        provenance_rows(smoke_rows),
    )
    jobs = table(
        ["Cluster", "Scope", "Job ID", "State", "Reason", "Script", "Log", "W&B"],
        job_rows(smoke_rows),
    )
    performance = table(
        [
            "Scope",
            "E2E step time (s)",
            "E2E tokens/s/GPU",
            "Generation time (s)",
            "Policy training time (s)",
            "Logprob time (s)",
        ],
        performance_rows(performance_rows_),
    )
    convergence = table(
        ["Scope", "Accuracy metric", "Baseline", "Candidate", "Delta"],
        convergence_rows(run_rows),
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latest-main NanoV3 CUDA Graph matrix</title>
<style>
body {{ background: #10141c; color: #e7edf7; font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; }}
h1, h2 {{ color: #9fd7ff; }}
section {{ background: #18212d; border: 1px solid #2b3a4c; border-radius: 8px; margin: 1rem 0; padding: 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #2b3a4c; padding: .55rem; text-align: left; vertical-align: top; word-break: break-word; }}
th {{ color: #9fd7ff; }} a {{ color: #7fe1b9; }} .table-wrap {{ overflow-x: auto; }} .empty {{ color: #aeb8c7; }}
</style>
</head>
<body>
<h1>Latest-main NanoV3 CUDA Graph matrix</h1>
<p>Updated (UTC): {escape(updated_at)}</p>
<section id="provenance"><h2>Provenance</h2>{provenance}</section>
<section id="job-status"><h2>Job status</h2>{jobs}</section>
<section id="performance"><h2>Performance</h2>{performance}</section>
<section id="convergence"><h2>Convergence</h2>{convergence}</section>
</body>
</html>
"""


def main() -> None:
    """Load the available indexes and refresh the report path."""
    args = parse_args()
    smoke_rows = read_csv(args.results_dir / SMOKE_CSV)
    run_rows = read_csv(args.results_dir / RUNS_CSV)
    performance_rows_ = read_csv(args.results_dir / PERFORMANCE_CSV)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(
            smoke_rows=smoke_rows,
            run_rows=run_rows,
            performance_rows_=performance_rows_,
        )
    )


if __name__ == "__main__":
    main()
