"""Write the run tables as a static HTML page that opens without a server.

    python report_html.py [report/data/runs.json] [report/index.html]

One table per comparison group. Stage times are means over the report window
(steps 3-20, SWE steps 1-20). Speedups are control / arm on the steps that both
runs logged, so above 1x means the arm is faster. The V2 (single-controller) SWE
runs have no W&B history, so their tables come from swe_driver_steps.csv
(swe_driver_logs.py) and use steps 2 onward.
"""

import csv
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MIN_PAIRED = 8
DRAFTERS = {"dflash": "DFlash", "dspark": "DSpark"}
TIMES = (
    ("gen_s", "Gen s"),
    ("train_s", "Train s"),
    ("logprob_s", "LogProb s"),
    ("refit_s", "Refit s"),
    ("step_s", "E2E step s"),
)
SPEEDUPS = (("pair_step_delta", "E2E speedup"), ("pair_gen_delta", "Gen speedup"))
DRIVER_TIMES = (
    ("exposed_generation_s", "Gen s"),
    ("policy_training_s", "Train s"),
    ("logprob_s", "LogProb s"),
    ("weight_sync_s", "Refit s"),
    ("total_s", "E2E step s"),
)
CSS = """
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e; --grid: #e4e3df;
  --head: #f1f0ec; --good: #1d7a3e; --bad: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #161615; --ink: #ededeb; --muted: #a3a29d; --grid: #34332f;
    --head: #22211f; --good: #5cc283; --bad: #f2867c;
  }
}
body {
  background: var(--surface); color: var(--ink); margin: 0 auto; max-width: 1280px;
  padding: 16px; font: 14px/1.45 system-ui, -apple-system, sans-serif;
}
h1 { font-size: 22px; margin: 8px 0; }
h2 { font-size: 16px; margin: 28px 0 8px; }
p { color: var(--muted); max-width: 80ch; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { background: var(--head); text-align: right; font-weight: 600; }
td { text-align: right; }
th:first-child, td:first-child, td.text { text-align: left; }
tr.superseded td { color: var(--muted); }
.good { color: var(--good); }
.bad { color: var(--bad); }
a { color: inherit; }
"""


def num(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def speedup(run: dict, key: str) -> str:
    delta = run.get(key)
    if delta is None or (run.get("pair_steps") or 0) < MIN_PAIRED:
        return "<td>-</td>"
    x = 1 / (1 + delta)
    tone = "good" if x >= 1.02 else ("bad" if x <= 0.98 else "")
    return f'<td class="{tone}">{x:.2f}x</td>'


def drafter(run: dict) -> str:
    if run["drafter"] == "none":
        return "-"
    return f"{DRAFTERS.get(run['drafter'], run['drafter'])} {run['drafter_mode']}"


def sort_key(run: dict) -> tuple:
    return (
        run["drafter"] != "none",
        run["drafter"],
        run["drafter_mode"],
        run["k"] or 0,
        run["arm"],
    )


def group_title(runs: list[dict]) -> str:
    # Some groups hold sync and async runs, so list every value
    modes = ", ".join(sorted({r["mode"] for r in runs}))
    seqlens = ", ".join(sorted({r["seqlen"] for r in runs}))
    return f"{runs[0]['group']}: {runs[0]['model']}, {modes}, {seqlens}"


def row(run: dict) -> str:
    arm = html.escape(run["arm"])
    link = (
        f'<a href="{html.escape(run["wandb_url"])}">{arm}</a>'
        if run.get("wandb_url")
        else arm
    )
    ref = run.get("ref") or ""
    ref_arm = ref.split("-", 2)[-1] if ref else "-"
    state = str(run.get("state") or "")
    cells = [
        f"<td>{link}</td>",
        f"<td class='text'>{drafter(run)}</td>",
        f"<td>{run['k'] or '-'}</td>",
        f"<td>{run.get('window_steps') or 0}</td>",
        *(
            f"<td>{num(run.get(key), 3 if key == 'logprob_s' else 1)}</td>"
            for key, _ in TIMES
        ),
        f"<td>{num(run.get('tok_s_gpu'))}</td>",
        f"<td>{num(run.get('gen_tok_s_gpu'))}</td>",
        f"<td>{num(run.get('accept_len'), 2)}</td>",
        *(speedup(run, key) for key, _ in SPEEDUPS),
        f"<td class='text'>{html.escape(ref_arm)}</td>",
        f"<td class='text'>{html.escape(state)}</td>",
    ]
    cls = ' class="superseded"' if state.startswith("superseded") else ""
    return f"<tr{cls}>{''.join(cells)}</tr>"


def table(runs: list[dict]) -> str:
    heads = [
        "Arm",
        "Drafter",
        "K",
        "Steps",
        *(label for _, label in TIMES),
        "tok/s/GPU",
        "Gen tok/s/GPU",
        "Accept len",
        *(label for _, label in SPEEDUPS),
        "Control",
        "State",
    ]
    head = "".join(f"<th>{h}</th>" for h in heads)
    body = "\n".join(row(r) for r in sorted(runs, key=sort_key))
    return (
        f"<h2>{html.escape(group_title(runs))}</h2>\n"
        f'<div class="wrap"><table><thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody></table></div>'
    )


def driver_runs(path: Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    # Run names are <cohort>-sc-<arm>, for example swe-tosd1030-sc-...-baseline-l0-staleness2
    cohorts: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cohort, arm = r["run"].split("-sc-", 1)
            step = int(r["step"])
            if step < 2:
                continue
            vals = {k: float(v) for k, v in r.items() if k not in ("run", "step") and v}
            cohorts.setdefault(cohort, {}).setdefault(arm, {})[step] = vals
    return cohorts


def driver_row(arm: str, steps: dict[int, dict], ref: dict[int, dict] | None) -> str:
    def avg(key: str, on: list[int]) -> float | None:
        xs = [steps[s][key] for s in on if key in steps[s]]
        return sum(xs) / len(xs) if xs else None

    window = sorted(steps)
    lags = [steps[s]["trainer_lag"] for s in window]
    cells = [
        f"<td>{html.escape(arm)}</td>",
        f"<td>{min(window)}-{max(window)}</td>",
        *(f"<td>{num(avg(key, window))}</td>" for key, _ in DRIVER_TIMES),
        f"<td>{sum(lags) / len(lags):.2f}</td>",
        f"<td>{max(lags):.0f}</td>",
    ]
    paired = sorted(set(steps) & set(ref)) if ref is not None else []
    if len(paired) >= MIN_PAIRED:
        ref_mean = sum(ref[s]["total_s"] for s in paired) / len(paired)
        x = ref_mean / avg("total_s", paired)
        tone = "good" if x >= 1.02 else ("bad" if x <= 0.98 else "")
        cells.append(f'<td class="{tone}">{x:.2f}x</td>')
    else:
        cells.append("<td>-</td>")
    return f"<tr>{''.join(cells)}</tr>"


def driver_table(cohort: str, arms: dict[str, dict[int, dict]]) -> str:
    heads = [
        "Arm",
        "Steps",
        *(label for _, label in DRIVER_TIMES),
        "Lag mean",
        "Lag max",
        "E2E speedup",
    ]
    head = "".join(f"<th>{h}</th>" for h in heads)
    body = "\n".join(
        driver_row(arm, arms[arm], None if arm == ref else arms.get(ref))
        for arm in sorted(arms, key=lambda a: ("dspark" in a, a))
        for ref in [arm.replace("dspark-k8", "baseline")]
    )
    return (
        f"<h2>{html.escape(cohort)}: Qwen3-30B-A3B SWE, V2 driver logs</h2>\n"
        f'<div class="wrap"><table><thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody></table></div>'
    )


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report/data/runs.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "report/index.html"
    groups: dict[str, list[dict]] = {}
    for run in json.loads(src.read_text()):
        groups.setdefault(run["group"], []).append(run)
    tables = "\n".join(table(groups[g]) for g in sorted(groups))
    driver = src.parent / "swe_driver_steps.csv"
    if driver.exists():
        cohorts = driver_runs(driver)
        tables += "\n" + "\n".join(driver_table(c, cohorts[c]) for c in sorted(cohorts))
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpecDec RL Runs</title>
<style>{CSS}</style>
</head>
<body>
<h1>SpecDec RL runs: baseline vs DFlash vs DSpark</h1>
<p>Stage times are means over steps 3-20 (SWE steps 1-20). Speedup is control step time over
arm step time on the steps that both runs logged, shown when at least {MIN_PAIRED} steps pair.
Above 1x means the arm is faster. SWE runs skip the logprob pass, so LogProb is 0 or about 1 ms.
The V2 SWE tables at the end come from the driver logs and use steps 2 onward. Their speedup
pairs each DSpark K8 run with the baseline run of the same cohort and fixes.
The report text is in <a href="README.md">README.md</a>.</p>
{tables}
</body>
</html>"""
    out.write_text(page)
    print(f"wrote {out} ({len(groups)} groups)")


if __name__ == "__main__":
    main()
