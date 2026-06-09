#!/usr/bin/env python3 -S
"""Parse per-step Async-GRPO timing breakdown from ray-driver.log into a table.

Output columns: step, phase, total, refit, logprobs, training, exposed_generation
(prefixed with job,variant when multiple logs are passed).

Handles BOTH known log formats:
  A) 256H100 perf-patch "Performance Metrics" block:
       Step N ...
       train_step:            60.31s          (-> training)
       logprob_compute:       12.05s          (-> logprobs)
       generation_compute:   240.40s          (-> exposed_generation)
       total_step:           411.40s          (-> total)
       weight_sync:           18.xx s          (-> refit)
  B) ruit SWE format:
       Step N ... / step N ...
       • Total step time: 445.10s             (-> total)
       • policy_training: 406.30s (91.3%)     (-> training)
       • exposed_generation: 33.42s (7.5%)    (-> exposed_generation)
       • weight_sync: 4.53s (1.0%)            (-> refit)
       (ruit skips reference logprobs when KL=0, so logprobs may be blank)

Phase (heuristic, tune with --factor): step 1 = cold; for later steps,
"long_tail" if total > factor x median(non-cold totals), else "steady".
The steady-mean summary excludes cold + long_tail. If the heuristic mislabels
a run, pass --factor or eyeball the per-step rows.

Usage:
  # one job, markdown table:
  python3 -S parse_step_breakdown.py --markdown \
      --job 12673673 --variant 192g64t_async4 --log ray-driver.log
  # several jobs at once (CSV):
  python3 -S parse_step_breakdown.py \
      --job 12561705 --variant async4_BF16  --log a.log \
      --job 12561710 --variant async4_FP8KV --log b.log
"""

import argparse
import re
import statistics
import sys

NUM = r"([0-9]+(?:\.[0-9]+)?)"
GAP_TOL = 2  # non-timing lines tolerated inside a block (bridges ray interleave)
# field -> regexes (first match wins), covering both log formats
PATTERNS = {
    "total": [rf"Total step time:\s*{NUM}\s*s", rf"total_step:\s*{NUM}\s*s"],
    "refit": [rf"weight_sync:\s*{NUM}\s*s", rf"refit:\s*{NUM}\s*s"],
    "logprobs": [rf"logprob_compute:\s*{NUM}\s*s", rf"logprobs?:\s*{NUM}\s*s"],
    "training": [rf"policy_training:\s*{NUM}\s*s", rf"train_step:\s*{NUM}\s*s"],
    "exposed_generation": [
        rf"exposed_generation:\s*{NUM}\s*s",
        rf"generation_compute:\s*{NUM}\s*s",
    ],
}
FIELDS = [
    "step",
    "phase",
    "total",
    "refit",
    "logprobs",
    "training",
    "exposed_generation",
]


def _first_match(line, regs):
    for rg in regs:
        mm = re.search(rg, line)
        if mm:
            return mm.group(1)
    return None


def _line_fields(line):
    """field -> value for every timing pattern matching this line ({} if none)."""
    out = {}
    for field, regs in PATTERNS.items():
        v = _first_match(line, regs)
        if v is not None:
            out[field] = v
    return out


def parse_log(path):
    """Return list of per-step dicts (values are strings; missing -> '').

    A step's timing fields are logged as a contiguous run of lines (the ruit
    "- ..." block or the 256H100 "Performance Metrics" block); runs are separated
    by non-timing lines ("Step N", headers). We segment the log into maximal runs
    of timing-matching lines and emit one row per run containing a `total`. This
    is order-agnostic (works whether total is logged first or last) and immune to
    the "step N" log spam. Steps numbered 1-based by run order.
    """
    try:
        text = open(path, errors="ignore").read()
    except OSError as e:
        print(f"# ERROR reading {path}: {e}", file=sys.stderr)
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # strip ANSI color codes
    rows: list[dict] = []
    block: dict = {}
    gap = 0  # consecutive non-timing lines tolerated inside a block (ray interleave)

    def flush():
        nonlocal block
        if "total" in block:
            block["step"] = str(len(rows) + 1)
            rows.append(block)
        block = {}

    for line in text.splitlines():
        lf = _line_fields(line)
        if lf:
            # a new total while one is already open => previous step's block ended
            if "total" in lf and "total" in block:
                flush()
            for f, v in lf.items():
                block.setdefault(f, v)  # first occurrence in the run wins
            gap = 0
        elif block:
            gap += 1
            if gap > GAP_TOL:  # real steps are far apart; this only bridges noise
                flush()
                gap = 0
    flush()
    return rows


def classify(rows, factor):
    totals = [float(r["total"]) for r in rows if r.get("total")]
    noncold = totals[1:] if len(totals) > 1 else totals
    med = statistics.median(noncold) if noncold else 0.0
    for i, r in enumerate(rows):
        if i == 0:
            r["phase"] = "cold"
        elif med and float(r["total"]) > factor * med:
            r["phase"] = "long_tail"
        else:
            r["phase"] = "steady"
    return rows


def steady_mean(rows):
    steady = [r for r in rows if r["phase"] == "steady"]

    def mean(f):
        vals = [float(r[f]) for r in steady if r.get(f)]
        return round(statistics.mean(vals), 2) if vals else ""

    return len(steady), {f: mean(f) for f in FIELDS if f not in ("step", "phase")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--job", action="append", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument(
        "--factor", type=float, default=1.30, help="long_tail = total > factor*median"
    )
    ap.add_argument(
        "--markdown", action="store_true", help="emit GitHub markdown tables"
    )
    a = ap.parse_args()
    if not (len(a.log) == len(a.job) == len(a.variant)):
        sys.exit("--log/--job/--variant counts must match")

    md = a.markdown
    multi = len(a.log) > 1
    cols = (["job", "variant"] if multi else []) + FIELDS
    sep = " | " if md else ","
    if md:
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join("---" for _ in cols) + " |")
    else:
        print(sep.join(cols))

    summary = []
    for log, job, variant in zip(a.log, a.job, a.variant):
        rows = classify(parse_log(log), a.factor)
        for r in rows:
            vals = ([job, variant] if multi else []) + [
                str(r.get(c, "")) for c in FIELDS
            ]
            print(("| " + " | ".join(vals) + " |") if md else sep.join(vals))
        n, m = steady_mean(rows)
        summary.append((job, variant, n, m))

    scols = [
        "job",
        "variant",
        "n_steady",
        "total",
        "refit",
        "logprobs",
        "training",
        "exposed_generation",
    ]
    print(
        "\n"
        + (
            "**steady-mean** (cold + long_tail excluded)"
            if md
            else "# steady-mean (cold + long_tail excluded)"
        )
    )
    if md:
        print("| " + " | ".join(scols) + " |")
        print("| " + " | ".join("---" for _ in scols) + " |")
    else:
        print("# " + ",".join(scols))
    for job, variant, n, m in summary:
        vals = [job, variant, str(n)] + [
            str(m[f])
            for f in ["total", "refit", "logprobs", "training", "exposed_generation"]
        ]
        print(("| " + " | ".join(vals) + " |") if md else "# " + ",".join(vals))


if __name__ == "__main__":
    main()
