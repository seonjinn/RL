"""Parse SC (V2) SWE GRPO driver logs into per-step tables and compare arms.

    python swe_driver_logs.py <report/data> <run>/<jobid>-logs/ray-driver.log ...

The SC driver prints a timing block and a `train step N/M trainer_v=V lag=L`
line for each step, plus one step_metrics dict. The script writes two tables
keyed by run directory name: swe_driver_steps.csv (phase times and trainer lag)
and swe_driver_metrics.csv (reward and off-policy metrics). It then prints the
GBS 512 pair, the staleness 2 pairs, the staleness 1 vs 2 quality check and
the env run-ahead pairs. Step 1 holds the cold start, so step times use steps
2 onward.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

TIMING = {
    "Total step time": "total_s",
    "exposed_generation": "exposed_generation_s",
    "policy_training": "policy_training_s",
    "policy_and_reference_logprobs": "logprob_s",
    "weight_sync": "weight_sync_s",
}
METRICS = {
    "total_reward/mean": "reward",
    "swe_agents_train/resolved/mean": "resolved",
    "zero_advantage_group_pct": "zero_advantage_group_pct",
    "gen_kl_error": "gen_kl_error",
    "token_mult_prob_error": "token_mult_prob_error",
    "sampling_importance_ratio": "sampling_importance_ratio",
    "is_oob_ratio": "is_oob_ratio",
    "probs_ratio_clamped": "probs_ratio_clamped",
    "grad_norm": "grad_norm",
    "approx_entropy": "approx_entropy",
    "swe_agents_train/prewarm_adopted/mean": "prewarm_adopted",
    "swe_agents_train/prewarm_mismatch/mean": "prewarm_mismatch",
    "swe_agents_train/prewarm_expired/mean": "prewarm_expired",
    "swe_agents_train/prewarm_hold_time/mean": "prewarm_hold_s",
}
NUM = r"(?:np\.float\d+\()?(-?[\d.]+(?:e[+-]?\d+)?|nan|inf)"
# Ray prefixes actor output with "(ActorName pid=N)" in ANSI color.
RAY_PREFIX = re.compile(r"^(?:\x1b\[[\d;]*m)?\([^)]*\)(?:\x1b\[[\d;]*m)?\s*")
STEP_PAIRS = [
    ("GBS 512, L5", "baseline-l5", "dspark-k8-l5", 512),
    (
        "GBS 32, staleness 2, R1",
        "baseline-l0-staleness2",
        "dspark-k8-l0-staleness2",
        32,
    ),
    (
        "GBS 32, staleness 2, R2",
        "baseline-l0-staleness2-r2",
        "dspark-k8-l0-staleness2-r2",
        32,
    ),
]
# Env run-ahead A/B at GBS 32, staleness 1: (label, off arm, prewarm arm).
RUNAHEAD_PAIRS = [
    ("R1", "baseline-l0-runahead-off", "baseline-l0-runahead-p2"),
    ("R2", "baseline-l0-runahead-off-r2", "baseline-l0-runahead-p2-r2"),
]
QUALITY = {
    1: [
        "baseline-l0-control",
        "baseline-l0-control-r2",
        "dspark-k8-l0-control",
        "dspark-k8-l0-control-r2",
    ],
    2: [
        "baseline-l0-staleness2",
        "baseline-l0-staleness2-r2",
        "dspark-k8-l0-staleness2",
        "dspark-k8-l0-staleness2-r2",
    ],
}


def parse(path: Path) -> tuple[list[dict], list[dict]]:
    steps, metrics, block = [], [], {}
    for line in path.read_text(errors="replace").splitlines():
        # Newer SC drivers print the per-step dict as `step_metrics={...}`.
        line = RAY_PREFIX.sub("", line).strip().removeprefix("step_metrics=")
        if m := re.match(r"• ([\w ]+?): ([\d.]+)s", line):
            if m.group(1) in TIMING:
                block[TIMING[m.group(1)]] = float(m.group(2))
        elif m := re.match(r"train step (\d+)/\d+\s+trainer_v=\d+\s+lag=(\d+)", line):
            steps.append(
                {"step": int(m.group(1)), "trainer_lag": int(m.group(2)), **block}
            )
            block = {}
        elif line.startswith("{") and "'total_reward/mean'" in line:
            vals = dict(re.findall(r"'([^']+)': " + NUM, line))
            metrics.append(
                {
                    "step": len(metrics) + 1,
                    **{v: float(vals[k]) for k, v in METRICS.items() if k in vals},
                }
            )
    return steps, metrics


def by_suffix(table: dict[str, list[dict]], suffix: str) -> list[dict]:
    hits = [k for k in table if k.endswith("-" + suffix)]
    if len(hits) != 1:
        raise SystemExit(f"{suffix}: expected one run, got {hits}")
    return table[hits[0]]


def boot_ratio(num: np.ndarray, den: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    r = [
        rng.choice(num, len(num)).mean() / rng.choice(den, len(den)).mean()
        for _ in range(n)
    ]
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def write(path: Path, table: dict[str, list[dict]], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run", "step", *cols])
        for run, rows in table.items():
            for r in rows:
                w.writerow([run, r["step"], *(r.get(c, "") for c in cols)])


def main() -> None:
    out = Path(sys.argv[1])
    steps, metrics = {}, {}
    for p in map(Path, sys.argv[2:]):
        run = p.parent.parent.name
        steps[run], metrics[run] = parse(p)
    write(
        out / "swe_driver_steps.csv",
        {k: v for k, v in steps.items() if v},
        ["trainer_lag", *TIMING.values()],
    )
    write(
        out / "swe_driver_metrics.csv",
        {k: v for k, v in metrics.items() if v},
        list(METRICS.values()),
    )

    pooled = {"b": [], "d": []}
    for label, base, spec, gbs in STEP_PAIRS:
        b, d = by_suffix(steps, base), by_suffix(steps, spec)
        n = min(len(b), len(d))
        bt, dt = (np.array([s["total_s"] for s in x[1:n]]) for x in (b, d))
        be, de = (np.array([s["exposed_generation_s"] for s in x[1:n]]) for x in (b, d))
        lo, hi = boot_ratio(bt, dt)
        lags = [s["trainer_lag"] for s in b[1:n] + d[1:n]]
        print(
            f"{label}, steps 2-{n}: base {bt.mean():.1f} s ({gbs / bt.mean():.2f} samples/s), DSpark {dt.mean():.1f} s "
            f"({gbs / dt.mean():.2f}), step {bt.mean() / dt.mean():.2f}x [{lo:.2f}, {hi:.2f}], exposed {be.mean():.1f} -> "
            f"{de.mean():.1f} s, lag mean {np.mean(lags):.1f} max {max(lags)}"
        )
        for k in TIMING.values():
            print(
                f"    {k:<22} base {np.mean([s[k] for s in b[1:n]]):6.1f}  DSpark {np.mean([s[k] for s in d[1:n]]):6.1f}"
            )
        if gbs == 32:
            pooled["b"] += list(bt)
            pooled["d"] += list(dt)
    bt, dt = np.array(pooled["b"]), np.array(pooled["d"])
    lo, hi = boot_ratio(bt, dt)
    print(
        f"GBS 32, staleness 2, pooled n={len(bt)}: base {bt.mean():.1f} s, DSpark {dt.mean():.1f} s, step {bt.mean() / dt.mean():.2f}x [{lo:.2f}, {hi:.2f}]"
    )

    for arm, prefix in (("base", "baseline"), ("DSpark K8", "dspark-k8")):
        t = {
            k: np.array(
                [
                    s["total_s"]
                    for run in runs
                    if run.startswith(prefix)
                    for s in by_suffix(steps, run)[1:]
                ]
            )
            for k, runs in QUALITY.items()
        }
        lo, hi = boot_ratio(t[1], t[2])
        print(
            f"{arm} staleness 1 -> 2, steps 2-20 pooled: {t[1].mean():.1f} -> {t[2].mean():.1f} s, {t[1].mean() / t[2].mean():.2f}x [{lo:.2f}, {hi:.2f}]"
        )

    reward = {}
    for stale, runs in QUALITY.items():
        rows = [r for run in runs for r in by_suffix(metrics, run)]
        reward[stale] = np.array([r["reward"] for r in rows])
        means = " ".join(
            f"{k}={np.nanmean([r[k] for r in rows if k in r]):.4f}"
            for k in METRICS.values()
        )
        print(f"staleness {stale}, n={len(rows)}: {means}")
        print(
            f"    reward per run: {[round(float(np.mean([r['reward'] for r in by_suffix(metrics, run)])), 4) for run in runs]}"
        )
    lo, hi = boot_diff(reward[1], reward[2])
    print(
        f"reward staleness 2 - staleness 1: {100 * (reward[2].mean() - reward[1].mean()):+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    )

    runahead(steps, metrics)


def runahead(steps: dict[str, list[dict]], metrics: dict[str, list[dict]]) -> None:
    pooled = {k: [] for k in ("off", "p2", "off_e", "p2_e", "off_r", "p2_r")}
    for label, off_run, p2_run in RUNAHEAD_PAIRS:
        if not any(k.endswith("-" + p2_run) for k in steps):
            continue
        off, p2 = by_suffix(steps, off_run), by_suffix(steps, p2_run)
        n = min(len(off), len(p2))
        ot, pt = (np.array([s["total_s"] for s in x[1:n]]) for x in (off, p2))
        oe, pe = (
            np.array([s["exposed_generation_s"] for s in x[1:n]]) for x in (off, p2)
        )
        lo, hi = boot_ratio(ot, pt)
        elo, ehi = boot_ratio(oe, pe)
        print(
            f"Run-ahead {label}, steps 2-{n}: off {ot.mean():.1f} s, p2 {pt.mean():.1f} s, step {ot.mean() / pt.mean():.2f}x "
            f"[{lo:.2f}, {hi:.2f}], exposed {oe.mean():.1f} -> {pe.mean():.1f} s {oe.mean() / pe.mean():.2f}x [{elo:.2f}, {ehi:.2f}]"
        )
        for k in TIMING.values():
            print(
                f"    {k:<22} off {np.mean([s[k] for s in off[1:n]]):6.1f}  p2 {np.mean([s[k] for s in p2[1:n]]):6.1f}"
            )
        om, pm = by_suffix(metrics, off_run)[1:n], by_suffix(metrics, p2_run)[1:n]
        print(
            f"    reward off {np.mean([r['reward'] for r in om]):.4f} p2 {np.mean([r['reward'] for r in pm]):.4f}, "
            + " ".join(
                f"{k} {np.mean([r[k] for r in pm]):.2f}"
                for k in (
                    "prewarm_adopted",
                    "prewarm_mismatch",
                    "prewarm_expired",
                    "prewarm_hold_s",
                )
            )
        )
        pooled["off"] += list(ot)
        pooled["p2"] += list(pt)
        pooled["off_e"] += list(oe)
        pooled["p2_e"] += list(pe)
        pooled["off_r"] += [r["reward"] for r in om]
        pooled["p2_r"] += [r["reward"] for r in pm]
    if len(pooled["off"]) > 0:
        ot, pt, oe, pe, orw, prw = (np.array(v) for v in pooled.values())
        lo, hi = boot_ratio(ot, pt)
        elo, ehi = boot_ratio(oe, pe)
        rlo, rhi = boot_diff(orw, prw)
        print(
            f"Run-ahead pooled n={len(ot)}: off {ot.mean():.1f} s, p2 {pt.mean():.1f} s, step {ot.mean() / pt.mean():.2f}x [{lo:.2f}, {hi:.2f}], "
            f"exposed {oe.mean():.1f} -> {pe.mean():.1f} s {oe.mean() / pe.mean():.2f}x [{elo:.2f}, {ehi:.2f}], "
            f"reward p2 - off {100 * (prw.mean() - orw.mean()):+.1f}pp [{100 * rlo:+.1f}, {100 * rhi:+.1f}]"
        )


if __name__ == "__main__":
    main()
