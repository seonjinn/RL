"""Separate the SWE SpecDec gain from run-to-run variation.

    python swe_variation.py <csv_dir> [report/data/swe_variation.csv]

<csv_dir> holds the per-run W&B history CSVs that extract_metrics.py writes.
For each run: the agent workload (turns, generated and total tokens, masked
samples, tool time), then model-call tok/s and step time compared with the
cohort control step by step. A step pairs the same prompts in both runs. The
90% interval resamples the paired steps, so it shows how far step-to-step
variation alone can move the ratio.
"""

import csv
import math
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
CONTROL = {
    "swe-sd1020-scale2-r16-baseline": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-frozen-dflash-k3": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-frozen-dspark-k4": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-online-dflash-k3-lr1e4-tp4": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-online-dspark-k4-lr1e4-tp4": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-online-dspark-k8-lr1e4-tp4": "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-sc-r16-baseline": "swe-sd1020-sc-r16-baseline",
    "swe-sd1020-sc-r16-frozen-dspark-k4": "swe-sd1020-sc-r16-baseline",
}
STEPS = range(1, 21)
BOOTSTRAP = 2000
P = "train/swe_agents_train/"
KEYS = {
    "gen_tok": P + "gen_tokens_per_sample/mean",
    "total_tok": P + "total_tokens_per_sample/mean",
    "model": P + "total_model_call_time/mean",
    "tool": P + "total_command_exec_time/mean",
    "mask": P + "mask_sample/mean",
    "turns": "train/turns_per_sample/mean",
    "step": "timing/train/total_step_time",
}
FIELDS = [
    "run",
    "turns",
    "gen_tokens",
    "total_tokens",
    "masked_frac",
    "tool_exec_s",
    "model_call_tok_per_s",
    "model_call_cv",
    "model_call_ratio",
    "model_call_ratio_p05",
    "model_call_ratio_p95",
    "model_call_steps_faster",
    "step_cv",
    "step_ratio",
    "step_ratio_p05",
    "step_ratio_p95",
    "step_steps_faster",
]


def per_step(path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(path.open()):
        try:
            step = int(float(row["nemo_rl/step"]))
        except (KeyError, TypeError, ValueError):
            continue
        for name, key in KEYS.items():
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                steps.setdefault(step, {})[name] = value
    steps = {s: v for s, v in steps.items() if s in STEPS}
    for v in steps.values():
        if v.get("model"):
            v["tok_per_s"] = v["gen_tok"] / v["model"]
    return steps


def paired(
    arm: dict[int, dict[str, float]],
    control: dict[int, dict[str, float]],
    name: str,
    higher_is_faster: bool,
) -> tuple[float, float, float, int, int]:
    steps = [s for s in arm if name in arm[s] and name in control.get(s, {})]

    def ratio(sample: list[int]) -> float:
        a = sum(arm[s][name] for s in sample)
        c = sum(control[s][name] for s in sample)
        return a / c if higher_is_faster else c / a

    rng = random.Random(0)
    boot = sorted(ratio([rng.choice(steps) for _ in steps]) for _ in range(BOOTSTRAP))
    faster = sum(ratio([s]) > 1 for s in steps)
    return (
        ratio(steps),
        boot[int(0.05 * BOOTSTRAP)],
        boot[int(0.95 * BOOTSTRAP)],
        faster,
        len(steps),
    )


def summarize(
    run: str, arm: dict[int, dict[str, float]], control: dict[int, dict[str, float]]
) -> dict[str, str]:
    def mean(name: str) -> float:
        return st.mean(v[name] for v in arm.values() if name in v)

    def cv(name: str) -> float:
        xs = [v[name] for v in arm.values() if name in v]
        return st.pstdev(xs) / st.mean(xs)

    tok = paired(arm, control, "tok_per_s", higher_is_faster=True)
    step = paired(arm, control, "step", higher_is_faster=False)
    out = {
        "run": run.removeprefix("swe-sd1020-"),
        "turns": f"{mean('turns'):.2f}",
        "gen_tokens": f"{mean('gen_tok'):.0f}",
        "total_tokens": f"{mean('total_tok'):.0f}",
        "masked_frac": f"{mean('mask'):.4f}",
        "tool_exec_s": f"{mean('tool'):.2f}",
        "model_call_tok_per_s": f"{mean('tok_per_s'):.1f}",
        "model_call_cv": f"{cv('tok_per_s'):.3f}",
        "step_cv": f"{cv('step'):.3f}",
    }
    for prefix, (r, lo, hi, won, n) in (("model_call", tok), ("step", step)):
        out[f"{prefix}_ratio"] = f"{r:.3f}"
        out[f"{prefix}_ratio_p05"] = f"{lo:.3f}"
        out[f"{prefix}_ratio_p95"] = f"{hi:.3f}"
        out[f"{prefix}_steps_faster"] = f"{won}/{n}"
    return out


def main() -> None:
    csv_dir = Path(sys.argv[1])
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else HERE / "report/data/swe_variation.csv"
    )
    runs = {r: per_step(csv_dir / f"{r}.csv") for r in CONTROL}
    rows = [summarize(r, runs[r], runs[c]) for r, c in CONTROL.items()]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
