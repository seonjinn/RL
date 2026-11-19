"""Model the SWE GRPO step as a ready_first gate and score rollout levers.

    python swe_gate_model.py <report/data>

Reads swe_gate_arms.csv (per arm: prompts per step, train-side time, measured
exposed generation and step, env startup per rollout) and swe_gate_spans.csv
(measured group spans, dispatch to last trajectory). Writes swe_gate_levers.csv.

The model follows ReadyFirstSampler. It never drops a rollout. At each step
start it refills the pool of in-flight groups to the cap, and the step trains on
the first PPS groups to finish. The step waits for that last group (exposed
generation), then spends the train-side time. Staleness 1 gives a cap of
2 x PPS. A larger cap stands for staleness 2. Prewarm cuts the env startup from
the first k groups of each refill, because the sandbox starts while the batch
waits for admission.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STEPS = 3000
WARMUP = 100

# (lever, pool size in PPS units plus extra fraction, prewarmed fraction of a refill, startup fraction hidden)
LEVERS = [
    ("baseline (staleness 1)", 2, 0.0, 0.0, 0.0),
    ("staleness 2, pool 2xPPS+PPS/8", 2, 0.125, 0.0, 0.0),
    ("staleness 2, pool 2xPPS+PPS/4", 2, 0.25, 0.0, 0.0),
    ("staleness 2, pool 2xPPS+PPS/2", 2, 0.5, 0.0, 0.0),
    ("staleness 2, pool 3xPPS", 3, 0.0, 0.0, 0.0),
    ("prewarm all, hide 75% startup", 2, 0.0, 1.0, 0.75),
    ("prewarm all, hide 100% startup", 2, 0.0, 1.0, 1.0),
    ("prewarm 1/2 batch", 2, 0.0, 0.5, 1.0),
    ("prewarm 1/4 batch", 2, 0.0, 0.25, 1.0),
    ("staleness 2 + prewarm all", 3, 0.0, 1.0, 1.0),
]


def simulate(
    spans: np.ndarray, train_s: float, pps: int, pool: int, prewarm: int, cut: float
) -> float:
    rng = np.random.default_rng(0)
    t, inflight, exposed = 0.0, [], []
    for _ in range(STEPS):
        new = rng.choice(spans, pool - len(inflight))
        new[:prewarm] = np.maximum(new[:prewarm] - cut, 1.0)
        inflight = sorted(inflight + list(t + new))
        e = max(0.0, inflight[pps - 1] - t)
        inflight = inflight[pps:]
        exposed.append(e)
        t += train_s + e
    return float(np.mean(exposed[WARMUP:]))


def main() -> None:
    data = Path(sys.argv[1])
    spans: dict[str, list[float]] = defaultdict(list)
    with open(data / "swe_gate_spans.csv") as f:
        for r in csv.DictReader(f):
            spans[r["arm"]].append(float(r["group_span_s"]))
    with open(data / "swe_gate_arms.csv") as f:
        arms = list(csv.DictReader(f))

    rows = []
    for a in arms:
        name, pps = a["arm"], int(a["prompts_per_step"])
        train_s, startup = float(a["train_side_s"]), float(a["startup_s"])
        s = np.array(spans[name])
        base_step = None
        print(
            f"{name}: measured exposed {float(a['exposed_generation_s']):.0f} s, step {float(a['step_s']):.0f} s"
        )
        for lever, mult, extra, frac, hide in LEVERS:
            pool = mult * pps + (max(1, round(extra * pps)) if extra else 0)
            prewarm = round(frac * pps)
            e = simulate(s, train_s, pps, pool, prewarm, hide * startup)
            step = e + train_s
            base_step = base_step or step
            rows.append(
                [
                    name,
                    lever,
                    pool,
                    prewarm,
                    f"{hide * startup:.1f}",
                    f"{e:.1f}",
                    f"{step:.1f}",
                    f"{base_step / step:.2f}",
                ]
            )
            print(
                f"  {lever:<34} exposed {e:5.0f} s  step {step:5.0f} s  {base_step / step:.2f}x"
            )

    with open(data / "swe_gate_levers.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "arm",
                "lever",
                "pool_groups",
                "prewarmed_groups",
                "startup_hidden_s",
                "model_exposed_s",
                "model_step_s",
                "speedup_vs_baseline",
            ]
        )
        w.writerows(rows)


if __name__ == "__main__":
    main()
