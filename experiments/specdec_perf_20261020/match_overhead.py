"""Pool the drafter training overhead over the first matched TP1 set and the two same-window reps.

    python match_overhead.py <dir with one extract_metrics.py CSV per run>

Each rep pairs an online arm with the baseline of the same impl and the same rep over steps 3-20.
Per rep: train overhead = mean(arm train) / mean(base train) - 1, plus added seconds.
Pooled = mean over reps; the CI resamples steps inside every run (4000 draws, 90%).
Impl gap = ours - teammate, with the same joint resampling.
"""

import csv
import random
import sys
from pathlib import Path

D = Path(sys.argv[1])
FIRST, LAST = 3, 20
K = {
    "train": "timing/train/policy_training",
    "step": "timing/train/total_step_time",
    "gen": "timing/train/generation",
    "logprob": "timing/train/policy_and_reference_logprobs",
    "refit": "timing/train/prepare_for_generation/total",
    "al": "train/vllm/spec_acceptance_length",
}
REPS = ["", "-rep1025a", "-rep1025b"]
IMPL = {"ours": "q30-sd1020-match-tp1", "teammate": "q30-markx1021-match-tp1"}
ARMS = ["online-dflash-k7-lr1e4", "online-dspark-k8-lr1e4"]


def load(name: str) -> dict[int, dict[str, float]]:
    out = {}
    for r in csv.DictReader(open(D / f"{name}.csv")):
        s = r.get("nemo_rl/step")
        if not s:
            continue
        s = int(float(s))
        if FIRST <= s <= LAST:
            out[s] = {
                k: float(r[v]) for k, v in K.items() if r.get(v) not in (None, "")
            }
    return out


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def pair(
    impl: str, arm: str, rep: str
) -> tuple[list[float], list[float], list[float], list[float], dict]:
    B, A = load(f"{IMPL[impl]}-baseline{rep}"), load(f"{IMPL[impl]}-{arm}{rep}")
    st = [s for s in sorted(set(B) & set(A)) if "train" in B[s] and "train" in A[s]]
    extra = {
        k: mean([A[s][k] for s in st if k in A[s]])
        - mean([B[s][k] for s in st if k in B[s]])
        for k in ("logprob", "refit", "gen", "step")
    }
    extra["al"] = mean([A[s]["al"] for s in st if "al" in A[s]])
    extra["n"] = len(st)
    return (
        [B[s]["train"] for s in st],
        [A[s]["train"] for s in st],
        [B[s]["step"] for s in st],
        [A[s]["step"] for s in st],
        extra,
    )


rnd = random.Random(0)
N = 4000


def resample(xs: list[float]) -> list[float]:
    return [rnd.choice(xs) for _ in xs]


def ci(v: list[float]) -> tuple[float, float]:
    v = sorted(v)
    return v[int(0.05 * len(v))], v[int(0.95 * len(v))]


rows = []
pooled: dict[tuple[str, str], list[float]] = {}
print(
    f"{'impl':9} {'arm':24} {'rep':10} {'n':>2} {'base s':>7} {'arm s':>7} {'+train s':>8} {'train %':>7} {'step x':>6} {'AL':>5}"
)
for arm in ARMS:
    for impl in IMPL:
        data = [pair(impl, arm, rep) for rep in REPS]
        for rep, (tb, ta, sb, sa, ex) in zip(REPS, data):
            ov = mean(ta) / mean(tb) - 1
            rows.append(
                dict(
                    impl=impl,
                    arm=arm,
                    rep=rep.strip("-") or "first",
                    n=ex["n"],
                    base_train_s=round(mean(tb), 2),
                    arm_train_s=round(mean(ta), 2),
                    d_train_s=round(mean(ta) - mean(tb), 2),
                    train_overhead=round(ov, 4),
                    step_x=round(mean(sb) / mean(sa), 4),
                    d_step_s=round(ex["step"], 2),
                    d_gen_s=round(ex["gen"], 2),
                    d_logprob_s=round(ex["logprob"], 2),
                    d_refit_s=round(ex["refit"], 2),
                    al=round(ex["al"], 3),
                )
            )
            r = rows[-1]
            print(
                f"{impl:9} {arm:24} {r['rep']:10} {r['n']:2} {r['base_train_s']:7.1f} {r['arm_train_s']:7.1f} {r['d_train_s']:+8.2f} {100 * ov:6.1f}% {r['step_x']:6.3f} {r['al']:5.2f}"
            )
        draws_pct, draws_s = [], []
        for _ in range(N):
            o, d = [], []
            for tb, ta, *_ in data:
                b, a = mean(resample(tb)), mean(resample(ta))
                o.append(a / b - 1)
                d.append(a - b)
            draws_pct.append(mean(o))
            draws_s.append(mean(d))
        pooled[(impl, arm)] = draws_pct
        point = mean([mean(ta) / mean(tb) - 1 for tb, ta, *_ in data])
        point_s = mean([mean(ta) - mean(tb) for tb, ta, *_ in data])
        lo, hi = ci(draws_pct)
        slo, shi = ci(draws_s)
        print(
            f"  POOLED {impl:9} {arm:24} train +{100 * point:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  +{point_s:.2f} s [{slo:.2f}, {shi:.2f}]"
        )
        rows.append(
            dict(
                impl=impl,
                arm=arm,
                rep="pooled",
                train_overhead=round(point, 4),
                train_lo=round(lo, 4),
                train_hi=round(hi, 4),
                d_train_s=round(point_s, 2),
                d_train_lo=round(slo, 2),
                d_train_hi=round(shi, 2),
            )
        )
    gap = [o - t for o, t in zip(pooled[("ours", arm)], pooled[("teammate", arm)])]
    pt = mean(
        [
            r["train_overhead"]
            for r in rows
            if r["impl"] == "ours" and r["arm"] == arm and r["rep"] == "pooled"
        ]
    ) - mean(
        [
            r["train_overhead"]
            for r in rows
            if r["impl"] == "teammate" and r["arm"] == arm and r["rep"] == "pooled"
        ]
    )
    lo, hi = ci(gap)
    print(
        f"  GAP ours-teammate {arm}: {100 * pt:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    )
    rows.append(
        dict(
            impl="ours-teammate",
            arm=arm,
            rep="gap",
            train_overhead=round(pt, 4),
            train_lo=round(lo, 4),
            train_hi=round(hi, 4),
        )
    )

# Run-to-run noise: spread of the baseline train time over the 3 reps of one impl.
for impl in IMPL:
    bs = [
        mean(
            [
                v["train"]
                for v in load(f"{IMPL[impl]}-baseline{rep}").values()
                if "train" in v
            ]
        )
        for rep in REPS
    ]
    print(
        f"baseline train s {impl}: {', '.join(f'{b:.1f}' for b in bs)} (spread {100 * (max(bs) / min(bs) - 1):.1f}%)"
    )

fields = list(dict.fromkeys(k for r in rows for k in r))
with open(D / "overhead_pooled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(rows)
