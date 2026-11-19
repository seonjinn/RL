"""Show how many resolved SWE samples each final-eval timeout would cut, and which instances time out.

    python swe_eval_timeout.py <samples.csv>

<samples.csv> is the output of swe_eval_samples.py. An eval at or above 1190 s counts as a
hit on the Gym default 1200 s cap.
"""

import collections
import csv
import statistics as st
import sys
from dataclasses import dataclass

CAP_S = 1190
TIMEOUTS_S = (120, 180, 240, 300, 360, 420, 450, 480, 540, 600, 900, 1200)
SLOW_RESOLVED_S = 300


@dataclass(frozen=True)
class Sample:
    run: str
    instance: str
    eval_s: float
    resolved: bool


def quantile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load(path: str) -> list[Sample]:
    samples = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["final_eval_time"])
            except (TypeError, ValueError):
                continue
            samples.append(Sample(r["run"], r["instance"], t, r["resolved"] == "True"))
    return samples


def main() -> None:
    samples = load(sys.argv[1])
    resolved = [s.eval_s for s in samples if s.resolved]
    every = [s.eval_s for s in samples]
    print(
        f"samples {len(samples)} runs {len({s.run for s in samples})} "
        f"instances {len({s.instance for s in samples})} resolved {len(resolved)}"
    )
    for name, xs in (("resolved", resolved), ("all", every)):
        qs = " ".join(
            f"p{p * 100:g} {quantile(xs, p):.0f}" for p in (0.5, 0.9, 0.95, 0.99, 0.999)
        )
        print(f"{name} eval s: {qs} max {max(xs):.0f}")

    print("\ntimeout_s resolved_lost frac_resolved_lost samples_over frac_over")
    for t in TIMEOUTS_S:
        lost = sum(x > t for x in resolved)
        over = sum(x > t for x in every)
        print(
            f"{t:9d} {lost:13d} {lost / len(resolved):18.4f} {over:12d} {over / len(every):9.4f}"
        )

    by_inst: dict[str, list[Sample]] = collections.defaultdict(list)
    for s in samples:
        by_inst[s.instance].append(s)

    print(f"\ninstances with an eval >= {CAP_S} s:")
    for inst, ss in sorted(
        by_inst.items(), key=lambda kv: -sum(s.eval_s >= CAP_S for s in kv[1])
    ):
        capped = sum(s.eval_s >= CAP_S for s in ss)
        if capped == 0:
            break
        print(
            f"  {inst:45s} n={len(ss):4d} capped={capped:4d} resolved={sum(s.resolved for s in ss):3d}"
        )

    print(f"\ninstances with a resolved eval > {SLOW_RESOLVED_S} s:")
    for inst, ss in sorted(by_inst.items()):
        rr = [s.eval_s for s in ss if s.resolved]
        if rr and max(rr) > SLOW_RESOLVED_S:
            print(
                f"  {inst:45s} n={len(ss):4d} resolved={len(rr):3d} "
                f"resolved eval med={st.median(rr):.0f} max={max(rr):.0f} "
                f"all med={st.median([s.eval_s for s in ss]):.0f}"
            )


if __name__ == "__main__":
    main()
