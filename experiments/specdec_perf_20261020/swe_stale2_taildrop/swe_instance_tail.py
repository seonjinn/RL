"""Rank SWE instances by group span and resolve count across finished runs.

    python swe_instance_tail.py <swebench_results_dir> [<swebench_results_dir> ...]

Each results dir is the "SWE agents results root for this run" from ray-driver.log.
One instance in one root is one GRPO group. The group span is the slowest trajectory
(ray queue + openhands run + final eval), which is what the ready-first gate waits for.
"""

import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Instance:
    groups: int = 0
    attempts: int = 0
    missing: int = 0
    resolved: int = 0
    no_patch: int = 0
    spans: list[float] = field(default_factory=list)
    medians: list[float] = field(default_factory=list)
    init_max: list[float] = field(default_factory=list)


def trajectory(path: str) -> tuple[float, bool, float, bool]:
    m = json.load(open(path))
    total = sum(
        m.get(k) or 0.0
        for k in ("ray_queue_time", "openhands_run_time", "final_eval_time")
    )
    return (
        total,
        bool(m.get("resolved")),
        m.get("initialize_runtime_time") or 0.0,
        bool(m.get("patch_exists")),
    )


def main() -> None:
    instances: defaultdict[str, Instance] = defaultdict(Instance)
    spans: list[float] = []
    for root in sys.argv[1:]:
        groups: defaultdict[str, list[tuple[float, bool, float, bool] | None]] = (
            defaultdict(list)
        )
        for d in glob.glob(f"{root}/*/"):
            iid = re.sub(r"_\d{10,}_[0-9a-f]{8}$", "", os.path.basename(d.rstrip("/")))
            p = f"{d}nemo_gym_metrics.json"
            groups[iid].append(trajectory(p) if os.path.exists(p) else None)
        for iid, trajs in groups.items():
            ok = [t for t in trajs if t]
            a = instances[iid]
            a.groups += 1
            a.attempts += len(trajs)
            a.missing += len(trajs) - len(ok)
            a.resolved += sum(t[1] for t in ok)
            a.no_patch += sum(not t[3] for t in ok)
            if ok:
                a.spans.append(max(t[0] for t in ok))
                a.medians.append(st.median(t[0] for t in ok))
                a.init_max.append(max(t[2] for t in ok))
                spans.append(a.spans[-1])

    q = sorted(spans)
    print(
        f"groups={len(q)} instances={len(instances)} span p50={q[len(q) // 2]:.0f} p90={q[int(len(q) * 0.9)]:.0f} max={q[-1]:.0f}"
    )
    print(
        "instance groups attempts missing resolved no_patch span_mean span_max median_traj init_max"
    )
    for iid, a in sorted(
        instances.items(), key=lambda kv: -st.mean(kv[1].spans or [0.0])
    ):
        if not a.spans:
            print(iid, a.groups, a.attempts, a.missing, "-")
            continue
        print(
            iid,
            a.groups,
            a.attempts,
            a.missing,
            a.resolved,
            a.no_patch,
            f"{st.mean(a.spans):.0f}",
            f"{max(a.spans):.0f}",
            f"{st.mean(a.medians):.0f}",
            f"{max(a.init_max):.0f}",
        )


if __name__ == "__main__":
    main()
