"""Collect per-sample SWE final-eval metrics from the Gym results dirs of the Q30 SWE runs.

    python swe_eval_samples.py <out.csv> [<log glob>]

The default glob covers every swe-sd102* run under the user's clones. Each driver log
names its Gym results root, and each sample dir holds one nemo_gym_metrics.json.
"""

import csv
import glob
import json
import os
import re
import sys

U = "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna"
LOG_GLOB = (
    f"{U}/*/experiments/specdec_perf_20261020/logs/swe-sd102*/*-logs/ray-driver.log"
)
ROOT_RE = re.compile(r"SWE agents results root for this run: (\S+)")
SAMPLE_SUFFIX_RE = re.compile(r"_\d{13}_[0-9a-f]{8}$")
METRICS = [
    "final_eval_time",
    "resolved",
    "eval_timed_out",
    "eval_launch_failed",
    "patch_exists",
    "openhands_run_time",
    "total_model_call_time",
    "evaluation_start_timestamp",
]


def results_roots(log: str) -> list[str]:
    roots: list[str] = []
    with open(log, errors="replace") as f:
        for line in f:
            m = ROOT_RE.search(line)
            if m and m.group(1) not in roots:
                roots.append(m.group(1))
    return roots


def main() -> None:
    log_glob = sys.argv[2] if len(sys.argv) > 2 else LOG_GLOB
    seen: set[str] = set()
    with open(sys.argv[1], "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["run", "job", "instance", *METRICS])
        for log in sorted(glob.glob(log_glob)):
            parts = log.split("/")
            run, job = parts[-3], parts[-2].removesuffix("-logs")
            n = 0
            for root in results_roots(log):
                # A resumed or re-queued job can log a root that an earlier job already wrote.
                if root in seen or not os.path.isdir(root):
                    continue
                seen.add(root)
                for path in glob.glob(f"{root}/*/nemo_gym_metrics.json"):
                    try:
                        with open(path) as mf:
                            d = json.load(mf)
                    except (OSError, ValueError):
                        continue
                    inst = SAMPLE_SUFFIX_RE.sub(
                        "", os.path.basename(os.path.dirname(path))
                    )
                    out.writerow([run, job, inst, *(d.get(k) for k in METRICS)])
                    n += 1
            print(run, job, n, flush=True)


if __name__ == "__main__":
    main()
