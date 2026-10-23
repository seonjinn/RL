"""Split the SWE per-trajectory time into model calls, tool execution, agent overhead, final eval and setup.

    python swe_breakdown.py <csv_dir> [report/data/swe_time_breakdown.csv]

<csv_dir> holds the per-run W&B history CSVs that extract_metrics.py writes.
Each value is the mean over trainer steps 1-20 of the per-step trajectory mean.
final_eval also splits into steps 4-8 and the other steps, because the frozen
scale2 arms hit final-eval bursts in steps 4-8 that the control did not.
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = [
    "swe-sd1020-scale2-r16-baseline",
    "swe-sd1020-scale2-r16-frozen-dflash-k3",
    "swe-sd1020-scale2-r16-frozen-dspark-k4",
    "swe-sd1020-scale2-r16-online-dflash-k3-lr1e4-tp4",
    "swe-sd1020-scale2-r16-online-dspark-k4-lr1e4-tp4",
    "swe-sd1020-scale2-r16-online-dspark-k8-lr1e4-tp4",
    "swe-sd1020-sc-r16-baseline",
    "swe-sd1020-sc-r16-frozen-dspark-k4",
    "swe-sd1020-scale2-r16-online-dflash-k7-lr1e4-tp4",
    "swe-sd1022-sc-r16-baseline",
    "swe-sd1022-sc-r16-frozen-dflash-k3",
    "swe-sd1022-sc-r16-frozen-dspark-k4-r2",
    "swe-sd1022-sc-r16-online-dflash-k3-lr1e4-tp4",
    "swe-sd1022-sc-r16-online-dspark-k4-lr1e4-tp4",
    "swe-sd1023-sc-r16-baseline",
    "swe-sd1023-sc-r16-regen-frozen-dflash-k3",
    "swe-sd1023-sc-r16-regen-frozen-dflash-k7",
    "swe-sd1023-sc-r16-regen-online-dflash-k3-lr1e4-sw8k",
    "swe-sd1023-sc-r16-regen-online-dflash-k7-lr1e4-sw8k",
]
STEPS = range(1, 21)
BURST_STEPS = range(4, 9)
P = "train/swe_agents_train/"
KEYS = {
    "model": P + "total_model_call_time/mean",
    "tool": P + "total_command_exec_time/mean",
    "agent": P + "openhands_run_time/mean",
    "eval": P + "final_eval_time/mean",
    "spinup": P + "generation_apptainer_spinup_time/mean",
    "rt_create": P + "create_runtime_time/mean",
    "rt_connect": P + "connect_to_runtime_time/mean",
    "rt_init": P + "initialize_runtime_time/mean",
    "queue": P + "ray_queue_time/mean",
    "traj": "train/trajectory_duration_s",
    "gen_tok": P + "gen_tokens_per_sample/mean",
    "step": "timing/train/total_step_time",
    "train": "timing/train/policy_training",
}
FIELDS = [
    "run",
    "steps",
    "model_call_s",
    "tool_exec_s",
    "agent_other_s",
    "final_eval_s",
    "final_eval_steps4_8_s",
    "final_eval_other_steps_s",
    "setup_s",
    "queue_s",
    "trajectory_s",
    "model_call_tok_per_s",
    "step_s",
    "policy_training_s",
]


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def per_step(path: Path) -> dict[int, dict[str, float]]:
    # One trainer step spans several W&B rows; keep the last value of each key per step.
    steps: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(path.open()):
        try:
            step = int(float(row["nemo_rl/step"]))
        except (KeyError, TypeError, ValueError):
            continue
        for name, key in KEYS.items():
            try:
                steps.setdefault(step, {})[name] = float(row[key])
            except (KeyError, TypeError, ValueError):
                pass
    return steps


def summarize(run: str, steps: dict[int, dict[str, float]]) -> dict[str, str]:
    def avg(name: str, window: range = STEPS) -> float:
        return mean([steps[s][name] for s in window if s in steps and name in steps[s]])

    other_steps = [s for s in STEPS if s not in BURST_STEPS]
    setup = sum(avg(k) for k in ("spinup", "rt_create", "rt_connect", "rt_init"))
    values = {
        "steps": len([s for s in STEPS if s in steps]),
        "model_call_s": avg("model"),
        "tool_exec_s": avg("tool"),
        "agent_other_s": avg("agent") - avg("model") - avg("tool"),
        "final_eval_s": avg("eval"),
        "final_eval_steps4_8_s": avg("eval", BURST_STEPS),
        "final_eval_other_steps_s": mean(
            [steps[s]["eval"] for s in other_steps if "eval" in steps.get(s, {})]
        ),
        "setup_s": setup,
        "queue_s": avg("queue"),
        "trajectory_s": avg("traj"),
        "model_call_tok_per_s": avg("gen_tok") / avg("model"),
        "step_s": avg("step"),
        "policy_training_s": avg("train"),
    }
    out = {"run": run.removeprefix("swe-").removeprefix("sd1020-")}
    for k, v in values.items():
        out[k] = str(v) if isinstance(v, int) else ("" if math.isnan(v) else f"{v:.2f}")
    return out


def main() -> None:
    csv_dir = Path(sys.argv[1])
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else HERE / "report/data/swe_time_breakdown.csv"
    )
    rows = [
        summarize(run, per_step(csv_dir / f"{run}.csv"))
        for run in RUNS
        if (csv_dir / f"{run}.csv").exists()
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
