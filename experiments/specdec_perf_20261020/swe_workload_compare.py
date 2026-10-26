"""Compare the SWE agent rollout shape with single-turn math GRPO rollouts.

    python swe_workload_compare.py <char_dir> <out_dir> [label=wandb.csv ...]

<char_dir> holds the <run>_traj.csv and <run>_calls.csv files that
swe_workload_char.py writes. Each label=wandb.csv pair is a per-step W&B history
CSV from extract_metrics.py. Writes into <out_dir>:
  swe_workload_runs.csv     one row per SWE run: turns, context growth, tokens
                            per call, tool gaps, trajectory phases, outcomes
  swe_workload_by_turn.csv  per run and call-index bucket: context, output,
                            new input, latency, share of trajectories still alive
  swe_workload_by_call.csv  per run and call index: trajectories that reach it,
                            mean context and output (plot input)
  workload_wandb.csv        per W&B run, steps 2-20: samples, turns, tokens per
                            sample, truncation, reward, step phase times
"""

import csv
import math
import statistics as st
import sys
from pathlib import Path

BUCKETS = [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 39), (40, 79), (80, 10**6)]
OBS = ["CmdOutput", "FileEdit", "FileRead", "TaskTracking"]
WANDB = {
    "samples": ["train/num_valid_samples"],
    "turns": ["train/turns_per_sample/mean", "train/avg_turns_per_sample"],
    "turns_max": ["train/turns_per_sample/max", "train/max_turns_per_sample"],
    "gen_tokens": [
        "train/gen_tokens_per_sample/mean",
        "train/mean_gen_tokens_per_sample",
    ],
    "gen_tokens_max": [
        "train/gen_tokens_per_sample/max",
        "train/max_gen_tokens_per_sample",
    ],
    "total_tokens": [
        "train/total_tokens_per_sample/mean",
        "train/mean_total_tokens_per_sample",
    ],
    "prompt_tokens": ["train/mean_prompt_length"],
    "truncation": ["train/truncation_rate"],
    "reward": ["train/total_reward/mean", "train/reward"],
    "zero_adv_groups_pct": ["train/zero_advantage_group_pct"],
    "masked": ["train/swe_agents_train/mask_sample/mean"],
    "groups_all_wrong": ["train/baseline_reward/pct_0"],
    "groups_all_right": ["train/baseline_reward/pct_1"],
    "step_s": ["timing/train/total_step_time"],
    "gen_s": ["timing/train/exposed_generation", "timing/train/generation"],
    "train_s": ["timing/train/policy_training"],
    "logprob_s": ["timing/train/policy_and_reference_logprobs"],
    "refit_s": [
        "timing/train/weight_sync",
        "timing/train/prepare_for_generation/total",
    ],
}


def num(x: str | None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q / 100 * len(s)))] if s else math.nan


def mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else math.nan


def col(rows: list[dict], key: str) -> list[float]:
    return [v for v in (num(r.get(key)) for r in rows) if v is not None]


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def concurrency(spans: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Time-weighted mean, p90 and max of open spans between the first start and last end."""
    events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    if not events:
        return math.nan, math.nan, math.nan
    level, prev, peak = 0, events[0][0], 0
    weight: dict[int, float] = {}
    for t, step in events:
        weight[level] = weight.get(level, 0.0) + t - prev
        level, prev = level + step, t
        peak = max(peak, level)
    total = sum(weight.values())
    if total <= 0:
        return math.nan, math.nan, float(peak)
    acc, p90 = 0.0, 0
    for lvl in sorted(weight):
        acc += weight[lvl]
        if acc >= 0.9 * total:
            p90 = lvl
            break
    return sum(k * v for k, v in weight.items()) / total, float(p90), float(peak)


def run_row(label: str, traj: list[dict], calls: list[dict]) -> dict:
    n = len(traj)
    ncalls = col(traj, "calls")
    comp = col(calls, "completion_tokens")
    later = [c for c in calls if c["call"] != "0"]
    new_in = col(later, "new_input_tokens")
    lat = col(calls, "latency")
    rate = [c / l for c, l in zip(comp, lat) if l > 0 and c >= 64]
    gaps = col(calls, "gap")
    startup = [
        sum(
            num(r.get(k)) or 0.0
            for k in (
                "generation_apptainer_spinup_time",
                "create_runtime_time",
                "connect_to_runtime_time",
                "initialize_runtime_time",
            )
        )
        for r in traj
    ]
    model = col(traj, "total_model_call_time")
    run = col(traj, "openhands_run_time")
    final_eval = col(traj, "final_eval_time")
    wall = [
        (num(r.get("ray_queue_time")) or 0.0)
        + (num(r.get("openhands_run_time")) or 0.0)
        + (num(r.get("final_eval_time")) or 0.0)
        for r in traj
    ]
    obs_total = sum(sum(col(traj, f"obs_{t}")) for t in [*OBS, "Agent", "Error"]) + sum(
        col(traj, "obs_other")
    )
    call_spans = [
        (e - l, e)
        for e, l in ((num(c["end"]), num(c["latency"])) for c in calls)
        if e is not None and l is not None
    ]
    active_mean, active_p90, active_max = concurrency(call_spans)
    sum_prompt = sum(col(traj, "sum_prompt"))
    sum_new = sum(col(traj, "sum_new_input"))
    sum_comp = sum(col(traj, "sum_completion"))
    return {
        "run": label,
        "trajectories": n,
        "calls_mean": mean(ncalls),
        "calls_p50": pct(ncalls, 50),
        "calls_p90": pct(ncalls, 90),
        "calls_p99": pct(ncalls, 99),
        "calls_max": max(ncalls, default=math.nan),
        "first_context": mean(col(traj, "first_prompt")),
        "final_context_mean": mean(col(traj, "last_prompt")),
        "final_context_p90": pct(col(traj, "last_prompt"), 90),
        "final_context_max": max(col(traj, "last_prompt"), default=math.nan),
        "output_per_traj": mean(col(traj, "sum_completion")),
        "output_per_call_mean": mean(comp),
        "output_per_call_p50": pct(comp, 50),
        "output_per_call_p90": pct(comp, 90),
        "output_per_call_p99": pct(comp, 99),
        "new_input_per_call_p50": pct(new_in, 50),
        "new_input_per_call_mean": mean(new_in),
        "input_to_output_no_cache": sum_prompt / sum_comp if sum_comp else math.nan,
        "input_to_output_prefix_cache": sum_new / sum_comp if sum_comp else math.nan,
        "call_latency_p50": pct(lat, 50),
        "call_latency_p90": pct(lat, 90),
        "decode_tok_s_p50": pct(rate, 50),
        "active_calls_mean": active_mean,
        "active_calls_p90": active_p90,
        "active_calls_max": active_max,
        "gap_p50": pct(gaps, 50),
        "gap_p90": pct(gaps, 90),
        "startup_s": mean(startup),
        "model_s": mean(model),
        "exec_s": mean(col(traj, "total_command_exec_time")),
        "openhands_run_s": mean(run),
        "final_eval_s": mean(final_eval),
        "wall_mean_s": mean(wall),
        "wall_p90_s": pct(wall, 90),
        "wall_max_s": max(wall, default=math.nan),
        "model_share": sum(model) / sum(wall) if wall else math.nan,
        **{
            f"tool_{t}": sum(col(traj, f"obs_{t}")) / obs_total
            if obs_total
            else math.nan
            for t in OBS
        },
        "resolved": mean([1.0 if r.get("resolved") == "True" else 0.0 for r in traj]),
        "patch": mean([1.0 if r.get("patch_exists") == "True" else 0.0 for r in traj]),
        "agent_finish": mean(col(traj, "agent_finish")),
    }


def by_turn(label: str, traj: list[dict], calls: list[dict]) -> list[dict]:
    n = len(traj)
    out = []
    for lo, hi in BUCKETS:
        sel = [c for c in calls if lo <= int(c["call"]) <= hi]
        if not sel:
            continue
        alive = sum(1 for r in traj if (num(r.get("calls")) or 0) > lo)
        out.append(
            {
                "run": label,
                "calls": f"{lo + 1}"
                if lo == hi
                else f"{lo + 1}-{hi + 1}"
                if hi < 10**6
                else f"{lo + 1}+",
                "n_calls": len(sel),
                "alive_share": alive / n if n else math.nan,
                "context_mean": mean(col(sel, "prompt_tokens")),
                "output_mean": mean(col(sel, "completion_tokens")),
                "new_input_mean": mean(col(sel, "new_input_tokens")),
                "latency_mean": mean(col(sel, "latency")),
            }
        )
    return out


def by_call(label: str, calls: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for c in calls:
        groups.setdefault(int(c["call"]), []).append(c)
    return [
        {
            "run": label,
            "call": i + 1,
            "trajectories": len(groups[i]),
            "context_mean": mean(col(groups[i], "prompt_tokens")),
            "output_mean": mean(col(groups[i], "completion_tokens")),
        }
        for i in sorted(groups)
    ]


def wandb_row(label: str, path: Path) -> dict:
    rows = [r for r in read(path) if num(r.get("timing/train/total_step_time"))]
    window = [
        r
        for r in rows
        if 2 <= (num(r.get("nemo_rl/step")) or num(r.get("step")) or 0) <= 20
    ] or rows
    rec: dict = {"run": label, "steps": len(window)}
    for name, keys in WANDB.items():
        rec[name] = math.nan
        for k in keys:
            vals = col(window, k)
            if vals:
                rec[name] = mean(vals)
                break
    rec["gen_tokens_per_step"] = rec["samples"] * rec["gen_tokens"]
    rec["gen_share"] = rec["gen_s"] / rec["step_s"]
    return rec


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: f"{v:.4g}" if isinstance(v, float) else v for k, v in r.items()}
            )


def main() -> None:
    char_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, turns, per_call = [], [], []
    for tpath in sorted(char_dir.glob("*_traj.csv")):
        label = tpath.name.removesuffix("_traj.csv")
        traj = read(tpath)
        calls = read(char_dir / f"{label}_calls.csv")
        runs.append(run_row(label, traj, calls))
        turns.extend(by_turn(label, traj, calls))
        per_call.extend(by_call(label, calls))
    write(out_dir / "swe_workload_runs.csv", runs)
    write(out_dir / "swe_workload_by_turn.csv", turns)
    write(out_dir / "swe_workload_by_call.csv", per_call)
    pairs = [a.split("=", 1) for a in sys.argv[3:]]
    if pairs:
        write(out_dir / "workload_wandb.csv", [wandb_row(k, Path(v)) for k, v in pairs])


if __name__ == "__main__":
    main()
