"""Extract per-call and per-trajectory SWE agent workload shape from one run.

    python swe_workload_char.py <run_label> <swebench_results_dir> <out_dir>

<swebench_results_dir> is the per-run directory that the swe_agents server logs as
"SWE agents results root for this run". Writes two CSVs into <out_dir>:
  <run_label>_calls.csv  one row per LLM call: prompt and completion tokens,
                         latency, end time, gap since the previous call ended
  <run_label>_traj.csv   one row per trajectory: call count, context growth,
                         tool mix, phase times, resolved

prompt_tokens is the full chat context of each call. vLLM prefix caching reuses
the shared prefix, so the new prefill work of call i is about
prompt_i - prompt_(i-1) - completion_(i-1) (the tool observation) when the
cached blocks survive.
"""

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

CALL_FIELDS = [
    "traj",
    "call",
    "prompt_tokens",
    "completion_tokens",
    "new_input_tokens",
    "latency",
    "end",
    "gap",
]
OBS_TYPES = ["CmdOutput", "FileEdit", "FileRead", "TaskTracking", "Agent", "Error"]
METRIC_FIELDS = [
    "ray_queue_time",
    "generation_apptainer_spinup_time",
    "create_runtime_time",
    "connect_to_runtime_time",
    "deep_reset_time",
    "initialize_runtime_time",
    "total_model_call_time",
    "total_command_exec_time",
    "complete_runtime_time",
    "close_runtime_time",
    "openhands_run_time",
    "final_eval_time",
    "agent_peak_rss_mb",
    "eval_peak_rss_mb",
    "resolved",
    "patch_exists",
    "agent_error_kind",
]
TRAJ_FIELDS = [
    "traj",
    "gen_start",
    "calls",
    "first_prompt",
    "last_prompt",
    "sum_prompt",
    "sum_new_input",
    "sum_completion",
    "max_completion",
    "sum_latency",
    "agent_finish",
    "patch_lines",
    *[f"obs_{t}" for t in OBS_TYPES],
    "obs_other",
    *[f"exec_{t}" for t in OBS_TYPES],
    *METRIC_FIELDS,
]


def epoch(ts: str) -> float:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def obs_type(name: str) -> str:
    base = name.removesuffix("Observation")
    return base if base in OBS_TYPES else "other"


def one_traj(d: str) -> tuple[list[dict], dict] | None:
    mpath = os.path.join(d, "nemo_gym_metrics.json")
    outs = glob.glob(os.path.join(d, "trajectories", "*", "output.jsonl"))
    if not os.path.exists(mpath) or not outs:
        return None
    with open(mpath) as f:
        m = json.load(f)
    with open(outs[0]) as f:
        o = json.loads(f.readline())
    om = o.get("metrics") or {}
    usage = om.get("token_usages") or []
    lat = om.get("response_latencies") or []
    name = os.path.basename(d)
    calls = []
    prev_end = None
    prev_ctx = 0
    for i, (u, r) in enumerate(zip(usage, lat)):
        p, c = u["prompt_tokens"], u["completion_tokens"]
        end = epoch(r["timestamp"])
        gap = end - r["latency"] - prev_end if prev_end is not None else ""
        calls.append(
            {
                "traj": name,
                "call": i,
                "prompt_tokens": p,
                "completion_tokens": c,
                "new_input_tokens": p - prev_ctx,
                "latency": round(r["latency"], 4),
                "end": round(end, 3),
                "gap": round(gap, 4) if gap != "" else "",
            }
        )
        prev_end, prev_ctx = end, p + c
    obs = Counter()
    exec_s: defaultdict[str, float] = defaultdict(float)
    for a in om.get("action_execution_latencies") or []:
        t = obs_type(a.get("observation_type", ""))
        obs[t] += 1
        exec_s[t] += a.get("latency") or 0.0
    history = o.get("history") or []
    agent_actions = [h.get("action") for h in history if h.get("source") == "agent"]
    patch = m.get("model_patch") or ""
    row = {
        "traj": name,
        "gen_start": m.get("generation_start_timestamp", ""),
        "calls": len(calls),
        "first_prompt": calls[0]["prompt_tokens"] if calls else 0,
        "last_prompt": calls[-1]["prompt_tokens"] if calls else 0,
        "sum_prompt": sum(c["prompt_tokens"] for c in calls),
        "sum_new_input": sum(c["new_input_tokens"] for c in calls),
        "sum_completion": sum(c["completion_tokens"] for c in calls),
        "max_completion": max((c["completion_tokens"] for c in calls), default=0),
        "sum_latency": round(sum(c["latency"] for c in calls), 3),
        "agent_finish": int("finish" in agent_actions),
        "patch_lines": patch.count("\n"),
        **{f"obs_{t}": obs[t] for t in OBS_TYPES},
        "obs_other": obs["other"],
        **{f"exec_{t}": round(exec_s[t], 3) for t in OBS_TYPES},
        **{k: m.get(k, "") for k in METRIC_FIELDS},
    }
    return calls, row


def main() -> None:
    label, root, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    dirs = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = [r for r in ex.map(one_traj, dirs, chunksize=8) if r]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{label}_calls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, CALL_FIELDS)
        w.writeheader()
        for calls, _ in results:
            w.writerows(calls)
    with open(os.path.join(out_dir, f"{label}_traj.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, TRAJ_FIELDS)
        w.writeheader()
        w.writerows(row for _, row in results)
    print(f"{label}: {len(results)}/{len(dirs)} trajectories")


if __name__ == "__main__":
    main()
