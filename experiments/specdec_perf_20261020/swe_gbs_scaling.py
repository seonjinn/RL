"""Compare SWE GRPO steps at two rollout batch sizes on the same nodes.

    python swe_gbs_scaling.py <out.csv> gbs=label=wandb.csv ...

Each argument names the global batch (prompts x generations), a row label and a
per-step W&B history CSV from extract_metrics.py. For each run the script
averages steps 2-20 (step 1 holds the cold start) and writes one row: phase
times, samples per second, training tokens per second, agent time split and
container memory. Throughput columns divide by the step time, so they compare
across batch sizes. Step ratios do not, because the steps hold different work.
"""

import csv
import math
import statistics as st
import sys
from pathlib import Path

STEPS = range(2, 21)
P = "train/swe_agents_train/"
KEYS = {
    "step_s": ["timing/train/total_step_time"],
    "gen_s": ["timing/train/exposed_generation"],
    "train_s": ["timing/train/policy_training"],
    "logprob_s": ["timing/train/policy_and_reference_logprobs"],
    "refit_s": ["timing/train/weight_sync"],
    "samples": ["train/num_valid_samples"],
    "train_tokens": ["train/global_valid_toks"],
    "gen_tokens": [P + "gen_tokens_per_sample/mean"],
    "turns": ["train/turns_per_sample/mean"],
    "queue_s": [P + "ray_queue_time/mean"],
    "model_s": [P + "total_model_call_time/mean"],
    "model_max_s": [P + "total_model_call_time/max"],
    "run_s": [P + "openhands_run_time/mean"],
    "run_max_s": [P + "openhands_run_time/max"],
    "eval_s": [P + "final_eval_time/mean"],
    "agent_rss_mb": [P + "agent_peak_rss_mb/mean"],
    "agent_rss_max_mb": [P + "agent_peak_rss_mb/max"],
    "accept_len": ["train/vllm/spec_acceptance_length"],
    "reward": ["train/total_reward/mean", "train/reward"],
    "masked": [P + "mask_sample/mean"],
}


def num(x: str | None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def per_step(path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            step = num(row.get("nemo_rl/step"))
            if step is None or int(step) not in STEPS:
                continue
            for name, keys in KEYS.items():
                for k in keys:
                    v = num(row.get(k))
                    if v is not None:
                        steps.setdefault(int(step), {})[name] = v
                        break
    return steps


def summarize(gbs: int, label: str, path: Path) -> dict:
    steps = per_step(path)

    def mean(name: str) -> float:
        xs = [v[name] for v in steps.values() if name in v]
        # The V2 counters skip steps with no finished SpecDec request
        if name == "accept_len":
            xs = [x for x in xs if x > 0]
        return st.mean(xs) if xs else math.nan

    def total(name: str) -> float:
        return sum(v[name] for v in steps.values() if name in v and "step_s" in v)

    wall = total("step_s")
    rec = {"gbs": gbs, "run": label, "steps": len(steps)}
    rec.update({name: mean(name) for name in KEYS})
    rec["samples_per_s"] = total("samples") / wall if wall else math.nan
    rec["train_tok_per_s"] = total("train_tokens") / wall if wall else math.nan
    rec["model_call_tok_s"] = rec["gen_tokens"] / rec["model_s"]
    for phase in ("gen_s", "train_s", "logprob_s", "refit_s"):
        rec[phase.removesuffix("_s") + "_share"] = rec[phase] / rec["step_s"]
    rec["train_s_per_ksample"] = 1000 * rec["train_s"] / rec["samples"]
    return rec


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for arg in sys.argv[2:]:
        gbs, label, path = arg.split("=", 2)
        rows.append(summarize(int(gbs), label, Path(path)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: f"{v:.4g}" if isinstance(v, float) else v for k, v in r.items()}
            )


if __name__ == "__main__":
    main()
