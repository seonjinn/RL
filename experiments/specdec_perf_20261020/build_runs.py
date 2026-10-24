"""Turn extract_metrics.py output into the per-run and per-step JSON of the dashboard.

    python build_runs.py <src_dir> <out_dir> [jobs.json]

<src_dir> holds summary.csv (run, run_id, state) and one W&B history CSV per run.
jobs.json maps a run name to its Slurm job id. Writes <out_dir>/runs.json and
<out_dir>/steps.json. report/data/runs.json merges the runs.json of each batch.
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
JOBS = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
WANDB = "https://wandb.ai/nvidia/sna-async-grpo-gb200/runs/"
FIRST, LAST = 3, 20

# Sync and async log the same phase under different keys; the first key present wins.
COLS = {
    "step_s": ("timing/train/total_step_time",),
    "gen_s": ("timing/train/generation", "timing/train/exposed_generation"),
    "train_s": ("timing/train/policy_training",),
    "logprob_s": ("timing/train/policy_and_reference_logprobs",),
    "refit_s": (
        "timing/train/prepare_for_generation/total",
        "timing/train/weight_sync",
    ),
    "tok_s_gpu": ("performance/tokens_per_sec_per_gpu",),
    "gen_tok_s_gpu": ("performance/generation_tokens_per_sec_per_gpu",),
    "train_tok_s_gpu": ("performance/policy_training_tokens_per_sec_per_gpu",),
    "logprob_tok_s_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    ),
    "num_tokens": ("train/total_num_tokens",),
    "accept_len": ("train/vllm/spec_acceptance_length",),
    "accept_rate": ("train/vllm/spec_acceptance_rate",),
    "gen_len": ("train/mean_gen_tokens_per_sample",),
    "reward": ("train/reward",),
    "gen_kl": ("train/gen_kl_error",),
    "draft_loss": ("train/draft_loss",),
    # Async only: mean wall time per trajectory on the generation workers, hidden or not.
    "rollout_s": ("train/trajectory_duration_s",),
    # SWE only: per-trajectory time inside model calls, and tokens generated in them.
    "model_call_s": ("train/swe_agents_train/total_model_call_time/mean",),
    "traj_gen_tok": ("train/swe_agents_train/gen_tokens_per_sample/mean",),
}
# vLLM spec-decode counters: accepted tokens, drafts, draft tokens.
SPEC = (
    "train/vllm/spec_num_accepted_tokens",
    "train/vllm/spec_num_drafts",
    "train/vllm/spec_num_draft_tokens",
)
STEP_COLS = (
    "step_s",
    "gen_s",
    "train_s",
    "logprob_s",
    "refit_s",
    "tok_s_gpu",
    "accept_len",
)

MODELS = {
    "q30": "Qwen3-30B-A3B",
    "n35": "Nemotron 3.5 Lightning 30B-A3B",
    "q35": "Qwen3.5-35B-A3B",
    "q235": "Qwen3-235B-A22B",
    "swe": "Qwen3-30B-A3B SWE",
    "swe235": "Qwen3-235B-A22B SWE",
}

# Each arm's control: the run that differs from it in exactly the axis the arm isolates
# (README "Arms"). Keys and values drop the "<model>-sd1020-" prefix.
REF = {
    "frozen-dflash-k7": "baseline",
    "frozen-dspark-k8": "baseline",
    "tp2-baseline": "baseline",
    "tp8-baseline-s64": "baseline",
    "online-dflash-k7": "tp2-baseline",
    "online-dspark-k8": "tp2-baseline",
    "async-baseline": "baseline",
    "async-baseline-mbt16k": "async-baseline",
    "async-baseline-s256-mbt16k": "async-baseline",
    "async-frozen-dflash-k7-s128": "async-baseline-mbt16k",
    "async-frozen-dspark-k8-s128": "async-baseline-mbt16k",
    "async-frozen-dflash-k7-s256": "async-baseline-s256-mbt16k",
    "async-frozen-dspark-k8-s256": "async-baseline-s256-mbt16k",
    "async-tp2-baseline-mbt16k": "async-baseline-mbt16k",
    "async-online-dflash-k7": "async-tp2-baseline-mbt16k",
    "async-online-dspark-k8": "async-tp2-baseline-mbt16k",
    "seq32k-frozen-dflash-k7": "seq32k-baseline",
    "seq32k-frozen-dspark-k8": "seq32k-baseline",
    "seq32k-async-baseline-mbt16k": "seq32k-baseline",
    "seq32k-async-frozen-dflash-k7-s128": "seq32k-async-baseline-mbt16k",
    "seq32k-async-frozen-dspark-k8-s128": "seq32k-async-baseline-mbt16k",
}


# Q235 SpecDec arms move generation to vLLM TP8; their control keeps that layout without SpecDec.
# N35 SpecDec arms turn prefix caching off (vLLM 0.29 hybrid Mamba limit); so do their controls.
REF_BY_MODEL = {
    "q235": {
        "frozen-dflash-k7": "tp8-baseline-s64",
        "frozen-dspark-k8": "tp8-baseline-s64",
    },
    "q35": {
        "baseline-nopc": "baseline",
        "frozen-dflash-k7": "baseline-nopc",
        "frozen-dspark-k8": "baseline-nopc",
        "async-baseline-mbt16k-nopc": "async-baseline",
        "async-frozen-dflash-k7-s128": "async-baseline-mbt16k-nopc",
        "async-frozen-dspark-k8-s128": "async-baseline-mbt16k-nopc",
        # gpu_memory_utilization 0.8 reruns: SpecDec holds 1+K GDN state blocks per request
        "baseline-nopc-gmu08": "baseline-nopc",
        "frozen-dflash-k7-gmu08": "baseline-nopc-gmu08",
        "frozen-dspark-k8-gmu08": "baseline-nopc-gmu08",
        "frozen-dflash-k3-gmu08": "baseline-nopc-gmu08",
        "frozen-dspark-k4-gmu08": "baseline-nopc-gmu08",
        # vLLM GDN metadata reuse backport: the baseline isolates the backport, SpecDec arms compare
        # against the baseline with the same backport
        "gdnreuse-baseline-nopc-gmu08": "baseline-nopc-gmu08",
        "gdnreuse-frozen-dflash-k3-gmu08": "gdnreuse-baseline-nopc-gmu08",
        "gdnreuse-frozen-dflash-k7-gmu08": "gdnreuse-baseline-nopc-gmu08",
        "gdnreuse-frozen-dspark-k4-gmu08": "gdnreuse-baseline-nopc-gmu08",
        "gdnreuse-frozen-dspark-k8-gmu08": "gdnreuse-baseline-nopc-gmu08",
    },
    "n35": {
        "baseline-nopc": "baseline",
        "frozen-dflash-k7": "baseline-nopc",
        "frozen-dspark-k8": "baseline-nopc",
        "async-baseline-mbt16k-nopc": "async-baseline-mbt16k",
        "async-frozen-dflash-k7-s128": "async-baseline-mbt16k-nopc",
        "async-frozen-dspark-k8-s128": "async-baseline-mbt16k-nopc",
    },
}


def num(x: str | None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def model_of(run: str) -> str:
    return re.split(r"-sd10\d\d-", run, maxsplit=1)[0]


# Cohorts submitted together with their own baselines (K sweep, DAPO dataset). Each is its own
# comparison group, and a frozen arm's control is the cohort baseline of the same mode.
COHORTS = {
    "ksweep": "ksweep",
    "dapo-seq32k": "dapo32k",
    "dapo40k": "dapo40k",
    "dapo": "dapo",
    "scale-r16": "scale",
    "scale2-r16": "scale",
    # Single-controller (V2) stack; its own baseline, so not compared against legacy async arms
    "sc-r16": "sc",
    "sc-r8": "sc",
}
# Follow-up cohorts fall back to an earlier cohort's control until their own has enough steps.
COHORT_CONTROL = {"scale2-r16": "scale-r16"}
# Steps an in-cohort control needs before it replaces the fallback (the page's MIN_PAIRED)
MIN_CONTROL_STEPS = 8
# Runs that stopped at step 0 and were resubmitted; the state names the replacement
SUPERSEDED = {
    "q235-sd1020-ksweep-frozen-dflash-k3": "superseded: r2 job 8150700",
    "q235-sd1020-ksweep-frozen-dspark-k4": "superseded: r2 job 8151041",
    "q30-sd1020-dapo-seq32k-async-frozen-dspark-k8-s128": "superseded: r2 job 8150978",
    "swe-sd1020-scale2-r16-online-dspark-k8": "superseded: lr1e4-tp4 run",
}
# SWE rollout-setting variants change the final-eval timeout, the sampler or the dataset of a
# cohort. A variant SpecDec arm keeps the variant prefix, so its control is the variant baseline.
# The variant baseline compares with the baseline it changes, which can run in an earlier batch.
VARIANT_BASELINE_REF = {
    "sc-r16-eval600-baseline": "sd1023-sc-r16-baseline",
    "sc-r16-readyfirst-baseline": "sd1023-sc-r16-baseline",
    "sc-r16-rf-eval450-drop4-baseline": "sd1023-sc-r16-readyfirst-baseline",
}
SPEC_ARM_PREFIX = re.compile(r"(.*?)(?:regen-)?(?:frozen|online)-")


def cohort_ref(model: str, cohort: str, core: str) -> str | None:
    if "frozen" not in core and "online" not in core:
        return None
    if core.startswith("async-"):
        base = "async-baseline-mbt16k"
    else:
        # Q235 sync spec arms run vLLM TP8 PP1 (DSpark rejects PP>1), so their control does too
        base = "tp8-baseline-s64" if model == "q235" else "baseline"
        if variant := SPEC_ARM_PREFIX.match(core):
            base = variant.group(1) + base
    return f"{cohort}-{base}" + ("-nopc" if model == "n35" else "")


def describe(run: str) -> dict:
    model, batch, arm = re.match(r"(.+?)-(sd10\d\d)-(.+)", run).groups()
    cohort = next((c for c in COHORTS if arm.startswith(f"{c}-")), None)
    core = arm.removeprefix(f"{cohort}-") if cohort else arm
    seq32k = core.startswith("seq32k-") or cohort == "dapo-seq32k"
    swe = model.startswith("swe")
    # SWE arms are async-1off without the "async-" tag in their names
    mode = "async" if "async-" in arm or swe else "sync"
    drafter = next((d for d in ("dflash", "dspark") if d in arm), "none")
    k = re.search(r"-k(\d+)", arm)
    seqs = re.search(r"-s(\d+)(?:-|$)", arm)
    # Online arms train at TP2, so they and their TP2 controls form their own comparison
    tp2 = "online" in arm or "tp2" in arm
    q30_group = "q30-32k" if seq32k else ("q30-online" if tp2 else f"q30-{mode}")
    group = (
        f"{model}-{COHORTS[cohort]}{'' if batch == 'sd1020' else '-' + batch[2:]}-{mode}"
        if cohort
        else (q30_group if model == "q30" else model)
    )
    # Resubmits carry a "-vN" checkout suffix; their control is the resubmit with the same suffix
    base_arm = re.sub(r"-v\d+$", "", arm)
    retry = arm[len(base_arm) :]
    ref = (
        cohort_ref(model, cohort, core)
        if cohort
        else REF_BY_MODEL.get(model, {}).get(base_arm, REF.get(base_arm))
    )
    if ref and retry and not cohort:
        ref += retry
    fallback = (
        ref.replace(cohort, COHORT_CONTROL[cohort], 1)
        if ref and cohort in COHORT_CONTROL
        else None
    )
    if arm in VARIANT_BASELINE_REF:
        ref_run = f"{model}-{VARIANT_BASELINE_REF[arm]}"
    else:
        ref_run = f"{model}-{batch}-{ref}" if ref else None
    return {
        "run": run,
        "group": group,
        "model": MODELS[model],
        "mode": mode,
        "seqlen": "32K"
        if seq32k
        else ("40K" if cohort == "dapo40k" else ("SWE" if swe else "default")),
        "arm": arm,
        "drafter": drafter,
        "drafter_mode": "online"
        if "online" in arm
        else ("frozen" if "frozen" in arm else "none"),
        "k": int(k.group(1)) if k else None,
        "seqs": int(seqs.group(1)) if seqs else None,
        "ref": ref_run,
        "ref_fallback": f"{model}-{batch}-{fallback}" if fallback else None,
    }


def pick(row: dict, keys: tuple[str, ...]) -> float | None:
    return next((v for v in (num(row.get(k)) for k in keys) if v is not None), None)


def mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b


def rel(a: float | None, b: float | None) -> float | None:
    r = ratio(a, b)
    return None if r is None else r - 1


# NeMo-RL logs each throughput per step as that step's tokens over its phase time.
RATE_COLS = ("tok_s_gpu", "gen_tok_s_gpu", "train_tok_s_gpu", "logprob_tok_s_gpu")


def agg(rows: list[dict], col: str) -> float | None:
    if col not in RATE_COLS:
        return mean([r[col] for r in rows])
    # Total tokens over total time. A plain mean of per-step rates lets short steps dominate:
    # async SWE alternates ~13 s buffered steps with 100-1000 s rollout steps.
    pts = [(r["num_tokens"], r[col]) for r in rows if r["num_tokens"] and r[col]]
    return sum(n for n, _ in pts) / sum(n / v for n, v in pts) if pts else None


def spec_window(rows: list[dict]) -> tuple[float, float | None] | None:
    # Ratio of summed counters. A step whose window held no drafts logs acceptance length 1.0,
    # which drags a plain mean down; async SWE has several such steps.
    pts = [r["spec"] for r in rows if None not in r["spec"] and r["spec"][1]]
    if not pts:
        return None
    acc, drafts, draft_tok = (sum(p[i] for p in pts) for i in range(3))
    return 1 + acc / drafts, acc / draft_tok if draft_tok else None


def traj_tok_s(rows: list[dict]) -> float | None:
    # Token-weighted: total tokens over total model-call time, not a mean of per-step rates.
    return ratio(
        mean([r["traj_gen_tok"] for r in rows]), mean([r["model_call_s"] for r in rows])
    )


# Each column compared on the steps both runs logged, and the key its delta is stored under.
PAIR_COLS = {
    "step_s": "pair_step_delta",
    "gen_s": "pair_gen_delta",
    "logprob_s": "pair_logprob_delta",
    "train_s": "pair_train_delta",
    "refit_s": "pair_refit_delta",
    "tok_s_gpu": "pair_tput_delta",
    "gen_tok_s_gpu": "pair_gen_tok_delta",
    "logprob_tok_s_gpu": "pair_logprob_tok_delta",
    "train_tok_s_gpu": "pair_train_tok_delta",
    "rollout_s": "pair_rollout_delta",
}


# A rate column inherits the noise of the phase timer it divides by.
PHASE_OF = {
    "gen_tok_s_gpu": "gen_s",
    "logprob_tok_s_gpu": "logprob_s",
    "train_tok_s_gpu": "train_s",
}
# SWE skips the logprob pass and its timer logs about 1 ms; a ratio of two idle timers is noise.
MIN_PHASE_S = 0.5


def pair_deltas(arm: dict[int, dict], ref: dict[int, dict]) -> dict:
    # A run that hits the wall clock early would otherwise be compared on fewer, earlier steps.
    common = sorted(arm.keys() & ref.keys())
    out: dict = {"pair_steps": len(common)}
    ref_rows = [ref[s] for s in common]
    for col, key in PAIR_COLS.items():
        phase = agg(ref_rows, PHASE_OF.get(col, col)) if col != "tok_s_gpu" else None
        if phase is not None and phase < MIN_PHASE_S:
            out[key] = None
            continue
        out[key] = rel(agg([arm[s] for s in common], col), agg(ref_rows, col))
    out["pair_traj_delta"] = rel(
        traj_tok_s([arm[s] for s in common]), traj_tok_s([ref[s] for s in common])
    )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {r["run"]: r for r in csv.DictReader(open(SRC / "summary.csv"))}
    runs, steps = [], []
    windows: dict[str, dict[int, dict]] = {}
    for name in sorted(summary):
        if name.endswith("-smoke"):
            continue
        s = summary[name]
        per_run = SRC / f"{name}.csv"
        rows = []
        for r in csv.DictReader(open(per_run)) if per_run.exists() else []:
            step = num(r.get("nemo_rl/step"))
            if step is None or step < 1 or pick(r, COLS["step_s"]) is None:
                continue
            row = {"step": int(step)} | {k: pick(r, keys) for k, keys in COLS.items()}
            row["spec"] = tuple(num(r.get(k)) for k in SPEC)
            if row["spec"][1] == 0:
                row["accept_len"] = row["accept_rate"] = None
            # SWE skips the logprob pass (force_on_policy_ratio, KL penalty 0). The legacy stack
            # still logs a ~1 ms timer, the V2 stack logs none.
            if row["logprob_s"] is None and model_of(name).startswith("swe"):
                row["logprob_s"] = 0.0
            rows.append(row)
        # Async SWE buffers whole rollouts, so a 1000 s rollout lands on step 2 in one arm and is
        # split over steps 2-3 in another. Any cut after step 1 drops a different share of rollout
        # per arm; from step 1 the window sum is the wall time to finish N steps.
        first = 1 if model_of(name).startswith("swe") else FIRST
        window = [r for r in rows if first <= r["step"] <= LAST]
        windows[name] = {r["step"]: r for r in window}
        row = describe(name)
        row.update(
            job=JOBS.get(name),
            wandb_id=s["run_id"],
            wandb_url=WANDB + s["run_id"],
            state=SUPERSEDED.get(name, s["state"]),
            window_steps=len(window),
            max_step=max((r["step"] for r in rows), default=0),
        )
        row.update({k: agg(window, k) for k in COLS})
        row["traj_tok_s"] = traj_tok_s(window)
        if spec := spec_window(window):
            row["accept_len"], row["accept_rate"] = spec
        if row["group"] == "swe":
            # Runs before 8f1959fe65 read the vLLM counter delta only from batch arrival to
            # metrics time, which misses most of a SWE rollout. Later SWE runs tile the windows.
            row["accept_len"] = row["accept_rate"] = None
        runs.append(row)
        steps += [
            {"run": name, "step": r["step"], "in_window": first <= r["step"] <= LAST}
            | {k: r[k] for k in STEP_COLS}
            for r in rows
        ]
    for r in runs:
        fallback = r.pop("ref_fallback")
        if len(windows.get(r["ref"], {})) < MIN_CONTROL_STEPS and fallback in windows:
            r["ref"] = fallback
        if r["ref"] not in windows:
            r["ref"] = None
        else:
            r.update(pair_deltas(windows[r["run"]], windows[r["ref"]]))
    (OUT / "runs.json").write_text(json.dumps(runs, indent=1))
    (OUT / "steps.json").write_text(json.dumps(steps))
    print(len(runs), "runs,", len(steps), "step rows")


if __name__ == "__main__":
    main()
