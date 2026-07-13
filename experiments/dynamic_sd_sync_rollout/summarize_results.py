"""Aggregate DynamicSD profile/rollout results.json files into tidy CSVs.

Run identity (model, bench, mode, K, draft sampling, dynamic-or-not) is
recovered from each file's embedded config, not from filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

BENCH_BY_PROMPT_FILE = {
    "math500_prompts_full": "math500",
    "math_500_data_prompts_20260612": "math500",
    "openmath2_prompts_2048": "openmath",
    "dapo_math_prompts_2048": "dapo",
    "swebench_verified_prompts_all": "swe_verified",
    "swebench_full_test_prompts_all": "swe_full",
}

MODEL_LABELS = {
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "Qwen/Qwen3-235B-A22B": "Qwen3-235B-A22B",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "Nemotron3-Super-120B (FP8)",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "Nemotron3-Ultra-550B (NVFP4)",
}


def classify(config: dict[str, Any]) -> dict[str, Any]:
    spec = config.get("speculative_config_resolved") or {}
    prompt_stem = Path(config.get("prompt_jsonl") or "synthetic").stem
    model_name = str(config.get("model") or "")
    model_label = MODEL_LABELS.get(model_name, model_name)
    if int(config.get("max_model_len") or 0) >= 40960:
        model_label += " (40K)"

    if not spec:
        variant = "baseline"
    elif spec.get("num_speculative_tokens_per_batch_size"):
        variant = "dynamic"
    elif spec.get("method") == "suffix":
        variant = "suffix"
    else:
        variant = f"fixed_k{spec.get('num_speculative_tokens', '?')}"
    sample_method = spec.get("draft_sample_method", "greedy") if spec else "-"
    if sample_method == "probabilistic":
        variant += "_prob"

    return {
        "model": model_label,
        "bench": BENCH_BY_PROMPT_FILE.get(prompt_stem, prompt_stem),
        "variant": variant,
        "k": int(spec.get("num_speculative_tokens", 0)) if spec else 0,
        "sample_method": sample_method,
        "tp": int(config.get("tp") or 1),
        "tag": config.get("tag", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="results.json files or dirs")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        paths.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])

    profile_rows: list[dict[str, Any]] = []
    rollout_step_rows: list[dict[str, Any]] = []
    rollout_summary_rows: list[dict[str, Any]] = []
    drain_rows: list[dict[str, Any]] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        ident = classify(config)
        results = payload.get("results", [])
        if config.get("mode") == "profile":
            for row in results:
                profile_rows.append(
                    {
                        **ident,
                        "batch_size": row["batch_size"],
                        "output_tok_s": row["output_tok_s"],
                        "output_tok_s_per_gpu": row["output_tok_s"] / ident["tp"],
                        "wall_ms_per_output_token": row.get("wall_ms_per_output_token"),
                        "mean_acceptance_length": row.get("mean_acceptance_length"),
                        "acceptance_rate": row.get("acceptance_rate"),
                        "partial": payload.get("partial", False),
                    }
                )
            continue

        walls, toks = [], []
        for row in results:
            spec = row.get("spec_decode") or {}
            lengths = row.get("output_lengths") or {}
            rollout_step_rows.append(
                {
                    **ident,
                    "step": row["step"],
                    "num_sequences": row.get("num_sequences"),
                    "wall_s": row["wall_s"],
                    "output_tok_s": row["output_tok_s"],
                    "len_p50": lengths.get("p50"),
                    "len_p90": lengths.get("p90"),
                    "len_max": lengths.get("max"),
                    "total_tokens": lengths.get("total"),
                    "mean_acceptance_length": spec.get("mean_acceptance_length"),
                }
            )
            walls.append(row["wall_s"])
            toks.append(row["output_tok_s"])
            for req in row.get("request_timing", []):
                if "finished_s" in req:
                    drain_rows.append(
                        {
                            **ident,
                            "step": row["step"],
                            "finished_s": req["finished_s"],
                            "output_tokens": req.get("output_tokens"),
                        }
                    )
        if walls:
            rollout_summary_rows.append(
                {
                    **ident,
                    "num_steps": len(walls),
                    "mean_step_wall_s": statistics.fmean(walls),
                    "mean_output_tok_s": statistics.fmean(toks),
                    "mean_output_tok_s_per_gpu": statistics.fmean(toks) / ident["tp"],
                    "partial": payload.get("partial", False),
                }
            )

    write_csv(args.out_dir / "profile_grid.csv", profile_rows)
    write_csv(args.out_dir / "rollout_steps.csv", rollout_step_rows)
    write_csv(args.out_dir / "rollout_summary.csv", rollout_summary_rows)
    write_csv(args.out_dir / "drain_curves.csv", drain_rows)


if __name__ == "__main__":
    main()
