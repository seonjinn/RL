#!/usr/bin/env python3
"""Analyze trained-draft Eagle3 num_speculative_tokens sweep logs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from analyze_specdec_smoke import analyze, gate_result, summarize_result


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_FILE = ROOT / "latest_trained_draft_spec_tokens_sweep_jobs.txt"
DEFAULT_CLUSTER_REPO = Path(
    os.environ.get(
        "SMOKE_REPO_ROOT",
        os.environ.get(
            "NEMO_RL_REPO_ROOT",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL",
        ),
    )
)
SPEC_JOB_RE = re.compile(r"specdec_tokens_(\d+)_job")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, default=DEFAULT_JOB_FILE)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_CLUSTER_REPO)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument(
        "--specdec-log",
        action="append",
        default=[],
        metavar="TOKENS=PATH",
        help="Explicit specdec log path for a token setting, e.g. 3=/path/ray-driver.log.",
    )
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--drop-first-step", action="store_true")
    parser.add_argument("--gen-outlier-threshold-s", type=float, default=800.0)
    parser.add_argument("--min-generation-speedup-pct", type=float, default=10.0)
    parser.add_argument("--min-acceptance-rate", type=float, default=0.45)
    parser.add_argument("--fail-on-missing-spec-metrics", action="store_true")
    parser.add_argument("--fail-if-no-pass", action="store_true")
    return parser.parse_args()


def parse_job_file(path: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    if not path.exists():
        return jobs
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        jobs[key.strip()] = value.strip()
    return jobs


def job_log_path(repo_root: Path, job_id: str) -> Path:
    if not job_id or not job_id.isdigit():
        raise ValueError(f"invalid Slurm job id: {job_id!r}")
    ray_driver = repo_root / f"{job_id}-logs" / "ray-driver.log"
    if ray_driver.exists():
        return ray_driver
    return repo_root / f"{job_id}-logs"


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} path is not visible: {path}")
    return path


def parse_explicit_spec_logs(items: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--specdec-log must be TOKENS=PATH, got {item!r}")
        token_text, path_text = item.split("=", 1)
        tokens = int(token_text)
        parsed[tokens] = Path(path_text)
    return parsed


def metric(summary: dict[str, Any], key: str, section: str = "timing") -> float | None:
    item = summary.get(section, {}).get(key, {})
    value = item.get("median")
    return float(value) if value is not None else None


def acceptance(summary: dict[str, Any]) -> float | None:
    value = metric(summary, "acceptance_rate", "spec_metrics")
    if value is None:
        value = metric(summary, "derived_acceptance_rate", "spec_metrics")
    return value


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.2f}{suffix}"


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def analyze_one(
    path: Path,
    args: argparse.Namespace,
    baseline_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = analyze([path])
    summary = summarize_result(result, args.drop_first_step, args.gen_outlier_threshold_s)
    gate = gate_result(
        summary,
        baseline_summary,
        args.min_generation_speedup_pct,
        args.min_acceptance_rate,
        args.fail_on_missing_spec_metrics,
    )
    return summary, gate


def choose_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [row for row in rows if row["gate_status"] == "pass"]
    candidates = passed or rows
    candidates = [row for row in candidates if row.get("exposed_generation_median_s") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["exposed_generation_median_s"], row["tokens"]))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trained Eagle3 Spec-Token Sweep",
        "",
        f"Overall: **{payload['overall_status'].upper()}**",
        "",
    ]
    recommendation = payload.get("recommendation")
    if recommendation:
        lines.append(
            "Recommendation: "
            f"**{recommendation['tokens']} speculative tokens** "
            f"(gate={recommendation['gate_status']}, "
            f"exposed_generation={fmt(recommendation.get('exposed_generation_median_s'), 's')})."
        )
        lines.append("")
    lines += [
        "| tokens | gate | exposed gen median | gen speedup | total step median | acceptance | warnings | log |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['tokens']} | {row['gate_status'].upper()} | "
            f"{fmt(row.get('exposed_generation_median_s'), 's')} | "
            f"{pct(row.get('generation_speedup_pct'))} | "
            f"{fmt(row.get('total_step_median_s'), 's')} | "
            f"{fmt(row.get('acceptance_rate'))} | "
            f"{row.get('warning_count', 0)} | {row['log']} |"
        )
    if payload.get("baseline"):
        baseline = payload["baseline"]
        lines += [
            "",
            "Baseline:",
            f"- log: {baseline['log']}",
            f"- exposed_generation median: {fmt(baseline.get('exposed_generation_median_s'), 's')}",
            f"- total_step_time median: {fmt(baseline.get('total_step_median_s'), 's')}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    jobs = parse_job_file(args.job_file)
    vllm_draft_dir = jobs.get("vllm_draft_dir")
    artifact_root = jobs.get("artifact_root")
    repo_root_recorded = jobs.get("repo_root")
    swe_repo_root = jobs.get("swe_repo_root")
    config_file = jobs.get("config_file")
    env_file = jobs.get("env_file")
    chat_template = jobs.get("chat_template")
    max_num_steps = jobs.get("max_num_steps")
    spec_tokens_list = jobs.get("spec_tokens_list")
    eagle3_draft_tp = jobs.get("eagle3_draft_tp")

    explicit = parse_explicit_spec_logs(args.specdec_log)
    baseline_log = args.baseline_log
    if baseline_log is None and jobs.get("baseline_job"):
        baseline_log = job_log_path(args.repo_root, jobs["baseline_job"])
    if baseline_log is None:
        raise SystemExit("baseline log is required via --baseline-log or baseline_job in --job-file")
    baseline_log = existing(baseline_log, "baseline")

    spec_logs: dict[int, Path] = dict(explicit)
    for key, value in jobs.items():
        match = SPEC_JOB_RE.fullmatch(key)
        if not match or int(match.group(1)) in spec_logs:
            continue
        try:
            spec_logs[int(match.group(1))] = job_log_path(args.repo_root, value)
        except ValueError:
            continue
    if not spec_logs:
        raise SystemExit("no specdec sweep logs found via --specdec-log or job file")

    baseline_summary, baseline_gate = analyze_one(baseline_log, args, None)
    baseline_gen = metric(baseline_summary, "exposed_generation")
    baseline_total = metric(baseline_summary, "total_step_time")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for tokens, path in sorted(spec_logs.items()):
        path = existing(path, f"specdec {tokens}")
        summary, gate = analyze_one(path, args, baseline_summary)
        current_gen = metric(summary, "exposed_generation")
        speedup = None
        if baseline_gen and current_gen is not None:
            speedup = (1.0 - current_gen / baseline_gen) * 100.0
        row = {
            "tokens": tokens,
            "log": str(path),
            "gate_status": gate["status"],
            "exposed_generation_median_s": current_gen,
            "generation_speedup_pct": speedup,
            "total_step_median_s": metric(summary, "total_step_time"),
            "acceptance_rate": acceptance(summary),
            "warning_count": len(summary.get("warnings", [])),
        }
        rows.append(row)
        details[str(tokens)] = {"summary": summary, "gate": gate}

    recommendation = choose_recommendation(rows)
    overall = "pass" if any(row["gate_status"] == "pass" for row in rows) else "fail"
    payload = {
        "overall_status": overall,
        "job_file": str(args.job_file),
        "repo_root": str(args.repo_root),
        "vllm_draft_dir": vllm_draft_dir,
        "execution_context": {
            "artifact_root": artifact_root,
            "repo_root": repo_root_recorded,
            "swe_repo_root": swe_repo_root,
            "config_file": config_file,
            "env_file": env_file,
            "chat_template": chat_template,
        },
        "max_num_steps": max_num_steps,
        "spec_tokens_list": spec_tokens_list,
        "eagle3_draft_tp": eagle3_draft_tp,
        "baseline": {
            "log": str(baseline_log),
            "gate": baseline_gate,
            "exposed_generation_median_s": baseline_gen,
            "total_step_median_s": baseline_total,
            "summary": baseline_summary,
        },
        "rows": rows,
        "recommendation": recommendation,
        "details": details,
    }

    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.fail_if_no_pass and overall != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
