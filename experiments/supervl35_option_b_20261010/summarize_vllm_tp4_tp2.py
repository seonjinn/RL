"""Summarize matched SuperVL3.5 BF16 TP4 and MXFP8 TP2/DP2 vLLM runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
    "nemo-rl/supervl35-optionb-20261010"
)


def summarize_matched_runs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], concurrency: int
) -> dict[str, float | int]:
    if not pairs or concurrency < 2:
        raise ValueError("at least one pair and concurrency >= 2 are required")
    baseline_elapsed = 0.0
    target_elapsed = 0.0
    requests = 0
    output_tokens = 0
    for baseline, target in pairs:
        if len(baseline["runs"]) != 1 or len(target["runs"]) != 1:
            raise ValueError("one unprofiled trial is required per job")
        base_run = baseline["runs"][0]
        target_run = target["runs"][0]
        if base_run["requests"] != target_run["requests"]:
            raise ValueError("request counts differ between arms")
        if (
            baseline["request_multiplier"] != target["request_multiplier"]
            or base_run["requests"] != concurrency * baseline["request_multiplier"]
        ):
            raise ValueError("request multiplier does not match offered concurrency")
        if not baseline["enable_prefix_caching"] or not target["enable_prefix_caching"]:
            raise ValueError("prefix caching must be enabled in both arms")
        if (
            not baseline["enable_chunked_prefill"]
            or not target["enable_chunked_prefill"]
        ):
            raise ValueError("chunked prefill must be enabled in both arms")
        if (
            baseline["vllm_version"] != target["vllm_version"]
            or baseline["prompt_sha256"] != target["prompt_sha256"]
        ):
            raise ValueError("vLLM version or prompt differs between arms")
        for key in ("batch_size", "output_length", "prompt_tokens", "output_tokens"):
            if base_run[key] != target_run[key]:
                raise ValueError(f"{key} differs between arms")
        if base_run["batch_size"] != concurrency:
            raise ValueError("reported batch size differs from offered concurrency")
        if base_run["elapsed_s"] <= 0 or target_run["elapsed_s"] <= 0:
            raise ValueError("elapsed time must be positive")
        baseline_elapsed += base_run["elapsed_s"]
        target_elapsed += target_run["elapsed_s"]
        requests += base_run["requests"]
        output_tokens += base_run["output_tokens"]
    return {
        "pairs": len(pairs),
        "requests": requests,
        "output_tokens": output_tokens,
        "baseline_elapsed_s": baseline_elapsed,
        "target_elapsed_s": target_elapsed,
        "target_vs_baseline": baseline_elapsed / target_elapsed,
    }


def load_completed_result(path: Path) -> dict[str, Any] | None:
    exit_file = path.with_name("exit-code")
    if not exit_file.exists():
        return None
    exit_code = exit_file.read_text(encoding="utf-8").strip()
    if exit_code != "0":
        raise ValueError(f"{path.parent} failed with exit code {exit_code}")
    if not path.exists():
        raise ValueError(f"{path.parent} exited successfully but has no result JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("vllm_tp4_tp2_matrix_20261014.csv"),
    )
    args = parser.parse_args()
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    with args.manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups[(int(row["isl"]), int(row["osl"]), int(row["concurrency"]))].append(
                row
            )

    print(
        "| ISL/OSL | Offered C | Jobs | BF16 TP4 (s) | MXFP8 TP2/DP2 (s) | Throughput ratio |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for (isl, osl, concurrency), rows in sorted(groups.items()):
        pairs = []
        pending = False
        failure = None
        for row in rows:
            suffix = f"isl{isl}_osl{osl}"
            baseline_path = (
                args.root / "vllm_length_sweep" / suffix / row["bf16_job"] / "bf16.json"
            )
            target_path = (
                args.root
                / "vllm_dp2_length_sweep"
                / suffix
                / row["mxfp8_job"]
                / "moe.json"
            )
            try:
                baseline = load_completed_result(baseline_path)
                target = load_completed_result(target_path)
            except ValueError as error:
                failure = str(error)
                break
            if baseline is None or target is None:
                pending = True
                continue
            pairs.append((baseline, target))
        label = f"{isl // 1024}K/{osl // 1024}K"
        if failure:
            print(f"| {label} | {concurrency} | {len(rows)} | FAILED | {failure} | - |")
        elif pending or len(pairs) != len(rows):
            print(f"| {label} | {concurrency} | {len(rows)} | pending | pending | - |")
        else:
            summary = summarize_matched_runs(pairs, concurrency)
            print(
                f"| {label} | {concurrency} | {len(rows)} | "
                f"{summary['baseline_elapsed_s']:.2f} | "
                f"{summary['target_elapsed_s']:.2f} | "
                f"{summary['target_vs_baseline']:.3f}x |"
            )


if __name__ == "__main__":
    main()
