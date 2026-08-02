from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict


class RunSummary(TypedDict):
    arm: str | None
    complete: bool
    elapsed_seconds: float | None
    generation_calls: int | None
    generation_seconds: float | None
    gpu_count: int
    measurement_scope: str
    model_load_seconds: float | None
    output_tokens: int | None
    tokens_per_second: float | None
    tokens_per_second_per_gpu: float | None


class SummaryReport(TypedDict):
    runs: list[RunSummary]
    adaptive_vs_baseline_speedup: float | None


_MARKER = re.compile(r"^NEMORL_CANARY\s+(?P<fields>.+)$")


def _fields(line: str) -> dict[str, str]:
    match = _MARKER.match(line.strip())
    if match is None:
        return {}
    parsed: dict[str, str] = {}
    for item in match.group("fields").split():
        key, separator, value = item.partition("=")
        if separator:
            parsed[key] = value
    return parsed


def _epoch(fields: dict[str, str]) -> float | None:
    value = fields.get("epoch")
    if value is None:
        return None
    try:
        epoch = float(value)
    except ValueError:
        return None
    return epoch if math.isfinite(epoch) else None


def _token_count(fields: dict[str, str]) -> int | None:
    value = fields.get("tokens")
    if value is None:
        return None
    try:
        tokens = int(value)
    except ValueError:
        return None
    return tokens if tokens >= 0 else None


def _nonnegative_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _positive_count(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _duration(end: float | None, start: float | None) -> float | None:
    if end is None or start is None or end < start:
        return None
    return end - start


def summarize_log(path: Path, *, gpu_count: int = 8) -> RunSummary:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")

    arm: str | None = None
    start: float | None = None
    model_ready: float | None = None
    complete: float | None = None
    output_tokens: int | None = None
    direct_generation_seconds: float | None = None
    generation_calls: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = _fields(line)
        if not fields:
            continue
        arm = fields.get("arm", arm)
        event = fields.get("event")
        if event == "start":
            parsed_epoch = _epoch(fields)
            start = parsed_epoch if parsed_epoch is not None else start
        elif event == "model_ready":
            parsed_epoch = _epoch(fields)
            model_ready = parsed_epoch if parsed_epoch is not None else model_ready
        elif event == "outputs":
            parsed_tokens = _token_count(fields)
            output_tokens = (
                parsed_tokens if parsed_tokens is not None else output_tokens
            )
        elif event == "generation":
            parsed_seconds = _nonnegative_float(fields.get("seconds"))
            parsed_calls = _positive_count(fields.get("calls"))
            if parsed_seconds is not None and parsed_calls is not None:
                direct_generation_seconds = parsed_seconds
                generation_calls = parsed_calls
        elif event == "complete":
            parsed_epoch = _epoch(fields)
            complete = parsed_epoch if parsed_epoch is not None else complete

    elapsed_seconds = _duration(complete, start)
    model_load_seconds = _duration(model_ready, start)
    rollout_eval_seconds = _duration(complete, model_ready)
    if direct_generation_seconds is None:
        generation_seconds = rollout_eval_seconds
        measurement_scope = "rollout_eval_wall"
    else:
        generation_seconds = direct_generation_seconds
        measurement_scope = "generation_calls"
    is_complete = elapsed_seconds is not None
    tokens_per_second: float | None = None
    if (
        is_complete
        and model_load_seconds is not None
        and generation_seconds is not None
        and generation_seconds > 0
        and output_tokens is not None
    ):
        tokens_per_second = output_tokens / generation_seconds

    return {
        "arm": arm,
        "complete": is_complete,
        "elapsed_seconds": elapsed_seconds,
        "generation_calls": generation_calls,
        "generation_seconds": generation_seconds,
        "gpu_count": gpu_count,
        "measurement_scope": measurement_scope,
        "model_load_seconds": model_load_seconds,
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": (
            tokens_per_second / gpu_count if tokens_per_second is not None else None
        ),
    }


def summarize_logs(paths: Sequence[Path], *, gpu_count: int = 8) -> SummaryReport:
    runs = [summarize_log(path, gpu_count=gpu_count) for path in paths]
    speedup: float | None = None
    if len(runs) == 2:
        runs_by_arm = {run["arm"]: run for run in runs}
        if set(runs_by_arm) == {"adaptive", "baseline"}:
            adaptive_throughput = runs_by_arm["adaptive"]["tokens_per_second"]
            baseline_throughput = runs_by_arm["baseline"]["tokens_per_second"]
            if (
                adaptive_throughput is not None
                and baseline_throughput is not None
                and baseline_throughput > 0
            ):
                speedup = adaptive_throughput / baseline_throughput
    return {"runs": runs, "adaptive_vs_baseline_speedup": speedup}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--gpu-count", type=_positive_int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize_logs(args.logs, gpu_count=args.gpu_count)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
