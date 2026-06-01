#!/usr/bin/env python3
"""Summarize NeMo-RL static SpecDec smoke-test logs.

The smoke test has two jobs:
1. prove that vLLM/NeMo-RL can load an Eagle3 draft checkpoint; and
2. decide whether generation time improves enough to justify training a
   Thinking-2507/SWE-specific draft.

This parser is intentionally tolerant. NeMo-RL and vLLM log formats vary across
versions, and speculative decoding metrics are not always printed with the same
names. The script extracts the stable NeMo-RL stage timings first, then gathers
any log lines that look like speculative/draft/acceptance metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


LOG_SUFFIXES = {".log", ".out", ".txt"}
CSV_SUFFIXES = {".csv"}
DEFAULT_TIMING_KEYS = (
    "exposed_generation",
    "policy_training",
    "policy_and_reference_logprobs",
    "weight_sync",
    "total_step_time",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_RE = re.compile(
    r"(?:training_step\s*=\s*|(?:^|=+\s*)Step\s+)(\d+)(?:/\d+)?",
    re.IGNORECASE,
)
FILE_STEP_RE = re.compile(r"(?:^|[_-])step[_-]?(\d+)(?:\D|$)", re.IGNORECASE)
TOTAL_RE = re.compile(r"Total step time:\s*([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
TIMING_RE = re.compile(
    r"(?:^|\s)[*-]?\s*(?:[•-]\s*)?"
    r"([A-Za-z0-9_./-]+):\s*([0-9]+(?:\.[0-9]+)?)s"
    r"(?:\s*\(([0-9]+(?:\.[0-9]+)?)%\))?",
)
THROUGHPUT_RE = re.compile(
    r"(?:^|\s)[*-]?\s*(?:[•-]\s*)?"
    r"([A-Za-z0-9 /()_-]*Tokens/sec(?:/gpu)?[A-Za-z0-9 /()_-]*):\s*"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
KEY_VALUE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_:./-]*)\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)%?",
    re.IGNORECASE,
)
PROM_METRIC_RE = re.compile(
    r"([A-Za-z_:][A-Za-z0-9_:./-]*)(?:\{[^}]*\})?\s+(-?[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
ACCEPTANCE_RATE_TEXT_RE = re.compile(
    r"(?:avg\s+draft\s+acceptance\s+rate|draft\s+acceptance\s+rate|"
    r"acceptance\s+rate)[^0-9-]*"
    r"(-?[0-9]+(?:\.[0-9]+)?)\s*(%)?",
    re.IGNORECASE,
)
ENV_ASSIGN_RE = re.compile(
    r"(?:^|[\s'\"{,])([A-Z][A-Z0-9_]+)=([^\\\s,'\"\]}]+)"
)
HYDRA_ASSIGN_RE = re.compile(
    r"(?:^|[\s'\",])(\+{0,2}[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)=([^\\\s,'\"\]}]+)"
)
LOGPROB_CONFIG_RE = re.compile(
    r"(spec_decode_requested|omit_generation_logprobs|"
    r"force_specdec_request_logprobs|request_logprobs)=([A-Za-z0-9_.-]+)"
)
CONFIG_KEY_ALIASES = {
    "policy.generation.vllm_kwargs.max_num_seqs": "VLLM_MAX_NUM_SEQS",
    "policy.generation.vllm_kwargs.max_num_batched_tokens": "VLLM_MAX_NUM_BATCHED_TOKENS",
}
RUN_CONFIG_KEYS = {
    "NRL_VLLM_OMIT_GENERATION_LOGPROBS",
    "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_ATTENTION_BACKEND",
    "grpo.num_prompts_per_step",
    "grpo.num_generations_per_prompt",
    "policy.train_global_batch_size",
    "policy.generation.max_new_tokens",
    "policy.generation.vllm_cfg.enforce_eager",
}
BASELINE_EXACT_MATCH_KEYS = (
    "grpo.num_prompts_per_step",
    "grpo.num_generations_per_prompt",
    "policy.train_global_batch_size",
    "policy.generation.max_new_tokens",
    "policy.generation.vllm_cfg.enforce_eager",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
)


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "")


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.2f}{suffix}"


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def normalize_spec_metric(name: str) -> str:
    normalized = normalize_name(name)
    if "reliable" in normalized:
        return normalized
    if "accept" in normalized and "rate" in normalized and (
        "per_position" in normalized or re.search(r"(?:^|_)pos_?\d+", normalized)
    ):
        pos_match = re.search(r"(?:^|_)pos_?(\d+)", normalized)
        if pos_match:
            return f"acceptance_rate_pos_{pos_match.group(1)}"
        return "acceptance_rate_per_pos"
    if "metric" in normalized and "available" in normalized:
        return "metrics_available"
    if "accept" in normalized and "rate" in normalized:
        return "acceptance_rate"
    if "accepted" in normalized and "token" in normalized:
        return "accepted_tokens"
    if "accept" in normalized and "token" in normalized:
        return "accepted_tokens"
    if "draft" in normalized and "token" in normalized:
        return "draft_tokens"
    if "draft" in normalized and "accept" in normalized:
        return "acceptance_rate"
    return normalized


def likely_named_metric(name: str) -> bool:
    # Avoid treating file paths like retro_decoder_spec.py:39 as metrics.
    if "\\" in name or ".py" in name or len(name) > 96:
        return False
    if "/" in name and not name.startswith(("spec_decode/", "spec_decode_gate/", "vllm/")):
        return False
    return True


def as_rate(name: str, value: float, source_text: str) -> float:
    if "rate" not in name and "acceptance" not in name:
        return value
    if "%" in source_text or value > 1.0:
        return value / 100.0
    return value


def maybe_step_from_filename(path: Path) -> int | None:
    match = FILE_STEP_RE.search(path.stem)
    return int(match.group(1)) if match else None


@dataclass
class StepRecord:
    step: int
    source: str
    timings: dict[str, float] = field(default_factory=dict)
    timing_pct: dict[str, float] = field(default_factory=dict)


@dataclass
class ParseResult:
    paths: list[str]
    files: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    spec_metrics: dict[str, list[float]] = field(default_factory=dict)
    throughput_metrics: dict[str, list[float]] = field(default_factory=dict)
    spec_lines: list[str] = field(default_factory=list)
    spec_metric_sources: list[str] = field(default_factory=list)
    timing_sources: list[str] = field(default_factory=list)
    standalone_vllm_spec_sources: list[str] = field(default_factory=list)
    standalone_vllm_throughput_sources: list[str] = field(default_factory=list)
    mixed_context_sources: list[str] = field(default_factory=list)
    evidence: dict[str, bool] = field(
        default_factory=lambda: {
            "speculative_config_seen": False,
            "eagle3_seen": False,
            "vllm_seen": False,
        }
    )
    run_config: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def add_metric(bucket: dict[str, list[float]], key: str, value: float) -> None:
    bucket.setdefault(key, []).append(value)


def add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def clean_config_value(value: str) -> str:
    return value.strip().strip("\\").strip(",").strip("'\"")


def canonical_config_key(key: str) -> str:
    cleaned = key.lstrip("+")
    return CONFIG_KEY_ALIASES.get(cleaned, cleaned)


def add_config_value(result: ParseResult, key: str, value: str) -> None:
    value = clean_config_value(value)
    if not value:
        return
    bucket = result.run_config.setdefault(key, [])
    if value not in bucket:
        bucket.append(value)


def parse_run_config(line: str, result: ParseResult) -> None:
    for match in ENV_ASSIGN_RE.finditer(line):
        key, value = match.groups()
        if key in RUN_CONFIG_KEYS:
            add_config_value(result, key, value)

    for match in HYDRA_ASSIGN_RE.finditer(line):
        key, value = match.groups()
        key = canonical_config_key(key)
        if key in RUN_CONFIG_KEYS:
            add_config_value(result, key, value)

    lower = line.lower()
    if "generation sampling logprob config" in lower or "specdec effective config" in lower:
        for key, value in LOGPROB_CONFIG_RE.findall(line):
            if key == "request_logprobs":
                key = "effective_generation_request_logprobs"
            else:
                key = f"effective_{key}"
            add_config_value(result, key, value)


def collect_paths(paths: Iterable[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in LOG_SUFFIXES | CSV_SUFFIXES:
                    collected.append(child)
        elif path.is_file():
            collected.append(path)
    return collected


def record_for_step(
    records: dict[tuple[str, int], StepRecord], path: Path, step: int
) -> StepRecord:
    key = (str(path), step)
    if key not in records:
        records[key] = StepRecord(step=step, source=str(path))
    return records[key]


def parse_csv(path: Path, result: ParseResult, records: dict[tuple[str, int], StepRecord]) -> None:
    try:
        with path.open(newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            if "step" not in fields:
                return
            timing_map = {
                "total_s": "total_step_time",
                "exposed_gen_s": "exposed_generation",
                "training_s": "policy_training",
                "logprobs_s": "policy_and_reference_logprobs",
                "weight_sync_s": "weight_sync",
                "ckpt_s": "checkpoint",
            }
            if not fields.intersection(timing_map):
                return
            result.files.append(str(path))
            for row in reader:
                if not row.get("step"):
                    continue
                step = int(float(row["step"]))
                record = record_for_step(records, path, step)
                for source_key, target_key in timing_map.items():
                    value = row.get(source_key)
                    if value in (None, ""):
                        continue
                    record.timings[target_key] = float(value)
                    add_unique(result.timing_sources, str(path))
    except (OSError, ValueError, csv.Error) as exc:
        result.warnings.append(f"Could not parse CSV {path}: {exc}")


def is_standalone_vllm_line(lower: str) -> bool:
    return any(
        token in lower
        for token in (
            "launch_vllm.py",
            "standalone vllm",
            "standalone_vllm",
            "vllm standalone",
            "vllm serve",
            "vllm.entrypoints.openai.api_server",
        )
    )


def is_standalone_vllm_server_line(lower: str) -> bool:
    return any(
        token in lower
        for token in (
            "application startup complete",
            "uvicorn running",
            "/health",
            "openai api server",
        )
    )


def is_nemo_rl_line(lower: str) -> bool:
    return any(
        token in lower
        for token in (
            "vllmgenerationworker",
            "generation worker group",
            "nemo_rl",
            "total step time",
            "========================= step ",
        )
    )


def parse_spec_metrics(
    line: str,
    result: ParseResult,
    *,
    source: Path | None = None,
    standalone_context: bool = False,
    nemo_context: bool = False,
) -> None:
    lower = line.lower()
    if "vllm" in lower:
        result.evidence["vllm_seen"] = True
    if "speculative_config" in lower or "speculative decoding" in lower:
        result.evidence["speculative_config_seen"] = True
    if "eagle3" in lower:
        result.evidence["eagle3_seen"] = True
    if not any(token in lower for token in ("spec", "draft", "accept", "eagle3")):
        return

    captured = False
    seen_values: set[tuple[str, float]] = set()
    for match in KEY_VALUE_RE.finditer(line):
        raw_name, raw_value = match.groups()
        if not likely_named_metric(raw_name):
            continue
        if not any(token in raw_name.lower() for token in ("spec", "draft", "accept", "eagle")):
            continue
        metric_name = normalize_spec_metric(raw_name)
        value = as_rate(metric_name, float(raw_value), match.group(0))
        seen_values.add((metric_name, value))
        add_metric(result.spec_metrics, metric_name, value)
        captured = True

    for match in PROM_METRIC_RE.finditer(line):
        raw_name, raw_value = match.groups()
        if not likely_named_metric(raw_name):
            continue
        if not any(token in raw_name.lower() for token in ("spec", "draft", "accept", "eagle")):
            continue
        metric_name = normalize_spec_metric(raw_name)
        value = as_rate(metric_name, float(raw_value), match.group(0))
        if (metric_name, value) in seen_values:
            continue
        seen_values.add((metric_name, value))
        add_metric(result.spec_metrics, metric_name, value)
        captured = True

    if "accept" in lower and "rate" in lower and "per-position" not in lower:
        match = ACCEPTANCE_RATE_TEXT_RE.search(line)
        if match:
            value = float(match.group(1))
            if match.group(2) or value > 1.0:
                value /= 100.0
            if ("acceptance_rate", value) not in seen_values:
                add_metric(result.spec_metrics, "acceptance_rate", value)
                captured = True

    if captured:
        if source is not None:
            add_unique(result.spec_metric_sources, str(source))
            if standalone_context and not nemo_context:
                add_unique(result.standalone_vllm_spec_sources, str(source))
        if len(result.spec_lines) < 20:
            result.spec_lines.append(line.strip())
    elif any(token in lower for token in ("speculative", "draft", "eagle3")):
        if len(result.spec_lines) < 20:
            result.spec_lines.append(line.strip())


def parse_text_log(path: Path, result: ParseResult, records: dict[tuple[str, int], StepRecord]) -> None:
    result.files.append(str(path))
    fallback_step = maybe_step_from_filename(path)
    current_step = fallback_step
    synthetic_step = 0
    standalone_context = False
    nemo_context = False
    file_saw_strong_standalone = False
    file_saw_weak_standalone = False
    file_saw_nemo = False

    try:
        with path.open(errors="replace") as fh:
            for raw_line in fh:
                line = clean_line(raw_line)
                if not line.strip():
                    continue
                lower = line.lower()
                parse_run_config(line, result)
                line_is_strong_standalone = is_standalone_vllm_line(lower)
                line_is_weak_standalone = is_standalone_vllm_server_line(lower)
                line_is_nemo = is_nemo_rl_line(lower)
                line_is_standalone = line_is_strong_standalone or (
                    line_is_weak_standalone and not file_saw_nemo
                )
                file_saw_strong_standalone = (
                    file_saw_strong_standalone or line_is_strong_standalone
                )
                file_saw_weak_standalone = (
                    file_saw_weak_standalone or line_is_weak_standalone
                )
                file_saw_nemo = file_saw_nemo or line_is_nemo
                standalone_context = standalone_context or line_is_standalone
                nemo_context = nemo_context or line_is_nemo

                step_match = STEP_RE.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                    nemo_context = True
                    file_saw_nemo = True

                parse_spec_metrics(
                    line,
                    result,
                    source=path,
                    standalone_context=standalone_context,
                    nemo_context=nemo_context,
                )

                throughput_match = THROUGHPUT_RE.search(line)
                if throughput_match:
                    key = normalize_name(throughput_match.group(1))
                    add_metric(result.throughput_metrics, key, float(throughput_match.group(2)))
                    if standalone_context and not nemo_context:
                        add_unique(
                            result.standalone_vllm_throughput_sources, str(path)
                        )

                total_match = TOTAL_RE.search(line)
                if total_match:
                    if current_step is None:
                        current_step = fallback_step if fallback_step is not None else synthetic_step
                        synthetic_step += 1
                    record = record_for_step(records, path, current_step)
                    record.timings["total_step_time"] = float(total_match.group(1))
                    add_unique(result.timing_sources, str(path))
                    continue

                timing_match = TIMING_RE.search(line)
                if timing_match:
                    raw_key, raw_value, pct = timing_match.groups()
                    key = normalize_name(raw_key)
                    if key == "total_step_time":
                        continue
                    if key.endswith("_s"):
                        key = key[:-2]
                    if current_step is None:
                        current_step = fallback_step if fallback_step is not None else synthetic_step
                        synthetic_step += 1
                    record = record_for_step(records, path, current_step)
                    record.timings[key] = float(raw_value)
                    add_unique(result.timing_sources, str(path))
                    if pct is not None:
                        record.timing_pct[key] = float(pct)
    except OSError as exc:
        result.warnings.append(f"Could not read {path}: {exc}")
    if file_saw_strong_standalone and file_saw_nemo:
        add_unique(result.mixed_context_sources, str(path))
    elif file_saw_weak_standalone and not file_saw_nemo:
        add_unique(result.standalone_vllm_throughput_sources, str(path))


def analyze(paths: list[Path]) -> ParseResult:
    result = ParseResult(paths=[str(p) for p in paths])
    records: dict[tuple[str, int], StepRecord] = {}
    files = collect_paths(paths)
    if not files:
        result.warnings.append("No readable log, out, txt, or csv files found.")
        return result

    for path in files:
        suffix = path.suffix.lower()
        if suffix in CSV_SUFFIXES:
            parse_csv(path, result, records)
        elif suffix in LOG_SUFFIXES:
            parse_text_log(path, result, records)

    result.steps = sorted(records.values(), key=lambda r: (r.source, r.step))
    if not result.steps:
        result.warnings.append("No NeMo-RL step timings were found.")
    if not result.spec_metrics:
        result.warnings.append(
            "No numeric speculative decoding acceptance/draft metrics were found. "
            "This can be normal if the vLLM build does not print them to the driver log."
        )
    return result


def selected_steps(
    result: ParseResult, drop_first: bool, gen_outlier_threshold: float | None
) -> list[StepRecord]:
    steps = list(result.steps)
    if drop_first and steps:
        min_step = min(step.step for step in steps)
        steps = [step for step in steps if step.step != min_step]
    if gen_outlier_threshold is not None:
        steps = [
            step
            for step in steps
            if step.timings.get("exposed_generation", 0.0) <= gen_outlier_threshold
        ]
    return steps


def summarize_result(
    result: ParseResult, drop_first: bool, gen_outlier_threshold: float | None
) -> dict[str, Any]:
    steps = selected_steps(result, drop_first, gen_outlier_threshold)
    timing: dict[str, dict[str, float | int | None]] = {}
    all_keys = sorted({key for step in steps for key in step.timings})
    preferred_keys = [key for key in DEFAULT_TIMING_KEYS if key in all_keys]
    for key in preferred_keys + [key for key in all_keys if key not in preferred_keys]:
        timing[key] = stats([step.timings[key] for step in steps if key in step.timings])

    spec_values = {key: list(values) for key, values in result.spec_metrics.items()}
    if "acceptance_rate" not in spec_values:
        accepted = spec_values.get("accepted_tokens", [])
        drafted = spec_values.get("draft_tokens", [])
        if accepted and drafted and sum(drafted) > 0:
            spec_values["derived_acceptance_rate"] = [sum(accepted) / sum(drafted)]
    spec = {key: stats(values) for key, values in sorted(spec_values.items())}
    throughput = {key: stats(values) for key, values in sorted(result.throughput_metrics.items())}
    return {
        "files": result.files,
        "num_steps_raw": len(result.steps),
        "num_steps_selected": len(steps),
        "timing": timing,
        "spec_metrics": spec,
        "throughput_metrics": throughput,
        "evidence": result.evidence,
        "source_provenance": {
            "spec_metric_sources": result.spec_metric_sources,
            "timing_sources": result.timing_sources,
            "standalone_vllm_spec_sources": result.standalone_vllm_spec_sources,
            "standalone_vllm_throughput_sources": result.standalone_vllm_throughput_sources,
            "mixed_context_sources": result.mixed_context_sources,
        },
        "run_config": {key: values for key, values in sorted(result.run_config.items())},
        "warnings": result.warnings,
        "spec_lines": result.spec_lines,
    }


def spec_metric_median(summary: dict[str, Any], *names: str) -> float | None:
    metrics = summary.get("spec_metrics", {})
    for name in names:
        for key in (name, f"spec_decode_{name}"):
            value = metrics.get(key, {}).get("median")
            if value is not None:
                return value
    for name in names:
        for key, stats_dict in sorted(metrics.items()):
            if key.startswith("derived_") and key not in names:
                continue
            if key.endswith(f"_{name}"):
                value = stats_dict.get("median")
                if value is not None:
                    return value
    return None


def config_values(summary: dict[str, Any], key: str) -> list[str]:
    values = summary.get("run_config", {}).get(key, [])
    return list(values) if isinstance(values, list) else []


def single_config_value(summary: dict[str, Any], key: str) -> str | None:
    values = config_values(summary, key)
    return values[-1] if values else None


def boolish(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def generation_request_logprobs(summary: dict[str, Any]) -> tuple[bool | None, str]:
    effective = boolish(single_config_value(summary, "effective_generation_request_logprobs"))
    if effective is not None:
        return effective, "effective_generation_request_logprobs"

    force = boolish(single_config_value(summary, "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS"))
    if force is True:
        return True, "NRL_VLLM_SPECDEC_REQUEST_LOGPROBS"

    omit = boolish(single_config_value(summary, "NRL_VLLM_OMIT_GENERATION_LOGPROBS"))
    if omit is True:
        return False, "NRL_VLLM_OMIT_GENERATION_LOGPROBS"
    if omit is False:
        return True, "NRL_VLLM_OMIT_GENERATION_LOGPROBS"

    if summary.get("evidence", {}).get("speculative_config_seen") and force is not True:
        return False, "specdec_default_without_request_logprobs"

    return None, "missing_logprob_mode_evidence"


def baseline_compatibility_check(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    mismatches: list[dict[str, str | None]] = []
    missing: list[dict[str, str | None]] = []

    current_request_logprobs, current_source = generation_request_logprobs(current)
    baseline_request_logprobs, baseline_source = generation_request_logprobs(baseline)
    if current_request_logprobs is None or baseline_request_logprobs is None:
        missing.append(
            {
                "field": "generation_request_logprobs",
                "current_source": current_source,
                "baseline_source": baseline_source,
            }
        )
    elif current_request_logprobs != baseline_request_logprobs:
        mismatches.append(
            {
                "field": "generation_request_logprobs",
                "current": str(current_request_logprobs),
                "baseline": str(baseline_request_logprobs),
            }
        )

    for key in BASELINE_EXACT_MATCH_KEYS:
        current_value = single_config_value(current, key)
        baseline_value = single_config_value(baseline, key)
        if current_value is None and baseline_value is None:
            continue
        if current_value is None or baseline_value is None:
            missing.append(
                {
                    "field": key,
                    "current": current_value,
                    "baseline": baseline_value,
                }
            )
        elif current_value != baseline_value:
            mismatches.append(
                {
                    "field": key,
                    "current": current_value,
                    "baseline": baseline_value,
                }
            )

    passed = not mismatches and baseline_request_logprobs is not None and current_request_logprobs is not None
    reason = None
    if mismatches:
        reason = "baseline/current generation configs differ"
    elif baseline_request_logprobs is None or current_request_logprobs is None:
        reason = "missing generation logprob mode evidence"

    return {
        "name": "baseline_config_compatibility",
        "passed": passed,
        "reason": reason,
        "current_generation_request_logprobs": current_request_logprobs,
        "baseline_generation_request_logprobs": baseline_request_logprobs,
        "current_logprob_source": current_source,
        "baseline_logprob_source": baseline_source,
        "mismatches": mismatches,
        "missing_context": missing,
    }


def gate_result(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    min_gen_speedup_pct: float | None,
    min_acceptance_rate: float | None,
    fail_on_missing_spec_metrics: bool,
    allow_standalone_vllm: bool,
    require_comparable_baseline_config: bool = True,
) -> dict[str, Any]:
    status = "pass"
    checks: list[dict[str, Any]] = []

    current_gen = current["timing"].get("exposed_generation", {}).get("median")
    if baseline is not None:
        compatibility = baseline_compatibility_check(current, baseline)
        checks.append(compatibility)
        if require_comparable_baseline_config and not compatibility["passed"]:
            status = "fail"

    if baseline is not None and min_gen_speedup_pct is not None:
        baseline_gen = baseline["timing"].get("exposed_generation", {}).get("median")
        if baseline_gen and current_gen is not None:
            speedup_pct = (1.0 - current_gen / baseline_gen) * 100.0
            passed = speedup_pct >= min_gen_speedup_pct
            checks.append(
                {
                    "name": "generation_speedup",
                    "passed": passed,
                    "value_pct": speedup_pct,
                    "threshold_pct": min_gen_speedup_pct,
                    "baseline_median_s": baseline_gen,
                    "current_median_s": current_gen,
                }
            )
            if not passed:
                status = "fail"
        else:
            checks.append(
                {
                    "name": "generation_speedup",
                    "passed": False,
                    "reason": "missing baseline or current exposed_generation median",
                }
            )
            status = "fail"

    if min_acceptance_rate is not None:
        acceptance = spec_metric_median(current, "acceptance_rate")
        if acceptance is None:
            acceptance = spec_metric_median(current, "derived_acceptance_rate")
        reliable = spec_metric_median(current, "acceptance_rate_reliable")
        complete = spec_metric_median(current, "metrics_complete")
        partial = spec_metric_median(current, "metrics_partial")
        if acceptance is None:
            passed = not fail_on_missing_spec_metrics
            checks.append(
                {
                    "name": "acceptance_rate",
                    "passed": passed,
                    "reason": "missing acceptance_rate metric",
                    "threshold": min_acceptance_rate,
                }
            )
            if not passed:
                status = "fail"
        else:
            reliability_fail_reason = None
            if reliable is None and complete is None and partial is None:
                if fail_on_missing_spec_metrics:
                    reliability_fail_reason = "missing reliability metrics"
            elif reliable is not None and reliable < 0.5:
                reliability_fail_reason = "acceptance_rate_reliable is false"
            elif complete is not None and complete < 0.5:
                reliability_fail_reason = "metrics_complete is false"
            elif partial is not None and partial >= 0.5:
                reliability_fail_reason = "metrics are partial"
            checks.append(
                {
                    "name": "acceptance_rate_reliability",
                    "passed": reliability_fail_reason is None,
                    "acceptance_rate_reliable": reliable,
                    "metrics_complete": complete,
                    "metrics_partial": partial,
                    "reason": reliability_fail_reason,
                }
            )
            if reliability_fail_reason is not None:
                status = "fail"
            passed = acceptance >= min_acceptance_rate
            checks.append(
                {
                    "name": "acceptance_rate",
                    "passed": passed,
                    "value": acceptance,
                    "threshold": min_acceptance_rate,
                }
            )
            if not passed:
                status = "fail"

    if fail_on_missing_spec_metrics and not current["spec_metrics"]:
        checks.append(
            {
                "name": "spec_metrics_present",
                "passed": False,
                "reason": "no numeric speculative decoding metrics found",
            }
        )
        status = "fail"
    if fail_on_missing_spec_metrics and current["spec_metrics"]:
        metrics_available = current["spec_metrics"].get("metrics_available", {}).get(
            "median"
        )
        if metrics_available is not None and metrics_available < 0.5:
            checks.append(
                {
                    "name": "spec_metrics_available",
                    "passed": False,
                    "value": metrics_available,
                    "reason": "vLLM reported speculative decoding metrics unavailable",
                }
            )
            status = "fail"
        draft_tokens = current["spec_metrics"].get("draft_tokens", {}).get("median")
        if draft_tokens is None:
            draft_tokens = current["spec_metrics"].get("num_draft_tokens", {}).get(
                "median"
            )
        if draft_tokens is None or draft_tokens <= 0:
            checks.append(
                {
                    "name": "spec_decode_active",
                    "passed": False,
                    "value": draft_tokens,
                    "reason": "no positive draft-token metric was found",
                }
            )
            status = "fail"

    provenance = current.get("source_provenance", {})
    standalone_sources = provenance.get("standalone_vllm_spec_sources") or []
    standalone_throughput_sources = (
        provenance.get("standalone_vllm_throughput_sources") or []
    )
    mixed_context_sources = provenance.get("mixed_context_sources") or []
    if (
        (standalone_sources or standalone_throughput_sources or mixed_context_sources)
        and not allow_standalone_vllm
    ):
        checks.append(
            {
                "name": "nemo_rl_metric_provenance",
                "passed": False,
                "reason": (
                    "standalone vLLM metrics or mixed standalone/NeMo-RL log contexts "
                    "found in current inputs; "
                    "rerun with --allow-standalone-vllm only for standalone generation reports"
                ),
                "sources": standalone_sources[:8],
                "throughput_sources": standalone_throughput_sources[:8],
                "mixed_context_sources": mixed_context_sources[:8],
            }
        )
        status = "fail"

    return {"status": status, "checks": checks}


def to_jsonable(result: ParseResult, summary: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": result.paths,
        "files": result.files,
        "steps": [
            {
                "step": step.step,
                "source": step.source,
                "timings": step.timings,
                "timing_pct": step.timing_pct,
            }
            for step in result.steps
        ],
        "source_provenance": summary.get("source_provenance", {}),
        "run_config": summary.get("run_config", {}),
        "summary": summary,
        "gate": gate,
    }


def render_table(title: str, summary: dict[str, Any]) -> str:
    lines = [title]
    lines.append(
        f"files={len(summary['files'])} raw_steps={summary['num_steps_raw']} "
        f"selected_steps={summary['num_steps_selected']}"
    )
    lines.append("")
    lines.append("Timing medians:")
    lines.append("| metric | n | median | mean | p95 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key, item in summary["timing"].items():
        lines.append(
            f"| {key} | {item['n']} | {fmt(item['median'], 's')} | "
            f"{fmt(item['mean'], 's')} | {fmt(item['p95'], 's')} | {fmt(item['max'], 's')} |"
        )

    if summary["spec_metrics"]:
        lines.append("")
        lines.append("SpecDec metrics:")
        lines.append("| metric | n | median | mean | min | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for key, item in summary["spec_metrics"].items():
            suffix = "" if "rate" not in key else ""
            lines.append(
                f"| {key} | {item['n']} | {fmt(item['median'], suffix)} | "
                f"{fmt(item['mean'], suffix)} | {fmt(item['min'], suffix)} | "
                f"{fmt(item['max'], suffix)} |"
            )
    else:
        lines.append("")
        lines.append("SpecDec metrics: none found")

    if summary["throughput_metrics"]:
        lines.append("")
        lines.append("Throughput metrics:")
        lines.append("| metric | n | median | mean |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key, item in summary["throughput_metrics"].items():
            lines.append(f"| {key} | {item['n']} | {fmt(item['median'])} | {fmt(item['mean'])} |")

    if summary["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")

    if summary.get("run_config"):
        lines.append("")
        lines.append("Run config evidence:")
        for key, values in sorted(summary["run_config"].items()):
            rendered_values = ", ".join(values[:3])
            if len(values) > 3:
                rendered_values += ", ..."
            lines.append(f"- {key}: {rendered_values}")

    return "\n".join(lines)


def render_comparison(
    current: dict[str, Any], baseline: dict[str, Any] | None, gate: dict[str, Any]
) -> str:
    lines: list[str] = []
    if baseline is not None:
        base_gen = baseline["timing"].get("exposed_generation", {}).get("median")
        cur_gen = current["timing"].get("exposed_generation", {}).get("median")
        base_total = baseline["timing"].get("total_step_time", {}).get("median")
        cur_total = current["timing"].get("total_step_time", {}).get("median")
        lines.append("")
        lines.append("Baseline comparison:")
        lines.append("| metric | baseline median | current median | delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, base, cur in (
            ("exposed_generation", base_gen, cur_gen),
            ("total_step_time", base_total, cur_total),
        ):
            if base and cur is not None:
                delta = (cur / base - 1.0) * 100.0
                lines.append(f"| {name} | {base:.2f}s | {cur:.2f}s | {delta:+.1f}% |")
            else:
                lines.append(f"| {name} | {fmt(base, 's')} | {fmt(cur, 's')} | - |")

    if gate["checks"]:
        lines.append("")
        lines.append(f"Gate: {gate['status'].upper()}")
        for check in gate["checks"]:
            mark = "PASS" if check.get("passed") else "FAIL"
            details = ", ".join(
                f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in check.items()
                if key not in {"name", "passed"}
            )
            lines.append(f"- {mark} {check['name']}: {details}")

    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Current smoke log files or directories.")
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=Path,
        help="Optional baseline log files/directories for speedup comparison.",
    )
    parser.add_argument("--json-out", type=Path, help="Write full parsed output as JSON.")
    parser.add_argument("--markdown-out", type=Path, help="Write the human summary as Markdown.")
    parser.add_argument(
        "--drop-first-step",
        action="store_true",
        help="Drop the lowest step id from summaries to reduce cold-start noise.",
    )
    parser.add_argument(
        "--gen-outlier-threshold-s",
        type=float,
        default=None,
        help="Exclude selected steps whose exposed_generation exceeds this threshold.",
    )
    parser.add_argument(
        "--min-generation-speedup-pct",
        type=float,
        default=None,
        help="With --baseline, fail the gate unless median exposed_generation improves by this percent.",
    )
    parser.add_argument(
        "--min-acceptance-rate",
        type=float,
        default=None,
        help="Fail the gate unless median acceptance_rate is at least this value, e.g. 0.45.",
    )
    parser.add_argument(
        "--fail-on-missing-spec-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the gate if acceptance/draft metrics are absent from the logs.",
    )
    parser.add_argument(
        "--allow-standalone-vllm",
        action="store_true",
        help="Allow standalone vLLM SpecDec metrics in current inputs. Leave unset for NeMo-RL evidence.",
    )
    parser.add_argument(
        "--allow-unmatched-baseline-config",
        action="store_true",
        help=(
            "Allow speedup reporting even when baseline/current generation-logprob "
            "or batch-shape parity cannot be proven. Use only for legacy diagnostics."
        ),
    )
    args = parser.parse_args()

    current_result = analyze(args.paths)
    current_summary = summarize_result(
        current_result, args.drop_first_step, args.gen_outlier_threshold_s
    )

    baseline_summary = None
    baseline_result = None
    if args.baseline:
        baseline_result = analyze(args.baseline)
        baseline_summary = summarize_result(
            baseline_result, args.drop_first_step, args.gen_outlier_threshold_s
        )

    gate = gate_result(
        current_summary,
        baseline_summary,
        args.min_generation_speedup_pct,
        args.min_acceptance_rate,
        args.fail_on_missing_spec_metrics,
        args.allow_standalone_vllm,
        not args.allow_unmatched_baseline_config,
    )

    rendered = render_table("SpecDec smoke summary", current_summary)
    if baseline_summary is not None:
        rendered += "\n" + render_table("\nBaseline summary", baseline_summary)
    rendered += render_comparison(current_summary, baseline_summary, gate)
    print(rendered)

    if args.markdown_out:
        write_text(args.markdown_out, rendered)

    if args.json_out:
        payload = {
            "current": to_jsonable(current_result, current_summary, gate),
            "baseline": (
                to_jsonable(baseline_result, baseline_summary, {"status": "not_evaluated", "checks": []})
                if baseline_result is not None and baseline_summary is not None
                else None
            ),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return 0 if gate["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
