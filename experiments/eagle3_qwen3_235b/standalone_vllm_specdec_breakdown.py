#!/usr/bin/env python3
"""Standalone vLLM speculative-decoding timing breakdown.

The benchmark runs controlled static batches through ``vllm.LLM.generate`` and
captures vLLM's embedded torch-profiler traces. It summarizes the traces into
Figure-4-style buckets:

* Drafting
* Verification
* Rejection Sampling
* Other vLLM overheads

The attribution is intentionally conservative. Only events with explicit names
from vLLM or the optional ``specdec_breakdown`` instrumentation are counted in a
non-Other bucket; unclassified wall time remains Other.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from pathlib import Path
from typing import Any


BUCKETS = ("drafting", "verification", "rejection_sampling")


def read_trace_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    return json.loads(path.read_text(errors="replace"))


def load_json_arg(value: str) -> dict[str, Any]:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def classify_event(name: str) -> str | None:
    lowered = name.lower()
    if "specdec_breakdown.drafting" in lowered:
        return "drafting"
    if "specdec_breakdown.verification" in lowered:
        return "verification"
    if "specdec_breakdown.rejection_sampling" in lowered:
        return "rejection_sampling"
    if "rejection" in lowered or "reject" in lowered:
        return "rejection_sampling"
    if "verify" in lowered or "verification" in lowered:
        return "verification"
    if re.search(r"\b(sample|sampler|sampling)\b", lowered) and (
        "spec" in lowered or "reject" in lowered
    ):
        return "rejection_sampling"
    if (
        "draft" in lowered
        or "drafter" in lowered
        or "eagle" in lowered
        or "propose" in lowered
        or "proposal" in lowered
    ):
        return "drafting"
    return None


def trace_files(profile_dir: Path) -> list[Path]:
    patterns = ("*.pt.trace.json", "*.pt.trace.json.gz", "*.trace.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(profile_dir.rglob(pattern))
    return sorted(set(files))


def analyze_trace_files(files: list[Path], wall_time_s: float) -> dict[str, Any]:
    bucket_us = {bucket: 0.0 for bucket in BUCKETS}
    bucket_events = {bucket: 0 for bucket in BUCKETS}
    matched_events: list[dict[str, Any]] = []
    total_events = 0

    for path in files:
        try:
            payload = read_trace_json(path)
        except Exception as exc:
            matched_events.append(
                {"trace": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for event in payload.get("traceEvents", []):
            total_events += 1
            name = str(event.get("name", ""))
            dur = event.get("dur")
            if dur is None or event.get("ph") != "X":
                continue
            bucket = classify_event(name)
            if bucket is None:
                continue
            try:
                dur_us = float(dur)
            except (TypeError, ValueError):
                continue
            bucket_us[bucket] += dur_us
            bucket_events[bucket] += 1
            if len(matched_events) < 80:
                matched_events.append(
                    {
                        "bucket": bucket,
                        "name": name,
                        "dur_us": dur_us,
                        "trace": str(path),
                    }
                )

    wall_us = wall_time_s * 1_000_000.0
    attributed_us = sum(bucket_us.values())
    other_us = max(wall_us - attributed_us, 0.0)
    all_buckets = dict(bucket_us)
    all_buckets["other_vllm_overheads"] = other_us

    percentages = {
        bucket: (value / wall_us * 100.0 if wall_us > 0 else 0.0)
        for bucket, value in all_buckets.items()
    }
    return {
        "wall_time_s": wall_time_s,
        "trace_files": [str(path) for path in files],
        "total_trace_events": total_events,
        "bucket_duration_us": all_buckets,
        "bucket_percent_of_wall": percentages,
        "bucket_event_counts": bucket_events,
        "matched_event_examples": matched_events,
        "attribution_coverage_pct": (
            min(attributed_us, wall_us) / wall_us * 100.0 if wall_us > 0 else 0.0
        ),
        "overlap_or_overcount_us": max(attributed_us - wall_us, 0.0),
    }


def build_llm(args: argparse.Namespace, capture_sizes: list[int]):
    from vllm import LLM

    max_bs = max(capture_sizes)
    compilation_config: dict[str, Any] = {
        "cudagraph_capture_sizes": sorted(set(capture_sizes)),
    }
    if args.compilation_config_json:
        compilation_config.update(json.loads(args.compilation_config_json))

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "trust_remote_code": True,
        "max_num_seqs": max(args.batch_sizes),
        "max_num_batched_tokens": args.max_num_batched_tokens
        or max(max_bs * args.isl, args.isl + args.osl),
        "max_model_len": args.max_model_len or (args.isl + args.osl + 1024),
        "enable_chunked_prefill": False,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 0,
        "compilation_config": compilation_config,
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.distributed_executor_backend != "none":
        kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.speculative_config:
        kwargs["speculative_config"] = load_json_arg(args.speculative_config)
    return LLM(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--speculative-config")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", default="none")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--isl", type=int, default=1000)
    parser.add_argument("--osl", type=int, default=1000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--compilation-config-json")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(profile_dir)

    from vllm import SamplingParams

    batch_sizes = sorted(set(args.batch_sizes))
    num_spec_tokens = 0
    if args.speculative_config:
        num_spec_tokens = int(load_json_arg(args.speculative_config).get("num_speculative_tokens", 0))
    capture_sizes = sorted(
        set(batch_sizes + [bs * max(1, num_spec_tokens + 1) for bs in batch_sizes])
    )

    llm = build_llm(args, capture_sizes=capture_sizes)
    sampling_params = SamplingParams(
        min_tokens=args.osl,
        max_tokens=args.osl,
        ignore_eos=True,
        temperature=0.0,
        seed=0,
    )
    dummy_ids = list(range(args.isl))
    total_gpus = args.tp * args.pp

    rows: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        output.write_text(
            json.dumps(
                {
                    "config": {
                        "tag": args.tag,
                        "model": args.model,
                        "speculative_config": args.speculative_config,
                        "tp": args.tp,
                        "pp": args.pp,
                        "total_gpus": total_gpus,
                        "isl": args.isl,
                        "osl": args.osl,
                        "batch_sizes": batch_sizes,
                        "capture_sizes": capture_sizes,
                        "profile_dir": str(profile_dir),
                        "bucket_definition": [
                            "drafting",
                            "verification",
                            "rejection_sampling",
                            "other_vllm_overheads",
                        ],
                    },
                    "results": rows,
                },
                indent=2,
            )
        )

    for bs in batch_sizes:
        prompts = [{"prompt_token_ids": dummy_ids} for _ in range(bs)]
        for _ in range(args.warmup_repeats):
            llm.generate(prompts, sampling_params)

        before = set(trace_files(profile_dir))
        if hasattr(llm, "start_profile"):
            llm.start_profile()
        started = time.perf_counter()
        llm.generate(prompts, sampling_params)
        latency_s = time.perf_counter() - started
        if hasattr(llm, "stop_profile"):
            llm.stop_profile()

        time.sleep(2.0)
        after = set(trace_files(profile_dir))
        new_files = sorted(after - before) or sorted(after)
        breakdown = analyze_trace_files(new_files, latency_s)
        row = {
            "bs": bs,
            "latency_s": latency_s,
            "output_tok_s": bs * args.osl / latency_s,
            "output_tok_s_per_gpu": bs * args.osl / latency_s / total_gpus,
            "breakdown": breakdown,
        }
        rows.append(row)
        flush()
        print(
            f"bs={bs} latency={latency_s:.3f}s "
            f"out/gpu={row['output_tok_s_per_gpu']:.2f} "
            f"coverage={breakdown['attribution_coverage_pct']:.1f}%"
        )

    flush()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
