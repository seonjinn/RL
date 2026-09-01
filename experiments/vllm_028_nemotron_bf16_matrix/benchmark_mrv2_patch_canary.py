#!/usr/bin/env python3
"""Run one patched-MRV2 engine through controlled DynamicMTP K depths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
CANARY_BATCH_SIZES = (1, 2, 4, 8, 16)
CANARY_DYNAMIC_SCHEDULE = "1:1:5,2:2:3,3:4:2,5:8:1,9:512:0"
CANARY_MAX_K = 5


def _load_sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load sibling module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_k_by_batch() -> dict[int, int]:
    return {1: 5, 2: 3, 4: 2, 8: 1, 16: 0}


def build_canary_argv(
    *, model_key: str, isl: int, osl: int, output: Path
) -> list[str]:
    benchmark = _load_sibling("benchmark")
    argv = benchmark.build_legacy_benchmark_argv(
        {
            "model_key": model_key,
            "runner_key": "mrv2",
            "method_key": "mtp_dynamic_max_k5",
            "isl": isl,
            "osl": osl,
            "batch_sizes": list(CANARY_BATCH_SIZES),
            "max_num_seqs": 512,
            "speculative_config": None,
            "result_path": str(output),
        }
    )
    schedule_index = argv.index("--dynamic-schedule") + 1
    argv[schedule_index] = CANARY_DYNAMIC_SCHEDULE
    return argv


def validate_canary_payload(payload: dict[str, Any], *, osl: int) -> dict[str, Any]:
    if payload.get("status") != "complete":
        raise ValueError("canary payload must be complete")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("canary payload must contain results")
    expected = expected_k_by_batch()
    observed_batches = [int(row.get("bs", row.get("batch_size", 0))) for row in rows]
    if observed_batches != list(expected):
        raise ValueError(
            f"canary batches must be {list(expected)}, got {observed_batches}"
        )

    for row in rows:
        batch_size = int(row.get("bs", row.get("batch_size", 0)))
        actual_tokens = int(row["output_tokens"])
        expected_tokens = batch_size * osl
        row["expected_output_tokens"] = expected_tokens
        row["actual_output_tokens"] = actual_tokens
        row["tokens_ok"] = actual_tokens == expected_tokens
        if not row["tokens_ok"]:
            raise ValueError(
                f"output tokens mismatch at batch {batch_size}: "
                f"actual={actual_tokens}, expected={expected_tokens}"
            )

        metrics = row.get("spec_decode_metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"missing speculative metrics at batch {batch_size}")
        drafts = float(metrics.get("num_drafts", 0.0))
        draft_tokens = float(metrics.get("num_draft_tokens", 0.0))
        expected_k = expected[batch_size]
        row["draft_counter_residual_at_k0"] = False
        if expected_k == 0:
            observed_width = 0.0
            row["draft_counter_async_skew_limit_tokens"] = float(
                batch_size * CANARY_MAX_K
            )
            residual_is_bounded = (
                drafts <= batch_size
                and draft_tokens <= batch_size * CANARY_MAX_K
            )
            if not residual_is_bounded:
                raise ValueError(
                    "K0 canary emitted substantive draft work beyond one "
                    "offered batch of asynchronous counter skew"
                )
            row["draft_counter_async_skew_tokens"] = draft_tokens
            row["draft_counter_residual_at_k0"] = draft_tokens > 0.0
        else:
            if drafts <= 0.0:
                raise ValueError(f"K{expected_k} canary emitted no drafts")
            observed_width = draft_tokens / drafts
            counter_skew = draft_tokens - drafts * expected_k
            counter_skew_limit = float(batch_size * CANARY_MAX_K)
            row["draft_counter_async_skew_limit_tokens"] = counter_skew_limit
            row["draft_counter_async_skew_tokens"] = counter_skew
            if abs(counter_skew) > counter_skew_limit:
                raise ValueError(
                    f"draft width mismatch at batch {batch_size}: "
                    f"observed={observed_width}, expected={expected_k}"
                )
        row["draft_counter_attribution"] = (
            "prometheus_async_delta_bounded"
            if row["draft_counter_async_skew_tokens"] != 0.0
            else "prometheus_async_delta_exact"
        )
        row["expected_dynamic_k"] = expected_k
        row["observed_mean_draft_width"] = observed_width
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("super", "ultra"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--distributed-executor-backend", choices=("ray",))
    parser.add_argument("--isl", type=int, default=1000)
    parser.add_argument("--osl", type=int, default=128)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--dynamic-schedule", required=True)
    parser.add_argument("--patchset-manifest-sha256", required=True)
    parser.add_argument("--container-artifact", required=True)
    parser.add_argument("--container-artifact-sha256", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--harness-manifest-sha256", required=True)
    parser.add_argument("--ray-version", default="")
    parser.add_argument("--ray-bundle", default="")
    parser.add_argument("--ray-bundle-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parsed = _build_parser().parse_args()
    if tuple(parsed.batch_sizes) != CANARY_BATCH_SIZES:
        raise ValueError(f"batch sizes must be {CANARY_BATCH_SIZES}")
    if parsed.dynamic_schedule != CANARY_DYNAMIC_SCHEDULE:
        raise ValueError("dynamic schedule does not match the controlled canary")
    expected_tp = 2 if parsed.model_key == "super" else 8
    if parsed.tensor_parallel_size != expected_tp:
        raise ValueError("tensor parallel size does not match model topology")
    if parsed.enable_expert_parallel != (parsed.model_key == "ultra"):
        raise ValueError("expert parallel setting does not match model topology")
    if (parsed.distributed_executor_backend == "ray") != (
        parsed.model_key == "ultra"
    ):
        raise ValueError("Ray setting does not match model topology")

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = parsed.output.with_name(f"{parsed.output.stem}.raw.json")
    attempt_output = parsed.output.with_name("attempt.json")
    provenance = {
        "vllm_version": "0.28.0",
        "vllm_base_commit": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
        "patchset_manifest_sha256": parsed.patchset_manifest_sha256,
        "container_artifact": parsed.container_artifact,
        "container_artifact_sha256": parsed.container_artifact_sha256,
        "harness_commit": parsed.harness_commit,
        "harness_manifest_sha256": parsed.harness_manifest_sha256,
        "ray_version": parsed.ray_version or None,
        "ray_bundle": parsed.ray_bundle or None,
        "ray_bundle_sha256": parsed.ray_bundle_sha256 or None,
    }
    attempt = {
        "status": "running",
        "config": {
            "model_key": parsed.model_key,
            "runner_key": "mrv2",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "isl": parsed.isl,
            "osl": parsed.osl,
            "batch_sizes": list(CANARY_BATCH_SIZES),
            "dynamic_schedule": CANARY_DYNAMIC_SCHEDULE,
        },
        "runtime_provenance": provenance,
    }
    attempt_output.write_text(json.dumps(attempt, indent=2) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            build_canary_argv(
                model_key=parsed.model_key,
                isl=parsed.isl,
                osl=parsed.osl,
                output=raw_output,
            ),
            check=True,
        )
        payload = json.loads(raw_output.read_text(encoding="utf-8"))
        validated = validate_canary_payload(payload, osl=parsed.osl)
    except BaseException as error:
        attempt["status"] = "failed"
        attempt["error_type"] = type(error).__name__
        attempt["error"] = str(error)
        attempt_output.write_text(
            json.dumps(attempt, indent=2) + "\n", encoding="utf-8"
        )
        raise

    validated["runtime_provenance"] = provenance
    validated["config"].update(attempt["config"])
    temporary = parsed.output.with_name(f"{parsed.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
    temporary.replace(parsed.output)
    attempt["status"] = "complete"
    attempt_output.write_text(json.dumps(attempt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
