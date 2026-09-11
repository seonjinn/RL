#!/usr/bin/env python3
"""Validate and aggregate the 16 external vLLM engines at the rollout barrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .contract import ExperimentContract, worker_assignment


def aggregate_workers(
    paths: Sequence[Path], *, expected_arm: str
) -> dict[str, Any]:
    contract = ExperimentContract()
    if len(paths) != contract.worker_count:
        raise ValueError(f"expected exactly {contract.worker_count} worker results")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seen_workers: set[int] = set()
    expected_tokens = 0
    actual_tokens = 0
    actual_samples = 0
    barrier_seconds = 0.0
    spec_totals = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    for payload in payloads:
        receipt = payload.get("runtime_contract") or {}
        worker_index = receipt.get("worker_index")
        if type(worker_index) is not int or worker_index in seen_workers:
            raise ValueError("worker indices must be unique integers")
        assignment = worker_assignment(contract, worker_index)
        if receipt.get("arm") != expected_arm:
            raise ValueError("worker arm does not match aggregate arm")
        if receipt.get("prompt_offset") != assignment.prompt_offset:
            raise ValueError("worker prompt partition does not match the contract")
        if receipt.get("seed") != assignment.seed:
            raise ValueError("worker seed does not match the contract")
        if receipt.get("vllm_version") != contract.vllm_version:
            raise ValueError("worker vLLM version does not match the contract")
        if receipt.get("vllm_commit") != contract.vllm_commit:
            raise ValueError("worker vLLM commit does not match the contract")
        rows = payload.get("results")
        if payload.get("partial") is not False or not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("worker result is incomplete")
        row = rows[0]
        lengths = row.get("output_lengths") or {}
        timing = row.get("request_timing") or []
        actual_samples += int(row.get("num_sequences", 0))
        expected_tokens += int(lengths.get("total", 0))
        actual_tokens += sum(int(item["output_tokens"]) for item in timing)
        barrier_seconds = max(barrier_seconds, float(row["wall_s"]))
        spec = row.get("spec_decode") or {}
        for name in spec_totals:
            spec_totals[name] += int(spec.get(name, 0))
        seen_workers.add(worker_index)

    if seen_workers != set(range(contract.worker_count)):
        raise ValueError("aggregate does not contain the exact worker set")
    if actual_samples != contract.global_sample_count:
        raise ValueError("aggregate sample count does not match GBS 2048")
    if actual_tokens != expected_tokens:
        raise ValueError("aggregate token accounting mismatch")
    if actual_tokens <= 0 or barrier_seconds <= 0:
        raise ValueError("aggregate contains no measurable generation work")

    if spec_totals["num_draft_tokens"] > 0:
        spec_totals["acceptance_rate"] = (
            spec_totals["num_accepted_tokens"] / spec_totals["num_draft_tokens"]
        )
    if spec_totals["num_drafts"] > 0:
        spec_totals["mean_acceptance_length"] = 1.0 + (
            spec_totals["num_accepted_tokens"] / spec_totals["num_drafts"]
        )
    return {
        "status": "complete",
        "arm": expected_arm,
        "vllm_version": contract.vllm_version,
        "vllm_commit": contract.vllm_commit,
        "expected_samples": contract.global_sample_count,
        "actual_samples": actual_samples,
        "expected_output_tokens": expected_tokens,
        "actual_output_tokens": actual_tokens,
        "tokens_ok": True,
        "barrier_seconds": barrier_seconds,
        "output_tokens_per_second": actual_tokens / barrier_seconds,
        "spec_decode": spec_totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args()
    paths = sorted(parsed.workers_dir.glob("worker-*.json"))
    summary = aggregate_workers(paths, expected_arm=parsed.arm)
    parsed.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

