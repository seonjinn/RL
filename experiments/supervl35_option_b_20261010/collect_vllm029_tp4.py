"""Validate and collect matched SuperVL3.5 vLLM 0.29 MXFP8 TP4 jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


MATRIX = Path(__file__).resolve().parent / "vllm_tp4_tp2_matrix_20261014.csv"
CONTROL_JOB = "7948449"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"Expected JSON object: {path}")
    return data


def baseline_jobs() -> dict[tuple[int, int, int, int], str]:
    jobs: dict[tuple[int, int, int, int], str] = {}
    with MATRIX.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (int(row["isl"]), int(row["osl"]), int(row["concurrency"]), int(row["replicate"]))
            require(key not in jobs, f"Duplicate matrix key: {key}")
            jobs[key] = row["bf16_job"]
    return jobs


def result_path(root: Path, isl: int, osl: int, job_id: str, arm: str) -> Path:
    return root / "vllm_length_sweep" / f"isl{isl}_osl{osl}" / job_id / f"{arm}.json"


def collect_one(
    root: Path, row: dict[str, Any], baselines: dict[tuple[int, int, int, int], str], control_hash: str
) -> dict[str, Any]:
    isl, osl, concurrency, replicate = (
        int(row["isl"]), int(row["osl"]), int(row["concurrency"]), int(row["replicate"])
    )
    job_id = str(row["job_id"])
    key = (isl, osl, concurrency, replicate)
    baseline_job = baselines[key]
    result_dir = result_path(root, isl, osl, job_id, "moe").parent
    require((result_dir / "exit-code").read_text().strip() == "0", f"Nonzero workload exit: {job_id}")
    baseline_dir = result_path(root, isl, osl, baseline_job, "bf16").parent
    require((baseline_dir / "exit-code").read_text().strip() == "0", f"Nonzero baseline exit: {baseline_job}")
    result = read_json(result_dir / "moe.json")
    baseline = read_json(result_path(root, isl, osl, baseline_job, "bf16"))
    multiplier = int(row["request_multiplier"])
    expected_requests = concurrency * multiplier
    require(result["vllm_version"] == baseline["vllm_version"] == "0.29.0", f"Version mismatch: {job_id}")
    require(result["prompt_sha256"] == baseline["prompt_sha256"], f"Prompt mismatch: {job_id}")
    require(result["config_sha256"] == control_hash, f"MXFP8 TP4 config mismatch: {job_id}")
    require(result["scoped_mxfp8_ignore"] and not baseline["scoped_mxfp8_ignore"], f"Quantization scope mismatch: {job_id}")
    require(result["model"] == baseline["model"], f"Model mismatch: {job_id}")
    require(result["request_multiplier"] == baseline["request_multiplier"] == multiplier, f"Request multiplier mismatch: {job_id}")
    require(result["max_num_seqs"] == baseline["max_num_seqs"] == min(concurrency, 128), f"Sequence cap mismatch: {job_id}")
    require(result["enable_prefix_caching"] == baseline["enable_prefix_caching"] is True, f"Prefix caching mismatch: {job_id}")
    require(result["enable_chunked_prefill"] == baseline["enable_chunked_prefill"] is True, f"Chunked prefill mismatch: {job_id}")
    require(result["moe_backend"] == baseline["moe_backend"] == "flashinfer_trtllm", f"MoE backend mismatch: {job_id}")
    require(result["linear_backend"] == baseline["linear_backend"] == "flashinfer_cutedsl", f"Linear backend mismatch: {job_id}")
    require(result["attention_backend"] == baseline["attention_backend"] == "FLASHINFER", f"Attention backend mismatch: {job_id}")
    require(result["concurrencies"] == baseline["concurrencies"] == [concurrency], f"Concurrency mismatch: {job_id}")
    require(len(result["runs"]) == len(baseline["runs"]) == 1, f"Unexpected trial count: {job_id}")
    run = result["runs"][0]
    base_run = baseline["runs"][0]
    for name, expected in (
        ("requests", expected_requests),
        ("prompt_tokens", expected_requests * isl),
        ("output_tokens", expected_requests * osl),
        ("batch_size", concurrency),
        ("output_length", osl),
    ):
        require(run[name] == base_run[name] == expected, f"{name} mismatch: {job_id}")
    require(run["elapsed_s"] > 0 and base_run["elapsed_s"] > 0, f"Invalid elapsed time: {job_id}")
    expected_rate = run["output_tokens"] / run["elapsed_s"] / 4
    require(abs(run["output_tokens_per_second_per_gpu"] - expected_rate) < 1e-4, f"Throughput mismatch: {job_id}")
    return {
        "isl": isl, "osl": osl, "concurrency": concurrency, "replicate": replicate,
        "request_multiplier": multiplier, "job_id": job_id, "baseline_job": baseline_job,
        "requests": expected_requests, "output_tokens": run["output_tokens"],
        "elapsed_s": run["elapsed_s"], "output_tok_s_gpu": expected_rate,
        "prompt_sha256": result["prompt_sha256"], "config_sha256": result["config_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="OCI-HSG experiment root")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Skip jobs whose result files do not yet exist")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    jobs = manifest["jobs"]
    require(len(jobs) == 20, f"Expected 20 submitted jobs, found {len(jobs)}")
    require(len({str(row["job_id"]) for row in jobs}) == len(jobs), "Duplicate submitted job IDs")
    require(
        {(int(row["isl"]), int(row["osl"]), int(row["concurrency"]), int(row["replicate"])) for row in jobs}
        == set(baseline_jobs()) - {(1024, 1024, 16, 1), (1024, 1024, 128, 1)},
        "Submitted workload keys do not match the missing TP4 points",
    )
    control = read_json(result_path(args.root, 1024, 1024, CONTROL_JOB, "moe"))
    baselines = baseline_jobs()
    collected: list[dict[str, Any]] = []
    pending: list[str] = []
    for row in jobs:
        job_id = str(row["job_id"])
        result_dir = result_path(args.root, int(row["isl"]), int(row["osl"]), job_id, "moe").parent
        exit_file = result_dir / "exit-code"
        if exit_file.exists():
            require(exit_file.read_text().strip() == "0", f"Nonzero workload exit: {job_id}")
        if not exit_file.exists() or not (result_dir / "moe.json").exists():
            if args.allow_partial:
                pending.append(job_id)
                continue
            raise FileNotFoundError(f"Missing result for job {job_id}: {result_dir}")
        collected.append(collect_one(args.root, row, baselines, control["config_sha256"]))

    output = {"completed": len(collected), "expected": len(jobs), "pending_jobs": pending, "rows": collected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Validated {len(collected)}/{len(jobs)} jobs; pending: {pending}", flush=True)


if __name__ == "__main__":
    main()
