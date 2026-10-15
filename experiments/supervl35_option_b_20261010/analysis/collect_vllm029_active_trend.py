"""Validate the matched SuperVL3.5 TP4 active-concurrency sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010")
CONCURRENCIES = (64, 128, 256, 512, 1024)
SCHEDULER_STATS = re.compile(r"Running:\s*(\d+)\s+reqs,\s*Waiting:\s*(\d+)\s+reqs")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def collect_job(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    arm = str(job["arm"])
    concurrency = int(job["concurrency"])
    job_id = str(job["job_id"])
    multiplier = int(job["request_multiplier"])
    require(arm in {"bf16", "moe"}, f"Unknown arm: {arm}")
    require(concurrency in CONCURRENCIES, f"Unknown concurrency: {concurrency}")
    require(concurrency * multiplier == 2048, f"Wrong request count: {job_id}")
    run_dir = root / "vllm_length_sweep" / "isl1024_osl1024" / job_id
    require((run_dir / "exit-code").read_text(encoding="utf-8").strip() == "0", f"Failed job: {job_id}")
    result = read_json(run_dir / f"{arm}.json")
    require(result["vllm_version"] == "0.29.0", f"Wrong vLLM version: {job_id}")
    require(result["concurrencies"] == [concurrency], f"Wrong concurrency: {job_id}")
    require(result["max_num_seqs"] == concurrency, f"Wrong sequence cap: {job_id}")
    require(result["request_multiplier"] == multiplier, f"Wrong request multiplier: {job_id}")
    require(result["prompt_tokens"] == 1024, f"Wrong prompt length: {job_id}")
    require(result["enable_prefix_caching"] is True, f"Prefix cache disabled: {job_id}")
    require(result["enable_chunked_prefill"] is True, f"Chunked prefill disabled: {job_id}")
    require(result["moe_backend"] == "flashinfer_trtllm", f"Wrong MoE backend: {job_id}")
    require(result["linear_backend"] == "flashinfer_cutedsl", f"Wrong linear backend: {job_id}")
    require(result["attention_backend"] == "FLASHINFER", f"Wrong attention backend: {job_id}")
    require(bool(result["scoped_mxfp8_ignore"]) == (arm == "moe"), f"Wrong quantization scope: {job_id}")
    require(len(result["runs"]) == 2, f"Expected two trials: {job_id}")
    throughput: list[float] = []
    elapsed: list[float] = []
    for trial, run in enumerate(result["runs"]):
        require(run["trial"] == trial, f"Wrong trial index: {job_id}")
        require(run["requests"] == 2048, f"Wrong request count: {job_id}")
        require(run["batch_size"] == concurrency, f"Wrong batch size: {job_id}")
        require(run["output_length"] == 1024, f"Wrong output length: {job_id}")
        require(run["prompt_tokens"] == 2048 * 1024, f"Wrong prompt tokens: {job_id}")
        require(run["output_tokens"] == 2048 * 1024, f"Wrong output tokens: {job_id}")
        seconds = float(run["elapsed_s"])
        require(seconds > 0, f"Invalid elapsed time: {job_id}")
        rate = 2048 * 1024 / seconds / 4
        require(abs(float(run["output_tokens_per_second_per_gpu"]) - rate) < 1e-4, f"Wrong throughput: {job_id}")
        elapsed.append(seconds)
        throughput.append(rate)
    log = (run_dir / "driver.log").read_text(encoding="utf-8", errors="replace")
    require("enforce_eager=False" in log, f"Eager execution enabled: {job_id}")
    require("CUDAGraphMode.PIECEWISE" in log, f"Wrong CUDA graph mode: {job_id}")
    capture_matches = re.findall(
        r"cudagraph_capture_sizes['\"]?: \[([^\]]+)\].*?max_cudagraph_capture_size['\"]?: (\d+)",
        log,
    )
    require(bool(capture_matches), f"Missing effective CUDA graph config: {job_id}")
    capture_sizes = [int(value) for value in capture_matches[-1][0].split(", ")]
    capture_max = int(capture_matches[-1][1])
    expected_capture_max = min(concurrency * 2, 1024)
    require(
        capture_max == capture_sizes[-1] == expected_capture_max,
        f"Unexpected CUDA graph capture ceiling: {job_id}",
    )
    capture_completions = log.count("Graph capturing finished")
    require(capture_completions > 0, f"CUDA graph capture did not finish: {job_id}")
    samples = SCHEDULER_STATS.findall(log)
    require(bool(samples), f"Missing scheduler samples: {job_id}")
    max_running = max(int(running) for running, _ in samples)
    return {
        "arm": arm,
        "concurrency": concurrency,
        "job_id": job_id,
        "request_multiplier": multiplier,
        "model": result["model"],
        "prompt_sha256": result["prompt_sha256"],
        "config_sha256": result["config_sha256"],
        "max_running": max_running,
        "reached_target": max_running >= concurrency,
        "cudagraph_capture_max": capture_max,
        "cudagraph_capture_completions": capture_completions,
        "elapsed_s": elapsed,
        "output_tok_s_gpu": throughput,
        "mean_output_tok_s_gpu": mean(throughput),
    }


def collect(root: Path, manifest: dict[str, Any], allow_partial: bool = False) -> dict[str, Any]:
    jobs = manifest["jobs"]
    require(len(jobs) == 10, "Expected five matched BF16/MXFP8 job pairs")
    keys = {(job["arm"], int(job["concurrency"])) for job in jobs}
    require(keys == {(arm, c) for arm in ("bf16", "moe") for c in CONCURRENCIES}, "Wrong job matrix")
    require(len({str(job["job_id"]) for job in jobs}) == 10, "Duplicate job IDs")
    records: list[dict[str, Any]] = []
    pending: list[str] = []
    for job in jobs:
        job_id = str(job["job_id"])
        run_dir = root / "vllm_length_sweep" / "isl1024_osl1024" / job_id
        if not (run_dir / "exit-code").exists():
            if allow_partial:
                pending.append(job_id)
                continue
            raise FileNotFoundError(f"Missing completed result: {job_id}")
        records.append(collect_job(root, job))
    require(len({record["model"] for record in records}) <= 1, "Model mismatch")
    require(len({record["prompt_sha256"] for record in records}) <= 1, "Prompt mismatch")
    for arm in ("bf16", "moe"):
        require(len({record["config_sha256"] for record in records if record["arm"] == arm}) <= 1, f"{arm} config mismatch")
    return {"manifest": manifest, "records": records, "pending_jobs": pending}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = collect(args.root, read_json(args.manifest), args.allow_partial)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(summary['records'])}/10 jobs; pending={summary['pending_jobs']}")


if __name__ == "__main__":
    main()
