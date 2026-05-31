#!/usr/bin/env python3
"""Submit restartable continuation jobs for direct-vLLM target chunks.

The first target-generation jobs may hit the cluster wall time before a full
5k-row chunk is complete, especially for Qwen3-235B. These continuation jobs
depend on the previous wave with afterany, compute the current JSONL line count
at runtime, and append only the missing prompt slice.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"


MODEL_DEFAULTS = {
    "qwen3_30ba3b": {
        "model_path": "Qwen/Qwen3-30B-A3B",
        "num_nodes": "1",
        "vllm_tp": "4",
        "generation_concurrency": "64",
        "vllm_max_num_seqs": "64",
        "vllm_extra_args": "--generation-config vllm",
    },
    "qwen3_235b": {
        "model_path": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "num_nodes": "2",
        "vllm_tp": "8",
        "generation_concurrency": "16",
        "vllm_max_num_seqs": "16",
        "vllm_extra_args": "--disable-frontend-multiprocessing --generation-config vllm",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", choices=MODEL_DEFAULTS, required=True)
    parser.add_argument("--artifact-root", default="/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
    parser.add_argument("--remote-repo-root", default="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Nemo-RL_Qwen3_Roadmap")
    parser.add_argument("--prompt-data", default=None)
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--start-chunk", type=int, default=0, help="First chunk index to submit.")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--wave-limit", type=int, default=1000)
    parser.add_argument("--start-wave", type=int, default=1)
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--account", default="coreai_dlalgo_nemorl")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--previous-job-ids", default="", help="Comma-separated dependency job ids, one per chunk.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_submit(env: dict[str, str], dry_run: bool) -> tuple[str | None, str]:
    cmd = ["bash", str(EXP / "submit_direct_vllm_math_rollout_smoke.sh")]
    if dry_run:
        env = {**env, "DRY_RUN": "true"}
    else:
        env = {**env, "DRY_RUN": "false"}
    proc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"submit failed with rc={proc.returncode}\n{output}")
    job_id = None
    for line in output.splitlines():
        if line.startswith("job_id="):
            job_id = line.split("=", 1)[1].strip()
        elif "Submitted batch job" in line:
            job_id = line.rsplit(" ", 1)[-1].strip()
    return job_id, output


def main() -> int:
    args = parse_args()
    defaults = MODEL_DEFAULTS[args.model_label]
    artifact_root = Path(args.artifact_root)
    prompt_data = args.prompt_data or str(artifact_root / "data" / "mixed_math_nonopenmath_500k_prompts.jsonl")
    chunk_dir = Path(args.chunk_dir or artifact_root / "data" / "mixed_target_chunks")
    report_dir = Path(args.report_dir or artifact_root / "reports" / "mixed_target_chunks")

    previous = [item for item in args.previous_job_ids.split(",") if item]
    if previous and len(previous) != args.chunks:
        raise SystemExit("--previous-job-ids must contain one job id per chunk")

    submitted: list[dict[str, str | int | None]] = []
    deps: list[str | None] = previous if previous else [None] * args.chunks

    for wave in range(args.start_wave, args.start_wave + args.waves):
        next_deps: list[str | None] = []
        for dep_index, chunk_index in enumerate(range(args.start_chunk, args.start_chunk + args.chunks)):
            output = chunk_dir / f"{args.model_label}_{chunk_index:03d}.jsonl"
            tag = f"{args.model_label}_{chunk_index:03d}_resume_w{wave}"
            env = {
                "ARTIFACT_ROOT": str(artifact_root),
                "REMOTE_REPO_ROOT": args.remote_repo_root,
                "REPORT_DIR": str(report_dir),
                "DATA_DIR": str(artifact_root / "data"),
                "RUN_SCRIPT": str(EXP / "run_direct_vllm_math_resume_chunk.sh"),
                "MODEL_LABEL": args.model_label,
                "CHUNK_INDEX": str(chunk_index),
                "CHUNK_SIZE": str(args.chunk_size),
                "WAVE_LIMIT": str(args.wave_limit),
                "BASE_OFFSET": str(chunk_index * args.chunk_size),
                "MODEL_PATH": defaults["model_path"],
                "PROMPT_DATA": prompt_data,
                "OUTPUT_CONVERSATIONS": str(output),
                "SUMMARY_JSON": str(report_dir / f"{tag}_summary.json"),
                "SERVER_LOG": str(report_dir / f"{tag}_server.log"),
                "GENERATION_LOG": str(report_dir / f"{tag}_generation.log"),
                "SKIP_PROMPT_MATERIALIZE": "true",
                "OUTPUT_SCHEMA": "speculators",
                "NUM_RESPONSES": "1",
                "GENERATION_CONCURRENCY": defaults["generation_concurrency"],
                "TEMPERATURE": "1.0",
                "TOP_P": "1.0",
                "MAX_TOKENS": "4096",
                "MAX_MODEL_LEN": "8192",
                "VLLM_TP": defaults["vllm_tp"],
                "VLLM_GPU_UTIL": "0.82",
                "VLLM_MAX_NUM_SEQS": defaults["vllm_max_num_seqs"],
                "VLLM_MAX_NUM_BATCHED_TOKENS": "32768",
                "VLLM_ENFORCE_EAGER": "false",
                "VLLM_DISABLE_LOG_STATS": "true",
                "VLLM_EXTRA_ARGS": defaults["vllm_extra_args"],
                "ACCOUNT": args.account,
                "PARTITION": args.partition,
                "NUM_NODES": defaults["num_nodes"],
                "GPUS_PER_NODE": "4",
                "TIME_LIMIT": args.time_limit,
                "JOB_NAME": f"{args.model_label}-targetgen-resume-{chunk_index}-w{wave}",
            }
            if deps[dep_index]:
                env["SBATCH_EXTRA_ARGS"] = f"--dependency=afterany:{deps[dep_index]}"

            job_id, output_text = run_submit(env, args.dry_run)
            submitted.append(
                {
                    "wave": wave,
                    "chunk_index": chunk_index,
                    "dependency": deps[dep_index],
                    "job_id": job_id,
                    "output_conversations": str(output),
                }
            )
            next_deps.append(job_id or deps[dep_index])
            print(output_text, end="" if output_text.endswith("\n") else "\n")
        deps = next_deps

    summary = {
        "model_label": args.model_label,
        "chunk_size": args.chunk_size,
        "start_chunk": args.start_chunk,
        "wave_limit": args.wave_limit,
        "waves": args.waves,
        "dry_run": args.dry_run,
        "submitted": submitted,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
