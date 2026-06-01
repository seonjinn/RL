#!/usr/bin/env python3
"""Submit a safe Qwen3-235B missing-id repair chain after wave3.

The first wave3 jobs may time out after producing partial generated JSONL
files.  This submitter keeps those running jobs, replaces the unsafe pending
tail, and wires:

1. CPU partial-apply jobs afterany on each wave3 generation job.
2. Smaller vLLM repair jobs afterok on the corresponding partial apply.
3. Finalizer afterok on all new repair jobs.
4. Optional Speculators training chain after the finalizer.

The repair jobs always use the full mixed 500K prompt file, never the OpenMath
smoke prompt default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "eagle3_qwen3_235b"

ARTIFACT_ROOT = Path(
    os.environ.get("ARTIFACT_ROOT", "/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
)
REPORT_DIR = ARTIFACT_ROOT / "reports" / "mixed_target_chunks"
CHUNK_DIR = ARTIFACT_ROOT / "data" / "mixed_target_chunks"
PROMPT_DATA = ARTIFACT_ROOT / "data" / "mixed_math_nonopenmath_500k_prompts.jsonl"
REPLACEMENT_CONVERSATIONS = (
    ARTIFACT_ROOT / "data" / "mixed_math_nonopenmath_qwen3_235b_replacement_conversations_dapo100.jsonl"
)

MODEL_LABEL = "qwen3_235b"
MODEL_PATH = "Qwen/Qwen3-235B-A22B-Thinking-2507"


def run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> str:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return "Submitted batch job DRYRUN"
    try:
        out = subprocess.check_output(
            cmd,
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        if exc.output:
            print(exc.output, flush=True)
        raise
    print(out.strip(), flush=True)
    return out


def parse_job_id(output: str) -> str:
    if "DRYRUN" in output:
        return "DRYRUN"
    for line in output.splitlines():
        if line.startswith("job_id="):
            return line.split("=", 1)[1].strip()
        parts = line.strip().split()
        if len(parts) >= 4 and parts[:3] == ["Submitted", "batch", "job"]:
            return parts[3]
    raise RuntimeError(f"could not parse job id from output:\n{output}")


def parse_wave3_jobs(path: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"^group_(?P<group>\d+)_job=(?P<job>\d+)\s+chunks=(?P<chunks>[0-9,]+)")
    groups: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        groups.append(match.groupdict())
    if not groups:
        raise RuntimeError(f"no wave3 groups parsed from {path}")
    return groups


def load_missing_counts(group: str, chunks: str) -> dict[int, int]:
    prepare_json = REPORT_DIR / f"{MODEL_LABEL}_missing_id_repair_w3_g{group}_prepare.json"
    payload = json.loads(prepare_json.read_text(encoding="utf-8"))
    counts = payload.get("missing_counts", {})
    selected = [int(item) for item in chunks.split(",") if item]
    return {chunk: int(counts.get(f"{chunk:03d}", 0)) for chunk in selected}


def pack_chunks(
    counts: dict[int, int], *, target_missing_per_job: int, max_chunks_per_job: int
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_missing = 0
    for chunk, missing in counts.items():
        would_exceed = (
            current
            and current_missing + missing > target_missing_per_job
            and current_missing > 0
        )
        too_many_chunks = current and len(current) >= max_chunks_per_job
        if would_exceed or too_many_chunks:
            groups.append(current)
            current = []
            current_missing = 0
        current.append(chunk)
        current_missing += missing
    if current:
        groups.append(current)
    return groups


def submit_partial_apply(
    *,
    group: str,
    chunks: str,
    dependency_job: str,
    account: str,
    partition: str,
    dry_run: bool,
) -> str:
    json_out = REPORT_DIR / f"{MODEL_LABEL}_missing_id_repair_w3_g{group}_partial_apply_safe.json"
    prepare_json = REPORT_DIR / f"{MODEL_LABEL}_missing_id_repair_w3_g{group}_prepare.json"
    generated = REPORT_DIR / f"{MODEL_LABEL}_missing_id_repair_w3_g{group}_generated.jsonl"
    repair_py = EXP / "repair_direct_vllm_missing_chunk_ids.py"
    script = f"""set -euo pipefail
cd {shlex.quote(str(ROOT))}
status=0
python3 {shlex.quote(str(repair_py))} apply \
  --prepare-json {shlex.quote(str(prepare_json))} \
  --generated-output {shlex.quote(str(generated))} \
  --chunk-dir {shlex.quote(str(CHUNK_DIR))} \
  --model-label {MODEL_LABEL} \
  --chunk-size 5000 \
  --chunks {shlex.quote(chunks)} \
  --json-out {shlex.quote(str(json_out))} || status=$?
if [[ "$status" == "2" ]]; then
  python3 - {shlex.quote(str(json_out))} <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    payload.get("status") == "incomplete"
    and not payload.get("overfull_chunks")
    and not payload.get("unexpected_ids")
    and not payload.get("duplicate_generated_ids")
):
    print("partial apply accepted incomplete generated coverage; wave4 will repair remaining ids")
    raise SystemExit(0)
raise SystemExit(2)
PY
else
  exit "$status"
fi
"""
    out = run(
        [
            "sbatch",
            "--nodes=1",
            "--ntasks=1",
            f"--account={account}",
            f"--partition={partition}",
            "--gres=gpu:4",
            "--time=00:20:00",
            "--mem=0",
            f"--dependency=afterany:{dependency_job}",
            f"--job-name={MODEL_LABEL}_partial_apply_safe_w3_g{group}",
            "--output=logs/%x_%j.out",
            "--error=logs/%x_%j.err",
            "--wrap",
            f"bash -lc {shlex.quote(script)}",
        ],
        dry_run=dry_run,
    )
    return parse_job_id(out)


def submit_repair_job(
    *,
    group: str,
    index: int,
    chunks: list[int],
    dependency_job: str,
    account: str,
    partition: str,
    time_limit: str,
    dry_run: bool,
) -> str:
    chunk_text = ",".join(f"{chunk:03d}" for chunk in chunks)
    tag = f"{MODEL_LABEL}_missing_id_repair_safe_w4_g{group}_{index:02d}"
    env = {
        "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
        "REMOTE_REPO_ROOT": str(ROOT),
        "REPORT_DIR": str(REPORT_DIR),
        "DATA_DIR": str(ARTIFACT_ROOT / "data"),
        "RUN_SCRIPT": str(EXP / "run_direct_vllm_math_missing_id_repair.sh"),
        "MODEL_LABEL": MODEL_LABEL,
        "MODEL_PATH": MODEL_PATH,
        "PROMPT_DATA": str(PROMPT_DATA),
        "CHUNK_DIR": str(CHUNK_DIR),
        "REPAIR_CHUNKS": chunk_text,
        "REPAIR_TAG": tag,
        "OUTPUT_SCHEMA": "speculators",
        "NUM_RESPONSES": "1",
        "GENERATION_CONCURRENCY": "16",
        "TEMPERATURE": "1.0",
        "TOP_P": "1.0",
        "MAX_TOKENS": "4096",
        "MAX_MODEL_LEN": "8192",
        "VLLM_TP": "8",
        "VLLM_GPU_UTIL": "0.82",
        "VLLM_MAX_NUM_SEQS": "16",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "32768",
        "VLLM_ENFORCE_EAGER": "false",
        "VLLM_DISABLE_LOG_STATS": "true",
        "VLLM_EXTRA_ARGS": "--disable-frontend-multiprocessing --generation-config vllm",
        "SKIP_PROMPT_MATERIALIZE": "true",
        "ACCOUNT": account,
        "PARTITION": partition,
        "NUM_NODES": "2",
        "GPUS_PER_NODE": "4",
        "TIME_LIMIT": time_limit,
        "SBATCH_EXTRA_ARGS": f"--dependency=afterok:{dependency_job}",
        "JOB_NAME": f"{MODEL_LABEL}_missing_id_repair_safe_w4_g{group}_{index:02d}",
        "DRY_RUN": "true" if dry_run else "false",
    }
    out = run(["bash", str(EXP / "submit_direct_vllm_math_rollout_smoke.sh")], env=env, dry_run=False)
    if dry_run:
        return f"DRYRUN_REPAIR_G{group}_{index:02d}"
    return parse_job_id(out)


def submit_finalizer(
    *,
    repair_jobs: list[str],
    extra_dependency_jobs: list[str],
    account: str,
    partition: str,
    dry_run: bool,
) -> str:
    dependency_jobs = [*repair_jobs, *extra_dependency_jobs]
    dependency = "afterok:" + ":".join(dependency_jobs)
    env = {
        "ACCOUNT": account,
        "PARTITION": partition,
        "DEPENDENCY": dependency,
        "JOB_NAME": f"{MODEL_LABEL}-finalize-mixed-500k-safe-w4",
        "GPUS_PER_NODE": "4",
        "REPLACEMENT_CONVERSATIONS": str(REPLACEMENT_CONVERSATIONS),
        "DRY_RUN": "true" if dry_run else "false",
    }
    out = run(["bash", str(EXP / "submit_qwen235b_finalize_after_targetgen.sh")], env=env, dry_run=False)
    if dry_run:
        return "DRYRUN_FINALIZER"
    return parse_job_id(out)


def submit_speculators(*, finalizer_job: str, dry_run: bool) -> dict[str, object] | None:
    out = run(
        [
            "python3",
            str(EXP / "submit_qwen235b_mixed_500k_speculators_after_finalize.py"),
            "--finalizer-job-id",
            finalizer_job,
            "--allow-pending-finalizer",
            "--job-suffix",
            "-safe-w4",
        ]
        + (["--dry-run"] if dry_run else []),
        dry_run=False,
    )
    try:
        start = out.index("{")
        return json.loads(out[start:])
    except Exception:
        return {"raw_output": out}


def cancel_jobs(job_ids: list[str], *, dry_run: bool) -> None:
    cleaned = [job for job in job_ids if job and job != "DRYRUN"]
    if not cleaned:
        return
    run(["scancel", *cleaned], dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave3-job-file", type=Path, default=ROOT / "latest_qwen235b_missing_id_repair_jobs.txt")
    parser.add_argument("--account", default=os.environ.get("ACCOUNT", "coreai_dlalgo_nemorl"))
    parser.add_argument("--partition", default=os.environ.get("PARTITION", "batch"))
    parser.add_argument("--repair-time", default="04:00:00")
    parser.add_argument("--target-missing-per-job", type=int, default=700)
    parser.add_argument("--max-chunks-per-job", type=int, default=40)
    parser.add_argument("--cancel-stale-job-ids", default="")
    parser.add_argument(
        "--extra-finalizer-job-ids",
        default="",
        help="Comma/space separated jobs, such as replacement generation, that finalizer must wait for.",
    )
    parser.add_argument("--skip-speculators", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (args.wave3_job_file, PROMPT_DATA):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.cancel_stale_job_ids:
        cancel_jobs([item for item in re.split(r"[,\s]+", args.cancel_stale_job_ids) if item], dry_run=args.dry_run)

    partial_jobs: dict[str, str] = {}
    repair_jobs: list[str] = []
    repair_plan: list[dict[str, object]] = []

    for group_info in parse_wave3_jobs(args.wave3_job_file):
        group = group_info["group"]
        chunks = group_info["chunks"]
        partial_id = submit_partial_apply(
            group=group,
            chunks=chunks,
            dependency_job=group_info["job"],
            account=args.account,
            partition=args.partition,
            dry_run=args.dry_run,
        )
        partial_jobs[group] = partial_id
        counts = load_missing_counts(group, chunks)
        for idx, chunk_group in enumerate(
            pack_chunks(
                counts,
                target_missing_per_job=args.target_missing_per_job,
                max_chunks_per_job=args.max_chunks_per_job,
            )
        ):
            repair_id = submit_repair_job(
                group=group,
                index=idx,
                chunks=chunk_group,
                dependency_job=partial_id,
                account=args.account,
                partition=args.partition,
                time_limit=args.repair_time,
                dry_run=args.dry_run,
            )
            repair_jobs.append(repair_id)
            repair_plan.append(
                {
                    "group": group,
                    "repair_job": repair_id,
                    "chunks": [f"{chunk:03d}" for chunk in chunk_group],
                    "estimated_missing_from_wave3_prepare": sum(counts.get(chunk, 0) for chunk in chunk_group),
                    "dependency": f"afterok:{partial_id}",
                }
            )

    finalizer_id = submit_finalizer(
        repair_jobs=repair_jobs,
        extra_dependency_jobs=[
            item for item in re.split(r"[,\s]+", args.extra_finalizer_job_ids) if item
        ],
        account=args.account,
        partition=args.partition,
        dry_run=args.dry_run,
    )
    speculators = None if args.skip_speculators else submit_speculators(finalizer_job=finalizer_id, dry_run=args.dry_run)
    extra_finalizer_jobs = [
        item for item in re.split(r"[,\s]+", args.extra_finalizer_job_ids) if item
    ]

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": "qwen3_235b_safe_missing_id_repair_after_wave3",
        "wave3_job_file": str(args.wave3_job_file),
        "prompt_data": str(PROMPT_DATA),
        "replacement_conversations": str(REPLACEMENT_CONVERSATIONS),
        "partial_apply_jobs": partial_jobs,
        "repair_jobs": repair_jobs,
        "repair_plan": repair_plan,
        "extra_finalizer_jobs": extra_finalizer_jobs,
        "finalizer_job": finalizer_id,
        "speculators": speculators,
        "dry_run": args.dry_run,
    }
    report = ARTIFACT_ROOT / "reports/qwen3_235b_safe_missing_id_repair_chain.json"
    if not args.dry_run:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (ROOT / "latest_qwen235b_safe_missing_id_repair_chain.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
