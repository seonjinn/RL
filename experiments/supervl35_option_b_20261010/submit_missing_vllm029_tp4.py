"""Submit missing matched MXFP8 TP4 points from the vLLM 0.29 sweep matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "vllm_tp4_tp2_matrix_20261014.csv"
LAUNCHER = ROOT / "submit_vllm_tp4_tp2_sweep.sh"
JOB_ID = re.compile(r"^\d+$", re.MULTILINE)


def plan() -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with MATRIX.open(newline="", encoding="utf-8") as source:
        for item in csv.DictReader(source):
            isl = int(item["isl"])
            osl = int(item["osl"])
            concurrency = int(item["concurrency"])
            replicate = int(item["replicate"])
            if (isl, osl, concurrency) in {(1024, 1024, 16), (1024, 1024, 128)}:
                continue
            multiplier = 2 if (isl, osl) == (1024, 10240) and concurrency >= 512 else 10
            rows.append(
                {
                    "isl": isl,
                    "osl": osl,
                    "concurrency": concurrency,
                    "replicate": replicate,
                    "request_multiplier": multiplier,
                }
            )
    keys = {(row["isl"], row["osl"], row["concurrency"], row["replicate"]) for row in rows}
    if len(rows) != len(keys) or len(rows) != 20:
        raise ValueError("Expected 20 unique jobs: 19 missing points plus one C128 repeat")
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"purpose": "SuperVL3.5 vLLM 0.29 matched MXFP8 TP4 sweep", "jobs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("purpose") != "SuperVL3.5 vLLM 0.29 matched MXFP8 TP4 sweep":
        raise ValueError(f"Unexpected manifest purpose in {path}")
    if not isinstance(data.get("jobs"), list):
        raise ValueError(f"Invalid jobs list in {path}")
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["isl"]), int(row["osl"]), int(row["concurrency"]), int(row["replicate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Durable JSON manifest outside the source worktree")
    parser.add_argument("--dry-run", action="store_true", help="Print pending jobs without submitting")
    parser.add_argument("--max-new", type=int, default=0, help="Stop after this many submissions (0 means all)")
    args = parser.parse_args()
    if args.max_new < 0:
        parser.error("--max-new must be nonnegative")

    manifest = load_manifest(args.manifest)
    submitted = {key(row) for row in manifest["jobs"]}
    pending = [row for row in plan() if key(row) not in submitted]
    print(f"Planned: 20 jobs; recorded: {len(submitted)}; pending: {len(pending)}", flush=True)
    if args.dry_run:
        for row in pending:
            print(json.dumps(row, sort_keys=True))
        return

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    count = 0
    for row in pending:
        if args.max_new and count >= args.max_new:
            break
        env = os.environ.copy()
        env["BENCH_REQUEST_MULTIPLIER"] = str(row["request_multiplier"])
        command = [
            "bash", str(LAUNCHER), "mxfp8_tp4", str(row["isl"]),
            str(row["osl"]), str(row["concurrency"]),
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"Submission failed for {row}; reconcile broker state before retrying")
        ids = JOB_ID.findall(result.stdout)
        if len(ids) != 1:
            raise RuntimeError(f"Cannot identify the submitted job for {row}; reconcile broker state before retrying")
        manifest["jobs"].append(
            {**row, "job_id": ids[0], "launcher_commit": commit,
             "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        )
        save_manifest(args.manifest, manifest)
        count += 1
    print(f"New submissions: {count}; recorded total: {len(manifest['jobs'])}", flush=True)


if __name__ == "__main__":
    main()
