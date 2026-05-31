#!/usr/bin/env python3
"""Analyze Qwen3 RL rollout capture artifacts for Eagle3 corpus readiness."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from normalize_rl_rollouts_to_conversations import extract_from_record


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_SOURCE_VLLM_PIP_SPEC = (
    "https://files.pythonhosted.org/packages/7d/0a/278d7bbf454f7de5322a5007427eed3e8b34ed6c2802491b56bbdfd7bbb4/"
    "vllm-0.10.2.tar.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--rollout-log-dir", type=Path)
    parser.add_argument("--output-data", type=Path)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--sample-lines", type=int, default=200)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--infer-flat-content-roles", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def default_rollout_log_dir(args: argparse.Namespace) -> Path:
    return args.rollout_log_dir or args.artifact_root / "rl_rollout_capture_logs/qwen3_235b_swe_capture_smoke"


def default_output_data(args: argparse.Namespace) -> Path:
    return args.output_data or args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def proven_source_vllm_env(artifact_root: Path) -> dict[str, str]:
    source_build = load_json(artifact_root / "reports/vllm_native_source_build.json")
    abi_probe = load_json(artifact_root / "reports/vllm_native_abi_probe.json")
    source_site = str(source_build.get("output_site") or "")
    if source_build.get("overall_status") != "pass" or not source_site:
        return {}
    if abi_probe.get("overall_status") != "pass":
        return {}
    for result in abi_probe.get("results") or []:
        if not isinstance(result, dict) or str(result.get("site") or "") != source_site:
            continue
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        if result.get("returncode") == 0 and parsed.get("vllm_c_ok") and parsed.get("compilation_config_ok"):
            return {
                "INSTALL_VLLM_IN_SYSTEM": "true",
                "SHARED_VLLM_SITE": source_site,
                "VLLM_PIP_SPEC": DEFAULT_SOURCE_VLLM_PIP_SPEC,
                "VLLM_ENFORCE_EAGER": "True",
                "VLLM_COMPILATION_LEVEL": "0",
                "VLLM_USE_INDUCTOR": "False",
            }
    return {}


def shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def iter_jsonl(path: Path, limit: int):
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            if line_num > limit:
                break
            text = line.strip()
            if not text:
                continue
            try:
                yield line_num, json.loads(text), None
            except json.JSONDecodeError as exc:
                yield line_num, None, str(exc)


def inspect_train_data(files: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    rows = 0
    invalid_json = 0
    extracted = 0
    key_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for path in files:
        for line_num, record, error in iter_jsonl(path, args.sample_lines):
            rows += 1
            if error:
                invalid_json += 1
                continue
            if not isinstance(record, dict):
                continue
            for key in record:
                key_counts[key] = key_counts.get(key, 0) + 1
            found = extract_from_record(
                record,
                path,
                line_num,
                None,
                args.min_assistant_chars,
                False,
                "<think>\n",
                "\n</think>\n\n",
                args.infer_flat_content_roles,
            )
            extracted += len(found)
            if found and len(samples) < 3:
                sample = dict(found[0])
                sample["messages"] = sample["messages"][:2]
                samples.append(sample)
    return {
        "files": [str(path) for path in files],
        "file_count": len(files),
        "rows_sampled": rows,
        "invalid_json": invalid_json,
        "extractable_conversations": extracted,
        "key_counts": key_counts,
        "samples": samples,
    }


def train_data_files(rollout_log_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in ("train_data_step*.jsonl", "exp_*/train_data_step*.jsonl"):
        for path in sorted(rollout_log_dir.glob(pattern)):
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files


def validate_output(output_data: Path, validation_json: Path, max_seq_len: int) -> dict[str, Any]:
    if not output_data.exists():
        return {"exists": False, "status": "missing", "path": str(output_data)}
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/validate_training_conversations.py",
        str(output_data),
        "--max-seq-len",
        str(max_seq_len),
        "--json-out",
        str(validation_json),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload: dict[str, Any] = {
        "exists": True,
        "status": "pass" if result.returncode == 0 else "fail",
        "path": str(output_data),
        "validation_json": str(validation_json),
        "returncode": result.returncode,
        "output_tail": result.stdout[-4000:],
    }
    if validation_json.exists():
        try:
            payload["validation"] = json.loads(validation_json.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["validation_parse_error"] = str(exc)
    return payload


def recommendation(args: argparse.Namespace, rollout_log_dir: Path, output_data: Path) -> dict[str, str]:
    base_env = {
        "ARTIFACT_ROOT": str(args.artifact_root),
        "ROLLOUT_LOG_DIR": str(rollout_log_dir),
        "OUTPUT_CONVERSATIONS": str(output_data),
    }
    runtime_env = proven_source_vllm_env(args.artifact_root)
    return {
        "capture_plan_command": (
            f"{shell_env({'DRY_RUN': 'true', **base_env, **runtime_env})} "
            "bash experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"
        ),
        "capture_submit_command": (
            f"{shell_env({'DRY_RUN': 'false', **base_env, **runtime_env})} "
            "bash experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"
        ),
        "materialize_command": (
            f"ARTIFACT_ROOT={args.artifact_root} ROLLOUT_LOG_DIR={rollout_log_dir} "
            f"OUTPUT_DATA={output_data} bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh"
        ),
        "pipeline_dry_run_command": (
            f"INPUT_DATA={output_data} ARTIFACT_ROOT={args.artifact_root} "
            "SUBMIT=false RUN_PILOT=true bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh"
        ),
    }


def overall_status(train: dict[str, Any], output: dict[str, Any]) -> str:
    if train["file_count"] == 0:
        return "missing_capture"
    if train["invalid_json"]:
        return "fail"
    if train["extractable_conversations"] == 0:
        return "fail"
    if output["status"] == "missing":
        return "needs_materialize"
    if output["status"] == "pass":
        return "pass"
    return "fail"


def render_markdown(data: dict[str, Any]) -> str:
    train = data["train_data"]
    output = data["output_data"]
    rec = data["recommendation"]
    lines = [
        "# Rollout Capture Analysis",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        f"Rollout log dir: `{data['rollout_log_dir']}`",
        f"Output data: `{output['path']}`",
        "",
        "| check | value |",
        "| --- | --- |",
        f"| train files | {train['file_count']} |",
        f"| rows sampled | {train['rows_sampled']} |",
        f"| extractable conversations | {train['extractable_conversations']} |",
        f"| invalid JSON | {train['invalid_json']} |",
        f"| output status | {output['status']} |",
        "",
        "Next commands:",
        "",
        "```bash",
    ]
    if data["overall_status"] == "missing_capture":
        lines.append(rec["capture_plan_command"])
    elif data["overall_status"] == "needs_materialize":
        lines.append(rec["materialize_command"])
    elif data["overall_status"] == "pass":
        lines.append(rec["pipeline_dry_run_command"])
    else:
        lines.append("# inspect the JSON report and source logs before continuing")
    lines += ["```", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rollout_log_dir = default_rollout_log_dir(args)
    output_data = default_output_data(args)
    validation_json = args.validation_json or output_data.with_suffix(".validation.json")
    files = train_data_files(rollout_log_dir)
    train = inspect_train_data(files, args)
    output = validate_output(output_data, validation_json, args.max_seq_len)
    data = {
        "overall_status": overall_status(train, output),
        "artifact_root": str(args.artifact_root),
        "rollout_log_dir": str(rollout_log_dir),
        "train_data": train,
        "output_data": output,
        "recommendation": recommendation(args, rollout_log_dir, output_data),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    text = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    print(text)
    return 1 if data["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
