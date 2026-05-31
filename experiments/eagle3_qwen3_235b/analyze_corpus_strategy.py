#!/usr/bin/env python3
"""Summarize which corpus path should feed Qwen3 Eagle3 draft training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument("--target-context", choices=("swe_rl", "math", "general"), default=os.environ.get("EAGLE3_TARGET_CONTEXT", "swe_rl"))
    parser.add_argument("--input-data", type=Path, default=Path(os.environ["INPUT_DATA"]) if os.environ.get("INPUT_DATA") else None)
    parser.add_argument(
        "--rollout-capture-analysis-json",
        type=Path,
        default=Path(os.environ["ROLLOUT_CAPTURE_ANALYSIS_JSON"]) if os.environ.get("ROLLOUT_CAPTURE_ANALYSIS_JSON") else None,
    )
    parser.add_argument("--math-data", action="append", type=Path, default=[])
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"parse_error": str(exc), "path": str(path)}


def validate_input(path: Path | None, max_seq_len: int, sample_limit: int) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "status": "not_set"}
    if not path.exists():
        return {"exists": False, "status": "missing", "path": str(path)}
    validation_json = path.with_suffix(".strategy_validation.json")
    cmd = [
        "python3",
        "experiments/eagle3_qwen3_235b/validate_training_conversations.py",
        str(path),
        "--limit",
        str(sample_limit),
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
    data: dict[str, Any] = {
        "exists": True,
        "status": "pass" if result.returncode == 0 else "fail",
        "path": str(path),
        "validation_json": str(validation_json),
        "returncode": result.returncode,
        "output_tail": result.stdout[-4000:],
    }
    if validation_json.exists():
        try:
            data["validation"] = json.loads(validation_json.read_text(encoding="utf-8"))
        except Exception as exc:
            data["validation_parse_error"] = str(exc)
    return data


def inspect_paths(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        try:
            if path.exists():
                stat = path.stat()
                item["size_bytes"] = stat.st_size
                item["is_dir"] = path.is_dir()
        except OSError as exc:
            item["error"] = str(exc)
        out.append(item)
    return out


def rollout_output_path(rollout: dict[str, Any] | None) -> Path | None:
    if not rollout:
        return None
    output = rollout.get("output_data")
    if output is None and isinstance(rollout.get("artifacts"), dict):
        output = rollout["artifacts"].get("output_data")
    if isinstance(output, dict) and output.get("path"):
        return Path(str(output["path"]))
    if isinstance(output, str) and output:
        return Path(output)
    return None


def rollout_status(rollout: dict[str, Any] | None) -> str:
    if not rollout:
        return "missing_report"
    candidates = [
        rollout.get("overall_status"),
        (rollout.get("decision") or {}).get("overall_status") if isinstance(rollout.get("decision"), dict) else None,
        (rollout.get("artifacts") or {}).get("overall_status") if isinstance(rollout.get("artifacts"), dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value)
    return "unknown"


def same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def input_valid(input_validation: dict[str, Any]) -> bool:
    if input_validation.get("status") != "pass":
        return False
    validation = input_validation.get("validation") if isinstance(input_validation.get("validation"), dict) else {}
    if validation and int(validation.get("valid_rows") or 0) <= 0:
        return False
    return True


def rollout_alignment(rollout: dict[str, Any] | None, input_data: Path | None, input_validation: dict[str, Any]) -> dict[str, Any]:
    output = rollout_output_path(rollout)
    status = rollout_status(rollout)
    aligned = same_path(output, input_data)
    validation_ok = input_valid(input_validation)
    return {
        "rollout_status": status,
        "rollout_output_path": str(output) if output else None,
        "input_data_path": str(input_data) if input_data else None,
        "output_matches_input": aligned,
        "input_validation_status": input_validation.get("status"),
        "input_valid": validation_ok,
        "valid_rows": ((input_validation.get("validation") or {}).get("valid_rows") if isinstance(input_validation.get("validation"), dict) else None),
        "proves_actual_rollout_corpus": status == "pass" and aligned and validation_ok,
    }


def commands(args: argparse.Namespace, rollout: dict[str, Any] | None) -> dict[str, str]:
    is_math = args.target_context == "math"
    default_log_name = "qwen3_235b_math_capture_smoke" if is_math else "qwen3_235b_swe_capture_smoke"
    default_output_name = "qwen3_235b_math_rollout_conversations.jsonl" if is_math else "qwen3_235b_swe_rollout_conversations.jsonl"
    capture_script = "run_math_rollout_capture_smoke.sh" if is_math else "run_rollout_capture_smoke.sh"
    rollout_log_dir = (
        Path(rollout["rollout_log_dir"])
        if rollout and rollout.get("rollout_log_dir")
        else args.artifact_root / f"rl_rollout_capture_logs/{default_log_name}"
    )
    rollout_output = rollout_output_path(rollout) or args.artifact_root / f"data/{default_output_name}"
    input_data = args.input_data or rollout_output
    return {
        "rollout_capture_dry_run": (
            f"DRY_RUN=true ARTIFACT_ROOT={args.artifact_root} "
            f"ROLLOUT_LOG_DIR={rollout_log_dir} OUTPUT_CONVERSATIONS={rollout_output} "
            f"EAGLE3_TARGET_CONTEXT={args.target_context} bash experiments/eagle3_qwen3_235b/{capture_script}"
        ),
        "rollout_capture_submit": (
            f"DRY_RUN=false ARTIFACT_ROOT={args.artifact_root} "
            f"ROLLOUT_LOG_DIR={rollout_log_dir} OUTPUT_CONVERSATIONS={rollout_output} "
            f"EAGLE3_TARGET_CONTEXT={args.target_context} bash experiments/eagle3_qwen3_235b/{capture_script}"
        ),
        "materialize_rollout": (
            f"ARTIFACT_ROOT={args.artifact_root} ROLLOUT_LOG_DIR={rollout_log_dir} "
            f"OUTPUT_DATA={rollout_output} bash experiments/eagle3_qwen3_235b/materialize_rollout_capture_corpus.sh"
        ),
        "validate_input": (
            f"python3 experiments/eagle3_qwen3_235b/validate_training_conversations.py {input_data} "
            f"--max-seq-len {args.max_seq_len}"
        ),
        "bootstrap_dry_run": (
            f"INPUT_DATA={input_data} ARTIFACT_ROOT={args.artifact_root} "
            f"EAGLE3_TARGET_CONTEXT={args.target_context} SUBMIT=false RUN_PILOT=true bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh"
        ),
        "refresh_operator_state": (
            f"ARTIFACT_ROOT={args.artifact_root} "
            "python3 experiments/eagle3_qwen3_235b/refresh_eagle3_operator_state.py"
        ),
    }


def decide(
    args: argparse.Namespace,
    rollout: dict[str, Any] | None,
    input_validation: dict[str, Any],
    math_paths: list[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    status = rollout_status(rollout)
    input_status = input_validation.get("status")
    visible_math = [item for item in math_paths if item.get("exists")]

    if args.target_context == "swe_rl":
        if status == "pass":
            if alignment["proves_actual_rollout_corpus"]:
                return {
                    "overall_status": "pass",
                    "primary_source": "actual_rl_rollout",
                    "detail": "materialized RL rollout corpus validates and matches the rollout analysis output path",
                    "next_action": "run bootstrap/pipeline dry-run with the materialized rollout conversation JSONL",
                    "provenance": alignment,
                }
            if not alignment["output_matches_input"]:
                return {
                    "overall_status": "fail",
                    "primary_source": "actual_rl_rollout",
                    "detail": "rollout analysis is PASS, but INPUT_DATA does not match the materialized rollout output path",
                    "next_action": "point INPUT_DATA at the rollout output or regenerate corpus reports for the selected file",
                    "provenance": alignment,
                }
            return {
                "overall_status": "fail",
                "primary_source": "actual_rl_rollout",
                "detail": "rollout analysis is PASS, but the selected conversation JSONL did not pass validation",
                "next_action": "rerun materialization and validate_training_conversations.py for the rollout output",
                "provenance": alignment,
            }
        if status == "needs_materialize":
            return {
                "overall_status": "needs_materialize",
                "primary_source": "actual_rl_rollout",
                "detail": "rollout train_data exists but has not been materialized into ModelOpt conversation JSONL",
                "next_action": "run materialize_rollout_capture_corpus.sh",
            }
        if status == "running":
            return {
                "overall_status": "missing_capture",
                "primary_source": "actual_rl_rollout",
                "detail": "rollout capture is submitted but still queued/running; wait for train_data artifacts before training",
                "next_action": "poll the active rollout capture watcher",
            }
        if input_status == "pass":
            return {
                "overall_status": "bootstrap_data_only",
                "primary_source": "existing_or_generated_conversations",
                "detail": "INPUT_DATA validates, but the report does not prove it came from the target RL rollout distribution",
                "next_action": "use only for smoke/bootstrap unless provenance confirms Qwen3 SWE/RL rollout origin",
                "provenance": alignment,
            }
        return {
            "overall_status": "missing_capture",
            "primary_source": "actual_rl_rollout",
            "detail": "SWE/RL target needs Qwen3 policy responses from the same rollout loop before final Eagle3 training",
            "next_action": "run a short rollout capture, then materialize and validate the conversation corpus",
            "provenance": alignment,
        }

    if args.target_context == "math":
        if input_status == "pass":
            return {
                "overall_status": "pass",
                "primary_source": "math_instruction_or_math_rollout",
                "detail": "INPUT_DATA validates for a math-target draft; confirm model/provenance before full hidden-state dump",
                "next_action": "run tokenizer/template validation and hidden-state pilot",
                "provenance": alignment,
            }
        if visible_math:
            return {
                "overall_status": "needs_materialize",
                "primary_source": "math_instruction_or_math_rollout",
                "detail": "math data paths are visible but still need conversion into ModelOpt conversation JSONL",
                "next_action": "convert math prompts/responses through MODE=existing or MODE=generate, then validate",
                "provenance": alignment,
            }
        return {
            "overall_status": "missing_math_corpus",
            "primary_source": "math_instruction_or_math_rollout",
            "detail": "math-target draft needs DAPO/OpenMathInstruct-style conversations or math rollouts",
            "next_action": "provide math data paths or generate Qwen3 Thinking math responses",
            "provenance": alignment,
        }

    if input_status == "pass":
        return {
            "overall_status": "pass",
            "primary_source": "validated_conversations",
            "detail": "INPUT_DATA validates for a general draft; domain match still controls acceptance rate",
            "next_action": "run hidden-state pilot and acceptance smoke",
            "provenance": alignment,
        }
    return {
        "overall_status": "missing_corpus",
        "primary_source": "target_domain_conversations",
        "detail": "general-target draft still needs validated conversations from the target generation distribution",
        "next_action": "prepare or generate ModelOpt conversation JSONL",
        "provenance": alignment,
    }


def render_markdown(data: dict[str, Any]) -> str:
    decision = data["decision"]
    lines = [
        "# Eagle3 Corpus Strategy",
        "",
        f"Overall: **{decision['overall_status'].upper()}**",
        f"Target context: `{data['target_context']}`",
        f"Primary source: `{decision['primary_source']}`",
        "",
        decision["detail"],
        "",
        "## Policy",
        "",
        "- Eagle3 draft training here is supervised hidden-state based training, not DAPO/GRPO reward training.",
        "- Train the draft on the same prompt, chat-template, and assistant-output distribution where it will be used.",
        "- DAPO/OpenMathInstruct-style data is primary for math workloads and supplemental for SWE/RL unless the target rollout is math.",
        "",
        "## Evidence",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| rollout status | {rollout_status(data.get('rollout_capture_analysis'))} |",
        f"| input data status | {data['input_data']['status']} |",
        f"| rollout output matches input | {data['rollout_alignment']['output_matches_input']} |",
        f"| input validation proves rows | {data['rollout_alignment']['input_valid']} |",
        f"| visible math paths | {sum(1 for item in data['math_data'] if item.get('exists'))} |",
        "",
        "## Next Command",
        "",
        "```bash",
    ]
    status = decision["overall_status"]
    cmd = data["commands"]["rollout_capture_dry_run"]
    if status == "needs_materialize":
        cmd = data["commands"]["materialize_rollout"]
    elif status in ("pass", "bootstrap_data_only"):
        cmd = data["commands"]["bootstrap_dry_run"]
    elif decision.get("next_action") == "poll the active rollout capture watcher":
        cmd = data["commands"]["refresh_operator_state"]
    elif status == "missing_math_corpus":
        cmd = "# provide --math-data paths or generate Qwen3 Thinking math responses, then rerun this report"
    lines += [cmd, "```", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rollout_path = args.rollout_capture_analysis_json or args.artifact_root / "reports/rollout_capture_analysis.json"
    rollout = load_json(rollout_path)
    input_data = args.input_data or args.artifact_root / "data/qwen3_235b_swe_rollout_conversations.jsonl"
    input_validation = validate_input(input_data, args.max_seq_len, args.sample_limit)
    math_paths = inspect_paths(args.math_data)
    alignment = rollout_alignment(rollout, input_data, input_validation)
    decision = decide(args, rollout, input_validation, math_paths, alignment)
    data = {
        "overall_status": decision["overall_status"],
        "target_context": args.target_context,
        "artifact_root": str(args.artifact_root),
        "rollout_capture_analysis_json": str(rollout_path),
        "rollout_capture_analysis": rollout,
        "rollout_alignment": alignment,
        "input_data": input_validation,
        "math_data": math_paths,
        "decision": decision,
        "commands": commands(args, rollout),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    text = render_markdown(data)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    print(text)
    return 1 if decision["overall_status"].endswith("fail") else 0


if __name__ == "__main__":
    raise SystemExit(main())
