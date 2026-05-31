#!/usr/bin/env python3
"""Estimate Qwen3-235B Eagle3 draft-training scale and run stages.

This is a planning report, not a GPU benchmark. It turns the current corpus
state and ModelOpt wrapper defaults into concrete pilot, calibration, and
production-candidate step counts so the next operator can decide how much
offline Eagle3 training to run before spending cluster time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARTIFACT_ROOT = Path("/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3")
DEFAULT_SCENARIOS = (8, 1_000, 2_438, 5_000, 10_000, 50_000, 100_000, 300_000, 500_000)
VALID_ROLES = {"system", "user", "assistant", "tool", "function"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)))
    parser.add_argument(
        "--input-data",
        type=Path,
        default=Path(os.environ["INPUT_DATA"]) if os.environ.get("INPUT_DATA") else None,
        help="ModelOpt/SpecForge conversation JSONL. Missing input still produces a planning report.",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=Path(os.environ["CONVERSATION_VALIDATION_JSON"]) if os.environ.get("CONVERSATION_VALIDATION_JSON") else None,
        help="Optional validate_training_conversations.py JSON output to reuse.",
    )
    parser.add_argument(
        "--corpus-strategy-json",
        type=Path,
        default=Path(os.environ["CORPUS_STRATEGY_JSON"]) if os.environ.get("CORPUS_STRATEGY_JSON") else None,
    )
    parser.add_argument(
        "--pipeline-submit-preflight-json",
        type=Path,
        default=Path(os.environ["PIPELINE_SUBMIT_PREFLIGHT_JSON"]) if os.environ.get("PIPELINE_SUBMIT_PREFLIGHT_JSON") else None,
    )
    parser.add_argument("--target-context", default=os.environ.get("EAGLE3_TARGET_CONTEXT", "swe_rl"))
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=16_384)
    parser.add_argument("--approx-chars-per-token", type=float, default=4.0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-hidden-copies", type=int, default=4)
    parser.add_argument("--bytes-per-value", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("TRAIN_GPUS_PER_NODE", "8")))
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")),
    )
    parser.add_argument("--epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    parser.add_argument("--pilot-examples", type=int, default=int(os.environ.get("DATA_SAMPLE_SIZE", "8") or "8"))
    parser.add_argument("--pilot-max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "20") or "20"))
    parser.add_argument("--smoke-examples", type=int, default=int(os.environ.get("SMOKE_EXAMPLES", "5") or "5"))
    parser.add_argument(
        "--first-calibration-examples",
        type=int,
        default=int(os.environ.get("FIRST_CALIBRATION_EXAMPLES", "2438") or "2438"),
        help="Expected first full SWE-Gym calibration size when no rollout corpus exists yet.",
    )
    parser.add_argument("--target-calibration-min-examples", type=int, default=10_000)
    parser.add_argument("--target-calibration-max-examples", type=int, default=50_000)
    parser.add_argument("--generic-min-examples", type=int, default=300_000)
    parser.add_argument("--generic-preferred-examples", type=int, default=500_000)
    parser.add_argument(
        "--full-rollout-materialization-json",
        type=Path,
        default=None,
        help="Optional SWE-Gym materialization report; defaults to reports/swegym_hf_materialize_full.json.",
    )
    parser.add_argument(
        "--scenario-examples",
        type=int,
        action="append",
        default=None,
        help="Example count scenario. Repeat to override defaults.",
    )
    parser.add_argument("--min-production-examples", type=int, default=50_000)
    parser.add_argument("--preferred-production-examples", type=int, default=100_000)
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


def input_files(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    if path.is_file():
        return [path]
    return []


def iter_jsonl(files: Iterable[Path], sample_limit: int):
    sampled = 0
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                sampled += 1
                if sampled > sample_limit:
                    return
                try:
                    yield path, line_num, json.loads(text), None
                except json.JSONDecodeError as exc:
                    yield path, line_num, None, str(exc)


def count_jsonl_rows(files: Iterable[Path]) -> int:
    rows = 0
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as fh:
            rows += sum(1 for line in fh if line.strip())
    return rows


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    raw_messages = record.get("messages")
    schema = "modelopt"
    if raw_messages is None:
        raw_messages = record.get("conversations")
        schema = "specforge_conversation"
    if not isinstance(raw_messages, list):
        return [], "unknown"

    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", raw.get("from"))
        content = raw.get("content", raw.get("value", raw.get("text")))
        if role is None or content in (None, ""):
            continue
        role_text = str(role).lower()
        role_text = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role_text, role_text)
        if role_text not in VALID_ROLES:
            continue
        text = content_text(content)
        if text.strip():
            messages.append({"role": role_text, "content": text})
    return messages, schema


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def gib(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def estimate_tokens(messages: list[dict[str, str]], chars_per_token: float) -> int:
    text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    return max(1, math.ceil(len(text) / chars_per_token))


def inspect_corpus(args: argparse.Namespace, validation: dict[str, Any] | None) -> dict[str, Any]:
    files = input_files(args.input_data)
    if not files:
        return {
            "status": "missing",
            "path": str(args.input_data) if args.input_data else None,
            "files": [],
            "total_rows": 0,
            "sampled_rows": 0,
            "valid_sample_rows": 0,
            "token_estimate_source": "default_assumption",
            "estimated_tokens": {"avg": args.max_seq_len // 3, "p50": args.max_seq_len // 3, "p95": args.max_seq_len, "max": 0},
            "assistant_chars": {"avg": 0, "p50": 0, "p95": 0, "max": 0},
            "schema_counts": {},
            "warning": "input corpus is not visible; estimates use conservative default token assumptions",
        }

    total_rows = count_jsonl_rows(files)
    token_counts: list[int] = []
    assistant_chars: list[int] = []
    schema_counts: dict[str, int] = {}
    invalid_json = 0
    empty_or_invalid_messages = 0

    for _, _, record, error in iter_jsonl(files, args.sample_limit):
        if error:
            invalid_json += 1
            continue
        if not isinstance(record, dict):
            empty_or_invalid_messages += 1
            continue
        messages, schema = normalize_messages(record)
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        if not messages:
            empty_or_invalid_messages += 1
            continue
        token_counts.append(estimate_tokens(messages, args.approx_chars_per_token))
        assistant_chars.append(sum(len(item["content"].strip()) for item in messages if item["role"] == "assistant"))

    avg_tokens = int(round(sum(token_counts) / len(token_counts))) if token_counts else args.max_seq_len // 3
    avg_assistant = int(round(sum(assistant_chars) / len(assistant_chars))) if assistant_chars else 0
    validation_status = None
    if validation:
        validation_status = "parse_error" if validation.get("parse_error") else (
            "pass" if validation.get("failure_count", 1) == 0 else "fail"
        )
    status = "pass" if token_counts and invalid_json == 0 and empty_or_invalid_messages == 0 else "warn"
    if validation_status == "fail":
        status = "warn"
    return {
        "status": status,
        "path": str(args.input_data),
        "files": [str(path) for path in files],
        "total_rows": total_rows,
        "sampled_rows": min(total_rows, args.sample_limit),
        "valid_sample_rows": len(token_counts),
        "invalid_json": invalid_json,
        "empty_or_invalid_messages": empty_or_invalid_messages,
        "validation_status": validation_status,
        "validation_json": str(args.validation_json) if args.validation_json else None,
        "token_estimate_source": "sampled_chars_estimate",
        "estimated_tokens": {
            "avg": avg_tokens,
            "p50": percentile(token_counts, 0.50),
            "p95": percentile(token_counts, 0.95),
            "max": max(token_counts) if token_counts else 0,
        },
        "assistant_chars": {
            "avg": avg_assistant,
            "p50": percentile(assistant_chars, 0.50),
            "p95": percentile(assistant_chars, 0.95),
            "max": max(assistant_chars) if assistant_chars else 0,
        },
        "schema_counts": schema_counts,
    }


def effective_batch(args: argparse.Namespace) -> int:
    return max(1, args.gpus * args.per_device_train_batch_size * args.gradient_accumulation_steps)


def storage_gib(examples: int, tokens_per_example: int, args: argparse.Namespace) -> float:
    total_bytes = examples * tokens_per_example * args.hidden_size * args.num_hidden_copies * args.bytes_per_value
    return round(gib(total_bytes), 2)


def rollout_input_plan(args: argparse.Namespace, materialization: dict[str, Any] | None) -> dict[str, Any]:
    if materialization and materialization.get("overall_status") == "pass":
        rows = int(materialization.get("rows_written") or materialization.get("rows_seen") or 0)
        if rows > 0:
            return {
                "status": "available",
                "planned_first_calibration_examples": rows,
                "source": "swegym_hf_materialize_full",
                "path": materialization.get("output_jsonl"),
            }
    return {
        "status": "planned",
        "planned_first_calibration_examples": args.first_calibration_examples,
        "source": "default_swegym_train_split_expectation",
        "path": None,
    }


def stage_row(
    *,
    name: str,
    purpose: str,
    examples: int,
    max_steps: int | None,
    gate: str,
    global_batch: int,
    avg_tokens: int,
    args: argparse.Namespace,
    examples_display: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "purpose": purpose,
        "examples": examples,
        "examples_display": examples_display or str(examples),
        "max_steps": max_steps,
        "nominal_epoch_steps": math.ceil(max(1, examples) / global_batch),
        "hidden_state_storage_gib_avg_tokens": storage_gib(examples, avg_tokens, args),
        "gate": gate,
    }


def stage_plan(args: argparse.Namespace, corpus: dict[str, Any], rollout_plan: dict[str, Any]) -> list[dict[str, Any]]:
    total_rows = int(corpus.get("total_rows") or 0)
    scenarios = args.scenario_examples or list(DEFAULT_SCENARIOS)
    planned_first_calibration = int(rollout_plan.get("planned_first_calibration_examples") or args.first_calibration_examples)
    if total_rows and total_rows not in scenarios:
        scenarios = sorted(set(scenarios + [total_rows]))
    if planned_first_calibration and planned_first_calibration not in scenarios:
        scenarios = sorted(set(scenarios + [planned_first_calibration]))

    global_batch = effective_batch(args)
    avg_tokens = int(corpus["estimated_tokens"]["avg"])
    p95_tokens = int(corpus["estimated_tokens"]["p95"] or args.max_seq_len)
    first_calibration_examples = total_rows if total_rows else planned_first_calibration

    stages: list[dict[str, Any]] = [
        stage_row(
            name="smoke",
            purpose="runtime/capture proof only; do not train a serious draft on this",
            examples=args.smoke_examples,
            max_steps=0,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="train_data_step artifacts appear and normalize into valid conversations",
        ),
        stage_row(
            name="pilot",
            purpose="wiring only: hidden-state dump, loss mask, ModelOpt train, export, vLLM load",
            examples=args.pilot_examples,
            max_steps=args.pilot_max_steps,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="all pipeline stages complete; no quality claim",
        ),
        stage_row(
            name="swegym_first_calibration",
            purpose="first acceptance/speed direction on the materialized SWE-Gym train split",
            examples=first_calibration_examples,
            max_steps=1_000,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="trained draft loads in NeMo-RL and improves exposed_generation without reward/malformed regressions",
        ),
        stage_row(
            name="target_domain_calibration",
            purpose="larger target-domain SWE/RL calibration if the 2.4k run improves acceptance",
            examples=args.target_calibration_max_examples,
            examples_display=f"{args.target_calibration_min_examples}-{args.target_calibration_max_examples}",
            max_steps=2_000,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="acceptance, speed, reward, and malformed-output metrics all move in the right direction",
        ),
        stage_row(
            name="production_candidate",
            purpose="first serious SWE/RL draft candidate for longer runs",
            examples=args.preferred_production_examples,
            examples_display=f"{args.min_production_examples}-{args.preferred_production_examples}+",
            max_steps=None,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="num_spec_tokens sweep selects k and speedup is stable over multi-step RL smoke",
        ),
        stage_row(
            name="generic_optional",
            purpose="broad reusable Qwen3-235B draft outside the SWE/RL target",
            examples=args.generic_preferred_examples,
            examples_display=f"{args.generic_min_examples}-{args.generic_preferred_examples}",
            max_steps=None,
            global_batch=global_batch,
            avg_tokens=avg_tokens,
            args=args,
            gate="only pursue if the target changes from SWE/RL acceleration to general-purpose serving",
        ),
    ]

    scenario_rows = []
    for examples in sorted(set(scenarios)):
        scenario_rows.append(
            {
                "examples": examples,
                "steps_per_epoch": math.ceil(examples / global_batch),
                "steps_for_epochs": math.ceil(examples * args.epochs / global_batch),
                "hidden_state_storage_gib_avg_tokens": storage_gib(examples, avg_tokens, args),
                "hidden_state_storage_gib_p95_tokens": storage_gib(examples, p95_tokens, args),
            }
        )
    return stages + [{"name": "scenarios", "rows": scenario_rows}]


def recommendation(args: argparse.Namespace, corpus: dict[str, Any], corpus_strategy: dict[str, Any] | None, pipeline_preflight: dict[str, Any] | None) -> dict[str, Any]:
    strategy_status = (corpus_strategy or {}).get("overall_status")
    pipeline_ready = bool((pipeline_preflight or {}).get("submit_ready"))
    total_rows = int(corpus.get("total_rows") or 0)

    if corpus["status"] == "missing" or strategy_status in {"missing_capture", "missing_corpus", None}:
        return {
            "status": "needs_rollout_corpus",
            "summary": "capture/materialize actual Qwen3 SWE/RL rollout conversations before non-pilot Eagle3 training",
            "next_command": (
                f"DRY_RUN=true ARTIFACT_ROOT={args.artifact_root} "
                "bash experiments/eagle3_qwen3_235b/run_rollout_capture_smoke.sh"
            ),
        }
    if total_rows < args.pilot_examples:
        return {
            "status": "needs_more_data",
            "summary": f"visible corpus has {total_rows} rows, below pilot size {args.pilot_examples}",
            "next_command": "materialize additional rollout conversations, then rerun this report",
        }
    if total_rows < args.min_production_examples:
        return {
            "status": "pilot_or_calibration_only",
            "summary": (
                f"visible corpus has {total_rows} rows; use for pilot/calibration, not final production draft "
                f"until at least {args.min_production_examples} target-domain responses exist"
            ),
            "next_command": (
                f"INPUT_DATA={args.input_data} ARTIFACT_ROOT={args.artifact_root} "
                "SUBMIT=false RUN_PILOT=true bash experiments/eagle3_qwen3_235b/bootstrap_eagle3_path.sh"
            ),
        }
    if not pipeline_ready:
        return {
            "status": "preflight_before_submit",
            "summary": "corpus scale is plausible, but pipeline submit preflight has not proven submit_ready=true",
            "next_command": (
                f"INPUT_DATA={args.input_data} ARTIFACT_ROOT={args.artifact_root} "
                "python3 experiments/eagle3_qwen3_235b/preflight_eagle3_pipeline_submit.py"
            ),
        }
    return {
        "status": "ready_for_offline_train_submit",
        "summary": "corpus scale and submit preflight are ready for the fixed-draft offline pipeline",
        "next_command": (
            f"INPUT_DATA={args.input_data} ARTIFACT_ROOT={args.artifact_root} "
            "SUBMIT=true RUN_PILOT=true bash experiments/eagle3_qwen3_235b/run_eagle3_cluster_pilot.sh"
        ),
    }


def render_markdown(data: dict[str, Any]) -> str:
    corpus = data["corpus"]
    rec = data["recommendation"]
    train = data["training_defaults"]
    scenario = next(item for item in data["stage_plan"] if item["name"] == "scenarios")
    rollout = data["rollout_input_plan"]
    lines = [
        "# Eagle3 Training Scale Plan",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"Target context: `{data['target_context']}`",
        f"Recommendation: {rec['summary']}",
        "",
        "## Training Defaults",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| GPUs | {train['gpus']} |",
        f"| per-device batch | {train['per_device_train_batch_size']} |",
        f"| grad accumulation | {train['gradient_accumulation_steps']} |",
        f"| effective global batch | {train['effective_global_batch']} |",
        f"| epochs | {train['epochs']} |",
        f"| max sequence length | {train['max_seq_len']} |",
        "",
        "## Corpus Signal",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| status | {corpus['status']} |",
        f"| path | `{corpus.get('path')}` |",
        f"| total rows | {corpus['total_rows']} |",
        f"| sampled rows | {corpus['sampled_rows']} |",
        f"| avg estimated tokens | {corpus['estimated_tokens']['avg']} |",
        f"| p95 estimated tokens | {corpus['estimated_tokens']['p95']} |",
        "",
        "## Rollout Input Plan",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| status | {rollout['status']} |",
        f"| source | {rollout['source']} |",
        f"| planned first calibration examples | {rollout['planned_first_calibration_examples']} |",
        f"| source path | `{rollout.get('path')}` |",
        "",
        "## Stage Plan",
        "",
        "| stage | examples | max steps | nominal epoch steps | purpose | gate |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in data["stage_plan"]:
        if item["name"] == "scenarios":
            continue
        max_steps = item["max_steps"] if item["max_steps"] is not None else "epoch"
        lines.append(
            f"| {item['name']} | {item.get('examples_display', item['examples'])} | {max_steps} | "
            f"{item['nominal_epoch_steps']} | {item['purpose']} | {item['gate']} |"
        )
    lines += [
        "",
        "## Scenario Estimates",
        "",
        "| examples | steps/epoch | steps for epochs | hidden storage avg-token GiB | hidden storage p95-token GiB |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario["rows"]:
        lines.append(
            f"| {row['examples']} | {row['steps_per_epoch']} | {row['steps_for_epochs']} | "
            f"{row['hidden_state_storage_gib_avg_tokens']} | {row['hidden_state_storage_gib_p95_tokens']} |"
        )
    lines += [
        "",
        "## Public Sample-Size Anchors",
        "",
        "- vLLM Speculators Qwen3-8B online tutorial: 5k samples is a demo/getting-started scale.",
        "- Original EAGLE: about 68k ShareGPT dialogue iterations; 70B training reported at 1-2 days on 4x A100 40G.",
        "- Baseten EAGLE-3 guide: about 100k samples for specialized large-model task/format drafters, about 500k for large generic drafters, 1k-2k tokens/sample.",
        "- EAGLE-3 / NVIDIA Qwen3-235B-A22B-Eagle3 public artifacts: roughly 500k regenerated/synthetic samples for broad general-purpose modules.",
        "- TorchSpec Kimi K2.5 EAGLE-3: 600k samples / 6B tokens / 1500 H200 GPU hours is frontier-scale, not the first SWE/RL target.",
        "",
        "## Gates",
        "",
        "- Pilot proves the pipeline only; it is not acceptance-rate evidence.",
        "- Calibration should run a trained-draft smoke pair and `num_spec_tokens` sweep for k=2,3,4.",
        "- Production candidate should use target-domain SWE/RL rollout responses, not math-only DAPO/OpenMathInstruct data.",
        "- Stop or change data if acceptance does not translate into lower `exposed_generation` or if reward/malformed metrics regress.",
        "",
        "## Next Command",
        "",
        "```bash",
        rec["next_command"],
        "```",
        "",
    ]
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    validation = load_json(args.validation_json)
    corpus_strategy = load_json(args.corpus_strategy_json or args.artifact_root / "reports/corpus_strategy.json")
    pipeline_preflight = load_json(
        args.pipeline_submit_preflight_json or args.artifact_root / "reports/eagle3_pipeline_submit_preflight.json"
    )
    full_rollout_materialization_json = args.full_rollout_materialization_json or args.artifact_root / "reports/swegym_hf_materialize_full.json"
    full_rollout_materialization = load_json(full_rollout_materialization_json)
    corpus = inspect_corpus(args, validation)
    rollout_plan = rollout_input_plan(args, full_rollout_materialization)
    plan = stage_plan(args, corpus, rollout_plan)
    rec = recommendation(args, corpus, corpus_strategy, pipeline_preflight)
    status = "pass" if rec["status"] == "ready_for_offline_train_submit" else "planning"
    if rec["status"] in {"needs_rollout_corpus", "needs_more_data"}:
        status = "incomplete"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": status,
        "target_context": args.target_context,
        "artifact_root": str(args.artifact_root),
        "training_defaults": {
            "gpus": args.gpus,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": effective_batch(args),
            "epochs": args.epochs,
            "max_seq_len": args.max_seq_len,
            "hidden_size": args.hidden_size,
            "num_hidden_copies": args.num_hidden_copies,
            "bytes_per_value": args.bytes_per_value,
        },
        "corpus": corpus,
        "rollout_input_plan": {
            **rollout_plan,
            "materialization_json": str(full_rollout_materialization_json),
            "materialization_status": (full_rollout_materialization or {}).get("overall_status"),
        },
        "corpus_strategy": {
            "path": str(args.corpus_strategy_json or args.artifact_root / "reports/corpus_strategy.json"),
            "overall_status": (corpus_strategy or {}).get("overall_status"),
            "decision": (corpus_strategy or {}).get("decision"),
        },
        "pipeline_submit_preflight": {
            "path": str(args.pipeline_submit_preflight_json or args.artifact_root / "reports/eagle3_pipeline_submit_preflight.json"),
            "overall_status": (pipeline_preflight or {}).get("overall_status"),
            "submit_ready": (pipeline_preflight or {}).get("submit_ready"),
        },
        "stage_plan": plan,
        "recommendation": rec,
    }


def main() -> int:
    args = parse_args()
    data = build_payload(args)
    text = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
