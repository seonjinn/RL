#!/usr/bin/env python3
"""Discover candidate inputs for a Qwen3-235B Eagle3 run.

Run this on the machine where Lustre/model snapshots/NeMo-RL outputs are
mounted. It produces a ranked report and an optional bootstrap env file.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from normalize_rl_rollouts_to_conversations import extract_from_record
except Exception:  # pragma: no cover - import path depends on invocation style
    extract_from_record = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    ROOT,
    Path("/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso"),
    Path("/lustre/fsw/portfolios/coreai/users/sna"),
]
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".cache",
    ".venv",
    "node_modules",
    "wandb",
}


@dataclass
class FileBuckets:
    configs: list[Path] = field(default_factory=list)
    tokenizer_configs: list[Path] = field(default_factory=list)
    jsonl: list[Path] = field(default_factory=list)
    scanned_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[p for p in DEFAULT_ROOTS if p.exists()])
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "outputs/qwen3_235b_eagle3")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=20000)
    parser.add_argument("--sample-lines", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--env-out", type=Path)
    return parser.parse_args()


def depth_from(root: Path, path: Path) -> int:
    try:
        return len(path.resolve().relative_to(root.resolve()).parts)
    except Exception:
        return 999999


def collect_files(roots: Iterable[Path], max_depth: int, max_files: int) -> FileBuckets:
    buckets = FileBuckets()
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if buckets.scanned_files >= max_files or not path.is_file():
            return
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        buckets.scanned_files += 1
        name = path.name
        if name == "config.json":
            buckets.configs.append(path)
        elif name == "tokenizer_config.json":
            buckets.tokenizer_configs.append(path)
        elif path.suffix == ".jsonl":
            buckets.jsonl.append(path)

    for root in roots:
        if root.is_file():
            add(root)
            continue
        if not root.is_dir():
            continue
        for current_text, dirs, files in os.walk(root, onerror=lambda _: None):
            current = Path(current_text)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if depth_from(root, current) > max_depth:
                dirs[:] = []
                continue
            for filename in files:
                if filename in {"config.json", "tokenizer_config.json"} or filename.endswith(".jsonl"):
                    add(current / filename)
                    if buckets.scanned_files >= max_files:
                        return buckets
    return buckets


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def rope_theta(cfg: dict[str, Any]) -> Any:
    if cfg.get("rope_theta") is not None:
        return cfg.get("rope_theta")
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        return rope_scaling.get("rope_theta")
    return None


def score_verifier_config(path: Path, cfg: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    text = str(path).lower()
    model_type = str(cfg.get("model_type") or "").lower()
    if "qwen3" in text:
        score += 20
        reasons.append("path contains qwen3")
    if "235b" in text:
        score += 20
        reasons.append("path contains 235b")
    if "thinking" in text:
        score += 15
        reasons.append("path contains thinking")
    if model_type in {"qwen3", "qwen3_moe"}:
        score += 20
        reasons.append(f"model_type={model_type}")
    expected = {
        "num_hidden_layers": 94,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
    }
    for key, value in expected.items():
        if cfg.get(key) == value:
            score += 10
            reasons.append(f"{key}={value}")
    if rope_theta(cfg) == 5000000:
        score += 10
        reasons.append("rope_theta=5000000")
    if "architectures" in cfg:
        score += 2
    return score, reasons


def is_eagle_draft_config(cfg: dict[str, Any]) -> bool:
    return (
        cfg.get("speculators_model_type") == "eagle3"
        or cfg.get("architectures") == ["Eagle3Speculator"]
        or "eagle_aux_hidden_state_layer_ids" in cfg
        or isinstance(cfg.get("eagle_config"), dict)
        or isinstance(cfg.get("transformer_layer_config"), dict)
    )


def inspect_configs(configs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verifier: list[dict[str, Any]] = []
    drafts: list[dict[str, Any]] = []
    for path in configs:
        cfg = load_json(path)
        if not isinstance(cfg, dict):
            continue
        if is_eagle_draft_config(cfg):
            drafts.append(
                {
                    "path": str(path),
                    "parent": str(path.parent),
                    "model_type": cfg.get("model_type"),
                    "architectures": cfg.get("architectures"),
                    "speculators_model_type": cfg.get("speculators_model_type"),
                    "aux_layers": cfg.get("eagle_aux_hidden_state_layer_ids")
                    or (cfg.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids"),
                }
            )
            continue
        if all(key in cfg for key in ("hidden_size", "num_hidden_layers", "num_attention_heads")):
            score, reasons = score_verifier_config(path, cfg)
            verifier.append(
                {
                    "path": str(path),
                    "parent": str(path.parent),
                    "score": score,
                    "reasons": reasons,
                    "model_type": cfg.get("model_type"),
                    "num_hidden_layers": cfg.get("num_hidden_layers"),
                    "hidden_size": cfg.get("hidden_size"),
                    "num_attention_heads": cfg.get("num_attention_heads"),
                    "num_key_value_heads": cfg.get("num_key_value_heads"),
                    "intermediate_size": cfg.get("intermediate_size"),
                    "rope_theta": rope_theta(cfg),
                }
            )
    verifier.sort(key=lambda item: item["score"], reverse=True)
    return verifier, drafts


def inspect_tokenizers(paths: Iterable[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        cfg = load_json(path)
        if not isinstance(cfg, dict) or not isinstance(cfg.get("chat_template"), str):
            continue
        text = str(path).lower()
        score = 10
        reasons = ["has chat_template"]
        for token, points in (("qwen3", 15), ("235b", 15), ("thinking", 10)):
            if token in text:
                score += points
                reasons.append(f"path contains {token}")
        candidates.append(
            {
                "path": str(path),
                "parent": str(path.parent),
                "score": score,
                "reasons": reasons,
                "chat_template_chars": len(cfg["chat_template"]),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def inspect_jsonl(path: Path, sample_lines: int) -> dict[str, Any]:
    rows = 0
    json_rows = 0
    extracted = 0
    messages_rows = 0
    assistant_rows = 0
    invalid = 0
    key_counts: dict[str, int] = {}
    error = None
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                rows += 1
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    invalid += 1
                    if rows >= sample_lines:
                        break
                    continue
                if isinstance(record, dict):
                    json_rows += 1
                    for key in record:
                        key_counts[key] = key_counts.get(key, 0) + 1
                    messages = record.get("messages", record.get("conversations"))
                    if isinstance(messages, list):
                        messages_rows += 1
                        if any(isinstance(msg, dict) and str(msg.get("role", msg.get("from", ""))).lower() in {"assistant", "gpt", "bot"} for msg in messages):
                            assistant_rows += 1
                    if extract_from_record is not None:
                        extracted += len(
                            extract_from_record(
                                record,
                                path,
                                line_num,
                                None,
                                1,
                                False,
                                "<think>\n",
                                "\n</think>\n\n",
                            )
                        )
                if rows >= sample_lines:
                    break
    except Exception as exc:
        error = str(exc)
    return {
        "path": str(path),
        "rows_sampled": rows,
        "json_rows": json_rows,
        "messages_rows": messages_rows,
        "assistant_rows": assistant_rows,
        "extracted_rollout_conversations": extracted,
        "invalid_json": invalid,
        "key_counts": dict(sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))[:16]),
        "sample_error": error,
    }


def score_jsonl(item: dict[str, Any]) -> tuple[int, float]:
    positives = max(item["extracted_rollout_conversations"], item["assistant_rows"])
    ratio = positives / item["json_rows"] if item["json_rows"] else 0.0
    return positives, ratio


def inspect_jsonls(paths: Iterable[Path], sample_lines: int, top_k: int) -> list[dict[str, Any]]:
    inspected = [inspect_jsonl(path, sample_lines) for path in paths]
    positives = [item for item in inspected if max(item["extracted_rollout_conversations"], item["assistant_rows"]) > 0]
    positives.sort(key=score_jsonl, reverse=True)
    return positives[:top_k]


def quote(value: str) -> str:
    return shlex.quote(value)


def render_env(payload: dict[str, Any], artifact_root: Path) -> str:
    verifier = payload["verifier_candidates"][0] if payload["verifier_candidates"] else None
    tokenizer = payload["tokenizer_candidates"][0] if payload["tokenizer_candidates"] else None
    conversation = payload["conversation_candidates"][0] if payload["conversation_candidates"] else None
    lines = [
        "# shellcheck shell=bash",
        "# Source this before bootstrap_eagle3_path.sh, then inspect before SUBMIT=true.",
        f"ARTIFACT_ROOT={quote(str(artifact_root))}",
    ]
    if verifier:
        lines.append(f"VERIFIER_CONFIG_DIR={quote(verifier['parent'])}")
    if tokenizer:
        lines.append(f"TOKENIZER_CONFIG={quote(tokenizer['path'])}")
    if conversation:
        lines += [
            "MODE=rollout",
            f"INPUT_PATHS={quote(conversation['path'])}",
        ]
    lines += [
        f"REFERENCE_ARCH={quote(str(artifact_root / 'architecture/eagle3_architecture.json'))}",
        f"ARCH_ENV_FILE={quote(str(artifact_root / 'architecture/eagle3_architecture.env'))}",
        f"CHAT_TEMPLATE={quote(str(artifact_root / 'templates/qwen3_generation_template.jinja2'))}",
        f"INPUT_DATA={quote(str(artifact_root / 'data/qwen3_235b_swe_rollout_conversations.jsonl'))}",
        f"HIDDEN_STATES_DIR={quote(str(artifact_root / 'hidden_states'))}",
        f"OUTPUT_DIR={quote(str(artifact_root / 'modelopt_ckpt'))}",
        f"EXPORT_DIR={quote(str(artifact_root / 'exported_hf'))}",
        f"VLLM_DRAFT_DIR={quote(str(artifact_root / 'vllm_draft'))}",
    ]
    return "\n".join(lines) + "\n"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 Eagle3 Run Input Discovery",
        "",
        f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        f"Roots: `{payload['roots']}`",
        f"Files scanned: **{payload['files_scanned']}**",
        "",
        "## Verifier Config Candidates",
        "",
        "| rank | score | layers | hidden | heads | kv | rope | config |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, item in enumerate(payload["verifier_candidates"], 1):
        lines.append(
            f"| {idx} | {item['score']} | {item.get('num_hidden_layers')} | {item.get('hidden_size')} | "
            f"{item.get('num_attention_heads')} | {item.get('num_key_value_heads')} | {item.get('rope_theta')} | "
            f"`{item['path']}` |"
        )
    lines += ["", "## Tokenizer Config Candidates", "", "| rank | score | template chars | path |", "| ---: | ---: | ---: | --- |"]
    for idx, item in enumerate(payload["tokenizer_candidates"], 1):
        lines.append(f"| {idx} | {item['score']} | {item['chat_template_chars']} | `{item['path']}` |")
    lines += ["", "## Conversation/rollout JSONL Candidates", "", "| rank | assistant rows | extracted rollouts | sampled | path |", "| ---: | ---: | ---: | ---: | --- |"]
    for idx, item in enumerate(payload["conversation_candidates"], 1):
        lines.append(
            f"| {idx} | {item['assistant_rows']} | {item['extracted_rollout_conversations']} | "
            f"{item['rows_sampled']} | `{item['path']}` |"
        )
    lines += ["", "## Eagle3 Draft Config Candidates", "", "| rank | model type | arch | aux | config |", "| ---: | --- | --- | --- | --- |"]
    for idx, item in enumerate(payload["draft_candidates"], 1):
        lines.append(
            f"| {idx} | {item.get('model_type')} | `{item.get('architectures')}` | "
            f"`{item.get('aux_layers')}` | `{item['path']}` |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    if args.env_out:
        args.env_out.parent.mkdir(parents=True, exist_ok=True)
        args.env_out.write_text(render_env(payload, args.artifact_root))
    print(markdown, end="")


def main() -> int:
    args = parse_args()
    if not args.roots:
        raise SystemExit("No roots provided and no default roots exist on this host.")
    buckets = collect_files(args.roots, args.max_depth, args.max_files)
    verifier, drafts = inspect_configs(buckets.configs)
    tokenizers = inspect_tokenizers(buckets.tokenizer_configs)
    conversations = inspect_jsonls(buckets.jsonl, args.sample_lines, args.top_k)
    payload = {
        "roots": [str(root) for root in args.roots],
        "artifact_root": str(args.artifact_root),
        "files_scanned": buckets.scanned_files,
        "config_files": len(buckets.configs),
        "tokenizer_config_files": len(buckets.tokenizer_configs),
        "jsonl_files": len(buckets.jsonl),
        "verifier_candidates": verifier[: args.top_k],
        "tokenizer_candidates": tokenizers[: args.top_k],
        "conversation_candidates": conversations,
        "draft_candidates": drafts[: args.top_k],
    }
    write_outputs(payload, args)
    if not verifier:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
