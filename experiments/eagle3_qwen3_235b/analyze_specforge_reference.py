#!/usr/bin/env python3
"""Record how the SGLang SpecForge reference relates to this ModelOpt path."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/sgl-project/SpecForge/main/examples/"
    "run_qwen3_235b_a22b_eagle3.sh"
)
DEFAULT_REPO_URL = "https://github.com/sgl-project/SpecForge"
DEFAULT_DOCS_URL = "https://sgl-project.github.io/SpecForge/"
DEFAULT_DATA_PREP_URL = (
    "https://sgl-project.github.io/SpecForge/basic_usage/"
    "data_preparation.html#option-1-conversation-format"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-url", default=os.environ.get("SPECFORGE_SCRIPT_URL", DEFAULT_SCRIPT_URL))
    parser.add_argument("--repo-url", default=os.environ.get("SPECFORGE_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--docs-url", default=os.environ.get("SPECFORGE_DOCS_URL", DEFAULT_DOCS_URL))
    parser.add_argument(
        "--data-prep-url",
        default=os.environ.get("SPECFORGE_DATA_PREP_URL", DEFAULT_DATA_PREP_URL),
    )
    parser.add_argument("--local-script", type=Path)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def fetch_text(url: str, timeout: float = 20.0) -> tuple[str, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return raw, {"source": url, "status": "fetched"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", {"source": url, "status": "fetch_failed", "error": str(exc)}


def read_script(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.local_script:
        if not args.local_script.exists():
            return "", {"source": str(args.local_script), "status": "missing"}
        return args.local_script.read_text(encoding="utf-8", errors="replace"), {
            "source": str(args.local_script),
            "status": "local",
        }
    if args.no_fetch:
        return "", {"source": args.script_url, "status": "skipped"}
    return fetch_text(args.script_url)


def analyze_data_prep_doc(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_fetch:
        text = ""
        fetch = {"source": args.data_prep_url, "status": "skipped"}
    else:
        text, fetch = fetch_text(args.data_prep_url)
    normalized = " ".join(text.split())
    detected = {
        "conversation_format": all(part in normalized for part in ("conversations", "role", "content")),
        "uses_id_field": '"id"' in normalized or "{ \"id\"" in normalized,
        "preformatted_text_format": '"text"' in normalized and "--is-preformatted" in normalized,
        "regenerate_dataset_recommended": "regenerate" in normalized.lower()
        and "target model" in normalized.lower(),
    }
    return {
        "url": args.data_prep_url,
        "fetch": fetch,
        "detected": detected,
        "usable_schema": {
            "specforge_conversation": {"id": "string", "conversations": [{"role": "user|assistant", "content": "string"}]},
            "workspace_modelopt": {
                "conversation_id": "string",
                "messages": [{"role": "user|assistant", "content": "string"}],
            },
        },
        "implication": (
            "The same RL rollout messages can be emitted as ModelOpt conversation_id/messages "
            "or SpecForge id/conversations. Pre-formatted text needs a matching chat template "
            "and --is-preformatted in SpecForge, so it is a comparison path rather than the "
            "default ModelOpt hidden-state gate."
        ),
    }


def flag_value(text: str, flag: str) -> str | None:
    pattern = re.compile(rf"{re.escape(flag)}\s+([^\\\n ]+)")
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def analyze(text: str, fetch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    normalized = " ".join(text.split())
    target = flag_value(normalized, "--target-model-path")
    config = flag_value(normalized, "--draft-model-config")
    train_data = flag_value(normalized, "--train-data-path")
    backend = flag_value(normalized, "--target-model-backend")
    chat_template = flag_value(normalized, "--chat-template")
    tp_size = flag_value(normalized, "--tp-size")
    detected = {
        "uses_train_eagle3_py": "scripts/train_eagle3.py" in normalized,
        "target_model_path": target,
        "draft_model_config": config,
        "train_data_path": train_data,
        "target_model_backend": backend,
        "chat_template": chat_template,
        "tp_size": tp_size,
        "mentions_qwen3_235b_in_url": "qwen3_235b" in args.script_url.lower(),
        "targets_qwen3_next_80b": bool(target and "Qwen3-Next-80B" in target),
    }
    conclusions: list[str] = []
    if backend == "sglang":
        conclusions.append("SpecForge example targets the SGLang backend.")
    if detected["targets_qwen3_next_80b"]:
        conclusions.append("Current visible script content targets Qwen3-Next-80B-A3B, despite the 235B filename.")
    if detected["uses_train_eagle3_py"]:
        conclusions.append("The example is useful for EAGLE3 training-shape comparison: train_eagle3.py, qwen chat template, TP, and backend flags.")
    conclusions.append("Do not use this as a drop-in replacement for the current ModelOpt + vLLM/NeMo-RL path.")
    data_prep = analyze_data_prep_doc(args)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": "reference_only" if text else "missing_reference",
        "repo_url": args.repo_url,
        "docs_url": args.docs_url,
        "data_preparation": data_prep,
        "script_url": args.script_url,
        "fetch": fetch,
        "detected": detected,
        "conclusions": conclusions,
        "modelopt_path_implication": {
            "keep_primary_training_path": "ModelOpt Eagle3 wrappers in this repo",
            "keep_primary_serving_validation": "vLLM/NeMo-RL RL generation smoke and sweep",
            "use_specforge_for": "SGLang/SGLang-serving comparison and architecture/training-shape sanity checks",
        },
    }


def render_markdown(data: dict[str, Any]) -> str:
    detected = data["detected"]
    lines = [
        "# SpecForge / SGLang Reference",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        "",
        f"Repository: {data['repo_url']}",
        f"Docs: {data['docs_url']}",
        f"Script: {data['script_url']}",
        f"Data prep: {data['data_preparation']['url']}",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| fetch status | {data['fetch'].get('status')} |",
        f"| uses train_eagle3.py | {detected['uses_train_eagle3_py']} |",
        f"| target model path | `{detected.get('target_model_path')}` |",
        f"| draft config | `{detected.get('draft_model_config')}` |",
        f"| target backend | `{detected.get('target_model_backend')}` |",
        f"| chat template | `{detected.get('chat_template')}` |",
        f"| tp size | `{detected.get('tp_size')}` |",
        "",
        "## Data Format Notes",
        "",
        "- SpecForge conversation format uses `id` plus `conversations` with `role`/`content` messages.",
        "- SpecForge pre-formatted text format uses `id` plus `text` and requires `--is-preformatted` while still supplying a matching chat template for assistant-span loss masks.",
        "- This workspace's ModelOpt path keeps `conversation_id` plus `messages`; the converters can emit SpecForge format for SGLang comparison runs.",
        "",
        "## Conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in data["conclusions"])
    lines += [
        "",
        "## Implication For This Workspace",
        "",
        "- Keep the primary training path on ModelOpt wrappers.",
        "- Keep the primary serving validation on vLLM/NeMo-RL RL generation.",
        "- Use SpecForge only as an SGLang reference unless the serving target changes to SGLang.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    text, fetch = read_script(args)
    data = analyze(text, fetch, args)
    rendered = render_markdown(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(rendered)
    print(rendered)
    return 0 if data["overall_status"] == "reference_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
