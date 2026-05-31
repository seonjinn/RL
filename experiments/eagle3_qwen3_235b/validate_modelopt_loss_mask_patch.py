#!/usr/bin/env python3
"""Validate the ModelOpt TRT-LLM hidden-state loss-mask patch.

Qwen3 SWE/RL Eagle3 training uses answer-only offline loss. For the TRT-LLM
hidden-state path, that requires the dumper to compute the assistant-token mask
from the same tokenizer call as `input_ids` and save it as `loss_mask` in each
`.pt` file. This validator is static plus wrapper-dry-run only; it does not load
TensorRT-LLM or submit GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELOPT = ROOT / "Model-Optimizer"
DEFAULT_WRAPPER = ROOT / "experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh"

COMMON_REL = "examples/speculative_decoding/collect_hidden_states/common.py"
TRTLLM_REL = "examples/speculative_decoding/collect_hidden_states/compute_hidden_states_trtllm.py"

COMMON_SNIPPETS = {
    "answer_only_arg": "def add_answer_only_loss_args(",
    "chat_template_loader": "def load_chat_template(",
    "generation_tag_verifier": "def verify_generation_tags(",
    "tokenize_with_loss_mask": "def tokenize_with_loss_mask(",
    "assistant_mask_request": "return_assistant_tokens_mask=answer_only_loss",
    "assistant_masks_read": 'out["assistant_masks"]',
    "shape_guard": "assistant_masks length",
}

TRTLLM_SNIPPETS = {
    "imports_answer_only_arg": "add_answer_only_loss_args",
    "imports_tokenize_with_loss_mask": "tokenize_with_loss_mask",
    "registers_answer_only_arg": "add_answer_only_loss_args(parser)",
    "loads_chat_template": "load_chat_template(args.chat_template)",
    "verifies_generation_tags": "verify_generation_tags(tokenizer.chat_template)",
    "postprocess_receives_loss_mask": "loss_mask: torch.Tensor",
    "writes_loss_mask": 'trtllm_dumped["loss_mask"] = loss_mask.cpu()',
    "tokenizes_with_loss_mask": "tokenize_with_loss_mask(",
    "passes_loss_mask_to_dump": "dump_hidden_states(idx, conversation_id, input_ids, loss_mask)",
}

WRAPPER_SNIPPETS = {
    "answer_only_default": 'ANSWER_ONLY_LOSS="${ANSWER_ONLY_LOSS:-true}"',
    "passes_answer_only_loss": "cmd+=(--answer-only-loss)",
    "passes_chat_template": 'cmd+=(--chat-template "$CHAT_TEMPLATE")',
    "trtllm_default": 'BACKEND="${BACKEND:-trtllm}"',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelopt-dir", type=Path, default=Path(os.environ.get("MODELOPT_DIR", DEFAULT_MODELOPT)))
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def read_text(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "", f"not visible: {path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return "", f"cannot read {path}: {exc}"


def run(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "cmd": cmd}


def snippet_check(label: str, text: str, snippets: dict[str, str], error: str | None) -> dict[str, Any]:
    if error:
        return {
            "label": label,
            "status": "fail",
            "error": error,
            "checks": {name: False for name in snippets},
            "missing": list(snippets),
        }
    checks = {name: snippet in text for name, snippet in snippets.items()}
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "label": label,
        "status": "pass" if not missing else "fail",
        "checks": checks,
        "missing": missing,
    }


def py_compile(paths: list[Path]) -> dict[str, Any]:
    existing = [str(path) for path in paths if path.exists()]
    if not existing:
        return {"status": "fail", "returncode": 1, "stdout": "no files exist"}
    result = run(["python3", "-m", "py_compile", *existing])
    return {
        "status": "pass" if result["returncode"] == 0 else "fail",
        "returncode": result["returncode"],
        "stdout": result["stdout"][-4000:],
        "files": existing,
    }


def wrapper_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "DRY_RUN": "true",
        "MODELOPT_DIR": str(args.modelopt_dir),
        "BACKEND": "trtllm",
        "BASE_MODEL": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "INPUT_DATA": "/tmp/qwen3_swe_conversations.jsonl",
        "HIDDEN_STATES_DIR": "/tmp/qwen3_eagle3_hiddens",
        "CHAT_TEMPLATE": "/tmp/qwen3_generation_template.jinja2",
        "ANSWER_ONLY_LOSS": "true",
        "TP": "8",
    }
    result = run(["bash", str(args.wrapper)], env=env)
    stdout = result["stdout"]
    required = {
        "uses_trtllm_script": "compute_hidden_states_trtllm.py" in stdout,
        "passes_answer_only_loss": "--answer-only-loss" in stdout,
        "passes_chat_template": "--chat-template" in stdout and "/tmp/qwen3_generation_template.jinja2" in stdout,
        "passes_tp": "--tp 8" in stdout or "--tp' 8" in stdout or "--tp\\ 8" in stdout,
    }
    missing = [name for name, ok in required.items() if not ok]
    return {
        "status": "pass" if result["returncode"] == 0 and not missing else "fail",
        "returncode": result["returncode"],
        "checks": required,
        "missing": missing,
        "stdout_tail": stdout[-4000:],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    common_path = args.modelopt_dir / COMMON_REL
    trtllm_path = args.modelopt_dir / TRTLLM_REL
    common_text, common_error = read_text(common_path)
    trtllm_text, trtllm_error = read_text(trtllm_path)
    wrapper_text, wrapper_error = read_text(args.wrapper)
    checks = {
        "common_helpers": snippet_check("common helpers", common_text, COMMON_SNIPPETS, common_error),
        "trtllm_dumper": snippet_check("TRT-LLM dumper", trtllm_text, TRTLLM_SNIPPETS, trtllm_error),
        "wrapper": snippet_check("Qwen3 dump wrapper", wrapper_text, WRAPPER_SNIPPETS, wrapper_error),
        "syntax": py_compile([common_path, trtllm_path]),
        "wrapper_dry_run": wrapper_dry_run(args),
    }
    overall = "pass" if all(item.get("status") == "pass" for item in checks.values()) else "fail"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "overall_status": overall,
        "modelopt_dir": str(args.modelopt_dir),
        "wrapper": str(args.wrapper),
        "paths": {
            "common": str(common_path),
            "trtllm": str(trtllm_path),
        },
        "checks": checks,
        "recommendation": (
            "Proceed with answer-only TRT-LLM hidden-state preflight."
            if overall == "pass"
            else "Patch ModelOpt or use the staged patched checkout before hidden-state dump."
        ),
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# ModelOpt Loss-Mask Patch Validation",
        "",
        f"Overall: **{data['overall_status'].upper()}**",
        f"ModelOpt: `{data['modelopt_dir']}`",
        f"Wrapper: `{data['wrapper']}`",
        "",
        f"Recommendation: {data['recommendation']}",
        "",
        "| check | status | missing |",
        "| --- | --- | --- |",
    ]
    for name, item in data["checks"].items():
        missing = ", ".join(item.get("missing") or [])
        lines.append(f"| {name} | {item.get('status')} | {missing or '-'} |")
    dry_run = data["checks"]["wrapper_dry_run"]
    if dry_run.get("stdout_tail"):
        lines += ["", "## Wrapper Dry-Run Tail", "", "```text", dry_run["stdout_tail"].strip(), "```"]
    return "\n".join(lines) + "\n"


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
    print(text, end="")
    return 0 if data["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
