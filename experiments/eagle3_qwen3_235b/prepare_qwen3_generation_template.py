#!/usr/bin/env python3
"""Prepare a Qwen3 chat template with Transformers generation tags.

ModelOpt answer-only loss needs ``{% generation %}`` / ``{% endgeneration %}``
tags so ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
can produce a loss mask. Qwen3 templates usually do not ship with these tags,
so this helper extracts a template from a local file, tokenizer_config.json, or
Hugging Face raw tokenizer_config.json, then wraps the assistant branch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


ASSISTANT_BRANCH_RE = re.compile(
    r"(\{%-?\s*elif\s+message(?:\.role|\[['\"]role['\"]\])\s*==\s*['\"]assistant['\"]\s*-?%\})"
)
BRANCH_END_RE = re.compile(
    r"\{%-?\s*(?:elif\s+message(?:\.role|\[['\"]role['\"]\])\s*==\s*['\"][^'\"]+['\"]|else|endif)\s*-?%\}"
)
ROLE_BRANCH_RE = re.compile(
    r"\{%-?\s*elif\s+message(?:\.role|\[['\"]role['\"]\])\s*==\s*['\"][^'\"]+['\"]\s*-?%\}"
)
IM_END_RE = re.compile(r"\{\{\s*-?\s*['\"]<\|im_end\|>\\n['\"]\s*\}\}")
GENERATION_TAG_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")
ENDGENERATION_TAG_RE = re.compile(r"\{%-?\s*endgeneration\s*-?%\}")
QWEN3_ASSISTANT_OUTPUT_RE = re.compile(r"\{%-?\s*if\s+loop\.index0\s*>\s*ns\.last_query_index\s*-?%\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--template", type=Path, help="Existing chat template Jinja file")
    source.add_argument("--tokenizer-config", type=Path, help="Local tokenizer_config.json")
    source.add_argument(
        "--model",
        default=None,
        help="HF model id to fetch tokenizer_config.json from, e.g. Qwen/Qwen3-235B-A22B-Thinking-2507",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Only extract/copy the template and verify whether generation tags already exist.",
    )
    return parser.parse_args()


def load_template(args: argparse.Namespace) -> tuple[str, str]:
    if args.template:
        return args.template.read_text(encoding="utf-8"), str(args.template)
    if args.tokenizer_config:
        data = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
        template = data.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError(f"{args.tokenizer_config} has no non-empty chat_template")
        return template, str(args.tokenizer_config)
    assert args.model
    url = f"https://huggingface.co/{args.model}/raw/{args.revision}/tokenizer_config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    template = data.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError(f"{url} has no non-empty chat_template")
    return template, url


def has_generation_tags(template: str) -> bool:
    return bool(GENERATION_TAG_RE.search(template) and ENDGENERATION_TAG_RE.search(template))


def patch_qwen3_thinking_assistant_branch(template: str, branch_start: int, branch_end: int) -> tuple[str, str] | None:
    """Patch Qwen3's reasoning-aware assistant branch without wrapping control flow.

    Qwen3's official template computes ``reasoning_content`` with nested
    ``if/else`` blocks before it emits assistant text. Wrapping the whole
    assistant branch breaks that nested control flow in Jinja. Instead, wrap
    the emitted assistant text section: from the final output-selection block
    through any tool-call text, stopping before ``<|im_end|>``.
    """

    branch = template[branch_start:branch_end]
    output_match = QWEN3_ASSISTANT_OUTPUT_RE.search(branch)
    im_end_matches = list(IM_END_RE.finditer(branch))
    if not output_match or not im_end_matches:
        return None

    gen_at = branch_start + output_match.start()
    end_at = branch_start + im_end_matches[-1].start()
    if gen_at >= end_at:
        return None

    patched = template[:gen_at] + "{% generation %}\n" + template[gen_at:end_at] + "{% endgeneration %}\n" + template[end_at:]
    if not has_generation_tags(patched):
        raise ValueError("Qwen3 patched template still lacks generation tags")
    return patched, "qwen3_assistant_output_block"


def patch_assistant_branch(template: str) -> tuple[str, str]:
    if has_generation_tags(template):
        return template, "already_tagged"

    match = ASSISTANT_BRANCH_RE.search(template)
    if not match:
        raise ValueError(
            "Could not find Qwen-style assistant branch "
            "{% elif message.role == \"assistant\" %}."
        )

    branch_start = match.end()
    role_match = ROLE_BRANCH_RE.search(template, branch_start)
    role_branch_end = role_match.start() if role_match else len(template)
    qwen3_patch = patch_qwen3_thinking_assistant_branch(template, branch_start, role_branch_end)
    if qwen3_patch is not None:
        return qwen3_patch
    next_match = BRANCH_END_RE.search(template, branch_start)
    branch_end = next_match.start() if next_match else len(template)
    branch = template[branch_start:branch_end]

    patched_branch = "\n{% generation %}" + branch
    im_end_matches = list(IM_END_RE.finditer(patched_branch))
    if im_end_matches:
        end_insert = im_end_matches[-1].start()
        patched_branch = (
            patched_branch[:end_insert] + "{% endgeneration %}\n" + patched_branch[end_insert:]
        )
        strategy = "assistant_branch_before_im_end"
    else:
        patched_branch = patched_branch + "\n{% endgeneration %}\n"
        strategy = "assistant_branch_until_next_role"

    patched = template[:branch_start] + patched_branch + template[branch_end:]
    if not has_generation_tags(patched):
        raise ValueError("Patched template still lacks generation tags")
    return patched, strategy


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --force")

    template, source = load_template(args)
    strategy = "copied"
    if not args.no_patch:
        template, strategy = patch_assistant_branch(template)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf-8")

    status = "tagged" if has_generation_tags(template) else "untagged"
    print(f"source={source}")
    print(f"output={args.output}")
    print(f"status={status}")
    print(f"strategy={strategy}")
    if status != "tagged":
        print("warning: output lacks generation/endgeneration tags", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
