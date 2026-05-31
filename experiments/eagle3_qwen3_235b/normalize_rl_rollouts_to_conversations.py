#!/usr/bin/env python3
"""Normalize RL rollout logs into ModelOpt Eagle3 conversation JSONL.

Use this when NeMo-RL/SWE rollout logs already contain assistant text. It avoids
calling a generation endpoint and writes records consumable by ModelOpt hidden
state dump scripts by default:

    {"conversation_id": "...", "messages": [...], "source_id": "..."}

For SGLang/SpecForge or vLLM Speculators comparison runs, pass
``--output-schema specforge`` or ``--output-schema speculators`` to write
conversation-format rows:

    {"id": "...", "conversations": [...]}

The parser is intentionally schema-tolerant. It handles plain
``messages``/``conversations`` rows, ``prompt`` + ``response`` rows, prompt +
``responses`` lists, and nested containers such as ``trajectories`` or
``rollouts``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


MESSAGE_KEYS = ("messages", "conversations", "conversation")
FLAT_CONTENT_KEYS = ("content", "contents")
FLAT_ROLE_KEYS = ("role", "roles")
PROMPT_KEYS = (
    "prompt",
    "instruction",
    "question",
    "problem",
    "query",
    "input",
    "issue",
    "problem_statement",
)
ASSISTANT_KEYS = (
    "assistant",
    "assistant_response",
    "response",
    "completion",
    "output",
    "answer",
    "prediction",
    "model_output",
    "extracted_model_output",
)
ASSISTANT_LIST_KEYS = (
    "responses",
    "completions",
    "outputs",
    "answers",
    "predictions",
    "generations",
    "assistant_responses",
)
REASONING_KEYS = (
    "reasoning_content",
    "reasoning",
    "reasoning_trace",
    "thinking",
)
NESTED_KEYS = (
    "trajectory",
    "trajectories",
    "rollout",
    "rollouts",
    "episode",
    "episodes",
    "sample",
    "samples",
    "record",
    "records",
    "item",
    "items",
    "data",
)
ID_KEYS = (
    "conversation_id",
    "uuid",
    "id",
    "task_id",
    "instance_id",
    "prompt_id",
    "sample_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Input JSONL files or dirs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument(
        "--output-schema",
        choices=("modelopt", "specforge", "speculators"),
        default="modelopt",
        help="JSONL schema to write. SpecForge and vLLM Speculators use id/conversations.",
    )
    parser.add_argument("--id-key", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--inspect-limit", type=int, default=20)
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Copy non-message scalar metadata into output records.",
    )
    parser.add_argument(
        "--include-reasoning-content",
        action="store_true",
        help="Merge assistant reasoning_content/reasoning fields into assistant content.",
    )
    parser.add_argument(
        "--infer-flat-content-roles",
        action="store_true",
        help=(
            "For NeMo-RL train_data JSONL rows that have flat content but no role "
            "list, treat the final content item as assistant and earlier items as user. "
            "Prefer role-aware logs when possible."
        ),
    )
    parser.add_argument(
        "--compact-current-turn",
        action="store_true",
        help=(
            "For SWE/Codex traces with long tool history, keep system/developer "
            "context plus the final user turn and assistant response only."
        ),
    )
    parser.add_argument("--reasoning-open-tag", default="<think>\n")
    parser.add_argument("--reasoning-close-tag", default="\n</think>\n\n")
    return parser.parse_args()


def iter_input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    return files


def iter_jsonl(paths: Iterable[Path]):
    for path in iter_input_files(paths):
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
            if isinstance(value, list):
                for line_num, item in enumerate(value, 1):
                    yield path, line_num, item
            else:
                yield path, 1, value
            continue
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    yield path, line_num, json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc


def normalize_role(role: Any) -> str:
    value = str(role).lower()
    mapping = {
        "human": "user",
        "user": "user",
        "developer": "system",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "model": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
        "observation": "tool",
    }
    return mapping.get(value, value)


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def merge_reasoning_content(
    content: Any,
    reasoning: Any,
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
) -> str:
    text = stringify(content)
    if not include_reasoning_content or reasoning in (None, ""):
        return text
    reasoning_text = stringify(reasoning).strip()
    if not reasoning_text or "<think" in text[:200]:
        return text
    return f"{reasoning_open_tag}{reasoning_text}{reasoning_close_tag}{text}"


def first_reasoning_value(record: dict[str, Any]) -> Any:
    for key in REASONING_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_messages(
    value: Any,
    include_reasoning_content: bool = False,
    reasoning_open_tag: str = "<think>\n",
    reasoning_close_tag: str = "\n</think>\n\n",
) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ValueError(f"Expected message list or string, got {type(value).__name__}")

    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Expected message dict, got {type(item).__name__}")
        role = item.get("role", item.get("from", item.get("speaker")))
        content = item.get("content", item.get("value", item.get("text")))
        item_type = item.get("type")
        if role is None and item_type in {"function_call", "custom_tool_call"}:
            role = "assistant"
            content = {
                "type": item_type,
                "name": item.get("name"),
                "arguments": item.get("arguments", item.get("input")),
                "call_id": item.get("call_id"),
            }
        elif role is None and item_type in {"function_call_output", "custom_tool_call_output"}:
            role = "tool"
            content = item.get("output")
        elif role is None and item_type == "reasoning":
            continue
        if role is None or content in (None, ""):
            raise ValueError(f"Message lacks role/from/speaker or content/value/text: {item}")
        normalized_role = normalize_role(role)
        content = text_from_content_parts(content)
        if normalized_role == "assistant":
            content = merge_reasoning_content(
                content,
                first_reasoning_value(item),
                include_reasoning_content,
                reasoning_open_tag,
                reasoning_close_tag,
            )
        messages.append({"role": normalized_role, "content": stringify(content)})
    return [m for m in messages if m["content"].strip()]


def text_from_content_parts(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("output_text", item.get("content")))
                if text not in (None, ""):
                    parts.append(stringify(text))
            elif item not in (None, ""):
                parts.append(stringify(item))
        return "\n".join(part.strip() for part in parts if part.strip())
    return stringify(value)


def normalize_responses_api_input(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                messages.append({"role": "user", "content": text})
            continue
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role", "user"))
        content = item.get("content", item.get("text"))
        if content in (None, ""):
            continue
        text = text_from_content_parts(content).strip()
        if text:
            messages.append({"role": role, "content": text})
    return messages or None


def responses_api_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    params = record.get("responses_create_params")
    if not isinstance(params, dict):
        return None
    messages = normalize_responses_api_input(params.get("input"))
    if messages:
        return messages
    prompt = params.get("prompt")
    if prompt not in (None, ""):
        return [{"role": "user", "content": stringify(prompt)}]
    instructions = params.get("instructions")
    if instructions not in (None, ""):
        return [{"role": "system", "content": stringify(instructions)}]
    return None


def responses_api_output_texts(
    value: Any,
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
) -> list[str]:
    if not isinstance(value, dict):
        return []
    output = value.get("output")
    if not isinstance(output, list):
        return []
    reasoning_parts: list[str] = []
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        role = normalize_role(item.get("role", "assistant" if item_type == "message" else ""))
        if item_type == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    if isinstance(part, dict) and part.get("text") not in (None, ""):
                        reasoning_parts.append(stringify(part["text"]))
                    elif part not in (None, ""):
                        reasoning_parts.append(stringify(part))
            elif summary not in (None, ""):
                reasoning_parts.append(stringify(summary))
            continue
        if role != "assistant" and item_type != "message":
            continue
        content = item.get("content", item.get("text"))
        if content in (None, ""):
            continue
        text = text_from_content_parts(content).strip()
        if text:
            texts.append(text)
    reasoning = "\n".join(part.strip() for part in reasoning_parts if part.strip())
    if include_reasoning_content and reasoning:
        return [
            merge_reasoning_content(text, reasoning, True, reasoning_open_tag, reasoning_close_tag)
            for text in texts
        ]
    return texts


def has_assistant(messages: list[dict[str, str]], min_chars: int) -> bool:
    return any(m["role"] == "assistant" and len(m["content"].strip()) >= min_chars for m in messages)


def compact_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    """Keep stable instructions plus the last user turn before a response."""

    last_user_idx = None
    for idx, message in enumerate(messages):
        if message["role"] == "user" and message["content"].strip():
            last_user_idx = idx
    if last_user_idx is None:
        return None

    compacted: list[dict[str, str]] = []
    seen_system: set[str] = set()
    for message in messages[: last_user_idx + 1]:
        if message["role"] != "system":
            continue
        content = message["content"].strip()
        if not content or content in seen_system:
            continue
        compacted.append({"role": "system", "content": content})
        seen_system.add(content)

    user_content = messages[last_user_idx]["content"].strip()
    if not user_content:
        return None
    compacted.append({"role": "user", "content": user_content})
    return compacted


def compact_complete_messages(
    messages: list[dict[str, str]],
    min_assistant_chars: int,
) -> list[dict[str, str]] | None:
    """Compact a full conversation to final prompt turn plus final assistant."""

    assistant_idx = None
    for idx, message in enumerate(messages):
        if message["role"] == "assistant" and len(message["content"].strip()) >= min_assistant_chars:
            assistant_idx = idx
    if assistant_idx is None:
        return None

    prompt = compact_prompt_messages(messages[:assistant_idx])
    if not prompt:
        return None
    assistant_content = messages[assistant_idx]["content"].strip()
    if not assistant_content:
        return None
    return prompt + [{"role": "assistant", "content": assistant_content}]


def source_id(record: dict[str, Any], path: Path, line_num: int, id_key: str | None) -> str:
    explicit = explicit_source_id(record, id_key)
    if explicit is not None:
        return explicit
    return f"{path.stem}-line{line_num:08d}"


def explicit_source_id(record: dict[str, Any], id_key: str | None) -> str | None:
    keys = [id_key] if id_key else []
    keys.extend(ID_KEYS)
    for key in keys:
        if key and record.get(key) not in (None, ""):
            return str(record[key])
    return None


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key in FLAT_CONTENT_KEYS or key in FLAT_ROLE_KEYS:
            continue
        if key in MESSAGE_KEYS or key in PROMPT_KEYS or key in ASSISTANT_KEYS:
            continue
        if key in ASSISTANT_LIST_KEYS or key in NESTED_KEYS:
            continue
        if key in REASONING_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def first_list_value(record: dict[str, Any], keys: tuple[str, ...]) -> list[Any] | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return value
    return None


def flat_content_messages(record: dict[str, Any], infer_flat_content_roles: bool) -> list[dict[str, str]] | None:
    """Convert SpecDec-RL flat content/role arrays into OpenAI-style messages."""

    contents = first_list_value(record, FLAT_CONTENT_KEYS)
    if not contents:
        return None
    roles = first_list_value(record, FLAT_ROLE_KEYS)

    messages: list[dict[str, str]] = []
    if roles and len(roles) == len(contents):
        for role, content in zip(roles, contents):
            text = stringify(content).strip()
            if not text:
                continue
            messages.append({"role": normalize_role(role), "content": text})
        return messages or None

    if not infer_flat_content_roles or len(contents) < 2:
        return None

    for index, content in enumerate(contents):
        text = stringify(content).strip()
        if not text:
            continue
        role = "assistant" if index == len(contents) - 1 else "user"
        messages.append({"role": role, "content": text})
    return messages or None


def first_prompt_messages(
    record: dict[str, Any],
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
) -> list[dict[str, str]] | None:
    messages = responses_api_prompt_messages(record)
    if messages:
        return messages
    for key in MESSAGE_KEYS:
        if key in record and record[key] not in (None, ""):
            try:
                messages = normalize_messages(
                    record[key],
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                )
            except ValueError:
                continue
            if messages:
                return messages
    for key in PROMPT_KEYS:
        if key in record and record[key] not in (None, ""):
            try:
                return normalize_messages(record[key])
            except ValueError:
                return [{"role": "user", "content": stringify(record[key])}]
    return None


def assistant_texts(
    record: dict[str, Any],
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
) -> list[str]:
    texts: list[str] = []
    reasoning = first_reasoning_value(record)
    for key in ASSISTANT_KEYS:
        value = record.get(key)
        if value in (None, ""):
            continue
        if key == "response" and isinstance(value, dict):
            texts.extend(
                responses_api_output_texts(
                    value,
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                )
            )
            continue
        if isinstance(value, list):
            try:
                messages = normalize_messages(
                    value,
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                )
                texts.extend(m["content"] for m in messages if m["role"] == "assistant")
                continue
            except ValueError:
                pass
        texts.append(
            merge_reasoning_content(
                value,
                reasoning,
                include_reasoning_content,
                reasoning_open_tag,
                reasoning_close_tag,
            )
        )
    for key in ASSISTANT_LIST_KEYS:
        value = record.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                nested = assistant_texts(
                    item,
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                )
                if nested:
                    texts.extend(nested)
                    continue
                for subkey in ASSISTANT_KEYS:
                    if item.get(subkey) not in (None, ""):
                        texts.append(stringify(item[subkey]))
                        break
            elif item not in (None, ""):
                texts.append(stringify(item))
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        stripped = text.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        out.append(stripped)
    return out


def extract_from_record(
    record: dict[str, Any],
    path: Path,
    line_num: int,
    id_key: str | None,
    min_assistant_chars: int,
    include_reasoning_content: bool,
    reasoning_open_tag: str,
    reasoning_close_tag: str,
    infer_flat_content_roles: bool = False,
    parent_prompt: list[dict[str, str]] | None = None,
    parent_source_id: str | None = None,
    prefix: str = "",
    compact_current_turn: bool = False,
) -> list[dict[str, Any]]:
    sid = explicit_source_id(record, id_key) or parent_source_id or source_id(record, path, line_num, id_key)
    if prefix:
        sid = f"{sid}-{prefix}"

    results: list[dict[str, Any]] = []
    prompt_messages = (
        first_prompt_messages(
            record,
            include_reasoning_content,
            reasoning_open_tag,
            reasoning_close_tag,
        )
        or parent_prompt
    )
    if compact_current_turn and prompt_messages:
        prompt_messages = compact_prompt_messages(prompt_messages) or prompt_messages
    response_texts = (
        assistant_texts(
            record,
            include_reasoning_content,
            reasoning_open_tag,
            reasoning_close_tag,
        )
        if prompt_messages
        else []
    )

    flat_messages = flat_content_messages(record, infer_flat_content_roles)
    if compact_current_turn and flat_messages:
        flat_messages = compact_complete_messages(flat_messages, min_assistant_chars)
    if flat_messages and has_assistant(flat_messages, min_assistant_chars):
        results.append({"source_id": sid, "messages": flat_messages, "metadata": metadata(record)})

    for key in MESSAGE_KEYS:
        if key in record and record[key] not in (None, ""):
            if compact_current_turn and response_texts:
                continue
            try:
                messages = normalize_messages(
                    record[key],
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                )
            except ValueError:
                continue
            if compact_current_turn:
                messages = compact_complete_messages(messages, min_assistant_chars) or []
            if has_assistant(messages, min_assistant_chars):
                results.append({"source_id": sid, "messages": messages, "metadata": metadata(record)})

    if prompt_messages:
        for response_index, text in enumerate(response_texts):
            if len(text.strip()) < min_assistant_chars:
                continue
            messages = list(prompt_messages) + [{"role": "assistant", "content": text}]
            results.append(
                {
                    "source_id": sid,
                    "response_index": response_index,
                    "messages": messages,
                    "metadata": metadata(record),
                }
            )

    child_prompt = prompt_messages or parent_prompt
    for key in NESTED_KEYS:
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            results.extend(
                extract_from_record(
                    value,
                    path,
                    line_num,
                    id_key,
                    min_assistant_chars,
                    include_reasoning_content,
                    reasoning_open_tag,
                    reasoning_close_tag,
                    infer_flat_content_roles,
                    parent_prompt=child_prompt,
                    parent_source_id=sid,
                    prefix=f"{prefix}{key}" if not prefix else f"{prefix}-{key}",
                    compact_current_turn=compact_current_turn,
                )
            )
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    results.extend(
                        extract_from_record(
                            item,
                            path,
                            line_num,
                            id_key,
                            min_assistant_chars,
                            include_reasoning_content,
                            reasoning_open_tag,
                            reasoning_close_tag,
                            infer_flat_content_roles,
                            parent_prompt=child_prompt,
                            parent_source_id=sid,
                            prefix=f"{prefix}{key}{idx:03d}" if not prefix else f"{prefix}-{key}{idx:03d}",
                            compact_current_turn=compact_current_turn,
                        )
                    )
    return results


def record_key(item: dict[str, Any]) -> str:
    return json.dumps(item.get("messages", []), ensure_ascii=False, sort_keys=True)


def record_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("conversation_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                seen.add(record_id(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return seen


def output_record(
    schema: str,
    cid: str,
    item: dict[str, Any],
    response_index: int,
    model: str,
    include_metadata: bool,
) -> dict[str, Any]:
    if schema in {"specforge", "speculators"}:
        output: dict[str, Any] = {
            "id": cid,
            "conversations": item["messages"],
        }
        if include_metadata:
            output.update(
                {
                    "source_id": item["source_id"],
                    "response_index": item.get("response_index", response_index),
                    "model": model,
                }
            )
            if item.get("metadata"):
                output["metadata"] = item["metadata"]
        return output

    output = {
        "conversation_id": cid,
        "messages": item["messages"],
        "source_id": item["source_id"],
        "response_index": item.get("response_index", response_index),
        "model": model,
    }
    if include_metadata and item.get("metadata"):
        output["metadata"] = item["metadata"]
    return output


def inspect(args: argparse.Namespace) -> None:
    rows = 0
    extracted = 0
    key_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for path, line_num, record in iter_jsonl(args.input):
        rows += 1
        if isinstance(record, dict):
            for key in record:
                key_counts[key] = key_counts.get(key, 0) + 1
            found = extract_from_record(
                record,
                path,
                line_num,
                args.id_key,
                args.min_assistant_chars,
                args.include_reasoning_content,
                args.reasoning_open_tag,
                args.reasoning_close_tag,
                args.infer_flat_content_roles,
                compact_current_turn=args.compact_current_turn,
            )
            extracted += len(found)
            if found and len(samples) < 3:
                sample = dict(found[0])
                sample["messages"] = sample["messages"][:2]
                samples.append(sample)
        if rows >= args.inspect_limit:
            break
    print(
        json.dumps(
            {
                "rows_inspected": rows,
                "candidate_conversations": extracted,
                "key_counts": key_counts,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    args = parse_args()
    if args.inspect_only:
        inspect(args)
        return 0

    seen_ids = load_seen(args.output) if args.append else set()
    seen_payloads: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    rows_seen = 0
    rows_with_output = 0
    written = 0
    skipped = 0

    with args.output.open(mode, encoding="utf-8") as out_f:
        for path, line_num, record in iter_jsonl(args.input):
            if args.limit is not None and rows_seen >= args.limit:
                break
            rows_seen += 1
            if not isinstance(record, dict):
                skipped += 1
                continue
            conversations = extract_from_record(
                record,
                path,
                line_num,
                args.id_key,
                args.min_assistant_chars,
                args.include_reasoning_content,
                args.reasoning_open_tag,
                args.reasoning_close_tag,
                args.infer_flat_content_roles,
                compact_current_turn=args.compact_current_turn,
            )
            if conversations:
                rows_with_output += 1
            for idx, item in enumerate(conversations):
                cid = f"{item['source_id']}-c{idx:03d}"
                if cid in seen_ids:
                    skipped += 1
                    continue
                payload_key = record_key(item)
                if payload_key in seen_payloads:
                    skipped += 1
                    continue
                output = output_record(
                    args.output_schema,
                    cid,
                    item,
                    idx,
                    args.model,
                    args.include_metadata,
                )
                out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
                seen_ids.add(cid)
                seen_payloads.add(payload_key)
                written += 1

    print(
        json.dumps(
            {
                "rows_seen": rows_seen,
                "rows_with_output": rows_with_output,
                "records_written": written,
                "records_skipped": skipped,
                "output": str(args.output),
                "output_schema": args.output_schema,
                "compact_current_turn": args.compact_current_turn,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
