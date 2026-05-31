#!/usr/bin/env python3
"""Validate Eagle3 training conversation JSONL before hidden-state dump.

The primary path in this workspace uses ModelOpt-style records:

    {"conversation_id": "...", "messages": [...]}

The validator also accepts SpecForge's conversation-format records:

    {"id": "...", "conversations": [...]}

Both formats carry the same role/content message semantics. Text-only
pre-formatted rows are intentionally not accepted here because this validator is
used as a ModelOpt hidden-state pipeline gate and must prove assistant messages
are present before the expensive dump starts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


VALID_ROLES = {"system", "user", "assistant", "tool", "function"}
ID_KEYS = ("conversation_id", "id")
MESSAGE_KEYS = ("messages", "conversations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or directories")
    parser.add_argument("--limit", type=int, default=None, help="Validate at most this many records")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--num-hidden-copies",
        type=int,
        default=4,
        help="Last hidden state plus Eagle3 aux hidden states. Qwen3-235B default is 4.",
    )
    parser.add_argument("--bytes-per-value", type=int, default=2, help="bf16/fp16 default is 2")
    parser.add_argument("--approx-chars-per-token", type=float, default=4.0)
    parser.add_argument("--min-assistant-chars", type=int, default=1)
    parser.add_argument("--fail-on-overlength", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer/model path for exact chat-template token counts.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help="Optional chat template file to attach to the tokenizer before counting.",
    )
    parser.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="Fail instead of falling back to character estimates when tokenizer loading fails.",
    )
    parser.add_argument(
        "--fail-on-duplicate-user-prompts",
        action="store_true",
        help="Fail when two records have the same normalized user prompt.",
    )
    parser.add_argument(
        "--denylist-prompts-from",
        type=Path,
        action="append",
        default=[],
        help="Conversation JSONL files whose user prompts must not appear in inputs.",
    )
    return parser.parse_args()


def input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    return files


def load_tokenizer(args: argparse.Namespace) -> tuple[Any | None, list[str], list[str]]:
    warnings: list[str] = []
    failures: list[str] = []
    if not args.tokenizer:
        return None, warnings, failures

    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        if args.chat_template:
            tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
        return tokenizer, warnings, failures
    except Exception as exc:  # pragma: no cover - depends on optional deps/env
        msg = f"tokenizer load failed for {args.tokenizer}: {exc}"
        if args.require_tokenizer:
            failures.append(msg)
        else:
            warnings.append(msg + "; falling back to character/token estimate")
        return None, warnings, failures


def iter_jsonl(files: Iterable[Path], limit: int | None):
    seen = 0
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    yield path, line_num, None, f"invalid JSON: {exc}"
                    continue
                yield path, line_num, record, None
                seen += 1
                if limit is not None and seen >= limit:
                    return


def content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def joined_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def user_prompt_key(messages: list[dict[str, str]]) -> str | None:
    for message in messages:
        if message["role"] == "user":
            return normalize_prompt(message["content"])
    return None


def load_denylist_prompt_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in input_files(paths):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                raw_messages, _ = record_messages(record)
                if not isinstance(raw_messages, list):
                    continue
                messages: list[dict[str, str]] = []
                for raw_msg in raw_messages:
                    if not isinstance(raw_msg, dict):
                        continue
                    role = raw_msg.get("role", raw_msg.get("from"))
                    content = raw_msg.get("content", raw_msg.get("value", raw_msg.get("text")))
                    if role is None or content in (None, ""):
                        continue
                    role = str(role).lower()
                    role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(
                        role, role
                    )
                    messages.append({"role": role, "content": content_to_text(content)})
                key = user_prompt_key(messages)
                if key:
                    keys.add(key)
    return keys


def record_id(record: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in ID_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value), key
    return None, None


def record_messages(record: dict[str, Any]) -> tuple[Any, str | None]:
    for key in MESSAGE_KEYS:
        value = record.get(key)
        if value is not None:
            return value, key
    return None, None


def schema_name(id_key: str | None, message_key: str | None) -> str:
    if id_key == "conversation_id" and message_key == "messages":
        return "modelopt"
    if id_key == "id" and message_key == "conversations":
        return "id_conversations"
    if message_key in MESSAGE_KEYS:
        return f"mixed_{id_key or 'missing_id'}_{message_key}"
    return "unknown"


def count_tokens(
    messages: list[dict[str, str]],
    tokenizer: Any | None,
    approx_chars_per_token: float,
) -> tuple[int, str, str | None]:
    if tokenizer is not None:
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(tokens, dict):
                tokens = tokens.get("input_ids", [])
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            return len(tokens), "tokenizer_chat_template", None
        except Exception as exc:
            try:
                encoded = tokenizer.encode(joined_text(messages), add_special_tokens=True)
                return len(encoded), "tokenizer_joined_text", f"chat-template tokenization failed: {exc}"
            except Exception as fallback_exc:
                text_len = len(joined_text(messages))
                estimate = math.ceil(text_len / approx_chars_per_token)
                return estimate, "chars_estimate", (
                    f"tokenizer fallback failed after chat-template error {exc}: {fallback_exc}"
                )

    text_len = len(joined_text(messages))
    return math.ceil(text_len / approx_chars_per_token), "chars_estimate", None


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct))
    return values[max(0, min(idx, len(values) - 1))]


def gibibytes(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def validate_record(
    record: Any,
    path: Path,
    line_num: int,
    seen_ids: set[str],
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    where = f"{path}:{line_num}"

    if not isinstance(record, dict):
        return None, [f"{where}: record must be a JSON object"], warnings

    cid, id_key = record_id(record)
    if cid in (None, ""):
        failures.append(f"{where}: missing non-empty conversation_id or id")
        cid = f"{path.name}:{line_num}"
    else:
        if cid in seen_ids:
            failures.append(f"{where}: duplicate conversation/id {cid!r}")
        seen_ids.add(cid)

    raw_messages, message_key = record_messages(record)
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, failures + [f"{where}: messages/conversations must be a non-empty list"], warnings

    messages: list[dict[str, str]] = []
    assistant_chars = 0
    assistant_count = 0
    total_chars = 0

    for msg_idx, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            failures.append(f"{where}: messages[{msg_idx}] must be an object")
            continue
        role = raw_msg.get("role", raw_msg.get("from"))
        content = raw_msg.get("content", raw_msg.get("value", raw_msg.get("text")))
        if role is None:
            failures.append(f"{where}: messages[{msg_idx}] missing role")
            continue
        role = str(role).lower()
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        if role not in VALID_ROLES:
            failures.append(f"{where}: messages[{msg_idx}] has unsupported role {role!r}")
        if content in (None, ""):
            failures.append(f"{where}: messages[{msg_idx}] has empty content")
            continue
        text = content_to_text(content)
        if not text.strip():
            failures.append(f"{where}: messages[{msg_idx}] has blank content")
            continue
        messages.append({"role": role, "content": text})
        total_chars += len(text)
        if role == "assistant":
            assistant_count += 1
            assistant_chars += len(text.strip())

    if not messages:
        return None, failures + [f"{where}: no valid messages"], warnings
    if assistant_chars < args.min_assistant_chars:
        failures.append(
            f"{where}: no assistant content with at least {args.min_assistant_chars} chars"
        )
    if messages[-1]["role"] != "assistant":
        warnings.append(f"{where}: last valid message role is {messages[-1]['role']!r}, not assistant")

    token_count, token_count_source, token_warning = count_tokens(
        messages,
        tokenizer,
        args.approx_chars_per_token,
    )
    if token_warning:
        warnings.append(f"{where}: {token_warning}")
    if token_count > args.max_seq_len:
        msg = f"{where}: estimated tokens {token_count} exceed max_seq_len {args.max_seq_len}"
        if args.fail_on_overlength:
            failures.append(msg)
        else:
            warnings.append(msg)

    stats = {
        "conversation_id": cid,
        "path": str(path),
        "line": line_num,
        "messages": len(messages),
        "assistant_messages": assistant_count,
        "assistant_chars": assistant_chars,
        "total_chars": total_chars,
        "estimated_tokens": token_count,
        "token_count_source": token_count_source,
        "id_key": id_key,
        "message_key": message_key,
        "schema": schema_name(id_key, message_key),
        "user_prompt_key": user_prompt_key(messages),
    }
    return stats, failures, warnings


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    try:
        files = input_files(args.inputs)
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not files:
        print("FAIL no JSONL files found", file=sys.stderr)
        raise SystemExit(1)

    tokenizer, tokenizer_warnings, tokenizer_failures = load_tokenizer(args)
    warnings.extend(tokenizer_warnings)
    failures.extend(tokenizer_failures)
    try:
        denylist_prompt_keys = load_denylist_prompt_keys(args.denylist_prompts_from)
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    seen_ids: set[str] = set()
    seen_prompt_keys: dict[str, str] = {}
    rows = 0
    valid_rows = 0
    stats_rows: list[dict[str, Any]] = []
    overlength = 0
    duplicate_user_prompt_count = 0
    denylisted_user_prompt_count = 0

    for path, line_num, record, parse_error in iter_jsonl(files, args.limit):
        rows += 1
        if parse_error:
            failures.append(f"{path}:{line_num}: {parse_error}")
            continue
        stats, row_failures, row_warnings = validate_record(
            record,
            path,
            line_num,
            seen_ids,
            args,
            tokenizer,
        )
        failures.extend(row_failures)
        warnings.extend(row_warnings)
        if stats:
            prompt_key = stats.get("user_prompt_key")
            if isinstance(prompt_key, str) and prompt_key:
                where = f"{path}:{line_num}"
                if prompt_key in denylist_prompt_keys:
                    denylisted_user_prompt_count += 1
                    failures.append(f"{where}: user prompt overlaps denylist")
                    row_failures.append(f"{where}: user prompt overlaps denylist")
                previous = seen_prompt_keys.get(prompt_key)
                if previous is not None:
                    duplicate_user_prompt_count += 1
                    msg = f"{where}: duplicate user prompt; first seen at {previous}"
                    if args.fail_on_duplicate_user_prompts:
                        failures.append(msg)
                        row_failures.append(msg)
                    else:
                        warnings.append(msg)
                else:
                    seen_prompt_keys[prompt_key] = where
            stats_rows.append(stats)
            if stats["estimated_tokens"] > args.max_seq_len:
                overlength += 1
            if not row_failures:
                valid_rows += 1

    token_counts = [int(row["estimated_tokens"]) for row in stats_rows]
    assistant_chars = [int(row["assistant_chars"]) for row in stats_rows]
    total_tokens = sum(token_counts)
    storage_bytes = (
        total_tokens * args.hidden_size * args.num_hidden_copies * args.bytes_per_value
    )
    source_counts: dict[str, int] = {}
    schema_counts: dict[str, int] = {}
    id_key_counts: dict[str, int] = {}
    message_key_counts: dict[str, int] = {}
    for row in stats_rows:
        source = str(row["token_count_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
        schema = str(row["schema"])
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        id_key = str(row["id_key"])
        id_key_counts[id_key] = id_key_counts.get(id_key, 0) + 1
        message_key = str(row["message_key"])
        message_key_counts[message_key] = message_key_counts.get(message_key, 0) + 1

    summary = {
        "inputs": [str(path) for path in files],
        "limit": args.limit,
        "rows_scanned": rows,
        "valid_rows": valid_rows,
        "records_with_stats": len(stats_rows),
        "unique_conversation_ids": len(seen_ids),
        "unique_user_prompts": len(seen_prompt_keys),
        "duplicate_user_prompt_count": duplicate_user_prompt_count,
        "denylist_prompt_count": len(denylist_prompt_keys),
        "denylisted_user_prompt_count": denylisted_user_prompt_count,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "overlength_count": overlength,
        "max_seq_len": args.max_seq_len,
        "estimated_tokens": {
            "total": total_tokens,
            "p50": percentile(token_counts, 0.50),
            "p95": percentile(token_counts, 0.95),
            "max": max(token_counts) if token_counts else 0,
            "source_counts": source_counts,
        },
        "assistant_chars": {
            "p50": percentile(assistant_chars, 0.50),
            "p95": percentile(assistant_chars, 0.95),
            "max": max(assistant_chars) if assistant_chars else 0,
        },
        "estimated_hidden_state_storage_gib": round(gibibytes(storage_bytes), 3),
        "schema_counts": schema_counts,
        "id_key_counts": id_key_counts,
        "message_key_counts": message_key_counts,
        "storage_assumption": {
            "hidden_size": args.hidden_size,
            "num_hidden_copies": args.num_hidden_copies,
            "bytes_per_value": args.bytes_per_value,
        },
        "warnings": warnings[:200],
        "failures": failures[:200],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("Training conversation validation")
    print(f"- files: {len(files)}")
    print(f"- rows scanned: {rows}")
    print(f"- valid rows: {valid_rows}")
    print(f"- unique conversation/id: {len(seen_ids)}")
    print(f"- unique user prompts: {len(seen_prompt_keys)}")
    print(f"- duplicate user prompts: {duplicate_user_prompt_count}")
    print(f"- denylisted user prompts: {denylisted_user_prompt_count}")
    print(f"- schema counts: {schema_counts}")
    print(
        "- estimated tokens: "
        f"p50={summary['estimated_tokens']['p50']} "
        f"p95={summary['estimated_tokens']['p95']} "
        f"max={summary['estimated_tokens']['max']} "
        f"total={summary['estimated_tokens']['total']}"
    )
    print(
        "- estimated hidden-state storage: "
        f"{summary['estimated_hidden_state_storage_gib']} GiB "
        f"for scanned rows"
    )
    print(f"- warnings: {len(warnings)}")
    for msg in warnings[:10]:
        print(f"  WARN {msg}")
    if len(warnings) > 10:
        print(f"  WARN ... {len(warnings) - 10} more")
    print(f"- failures: {len(failures)}")
    for msg in failures[:20]:
        print(f"  FAIL {msg}")
    if len(failures) > 20:
        print(f"  FAIL ... {len(failures) - 20} more")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
