#!/usr/bin/env python3
"""Generate ModelOpt Eagle3 training conversations from prompts.

This script talks to an OpenAI-compatible chat-completions endpoint, such as a
vLLM server, and writes JSONL records that ModelOpt hidden-state dumpers can
consume by default. Output records have:

    {
      "conversation_id": "...-r00",
      "messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}],
      "source_id": "...",
      "response_index": 0,
      "model": "..."
    }

For SGLang/SpecForge or vLLM Speculators comparison runs, pass
``--output-schema specforge`` or ``--output-schema speculators`` to write:

    {"id": "...", "conversations": [...]}

Input records can contain one of:

- messages: OpenAI-style message list
- conversations: OpenAI-style message list, or ShareGPT-style {"from", "value"}
- prompt: string or message list
- instruction/question/problem/input: string fallback fields
- SWE/NemoGym-style rows with problem_statement, repo, base_commit, instance_id

For records that already contain an assistant response, pass
``--use-existing-assistant`` to convert them without calling the endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


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
)
SWE_METADATA_KEYS = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "created_at",
)
SWE_TEST_KEYS = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
    "test_patch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-235B-A22B-Thinking-2507"),
        help="Model name served by the endpoint",
    )
    parser.add_argument("--num-responses", type=int, default=1, help="Responses per prompt")
    parser.add_argument(
        "--output-schema",
        choices=("modelopt", "specforge", "speculators"),
        default="modelopt",
        help="JSONL schema to write. SpecForge and vLLM Speculators use id/conversations.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip prompts whose chat-completion request still fails after retries",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent chat-completion requests for generated responses",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max input records")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many input records before processing. Useful for chunked generation.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print input schema summary and exit without calling the endpoint",
    )
    parser.add_argument("--inspect-limit", type=int, default=20)
    parser.add_argument(
        "--use-existing-assistant",
        action="store_true",
        help="Use existing assistant messages from input records instead of generating",
    )
    parser.add_argument(
        "--response-key",
        default=None,
        help="Explicit assistant response field for --use-existing-assistant",
    )
    parser.add_argument(
        "--swe-system-message",
        default=None,
        help="Optional system message to prepend when building SWE/NemoGym prompts",
    )
    parser.add_argument(
        "--include-swe-tests",
        action="store_true",
        help="Include SWE test metadata/test_patch in generated prompts. Off by default.",
    )
    parser.add_argument(
        "--id-key",
        default=None,
        help="Optional source-id field. Falls back to conversation_id, uuid, id, or row index",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output and skip already-written conversation_id values",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc


def normalize_role(role: str) -> str:
    role = role.lower()
    mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return mapping.get(role, role)


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ValueError(f"Expected messages list or string, got {type(value).__name__}")

    messages: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Expected message dict, got {type(item).__name__}")

        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role is None or content is None:
            raise ValueError(f"Message lacks role/from or content/value: {item}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        messages.append({"role": normalize_role(str(role)), "content": content})

    return [m for m in messages if m["content"]]


def stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def has_swe_fields(record: dict[str, Any]) -> bool:
    return any(key in record for key in ("problem_statement", "instance_id", "repo", "base_commit"))


def build_swe_messages(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    parts: list[str] = []

    metadata = []
    for key in SWE_METADATA_KEYS:
        if key in record and record[key] not in (None, ""):
            metadata.append(f"{key}: {record[key]}")
    if metadata:
        parts.append("Repository metadata:\n" + "\n".join(metadata))

    problem = None
    for key in ("problem_statement", "issue", "problem", "prompt", "input"):
        if key in record and record[key] not in (None, ""):
            problem = stringify_value(record[key])
            break
    if problem:
        parts.append("Problem statement:\n" + problem)

    hints = record.get("hints_text") or record.get("hints")
    if hints:
        parts.append("Hints:\n" + stringify_value(hints))

    if args.include_swe_tests:
        tests = []
        for key in SWE_TEST_KEYS:
            if key in record and record[key] not in (None, "", []):
                tests.append(f"{key}:\n{stringify_value(record[key])}")
        if tests:
            parts.append("Test metadata:\n" + "\n\n".join(tests))

    if not parts:
        raise ValueError(f"Could not build SWE prompt from keys: {record.keys()}")

    user_content = (
        "You are working on a software engineering task. "
        "Use the information below to produce the next assistant response.\n\n"
        + "\n\n".join(parts)
    )

    messages = []
    if args.swe_system_message:
        messages.append({"role": "system", "content": args.swe_system_message})
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_messages(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    if "messages" in record:
        return normalize_messages(record["messages"])
    if "conversations" in record:
        return normalize_messages(record["conversations"])
    if has_swe_fields(record):
        return build_swe_messages(record, args)
    for key in PROMPT_KEYS:
        if key in record and record[key] not in (None, ""):
            try:
                return normalize_messages(record[key])
            except ValueError:
                continue
    raise ValueError(f"Could not find messages/conversations/prompt fields in: {record.keys()}")


def source_id(record: dict[str, Any], line_num: int, id_key: str | None) -> str:
    keys = [id_key] if id_key else []
    keys.extend(["conversation_id", "uuid", "id", "task_id", "instance_id"])
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            return str(record[key])
    return f"row-{line_num:08d}"


def without_final_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed = list(messages)
    while trimmed and trimmed[-1]["role"] == "assistant":
        trimmed.pop()
    return trimmed


def final_assistant(messages: list[dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg["content"].strip():
            return msg["content"]
    return None


def assistant_from_record(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    keys = [args.response_key] if args.response_key else []
    keys.extend(ASSISTANT_KEYS)
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            value = record[key]
            if isinstance(value, list):
                try:
                    msg_response = final_assistant(normalize_messages(value))
                    if msg_response:
                        return msg_response
                except ValueError:
                    pass
            return stringify_value(value)
    return None


def inspect_schema(args: argparse.Namespace) -> None:
    key_counts: dict[str, int] = {}
    samples: dict[str, Any] = {}
    rows = 0
    for line_num, record in iter_jsonl(args.input):
        rows += 1
        for key, value in record.items():
            key_counts[key] = key_counts.get(key, 0) + 1
            if key not in samples:
                if isinstance(value, str):
                    samples[key] = value[:240]
                elif isinstance(value, list):
                    samples[key] = {
                        "type": "list",
                        "len": len(value),
                        "first": value[0] if value else None,
                    }
                elif isinstance(value, dict):
                    samples[key] = {"type": "dict", "keys": sorted(value.keys())[:30]}
                else:
                    samples[key] = value
        if rows >= args.inspect_limit:
            break

    print(json.dumps({"rows_inspected": rows, "key_counts": key_counts, "samples": samples}, ensure_ascii=False, indent=2))


def record_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("conversation_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def load_written_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    for _, record in iter_jsonl(path):
        cid = record_id(record)
        if cid:
            seen.add(cid)
    return seen


def post_chat_completion(
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_one(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
) -> str:
    max_tokens = args.max_tokens
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "n": 1,
    }
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            result = post_chat_completion(args.api_base, args.api_key, payload, args.timeout)
            content = result["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            raise RuntimeError(f"Empty completion response: {result}")
        except urllib.error.HTTPError as exc:
            last_error = exc
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            match = re.search(
                r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
                body,
            )
            if match:
                max_context = int(match.group(1))
                input_tokens = int(match.group(2))
                reduced_max_tokens = max_context - input_tokens
                if 0 < reduced_max_tokens < max_tokens:
                    max_tokens = reduced_max_tokens
                    payload["max_tokens"] = max_tokens
                    print(
                        "request max_tokens reduced after context-limit response: "
                        f"max_tokens={max_tokens} input_tokens={input_tokens} "
                        f"max_context={max_context}",
                        file=sys.stderr,
                    )
                    continue
            detail = f"{exc}; body={body[:500]}" if body else str(exc)
            print(f"request failed attempt={attempt}/{args.retries}: {detail}", file=sys.stderr)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError) as exc:
            last_error = exc
            print(f"request failed attempt={attempt}/{args.retries}: {exc}", file=sys.stderr)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    raise RuntimeError(f"chat completion failed after {args.retries} attempts") from last_error


def write_record(
    out_f,
    cid: str,
    sid: str,
    response_index: int,
    prompt_messages: list[dict[str, str]],
    response: str,
    model: str,
    output_schema: str,
) -> None:
    messages = prompt_messages + [{"role": "assistant", "content": response}]
    if output_schema in {"specforge", "speculators"}:
        record = {
            "id": cid,
            "conversations": messages,
        }
    else:
        record = {
            "conversation_id": cid,
            "messages": messages,
            "source_id": sid,
            "response_index": response_index,
            "model": model,
        }
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.flush()


def main() -> None:
    args = parse_args()
    if args.inspect_only:
        inspect_schema(args)
        return
    if args.num_responses < 1:
        raise ValueError("--num-responses must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    seen = load_written_ids(args.output) if args.append else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    num_inputs = 0
    num_written = 0
    num_skipped = 0

    if not args.use_existing_assistant and args.concurrency > 1:
        jobs: list[tuple[str, str, int, list[dict[str, str]]]] = []
        for line_num, record in iter_jsonl(args.input):
            if line_num <= args.offset:
                continue
            if args.limit is not None and num_inputs >= args.limit:
                break
            num_inputs += 1

            sid = source_id(record, line_num, args.id_key)
            messages = extract_messages(record, args)
            prompt_messages = without_final_assistant(messages)
            if not prompt_messages:
                print(f"skip {sid}: no prompt messages", file=sys.stderr)
                num_skipped += 1
                continue

            for response_index in range(args.num_responses):
                cid = f"{sid}-r{response_index:02d}"
                if cid in seen:
                    num_skipped += 1
                    continue
                seen.add(cid)
                jobs.append((cid, sid, response_index, prompt_messages))

        with open(args.output, mode, encoding="utf-8") as out_f:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_job = {
                    executor.submit(generate_one, args, prompt_messages): (
                        idx,
                        cid,
                        sid,
                        response_index,
                        prompt_messages,
                    )
                    for idx, (cid, sid, response_index, prompt_messages) in enumerate(jobs)
                }
                pending_results: dict[
                    int, tuple[str, str, int, list[dict[str, str]], str] | None
                ] = {}
                next_to_write = 0
                for future in as_completed(future_to_job):
                    idx, cid, sid, response_index, prompt_messages = future_to_job[future]
                    try:
                        response = future.result()
                    except Exception as exc:
                        if not args.skip_failed:
                            raise
                        print(f"skip {cid}: generation failed: {exc}", file=sys.stderr)
                        num_skipped += 1
                        pending_results[idx] = None
                    else:
                        pending_results[idx] = (
                            cid,
                            sid,
                            response_index,
                            prompt_messages,
                            response,
                        )

                    while next_to_write in pending_results:
                        result = pending_results.pop(next_to_write)
                        next_to_write += 1
                        if result is None:
                            continue
                        (
                            out_cid,
                            out_sid,
                            out_response_index,
                            out_prompt_messages,
                            out_response,
                        ) = result
                        write_record(
                            out_f,
                            out_cid,
                            out_sid,
                            out_response_index,
                            out_prompt_messages,
                            out_response,
                            args.model,
                            args.output_schema,
                        )
                        num_written += 1
                        print(f"wrote {out_cid}", file=sys.stderr)

        print(
            json.dumps(
                {
                    "input_offset": args.offset,
                    "inputs_seen": num_inputs,
                    "records_written": num_written,
                    "records_skipped": num_skipped,
                    "output": str(args.output),
                    "output_schema": args.output_schema,
                },
                indent=2,
            )
        )
        return

    with open(args.output, mode, encoding="utf-8") as out_f:
        for line_num, record in iter_jsonl(args.input):
            if line_num <= args.offset:
                continue
            if args.limit is not None and num_inputs >= args.limit:
                break
            num_inputs += 1

            sid = source_id(record, line_num, args.id_key)
            messages = extract_messages(record, args)
            prompt_messages = without_final_assistant(messages)
            if not prompt_messages:
                print(f"skip {sid}: no prompt messages", file=sys.stderr)
                num_skipped += 1
                continue

            if args.use_existing_assistant:
                response = final_assistant(messages) or assistant_from_record(record, args)
                if response is None:
                    print(f"skip {sid}: no existing assistant response", file=sys.stderr)
                    num_skipped += 1
                    continue
                cid = f"{sid}-r00"
                if cid in seen:
                    num_skipped += 1
                    continue
                write_record(out_f, cid, sid, 0, prompt_messages, response, args.model, args.output_schema)
                seen.add(cid)
                num_written += 1
                continue

            for response_index in range(args.num_responses):
                cid = f"{sid}-r{response_index:02d}"
                if cid in seen:
                    num_skipped += 1
                    continue
                try:
                    response = generate_one(args, prompt_messages)
                except Exception as exc:
                    if not args.skip_failed:
                        raise
                    print(f"skip {cid}: generation failed: {exc}", file=sys.stderr)
                    num_skipped += 1
                    continue
                write_record(
                    out_f,
                    cid,
                    sid,
                    response_index,
                    prompt_messages,
                    response,
                    args.model,
                    args.output_schema,
                )
                seen.add(cid)
                num_written += 1
                print(f"wrote {cid}", file=sys.stderr)

    print(
        json.dumps(
            {
                "input_offset": args.offset,
                "inputs_seen": num_inputs,
                "records_written": num_written,
                "records_skipped": num_skipped,
                "output": str(args.output),
                "output_schema": args.output_schema,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
