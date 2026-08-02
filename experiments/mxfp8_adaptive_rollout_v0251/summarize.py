from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict


class RunSummary(TypedDict):
    arm: str | None
    complete: bool
    elapsed_seconds: float | None
    model_load_seconds: float | None
    output_tokens: int | None


_MARKER = re.compile(r"^NEMORL_CANARY\s+(?P<fields>.+)$")


def _fields(line: str) -> dict[str, str]:
    match = _MARKER.match(line.strip())
    if match is None:
        return {}
    parsed: dict[str, str] = {}
    for item in match.group("fields").split():
        key, separator, value = item.partition("=")
        if separator:
            parsed[key] = value
    return parsed


def summarize_log(path: Path) -> RunSummary:
    arm: str | None = None
    start: float | None = None
    model_ready: float | None = None
    complete: float | None = None
    output_tokens: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = _fields(line)
        if not fields:
            continue
        arm = fields.get("arm", arm)
        event = fields.get("event")
        if event == "start" and "epoch" in fields:
            start = float(fields["epoch"])
        elif event == "model_ready" and "epoch" in fields:
            model_ready = float(fields["epoch"])
        elif event == "outputs" and "tokens" in fields:
            output_tokens = int(fields["tokens"])
        elif event == "complete" and "epoch" in fields:
            complete = float(fields["epoch"])

    is_complete = start is not None and complete is not None
    elapsed_seconds = None
    if start is not None and complete is not None:
        elapsed_seconds = complete - start
    return {
        "arm": arm,
        "complete": is_complete,
        "elapsed_seconds": elapsed_seconds,
        "model_load_seconds": (
            model_ready - start if start is not None and model_ready is not None else None
        ),
        "output_tokens": output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize_log(path) for path in args.logs]
    encoded = json.dumps(summaries, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
