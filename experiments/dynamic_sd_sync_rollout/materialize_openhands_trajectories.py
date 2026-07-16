"""Materialize OpenHands SWE trajectories into replay JSONL for --mode replay.

Source rows (e.g. nvidia/SWE-Hero-openhands-trajectories) carry a `trajectory`
list of chat messages; we keep system/user/assistant/tool turns verbatim so
the replay prompt at each assistant turn is the exact recorded prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_role(msg: dict) -> dict | None:
    role = msg.get("role")
    content = msg.get("content")
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    if not isinstance(content, str) or role not in (
        "system",
        "user",
        "assistant",
        "tool",
    ):
        return None
    if role == "tool":
        role = "user"  # chat templates without tool role: fold into user turn
    return {"role": role, "content": content}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nvidia/SWE-Hero-openhands-trajectories")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    kept = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in ds:
            msgs = [m for m in map(normalize_role, row.get("trajectory") or []) if m]
            if sum(1 for m in msgs if m["role"] == "assistant") < args.min_turns:
                continue
            f.write(
                json.dumps(
                    {
                        "instance_id": row.get("instance_id"),
                        "messages": msgs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if kept >= args.limit:
                break
    print(f"wrote {kept} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
