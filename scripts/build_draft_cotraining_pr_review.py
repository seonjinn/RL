import json
from pathlib import Path
from typing import Any, Mapping


def load_context(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_context(context: Mapping[str, Any]) -> None:
    prs = context.get("prs")
    if not isinstance(prs, list) or len(prs) != 11:
        raise ValueError("context must contain exactly 11 PRs")

    ids = [pr.get("id") for pr in prs]
    if len(set(ids)) != len(ids):
        raise ValueError("context requires a unique PR id for every PR")

    expected = [f"pr-{number:02d}" for number in range(1, 12)]
    if ids != expected:
        raise ValueError("PR ids must be ordered from pr-01 through pr-11")
