#!/usr/bin/env python3
"""Create symlink-backed checkpoint views with a matched YaRN config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


MARKER_NAME = ".long_context_view.json"
ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 1_000_000


def _config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(
    *,
    source: Path,
    max_position_embeddings: int,
    rope_factor: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_config_sha256": _config_digest(source / "config.json"),
        "max_position_embeddings": max_position_embeddings,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": rope_factor,
            "original_max_position_embeddings": ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "rope_theta": ROPE_THETA,
        },
    }


def materialize_model_view(
    *,
    source: Path,
    destination: Path,
    max_position_embeddings: int,
    rope_factor: float,
) -> dict[str, object]:
    """Create a model view that owns config metadata and symlinks all assets."""
    source = source.resolve()
    source_config = source / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"missing source config: {source_config}")
    if max_position_embeddings <= ORIGINAL_MAX_POSITION_EMBEDDINGS:
        raise ValueError("extended max_position_embeddings must exceed 32768")
    if rope_factor <= 1.0:
        raise ValueError("rope_factor must be greater than 1")

    metadata = _metadata(
        source=source,
        max_position_embeddings=max_position_embeddings,
        rope_factor=rope_factor,
    )
    marker = destination / MARKER_NAME
    if destination.exists():
        if not marker.is_file():
            raise FileExistsError(
                f"refusing to replace unowned destination: {destination}"
            )
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current == metadata:
            return metadata

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()

    source_payload: dict[str, Any] = json.loads(
        source_config.read_text(encoding="utf-8")
    )
    source_payload.pop("rope_scaling", None)
    source_payload["max_position_embeddings"] = max_position_embeddings
    source_payload["rope_theta"] = ROPE_THETA
    source_payload["rope_parameters"] = metadata["rope_parameters"]
    (temporary / "config.json").write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for child in source.iterdir():
        if child.name == "config.json":
            continue
        (temporary / child.name).symlink_to(
            child.resolve(), target_is_directory=child.is_dir()
        )

    (temporary / MARKER_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return metadata


def _parse_model_view(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if Path(name).name != name or name in {".", ".."}:
        raise argparse.ArgumentTypeError(f"invalid view name: {name!r}")
    return name, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--model-view",
        action="append",
        type=_parse_model_view,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--max-position-embeddings", type=int, default=131072)
    parser.add_argument("--rope-factor", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seen: set[str] = set()
    for name, source in args.model_view:
        if name in seen:
            raise ValueError(f"duplicate model view name: {name}")
        seen.add(name)
        destination = args.view_root / name
        materialize_model_view(
            source=source,
            destination=destination,
            max_position_embeddings=args.max_position_embeddings,
            rope_factor=args.rope_factor,
        )
        print(f"model_view={name} path={destination}", flush=True)


if __name__ == "__main__":
    main()
