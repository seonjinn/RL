#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Resolve a valid SLURM segment size for one concrete allocation."""

from __future__ import annotations

import argparse
import sys


CLUSTERS = frozenset({"ptyche", "oci-hsg", "lyris"})
PTYCHE_PORTABLE_MAX_SEGMENT = 18


def resolve_segment_size(cluster: str, num_nodes: int, configured: str) -> int | None:
    """Return a validated explicit segment or a ptyche allocation-derived value."""
    if cluster not in CLUSTERS:
        raise ValueError("cluster must be ptyche, oci-hsg, or lyris")
    if num_nodes < 1:
        raise ValueError("requested node count must be positive")

    if configured:
        try:
            segment_size = int(configured)
        except ValueError as error:
            raise ValueError(
                "configured segment size must be a positive integer"
            ) from error
        if segment_size < 1:
            raise ValueError("configured segment size must be a positive integer")
        if cluster == "ptyche" and segment_size > PTYCHE_PORTABLE_MAX_SEGMENT:
            raise ValueError("ptyche segment size must be at most 18 nodes")
        if segment_size > num_nodes or num_nodes % segment_size:
            raise ValueError(
                "configured segment size must divide the requested node count"
            )
        return segment_size

    if cluster != "ptyche":
        return None

    for segment_size in range(min(num_nodes, PTYCHE_PORTABLE_MAX_SEGMENT), 0, -1):
        if num_nodes % segment_size == 0:
            return segment_size
    raise AssertionError("positive node counts always have a segment divisor")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--configured", default="")
    args = parser.parse_args()
    try:
        segment_size = resolve_segment_size(
            args.cluster, args.num_nodes, args.configured
        )
    except ValueError as error:
        print(f"SLURM segment rejected: {error}", file=sys.stderr)
        return 2
    if segment_size is not None:
        print(segment_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
