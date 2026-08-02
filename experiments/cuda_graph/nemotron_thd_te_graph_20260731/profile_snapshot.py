#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read a trusted cluster profile once and expose only literal assignments."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


CLUSTERS = frozenset({"ptyche", "oci-hsg", "lyris"})
PROFILE_FIELDS = (
    "PROFILE_ID", "ACCOUNT", "PARTITION", "CONTAINER", "CONTAINER_SHA256",
    "MOUNTS", "SBATCH_GPUS_PER_NODE", "SBATCH_GRES", "SBATCH_SEGMENT_SIZE",
    "TIME_LIMIT", "RUNTIME_ATTESTATION", "RUNTIME_PREFLIGHT_JOB_ID",
    "EXPECTED_TE_SHA", "EXPECTED_NEMORL_SHA", "EXPECTED_BRIDGE_SHA", "EXPECTED_MCORE_SHA",
)
PROFILE_ASSIGNMENT = re.compile(r"([A-Z][A-Z0-9_]*)=([A-Za-z0-9_./,:=-]*)\Z")


@dataclass(frozen=True)
class ProfileSnapshot:
    sha256: str
    values: dict[str, str]


def _read_regular_file(path: Path, label: str) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened as a non-symlink file: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        content = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            content.extend(chunk)
        return bytes(content)
    finally:
        os.close(descriptor)


def _validate_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("profile directory must be an absolute path")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"profile directory is not trusted: {error}") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("profile directory must be a directory")
    finally:
        os.close(descriptor)


def load_profile_snapshot(profile_dir: Path, cluster: str, profile_file: str | None) -> ProfileSnapshot:
    if cluster not in CLUSTERS:
        raise ValueError("CLUSTER must be ptyche, oci-hsg, or lyris")
    _validate_directory(profile_dir)
    candidate = Path(profile_file) if profile_file else profile_dir / f"{cluster}.env"
    if profile_file is None and not os.path.lexists(candidate):
        candidate = profile_dir / f"{cluster}.env.example"
    if not candidate.is_absolute() or candidate.parent != profile_dir:
        raise ValueError("profile file must be a direct child of the trusted profile directory")
    try:
        text = _read_regular_file(candidate, "profile file").decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"profile is not valid UTF-8: {error}") from error
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        match = PROFILE_ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"profile line {number} must be a literal NAME=value assignment")
        name, value = match.groups()
        if name not in PROFILE_FIELDS or name in values:
            raise ValueError(f"profile line {number} has an unknown or duplicate field")
        values[name] = value
    return ProfileSnapshot(hashlib.sha256(text.encode()).hexdigest(), values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    try:
        snapshot = load_profile_snapshot(Path(args.profile_dir), args.cluster, args.profile_file)
        if args.expected_sha256 and snapshot.sha256 != args.expected_sha256:
            raise ValueError("profile SHA256 does not match validated profile")
    except ValueError as error:
        print(f"Profile rejected: {error}", file=sys.stderr)
        return 2
    print(f"PROFILE_SHA256\t{snapshot.sha256}")
    for field in PROFILE_FIELDS:
        print(f"{field}\t{snapshot.values.get(field, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
