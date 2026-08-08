#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import errno
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


TRANSIENT_LUSTRE_ERRNOS = frozenset(
    {errno.EAGAIN, errno.EINTR, errno.EIO, errno.ESTALE, errno.ETIMEDOUT}
)
MAX_ATTEMPTS = 6
ChmodFn = Callable[[Path, int], None]
LstatFn = Callable[[os.PathLike[str] | str], os.stat_result]
SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class _FilesystemFailure:
    path: Path
    error: OSError


@dataclass(frozen=True)
class _WritablePath:
    path: Path
    mode: int


@dataclass(frozen=True)
class _ScanResult:
    writable_paths: tuple[_WritablePath, ...]
    transient_failures: tuple[_FilesystemFailure, ...]


def _tree_paths_bottom_up(root: Path) -> Iterator[Path]:
    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, child_directories, filenames in os.walk(
        root, topdown=False, onerror=raise_walk_error, followlinks=False
    ):
        parent = Path(directory)
        yield from (parent / filename for filename in filenames)
        yield from (parent / child for child in child_directories)
    yield root


def _is_transient(error: OSError) -> bool:
    return error.errno in TRANSIENT_LUSTRE_ERRNOS


def _chmod_path_no_follow(path: Path, mode: int) -> None:
    try:
        no_follow = os.O_NOFOLLOW
    except AttributeError as error:
        raise RuntimeError(
            "O_NOFOLLOW is required for safe stage finalization"
        ) from error
    flags = os.O_RDONLY | no_follow
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    file_descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(file_descriptor)
        if not (stat.S_ISREG(opened_stat.st_mode) or stat.S_ISDIR(opened_stat.st_mode)):
            raise ValueError(f"Finalization path changed type before chmod: {path}")
        os.fchmod(file_descriptor, mode)
    finally:
        os.close(file_descriptor)


def _canonical_trusted_roots(
    root: Path,
    trusted_symlink_roots: Sequence[Path],
    lstat_fn: LstatFn,
) -> tuple[tuple[Path, ...], tuple[_FilesystemFailure, ...]]:
    canonical_roots: list[Path] = []
    transient_failures: list[_FilesystemFailure] = []
    for path in dict.fromkeys((root, *trusted_symlink_roots)):
        try:
            path_stat = lstat_fn(path)
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
                raise ValueError(
                    f"Trusted symlink root must be a regular directory: {path}"
                )
            canonical_roots.append(path.resolve(strict=True))
        except OSError as error:
            if not _is_transient(error):
                raise
            transient_failures.append(_FilesystemFailure(path, error))
    return tuple(canonical_roots), tuple(transient_failures)


def _validate_symlink(
    path: Path, canonical_trusted_roots: Sequence[Path]
) -> _FilesystemFailure | None:
    try:
        resolved_path = path.resolve(strict=True)
    except RuntimeError as error:
        raise ValueError(f"Runtime stage contains a broken symlink: {path}") from error
    except OSError as error:
        if _is_transient(error):
            return _FilesystemFailure(path, error)
        if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise ValueError(
                f"Runtime stage contains a broken symlink: {path}"
            ) from error
        raise
    if not any(
        resolved_path == trusted_root or resolved_path.is_relative_to(trusted_root)
        for trusted_root in canonical_trusted_roots
    ):
        raise ValueError(f"Runtime stage symlink escapes trusted roots: {path}")
    return None


def _scan_writable_paths(
    root: Path,
    regular_files: Sequence[Path],
    trusted_symlink_roots: Sequence[Path],
    lstat_fn: LstatFn,
) -> _ScanResult:
    writable_paths: list[_WritablePath] = []
    transient_failures: list[_FilesystemFailure] = []

    try:
        root_stat = lstat_fn(root)
    except OSError as error:
        if not _is_transient(error):
            raise
        return _ScanResult((), (_FilesystemFailure(root, error),))
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("Runtime stage root must be a regular directory")

    canonical_trusted_roots, root_failures = _canonical_trusted_roots(
        root, trusted_symlink_roots, lstat_fn
    )
    if root_failures:
        return _ScanResult((), root_failures)

    try:
        tree_paths = _tree_paths_bottom_up(root)
        for path in tree_paths:
            try:
                path_stat = lstat_fn(path)
            except OSError as error:
                if not _is_transient(error):
                    raise
                transient_failures.append(_FilesystemFailure(path, error))
                continue
            if stat.S_ISLNK(path_stat.st_mode):
                symlink_failure = _validate_symlink(path, canonical_trusted_roots)
                if symlink_failure is not None:
                    transient_failures.append(symlink_failure)
                continue
            if not (stat.S_ISREG(path_stat.st_mode) or stat.S_ISDIR(path_stat.st_mode)):
                continue
            mode = stat.S_IMODE(path_stat.st_mode)
            if mode & 0o222:
                writable_paths.append(_WritablePath(path, mode))
    except OSError as error:
        if not _is_transient(error):
            raise
        failed_path = Path(error.filename) if error.filename else root
        transient_failures.append(_FilesystemFailure(failed_path, error))

    for path in regular_files:
        try:
            path_stat = lstat_fn(path)
        except OSError as error:
            if not _is_transient(error):
                raise
            transient_failures.append(_FilesystemFailure(path, error))
            continue
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"Extra finalization path must be a regular file: {path}")
        mode = stat.S_IMODE(path_stat.st_mode)
        if mode & 0o222:
            writable_paths.append(_WritablePath(path, mode))

    return _ScanResult(tuple(writable_paths), tuple(transient_failures))


def _remove_write_bits(
    writable_paths: Sequence[_WritablePath], chmod_fn: ChmodFn
) -> tuple[_FilesystemFailure, ...]:
    transient_failures: list[_FilesystemFailure] = []
    for writable_path in writable_paths:
        try:
            chmod_fn(writable_path.path, writable_path.mode & ~0o222)
        except OSError as error:
            if not _is_transient(error):
                raise
            transient_failures.append(_FilesystemFailure(writable_path.path, error))
    return tuple(transient_failures)


def _bounded_samples(paths: Sequence[Path]) -> str:
    return ", ".join(str(path) for path in paths[:3])


def make_tree_readonly(
    root: Path,
    *,
    regular_files: Sequence[Path] = (),
    trusted_symlink_roots: Sequence[Path] = (),
    max_attempts: int = MAX_ATTEMPTS,
    initial_delay_seconds: float = 1.0,
    max_delay_seconds: float = 8.0,
    chmod_fn: ChmodFn | None = None,
    lstat_fn: LstatFn | None = None,
    sleep_fn: SleepFn | None = None,
) -> None:
    if not root.is_absolute():
        raise ValueError("Runtime stage root must be absolute")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if initial_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("Retry delays must be nonnegative")
    if any(not path.is_absolute() for path in regular_files):
        raise ValueError("Extra finalization paths must be absolute")
    if any(not path.is_absolute() for path in trusted_symlink_roots):
        raise ValueError("Trusted symlink roots must be absolute")

    chmod_fn = chmod_fn or _chmod_path_no_follow
    lstat_fn = lstat_fn or os.lstat
    sleep_fn = sleep_fn or time.sleep

    last_failure_paths: list[Path] = []
    last_errno_names: set[str] = set()
    for attempt in range(1, max_attempts + 1):
        initial_scan = _scan_writable_paths(
            root, regular_files, trusted_symlink_roots, lstat_fn
        )
        chmod_failures = _remove_write_bits(initial_scan.writable_paths, chmod_fn)
        verification = _scan_writable_paths(
            root, regular_files, trusted_symlink_roots, lstat_fn
        )
        failures = (*chmod_failures, *verification.transient_failures)
        if not failures and not verification.writable_paths:
            return

        last_failure_paths = [failure.path for failure in failures]
        last_failure_paths.extend(
            writable_path.path for writable_path in verification.writable_paths
        )
        last_errno_names = {
            errno.errorcode.get(failure.error.errno or 0, "UNKNOWN")
            for failure in failures
        }
        if attempt < max_attempts:
            print(
                "Runtime stage read-only finalization encountered "
                f"{len(failures)} transient Lustre errors and "
                f"{len(verification.writable_paths)} writable paths on attempt "
                f"{attempt}/{max_attempts}; retrying; errno="
                f"{','.join(sorted(last_errno_names)) or 'none'}; sample paths: "
                f"{_bounded_samples(last_failure_paths)}",
                file=sys.stderr,
            )
            sleep_fn(
                min(initial_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            )

    raise RuntimeError(
        "Runtime stage read-only finalization exhausted "
        f"{max_attempts} attempts with errno="
        f"{','.join(sorted(last_errno_names)) or 'none'}; sample paths: "
        f"{_bounded_samples(last_failure_paths)}"
    )


def verify_tree_readonly(
    root: Path,
    *,
    regular_files: Sequence[Path] = (),
    trusted_symlink_roots: Sequence[Path] = (),
    max_attempts: int = MAX_ATTEMPTS,
    initial_delay_seconds: float = 1.0,
    max_delay_seconds: float = 8.0,
    lstat_fn: LstatFn | None = None,
    sleep_fn: SleepFn | None = None,
) -> None:
    if not root.is_absolute():
        raise ValueError("Runtime stage root must be absolute")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if initial_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("Retry delays must be nonnegative")
    if any(not path.is_absolute() for path in regular_files):
        raise ValueError("Extra verification paths must be absolute")
    if any(not path.is_absolute() for path in trusted_symlink_roots):
        raise ValueError("Trusted symlink roots must be absolute")

    lstat_fn = lstat_fn or os.lstat
    sleep_fn = sleep_fn or time.sleep
    last_failure_paths: list[Path] = []
    last_errno_names: set[str] = set()
    for attempt in range(1, max_attempts + 1):
        verification = _scan_writable_paths(
            root, regular_files, trusted_symlink_roots, lstat_fn
        )
        if not verification.transient_failures:
            if verification.writable_paths:
                writable_paths = [
                    writable_path.path for writable_path in verification.writable_paths
                ]
                raise RuntimeError(
                    "Runtime stage contains writable regular state: "
                    f"{_bounded_samples(writable_paths)}"
                )
            return

        last_failure_paths = [
            failure.path for failure in verification.transient_failures
        ]
        last_errno_names = {
            errno.errorcode.get(failure.error.errno or 0, "UNKNOWN")
            for failure in verification.transient_failures
        }
        if attempt < max_attempts:
            print(
                "Runtime stage read-only verification encountered "
                f"{len(verification.transient_failures)} transient Lustre errors "
                f"on attempt {attempt}/{max_attempts}; retrying; errno="
                f"{','.join(sorted(last_errno_names)) or 'none'}; sample paths: "
                f"{_bounded_samples(last_failure_paths)}",
                file=sys.stderr,
            )
            sleep_fn(
                min(initial_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            )

    raise RuntimeError(
        "Runtime stage read-only verification exhausted "
        f"{max_attempts} attempts with errno="
        f"{','.join(sorted(last_errno_names)) or 'none'}; sample paths: "
        f"{_bounded_samples(last_failure_paths)}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regular-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--trusted-symlink-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        operation = verify_tree_readonly if args.verify_only else make_tree_readonly
        operation(
            args.root,
            regular_files=tuple(args.regular_file),
            trusted_symlink_roots=tuple(args.trusted_symlink_root),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"Runtime stage read-only finalization failed: {error}",
            file=sys.stderr,
        )
        return 2
    print(f"RUNTIME_STAGE_READONLY_READY={args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
