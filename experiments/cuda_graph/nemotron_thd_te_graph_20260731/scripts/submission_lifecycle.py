#!/usr/bin/env python3
"""Content-addressed source snapshots and transactional submission intents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SubmissionMode(StrEnum):
    TEST_ONLY = "test_only"
    SBATCH_TEST_ONLY = "sbatch_test_only"
    ACTUAL = "actual"


@dataclass(frozen=True)
class ArchiveSource:
    repository: Path
    commit: str
    relative_destination: Path


@dataclass(frozen=True)
class SubmissionArtifacts:
    snapshot_root: Path
    snapshot_sha256: str
    intent_path: Path
    intent_sha256: str


@dataclass(frozen=True)
class _OwnedIntent:
    artifact_root: Path
    parent: Path
    device: int
    inode: int


_OWNED_INTENTS: dict[Path, _OwnedIntent] = {}


@dataclass
class SubmissionTransaction:
    artifacts: SubmissionArtifacts
    artifact_root: Path
    mode: SubmissionMode
    snapshot_created: bool
    _scheduler_accepted: bool = False
    _closed: bool = False

    def commit_scheduler_acceptance(self) -> None:
        if self._closed:
            raise RuntimeError("submission transaction is closed")
        self._scheduler_accepted = True

    def close(self) -> None:
        if self._closed:
            return
        if not self._scheduler_accepted:
            remove_owned_intent(self.artifacts.intent_path)
        self._closed = True

    def __enter__(self) -> SubmissionTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == ".snapshot-sha256":
            continue
        metadata = path.lstat()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(f"exec={stat.S_IMODE(metadata.st_mode) & 0o111:o}".encode())
        digest.update(b"\0")
        if stat.S_ISREG(metadata.st_mode):
            digest.update(b"file\0")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        elif stat.S_ISDIR(metadata.st_mode):
            digest.update(b"directory\0")
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            if not (path.parent / target).resolve().is_relative_to(root.resolve()):
                raise ValueError(f"snapshot contains an escaping symlink: {relative}")
            digest.update(b"symlink\0")
            digest.update(target.encode())
        else:
            raise ValueError(f"snapshot contains an unsupported file type: {relative}")
        digest.update(b"\0")
    return digest.hexdigest()


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _remove_temporary_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IWUSR)
    shutil.rmtree(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_claim(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_directory(path: Path) -> None:
    """Create a directory tree without accepting an existing symlink component."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"artifact path contains a symlink or non-directory: {current}")


def _safe_destination(root: Path, relative: Path) -> Path:
    destination = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        destination /= part
        try:
            metadata = destination.lstat()
        except FileNotFoundError:
            destination.mkdir()
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"archive destination contains a symlink: {destination}")
    return destination


def _safe_merge_tree(source: Path, destination: Path) -> None:
    """Copy an extracted archive without following destination symlinks."""
    _safe_destination(destination.parent, Path(destination.name))
    for root, directories, files in os.walk(source, followlinks=False):
        source_root = Path(root)
        relative_root = source_root.relative_to(source)
        target_root = _safe_destination(destination, relative_root)
        for name in directories:
            source_path = source_root / name
            target_path = target_root / name
            metadata = source_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                directories.remove(name)
                if target_path.exists() or target_path.is_symlink():
                    raise ValueError(f"archive destination already exists: {target_path}")
                target_path.symlink_to(os.readlink(source_path))
            else:
                _safe_destination(target_root, Path(name))
        for name in files:
            source_path = source_root / name
            target_path = target_root / name
            if target_path.exists() or target_path.is_symlink():
                raise ValueError(f"archive destination already exists: {target_path}")
            metadata = source_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                target_path.symlink_to(os.readlink(source_path))
            elif stat.S_ISREG(metadata.st_mode):
                shutil.copyfile(source_path, target_path, follow_symlinks=False)
                target_path.chmod(stat.S_IMODE(metadata.st_mode))
            else:
                raise ValueError(f"snapshot contains an unsupported file type: {source_path}")


def _write_file(path: Path, contents: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb", closefd=True) as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted((root, *(item for item in root.rglob("*") if item.is_dir())), reverse=True):
        _fsync_directory(path)


def _archive_commit(repository: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", "--format=tar", commit], stdout=subprocess.PIPE)
    assert archive.stdout is not None
    try:
        extracted = subprocess.run(["tar", "-xf", "-", "-C", str(destination)], stdin=archive.stdout, check=False)
    finally:
        archive.stdout.close()
    if archive.wait() != 0 or extracted.returncode != 0:
        raise RuntimeError(f"failed to archive {repository} at {commit}")


def verify_source_snapshot(*, source_root: Path, candidate_sha: str, expected_sha256: str) -> None:
    if FULL_COMMIT.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be a full lowercase 40-character SHA")
    if FULL_SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("snapshot SHA256 must be lowercase hexadecimal")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("candidate source root is missing or unsafe")
    for path in (source_root, *source_root.rglob("*")):
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode) and metadata.st_mode & 0o222:
            raise ValueError(f"snapshot contains a writable path: {path}")
    marker = source_root / ".candidate-sha"
    digest_marker = source_root / ".snapshot-sha256"
    if not marker.is_file() or marker.read_text().strip() != candidate_sha:
        raise ValueError("candidate source snapshot does not match candidate SHA")
    if not digest_marker.is_file() or digest_marker.read_text().strip() != expected_sha256:
        raise ValueError("snapshot SHA256 marker mismatch")
    if _directory_sha256(source_root) != expected_sha256:
        raise ValueError("snapshot SHA256 does not match snapshot contents")


def load_submission_intent(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if FULL_SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("submission intent SHA256 must be lowercase hexadecimal")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("submission intent must be a non-writable regular file")
    serialized = path.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != expected_sha256:
        raise ValueError("submission intent SHA256 does not match intent contents")
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError("submission intent is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("submission intent must be a JSON object")
    return payload


def _validate_request(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, mode: SubmissionMode, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str) -> None:
    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise ValueError("artifact root must be an absolute non-symlink path")
    if not isinstance(mode, SubmissionMode):
        raise ValueError("submission mode is invalid")
    if candidate_kind not in {"mcore", "bridge"}:
        raise ValueError("candidate kind must be mcore or bridge")
    if FULL_COMMIT.fullmatch(candidate_sha) is None:
        raise ValueError("candidate SHA must be a full lowercase 40-character SHA")
    if not archive_sources:
        raise ValueError("at least one archive source is required")
    for source in archive_sources:
        if FULL_COMMIT.fullmatch(source.commit) is None:
            raise ValueError("archive source commit must be a full lowercase SHA")
        if source.relative_destination.is_absolute() or ".." in source.relative_destination.parts:
            raise ValueError("archive destination must stay inside the snapshot")


def _publish_snapshot(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str) -> tuple[Path, str, bool]:
    parent = artifact_root / "source-snapshots" / candidate_kind / candidate_sha
    _safe_directory(parent)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{candidate_sha}.", suffix=".tmp", dir=parent))
    try:
        for source in archive_sources:
            extracted_root = Path(tempfile.mkdtemp(prefix=".archive.", suffix=".tmp", dir=parent))
            try:
                _archive_commit(source.repository, source.commit, extracted_root)
                _safe_merge_tree(extracted_root, temporary_root / source.relative_destination)
            finally:
                if extracted_root.exists():
                    _remove_temporary_tree(extracted_root)
        candidate_marker = temporary_root / ".candidate-sha"
        digest_marker = temporary_root / ".snapshot-sha256"
        if candidate_marker.exists() or candidate_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved snapshot marker")
        if digest_marker.exists() or digest_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved digest marker")
        _write_file(candidate_marker, f"{candidate_sha}\n".encode(), 0o600)
        snapshot_sha256 = _directory_sha256(temporary_root)
        _write_file(digest_marker, f"{snapshot_sha256}\n".encode(), 0o600)
        _fsync_tree(temporary_root)
        final_root = parent / snapshot_sha256
        if final_root.exists() or final_root.is_symlink():
            verify_source_snapshot(source_root=final_root, candidate_sha=candidate_sha, expected_sha256=snapshot_sha256)
            return final_root, snapshot_sha256, False
        claim_path = parent / f".{snapshot_sha256}.claim"
        try:
            descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if final_root.exists() or final_root.is_symlink():
                    verify_source_snapshot(source_root=final_root, candidate_sha=candidate_sha, expected_sha256=snapshot_sha256)
                    return final_root, snapshot_sha256, False
                time.sleep(0.01)
            raise ValueError("snapshot publication claim did not produce a snapshot")
        else:
            try:
                _fsync_claim(descriptor)
            except BaseException:
                os.close(descriptor)
                claim_path.unlink(missing_ok=True)
                _fsync_directory(parent)
                raise
            os.close(descriptor)
        try:
            try:
                os.mkdir(final_root, 0o700)
            except FileExistsError:
                verify_source_snapshot(source_root=final_root, candidate_sha=candidate_sha, expected_sha256=snapshot_sha256)
                return final_root, snapshot_sha256, False
            final_metadata = final_root.lstat()
            if stat.S_ISLNK(final_metadata.st_mode) or not stat.S_ISDIR(final_metadata.st_mode):
                raise ValueError("snapshot final destination is unsafe")
            final_descriptor = os.open(final_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            temporary_descriptor = os.open(temporary_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for child in temporary_root.iterdir():
                    os.rename(
                        child.name,
                        child.name,
                        src_dir_fd=temporary_descriptor,
                        dst_dir_fd=final_descriptor,
                    )
                _make_tree_read_only(final_root)
                os.fsync(final_descriptor)
            finally:
                os.close(temporary_descriptor)
                os.close(final_descriptor)
            _fsync_directory(parent)
            published_metadata = final_root.lstat()
            if (published_metadata.st_dev, published_metadata.st_ino) != (
                final_metadata.st_dev,
                final_metadata.st_ino,
            ):
                raise ValueError("snapshot final destination changed during publication")
            verify_source_snapshot(source_root=final_root, candidate_sha=candidate_sha, expected_sha256=snapshot_sha256)
            return final_root, snapshot_sha256, True
        finally:
            claim_path.unlink(missing_ok=True)
            _fsync_directory(parent)
    finally:
        if temporary_root.exists():
            _remove_temporary_tree(temporary_root)


def _publish_intent(*, artifact_root: Path, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str, snapshot_root: Path, snapshot_sha256: str, intent_payload: Mapping[str, object]) -> SubmissionArtifacts:
    payload = dict(intent_payload)
    payload["snapshot_path"] = str(snapshot_root)
    payload["snapshot_sha256"] = snapshot_sha256
    serialized = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    intent_sha256 = hashlib.sha256(serialized).hexdigest()
    parent = artifact_root / "submission-intents" / candidate_kind / candidate_sha
    _safe_directory(parent)
    root = artifact_root.resolve()
    resolved_parent = parent.resolve()
    if not resolved_parent.is_relative_to(root / "submission-intents"):
        raise ValueError("submission intent escaped artifact root")
    submission_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
    intent_path = parent / f"{submission_id}.json"
    temporary_intent = parent / f".{submission_id}.tmp"
    try:
        _write_file(temporary_intent, serialized, 0o600)
        temporary_intent.chmod(0o444)
        _fsync_path(temporary_intent)
        _fsync_directory(parent)
        os.link(temporary_intent, intent_path)
    finally:
        temporary_intent.unlink(missing_ok=True)
        _fsync_directory(parent)
    metadata = intent_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("submission intent publication is unsafe")
    _OWNED_INTENTS[intent_path.absolute()] = _OwnedIntent(root, resolved_parent, metadata.st_dev, metadata.st_ino)
    try:
        _fsync_directory(parent)
        load_submission_intent(intent_path, expected_sha256=intent_sha256)
    except BaseException:
        remove_owned_intent(intent_path)
        raise
    return SubmissionArtifacts(snapshot_root, snapshot_sha256, intent_path, intent_sha256)


def remove_owned_intent(path: Path) -> None:
    absolute_path = path.absolute()
    owned = _OWNED_INTENTS.get(absolute_path)
    if owned is None:
        raise ValueError("submission intent is not owned by this transaction")
    if absolute_path.parent.resolve() != owned.parent or not owned.parent.is_relative_to(owned.artifact_root / "submission-intents"):
        raise ValueError("submission intent escaped its owned parent")
    metadata = absolute_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != (owned.device, owned.inode):
        raise ValueError("submission intent is not the owned immutable file")
    absolute_path.unlink()
    _fsync_directory(owned.parent)
    del _OWNED_INTENTS[absolute_path]


def prepare_candidate_submission(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, mode: SubmissionMode, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str, intent_payload: Mapping[str, object]) -> SubmissionTransaction:
    _validate_request(archive_sources=archive_sources, artifact_root=artifact_root, mode=mode, candidate_kind=candidate_kind, candidate_sha=candidate_sha)
    snapshot_root, snapshot_sha256, snapshot_created = _publish_snapshot(archive_sources=archive_sources, artifact_root=artifact_root, candidate_kind=candidate_kind, candidate_sha=candidate_sha)
    artifacts = _publish_intent(artifact_root=artifact_root, candidate_kind=candidate_kind, candidate_sha=candidate_sha, snapshot_root=snapshot_root, snapshot_sha256=snapshot_sha256, intent_payload=intent_payload)
    return SubmissionTransaction(artifacts, artifact_root, mode, snapshot_created)
