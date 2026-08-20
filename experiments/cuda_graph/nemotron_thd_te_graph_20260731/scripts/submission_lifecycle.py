#!/usr/bin/env python3
"""Content-addressed source snapshots and transactional submission intents."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RENAME_NOREPLACE = 1
RENAME_EXCL = 0x00000004


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
    parent_device: int
    parent_inode: int
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
    descriptor = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        _make_tree_read_only_at(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_read_only_at(directory_descriptor: int) -> None:
    with os.scandir(directory_descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        metadata = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if stat.S_ISLNK(metadata.st_mode):
            continue
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if stat.S_ISDIR(metadata.st_mode):
            flags |= os.O_DIRECTORY
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        try:
            if stat.S_ISDIR(metadata.st_mode):
                _make_tree_read_only_at(descriptor)
            else:
                os.fchmod(
                    descriptor, stat.S_IMODE(metadata.st_mode) & ~0o222
                )
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
    metadata = os.fstat(directory_descriptor)
    os.fchmod(
        directory_descriptor, stat.S_IMODE(metadata.st_mode) & ~0o222
    )
    os.fsync(directory_descriptor)


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


def _open_directory_at(parent_descriptor: int, name: str, *, create: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileExistsError:
            pass
        try:
            return os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ValueError(
                f"artifact path contains a symlink or non-directory: {name}"
            ) from error
    except OSError as error:
        raise ValueError(
            f"artifact path contains a symlink or non-directory: {name}"
        ) from error


def _open_safe_directory(path: Path, *, create: bool) -> int:
    if not path.is_absolute():
        raise ValueError("artifact directory must be absolute")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_descriptor = _open_directory_at(
                descriptor, part, create=create
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(
    root_descriptor: int, relative: Path, *, create: bool
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for part in relative.parts:
            if part in {"", "."}:
                continue
            next_descriptor = _open_directory_at(
                descriptor, part, create=create
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _descriptor_path(descriptor: int) -> Path:
    proc_path = Path(f"/proc/self/fd/{descriptor}")
    try:
        return Path(os.readlink(proc_path))
    except OSError:
        get_path = getattr(fcntl, "F_GETPATH", None)
        if get_path is None:
            raise RuntimeError("cannot resolve retained directory descriptor")
        value = fcntl.fcntl(descriptor, get_path, b"\0" * 1024)
        return Path(value.split(b"\0", 1)[0].decode())


def _rename_noreplace(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        symbol = "renameat2"
        flag = RENAME_NOREPLACE
    elif sys.platform == "darwin":
        symbol = "renameatx_np"
        flag = RENAME_EXCL
    else:
        raise OSError(
            errno.ENOSYS,
            f"no no-replace rename primitive for {sys.platform}",
        )
    try:
        rename = getattr(library, symbol)
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            f"no no-replace rename primitive: {symbol}",
        ) from error
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        source_parent_descriptor,
        os.fsencode(source_name),
        destination_parent_descriptor,
        os.fsencode(destination_name),
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number, os.strerror(error_number), destination_name
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _create_private_directory(
    *, parent_descriptor: int, prefix: str, suffix: str
) -> tuple[Path, int]:
    for _ in range(100):
        name = f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = _open_directory_at(
            parent_descriptor, name, create=False
        )
        metadata = os.fstat(descriptor)
        named_metadata = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (metadata.st_dev, metadata.st_ino) != (
            named_metadata.st_dev,
            named_metadata.st_ino,
        ):
            os.close(descriptor)
            raise ValueError("private publication directory changed after creation")
        os.fsync(parent_descriptor)
        return _descriptor_path(descriptor), descriptor
    raise FileExistsError("could not allocate a private publication directory")


def _safe_merge_tree(
    source: Path, *, destination_descriptor: int
) -> None:
    """Copy an extracted archive without following destination symlinks."""
    try:
        for root, directories, files in os.walk(source, followlinks=False):
            source_root = Path(root)
            relative_root = source_root.relative_to(source)
            target_descriptor = os.dup(destination_descriptor)
            try:
                for part in relative_root.parts:
                    if part in {"", "."}:
                        continue
                    next_descriptor = _open_directory_at(
                        target_descriptor, part, create=True
                    )
                    os.close(target_descriptor)
                    target_descriptor = next_descriptor
                for name in list(directories):
                    source_path = source_root / name
                    metadata = source_path.lstat()
                    if stat.S_ISLNK(metadata.st_mode):
                        directories.remove(name)
                        try:
                            os.symlink(
                                os.readlink(source_path),
                                name,
                                dir_fd=target_descriptor,
                            )
                        except FileExistsError as error:
                            raise ValueError(
                                f"archive destination already exists: {name}"
                            ) from error
                    else:
                        child_descriptor = _open_directory_at(
                            target_descriptor, name, create=True
                        )
                        os.close(child_descriptor)
                for name in files:
                    source_path = source_root / name
                    metadata = source_path.lstat()
                    if stat.S_ISLNK(metadata.st_mode):
                        try:
                            os.symlink(
                                os.readlink(source_path),
                                name,
                                dir_fd=target_descriptor,
                            )
                        except FileExistsError as error:
                            raise ValueError(
                                f"archive destination already exists: {name}"
                            ) from error
                    elif stat.S_ISREG(metadata.st_mode):
                        try:
                            output_descriptor = os.open(
                                name,
                                os.O_WRONLY
                                | os.O_CREAT
                                | os.O_EXCL
                                | os.O_NOFOLLOW,
                                stat.S_IMODE(metadata.st_mode),
                                dir_fd=target_descriptor,
                            )
                        except FileExistsError as error:
                            raise ValueError(
                                f"archive destination already exists: {name}"
                            ) from error
                        try:
                            with open(source_path, "rb") as source_file:
                                with os.fdopen(
                                    output_descriptor, "wb", closefd=False
                                ) as output_file:
                                    shutil.copyfileobj(source_file, output_file)
                                    output_file.flush()
                            os.fchmod(
                                output_descriptor,
                                stat.S_IMODE(metadata.st_mode),
                            )
                        finally:
                            os.close(output_descriptor)
                    else:
                        raise ValueError(
                            f"snapshot contains an unsupported file type: {source_path}"
                        )
            finally:
                os.close(target_descriptor)
    finally:
        os.close(destination_descriptor)


def _write_file_at(
    *, parent_descriptor: int, name: str, contents: bytes, mode: int
) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
        dir_fd=parent_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


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


def _open_named_directory(parent_descriptor: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError("snapshot final destination is unsafe") from error


def _verify_named_directory(
    *, parent_descriptor: int, name: str, expected_identity: tuple[int, int]
) -> None:
    descriptor = _open_named_directory(parent_descriptor, name)
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise ValueError("snapshot final destination changed during publication")


def _verify_named_snapshot(
    *,
    parent_descriptor: int,
    name: str,
    path: Path,
    candidate_sha: str,
    snapshot_sha256: str,
) -> None:
    descriptor = _open_named_directory(parent_descriptor, name)
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        verify_source_snapshot(
            source_root=path,
            candidate_sha=candidate_sha,
            expected_sha256=snapshot_sha256,
        )
        _verify_named_directory(
            parent_descriptor=parent_descriptor,
            name=name,
            expected_identity=identity,
        )
    finally:
        os.close(descriptor)


def _unlink_owned_entry(
    *, parent_descriptor: int, name: str, expected_identity: tuple[int, int]
) -> None:
    try:
        metadata = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise ValueError("owned publication entry changed before cleanup")
    os.unlink(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _publish_snapshot(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str) -> tuple[Path, str, bool]:
    parent = artifact_root / "source-snapshots" / candidate_kind / candidate_sha
    parent_descriptor = _open_safe_directory(parent, create=True)
    temporary_root, temporary_root_descriptor = _create_private_directory(
        parent_descriptor=parent_descriptor,
        prefix=f".{candidate_sha}.",
        suffix=".tmp",
    )
    try:
        for source in archive_sources:
            extracted_root, extracted_descriptor = _create_private_directory(
                parent_descriptor=parent_descriptor,
                prefix=".archive.",
                suffix=".tmp",
            )
            try:
                _archive_commit(source.repository, source.commit, extracted_root)
                destination_descriptor = _open_relative_directory(
                    temporary_root_descriptor,
                    source.relative_destination,
                    create=True,
                )
                _safe_merge_tree(
                    extracted_root,
                    destination_descriptor=destination_descriptor,
                )
            finally:
                if extracted_root.exists():
                    _remove_temporary_tree(extracted_root)
                    os.fsync(parent_descriptor)
                os.close(extracted_descriptor)
        candidate_marker = temporary_root / ".candidate-sha"
        digest_marker = temporary_root / ".snapshot-sha256"
        if candidate_marker.exists() or candidate_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved snapshot marker")
        if digest_marker.exists() or digest_marker.is_symlink():
            raise ValueError("candidate archive contains a reserved digest marker")
        _write_file_at(
            parent_descriptor=temporary_root_descriptor,
            name=candidate_marker.name,
            contents=f"{candidate_sha}\n".encode(),
            mode=0o600,
        )
        snapshot_sha256 = _directory_sha256(temporary_root)
        _write_file_at(
            parent_descriptor=temporary_root_descriptor,
            name=digest_marker.name,
            contents=f"{snapshot_sha256}\n".encode(),
            mode=0o600,
        )
        _fsync_tree(temporary_root)
        final_root = parent / snapshot_sha256
        try:
            existing_descriptor = _open_named_directory(
                parent_descriptor, snapshot_sha256
            )
        except ValueError:
            try:
                os.stat(
                    snapshot_sha256,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise
        else:
            os.close(existing_descriptor)
            _verify_named_snapshot(
                parent_descriptor=parent_descriptor,
                name=snapshot_sha256,
                path=final_root,
                candidate_sha=candidate_sha,
                snapshot_sha256=snapshot_sha256,
            )
            return final_root, snapshot_sha256, False
        claim_name = f".{snapshot_sha256}.claim"
        claim_descriptor: int | None = None
        claim_identity: tuple[int, int] | None = None
        try:
            try:
                claim_descriptor = os.open(
                    claim_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    try:
                        winner_descriptor = _open_named_directory(
                            parent_descriptor, snapshot_sha256
                        )
                    except ValueError:
                        time.sleep(0.01)
                        continue
                    os.close(winner_descriptor)
                    _verify_named_snapshot(
                        parent_descriptor=parent_descriptor,
                        name=snapshot_sha256,
                        path=final_root,
                        candidate_sha=candidate_sha,
                        snapshot_sha256=snapshot_sha256,
                    )
                    return final_root, snapshot_sha256, False
                raise ValueError(
                    "snapshot publication claim did not produce a snapshot"
                )
            claim_metadata = os.fstat(claim_descriptor)
            claim_identity = (claim_metadata.st_dev, claim_metadata.st_ino)
            _fsync_claim(claim_descriptor)
            os.close(claim_descriptor)
            claim_descriptor = None
            _make_tree_read_only_at(temporary_root_descriptor)
            verify_source_snapshot(
                source_root=temporary_root,
                candidate_sha=candidate_sha,
                expected_sha256=snapshot_sha256,
            )
            private_metadata = os.fstat(temporary_root_descriptor)
            private_identity = (
                private_metadata.st_dev,
                private_metadata.st_ino,
            )
            try:
                _rename_noreplace(
                    parent_descriptor,
                    temporary_root.name,
                    parent_descriptor,
                    snapshot_sha256,
                )
            except FileExistsError:
                _verify_named_snapshot(
                    parent_descriptor=parent_descriptor,
                    name=snapshot_sha256,
                    path=final_root,
                    candidate_sha=candidate_sha,
                    snapshot_sha256=snapshot_sha256,
                )
                return final_root, snapshot_sha256, False
            os.fsync(parent_descriptor)
            _verify_named_directory(
                parent_descriptor=parent_descriptor,
                name=snapshot_sha256,
                expected_identity=private_identity,
            )
            _verify_named_snapshot(
                parent_descriptor=parent_descriptor,
                name=snapshot_sha256,
                path=final_root,
                candidate_sha=candidate_sha,
                snapshot_sha256=snapshot_sha256,
            )
            return final_root, snapshot_sha256, True
        finally:
            try:
                if claim_descriptor is not None:
                    os.close(claim_descriptor)
            finally:
                if claim_identity is not None:
                    _unlink_owned_entry(
                        parent_descriptor=parent_descriptor,
                        name=claim_name,
                        expected_identity=claim_identity,
                    )
    finally:
        if temporary_root.exists():
            _remove_temporary_tree(temporary_root)
            os.fsync(parent_descriptor)
        os.close(temporary_root_descriptor)
        os.close(parent_descriptor)


def _publish_intent(*, artifact_root: Path, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str, snapshot_root: Path, snapshot_sha256: str, intent_payload: Mapping[str, object]) -> SubmissionArtifacts:
    payload = dict(intent_payload)
    payload["snapshot_path"] = str(snapshot_root)
    payload["snapshot_sha256"] = snapshot_sha256
    serialized = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    intent_sha256 = hashlib.sha256(serialized).hexdigest()
    parent = artifact_root / "submission-intents" / candidate_kind / candidate_sha
    parent_descriptor = _open_safe_directory(parent, create=True)
    root = artifact_root.resolve()
    resolved_parent = parent.resolve()
    if not resolved_parent.is_relative_to(root / "submission-intents"):
        os.close(parent_descriptor)
        raise ValueError("submission intent escaped artifact root")
    parent_metadata = os.fstat(parent_descriptor)
    submission_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
    intent_path = parent / f"{submission_id}.json"
    intent_name = intent_path.name
    temporary_name = f".{submission_id}.tmp"
    temporary_owned = False
    intent_owned = False
    try:
        intent_identity = _write_file_at(
            parent_descriptor=parent_descriptor,
            name=temporary_name,
            contents=serialized,
            mode=0o600,
        )
        temporary_owned = True
        temporary_descriptor = os.open(
            temporary_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            os.fchmod(temporary_descriptor, 0o444)
            os.fsync(temporary_descriptor)
            metadata = os.fstat(temporary_descriptor)
            if (metadata.st_dev, metadata.st_ino) != intent_identity:
                raise ValueError("submission intent temporary file changed")
        finally:
            os.close(temporary_descriptor)
        os.fsync(parent_descriptor)
        os.link(
            temporary_name,
            intent_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _OWNED_INTENTS[intent_path.absolute()] = _OwnedIntent(
            root,
            parent.absolute(),
            parent_metadata.st_dev,
            parent_metadata.st_ino,
            *intent_identity,
        )
        intent_owned = True
        _unlink_owned_entry(
            parent_descriptor=parent_descriptor,
            name=temporary_name,
            expected_identity=intent_identity,
        )
        temporary_owned = False
        load_submission_intent(intent_path, expected_sha256=intent_sha256)
    except BaseException:
        if intent_owned:
            remove_owned_intent(intent_path)
        raise
    finally:
        if temporary_owned:
            _unlink_owned_entry(
                parent_descriptor=parent_descriptor,
                name=temporary_name,
                expected_identity=intent_identity,
            )
        os.close(parent_descriptor)
    return SubmissionArtifacts(snapshot_root, snapshot_sha256, intent_path, intent_sha256)


def remove_owned_intent(path: Path) -> None:
    absolute_path = path.absolute()
    owned = _OWNED_INTENTS.get(absolute_path)
    if owned is None:
        raise ValueError("submission intent is not owned by this transaction")
    if absolute_path.parent != owned.parent or not owned.parent.is_relative_to(
        owned.artifact_root / "submission-intents"
    ):
        raise ValueError("submission intent escaped its owned parent")
    parent_descriptor = _open_safe_directory(owned.parent, create=False)
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if (parent_metadata.st_dev, parent_metadata.st_ino) != (
            owned.parent_device,
            owned.parent_inode,
        ):
            raise ValueError("submission intent escaped its owned parent")
        metadata = os.stat(
            absolute_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (owned.device, owned.inode)
        ):
            raise ValueError("submission intent is not the owned immutable file")
        os.unlink(absolute_path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        del _OWNED_INTENTS[absolute_path]
    finally:
        os.close(parent_descriptor)


def prepare_candidate_submission(*, archive_sources: tuple[ArchiveSource, ...], artifact_root: Path, mode: SubmissionMode, candidate_kind: Literal["mcore", "bridge"], candidate_sha: str, intent_payload: Mapping[str, object]) -> SubmissionTransaction:
    _validate_request(archive_sources=archive_sources, artifact_root=artifact_root, mode=mode, candidate_kind=candidate_kind, candidate_sha=candidate_sha)
    snapshot_root, snapshot_sha256, snapshot_created = _publish_snapshot(archive_sources=archive_sources, artifact_root=artifact_root, candidate_kind=candidate_kind, candidate_sha=candidate_sha)
    artifacts = _publish_intent(artifact_root=artifact_root, candidate_kind=candidate_kind, candidate_sha=candidate_sha, snapshot_root=snapshot_root, snapshot_sha256=snapshot_sha256, intent_payload=intent_payload)
    return SubmissionTransaction(artifacts, artifact_root, mode, snapshot_created)
