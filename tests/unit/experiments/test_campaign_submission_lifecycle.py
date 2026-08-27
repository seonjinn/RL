from __future__ import annotations

import builtins
import errno
import importlib.util
import os
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LIFECYCLE_PATH = (
    REPO_ROOT
    / "experiments"
    / "cuda_graph"
    / "nemotron_thd_te_graph_20260731"
    / "scripts"
    / "submission_lifecycle.py"
)


def load_lifecycle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("submission_lifecycle", LIFECYCLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def git_repository(path: Path, *, reserved_marker: str | None = None) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", path, "config", "user.email", "fixture@example.com"],
        check=True,
    )
    (path / "payload.py").write_text("VALUE = 1\n")
    (path / "alias.py").symlink_to("payload.py")
    if reserved_marker is not None:
        (path / reserved_marker).write_text("reserved\n")
    subprocess.run(["git", "-C", path, "add", "."], check=True)
    subprocess.run(["git", "-C", path, "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, commit


def git_repository_with_file(
    path: Path, *, relative_path: Path, contents: str
) -> tuple[Path, str]:
    repository, _ = git_repository(path)
    target = repository / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents)
    subprocess.run(["git", "-C", repository, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repository, "commit", "-q", "-m", "additional file"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def prepare_actual(
    repository: Path, commit: str, artifact_root: Path, *, module: ModuleType | None = None
):
    module = module or load_lifecycle()
    return module.prepare_candidate_submission(
        archive_sources=(module.ArchiveSource(repository, commit, Path(".")),),
        artifact_root=artifact_root,
        mode=module.SubmissionMode.ACTUAL,
        candidate_kind="mcore",
        candidate_sha=commit,
        intent_payload={
            "schema_version": 1,
            "candidate_kind": "mcore",
            "candidate_sha": commit,
        },
    )


def prepare_actual_in_subprocess(
    repository: Path, commit: str, artifact_root: Path
) -> None:
    script = """
import importlib.util
import sys
from pathlib import Path

lifecycle_path = Path(sys.argv[1])
repository = Path(sys.argv[2])
commit = sys.argv[3]
artifact_root = Path(sys.argv[4])
spec = importlib.util.spec_from_file_location(
    "submission_lifecycle_subprocess", lifecycle_path
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
transaction = module.prepare_candidate_submission(
    archive_sources=(
        module.ArchiveSource(repository, commit, Path(".")),
    ),
    artifact_root=artifact_root,
    mode=module.SubmissionMode.ACTUAL,
    candidate_kind="mcore",
    candidate_sha=commit,
    intent_payload={
        "schema_version": 1,
        "candidate_kind": "mcore",
        "candidate_sha": commit,
    },
)
transaction.close()
"""
    subprocess.run(
        (
            sys.executable,
            "-c",
            script,
            str(LIFECYCLE_PATH),
            str(repository),
            commit,
            str(artifact_root),
        ),
        check=True,
        capture_output=True,
        text=True,
    )


def tree_identity(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            stat.S_IFMT(path.lstat().st_mode),
            stat.S_IMODE(path.lstat().st_mode),
        )
        for path in sorted(root.rglob("*"))
    )


def make_snapshot_mutable(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def mutate_snapshot(root: Path, mutation: str) -> None:
    make_snapshot_mutable(root)
    if mutation == "writable":
        return
    if mutation == "digest":
        (root / ".snapshot-sha256").write_text("0" * 64 + "\n")
        (root / ".snapshot-sha256").chmod(0o444)
        return
    if mutation == "symlink":
        target = root.parent / "outside.py"
        target.write_text("outside\n")
        payload = root / "payload.py"
        payload.unlink()
        payload.symlink_to(target)
        return
    raise AssertionError(f"unknown mutation: {mutation}")


def assert_no_temporary_or_claim_residue(root: Path) -> None:
    assert not tuple(root.rglob("*.tmp"))
    assert not tuple(root.rglob("*.claim"))
    assert not tuple(root.rglob("*.quarantine"))
    for isolation in root.rglob("*.isolation"):
        assert isolation.is_dir()
        assert tuple(sorted(path.name for path in isolation.iterdir())) == (
            ".owner",
        )


def entries_with_identity(
    parent: Path, identity: tuple[int, int]
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in parent.iterdir()
        if (path.lstat().st_dev, path.lstat().st_ino) == identity
    )


def descendant_entries_with_identity(
    parent: Path, identity: tuple[int, int]
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in parent.rglob("*")
        if (path.lstat().st_dev, path.lstat().st_ino) == identity
    )


def test_identical_candidate_reuses_one_content_addressed_snapshot(
    tmp_path: Path,
) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    first = prepare_actual(repository, commit, tmp_path / "logs")
    before = tree_identity(first.artifacts.snapshot_root)
    second = prepare_actual(repository, commit, tmp_path / "logs")

    assert second.artifacts.snapshot_root == first.artifacts.snapshot_root
    assert second.artifacts.snapshot_sha256 == first.artifacts.snapshot_sha256
    assert tree_identity(second.artifacts.snapshot_root) == before
    assert second.artifacts.intent_path != first.artifacts.intent_path

    first.close()
    second.close()


@pytest.mark.parametrize("mutation", ("writable", "digest", "symlink"))
def test_existing_unsafe_snapshot_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    transaction = prepare_actual(repository, commit, tmp_path / "logs")
    mutate_snapshot(transaction.artifacts.snapshot_root, mutation)

    with pytest.raises(ValueError, match="unsafe"):
        prepare_actual(repository, commit, tmp_path / "logs")


def test_concurrent_identical_publishers_converge(tmp_path: Path) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    module = load_lifecycle()
    with ThreadPoolExecutor(max_workers=2) as executor:
        transactions = tuple(
            executor.map(
                lambda _: prepare_actual(
                    repository, commit, tmp_path / "logs", module=module
                ),
                range(2),
            )
        )

    assert len({item.artifacts.snapshot_root for item in transactions}) == 1
    assert sum(item.snapshot_created for item in transactions) <= 1
    assert_no_temporary_or_claim_residue(tmp_path / "logs")
    for transaction in transactions:
        transaction.close()


def test_submission_preparation_removes_temporary_state_after_archive_failure(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    with pytest.raises(RuntimeError, match="failed to archive"):
        module.prepare_candidate_submission(
            archive_sources=(
                module.ArchiveSource(tmp_path / "missing", "a" * 40, Path(".")),
            ),
            artifact_root=tmp_path / "logs",
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="mcore",
            candidate_sha="a" * 40,
            intent_payload={},
        )

    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_submission_preparation_leaves_no_intent_temporary_file_after_intent_failure(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    with pytest.raises(TypeError):
        module.prepare_candidate_submission(
            archive_sources=(module.ArchiveSource(repository, commit, Path(".")),),
            artifact_root=tmp_path / "logs",
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="mcore",
            candidate_sha=commit,
            intent_payload={"not_json": object()},
        )

    assert_no_temporary_or_claim_residue(tmp_path / "logs")
    assert not tuple((tmp_path / "logs" / "submission-intents").rglob("*.json"))


def test_intent_sync_failure_before_identity_capture_cleans_exact_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_fsync = module.os.fsync
    injected = False

    def fail_first_intent_temporary_sync(descriptor: int) -> None:
        nonlocal injected
        descriptor_path = module._descriptor_path(descriptor)
        if (
            not injected
            and "submission-intents" in descriptor_path.parts
            and descriptor_path.name.endswith(".tmp")
        ):
            injected = True
            raise OSError("intent temporary sync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_intent_temporary_sync)
    with pytest.raises(OSError, match="intent temporary sync"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert_no_temporary_or_claim_residue(artifact_root)
    assert not tuple(
        (artifact_root / "submission-intents").rglob("*.json")
    )


@pytest.mark.parametrize("entry_kind", ("claim", "intent"))
def test_owned_file_fstat_failure_cleans_created_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_fstat = module.os.fstat
    injected = False

    def fail_owned_file_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        descriptor_path = module._descriptor_path(descriptor)
        matches = (
            descriptor_path.name.endswith(".claim")
            if entry_kind == "claim"
            else (
                "submission-intents" in descriptor_path.parts
                and descriptor_path.name.endswith(".tmp")
            )
        )
        if not injected and matches:
            injected = True
            raise OSError(f"{entry_kind} identity capture failed")
        return original_fstat(descriptor)

    monkeypatch.setattr(module.os, "fstat", fail_owned_file_fstat)
    with pytest.raises(OSError, match="identity capture"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert_no_temporary_or_claim_residue(artifact_root)
    assert not tuple(
        (artifact_root / "submission-intents").rglob("*.json")
    )


def test_intent_parent_identity_failure_closes_parent_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_root = (
        Path("/dev/fd")
        if Path("/dev/fd").is_dir()
        else Path("/proc/self/fd")
    )
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    intent_parent = (
        artifact_root / "submission-intents" / "mcore" / commit
    )
    original_fstat = module.os.fstat
    injected = False

    def fail_intent_parent_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and intent_parent.is_dir()
            and module._descriptor_path(descriptor) == intent_parent
        ):
            injected = True
            raise OSError("intent parent identity failed")
        return original_fstat(descriptor)

    before = len(tuple(descriptor_root.iterdir()))
    monkeypatch.setattr(module.os, "fstat", fail_intent_parent_fstat)
    with pytest.raises(OSError, match="intent parent identity"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert len(tuple(descriptor_root.iterdir())) == before
    assert_no_temporary_or_claim_residue(artifact_root)


def test_submission_preparation_rejects_escaping_archive_destination(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    with pytest.raises(ValueError, match="destination"):
        module.prepare_candidate_submission(
            archive_sources=(module.ArchiveSource(repository, commit, Path("..")),),
            artifact_root=tmp_path / "logs",
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="mcore",
            candidate_sha=commit,
            intent_payload={},
        )

    assert_no_temporary_or_claim_residue(tmp_path / "logs")


@pytest.mark.parametrize("marker", (".candidate-sha", ".snapshot-sha256"))
def test_submission_preparation_rejects_reserved_snapshot_markers(
    tmp_path: Path, marker: str
) -> None:
    repository, commit = git_repository(tmp_path / "candidate", reserved_marker=marker)

    with pytest.raises(ValueError, match="reserved"):
        prepare_actual(repository, commit, tmp_path / "logs")

    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_unsafe_snapshot_with_unsupported_file_type_fails_closed(tmp_path: Path) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    transaction = prepare_actual(repository, commit, tmp_path / "logs")
    snapshot = transaction.artifacts.snapshot_root
    snapshot.chmod(snapshot.stat().st_mode | stat.S_IWUSR)
    fifo = snapshot / "unexpected.fifo"
    os.mkfifo(fifo)
    fifo.chmod(0o444)
    snapshot.chmod(snapshot.stat().st_mode & ~stat.S_IWUSR)

    with pytest.raises(ValueError, match="unsupported file type"):
        prepare_actual(repository, commit, tmp_path / "logs")


def test_submission_preparation_close_removes_only_its_owned_intent(
    tmp_path: Path,
) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    transaction = prepare_actual(repository, commit, tmp_path / "logs")
    intent_path = transaction.artifacts.intent_path

    transaction.close()

    assert not intent_path.exists()
    assert transaction.artifacts.snapshot_root.is_dir()


def test_submission_preparation_rejects_replaced_owned_intent(tmp_path: Path) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    transaction = prepare_actual(repository, commit, tmp_path / "logs")
    intent_path = transaction.artifacts.intent_path
    intent_path.chmod(0o644)
    intent_path.unlink()
    intent_path.write_text("{}\n")
    intent_path.chmod(0o444)

    with pytest.raises(ValueError, match="owned"):
        transaction.close()

    assert intent_path.exists()


def test_submission_preparation_rejects_cross_archive_symlink_escape(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    outside = tmp_path / "outside"
    outside.mkdir()
    first, first_commit = git_repository(tmp_path / "first")
    (first / "integration").symlink_to(outside, target_is_directory=True)
    subprocess.run(["git", "-C", first, "add", "integration"], check=True)
    subprocess.run(
        ["git", "-C", first, "commit", "-q", "-m", "symlink destination"],
        check=True,
    )
    first_commit = subprocess.run(
        ["git", "-C", first, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    second, second_commit = git_repository_with_file(
        tmp_path / "second", relative_path=Path("payload.txt"), contents="second\n"
    )

    with pytest.raises(ValueError, match="symlink"):
        module.prepare_candidate_submission(
            archive_sources=(
                module.ArchiveSource(first, first_commit, Path(".")),
                module.ArchiveSource(second, second_commit, Path("integration")),
            ),
            artifact_root=tmp_path / "logs",
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="bridge",
            candidate_sha=first_commit,
            intent_payload={},
        )

    assert not tuple(outside.iterdir())
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


@pytest.mark.parametrize("namespace", ("source-snapshots", "submission-intents"))
def test_submission_preparation_rejects_symlinked_durable_parent(
    tmp_path: Path, namespace: str
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    outside = tmp_path / "outside"
    outside.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / namespace).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.prepare_candidate_submission(
            archive_sources=(module.ArchiveSource(repository, commit, Path(".")),),
            artifact_root=logs,
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="mcore",
            candidate_sha=commit,
            intent_payload={},
        )

    assert not tuple(outside.iterdir())
    for path in (logs, *logs.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_submission_preparation_fails_closed_when_final_snapshot_races_to_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    outside = tmp_path / "outside"
    outside.mkdir()
    final_parent = tmp_path / "logs" / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace

    def install_symlink_winner(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name.endswith(".tmp") and len(destination_name) == 64:
            assert source_parent_descriptor == destination_parent_descriptor
            (final_parent / destination_name).symlink_to(
                outside, target_is_directory=True
            )
            raise FileExistsError(
                errno.EEXIST, "symlink winner", destination_name
            )
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module, "_rename_noreplace", install_symlink_winner, raising=False
    )
    with pytest.raises(ValueError, match="unsafe"):
        prepare_actual(repository, commit, tmp_path / "logs", module=module)

    assert not tuple(outside.iterdir())
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_submission_preparation_removes_claim_when_claim_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")

    def fail_claim_sync(descriptor: int) -> None:
        raise OSError("claim sync failed")

    monkeypatch.setattr(module, "_fsync_claim", fail_claim_sync)
    with pytest.raises(OSError, match="claim sync"):
        prepare_actual(repository, commit, tmp_path / "logs", module=module)

    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_submission_preparation_removes_owned_intent_when_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")

    def fail_verification(path: Path, *, expected_sha256: str) -> dict[str, object]:
        raise ValueError("intent verification failed")

    monkeypatch.setattr(module, "load_submission_intent", fail_verification)
    with pytest.raises(ValueError, match="intent verification"):
        prepare_actual(repository, commit, tmp_path / "logs", module=module)

    assert not tuple((tmp_path / "logs" / "submission-intents").rglob("*.json"))
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_submission_preparation_removes_claim_when_claim_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    claim_descriptor: list[int] = []
    close_faults = 0
    original_fsync_claim = module._fsync_claim
    original_close = module.os.close

    def record_claim_descriptor(descriptor: int) -> None:
        claim_descriptor.append(descriptor)
        original_fsync_claim(descriptor)

    def fail_claim_close(descriptor: int) -> None:
        nonlocal close_faults
        if claim_descriptor == [descriptor]:
            close_faults += 1
            raise OSError("claim close failed")
        original_close(descriptor)

    monkeypatch.setattr(module, "_fsync_claim", record_claim_descriptor)
    monkeypatch.setattr(module.os, "close", fail_claim_close)
    with pytest.raises(OSError, match="claim close"):
        prepare_actual(repository, commit, tmp_path / "logs", module=module)

    assert close_faults == 1
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_claim_cleanup_pre_unlink_swap_preserves_unknown_and_removes_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_fsync_claim = module._fsync_claim
    original_unlink = module.os.unlink
    claim_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    shared_unlink_attempted = False

    def record_claim_identity(descriptor: int) -> None:
        nonlocal claim_identity
        metadata = module.os.fstat(descriptor)
        claim_identity = (metadata.st_dev, metadata.st_ino)
        original_fsync_claim(descriptor)

    def swap_claim_before_shared_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal replacement_identity, shared_unlink_attempted
        name = Path(path).name
        if (
            not shared_unlink_attempted
            and dir_fd is not None
            and name.endswith(".claim")
            and module._descriptor_path(dir_fd) == final_parent
        ):
            displaced_name = ".owned-claim-displaced"
            module.os.rename(
                name,
                displaced_name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            descriptor = module.os.open(
                name,
                module.os.O_WRONLY
                | module.os.O_CREAT
                | module.os.O_EXCL
                | module.os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                module.os.write(descriptor, b"unrelated claim\n")
                metadata = module.os.fstat(descriptor)
                replacement_identity = (metadata.st_dev, metadata.st_ino)
            finally:
                module.os.close(descriptor)
            shared_unlink_attempted = True
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(module, "_fsync_claim", record_claim_identity)
    monkeypatch.setattr(module.os, "unlink", swap_claim_before_shared_unlink)
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    transaction.close()

    assert claim_identity is not None
    if shared_unlink_attempted:
        assert replacement_identity is not None
        assert entries_with_identity(final_parent, replacement_identity)
    assert not entries_with_identity(final_parent, claim_identity)


def test_claim_identity_remains_pinned_until_isolated_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_fsync_claim = module._fsync_claim
    original_move = module._move_shared_entry_into_isolation
    original_close = module.os.close
    claim_descriptor: int | None = None
    claim_identity: tuple[int, int] | None = None
    claim_closed = False
    cleanup_observed = False

    def record_claim_descriptor(descriptor: int) -> None:
        nonlocal claim_descriptor, claim_identity
        claim_descriptor = descriptor
        metadata = module.os.fstat(descriptor)
        claim_identity = (metadata.st_dev, metadata.st_ino)
        original_fsync_claim(descriptor)

    def record_claim_close(descriptor: int) -> None:
        nonlocal claim_closed
        if descriptor == claim_descriptor:
            claim_closed = True
        original_close(descriptor)

    def require_pinned_claim(**kwargs: object) -> bool:
        nonlocal cleanup_observed
        source_name = kwargs["source_name"]
        if isinstance(source_name, str) and source_name.endswith(".claim"):
            assert claim_descriptor is not None
            assert claim_identity is not None
            assert not claim_closed
            metadata = module.os.fstat(claim_descriptor)
            assert (metadata.st_dev, metadata.st_ino) == claim_identity
            cleanup_observed = True
        return original_move(**kwargs)

    monkeypatch.setattr(module, "_fsync_claim", record_claim_descriptor)
    monkeypatch.setattr(module.os, "close", record_claim_close)
    monkeypatch.setattr(
        module, "_move_shared_entry_into_isolation", require_pinned_claim
    )
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )

    assert cleanup_observed
    transaction.close()


def test_submission_preparation_removes_owned_intent_when_post_link_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    sync_fault_raised = False
    original_fsync = module.os.fsync

    def fail_first_post_link_sync(descriptor: int) -> None:
        nonlocal sync_fault_raised
        if (
            stat.S_ISDIR(module.os.fstat(descriptor).st_mode)
            and any(name.endswith(".json") for name in module.os.listdir(descriptor))
            and not sync_fault_raised
        ):
            sync_fault_raised = True
            raise OSError("post-link sync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_post_link_sync)
    with pytest.raises(OSError, match="post-link sync"):
        prepare_actual(repository, commit, tmp_path / "logs", module=module)

    assert sync_fault_raised
    assert not tuple((tmp_path / "logs" / "submission-intents").rglob("*.json"))
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


def test_intent_cleanup_pre_unlink_swap_preserves_unknown_and_removes_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    intent_path = transaction.artifacts.intent_path
    intent_parent = intent_path.parent
    intent_metadata = intent_path.lstat()
    intent_identity = (intent_metadata.st_dev, intent_metadata.st_ino)
    original_unlink = module.os.unlink
    replacement_identity: tuple[int, int] | None = None
    shared_unlink_attempted = False

    def swap_intent_before_shared_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal replacement_identity, shared_unlink_attempted
        name = Path(path).name
        if (
            not shared_unlink_attempted
            and dir_fd is not None
            and name == intent_path.name
            and module._descriptor_path(dir_fd) == intent_parent
        ):
            displaced_name = ".owned-intent-displaced"
            module.os.rename(
                name,
                displaced_name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            descriptor = module.os.open(
                name,
                module.os.O_WRONLY
                | module.os.O_CREAT
                | module.os.O_EXCL
                | module.os.O_NOFOLLOW,
                0o444,
                dir_fd=dir_fd,
            )
            try:
                module.os.write(descriptor, b"unrelated intent\n")
                metadata = module.os.fstat(descriptor)
                replacement_identity = (metadata.st_dev, metadata.st_ino)
            finally:
                module.os.close(descriptor)
            shared_unlink_attempted = True
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "unlink", swap_intent_before_shared_unlink)
    transaction.close()

    if shared_unlink_attempted:
        assert replacement_identity is not None
        assert entries_with_identity(intent_parent, replacement_identity)
    assert not entries_with_identity(intent_parent, intent_identity)


def test_owned_intent_identity_is_pinned_until_transaction_close(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    transaction = prepare_actual(
        repository, commit, tmp_path / "logs", module=module
    )
    owned = module._OWNED_INTENTS[
        transaction.artifacts.intent_path.absolute()
    ]

    metadata = module.os.fstat(owned.descriptor)
    assert (metadata.st_dev, metadata.st_ino) == (owned.device, owned.inode)

    transaction.close()
    with pytest.raises(OSError) as error:
        module.os.fstat(owned.descriptor)
    assert error.value.errno == errno.EBADF


def test_submission_preparation_component_swap_cannot_write_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository_with_file(
        tmp_path / "candidate",
        relative_path=Path("attack.txt"),
        contents="must stay inside\n",
    )
    artifact_root = tmp_path / "logs"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = builtins.open
    component_swapped = False

    def swap_component_before_source_read(file: object, *args: object, **kwargs: object):
        nonlocal component_swapped
        try:
            source_path = Path(file)  # type: ignore[arg-type]
        except TypeError:
            source_path = Path()
        if (
            not component_swapped
            and source_path.name == "attack.txt"
            and any(part.startswith(".archive.") for part in source_path.parts)
        ):
            private_roots = tuple(
                (artifact_root / "source-snapshots" / "mcore" / commit).glob(
                    f".{commit}.*.tmp"
                )
            )
            assert len(private_roots) == 1
            component = private_roots[0] / "integration"
            displaced = component.with_name("integration.displaced")
            component.rename(displaced)
            component.symlink_to(outside, target_is_directory=True)
            component_swapped = True
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", swap_component_before_source_read)
    try:
        transaction = module.prepare_candidate_submission(
            archive_sources=(
                module.ArchiveSource(
                    repository, commit, Path("integration")
                ),
            ),
            artifact_root=artifact_root,
            mode=module.SubmissionMode.ACTUAL,
            candidate_kind="mcore",
            candidate_sha=commit,
            intent_payload={},
        )
    except (OSError, ValueError):
        pass
    else:
        transaction.close()

    assert component_swapped
    assert not tuple(outside.iterdir())
    assert_no_temporary_or_claim_residue(artifact_root)


def test_private_directory_parent_swap_cannot_create_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    durable_parent = artifact_root / "source-snapshots" / "mcore" / commit
    displaced_parent = durable_parent.with_name(f"{commit}.displaced")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = module.os.mkdir
    parent_swapped = False
    outside_directory_created = False

    def swap_parent_before_private_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal parent_swapped, outside_directory_created
        name = Path(path).name
        if (
            not parent_swapped
            and name.startswith(f".{commit}.")
            and name.endswith(".tmp")
        ):
            durable_parent.rename(displaced_parent)
            durable_parent.symlink_to(outside, target_is_directory=True)
            parent_swapped = True
        if dir_fd is None:
            original_mkdir(path, mode)
            candidate = Path(path)
            if parent_swapped and candidate.parent.resolve() == outside:
                outside_directory_created = True
        else:
            original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", swap_parent_before_private_mkdir)
    try:
        transaction = prepare_actual(
            repository, commit, artifact_root, module=module
        )
    except (OSError, ValueError):
        pass
    else:
        transaction.close()

    assert parent_swapped
    assert not outside_directory_created
    assert not tuple(outside.iterdir())


def test_submission_preparation_does_not_populate_post_mkdir_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_lstat = module.Path.lstat
    replacement: Path | None = None

    def replace_created_final_root(path: Path):
        nonlocal replacement
        metadata = original_lstat(path)
        if (
            replacement is None
            and path.parent.name == commit
            and len(path.name) == 64
        ):
            displaced = path.with_name(f".{path.name}.displaced")
            path.rename(displaced)
            path.mkdir(mode=0o700)
            replacement = path
        return metadata

    monkeypatch.setattr(module.Path, "lstat", replace_created_final_root)
    try:
        transaction = prepare_actual(
            repository, commit, artifact_root, module=module
        )
    except (OSError, ValueError):
        pass
    else:
        transaction.close()

    replacement_was_populated = (
        replacement is not None and (replacement / "payload.py").exists()
    )
    for path in (artifact_root, *artifact_root.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    assert replacement is not None
    assert not replacement_was_populated


def test_final_mkdir_return_replacement_is_not_accepted_or_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_mkdir = module.os.mkdir
    swapped = False
    accepted = False
    replacement: Path | None = None

    def replace_final_before_mkdir_returns(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal replacement, swapped
        name = Path(path).name
        is_final = len(name) == 64 and (
            dir_fd is not None or Path(path).parent == final_parent
        )
        if not swapped and is_final:
            if dir_fd is None:
                original_mkdir(path, mode)
            else:
                original_mkdir(path, mode, dir_fd=dir_fd)
            created = final_parent / name
            created.rename(created.with_name(f".{name}.displaced"))
            original_mkdir(created, mode)
            replacement = created
            swapped = True
            return
        if dir_fd is None:
            original_mkdir(path, mode)
        else:
            original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", replace_final_before_mkdir_returns)
    try:
        transaction = prepare_actual(
            repository, commit, artifact_root, module=module
        )
    except (OSError, ValueError):
        pass
    else:
        accepted = True
        transaction.close()

    replacement_populated = (
        replacement is not None and (replacement / "payload.py").exists()
    )
    for path in (artifact_root, *artifact_root.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    assert accepted
    assert not swapped
    assert replacement is None
    assert not replacement_populated


def test_source_name_swap_cannot_poison_final_or_leave_verified_private_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    displaced: Path | None = None
    final_root: Path | None = None

    def swap_source_name_then_publish_replacement(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal displaced, final_root
        if (
            displaced is None
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            assert source_parent_descriptor == destination_parent_descriptor
            source_root = final_parent / source_name
            displaced = source_root.with_name(f"{source_name}.displaced")
            source_root.rename(displaced)
            source_root.mkdir()
            final_root = final_parent / destination_name
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        swap_source_name_then_publish_replacement,
    )
    with pytest.raises(ValueError, match="changed"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert displaced is not None
    assert final_root is not None
    assert (final_root.exists(), displaced.exists()) == (False, False)
    recovered = prepare_actual(repository, commit, artifact_root, module=module)
    assert recovered.snapshot_created is True
    assert recovered.artifacts.snapshot_root == final_root
    assert_no_temporary_or_claim_residue(artifact_root)
    recovered.close()


def test_source_moved_outside_private_parent_preserves_external_data_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    outside = tmp_path / "outside"
    outside.mkdir()
    unrelated = outside / "unrelated.txt"
    unrelated.write_text("do not delete\n")
    external_private = outside / "retained-private"
    original_rename_noreplace = module._rename_noreplace
    final_root: Path | None = None
    swapped = False

    def move_source_outside_then_publish_replacement(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal final_root, swapped
        if (
            not swapped
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            assert source_parent_descriptor == destination_parent_descriptor
            source_root = final_parent / source_name
            source_root.chmod(
                stat.S_IMODE(source_root.stat().st_mode) | stat.S_IWUSR
            )
            source_root.rename(external_private)
            source_root.mkdir()
            final_root = final_parent / destination_name
            swapped = True
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        move_source_outside_then_publish_replacement,
    )
    with pytest.raises(ValueError, match="changed"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert swapped
    assert final_root is not None
    assert not final_root.exists()
    assert (external_private / "payload.py").read_text() == "VALUE = 1\n"
    assert unrelated.read_text() == "do not delete\n"
    assert_no_temporary_or_claim_residue(artifact_root)

    recovered = prepare_actual(repository, commit, artifact_root, module=module)
    assert recovered.snapshot_created is True
    assert recovered.artifacts.snapshot_root == final_root
    assert external_private.is_dir()
    assert unrelated.read_text() == "do not delete\n"
    assert_no_temporary_or_claim_residue(artifact_root)
    recovered.close()


def test_source_displaced_before_atomic_publication_cleans_exact_private_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    private_identity: tuple[int, int] | None = None
    displaced: Path | None = None
    injected = False

    def displace_source_then_raise_enoent(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal displaced, injected, private_identity
        if not injected and source_name.endswith(".tmp") and len(destination_name) == 64:
            source_root = final_parent / source_name
            metadata = source_root.lstat()
            private_identity = (metadata.st_dev, metadata.st_ino)
            displaced = source_root.with_name(".verified-private-displaced")
            source_root.rename(displaced)
            injected = True
            raise FileNotFoundError(errno.ENOENT, "source displaced", source_name)
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        displace_source_then_raise_enoent,
    )
    with pytest.raises(FileNotFoundError, match="source displaced"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert displaced is not None
    assert private_identity is not None
    assert not displaced.exists()
    assert not entries_with_identity(final_parent, private_identity)
    assert_no_temporary_or_claim_residue(artifact_root)

    recovered = prepare_actual(repository, commit, artifact_root, module=module)
    assert recovered.snapshot_created is True
    assert_no_temporary_or_claim_residue(artifact_root)
    recovered.close()


def test_successful_publication_preserves_recreated_unknown_source_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    replacement_identity: tuple[int, int] | None = None
    injected = False

    def publish_then_recreate_source_name(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal injected, replacement_identity
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )
        if not injected and source_name.endswith(".tmp") and len(destination_name) == 64:
            source_root = final_parent / source_name
            source_root.mkdir()
            (source_root / "unrelated.txt").write_text("preserve me\n")
            metadata = source_root.lstat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            injected = True

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        publish_then_recreate_source_name,
    )
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )

    assert injected
    assert transaction.snapshot_created is True
    assert transaction.artifacts.snapshot_root.is_dir()
    assert replacement_identity is not None
    preserved = entries_with_identity(final_parent, replacement_identity)
    assert len(preserved) == 1
    assert (preserved[0] / "unrelated.txt").read_text() == "preserve me\n"
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()

    reused = prepare_actual(repository, commit, artifact_root, module=module)
    assert reused.snapshot_created is False
    assert entries_with_identity(final_parent, replacement_identity) == preserved
    assert_no_temporary_or_claim_residue(artifact_root)
    reused.close()


def test_replacement_source_with_valid_eexist_winner_preserves_unknown_and_cleans_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    private_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    final_root: Path | None = None
    injected = False

    def install_replacement_source_and_valid_winner(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal final_root, injected, private_identity, replacement_identity
        if not injected and source_name.endswith(".tmp") and len(destination_name) == 64:
            source_root = final_parent / source_name
            metadata = source_root.lstat()
            private_identity = (metadata.st_dev, metadata.st_ino)
            displaced = source_root.with_name(".verified-private-displaced")
            source_root.rename(displaced)
            source_root.mkdir()
            (source_root / "unrelated.txt").write_text("preserve me\n")
            metadata = source_root.lstat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            final_root = final_parent / destination_name
            module.shutil.copytree(displaced, final_root, symlinks=True)
            module._make_tree_read_only(final_root)
            injected = True
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        install_replacement_source_and_valid_winner,
    )
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )

    assert injected
    assert transaction.snapshot_created is False
    assert final_root == transaction.artifacts.snapshot_root
    assert private_identity is not None
    assert replacement_identity is not None
    assert not entries_with_identity(final_parent, private_identity)
    preserved = entries_with_identity(final_parent, replacement_identity)
    assert len(preserved) == 1
    assert (preserved[0] / "unrelated.txt").read_text() == "preserve me\n"
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()

    reused = prepare_actual(repository, commit, artifact_root, module=module)
    assert reused.snapshot_created is False
    assert reused.artifacts.snapshot_root == final_root
    assert entries_with_identity(final_parent, replacement_identity) == preserved
    assert_no_temporary_or_claim_residue(artifact_root)
    reused.close()


def test_recovery_rename_swap_restores_valid_unknown_without_quarantine_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    original_rename = module.os.rename
    displaced_private: Path | None = None
    final_root: Path | None = None
    preserved_source: Path | None = None
    source_replacement_identity: tuple[int, int] | None = None
    winner_identity: tuple[int, int] | None = None
    publication_raced = False
    recovery_raced = False

    def inject_recovery_source_swap(
        source_parent_descriptor: int,
        source_name: str,
    ) -> None:
        nonlocal preserved_source, recovery_raced, winner_identity
        if (
            not publication_raced
            or recovery_raced
            or final_root is None
            or source_name != final_root.name
            or module._descriptor_path(source_parent_descriptor) != final_parent
        ):
            return
        assert displaced_private is not None
        preserved_source = final_parent / ".unrelated-source-preserved"
        original_rename(
            final_root.name,
            preserved_source.name,
            src_dir_fd=source_parent_descriptor,
            dst_dir_fd=source_parent_descriptor,
        )
        module.shutil.copytree(displaced_private, final_root, symlinks=True)
        module._make_tree_read_only(final_root)
        metadata = final_root.lstat()
        winner_identity = (metadata.st_dev, metadata.st_ino)
        recovery_raced = True

    def swap_publication_and_recovery_sources(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal displaced_private, final_root, preserved_source
        nonlocal publication_raced, recovery_raced, source_replacement_identity
        nonlocal winner_identity
        if (
            not publication_raced
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            source_root = final_parent / source_name
            displaced_private = source_root.with_name(".verified-private-displaced")
            source_root.rename(displaced_private)
            source_root.mkdir()
            (source_root / "unrelated.txt").write_text("source replacement\n")
            metadata = source_root.lstat()
            source_replacement_identity = (metadata.st_dev, metadata.st_ino)
            final_root = final_parent / destination_name
            publication_raced = True
            original_rename_noreplace(
                source_parent_descriptor,
                source_name,
                destination_parent_descriptor,
                destination_name,
            )
            return
        inject_recovery_source_swap(source_parent_descriptor, source_name)
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    def swap_before_isolation_rename(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if src_dir_fd is not None:
            inject_recovery_source_swap(src_dir_fd, Path(source).name)
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        swap_publication_and_recovery_sources,
    )
    monkeypatch.setattr(module.os, "rename", swap_before_isolation_rename)
    with pytest.raises(ValueError, match="changed"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert publication_raced
    assert recovery_raced
    assert displaced_private is not None
    assert not displaced_private.exists()
    assert preserved_source is not None
    assert source_replacement_identity is not None
    preserved_sources = entries_with_identity(
        final_parent, source_replacement_identity
    )
    assert preserved_sources == (preserved_source,)
    assert (
        preserved_sources[0] / "unrelated.txt"
    ).read_text() == "source replacement\n"
    assert not tuple(final_parent.glob(".unowned-*.preserved"))
    assert winner_identity is not None
    assert final_root is not None
    assert entries_with_identity(final_parent, winner_identity) == (final_root,)
    assert_no_temporary_or_claim_residue(artifact_root)

    reused = prepare_actual(repository, commit, artifact_root, module=module)
    assert reused.snapshot_created is False
    assert reused.artifacts.snapshot_root == final_root
    assert_no_temporary_or_claim_residue(artifact_root)
    reused.close()


def test_private_tree_cleanup_pre_rmdir_swap_never_deletes_shared_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    original_rmdir = module.os.rmdir
    displaced_name = ".verified-private-displaced"
    private_identity: tuple[int, int] | None = None
    publication_raced = False
    shared_rmdir_attempted = False

    def publish_replacement_source(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal private_identity, publication_raced
        if (
            not publication_raced
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            source_root = final_parent / source_name
            metadata = source_root.lstat()
            private_identity = (metadata.st_dev, metadata.st_ino)
            source_root.rename(final_parent / displaced_name)
            source_root.mkdir()
            publication_raced = True
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    def swap_tree_before_shared_rmdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal shared_rmdir_attempted
        name = Path(path).name
        if (
            not shared_rmdir_attempted
            and dir_fd is not None
            and name == displaced_name
            and module._descriptor_path(dir_fd) == final_parent
        ):
            module.os.rename(
                name,
                ".owned-after-rmdir-swap",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            module.os.mkdir(name, 0o700, dir_fd=dir_fd)
            shared_rmdir_attempted = True
        original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(module, "_rename_noreplace", publish_replacement_source)
    monkeypatch.setattr(module.os, "rmdir", swap_tree_before_shared_rmdir)
    with pytest.raises(ValueError, match="changed"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert publication_raced
    assert private_identity is not None
    assert not shared_rmdir_attempted
    assert not entries_with_identity(final_parent, private_identity)

    recovered = prepare_actual(repository, commit, artifact_root, module=module)
    assert recovered.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    recovered.close()


def test_cleanup_never_rmdirs_isolation_by_shared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rmdir = module.os.rmdir

    def reject_shared_isolation_rmdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        name = Path(path).name
        if (
            dir_fd is not None
            and name.endswith(".isolation")
            and module._descriptor_path(dir_fd) == final_parent
        ):
            raise AssertionError(
                "isolation teardown is not bound to its retained inode"
            )
        original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "rmdir", reject_shared_isolation_rmdir)
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )

    assert transaction.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()
    assert_no_temporary_or_claim_residue(artifact_root)


def test_isolation_bootstrap_swap_fails_closed_without_deleting_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    owned_descriptor = module.os.open(
        "owned",
        module.os.O_WRONLY
        | module.os.O_CREAT
        | module.os.O_EXCL
        | module.os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    owned_metadata = module.os.fstat(owned_descriptor)
    owned_identity = (owned_metadata.st_dev, owned_metadata.st_ino)
    original_open_named_directory = module._open_named_directory
    unknown_identity: tuple[int, int] | None = None
    swapped = False

    def swap_created_isolation_before_open(
        directory_descriptor: int, name: str
    ) -> int:
        nonlocal swapped, unknown_identity
        if not swapped and name.endswith(".isolation"):
            module.os.rename(
                name,
                ".created-isolation-displaced",
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            module.os.mkdir(name, 0o700, dir_fd=directory_descriptor)
            replacement_descriptor = original_open_named_directory(
                directory_descriptor, name
            )
            try:
                unknown_descriptor = module.os.open(
                    "entry",
                    module.os.O_WRONLY
                    | module.os.O_CREAT
                    | module.os.O_EXCL
                    | module.os.O_NOFOLLOW,
                    0o600,
                    dir_fd=replacement_descriptor,
                )
                try:
                    module.os.write(unknown_descriptor, b"unknown\n")
                    metadata = module.os.fstat(unknown_descriptor)
                    unknown_identity = (metadata.st_dev, metadata.st_ino)
                finally:
                    module.os.close(unknown_descriptor)
            finally:
                module.os.close(replacement_descriptor)
            swapped = True
        return original_open_named_directory(directory_descriptor, name)

    monkeypatch.setattr(
        module, "_open_named_directory", swap_created_isolation_before_open
    )
    try:
        with pytest.raises(ValueError, match="cleanup isolation"):
            module._unlink_owned_entry(
                parent_descriptor=parent_descriptor,
                name="owned",
                expected_identity=owned_identity,
                identity_descriptor=owned_descriptor,
            )
    finally:
        module.os.close(owned_descriptor)
        module.os.close(parent_descriptor)

    assert swapped
    assert unknown_identity is not None
    assert descendant_entries_with_identity(parent, unknown_identity)
    assert descendant_entries_with_identity(parent, owned_identity)


def test_isolation_marker_initialization_failure_rolls_back_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    original_write_file_at = module._write_file_at
    injected = False

    def fail_first_marker_write(**kwargs: object) -> tuple[int, int]:
        nonlocal injected
        if not injected and kwargs["name"] == module.ISOLATION_MARKER_NAME:
            injected = True
            raise OSError("marker initialization failed")
        return original_write_file_at(**kwargs)

    monkeypatch.setattr(module, "_write_file_at", fail_first_marker_write)
    try:
        with module._locked_parent_namespace(parent_descriptor):
            with pytest.raises(OSError, match="marker initialization"):
                module._create_isolation_directory(parent_descriptor)
        assert not (parent / module.ISOLATION_DIRECTORY_NAME).exists()

        with module._locked_parent_namespace(parent_descriptor):
            _, isolation_descriptor = module._create_isolation_directory(
                parent_descriptor
            )
        module.os.close(isolation_descriptor)
    finally:
        module.os.close(parent_descriptor)

    assert injected
    assert tuple(
        path.name
        for path in (parent / module.ISOLATION_DIRECTORY_NAME).iterdir()
    ) == (module.ISOLATION_MARKER_NAME,)


@pytest.mark.parametrize("failure_stage", ("stat", "open"))
def test_isolation_bootstrap_observation_failure_rolls_back_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    original_stat = module.os.stat
    original_open_named_directory = module._open_named_directory
    injected = False

    def fail_first_isolation_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and path == module.ISOLATION_DIRECTORY_NAME
            and dir_fd == parent_descriptor
            and not follow_symlinks
        ):
            injected = True
            raise OSError("isolation bootstrap observation failed")
        return original_stat(
            path, dir_fd=dir_fd, follow_symlinks=follow_symlinks
        )

    def fail_first_isolation_open(descriptor: int, name: str) -> int:
        nonlocal injected
        if not injected and name == module.ISOLATION_DIRECTORY_NAME:
            injected = True
            raise OSError("isolation bootstrap observation failed")
        return original_open_named_directory(descriptor, name)

    if failure_stage == "stat":
        monkeypatch.setattr(module.os, "stat", fail_first_isolation_stat)
    else:
        monkeypatch.setattr(
            module, "_open_named_directory", fail_first_isolation_open
        )
    try:
        with module._locked_parent_namespace(parent_descriptor):
            with pytest.raises(OSError, match="bootstrap observation"):
                module._create_isolation_directory(parent_descriptor)
        assert not (parent / module.ISOLATION_DIRECTORY_NAME).exists()

        with module._locked_parent_namespace(parent_descriptor):
            _, isolation_descriptor = module._create_isolation_directory(
                parent_descriptor
            )
        module.os.close(isolation_descriptor)
    finally:
        module.os.close(parent_descriptor)

    assert injected


def test_persistent_isolation_sync_failure_still_rolls_back_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    isolation = parent / module.ISOLATION_DIRECTORY_NAME
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    original_fsync = module.os.fsync

    def fail_isolation_sync(descriptor: int) -> None:
        if isolation.exists() and module._descriptor_path(descriptor) == isolation:
            raise OSError("persistent isolation sync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_isolation_sync)
    try:
        with module._locked_parent_namespace(parent_descriptor):
            with pytest.raises(OSError, match="persistent isolation sync"):
                module._create_isolation_directory(parent_descriptor)
        assert not isolation.exists()

        monkeypatch.setattr(module.os, "fsync", original_fsync)
        with module._locked_parent_namespace(parent_descriptor):
            _, isolation_descriptor = module._create_isolation_directory(
                parent_descriptor
            )
        module.os.close(isolation_descriptor)
    finally:
        module.os.close(parent_descriptor)


def test_private_directory_initial_stat_failure_rolls_back_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_stat = module.os.stat
    injected = False

    def fail_first_private_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and dir_fd is not None
            and final_parent.is_dir()
            and module._descriptor_path(dir_fd) == final_parent
            and Path(path).name.endswith(".tmp")
            and not follow_symlinks
        ):
            injected = True
            raise OSError("private directory stat failed")
        return original_stat(
            path, dir_fd=dir_fd, follow_symlinks=follow_symlinks
        )

    monkeypatch.setattr(module.os, "stat", fail_first_private_stat)
    with pytest.raises(OSError, match="private directory stat"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert_no_temporary_or_claim_residue(artifact_root)
    recovered = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    recovered.close()


def test_private_directory_creator_returns_captured_identity(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    descriptor: int | None = None
    try:
        path, descriptor, identity = module._create_locked_private_directory(
            parent_descriptor=parent_descriptor,
            prefix=".private.",
            suffix=".tmp",
        )
        metadata = module.os.fstat(descriptor)
        assert (metadata.st_dev, metadata.st_ino) == identity
        module._remove_owned_entry(
            parent_descriptor=parent_descriptor,
            preferred_name=path.name,
            expected_identity=identity,
            identity_descriptor=descriptor,
            relocate_preferred_replacement=True,
        )
    finally:
        if descriptor is not None:
            module.os.close(descriptor)
        module.os.close(parent_descriptor)

    assert not tuple(parent.glob("*.tmp"))


@pytest.mark.parametrize("failure_number", (1, 2))
def test_private_directory_sync_failure_rolls_back_and_closes_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_number: int,
) -> None:
    descriptor_root = (
        Path("/dev/fd")
        if Path("/dev/fd").is_dir()
        else Path("/proc/self/fd")
    )
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_fsync = module.os.fsync
    matching_syncs = 0

    def fail_selected_private_parent_sync(descriptor: int) -> None:
        nonlocal matching_syncs
        if (
            final_parent.is_dir()
            and module._descriptor_path(descriptor) == final_parent
            and tuple(final_parent.glob("*.tmp"))
        ):
            matching_syncs += 1
            if matching_syncs == failure_number:
                raise OSError("private directory sync failed")
        original_fsync(descriptor)

    before = len(tuple(descriptor_root.iterdir()))
    monkeypatch.setattr(
        module.os, "fsync", fail_selected_private_parent_sync
    )
    with pytest.raises(OSError, match="private directory sync"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert matching_syncs >= failure_number
    assert len(tuple(descriptor_root.iterdir())) == before
    assert_no_temporary_or_claim_residue(artifact_root)

    recovered = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    recovered.close()
    assert_no_temporary_or_claim_residue(artifact_root)


def test_release_permission_error_does_not_block_exact_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    owned = parent / ".owned-displaced"
    owned.write_text("owned\n")
    preferred = parent / "preferred"
    preferred.write_text("unknown\n")
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    owned_descriptor = module.os.open(
        owned, module.os.O_RDONLY | module.os.O_NOFOLLOW
    )
    owned_metadata = module.os.fstat(owned_descriptor)
    owned_identity = (owned_metadata.st_dev, owned_metadata.st_ino)
    unknown_metadata = preferred.lstat()
    unknown_identity = (unknown_metadata.st_dev, unknown_metadata.st_ino)
    original_release = module._release_isolated_entry
    injected = False

    def deny_first_release(**kwargs: object) -> str:
        nonlocal injected
        if not injected:
            injected = True
            raise PermissionError("isolated release denied")
        return original_release(**kwargs)

    monkeypatch.setattr(module, "_release_isolated_entry", deny_first_release)
    try:
        with pytest.raises(PermissionError, match="isolated release denied"):
            module._remove_owned_entry(
                parent_descriptor=parent_descriptor,
                preferred_name=preferred.name,
                expected_identity=owned_identity,
                identity_descriptor=owned_descriptor,
                relocate_preferred_replacement=True,
            )

        assert injected
        assert not descendant_entries_with_identity(parent, owned_identity)
        assert descendant_entries_with_identity(parent, unknown_identity)
        assert (
            module._remove_owned_entry(
                parent_descriptor=parent_descriptor,
                preferred_name=preferred.name,
                expected_identity=owned_identity,
                identity_descriptor=owned_descriptor,
                relocate_preferred_replacement=True,
            )
            is False
        )
    finally:
        module.os.close(owned_descriptor)
        module.os.close(parent_descriptor)


@pytest.mark.parametrize("root_kind", ("archive", "snapshot"))
def test_removable_sync_failure_defers_until_after_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_make_removable = module._make_retained_directory_removable
    original_rename_noreplace = module._rename_noreplace
    target_identity: tuple[int, int] | None = None
    removable_injected = False
    publication_injected = False

    def fail_target_removable(**kwargs: object) -> None:
        nonlocal removable_injected, target_identity
        descriptor = kwargs["descriptor"]
        assert isinstance(descriptor, int)
        name = module._descriptor_path(descriptor).name
        matches = (
            name.startswith(".archive.")
            if root_kind == "archive"
            else name.startswith(f".{commit}.")
        )
        if not removable_injected and matches:
            removable_injected = True
            expected_identity = kwargs["expected_identity"]
            assert isinstance(expected_identity, tuple)
            target_identity = expected_identity
            raise OSError("retained directory sync failed")
        original_make_removable(**kwargs)

    def fail_snapshot_publication(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal publication_injected
        if (
            root_kind == "snapshot"
            and not publication_injected
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            publication_injected = True
            raise OSError("snapshot publication failed")
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module, "_make_retained_directory_removable", fail_target_removable
    )
    monkeypatch.setattr(module, "_rename_noreplace", fail_snapshot_publication)
    with pytest.raises(OSError, match="retained directory sync"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert removable_injected
    assert target_identity is not None
    assert not descendant_entries_with_identity(final_parent, target_identity)
    assert_no_temporary_or_claim_residue(artifact_root)


@pytest.mark.parametrize("error_type", (OSError, PermissionError))
def test_post_isolation_rename_sync_error_still_cleans_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    owned = parent / "owned"
    owned.write_text("owned\n")
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    owned_descriptor = module.os.open(
        owned, module.os.O_RDONLY | module.os.O_NOFOLLOW
    )
    metadata = module.os.fstat(owned_descriptor)
    owned_identity = (metadata.st_dev, metadata.st_ino)
    original_fsync = module.os.fsync
    injected = False

    def fail_first_post_move_sync(descriptor: int) -> None:
        nonlocal injected
        isolation = parent / module.ISOLATION_DIRECTORY_NAME
        if (
            not injected
            and isolation.is_dir()
            and tuple(isolation.glob("entry-*"))
        ):
            injected = True
            raise error_type("post-isolation rename sync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_post_move_sync)
    try:
        with pytest.raises(error_type, match="post-isolation rename sync"):
            module._remove_owned_entry(
                parent_descriptor=parent_descriptor,
                preferred_name=owned.name,
                expected_identity=owned_identity,
                identity_descriptor=owned_descriptor,
                relocate_preferred_replacement=True,
            )
    finally:
        module.os.close(owned_descriptor)
        module.os.close(parent_descriptor)

    assert injected
    assert not descendant_entries_with_identity(parent, owned_identity)


def test_fail_once_isolated_delete_retries_to_zero_exact_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    owned = parent / "owned"
    owned.write_text("owned\n")
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    owned_descriptor = module.os.open(
        owned, module.os.O_RDONLY | module.os.O_NOFOLLOW
    )
    metadata = module.os.fstat(owned_descriptor)
    owned_identity = (metadata.st_dev, metadata.st_ino)
    original_remove = module._remove_isolated_tree_entry
    injected = False

    def fail_first_isolated_delete(**kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("isolated delete failed")
        original_remove(**kwargs)

    monkeypatch.setattr(
        module, "_remove_isolated_tree_entry", fail_first_isolated_delete
    )
    try:
        with pytest.raises(OSError, match="isolated delete"):
            module._remove_owned_entry(
                parent_descriptor=parent_descriptor,
                preferred_name=owned.name,
                expected_identity=owned_identity,
                identity_descriptor=owned_descriptor,
                relocate_preferred_replacement=True,
            )
    finally:
        module.os.close(owned_descriptor)
        module.os.close(parent_descriptor)

    assert injected
    assert not descendant_entries_with_identity(parent, owned_identity)


def test_cleanup_reuses_one_isolation_boundary_per_parent(tmp_path: Path) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit

    assert len(tuple(final_parent.glob("*.isolation"))) == 1
    transaction.close()
    intent_parent = transaction.artifacts.intent_path.parent
    assert len(tuple(intent_parent.glob("*.isolation"))) == 1


def test_cleanup_reuses_isolation_boundary_across_module_instances(
    tmp_path: Path,
) -> None:
    first_module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    first = prepare_actual(
        repository, commit, artifact_root, module=first_module
    )
    first.close()

    second_module = load_lifecycle()
    second = prepare_actual(
        repository, commit, artifact_root, module=second_module
    )
    second.close()

    snapshot_parent = (
        artifact_root / "source-snapshots" / "mcore" / commit
    )
    intent_parent = second.artifacts.intent_path.parent
    assert len(tuple(snapshot_parent.glob("*.isolation"))) == 1
    assert len(tuple(intent_parent.glob("*.isolation"))) == 1


def test_cleanup_reuses_isolation_boundary_across_processes(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    first = prepare_actual(repository, commit, artifact_root, module=module)
    first.close()

    prepare_actual_in_subprocess(repository, commit, artifact_root)

    snapshot_parent = (
        artifact_root / "source-snapshots" / "mcore" / commit
    )
    intent_parent = first.artifacts.intent_path.parent
    assert len(tuple(snapshot_parent.glob("*.isolation"))) == 1
    assert len(tuple(intent_parent.glob("*.isolation"))) == 1
    assert_no_temporary_or_claim_residue(artifact_root)


def test_cooperative_processes_serialize_parent_namespace(
    tmp_path: Path,
) -> None:
    module = load_lifecycle()
    parent = tmp_path / "parent"
    parent.mkdir()
    attempted = tmp_path / "attempted"
    entered = tmp_path / "entered"
    script = """
import importlib.util
import os
import sys
from pathlib import Path

lifecycle_path = Path(sys.argv[1])
parent = Path(sys.argv[2])
attempted = Path(sys.argv[3])
entered = Path(sys.argv[4])
spec = importlib.util.spec_from_file_location(
    "submission_lifecycle_lock_subprocess", lifecycle_path
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    attempted.write_text("attempted\\n")
    with module._locked_parent_namespace(descriptor):
        entered.write_text("entered\\n")
finally:
    os.close(descriptor)
"""
    parent_descriptor = module.os.open(
        parent, module.os.O_RDONLY | module.os.O_DIRECTORY
    )
    publisher: subprocess.Popen[str] | None = None
    try:
        with module._locked_parent_namespace(parent_descriptor):
            publisher = subprocess.Popen(
                (
                    sys.executable,
                    "-c",
                    script,
                    str(LIFECYCLE_PATH),
                    str(parent),
                    str(attempted),
                    str(entered),
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5.0
            while not attempted.exists() and time.monotonic() < deadline:
                if publisher.poll() is not None:
                    break
                time.sleep(0.01)
            assert attempted.is_file()
            assert publisher.poll() is None
            assert not entered.exists()
        _, stderr = publisher.communicate(timeout=5.0)
        assert publisher.returncode == 0, stderr
    finally:
        module.os.close(parent_descriptor)
        if publisher is not None and publisher.poll() is None:
            publisher.kill()
            publisher.wait()

    assert entered.read_text() == "entered\n"


def test_shared_publication_mutations_hold_parent_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_lock = module._locked_parent_namespace
    original_mkdir = module.os.mkdir
    original_open = module.os.open
    original_link = module.os.link
    original_rename_noreplace = module._rename_noreplace
    lock_depth = 0
    observed: set[str] = set()

    @contextmanager
    def record_parent_lock(descriptor: int) -> Iterator[None]:
        nonlocal lock_depth
        with original_lock(descriptor):
            lock_depth += 1
            try:
                yield
            finally:
                lock_depth -= 1

    def require_locked_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if dir_fd is not None:
            parent = module._descriptor_path(dir_fd)
            if parent == artifact_root or parent.is_relative_to(artifact_root):
                observed.add("mkdir")
                assert lock_depth, "shared mkdir executed outside parent flock"
        original_mkdir(path, mode, dir_fd=dir_fd)

    def require_locked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None and flags & os.O_CREAT and flags & os.O_EXCL:
            parent = module._descriptor_path(dir_fd)
            if (
                (parent == artifact_root or parent.is_relative_to(artifact_root))
                and not any(part.endswith(".tmp") for part in parent.parts)
            ):
                label = (
                    "claim"
                    if Path(path).name.endswith(".claim")
                    else "intent"
                )
                observed.add(label)
                assert lock_depth, f"shared {label} create missed parent flock"
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def require_locked_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        observed.add("intent-link")
        assert lock_depth, "intent publication link missed parent flock"
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def require_locked_rename(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name.endswith(".tmp") and len(destination_name) == 64:
            observed.add("snapshot")
            assert lock_depth, "snapshot publication missed parent flock"
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(module, "_locked_parent_namespace", record_parent_lock)
    monkeypatch.setattr(module.os, "mkdir", require_locked_mkdir)
    monkeypatch.setattr(module.os, "open", require_locked_open)
    monkeypatch.setattr(module.os, "link", require_locked_link)
    monkeypatch.setattr(module, "_rename_noreplace", require_locked_rename)

    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    transaction.close()

    assert observed >= {"mkdir", "claim", "snapshot", "intent", "intent-link"}


def test_cleanup_retains_no_isolation_file_descriptors(tmp_path: Path) -> None:
    descriptor_root = (
        Path("/dev/fd")
        if Path("/dev/fd").is_dir()
        else Path("/proc/self/fd")
    )
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    before = len(tuple(descriptor_root.iterdir()))

    transaction = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    transaction.close()

    assert len(tuple(descriptor_root.iterdir())) == before


def test_denied_unknown_source_does_not_block_exact_private_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    original_move = module._move_shared_entry_into_isolation
    private_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    denied_name: str | None = None
    injected = False

    def displace_source_then_raise_enoent(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal denied_name, injected, private_identity
        nonlocal replacement_identity
        if (
            not injected
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            source_root = final_parent / source_name
            metadata = source_root.lstat()
            private_identity = (metadata.st_dev, metadata.st_ino)
            source_root.chmod(
                stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR
            )
            source_root.rename(final_parent / ".verified-private-displaced")
            source_root.mkdir()
            (source_root / "unrelated.txt").write_text("preserve me\n")
            metadata = source_root.lstat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            denied_name = source_name
            injected = True
            raise FileNotFoundError(
                errno.ENOENT, "source displaced", source_name
            )
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    def deny_unowned_preferred_name(**kwargs: object) -> bool:
        if kwargs["source_name"] == denied_name:
            raise PermissionError(errno.EACCES, "denied replacement")
        return original_move(**kwargs)

    monkeypatch.setattr(
        module, "_rename_noreplace", displace_source_then_raise_enoent
    )
    monkeypatch.setattr(
        module,
        "_move_shared_entry_into_isolation",
        deny_unowned_preferred_name,
    )
    with pytest.raises(FileNotFoundError, match="source displaced"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert denied_name is not None
    assert private_identity is not None
    assert replacement_identity is not None
    assert not descendant_entries_with_identity(
        final_parent, private_identity
    )
    preserved = entries_with_identity(final_parent, replacement_identity)
    assert preserved == (final_parent / denied_name,)
    assert (preserved[0] / "unrelated.txt").read_text() == "preserve me\n"

    recovered = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    assert recovered.snapshot_created is True
    recovered.close()


def test_submission_preparation_accepts_identical_eexist_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    winner_published = False

    def publish_winner_then_raise(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal winner_published
        if source_name.endswith(".tmp") and len(destination_name) == 64:
            assert source_parent_descriptor == destination_parent_descriptor
            source_root = (
                module._descriptor_path(source_parent_descriptor) / source_name
            )
            final_root = final_parent / destination_name
            module.shutil.copytree(source_root, final_root, symlinks=True)
            module._make_tree_read_only(final_root)
            winner_published = True
            raise FileExistsError(
                errno.EEXIST, "winner exists", destination_name
            )
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module, "_rename_noreplace", publish_winner_then_raise, raising=False
    )
    transaction = prepare_actual(repository, commit, artifact_root, module=module)

    assert winner_published
    assert transaction.snapshot_created is False
    assert transaction.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()


@pytest.mark.parametrize("destination_kind", ("file", "symlink", "empty"))
def test_final_publication_never_clobbers_conflicting_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_kind: str,
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    outside = tmp_path / "outside"
    outside.mkdir()
    original_rename_noreplace = module._rename_noreplace
    installed: Path | None = None

    def install_conflict_then_raise(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal installed
        if source_name.endswith(".tmp") and len(destination_name) == 64:
            assert source_parent_descriptor == destination_parent_descriptor
            installed = final_parent / destination_name
            if destination_kind == "file":
                installed.write_text("conflict\n")
            elif destination_kind == "symlink":
                installed.symlink_to(outside, target_is_directory=True)
            else:
                installed.mkdir()
            raise FileExistsError(
                errno.EEXIST, "conflict exists", destination_name
            )
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module, "_rename_noreplace", install_conflict_then_raise, raising=False
    )
    with pytest.raises(ValueError, match="snapshot"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert installed is not None
    if destination_kind == "file":
        assert installed.read_text() == "conflict\n"
    elif destination_kind == "symlink":
        assert installed.is_symlink()
        assert installed.resolve() == outside
    else:
        assert installed.is_dir()
        assert not tuple(installed.iterdir())
    assert_no_temporary_or_claim_residue(artifact_root)


def test_unavailable_noreplace_primitive_uses_locked_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"

    def unavailable_noreplace(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        raise OSError(errno.ENOSYS, "no no-replace rename primitive")

    monkeypatch.setattr(
        module, "_rename_noreplace", unavailable_noreplace, raising=False
    )
    transaction = prepare_actual(repository, commit, artifact_root, module=module)

    assert transaction.snapshot_created is True
    assert transaction.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()


def test_snapshot_publication_falls_back_when_noreplace_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    original_rename_noreplace = module._rename_noreplace

    def reject_final_publication_flags(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name.endswith(".tmp") and len(destination_name) == 64:
            raise OSError(errno.EINVAL, "filesystem rejects rename flags")
        original_rename_noreplace(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        reject_final_publication_flags,
    )

    transaction = prepare_actual(repository, commit, artifact_root, module=module)

    assert transaction.snapshot_created is True
    assert transaction.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()


def test_unavailable_noreplace_replacement_preserves_unknown_and_cleans_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    original_rename_noreplace = module._rename_noreplace
    private_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    injected = False

    def replace_source_then_report_unavailable(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal injected, private_identity, replacement_identity
        if (
            not injected
            and source_name.endswith(".tmp")
            and len(destination_name) == 64
        ):
            source_root = final_parent / source_name
            metadata = source_root.lstat()
            private_identity = (metadata.st_dev, metadata.st_ino)
            source_root.chmod(
                stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR
            )
            source_root.rename(final_parent / ".verified-private-displaced")
            source_root.mkdir()
            (source_root / "unrelated.txt").write_text("preserve me\n")
            metadata = source_root.lstat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            injected = True
        raise OSError(errno.ENOSYS, "no no-replace rename primitive")

    monkeypatch.setattr(
        module,
        "_rename_noreplace",
        replace_source_then_report_unavailable,
    )
    with pytest.raises(ValueError, match="final destination changed"):
        prepare_actual(repository, commit, artifact_root, module=module)

    assert injected
    assert private_identity is not None
    assert replacement_identity is not None
    assert not descendant_entries_with_identity(
        final_parent, private_identity
    )
    preserved = descendant_entries_with_identity(
        final_parent, replacement_identity
    )
    assert len(preserved) == 1
    assert (preserved[0] / "unrelated.txt").read_text() == "preserve me\n"
    assert_no_temporary_or_claim_residue(artifact_root)

    monkeypatch.setattr(module, "_rename_noreplace", original_rename_noreplace)
    recovered = prepare_actual(
        repository, commit, artifact_root, module=module
    )
    assert recovered.snapshot_created is True
    assert not descendant_entries_with_identity(
        final_parent, private_identity
    )
    assert descendant_entries_with_identity(
        final_parent, replacement_identity
    ) == preserved
    recovered.close()


def test_successful_final_publication_leaves_no_private_state(tmp_path: Path) -> None:
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    transaction = prepare_actual(repository, commit, artifact_root)

    assert transaction.snapshot_created is True
    assert transaction.artifacts.snapshot_root.is_dir()
    assert_no_temporary_or_claim_residue(artifact_root)
    transaction.close()


def test_snapshot_chmod_is_followed_by_inode_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    events: list[tuple[str, tuple[int, int]]] = []
    original_chmod = module.os.chmod
    original_fchmod = module.os.fchmod
    original_fsync = module.os.fsync

    def record_chmod(path: object, mode: int, *args: object, **kwargs: object) -> None:
        original_chmod(path, mode, *args, **kwargs)  # type: ignore[arg-type]
        if mode & 0o222:
            return
        dir_fd = kwargs.get("dir_fd")
        metadata = module.os.stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=bool(kwargs.get("follow_symlinks", True)),
        )
        events.append(("chmod", (metadata.st_dev, metadata.st_ino)))

    def record_fchmod(descriptor: int, mode: int) -> None:
        original_fchmod(descriptor, mode)
        if not mode & 0o222:
            metadata = module.os.fstat(descriptor)
            events.append(("chmod", (metadata.st_dev, metadata.st_ino)))

    def record_fsync(descriptor: int) -> None:
        original_fsync(descriptor)
        metadata = module.os.fstat(descriptor)
        events.append(("fsync", (metadata.st_dev, metadata.st_ino)))

    monkeypatch.setattr(module.os, "chmod", record_chmod)
    monkeypatch.setattr(module.os, "fchmod", record_fchmod)
    monkeypatch.setattr(module.os, "fsync", record_fsync)
    transaction = prepare_actual(repository, commit, tmp_path / "logs", module=module)
    transaction.close()

    unsynced = tuple(
        identity
        for index, (operation, identity) in enumerate(events)
        if operation == "chmod"
        and not any(
            later_operation == "fsync" and later_identity == identity
            for later_operation, later_identity in events[index + 1 :]
        )
    )
    assert not unsynced
