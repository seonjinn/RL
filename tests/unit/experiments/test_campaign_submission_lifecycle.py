from __future__ import annotations

import builtins
import errno
import importlib.util
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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

    def install_symlink_winner(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        assert source_parent_descriptor == destination_parent_descriptor
        assert source_name.endswith(".tmp")
        (final_parent / destination_name).symlink_to(
            outside, target_is_directory=True
        )
        raise FileExistsError(errno.EEXIST, "symlink winner", destination_name)

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

    assert close_faults == 2
    assert_no_temporary_or_claim_residue(tmp_path / "logs")


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


def test_submission_preparation_accepts_identical_eexist_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_lifecycle()
    repository, commit = git_repository(tmp_path / "candidate")
    artifact_root = tmp_path / "logs"
    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    winner_published = False

    def publish_winner_then_raise(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal winner_published
        assert source_parent_descriptor == destination_parent_descriptor
        source_root = module._descriptor_path(source_parent_descriptor) / source_name
        final_root = final_parent / destination_name
        module.shutil.copytree(source_root, final_root, symlinks=True)
        module._make_tree_read_only(final_root)
        winner_published = True
        raise FileExistsError(errno.EEXIST, "winner exists", destination_name)

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
    installed: Path | None = None

    def install_conflict_then_raise(
        source_parent_descriptor: int,
        source_name: str,
        destination_parent_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal installed
        assert source_parent_descriptor == destination_parent_descriptor
        assert source_name.endswith(".tmp")
        installed = final_parent / destination_name
        if destination_kind == "file":
            installed.write_text("conflict\n")
        elif destination_kind == "symlink":
            installed.symlink_to(outside, target_is_directory=True)
        else:
            installed.mkdir()
        raise FileExistsError(errno.EEXIST, "conflict exists", destination_name)

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


def test_unavailable_noreplace_primitive_fails_closed_and_cleans_private_state(
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
    with pytest.raises(OSError, match="no-replace rename"):
        prepare_actual(repository, commit, artifact_root, module=module)

    final_parent = artifact_root / "source-snapshots" / "mcore" / commit
    assert not tuple(path for path in final_parent.iterdir() if len(path.name) == 64)
    assert_no_temporary_or_claim_residue(artifact_root)


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
