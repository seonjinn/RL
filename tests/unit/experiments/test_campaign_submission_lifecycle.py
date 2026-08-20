from __future__ import annotations

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
    original_mkdir = module.os.mkdir

    def raced_mkdir(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], mode: int = 0o777) -> None:
        candidate = Path(path)
        if candidate.parent.name == commit and len(candidate.name) == 64:
            candidate.symlink_to(outside, target_is_directory=True)
            raise FileExistsError
        original_mkdir(path, mode)

    monkeypatch.setattr(module.os, "mkdir", raced_mkdir)
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
