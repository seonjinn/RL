# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded staging for immutable drafter checkpoints in the matrix."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.vllm_0251_drafter_matrix.matrix import (
    CheckpointSpec,
    G_CLUSTERS,
    G_VARIANTS,
    validate_snapshot,
)


DEFAULT_CONTAINER = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
    "nemo_rl_nightly_20260715.sqsh"
)
DEFAULT_MOUNTS = "/lustre:/lustre"
MANIFEST_NAME = "drafter-staging-manifest.json"
CONTAINER_PYTHON = "/opt/nemo_rl_venv/bin/python"
_SHA_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Machine-readable state for one immutable checkpoint snapshot."""

    repo_id: str
    revision: str
    status: Literal["planned", "test-only", "queued", "staged", "failed"]
    path: str
    job_id: str | None


class StagingFailure(RuntimeError):
    """A staging error that retains manifest entries for failure reporting."""

    def __init__(self, message: str, entries: tuple[ManifestEntry, ...]) -> None:
        super().__init__(message)
        self.entries = entries


SnapshotDownload = Callable[..., str]
SnapshotDownloadFactory = Callable[[], SnapshotDownload]


def _lyris_hf_home() -> Path:
    for cluster in G_CLUSTERS:
        if cluster.key == "lyris":
            return cluster.hf_home
    raise RuntimeError("The drafter matrix has no Lyris cluster")


def collect_checkpoint_specs() -> tuple[CheckpointSpec, ...]:
    """Return unique immutable matrix checkpoints, ordered by repository identity."""
    unique: dict[tuple[str, str], CheckpointSpec] = {}
    for variant in G_VARIANTS:
        for checkpoint in variant.checkpoints:
            if not _SHA_REVISION.fullmatch(checkpoint.revision):
                raise RuntimeError(
                    "Drafter staging requires a full immutable SHA revision: "
                    f"{checkpoint.repo_id}@{checkpoint.revision}"
                )
            unique.setdefault((checkpoint.repo_id, checkpoint.revision), checkpoint)
    return tuple(unique[key] for key in sorted(unique))


def _entries_for(
    checkpoints: Sequence[CheckpointSpec],
    hf_home: Path,
    status: Literal["planned", "test-only", "queued", "staged", "failed"],
    job_id: str | None,
) -> tuple[ManifestEntry, ...]:
    return tuple(
        ManifestEntry(
            repo_id=checkpoint.repo_id,
            revision=checkpoint.revision,
            status=status,
            path=str(checkpoint.snapshot_path(hf_home)),
            job_id=job_id,
        )
        for checkpoint in checkpoints
    )


def stage_targets(
    checkpoints: Sequence[CheckpointSpec],
    hf_home: Path,
    snapshot_download: SnapshotDownload,
    job_id: str | None,
) -> tuple[ManifestEntry, ...]:
    """Download exact snapshots and reject any cache layout mismatch."""
    entries: list[ManifestEntry] = list(
        _entries_for(checkpoints, hf_home, "planned", job_id)
    )
    for index, checkpoint in enumerate(checkpoints):
        expected_path = checkpoint.snapshot_path(hf_home)
        try:
            downloaded_path = Path(
                snapshot_download(
                    repo_id=checkpoint.repo_id,
                    revision=checkpoint.revision,
                    cache_dir=hf_home / "hub",
                )
            )
            if downloaded_path != expected_path:
                raise RuntimeError(
                    "snapshot_download returned an unexpected snapshot path: "
                    f"expected {expected_path}, got {downloaded_path}"
                )
            validate_snapshot(
                downloaded_path,
                checkpoint.revision,
                f"Drafter {checkpoint.repo_id}",
            )
        except Exception as error:
            entries[index] = ManifestEntry(
                repo_id=checkpoint.repo_id,
                revision=checkpoint.revision,
                status="failed",
                path=str(expected_path),
                job_id=job_id,
            )
            raise StagingFailure(str(error), tuple(entries)) from error
        entries[index] = ManifestEntry(
            repo_id=checkpoint.repo_id,
            revision=checkpoint.revision,
            status="staged",
            path=str(expected_path),
            job_id=job_id,
        )
    return tuple(entries)


def write_manifest(
    output_dir: Path,
    entries: Sequence[ManifestEntry],
    *,
    status: str | None = None,
    error: str | None = None,
) -> Path:
    """Atomically write the checkpoint staging manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / MANIFEST_NAME
    payload: dict[str, object] = {
        "checkpoints": [asdict(entry) for entry in entries]
    }
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{MANIFEST_NAME}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    return destination


def build_sbatch_command(
    *,
    mode: Literal["test-only", "submit"],
    output_dir: Path,
    hf_home: Path,
    container: Path,
    mounts: str,
    wrapper_path: Path,
    worker_path: Path,
) -> tuple[str, ...]:
    """Build the bounded Lyris staging job without requesting GPU resources."""
    if mode == "test-only":
        submission_flag = "--test-only"
    elif mode == "submit":
        submission_flag = "--parsable"
    else:
        raise ValueError(f"Unsupported staging submission mode: {mode}")
    return (
        "sbatch",
        submission_flag,
        "--dependency=",
        "--account=coreai_dlalgo_llm",
        "--partition=gb200",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--exclusive",
        "--time=02:00:00",
        "--segment=1",
        "--job-name=nemorl-drafter-stage",
        f"--output={output_dir / 'slurm-%j.out'}",
        f"--export=ALL,HF_HOME={hf_home}",
        f"--container-image={container}",
        f"--container-mounts={mounts}",
        str(wrapper_path),
        "--worker-script",
        str(worker_path),
        "--output-dir",
        str(output_dir),
        "--hf-home",
        str(hf_home),
    )


def _parse_job_id(stdout: str) -> str:
    job_id = stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch --parsable returned an invalid job ID: {stdout!r}")
    return job_id


def _add_location_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, default=_lyris_hf_home())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("show", "test-only", "submit"):
        subparser = subparsers.add_parser(command)
        _add_location_arguments(subparser)
        if command in {"test-only", "submit"}:
            subparser.add_argument("--container", type=Path, default=Path(DEFAULT_CONTAINER))
            subparser.add_argument("--mounts", default=DEFAULT_MOUNTS)
            subparser.add_argument(
                "--wrapper-path",
                type=Path,
                default=Path(__file__).with_name("submit_stage_drafters.sh"),
            )
            subparser.add_argument(
                "--worker-path",
                type=Path,
                default=Path(__file__),
            )
    return parser


def _emit_manifest(path: Path) -> None:
    print(path.read_text(encoding="utf-8"), end="")


def _snapshot_download_factory() -> SnapshotDownload:
    module: Any = importlib.import_module("huggingface_hub")
    return cast(SnapshotDownload, module.snapshot_download)


def run_stage(
    output_dir: Path,
    hf_home: Path,
    *,
    checkpoints: Sequence[CheckpointSpec] | None = None,
    snapshot_download_factory: SnapshotDownloadFactory = _snapshot_download_factory,
) -> Path:
    """Stage every checkpoint and persist a terminal manifest on any failure."""
    resolved_checkpoints: tuple[CheckpointSpec, ...] = ()
    job_id = os.environ.get("SLURM_JOB_ID")
    try:
        resolved_checkpoints = tuple(
            collect_checkpoint_specs() if checkpoints is None else checkpoints
        )
        snapshot_download = snapshot_download_factory()
        entries = stage_targets(
            resolved_checkpoints, hf_home, snapshot_download, job_id
        )
    except StagingFailure as error:
        manifest_path = write_manifest(
            output_dir,
            error.entries,
            status="failed",
            error=str(error),
        )
        _emit_manifest(manifest_path)
        raise
    except Exception as error:
        failed_entries = _entries_for(
            resolved_checkpoints, hf_home, "failed", job_id
        )
        manifest_path = write_manifest(
            output_dir,
            failed_entries,
            status="failed",
            error=str(error),
        )
        _emit_manifest(manifest_path)
        raise
    manifest_path = write_manifest(output_dir, entries, status="staged")
    _emit_manifest(manifest_path)
    return manifest_path


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_location_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the staging CLI."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ("--worker",):
        args = _worker_parser().parse_args(arguments[1:])
        run_stage(args.output_dir, args.hf_home)
        return 0

    args = _build_parser().parse_args(arguments)
    checkpoints = collect_checkpoint_specs()
    if args.command == "show":
        manifest_path = write_manifest(
            args.output_dir, _entries_for(checkpoints, args.hf_home, "planned", None)
        )
        _emit_manifest(manifest_path)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    container = args.container.resolve()
    wrapper_path = args.wrapper_path.resolve()
    worker_path = args.worker_path.resolve()
    for label, path in (
        ("container", container),
        ("staging wrapper", wrapper_path),
        ("staging worker", worker_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    command = build_sbatch_command(
        mode=args.command,
        output_dir=args.output_dir,
        hf_home=args.hf_home,
        container=container,
        mounts=args.mounts,
        wrapper_path=wrapper_path,
        worker_path=worker_path,
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if args.command == "test-only":
        entries = _entries_for(checkpoints, args.hf_home, "test-only", None)
    else:
        job_id = _parse_job_id(result.stdout)
        entries = _entries_for(checkpoints, args.hf_home, "queued", job_id)
    manifest_path = write_manifest(args.output_dir, entries, status=args.command)
    _emit_manifest(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
