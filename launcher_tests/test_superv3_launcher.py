import os
import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "launch_superv3_main_short.sh"


def _run_launcher(tmp_path: Path, submit_nodelist: str | None) -> list[str]:
    workdir = tmp_path / "workdir"
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoint"
    pretrained_dir = tmp_path / "pretrained"
    tokenizer_dir = tmp_path / "tokenizer"
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    bin_dir = tmp_path / "bin"
    container = tmp_path / "container.sqsh"
    capture = tmp_path / "sbatch.argv"

    for directory in (
        workdir / "examples/configs",
        data_dir,
        checkpoint_dir,
        pretrained_dir,
        tokenizer_dir,
        cache_dir,
        log_dir,
        bin_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        workdir / "ray.sub",
        workdir / "examples/configs/sft_superv3_prepacked.yaml",
        data_dir / "train.jsonl.packed",
        container,
    ):
        path.touch()

    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$SBATCH_CAPTURE"\n')
    sbatch.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture),
        "WORKDIR": str(workdir),
        "DATA_DIR": str(data_dir),
        "TRAIN_FILE": "train.jsonl.packed",
        "ORIG_MLM_CKPT": str(checkpoint_dir),
        "PRETRAINED_DIR": str(pretrained_dir),
        "TOKENIZER_DIR": str(tokenizer_dir),
        "SHARED_CACHE_ROOT": str(cache_dir),
        "LOG_DIR": str(log_dir),
        "CONTAINER": str(container),
        "SLURM_NODELIST": "ambient-allocation",
    }
    if submit_nodelist is not None:
        env["SUBMIT_NODELIST"] = submit_nodelist

    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True)
    return [arg.decode() for arg in capture.read_bytes().split(b"\0") if arg]


@pytest.mark.parametrize("submit_nodelist", [None, ""])
def test_superv3_launcher_ignores_unset_or_empty_nodelist(
    tmp_path: Path, submit_nodelist: str | None
) -> None:
    args = _run_launcher(tmp_path, submit_nodelist)

    assert not any(arg.startswith("--nodelist=") for arg in args)


def test_superv3_launcher_forwards_an_explicit_nodelist_as_one_argument(
    tmp_path: Path,
) -> None:
    nodelist = "pool0-[00105,00116,00391-00393]"
    args = _run_launcher(tmp_path, nodelist)

    assert args.count(f"--nodelist={nodelist}") == 1
