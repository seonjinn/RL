"""Behavior tests for the cross-architecture rclone launcher."""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).parents[3]
WRAPPER = ROOT / "experiments/q235_rp25_perf_20260917/rclone_arch_wrapper.sh"


def test_rclone_wrapper_selects_arm_binary_and_forwards_every_argument(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "rclone-arm64"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    binary.chmod(0o755)
    environment = os.environ | {
        "RCLONE_ARCH": "aarch64",
        "RCLONE_ROOT": str(tmp_path),
    }

    result = subprocess.run(
        [str(WRAPPER), "copy", "remote:path", "/destination"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["copy", "remote:path", "/destination"]
