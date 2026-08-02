from __future__ import annotations

import argparse
from pathlib import Path


def prepare_symlink_parents(cubin_dir: Path) -> tuple[Path, Path]:
    parents = (
        cubin_dir / "flashinfer" / "trtllm" / "batched_gemm",
        cubin_dir / "flashinfer" / "trtllm" / "gemm",
    )
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    return parents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubin-dir", type=Path)
    args = parser.parse_args()

    cubin_dir = args.cubin_dir
    if cubin_dir is None:
        from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

        cubin_dir = FLASHINFER_CUBIN_DIR

    for prepared in prepare_symlink_parents(cubin_dir.resolve()):
        print(f"flashinfer_symlink_parent={prepared}")


if __name__ == "__main__":
    main()
