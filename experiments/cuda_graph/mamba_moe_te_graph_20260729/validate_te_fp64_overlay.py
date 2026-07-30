"""Validate the immutable Transformer Engine FP64 weak-reference overlay."""

import argparse
import hashlib
import json
import stat
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch.utils as te_utils


def validate_overlay(
    *,
    expected_version: str,
    expected_sha256: str,
) -> dict[str, str]:
    """Validate the mounted Transformer Engine FP64 weak-reference overlay.

    Args:
        expected_version: Exact Transformer Engine package version in the image.
        expected_sha256: SHA256 digest of the mounted ``utils.py`` source file.

    Returns:
        Immutable-overlay provenance after all version, source, and CUDA checks pass.

    Raises:
        RuntimeError: If the mounted package, source, or FP64 weak reference differs
            from the reviewed overlay.
    """
    version = transformer_engine.__version__
    if version != expected_version:
        raise RuntimeError(
            "Transformer Engine version mismatch: "
            f"expected {expected_version}, found {version}"
        )

    utils_path = Path(te_utils.__file__).resolve()
    actual_sha256 = hashlib.sha256(utils_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Transformer Engine utils.py SHA256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256} at {utils_path}"
        )

    actual_mode = stat.S_IMODE(utils_path.stat().st_mode)
    if actual_mode != 0o444:
        raise RuntimeError(
            "Transformer Engine utils.py mode mismatch: "
            f"expected 0444, found {actual_mode:04o} at {utils_path}"
        )

    try:
        typestr = te_utils._torch_dtype_to_np_typestr_dict[torch.float64]
    except KeyError as error:
        raise RuntimeError("Transformer Engine has no FP64 CUDA array typestring") from error
    if typestr != "<f8":
        raise RuntimeError(
            "Transformer Engine FP64 CUDA array typestring mismatch: "
            f"expected <f8, found {typestr}"
        )

    source = torch.arange(4, device="cuda", dtype=torch.float64)
    weak = te_utils.make_weak_ref(source)
    if weak.dtype is not torch.float64:
        raise RuntimeError(
            f"FP64 weak reference changed dtype: expected {torch.float64}, found {weak.dtype}"
        )
    if weak.shape != source.shape:
        raise RuntimeError(
            f"FP64 weak reference changed shape: expected {source.shape}, found {weak.shape}"
        )
    if weak.data_ptr() != source.data_ptr():
        raise RuntimeError(
            "FP64 weak reference changed data pointer: "
            f"expected {source.data_ptr()}, found {weak.data_ptr()}"
        )

    provenance = {
        "te_version": version,
        "te_utils_path": str(utils_path),
        "te_utils_sha256": actual_sha256,
        "te_utils_mode": f"{actual_mode:04o}",
        "fp64_typestr": typestr,
        "fp64_dtype": str(weak.dtype),
        "fp64_shape": str(tuple(weak.shape)),
        "fp64_data_ptr": str(weak.data_ptr()),
    }
    print(json.dumps(provenance, sort_keys=True))
    return provenance


def parse_args() -> argparse.Namespace:
    """Parse immutable overlay provenance requirements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    """Run the immutable overlay preflight before Ray startup."""
    args = parse_args()
    validate_overlay(
        expected_version=args.expected_version,
        expected_sha256=args.expected_sha256,
    )


if __name__ == "__main__":
    main()
