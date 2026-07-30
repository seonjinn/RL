"""Run pytest only after validating the mounted Transformer Engine overlay."""

import argparse
from collections.abc import Sequence

import pytest

from validate_te_fp64_overlay import validate_overlay


def run_pytest(
    *,
    expected_version: str,
    expected_sha256: str,
    pytest_args: Sequence[str],
) -> int:
    """Validate the loaded overlay and run pytest in the same Python process."""
    validate_overlay(
        expected_version=expected_version,
        expected_sha256=expected_sha256,
    )
    return pytest.main(list(pytest_args))


def parse_args() -> argparse.Namespace:
    """Parse overlay provenance requirements and pytest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """Validate the overlay and exit with pytest's status."""
    args = parse_args()
    if not args.pytest_args:
        raise ValueError("At least one pytest argument is required")
    raise SystemExit(
        run_pytest(
            expected_version=args.expected_version,
            expected_sha256=args.expected_sha256,
            pytest_args=args.pytest_args,
        )
    )


if __name__ == "__main__":
    main()
