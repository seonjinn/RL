import importlib.util
from pathlib import Path

import pytest


EXPERIMENT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("source_guard", EXPERIMENT / "source_guard.py")
assert SPEC is not None and SPEC.loader is not None
source_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source_guard)


@pytest.mark.parametrize(
    ("prefix", "meaning"),
    [
        ("-", "uninitialized"),
        ("+", "unexpected commit"),
        ("U", "merge conflict"),
    ],
)
def test_source_guard_rejects_non_exact_recursive_submodule_state(
    prefix: str,
    meaning: str,
) -> None:
    status = f"{prefix}0123456789abcdef0123456789abcdef01234567 3rdparty/example"

    with pytest.raises(ValueError, match=rf"{meaning}.*3rdparty/example"):
        source_guard.validate_submodule_status(status)


def test_source_guard_accepts_initialized_exact_recursive_submodules() -> None:
    status = " 0123456789abcdef0123456789abcdef01234567 3rdparty/example"

    assert source_guard.validate_submodule_status(status) == [
        "0123456789abcdef0123456789abcdef01234567 3rdparty/example"
    ]
