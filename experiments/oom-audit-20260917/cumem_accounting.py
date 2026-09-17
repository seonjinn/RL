"""Guarded arithmetic for the experimental awake-only CuMem accounting probe."""

from collections.abc import Iterable, Mapping, Set


def unmapped_idle_bytes(
    segments: Iterable[Mapping[str, int]],
    registered: Set[int],
    device: int,
    reserved: int,
) -> int:
    missing = 0
    for segment in segments:
        if segment["device"] != device or segment["address"] in registered:
            continue
        if segment["allocated_size"] != 0 or segment["active_size"] != 0:
            raise ValueError("Missing CuMem registration for a non-idle segment")
        if segment["total_size"] < 0:
            raise ValueError("Negative segment size")
        missing += segment["total_size"]
    if not 0 <= missing <= reserved:
        raise ValueError("Unmapped bytes exceed reported reserved memory")
    return missing
