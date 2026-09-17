"""Correlate CPU backup storage with Linux VMAs without claiming VMA ownership."""

import re


def backup_mappings(
    smaps: str, regions: list[tuple[int, int]]
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    current: dict[str, str | int] | None = None
    for line in smaps.splitlines():
        match = re.match(r"^([0-9a-f]+)-([0-9a-f]+)\s", line)
        if match:
            start, end = (int(value, 16) for value in match.groups())
            current = None
            if any(
                size > 0 and address < end and address + size > start
                for address, size in regions
            ):
                current = {"mapping": line}
                rows.append(current)
        elif current is not None:
            fields = line.split()
            if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
                current[fields[0].rstrip(":")] = int(fields[1]) * 1024
    return rows
