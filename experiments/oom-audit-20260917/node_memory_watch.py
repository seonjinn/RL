"""Bounded node/cgroup snapshots before framework imports; no CUDA dependency."""

import argparse
import json
import socket
import time
from pathlib import Path


def read_cgroup_chain(leaf: Path, root: Path) -> list[dict[str, str]]:
    leaf.relative_to(root)
    records: list[dict[str, str]] = []
    for path in (leaf, *leaf.parents):
        if not path.is_relative_to(root):
            break
        record = {"path": str(path)}
        for name in (
            "memory.current", "memory.peak", "memory.max", "memory.events",
            "memory.events.local", "memory.stat", "memory.numa_stat",
        ):
            try:
                record[name] = (path / name).read_text().strip()
            except OSError:
                continue
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=14400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    relative = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    root = Path("/sys/fs/cgroup")
    leaf = root / relative.lstrip("/")
    deadline = time.monotonic() + args.duration
    with (args.output / f"node-memory-{socket.gethostname()}.jsonl").open("a") as output:
        while time.monotonic() < deadline:
            record = {
                "timestamp_ns": time.time_ns(),
                "cgroups": read_cgroup_chain(leaf, root),
                "meminfo": Path("/proc/meminfo").read_text(),
                "pressure": Path("/proc/pressure/memory").read_text(),
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            time.sleep(15)


if __name__ == "__main__":
    main()
