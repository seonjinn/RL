"""Read-only container baseline: no model or framework imports."""

import json
import os
import sys
import time
from pathlib import Path

record = {"label": sys.argv[1], "timestamp_ns": time.time_ns()}
record["meminfo"] = Path("/proc/meminfo").read_text()
record["mounts"] = [
    line
    for line in Path("/proc/self/mountinfo").read_text().splitlines()
    if line.split()[4] in ("/", "/tmp", "/opt", "/dev/shm", "/raid/scratch")
]
relative = next(
    line.split(":", 2)[2]
    for line in Path("/proc/self/cgroup").read_text().splitlines()
    if line.startswith("0::")
)
cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
record["cgroups"] = {}
for path in (cgroup, cgroup.parent, cgroup.parent.parent):
    record["cgroups"][str(path)] = {
        name: (path / name).read_text().strip()
        for name in ("memory.current", "memory.peak", "memory.max", "memory.stat")
        if (path / name).is_file()
    }
record["filesystems"] = {}
for path in ("/", "/tmp", "/dev/shm", "/raid/scratch"):
    try:
        stats = os.statvfs(path)
        record["filesystems"][path] = {
            "total_bytes": stats.f_blocks * stats.f_frsize,
            "used_bytes": (stats.f_blocks - stats.f_bfree) * stats.f_frsize,
        }
    except OSError as exc:
        record["filesystems"][path] = {"error": type(exc).__name__}
print(json.dumps(record, sort_keys=True), flush=True)
