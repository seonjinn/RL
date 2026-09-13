"""Build only the diagnostic CUDA allocator on a compute node, then test it."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def main() -> None:
    source = Path("/cumem-source")
    build = Path(os.environ["XDG_CACHE_HOME"]).parent / "cumem-build"
    build.mkdir(parents=True, exist_ok=True)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    extension = build / f"cumem_allocator{sysconfig.get_config_var('EXT_SUFFIX')}"
    for name in ("cumem_allocator.cpp", "cumem_allocator_compat.h"):
        path = source / name
        print(f"SOURCE {name} {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    command = [
        "g++", "-std=c++17", "-O2", "-shared", "-fPIC",
        f"-I{sysconfig.get_path('include')}", f"-I{cuda / 'include'}",
        str(source / "cumem_allocator.cpp"),
        f"-L{cuda / 'lib64' / 'stubs'}", "-lcuda", "-o", str(extension),
    ]
    print(command, flush=True)
    subprocess.run(command, check=True, timeout=180)
    print(f"BINARY {hashlib.sha256(extension.read_bytes()).hexdigest()}", flush=True)
    environment = dict(os.environ, NRL_CUMEM_EXTENSION=str(extension))
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("probe_cumem_lifecycle.py"))],
        env=environment, check=True, timeout=600,
    )


if __name__ == "__main__":
    main()
