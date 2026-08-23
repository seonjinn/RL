"""Emit import provenance from the container Python before Ray starts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys


def package(name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    result: dict[str, object] = {
        "spec": None if spec is None else {"origin": spec.origin, "locations": list(spec.submodule_search_locations or [])},
        "distribution": None,
    }
    try:
        dist = importlib.metadata.distribution(name)
        result["distribution"] = {
            "version": dist.version,
            "metadata_name": dist.metadata.get("Name"),
            "root": str(dist.locate_file("")),
            "exceptions_py_exists": (dist.locate_file("urllib3/exceptions.py")).is_file() if name == "urllib3" else None,
        }
    except importlib.metadata.PackageNotFoundError:
        result["distribution"] = "missing"
    try:
        module = __import__(name)
        result["import_file"] = getattr(module, "__file__", None)
        result["import_version"] = getattr(module, "__version__", None)
    except Exception as error:
        result["import_error"] = f"{type(error).__name__}: {error}"
    return result


print(
    json.dumps(
        {
            "executable": sys.executable,
            "version": sys.version,
            "sys_path": sys.path,
            "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "UV_CACHE_DIR", "UV_CACHE_DIR_OVERRIDE", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")},
            "requests": package("requests"),
            "urllib3": package("urllib3"),
            "ray": package("ray"),
        },
        sort_keys=True,
        indent=2,
    )
)
