"""Optional vLLM SpecDec profiling annotations.

This file is activated only when its directory is placed on ``PYTHONPATH``.
It installs a small import hook that wraps selected vLLM speculative-decoding
functions with ``torch.profiler.record_function`` ranges. The goal is to make
torch-profiler/nsys traces easier to bucket into Drafting, Verification,
Rejection Sampling, and Other without changing vLLM source files.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import inspect
import os
import sys
from types import ModuleType
from typing import Any, Callable


ENABLED = os.environ.get("SPECDEC_BREAKDOWN_INSTRUMENTATION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


WRAPPED_ATTR = "_specdec_breakdown_wrapped"
WRAPPED_MODULES: set[str] = set()


def should_inspect_module(fullname: str) -> bool:
    if not ENABLED:
        return False
    if not fullname.startswith("vllm."):
        return False
    lowered = fullname.lower()
    return (
        "spec_decode" in lowered
        or "specdec" in lowered
        or "eagle" in lowered
        or "rejection" in lowered
        or (
            os.environ.get("SPECDEC_BREAKDOWN_WRAP_TARGET", "1").lower()
            in {"1", "true", "yes", "on"}
            and lowered.endswith("gpu_model_runner")
        )
    )


def bucket_for(module_name: str, attr_name: str) -> str | None:
    text = f"{module_name}.{attr_name}".lower()
    if "reject" in text or "rejection" in text:
        return "rejection_sampling"
    if "sample" in text and ("spec" in text or "reject" in text):
        return "rejection_sampling"
    if "verify" in text or "verification" in text:
        return "verification"
    if "gpu_model_runner" in module_name.lower() and attr_name == "execute_model":
        return "verification"
    if (
        "draft" in text
        or "drafter" in text
        or "eagle" in text
        or "propose" in text
        or "proposal" in text
    ):
        return "drafting"
    return None


def record_range(bucket: str, label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(fn, WRAPPED_ATTR, False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            import torch

            with torch.profiler.record_function(
                f"specdec_breakdown.{bucket}:{label}"
            ):
                return fn(*args, **kwargs)
        except Exception:
            return fn(*args, **kwargs)

    setattr(wrapped, WRAPPED_ATTR, True)
    return wrapped


def wrap_module(module: ModuleType) -> None:
    module_name = getattr(module, "__name__", "")
    if module_name in WRAPPED_MODULES or not should_inspect_module(module_name):
        return
    WRAPPED_MODULES.add(module_name)

    for attr_name, obj in list(vars(module).items()):
        bucket = bucket_for(module_name, attr_name)
        if bucket and inspect.isfunction(obj):
            try:
                setattr(
                    module,
                    attr_name,
                    record_range(bucket, f"{module_name}.{attr_name}", obj),
                )
            except Exception:
                pass
            continue

        if inspect.isclass(obj) and getattr(obj, "__module__", "") == module_name:
            for method_name, method in list(vars(obj).items()):
                bucket = bucket_for(module_name, method_name) or bucket_for(
                    module_name, attr_name
                )
                if not bucket:
                    continue
                if isinstance(method, staticmethod):
                    raw = method.__func__
                    if inspect.isfunction(raw):
                        try:
                            setattr(
                                obj,
                                method_name,
                                staticmethod(
                                    record_range(
                                        bucket,
                                        f"{module_name}.{attr_name}.{method_name}",
                                        raw,
                                    )
                                ),
                            )
                        except Exception:
                            pass
                elif isinstance(method, classmethod):
                    raw = method.__func__
                    if inspect.isfunction(raw):
                        try:
                            setattr(
                                obj,
                                method_name,
                                classmethod(
                                    record_range(
                                        bucket,
                                        f"{module_name}.{attr_name}.{method_name}",
                                        raw,
                                    )
                                ),
                            )
                        except Exception:
                            pass
                elif inspect.isfunction(method):
                    try:
                        setattr(
                            obj,
                            method_name,
                            record_range(
                                bucket,
                                f"{module_name}.{attr_name}.{method_name}",
                                method,
                            ),
                        )
                    except Exception:
                        pass


class WrapLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)  # type: ignore[attr-defined]
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)  # type: ignore[attr-defined]
        wrap_module(module)


class WrapFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not should_inspect_module(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, WrapLoader):
            return spec
        spec.loader = WrapLoader(spec.loader)
        return spec


if ENABLED:
    sys.meta_path.insert(0, WrapFinder())
