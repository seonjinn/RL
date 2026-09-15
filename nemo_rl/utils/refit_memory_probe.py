"""Experiment-only refit memory instrumentation; disabled by default."""

import functools
import json
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def external_memory() -> dict[str, int]:
    import psutil
    import pynvml

    from nemo_rl.utils.nvml import device_id_to_physical_device_id

    import torch

    host = psutil.virtual_memory()
    result = {
        "host_used_bytes": host.total - host.available,
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            device_id_to_physical_device_id(torch.cuda.current_device())
        )
        result["device_used_bytes"] = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()
    return result


def refit_memory_phase(label: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if os.environ.get("NRL_REFIT_MEMORY_PROBE") != "1":
                return function(*args, **kwargs)
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            before = external_memory()
            peak = dict(before)
            stop = threading.Event()
            sampling_errors: list[str] = []
            device = torch.cuda.current_device()

            def sample() -> None:
                torch.cuda.set_device(device)
                while not stop.wait(0.2):
                    try:
                        for key, value in external_memory().items():
                            peak[key] = max(peak[key], value)
                    except Exception as error:
                        sampling_errors.append(repr(error))
                        return

            thread = threading.Thread(target=sample, daemon=True)
            record: dict[str, Any] = {
                "phase": label,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "rank": getattr(args[0], "rank", "?") if args else os.environ.get("RANK", "?"),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "start_unix_s": start,
                "before": before,
                "allocated_before_bytes": torch.cuda.memory_allocated(),
                "reserved_before_bytes": torch.cuda.memory_reserved(),
            }
            thread.start()
            try:
                result = function(*args, **kwargs)
                torch.cuda.synchronize()
                after = external_memory()
                for key, value in after.items():
                    peak[key] = max(peak[key], value)
                record.update(
                    after=after,
                    allocated_after_bytes=torch.cuda.memory_allocated(),
                    reserved_after_bytes=torch.cuda.memory_reserved(),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    succeeded=True,
                )
                return result
            except BaseException as error:
                record.update(succeeded=False, error=repr(error))
                raise
            finally:
                stop.set()
                thread.join(timeout=2)
                record.update(
                    end_unix_s=time.time(),
                    sampled_peak=peak,
                    sampling_errors=sampling_errors,
                    sampler_stopped=not thread.is_alive(),
                )
                print("REFIT_MEMORY_JSON " + json.dumps(record, default=str), flush=True)

        return wrapped

    return decorate
