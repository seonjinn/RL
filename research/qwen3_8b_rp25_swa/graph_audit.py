"""Opt-in, CPU-only counters around vLLM MRv2 capture/dispatch/replay.

No kernel, descriptor or model output is changed. Dispatch counts are not
GPU-time-weighted coverage. FULL replay counts increment only after success.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import wraps
import inspect
import json
import os
from typing import Any
from weakref import WeakKeyDictionary


_states: WeakKeyDictionary[Any, dict[str, Any]] = WeakKeyDictionary()
_installed: set[type] = set()


def snapshot(manager: Any) -> dict[str, Any]:
    state = _states.setdefault(
        manager,
        {
            "dispatch": Counter(),
            "full_replays": 0,
            "uniform_decode_fallbacks": 0,
            "max_input_tokens": 0,
            "max_padded_tokens": 0,
        },
    )
    return {
        "manager": type(manager).__name__,
        "manager_id": id(manager),
        "pid": os.getpid(),
        "mode": manager.cudagraph_mode.name,
        "decode_query_len": manager.decode_query_len,
        "max_num_reqs": manager.max_num_reqs,
        "full_graphs": len(manager.graphs),
        **state,
        "dispatch": dict(state["dispatch"]),
    }


def emit(event: dict[str, Any]) -> None:
    print("Q8_CG_AUDIT " + json.dumps(event, sort_keys=True), flush=True)


def instrument(
    manager_class: type, sink: Callable[[dict[str, Any]], None] = emit
) -> None:
    if manager_class in _installed:
        return
    original_capture = manager_class.capture
    original_dispatch = manager_class.dispatch
    original_replay = manager_class.run_fullgraph

    @wraps(original_capture)
    def capture(manager: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_capture(manager, *args, **kwargs)
        sink({"event": "capture", **snapshot(manager)})
        if not manager.graphs:
            raise RuntimeError(
                f"Q8 graph gate: no FULL graphs for {type(manager).__name__}"
            )
        return result

    @wraps(original_dispatch)
    def dispatch(
        manager: Any,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
    ) -> Any:
        desc = original_dispatch(
            manager, num_reqs, num_tokens, uniform_token_count, num_active_loras
        )
        if manager._graphs_captured:
            snapshot(manager)
            state = _states[manager]
            state["dispatch"][desc.cg_mode.name] += 1
            state["max_input_tokens"] = max(state["max_input_tokens"], num_tokens)
            state["max_padded_tokens"] = max(
                state["max_padded_tokens"], desc.num_tokens
            )
            if (
                uniform_token_count == manager.decode_query_len
                and desc.cg_mode.name != "FULL"
            ):
                state["uniform_decode_fallbacks"] += 1
            if sum(state["dispatch"].values()) % 2048 == 0:
                sink({"event": "dispatch", **snapshot(manager)})
        return desc

    @wraps(original_replay)
    def replay(manager: Any, desc: Any) -> Any:
        result = original_replay(manager, desc)
        snapshot(manager)
        state = _states[manager]
        state["full_replays"] += 1
        if state["full_replays"] == 1 or state["full_replays"] % 2048 == 0:
            sink({"event": "replay", **snapshot(manager)})
        return result

    manager_class.capture = capture
    manager_class.dispatch = dispatch
    manager_class.run_fullgraph = replay
    _installed.add(manager_class)


def install() -> None:
    import vllm
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if vllm.__version__ != "0.25.1":
        raise RuntimeError("Q8 graph audit is validated only for vLLM 0.25.1")
    names = tuple(inspect.signature(CudaGraphManager.dispatch).parameters)
    if names != (
        "self",
        "num_reqs",
        "num_tokens",
        "uniform_token_count",
        "num_active_loras",
    ):
        raise RuntimeError(f"Q8 graph audit: unsupported dispatch signature {names}")
    instrument(CudaGraphManager)
    emit({"event": "installed", "vllm": vllm.__version__, "pid": os.getpid()})
