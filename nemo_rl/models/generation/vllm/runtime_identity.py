"""Authoritative post-initialization identity for a vLLM model runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _get(value: object | None, name: str) -> object | None:
    if value is None:
        return None
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _qualified_type(value: object | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    cls = type(value)
    return cls.__module__, cls.__qualname__


def _normalized_enum(value: object) -> str:
    candidate = _get(value, "name") or _get(value, "value") or value
    return str(candidate).split(".")[-1].upper()


def _model_id(config: object | None) -> str | None:
    if config is None:
        return None
    for name in ("model", "model_name", "model_name_or_path", "_name_or_path"):
        value = _get(config, name)
        if isinstance(value, str) and value:
            return value
    hf_config = _get(config, "hf_config")
    if hf_config is not None and hf_config is not config:
        return _model_id(hf_config)
    return None


def _unwrap_cudagraph_model(value: object | None) -> object | None:
    seen: set[int] = set()
    while value is not None and id(value) not in seen:
        seen.add(id(value))
        runnable = _get(value, "runnable")
        if runnable is None:
            break
        value = runnable
    return value


def _configured_method(speculative_config: object | None) -> str:
    method = _get(speculative_config, "method")
    return str(method).lower() if method else "none"


def _initialized_method(
    proposer: object | None, speculative_config: object | None
) -> str:
    """Resolve the method from the constructed proposer, never config alone."""
    if proposer is None:
        return "none"
    module, class_name = _qualified_type(proposer)
    identity = f"{module or ''}.{class_name or ''}".lower()
    configured = _configured_method(speculative_config)
    if "dspark" in identity:
        return "dspark"
    if "dflash" in identity:
        return "dflash"
    if "eagle" in identity and configured in {"eagle", "eagle3"}:
        return configured
    proposer_method = _get(proposer, "method")
    if isinstance(proposer_method, str) and proposer_method:
        return proposer_method.lower()
    if configured != "none":
        return configured
    return "unknown"


def _iter_values(value: object | None) -> Iterable[object]:
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return ()


def _descriptor_size(value: object) -> int | None:
    size = _get(value, "num_tokens")
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size
    return None


def _full_dispatch_sizes(model_runner: object) -> tuple[bool, list[int]]:
    dispatcher = _get(model_runner, "cudagraph_dispatcher")
    keys_initialized = _get(dispatcher, "keys_initialized") is True
    keys = _get(dispatcher, "cudagraph_keys")
    sizes: set[int] = set()
    if isinstance(keys, Mapping):
        for mode, descriptors in keys.items():
            if _normalized_enum(mode) != "FULL":
                continue
            for descriptor in _iter_values(descriptors):
                size = _descriptor_size(descriptor)
                if size is not None:
                    sizes.add(size)
    return keys_initialized, sorted(sizes)


def _full_wrapper_sizes(model_runner: object) -> list[int]:
    sizes: set[int] = set()
    owners = (
        _get(model_runner, "model"),
        _get(_get(model_runner, "drafter"), "model"),
        _get(_get(model_runner, "proposer"), "model"),
        _get(_get(model_runner, "speculator"), "model"),
    )
    for owner in owners:
        if _normalized_enum(_get(owner, "runtime_mode")) != "FULL":
            continue
        entries = _get(owner, "concrete_cudagraph_entries")
        if not isinstance(entries, Mapping):
            continue
        for descriptor, entry in entries.items():
            if _get(entry, "cudagraph") is None:
                continue
            size = _descriptor_size(descriptor)
            if size is not None:
                sizes.add(size)
    return sorted(sizes)


def _legacy_full_graph_state(
    model_runner: object,
) -> tuple[bool, list[int], list[int], str | None]:
    owners = (
        model_runner,
        _get(model_runner, "cudagraph_manager"),
        _get(model_runner, "cuda_graph_manager"),
        _get(model_runner, "model"),
    )
    for owner in owners:
        if owner is None or _get(owner, "_graphs_captured") is not True:
            continue
        graphs = _get(owner, "graphs")
        captured = sorted(
            {
                size
                for descriptor in (graphs.keys() if isinstance(graphs, Mapping) else ())
                if (size := _descriptor_size(descriptor)) is not None
            }
        )
        capture_descs = _get(owner, "_capture_descs")
        expected: set[int] = set()
        if isinstance(capture_descs, Mapping):
            for mode, descriptors in capture_descs.items():
                if _normalized_enum(mode) != "FULL":
                    continue
                expected.update(
                    size
                    for descriptor in _iter_values(descriptors)
                    if (size := _descriptor_size(descriptor)) is not None
                )
        if expected and sorted(expected) == captured:
            cls = type(owner)
            return (
                True,
                sorted(expected),
                captured,
                f"{cls.__module__}.{cls.__qualname__}._graphs_captured",
            )
    return False, [], [], None


def _full_graph_state(
    model_runner: object,
) -> tuple[bool, list[int], list[int], str | None]:
    initialized, expected = _full_dispatch_sizes(model_runner)
    captured = _full_wrapper_sizes(model_runner)
    if initialized and expected and expected == captured:
        dispatcher = _get(model_runner, "cudagraph_dispatcher")
        cls = type(dispatcher)
        return (
            True,
            expected,
            captured,
            f"{cls.__module__}.{cls.__qualname__}.keys_initialized+captured_entries",
        )
    legacy_ready, legacy_expected, legacy_captured, legacy_source = (
        _legacy_full_graph_state(model_runner)
    )
    if legacy_ready:
        return legacy_ready, legacy_expected, legacy_captured, legacy_source
    return False, expected, captured, None


def build_vllm_runtime_identity(model_runner: Any) -> dict[str, object]:
    """Inspect the initialized runner without trusting launch-time settings."""
    vllm_config = model_runner.vllm_config
    speculative_config = _get(vllm_config, "speculative_config")
    proposer = next(
        (
            value
            for name in ("drafter", "proposer", "speculator")
            if (value := _get(model_runner, name)) is not None
        ),
        None,
    )
    draft_model = _unwrap_cudagraph_model(_get(proposer, "model"))
    target_model = _unwrap_cudagraph_model(_get(model_runner, "model"))
    target_module, target_class = _qualified_type(target_model)
    proposer_module, proposer_class = _qualified_type(proposer)
    draft_module, draft_class = _qualified_type(draft_model)
    compilation = _get(vllm_config, "compilation_config")
    capture_sizes = _get(compilation, "cudagraph_capture_sizes")
    graph_ready, expected_sizes, captured_sizes, graph_ready_source = _full_graph_state(
        model_runner
    )
    draft_config = _get(speculative_config, "draft_model_config")
    draft_model_id = _model_id(draft_config)
    if draft_model_id is None:
        configured_draft = _get(speculative_config, "model")
        draft_model_id = configured_draft if isinstance(configured_draft, str) else None
    speculative_tokens = _get(speculative_config, "num_speculative_tokens")
    k = (
        speculative_tokens
        if isinstance(speculative_tokens, int)
        and not isinstance(speculative_tokens, bool)
        and proposer is not None
        else 0
    )
    return {
        "schema_version": 1,
        "method": _initialized_method(proposer, speculative_config),
        "configured_method": _configured_method(speculative_config),
        "proposer_module": proposer_module,
        "proposer_class": proposer_class,
        "draft_model_module": draft_module,
        "draft_model_class": draft_class,
        "target_model_module": target_module,
        "target_model_class": target_class,
        "target_model_id": _model_id(_get(vllm_config, "model_config")),
        "draft_model_id": draft_model_id,
        "k": k,
        "cudagraph_mode": (
            _normalized_enum(_get(compilation, "cudagraph_mode"))
            if _get(compilation, "cudagraph_mode") is not None
            else None
        ),
        "cudagraph_capture_sizes": (
            sorted({int(value) for value in capture_sizes})
            if isinstance(capture_sizes, (list, tuple))
            else []
        ),
        "full_cudagraph_expected_sizes": expected_sizes,
        "full_cudagraph_captured_sizes": captured_sizes,
        "full_graph_ready": graph_ready,
        "graph_ready_source": graph_ready_source,
    }
