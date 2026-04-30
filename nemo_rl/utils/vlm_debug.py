# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from typing import Any, Iterable

import torch

_DEBUG_ENV = "NRL_NEMOTRON_VL_DEBUG"
_DEBUG_DIR_ENV = "NRL_NEMOTRON_VL_DEBUG_DIR"
_DEBUG_FILE_ENV = "NRL_NEMOTRON_VL_DEBUG_FILE"
_DUMP_TENSORS_ENV = "NRL_NEMOTRON_VL_DEBUG_DUMP_TENSORS"
_RUN_LABEL_ENV = "NRL_NEMOTRON_VL_RUN_LABEL"
_SAMPLE_ID_ENV = "NRL_NEMOTRON_VL_FIXTURE_SAMPLE_ID"

_FILE_LOCK = threading.Lock()
_WRITE_ERROR_REPORTED = False
_SANITIZE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, "0") == "1"


def dump_tensors_enabled() -> bool:
    return os.environ.get(_DUMP_TENSORS_ENV, "0") == "1"


def get_run_label(default: str | None = None) -> str | None:
    return os.environ.get(_RUN_LABEL_ENV, default)


def get_sample_id(default: str | None = None) -> str | None:
    return os.environ.get(_SAMPLE_ID_ENV, default)


def stable_hash(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif torch.is_tensor(value):
        raw = _tensor_hash_bytes(value)
    else:
        raw = json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def full_tensor_hash(tensor: torch.Tensor | None) -> str | None:
    """Hash every element of a tensor after dtype-normalization.

    Unlike ``stable_hash(tensor)``, this does not sample. It is intended only
    for small debug tensors where full position-aligned equality matters.
    """
    if tensor is None:
        return None
    return hashlib.sha256(_full_tensor_hash_bytes(tensor)).hexdigest()[:16]


def tensor_summary(name: str, tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None

    detached = tensor.detach()
    cpu_tensor = detached.cpu()
    flat = cpu_tensor.reshape(-1)
    sample = _sample_tensor(flat)
    stats_tensor = _stats_tensor(sample)

    summary: dict[str, Any] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "sample_numel": int(sample.numel()),
        "sample_hash": stable_hash(sample),
        "first_values": _tensor_preview_values(sample),
    }

    if sample.numel() == 0:
        summary.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "sum": None,
                "l2": None,
            }
        )
    else:
        summary.update(
            {
                "min": float(stats_tensor.min().item()),
                "max": float(stats_tensor.max().item()),
                "mean": float(stats_tensor.mean().item()),
                "std": float(stats_tensor.std(unbiased=False).item()),
                "sum": float(stats_tensor.sum().item()),
                "l2": float(torch.linalg.vector_norm(stats_tensor).item()),
            }
        )

    if dump_tensors_enabled():
        summary["tensor_values"] = _tensor_values_for_json(cpu_tensor)

    return summary


def tensor_full_summary(name: str, tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """Tensor summary whose hash covers the entire tensor, not a sample."""
    if tensor is None:
        return None

    detached = tensor.detach()
    cpu_tensor = detached.cpu()
    stats_tensor = _stats_tensor(cpu_tensor.reshape(-1))
    summary: dict[str, Any] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "full_hash": full_tensor_hash(detached),
        "first_values": _tensor_preview_values(cpu_tensor.reshape(-1)),
    }
    if stats_tensor.numel() == 0:
        summary.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "sum": None,
                "l2": None,
            }
        )
    else:
        summary.update(
            {
                "min": float(stats_tensor.min().item()),
                "max": float(stats_tensor.max().item()),
                "mean": float(stats_tensor.mean().item()),
                "std": float(stats_tensor.std(unbiased=False).item()),
                "sum": float(stats_tensor.sum().item()),
                "l2": float(torch.linalg.vector_norm(stats_tensor).item()),
            }
        )
    return summary


_DEFAULT_LM_PARAM_PATTERNS: tuple[str, ...] = (
    # token / output embeddings (cover both Megatron and vLLM/HF naming)
    r"(^|\.)embed_tokens\.weight$",
    r"(^|\.)word_embeddings\.weight$",
    r"(^|\.)lm_head\.weight$",
    r"(^|\.)output_layer\.weight$",
    # broad sweep across language-model body for layers 0, 5, 10, 15. Matches
    # Megatron-style (linear_qkv, mlp.experts.linear_fc1/2, mlp.router) and
    # vLLM/HF-style (qkv_proj, q_proj/k_proj/v_proj, mlp.experts.w13_weight,
    # mlp.experts.w2_weight, mlp.gate, mlp.gate_up_proj, mlp.down_proj).
    r"(^|\.)language_model\..*\.layers\.0\..*\.weight$",
    r"(^|\.)language_model\..*\.layers\.5\..*\.weight$",
    r"(^|\.)language_model\..*\.layers\.10\..*\.weight$",
    r"(^|\.)language_model\..*\.layers\.15\..*\.weight$",
)


def lm_param_stats(param: torch.Tensor) -> dict[str, Any]:
    """Layout-invariant summary stats for a parameter tensor.

    Computed in float32 on CPU. Sums and L2 are layout-invariant under any
    permutation of the elements, so two parameter tensors that hold the same
    set of values (modulo permutation/transposition/sharding) will agree on
    these stats. This makes the stats useful when Megatron and vLLM may store
    the same parameter in slightly different layouts.
    """
    detached = param.detach().to("cpu", torch.float32)
    flat = detached.reshape(-1)
    if flat.numel() == 0:
        return {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "numel": 0,
            "mean": None,
            "std": None,
            "sum": None,
            "abs_sum": None,
            "sq_sum": None,
            "min": None,
            "max": None,
            "l2": None,
        }
    return {
        "shape": list(param.shape),
        "dtype": str(param.dtype),
        "numel": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "sq_sum": float((flat * flat).sum()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(torch.linalg.vector_norm(flat)),
    }


def dump_named_parameter_stats(
    model: torch.nn.Module,
    *,
    stage: str,
    name_patterns: Iterable[str] | None = None,
    side: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Walk model.named_parameters(), match by regex, emit stats via write_stage.

    If name_patterns is None, the default set targets the canonical small
    sample of language-model parameters used by the Sprint 3 weight comparison.
    All matched parameter records are written under a single stage record, so
    one call per side produces one JSONL line per side that can be diffed.
    """
    if not debug_enabled():
        return

    patterns_seq = tuple(name_patterns) if name_patterns is not None else _DEFAULT_LM_PARAM_PATTERNS
    compiled = [re.compile(p) for p in patterns_seq]

    records: list[dict[str, Any]] = []
    layer_indices: dict[str, int] = {}
    try:
        for name, param in model.named_parameters():
            if not any(p.search(name) for p in compiled):
                continue
            stats = lm_param_stats(param)
            stats["name"] = name
            records.append(stats)
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match:
                idx = int(match.group(1))
                layer_indices[name] = idx
    except Exception as exc:
        records.append({"name": "<iteration_error>", "error": str(exc)})

    name_listing: list[str] = []
    if len(records) < 6:
        try:
            for name, _param in model.named_parameters():
                if "language_model" in name:
                    name_listing.append(name)
                if len(name_listing) >= 200:
                    break
        except Exception as exc:
            name_listing.append(f"<listing_error: {exc}>")

    payload: dict[str, Any] = {
        "side": side,
        "patterns": list(patterns_seq),
        "records": records,
        "max_layer_index_seen": max(layer_indices.values()) if layer_indices else None,
        "name_listing_first_200_language_model_params": name_listing,
    }
    if extra is not None:
        for key, value in extra.items():
            payload.setdefault(key, value)

    write_stage(stage, payload)


def attach_moe_router_probe(
    *,
    module: torch.nn.Module,
    side: str,
    layer_idx: int,
    top_k: int = 2,
    max_calls: int = 4,
    max_tokens_emit: int = 96,
) -> Any:
    """Register a forward hook on a router/gate Linear module that records
    per-token routing decisions.

    The hook captures input hash, router logits hash, top-K expert ids and
    softmax weights for the first ``max_tokens_emit`` tokens of the first
    ``max_calls`` forward passes. Records are written via ``write_stage`` under
    stage ``moe_router_decisions``. Returns the hook handle for the caller to
    keep alive.
    """
    state = {"call_count": 0}

    def hook(_mod, args, output):  # type: ignore[no-untyped-def]
        if not debug_enabled():
            return
        if state["call_count"] >= max_calls:
            return
        state["call_count"] += 1
        try:
            x = args[0] if args else None
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            if not torch.is_tensor(logits):
                return

            logits_flat = logits.reshape(-1, logits.shape[-1])
            num_tokens_total = int(logits_flat.shape[0])
            num_emit = min(num_tokens_total, max_tokens_emit)
            logits_slice = logits_flat[:num_emit].detach()

            x_flat = None
            if torch.is_tensor(x):
                x_flat = x.reshape(-1, x.shape[-1])[:num_emit].detach()

            probs = torch.softmax(logits_slice.float(), dim=-1)
            effective_top_k = min(top_k, int(logits_slice.shape[-1]))
            top_w, top_idx = probs.topk(effective_top_k, dim=-1)

            payload: dict[str, Any] = {
                "side": side,
                "layer_idx": layer_idx,
                "call_count": state["call_count"],
                "num_tokens_total": num_tokens_total,
                "num_tokens_emit": num_emit,
                "top_k": effective_top_k,
                "input_hash": stable_hash(x_flat) if x_flat is not None else None,
                "logits_hash": stable_hash(logits_slice),
                "top_indices": top_idx.tolist(),
                "top_weights": [
                    [round(float(v), 6) for v in row] for row in top_w.tolist()
                ],
                "input_summary": (
                    tensor_summary("router_input", x_flat)
                    if x_flat is not None
                    else None
                ),
                "logits_summary": tensor_summary("router_logits", logits_slice),
            }
            write_stage("moe_router_decisions", payload)
        except Exception as exc:
            print(
                f"[VLM_DEBUG_WARN] router hook layer={layer_idx} side={side} failed: {exc}",
                flush=True,
            )

    return module.register_forward_hook(hook)


def attach_activation_probe(
    *,
    module: torch.nn.Module,
    side: str,
    probe_name: str,
    module_name: str | None = None,
    max_calls: int = 4,
    max_tokens_emit: int | None = None,
) -> Any:
    """Register a forward hook that records input/output activation hashes.

    The first tensor input and first tensor output are flattened as
    ``[tokens, hidden]`` when possible. By default the full flattened tensor is
    hashed and summarized; pass ``max_tokens_emit`` only when a truncated view is
    desired.
    """
    state = {"call_count": 0}

    def _first_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                found = _first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, Mapping):
            for preferred_key in ("hidden_states", "input_tensor", "inputs_embeds"):
                if preferred_key in value:
                    found = _first_tensor(value[preferred_key])
                    if found is not None:
                        return found
            for item in value.values():
                found = _first_tensor(item)
                if found is not None:
                    return found
        return None

    def _token_slice(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        detached = tensor.detach()
        if detached.ndim >= 2:
            flat = detached.reshape(-1, detached.shape[-1])
        else:
            flat = detached.reshape(-1, 1)
        total = int(flat.shape[0])
        if max_tokens_emit is None:
            return flat, total
        return flat[: min(total, max_tokens_emit)], total

    def _window_summaries(tensor: torch.Tensor | None) -> dict[str, Any]:
        if tensor is None:
            return {}
        # Fixed prompt windows for the one-sample Nemotron VL smoke. vLLM prefill
        # has 385 prompt tokens; Megatron rescore has prompt+response, so the
        # shared comparison window is Megatron[:385] versus vLLM[:385].
        windows = {
            "prompt_0_385": (0, 385),
            "first_0_16": (0, 16),
            "image_region_9_283": (9, 283),
            "post_image_283_385": (283, 385),
            "last_prompt_353_385": (353, 385),
        }
        result: dict[str, Any] = {}
        for name, (start, end) in windows.items():
            if tensor.shape[0] < end:
                result[name] = {
                    "available": False,
                    "tokens_total": int(tensor.shape[0]),
                    "start": start,
                    "end": end,
                }
                continue
            result[name] = {
                "available": True,
                "start": start,
                "end": end,
                "summary": tensor_full_summary(name, tensor[start:end]),
            }
        return result

    def _handle_activation(_mod, args, kwargs, output):  # type: ignore[no-untyped-def]
        if not debug_enabled():
            return
        if state["call_count"] >= max_calls:
            return
        state["call_count"] += 1
        try:
            inp = _first_tensor(kwargs) if kwargs else None
            if inp is None:
                inp = _first_tensor(args)
            out = _first_tensor(output)
            inp_slice = None
            out_slice = None
            inp_total = None
            out_total = None
            if inp is not None:
                inp_slice, inp_total = _token_slice(inp)
            if out is not None:
                out_slice, out_total = _token_slice(out)

            extra_payload: dict[str, Any] = {}
            # Megatron's TELayerNormColumnParallelLinear fuses RMSNorm+Linear, so
            # regular hooks only expose pre-norm input and post-linear output. If
            # the fused module exposes its LN/RMSNorm weight, compute a best-effort
            # post-norm/pre-linear preview for comparison against vLLM's explicit
            # pre-linear tensor.
            ln_weight = None
            ln_bias = None
            for pname, pval in getattr(_mod, "named_parameters", lambda recurse=False: [])(
                recurse=False
            ):
                if pname in {"layer_norm_weight", "layernorm_weight"}:
                    ln_weight = pval
                elif pname in {"layer_norm_bias", "layernorm_bias"}:
                    ln_bias = pval
            if inp_slice is not None and ln_weight is not None:
                eps = float(
                    getattr(_mod, "eps", getattr(_mod, "layer_norm_epsilon", 1e-5))
                )
                x = inp_slice.detach().to(torch.float32)
                w = ln_weight.detach().to(device=x.device, dtype=torch.float32)
                # Transformer Engine uses zero-centered gamma by adding 1 to the
                # stored scale when enabled. Megatron config carries this flag.
                zero_centered = bool(
                    getattr(
                        getattr(_mod, "config", None),
                        "layernorm_zero_centered_gamma",
                        False,
                    )
                )
                if zero_centered:
                    w = w + 1.0
                normed = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
                normed = normed * w
                if ln_bias is not None:
                    normed = normed + ln_bias.detach().to(device=x.device, dtype=torch.float32)
                extra_payload["manual_rmsnorm_hash"] = full_tensor_hash(normed)
                extra_payload["manual_rmsnorm_summary"] = tensor_full_summary(
                    "manual_rmsnorm_output", normed
                )
                extra_payload["manual_rmsnorm_windows"] = _window_summaries(normed)
                extra_payload["manual_rmsnorm_eps"] = eps
                extra_payload["manual_rmsnorm_zero_centered_gamma"] = zero_centered

            write_stage(
                "activation_probe",
                {
                    "side": side,
                    "probe_name": probe_name,
                    "module_name": module_name,
                    "module_type": type(module).__name__,
                    "call_count": state["call_count"],
                    "input_tokens_total": inp_total,
                    "output_tokens_total": out_total,
                "input_hash": full_tensor_hash(inp_slice) if inp_slice is not None else None,
                "output_hash": full_tensor_hash(out_slice) if out_slice is not None else None,
                    "input_summary": (
                        tensor_full_summary("input", inp_slice)
                        if inp_slice is not None
                        else None
                    ),
                    "input_windows": _window_summaries(inp_slice),
                    "output_summary": (
                        tensor_full_summary("output", out_slice)
                        if out_slice is not None
                        else None
                    ),
                    "output_windows": _window_summaries(out_slice),
                    **extra_payload,
                },
            )
        except Exception as exc:
            print(
                f"[VLM_DEBUG_WARN] activation hook {probe_name} side={side} failed: {exc}",
                flush=True,
            )

    def hook_with_kwargs(_mod, args, kwargs, output):  # type: ignore[no-untyped-def]
        return _handle_activation(_mod, args, kwargs, output)

    def hook_without_kwargs(_mod, args, output):  # type: ignore[no-untyped-def]
        return _handle_activation(_mod, args, {}, output)

    try:
        return module.register_forward_hook(hook_with_kwargs, with_kwargs=True)
    except TypeError:
        return module.register_forward_hook(hook_without_kwargs)


def write_stage(stage: str, payload: Mapping[str, Any] | None = None) -> None:
    if not debug_enabled():
        return

    debug_dir = os.path.abspath(
        os.environ.get(_DEBUG_DIR_ENV, "/tmp/nrl_nemotron_vl_debug")
    )
    record: dict[str, Any] = {
        "stage": stage,
        "timestamp_ms": int(time.time() * 1000),
        "pid": os.getpid(),
        "run_label": get_run_label("unknown"),
        "repo": get_run_label("unknown"),
        "sample_id": get_sample_id(),
    }

    if payload is not None:
        record.update(_json_safe(payload))

    try:
        os.makedirs(debug_dir, exist_ok=True)
        output_path = _debug_output_path(debug_dir)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _FILE_LOCK:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
    except Exception as exc:
        _report_write_error_once(exc)


def _debug_output_path(debug_dir: str) -> str:
    explicit = os.environ.get(_DEBUG_FILE_ENV)
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join(debug_dir, explicit)

    run_label = get_run_label("run") or "run"
    safe_run_label = _SANITIZE_FILENAME_RE.sub("_", run_label).strip("._")
    if not safe_run_label:
        safe_run_label = "run"
    return os.path.join(debug_dir, f"{safe_run_label}-pid{os.getpid()}.jsonl")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, type):
        return value.__name__

    if torch.is_tensor(value):
        return tensor_summary("tensor", value)

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


def _sample_tensor(tensor: torch.Tensor, max_values: int = 256) -> torch.Tensor:
    if tensor.numel() <= max_values:
        return tensor.contiguous()

    indices = torch.linspace(
        0,
        tensor.numel() - 1,
        steps=max_values,
        device=tensor.device,
        dtype=torch.float64,
    ).round().to(torch.long)
    return tensor.index_select(0, indices).contiguous()


def _stats_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32)
    if torch.is_floating_point(tensor):
        return tensor.to(torch.float32)
    return tensor.to(torch.float32)


def _tensor_preview_values(tensor: torch.Tensor, limit: int = 8) -> list[Any]:
    preview = tensor[:limit]
    if preview.dtype == torch.bool:
        return [bool(item) for item in preview.tolist()]
    if torch.is_floating_point(preview):
        return [float(item) for item in preview.to(torch.float32).tolist()]
    return [int(item) for item in preview.to(torch.int64).tolist()]


def _tensor_values_for_json(tensor: torch.Tensor) -> Any:
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype == torch.bool:
        return cpu_tensor.tolist()
    if torch.is_floating_point(cpu_tensor):
        return cpu_tensor.to(torch.float32).tolist()
    return cpu_tensor.to(torch.int64).tolist()


def _tensor_hash_bytes(tensor: torch.Tensor) -> bytes:
    sample = _sample_tensor(tensor.detach().cpu().reshape(-1))
    if sample.dtype == torch.bool:
        sample = sample.to(torch.uint8)
    elif torch.is_floating_point(sample):
        sample = sample.to(torch.float32)
    else:
        sample = sample.to(torch.int64)
    return sample.contiguous().numpy().tobytes()


def _full_tensor_hash_bytes(tensor: torch.Tensor) -> bytes:
    full = tensor.detach().cpu().reshape(-1)
    if full.dtype == torch.bool:
        full = full.to(torch.uint8)
    elif torch.is_floating_point(full):
        full = full.to(torch.float32)
    else:
        full = full.to(torch.int64)
    return full.contiguous().numpy().tobytes()


def _report_write_error_once(exc: Exception) -> None:
    global _WRITE_ERROR_REPORTED
    if _WRITE_ERROR_REPORTED:
        return
    _WRITE_ERROR_REPORTED = True
    print(f"[NRL_NEMOTRON_VL_DEBUG] failed to write debug record: {exc}", flush=True)
