import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LengthBucket:
    max_tokens: int
    weight: int
    min_tokens: int
    ignore_eos: bool


@dataclass(frozen=True)
class RequestPlan:
    name: str
    max_model_len: int
    buckets: tuple[LengthBucket, ...]
    plan_hash: str


@dataclass(frozen=True)
class ResolvedRequest:
    prompt_id: str
    sample_index: int
    seed: int
    max_tokens: int
    min_tokens: int
    ignore_eos: bool


def _require_json_object(field_name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return value


def _require_json_array(field_name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _require_int(field_name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_bool(field_name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_payload = _require_json_object("request plan", payload)
    buckets_payload = _require_json_array("buckets", normalized_payload["buckets"])
    buckets = [
        {
            "ignore_eos": _require_bool(
                "ignore_eos", bucket_payload.get("ignore_eos", False)
            ),
            "max_tokens": _require_int("max_tokens", bucket_payload["max_tokens"]),
            "min_tokens": _require_int(
                "min_tokens",
                bucket_payload.get("min_tokens", bucket_payload["max_tokens"]),
            ),
            "weight": _require_int("weight", bucket_payload["weight"]),
        }
        for bucket_payload in (
            _require_json_object("bucket", bucket) for bucket in buckets_payload
        )
    ]
    buckets.sort(
        key=lambda bucket: (
            bucket["max_tokens"],
            bucket["min_tokens"],
            bucket["weight"],
            bucket["ignore_eos"],
        )
    )
    return {
        "buckets": buckets,
        "max_model_len": _require_int(
            "max_model_len", normalized_payload["max_model_len"]
        ),
        "name": normalized_payload["name"],
    }


def _plan_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_request_plan(path: Path) -> RequestPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = _canonical_payload(payload)
    buckets = tuple(
        LengthBucket(
            max_tokens=bucket["max_tokens"],
            weight=bucket["weight"],
            min_tokens=bucket["min_tokens"],
            ignore_eos=bucket["ignore_eos"],
        )
        for bucket in normalized["buckets"]
    )
    if not buckets:
        raise ValueError("request plan must define at least one bucket")
    if any(bucket.weight <= 0 for bucket in buckets):
        raise ValueError("request plan bucket weights must be positive")
    if any(bucket.min_tokens <= 0 for bucket in buckets):
        raise ValueError("request plan min_tokens must be positive")
    if any(bucket.min_tokens > bucket.max_tokens for bucket in buckets):
        raise ValueError("request plan min_tokens cannot exceed max_tokens")
    return RequestPlan(
        name=normalized["name"],
        max_model_len=normalized["max_model_len"],
        buckets=buckets,
        plan_hash=_plan_hash(payload),
    )


def validate_context_window(
    prompt_tokens: int,
    output_cap: int,
    max_model_len: int,
) -> None:
    if prompt_tokens + output_cap > max_model_len:
        raise ValueError(
            f"context overflow: prompt={prompt_tokens} output={output_cap} "
            f"max={max_model_len}"
        )


def _weighted_prompt_counts(
    total_prompts: int, buckets: tuple[LengthBucket, ...]
) -> list[int]:
    total_weight = sum(bucket.weight for bucket in buckets)
    raw_counts = [total_prompts * bucket.weight / total_weight for bucket in buckets]
    counts = [int(count) for count in raw_counts]
    remainder = total_prompts - sum(counts)
    ranked_remainders = sorted(
        (
            (raw_counts[index] - counts[index], index)
            for index in range(len(buckets))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, index in ranked_remainders[:remainder]:
        counts[index] += 1
    return counts


def _normalize_prompt_lengths(
    prompt_ids: list[str],
    prompt_token_lengths: dict[str, int] | list[int] | None,
) -> dict[str, int]:
    if prompt_token_lengths is None:
        return {}
    if isinstance(prompt_token_lengths, dict):
        return {prompt_id: int(prompt_token_lengths[prompt_id]) for prompt_id in prompt_ids}
    if len(prompt_token_lengths) != len(prompt_ids):
        raise ValueError("prompt_token_lengths must match prompt_ids")
    return {
        prompt_id: int(prompt_token_lengths[index])
        for index, prompt_id in enumerate(prompt_ids)
    }


def resolve_request_plan(
    plan: RequestPlan,
    *,
    prompt_ids: list[str],
    samples_per_prompt: int,
    seed_start: int,
    prompt_token_lengths: dict[str, int] | list[int] | None = None,
    rollout_batch_index: int = 0,
    max_model_len: int | None = None,
) -> list[ResolvedRequest]:
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    if not prompt_ids:
        return []
    total_requests = len(prompt_ids) * samples_per_prompt
    resolved_max_model_len = max_model_len if max_model_len is not None else plan.max_model_len
    lengths_by_prompt = _normalize_prompt_lengths(prompt_ids, prompt_token_lengths)
    prompt_counts = _weighted_prompt_counts(len(prompt_ids), plan.buckets)
    requests: list[ResolvedRequest] = []
    seed = seed_start + (rollout_batch_index * total_requests)
    prompt_index = 0
    for bucket, prompt_count in zip(plan.buckets, prompt_counts, strict=True):
        for _ in range(prompt_count):
            prompt_id = prompt_ids[prompt_index]
            prompt_index += 1
            prompt_tokens = lengths_by_prompt.get(prompt_id)
            if prompt_tokens is not None:
                validate_context_window(
                    prompt_tokens=prompt_tokens,
                    output_cap=bucket.max_tokens,
                    max_model_len=resolved_max_model_len,
                )
            for sample_index in range(samples_per_prompt):
                requests.append(
                    ResolvedRequest(
                        prompt_id=prompt_id,
                        sample_index=sample_index,
                        seed=seed,
                        max_tokens=bucket.max_tokens,
                        min_tokens=bucket.min_tokens,
                        ignore_eos=bucket.ignore_eos,
                    )
                )
                seed += 1
    return requests


def summarize_barrier_tail(barrier_times_s: list[float]) -> dict[str, float]:
    if not barrier_times_s:
        return {
            "count": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "max_s": 0.0,
            "tail_gap_s": 0.0,
        }
    ordered = sorted(float(value) for value in barrier_times_s)
    return {
        "count": float(len(ordered)),
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "max_s": ordered[-1],
        "tail_gap_s": ordered[-1] - statistics.median(ordered),
    }
