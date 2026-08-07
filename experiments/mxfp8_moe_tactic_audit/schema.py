"""Typed schemas for MXFP8 MoE routing traces and tactic replay results."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast


ROUTING_SIGNATURE_FIELDS = frozenset(
    {
        "schema_version",
        "model_revision",
        "layer_family",
        "num_tokens",
        "global_num_experts",
        "local_num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
        "expert_counts",
        "sampled_gpu_time_us",
        "tp_size",
        "ep_size",
        "dp_size",
        "cuda_graph_state",
        "weight_layout",
        "quantization",
        "runtime_fingerprint",
    }
)
TACTIC_PAIR_FIELDS = frozenset({"gemm1", "gemm2"})
TACTIC_MEASUREMENT_FIELDS = frozenset(
    {
        "signature_key",
        "tactic",
        "median_us",
        "p95_us",
        "cv",
        "warmups",
        "repetitions",
        "finite",
        "deterministic",
        "max_abs_error",
        "cosine_similarity",
        "failure",
    }
)
REPLAY_PROFILE_FIELDS = frozenset(
    {
        "signature",
        "signature_key",
        "aggregate_gpu_time_us",
        "call_count",
        "normalized_weight",
        "skew_class",
    }
)

def _require_exact_fields(row: Mapping[str, object], expected: frozenset[str]) -> None:
    """Reject JSON objects whose fields differ from the schema."""
    actual = frozenset(row)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"invalid fields: missing={missing}, unexpected={unexpected}")


def _require_int(value: object, field_name: str) -> int:
    """Return a JSON integer while rejecting booleans and other types."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_float(value: object, field_name: str) -> float:
    """Return a JSON number while rejecting booleans and other types."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _require_str(value: object, field_name: str) -> str:
    """Return a JSON string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    """Return a JSON boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    """Return a positive JSON integer."""
    integer = _require_int(value, field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _require_finite(value: float, field_name: str) -> float:
    """Return a finite numeric value."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Return a string-keyed JSON object."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    """Return a JSON array while excluding strings and mappings."""
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return value


@dataclass(frozen=True)
class RoutingSignature:
    """One observed MoE routing shape and its measured GPU execution time."""

    schema_version: int
    model_revision: str
    layer_family: str
    num_tokens: int
    global_num_experts: int
    local_num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    expert_counts: tuple[int, ...]
    sampled_gpu_time_us: float
    tp_size: int
    ep_size: int
    dp_size: int
    cuda_graph_state: str
    weight_layout: str
    quantization: str
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        """Validate the structural and observation invariants."""
        object.__setattr__(self, "expert_counts", tuple(self.expert_counts))
        dimensions = {
            "num_tokens": self.num_tokens,
            "global_num_experts": self.global_num_experts,
            "local_num_experts": self.local_num_experts,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "dp_size": self.dp_size,
        }
        for field_name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if len(self.expert_counts) != self.global_num_experts:
            raise ValueError("expert_counts length must equal global_num_experts")
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in self.expert_counts):
            raise ValueError("expert_counts must contain nonnegative integers")
        if sum(self.expert_counts) != self.num_tokens * self.top_k:
            raise ValueError("sum(expert_counts) must equal num_tokens * top_k")
        if not math.isfinite(self.sampled_gpu_time_us) or self.sampled_gpu_time_us <= 0:
            raise ValueError("sampled_gpu_time_us must be finite and positive")
        if self.quantization != "MXFP8":
            raise ValueError("quantization must be MXFP8")

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> RoutingSignature:
        """Parse and validate one vLLM JSONL routing trace row."""
        _require_exact_fields(row, ROUTING_SIGNATURE_FIELDS)
        counts = tuple(
            _require_int(value, "expert_counts")
            for value in _require_sequence(row["expert_counts"], "expert_counts")
        )
        return cls(
            schema_version=_require_int(row["schema_version"], "schema_version"),
            model_revision=_require_str(row["model_revision"], "model_revision"),
            layer_family=_require_str(row["layer_family"], "layer_family"),
            num_tokens=_require_positive_int(row["num_tokens"], "num_tokens"),
            global_num_experts=_require_positive_int(
                row["global_num_experts"], "global_num_experts"
            ),
            local_num_experts=_require_positive_int(row["local_num_experts"], "local_num_experts"),
            top_k=_require_positive_int(row["top_k"], "top_k"),
            hidden_size=_require_positive_int(row["hidden_size"], "hidden_size"),
            intermediate_size=_require_positive_int(row["intermediate_size"], "intermediate_size"),
            expert_counts=counts,
            sampled_gpu_time_us=_require_float(row["sampled_gpu_time_us"], "sampled_gpu_time_us"),
            tp_size=_require_positive_int(row["tp_size"], "tp_size"),
            ep_size=_require_positive_int(row["ep_size"], "ep_size"),
            dp_size=_require_positive_int(row["dp_size"], "dp_size"),
            cuda_graph_state=_require_str(row["cuda_graph_state"], "cuda_graph_state"),
            weight_layout=_require_str(row["weight_layout"], "weight_layout"),
            quantization=_require_str(row["quantization"], "quantization"),
            runtime_fingerprint=_require_str(row["runtime_fingerprint"], "runtime_fingerprint"),
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the trace row with JSON-compatible value types."""
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "layer_family": self.layer_family,
            "num_tokens": self.num_tokens,
            "global_num_experts": self.global_num_experts,
            "local_num_experts": self.local_num_experts,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "expert_counts": list(self.expert_counts),
            "sampled_gpu_time_us": self.sampled_gpu_time_us,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "dp_size": self.dp_size,
            "cuda_graph_state": self.cuda_graph_state,
            "weight_layout": self.weight_layout,
            "quantization": self.quantization,
            "runtime_fingerprint": self.runtime_fingerprint,
        }

    def signature_key(self) -> str:
        """Return the SHA256 key for the structural routing signature."""
        structural_row = self.to_json()
        del structural_row["sampled_gpu_time_us"]
        canonical_json = json.dumps(
            structural_row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return sha256(canonical_json.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class TacticPair:
    """The FC1 and FC2 tactics used by one replay measurement."""

    gemm1: int
    gemm2: int

    def __post_init__(self) -> None:
        """Validate tactic identifiers."""
        for field_name, value in (("gemm1", self.gemm1), ("gemm2", self.gemm2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> TacticPair:
        """Parse a tactic-pair JSON object."""
        _require_exact_fields(row, TACTIC_PAIR_FIELDS)
        return cls(
            gemm1=_require_int(row["gemm1"], "gemm1"),
            gemm2=_require_int(row["gemm2"], "gemm2"),
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the tactic pair."""
        return {"gemm1": self.gemm1, "gemm2": self.gemm2}


@dataclass(frozen=True)
class TacticMeasurement:
    """One tactic-pair timing and numerical qualification measurement."""

    signature_key: str
    tactic: TacticPair
    median_us: float
    p95_us: float
    cv: float
    warmups: int
    repetitions: int
    finite: bool
    deterministic: bool
    max_abs_error: float
    cosine_similarity: float
    failure: str | None

    def __post_init__(self) -> None:
        """Validate measurement values."""
        if not isinstance(self.signature_key, str) or not self.signature_key:
            raise ValueError("signature_key must be a nonempty string")
        if not isinstance(self.tactic, TacticPair):
            raise ValueError("tactic must be a TacticPair")
        for field_name, value in (
            ("median_us", self.median_us),
            ("p95_us", self.p95_us),
            ("cv", self.cv),
            ("max_abs_error", self.max_abs_error),
            ("cosine_similarity", self.cosine_similarity),
        ):
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        for field_name, value in (("warmups", self.warmups), ("repetitions", self.repetitions)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not isinstance(self.finite, bool) or not isinstance(self.deterministic, bool):
            raise ValueError("finite and deterministic must be booleans")
        if self.failure is not None and not isinstance(self.failure, str):
            raise ValueError("failure must be a string or null")

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> TacticMeasurement:
        """Parse a tactic measurement JSON object."""
        _require_exact_fields(row, TACTIC_MEASUREMENT_FIELDS)
        failure = row["failure"]
        if failure is not None and not isinstance(failure, str):
            raise ValueError("failure must be a string or null")
        return cls(
            signature_key=_require_str(row["signature_key"], "signature_key"),
            tactic=TacticPair.from_json(_require_mapping(row["tactic"], "tactic")),
            median_us=_require_finite(_require_float(row["median_us"], "median_us"), "median_us"),
            p95_us=_require_finite(_require_float(row["p95_us"], "p95_us"), "p95_us"),
            cv=_require_finite(_require_float(row["cv"], "cv"), "cv"),
            warmups=_require_positive_int(row["warmups"], "warmups"),
            repetitions=_require_positive_int(row["repetitions"], "repetitions"),
            finite=_require_bool(row["finite"], "finite"),
            deterministic=_require_bool(row["deterministic"], "deterministic"),
            max_abs_error=_require_finite(
                _require_float(row["max_abs_error"], "max_abs_error"), "max_abs_error"
            ),
            cosine_similarity=_require_finite(
                _require_float(row["cosine_similarity"], "cosine_similarity"), "cosine_similarity"
            ),
            failure=failure,
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the tactic measurement."""
        return {
            "signature_key": self.signature_key,
            "tactic": self.tactic.to_json(),
            "median_us": self.median_us,
            "p95_us": self.p95_us,
            "cv": self.cv,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "finite": self.finite,
            "deterministic": self.deterministic,
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class ReplayProfile:
    """A weighted routing signature selected for tactic replay."""

    signature: RoutingSignature
    signature_key: str
    aggregate_gpu_time_us: float
    call_count: int
    normalized_weight: float
    skew_class: Literal["balanced", "median-skew", "high-skew"]

    def __post_init__(self) -> None:
        """Validate replay-profile aggregation values."""
        if not isinstance(self.signature, RoutingSignature):
            raise ValueError("signature must be a RoutingSignature")
        if self.signature_key != self.signature.signature_key():
            raise ValueError("signature_key must match signature")
        if not math.isfinite(self.aggregate_gpu_time_us) or self.aggregate_gpu_time_us <= 0:
            raise ValueError("aggregate_gpu_time_us must be finite and positive")
        if isinstance(self.call_count, bool) or not isinstance(self.call_count, int) or self.call_count <= 0:
            raise ValueError("call_count must be positive")
        if not math.isfinite(self.normalized_weight) or self.normalized_weight <= 0:
            raise ValueError("normalized_weight must be finite and positive")
        if self.skew_class not in {"balanced", "median-skew", "high-skew"}:
            raise ValueError("skew_class must be balanced, median-skew, or high-skew")

    @classmethod
    def from_signature(cls, signature: RoutingSignature, weight: float) -> ReplayProfile:
        """Create the initial profile for one observed routing signature."""
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        mean_count = sum(signature.expert_counts) / signature.global_num_experts
        max_count = max(signature.expert_counts)
        skew_ratio = max_count / mean_count
        if skew_ratio <= 1.0:
            skew_class: Literal["balanced", "median-skew", "high-skew"] = "balanced"
        elif skew_ratio <= 2.0:
            skew_class = "median-skew"
        else:
            skew_class = "high-skew"
        return cls(
            signature=signature,
            signature_key=signature.signature_key(),
            aggregate_gpu_time_us=signature.sampled_gpu_time_us,
            call_count=1,
            normalized_weight=weight,
            skew_class=skew_class,
        )

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> ReplayProfile:
        """Parse a replay-profile JSON object."""
        _require_exact_fields(row, REPLAY_PROFILE_FIELDS)
        skew_class = _require_str(row["skew_class"], "skew_class")
        if skew_class not in {"balanced", "median-skew", "high-skew"}:
            raise ValueError("skew_class must be balanced, median-skew, or high-skew")
        return cls(
            signature=RoutingSignature.from_json(_require_mapping(row["signature"], "signature")),
            signature_key=_require_str(row["signature_key"], "signature_key"),
            aggregate_gpu_time_us=_require_finite(
                _require_float(row["aggregate_gpu_time_us"], "aggregate_gpu_time_us"),
                "aggregate_gpu_time_us",
            ),
            call_count=_require_positive_int(row["call_count"], "call_count"),
            normalized_weight=_require_finite(
                _require_float(row["normalized_weight"], "normalized_weight"), "normalized_weight"
            ),
            skew_class=cast(Literal["balanced", "median-skew", "high-skew"], skew_class),
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the replay profile."""
        return {
            "signature": self.signature.to_json(),
            "signature_key": self.signature_key,
            "aggregate_gpu_time_us": self.aggregate_gpu_time_us,
            "call_count": self.call_count,
            "normalized_weight": self.normalized_weight,
            "skew_class": self.skew_class,
        }
