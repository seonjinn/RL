import json
from collections.abc import Mapping
from numbers import Real
from typing import Any


def format_spec_decode_metrics(
    metrics: Mapping[str, Any], step: int
) -> str | None:
    """Format scalar vLLM SpecDec metrics as one parser-friendly log line."""
    payload: dict[str, float | int] = {"step": step}
    for name, value in metrics.items():
        if (
            name.startswith("vllm/spec_")
            and isinstance(value, Real)
            and not isinstance(value, bool)
        ):
            payload[name] = float(value)

    if len(payload) == 1:
        return None
    return "VLLM_SPEC_DECODE_METRICS " + json.dumps(
        payload, allow_nan=False, sort_keys=True
    )
