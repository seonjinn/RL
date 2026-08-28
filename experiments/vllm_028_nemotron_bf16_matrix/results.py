#!/usr/bin/env python3
"""Result-contract validation for vLLM 0.28 DynamicMTP gates."""

from __future__ import annotations

from typing import Any


def validate_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete token work and non-functional K=0 DynamicSD canaries."""
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported result schema_version")
    if payload.get("status") != "complete":
        raise ValueError("result status must be complete")
    config = payload.get("config")
    summary = payload.get("summary")
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("result must include config and summary objects")
    if summary.get("tokens_ok") is not True:
        raise ValueError("result tokens_ok must be true")
    metrics = summary.get("spec_metrics")
    legacy_metrics = False
    if not isinstance(metrics, dict):
        metrics = summary.get("spec_decode_metrics")
        legacy_metrics = True
    if not isinstance(metrics, dict):
        raise ValueError("result must include per-run spec_metrics")
    required_metrics = (
        {"num_draft_tokens", "num_accepted_tokens", "acceptance_rate"}
        if legacy_metrics
        else {"draft_tokens", "accepted_tokens", "acceptance_rate"}
    )
    if not required_metrics.issubset(metrics):
        raise ValueError("spec_metrics are incomplete")
    draft_tokens_key = "num_draft_tokens" if legacy_metrics else "draft_tokens"
    if config.get("effective_k") == 0 and float(metrics[draft_tokens_key]) != 0.0:
        raise ValueError("K=0 canary produced nonzero draft tokens")
    return payload
