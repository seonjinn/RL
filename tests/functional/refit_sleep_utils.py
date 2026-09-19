# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only helpers for the refit-aware sleep correctness gate."""

import math
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

MISSING_KEY = "__task6_unsent_manifest_entry__.weight"


@dataclass(frozen=True)
class Observation:
    tokens: tuple[tuple[int, ...], ...]
    logprobs: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        assert self.tokens and len(self.tokens) == len(self.logprobs)
        for tokens, logprobs in zip(self.tokens, self.logprobs, strict=True):
            assert tokens and len(tokens) == len(logprobs)
            assert all(math.isfinite(lp) and lp <= 0 for lp in logprobs)


def observe(
    prompt_lengths: torch.Tensor, result: Mapping[str, torch.Tensor]
) -> Observation:
    tokens = []
    logprobs = []
    assert len(prompt_lengths) == len(result["output_ids"])
    for row, (start, stop) in enumerate(
        zip(
            prompt_lengths.tolist(),
            result["unpadded_sequence_lengths"].tolist(),
            strict=True,
        )
    ):
        assert 0 <= start < stop <= result["output_ids"].shape[1]
        assert stop <= result["logprobs"].shape[1]
        tokens.append(tuple(result["output_ids"][row, start:stop].tolist()))
        logprobs.append(tuple(result["logprobs"][row, start:stop].tolist()))
    return Observation(tuple(tokens), tuple(logprobs))


def assert_distinct(
    previous: Observation, current: Observation, *, logprob_atol: float
) -> None:
    if previous.tokens == current.tokens:
        assert fresh_error(previous, current) > 2 * logprob_atol, (
            "stale weights: consecutive states did not exceed the measured numerical noise"
        )


def fresh_error(candidate: Observation, fresh: Observation) -> float:
    assert candidate.tokens == fresh.tokens, "fresh-engine tokens differ"
    return max(
        abs(a - b)
        for left, right in zip(candidate.logprobs, fresh.logprobs, strict=True)
        for a, b in zip(left, right, strict=True)
    )


def parity_metrics(generation: Observation, policy: Observation) -> dict[str, float]:
    assert generation.tokens == policy.tokens
    differences = [
        p - g
        for gen, train in zip(generation.logprobs, policy.logprobs, strict=True)
        for g, p in zip(gen, train, strict=True)
    ]
    # Same selected-token multiplicative error and KL(P_gen || P_train) k3
    # as test_vllm_generation.py and ClippedPGLossFn, excluding prompt/pad.
    return {
        "token_mult_prob_error": sum(math.exp(abs(d)) for d in differences)
        / len(differences),
        "gen_kl_error": sum(math.expm1(d) - d for d in differences) / len(differences),
    }


@dataclass(frozen=True)
class Tolerances:
    fresh_logprob_atol: float
    token_mult_prob_error_max: float
    gen_kl_error_max: float

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Tolerances":
        if (
            record.get("hardware") != "GB200"
            or not re.fullmatch(r"[0-9a-f]{40}", record.get("source_sha", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", record.get("image_digest", ""))
            or not record.get("observation_artifact")
        ):
            raise ValueError(
                "Tolerances require a GB200 observation artifact, SHA and image digest"
            )
        values = [float(record[name]) for name in cls.__dataclass_fields__]
        if not all(math.isfinite(v) and v >= 0 for v in values) or values[1] < 1:
            raise ValueError(
                "Tolerances must be finite nonnegative bounds; multiplicative error >= 1"
            )
        return cls(*values)

    def check(self, metrics: Mapping[str, float]) -> None:
        for name, limit in (
            ("fresh_logprob_max_abs", self.fresh_logprob_atol),
            ("token_mult_prob_error", self.token_mult_prob_error_max),
            ("gen_kl_error", self.gen_kl_error_max),
        ):
            assert math.isfinite(metrics[name]) and metrics[name] <= limit, (
                name,
                metrics[name],
                limit,
            )


@torch.no_grad()
def scale_parameters(parameters: Iterable[torch.nn.Parameter], *, factor: float) -> int:
    """Apply a fixed, rank-independent update to real BF16 policy parameters."""
    assert math.isfinite(factor) and factor > 0 and factor != 1
    changed = 0
    for parameter in parameters:
        assert parameter.dtype in (torch.bfloat16, torch.float32)
        before = parameter.detach().clone()
        parameter.mul_(factor)
        assert torch.isfinite(parameter).all()
        changed += int(torch.count_nonzero(parameter != before))
    assert changed > 0, "deterministic update changed no parameters"
    return changed


def incomplete_receiver_manifest(source: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the existing drain-and-ACK manifest rejection, without a hook."""
    assert MISSING_KEY not in source
    return {**source, MISSING_KEY: (torch.Size([1]), torch.bfloat16)}


def assert_bf16_control_metrics(metrics: Mapping[str, Mapping[str, float]]) -> None:
    steps = {str(step) for step in range(1, 21)}
    for key in ("train/loss", "train/gen_kl_error"):
        assert steps <= metrics[key].keys(), (
            f"BF16 control did not finish 20 steps: {key}"
        )
        assert all(math.isfinite(metrics[key][step]) for step in steps)
    # Preserve the existing BF16 wrapper's bound, not a new MXFP8 tolerance.
    assert sum(metrics["train/gen_kl_error"][step] for step in steps) / 20 < 0.002


@contextmanager
def failure_deadline(seconds: float) -> Iterator[None]:
    """Fail the entire test process if discarded-state work or cleanup hangs.

    This is an outer test watchdog, not an IPC injection. Exit 124 is always
    failure; only an acknowledged manifest rejection can pass the real gate.
    """
    assert math.isfinite(seconds) and seconds > 0

    def terminate() -> None:
        os.write(2, b"Discarded-state test exceeded its configured deadline\n")
        os._exit(124)

    watchdog = threading.Timer(seconds, terminate)
    watchdog.daemon = True
    watchdog.start()
    try:
        yield
    finally:
        watchdog.cancel()
        watchdog.join()
