# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GB200 gate's numerical oracle and test-only update."""

import math
import subprocess
import sys
import time

import pytest
import torch

from tests.functional import refit_sleep_utils as oracle


def test_selected_tokens_exclude_prompt_and_padding() -> None:
    result = {
        "output_ids": torch.tensor([[10, 11, 20, 21, 0], [12, 22, 0, 0, 0]]),
        "unpadded_sequence_lengths": torch.tensor([4, 2]),
        "logprobs": torch.tensor([[99, 99, -1, -2, 99], [99, -3, 99, 99, 99]]),
    }
    observed = oracle.observe(torch.tensor([2, 1]), result)
    assert observed.tokens == ((20, 21), (22,))
    assert observed.logprobs == ((-1.0, -2.0), (-3.0,))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
def test_selected_logprobs_must_be_finite_raw_probabilities(value: float) -> None:
    with pytest.raises(AssertionError):
        oracle.Observation(((1,),), ((value,),))


def test_empty_or_misaligned_observations_cannot_pass() -> None:
    for tokens, logprobs in [((), ()), (((),), ((),)), (((1, 2),), ((-1.0,),))]:
        with pytest.raises(AssertionError):
            oracle.Observation(tokens, logprobs)


def test_fresh_comparison_rejects_tokens_and_stale_updates() -> None:
    a = oracle.Observation(((1,),), ((-1.0,),))
    b = oracle.Observation(((1,),), ((-1.5,),))
    c = oracle.Observation(((2,),), ((-1.5,),))
    oracle.assert_distinct(a, b, logprob_atol=0)
    oracle.assert_distinct(b, c, logprob_atol=0)
    with pytest.raises(AssertionError, match="stale"):
        oracle.assert_distinct(a, a, logprob_atol=0)
    with pytest.raises(AssertionError, match="tokens"):
        oracle.fresh_error(b, c)
    assert oracle.fresh_error(a, b) == 0.5


def test_stale_guard_exceeds_measured_numerical_noise() -> None:
    a = oracle.Observation(((1,),), ((-1.0,),))
    noisy_a = oracle.Observation(((1,),), ((-1.001,),))
    with pytest.raises(AssertionError, match="stale"):
        oracle.assert_distinct(a, noisy_a, logprob_atol=0.01)


def test_parity_matches_selected_token_multi_probability_and_k3() -> None:
    generation = oracle.Observation(((1, 2),), ((-1.0, -2.0),))
    policy = oracle.Observation(((1, 2),), ((-1.5, -1.5),))
    metrics = oracle.parity_metrics(generation, policy)
    assert metrics["token_mult_prob_error"] == pytest.approx(math.exp(0.5))
    assert metrics["gen_kl_error"] == pytest.approx(math.cosh(0.5) - 1)


def test_tolerances_require_measured_gb200_provenance() -> None:
    with pytest.raises(ValueError, match="GB200"):
        oracle.Tolerances.from_record({})
    record = {
        "hardware": "GB200",
        "source_sha": "8" * 40,
        "image_digest": "sha256:" + "a" * 64,
        "observation_artifact": "/lustre/experiment/observations.json",
        "fresh_logprob_atol": 0.01,
        "token_mult_prob_error_max": 1.1,
        "gen_kl_error_max": 0.01,
    }
    limits = oracle.Tolerances.from_record(record)
    limits.check(
        {"fresh_logprob_max_abs": 0, "token_mult_prob_error": 1, "gen_kl_error": 0}
    )
    with pytest.raises(AssertionError):
        limits.check(
            {
                "fresh_logprob_max_abs": 0.02,
                "token_mult_prob_error": 1,
                "gen_kl_error": 0,
            }
        )
    record["gen_kl_error_max"] = float("nan")
    with pytest.raises(ValueError):
        oracle.Tolerances.from_record(record)


def test_deterministic_update_changes_real_bf16_parameters() -> None:
    a = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.0], dtype=torch.bfloat16))
    b = torch.nn.Parameter(a.detach().clone())
    assert oracle.scale_parameters([a], factor=1.0625) == 2
    assert oracle.scale_parameters([b], factor=1.0625) == 2
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert oracle.scale_parameters([a], factor=0.875) == 2
    assert not torch.equal(a, b)


def test_missing_manifest_entry_does_not_modify_sender_metadata() -> None:
    source = {"model.weight": (torch.Size([1]), torch.bfloat16)}
    receiver = oracle.incomplete_receiver_manifest(source)
    assert set(receiver) - set(source) == {oracle.MISSING_KEY}
    assert list(source) == ["model.weight"]
    assert receiver["model.weight"] == source["model.weight"]


@pytest.mark.parametrize("overrun,expected", [(True, 124), (False, 0)])
def test_failure_deadline_is_process_bounded(overrun: bool, expected: int) -> None:
    program = (
        "import time\n"
        "from tests.functional.refit_sleep_utils import failure_deadline\n"
        "with failure_deadline(0.2):\n"
        f"    time.sleep({1 if overrun else 0})\n"
        "time.sleep(0.3)\n"
    )
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", program], timeout=10, capture_output=True
    )
    assert result.returncode == expected, result.stderr.decode()
    assert time.monotonic() - started < 10


def test_bf16_control_requires_twenty_complete_finite_steps() -> None:
    metrics = {
        "train/loss": {str(step): 0.1 for step in range(1, 21)},
        "train/gen_kl_error": {str(step): 0.001 for step in range(1, 21)},
    }
    oracle.assert_bf16_control_metrics(metrics)
    metrics["train/gen_kl_error"]["20"] = float("nan")
    with pytest.raises(AssertionError):
        oracle.assert_bf16_control_metrics(metrics)
    del metrics["train/gen_kl_error"]["20"]
    with pytest.raises(AssertionError):
        oracle.assert_bf16_control_metrics(metrics)
