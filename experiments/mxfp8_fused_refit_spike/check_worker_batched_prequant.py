#!/usr/bin/env python3
"""Check that the policy worker routes selected weights through batch prequant."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from nemo_rl.models.generation.vllm.quantization import fp8_train_utils
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


def main() -> None:
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    weight = torch.ones(2, 32, dtype=torch.bfloat16)
    calls = []

    def iter_batched(
        params, selected_names, *, scratch_cache, max_experts_per_batch
    ):
        calls.append(
            (list(params), selected_names, scratch_cache, max_experts_per_batch)
        )
        yield "batched.weight", weight

    original = fp8_train_utils.iter_mxfp8_prequantized_params
    fp8_train_utils.iter_mxfp8_prequantized_params = iter_batched
    os.environ["NRL_MXFP8_BATCHED_EXPERT_PREQUANTIZE"] = "1"
    try:
        worker = object.__new__(MegatronPolicyWorkerImpl)
        worker._refit_prequant_names = {name}
        worker.model = object()
        worker.draft_model = None
        worker.refit_conversion_tasks = []
        worker.cfg = {"megatron_cfg": {"enabled": True}}
        worker.megatron_bridge = SimpleNamespace(
            export_hf_weights=lambda *_args, **_kwargs: iter([(name, weight)])
        )

        first = list(worker._iter_params_with_optional_kv_scales())
        second = list(worker._iter_params_with_optional_kv_scales())
    finally:
        fp8_train_utils.iter_mxfp8_prequantized_params = original

    assert first == [("batched.weight", weight)]
    assert second == first
    assert len(calls) == 2
    assert calls[0][0] == [(name, weight)]
    assert calls[0][1] == {name}
    assert calls[0][2] is calls[1][2]
    assert calls[0][3] == 16


if __name__ == "__main__":
    main()
