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

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_symbols():
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[3] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.eagle_ttt")
    return (
        module.EagleTTTAttentionPlan,
        module.EagleTTTState,
        module.eagle_ttt_attention,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(2026)
    shape = (1, 4, 7, 8)
    tensors = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(9)]
    return tuple(tensors)


def _run_attention(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    EagleTTTAttentionPlan, EagleTTTState, eagle_ttt_attention = _load_symbols()
    differentiable = tuple(tensor.detach().requires_grad_(True) for tensor in tensors)
    query, trunk_key, trunk_value, *branches = differentiable
    branch_keys = tuple(branches[:3])
    branch_values = tuple(branches[3:])
    state = EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=4,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for key, value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(branch_key=key, branch_value=value)
    plan = EagleTTTAttentionPlan(
        pass_index=3,
        pass_count=4,
        max_passes=8,
        sequence_length=query.shape[2],
    )
    output = eagle_ttt_attention(query=query, state=state, plan=plan)
    gradients = torch.autograd.grad(output.float().square().sum(), differentiable)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _tp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        full_inputs = _inputs()
        local_inputs = tuple(
            tensor.chunk(world_size, dim=1)[rank].contiguous() for tensor in full_inputs
        )
        local_output, local_gradients = _run_attention(local_inputs)
        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output)
        gathered_gradients: list[list[torch.Tensor]] = []
        for gradient in local_gradients:
            gathered = [torch.empty_like(gradient) for _ in range(world_size)]
            dist.all_gather(gathered, gradient)
            gathered_gradients.append(gathered)

        if rank == 0:
            expected_output, expected_gradients = _run_attention(full_inputs)
            actual_output = torch.cat(gathered_outputs, dim=1)
            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=5e-2,
                rtol=5e-2,
            )
            for gathered, expected in zip(
                gathered_gradients,
                expected_gradients,
                strict=True,
            ):
                actual = torch.cat(gathered, dim=1)
                torch.testing.assert_close(actual, expected, atol=8e-2, rtol=8e-2)
                assert torch.count_nonzero(actual) > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Gloo distributed support is required",
)
def test_tp2_bfloat16_matches_dense_output_and_all_gradients(tmp_path: Path) -> None:
    init_file = tmp_path / "eagle-ttt-tp2-init"
    mp.spawn(
        _tp_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
