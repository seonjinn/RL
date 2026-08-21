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

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import nemo_rl.models.megatron.draft.utils as draft_utils


class _DFlashExportModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.config = SimpleNamespace(
            hidden_size=4,
            intermediate_size=6,
            num_key_value_heads=1,
            head_dim=2,
            num_target_taps=2,
        )
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


def _logical_state(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.arange(32, dtype=torch.bfloat16, device=device).view(4, 8),
        "hidden_norm.weight": torch.arange(4, dtype=torch.bfloat16, device=device),
        "layers.0.self_attn.q_proj.weight": torch.arange(
            32, 48, dtype=torch.bfloat16, device=device
        ).view(4, 4),
        "layers.0.self_attn.o_proj.weight": torch.arange(
            48, 64, dtype=torch.bfloat16, device=device
        ).view(4, 4),
        "layers.0.mlp.down_proj.weight": torch.arange(
            64, 88, dtype=torch.float32, device=device
        ).view(4, 6),
        "norm.weight": torch.arange(4, 8, dtype=torch.float32, device=device),
    }


def _local_state(rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    logical = _logical_state(device)
    return {
        "fc.weight": logical["fc.weight"].chunk(2, dim=0)[rank].contiguous(),
        "hidden_norm.weight": logical["hidden_norm.weight"],
        "layers.0.self_attn.q_proj.weight": logical["layers.0.self_attn.q_proj.weight"]
        .chunk(2, dim=0)[rank]
        .contiguous(),
        "layers.0.self_attn.o_proj.weight": logical["layers.0.self_attn.o_proj.weight"]
        .chunk(2, dim=1)[rank]
        .contiguous(),
        "layers.0.mlp.down_proj.weight": logical["layers.0.mlp.down_proj.weight"]
        .chunk(2, dim=1)[rank]
        .contiguous(),
        "norm.weight": logical["norm.weight"],
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
        draft_utils.unwrap_model = lambda wrapped: wrapped
        draft_utils.parallel_state.model_parallel_is_initialized = lambda: True
        draft_utils.parallel_state.get_tensor_model_parallel_group = lambda: (
            dist.group.WORLD
        )

        all_gather_calls = 0
        all_reduce_calls = 0
        real_all_gather = dist.all_gather
        real_all_reduce = dist.all_reduce

        def counted_all_gather(
            gathered: list[torch.Tensor],
            tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal all_gather_calls
            all_gather_calls += 1
            real_all_gather(gathered, tensor, *args, **kwargs)

        def counted_all_reduce(
            tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal all_reduce_calls
            all_reduce_calls += 1
            real_all_reduce(tensor, *args, **kwargs)

        dist.all_gather = counted_all_gather
        dist.all_reduce = counted_all_reduce

        local = _local_state(rank, device)
        exported = draft_utils.export_dflash_weights_to_hf(_DFlashExportModel(local))
        expected = _logical_state(device)
        assert [name for name, _ in exported] == list(local)
        for name, tensor in exported:
            torch.testing.assert_close(tensor, expected[name])
        assert dict(exported)["hidden_norm.weight"] is local["hidden_norm.weight"]
        assert dict(exported)["norm.weight"] is local["norm.weight"]
        assert all_gather_calls == 2
        assert all_reduce_calls == 1

        asymmetric = _local_state(rank, device)
        if rank == 1:
            del asymmetric["norm.weight"]
        try:
            draft_utils.export_dflash_weights_to_hf(_DFlashExportModel(asymmetric))
        except RuntimeError as error:
            assert "manifest differs across TP ranks" in str(error)
        else:
            raise AssertionError("rank-asymmetric manifest was accepted")
        assert all_gather_calls == 2
        assert all_reduce_calls == 2
        dist.barrier()
        if rank == 0:
            print(
                "result=PASS topology=TP2 backend=NCCL "
                "payload_all_gathers=2 manifest_all_reduces=2"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
