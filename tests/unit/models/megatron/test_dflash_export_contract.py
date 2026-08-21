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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nemo_rl.models.megatron.draft as draft_api
import nemo_rl.models.megatron.draft.utils as draft_utils
from nemo_rl.models.megatron.draft.utils import (
    validate_dflash_export_state_dict,
)


pytestmark = pytest.mark.mcore


def test_raw_dflash_export_is_not_a_public_api() -> None:
    """Raw TP-local state must not masquerade as an interoperable export."""
    assert not hasattr(draft_api, "export_dflash_weights")
    assert not hasattr(draft_utils, "export_dflash_weights")


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "lm_head.weight",
        "module.draft_model.output_layer.weight",
        "draft.mask_embedding.weight",
        "module.mask_token",
    ],
)
def test_dflash_export_rejects_target_owned_components(forbidden_key: str) -> None:
    """Target-head and mask-token ownership violations fail before export."""
    with pytest.raises(ValueError, match=forbidden_key):
        validate_dflash_export_state_dict({forbidden_key: torch.ones(1)})


def test_dflash_export_checks_components_instead_of_substrings() -> None:
    """Related body parameter names are not rejected by substring matching."""
    allowed = {
        "head_projection.weight": torch.ones(1),
        "output_layernorm.weight": torch.ones(1),
        "mask_tokenizer_projection.weight": torch.ones(1),
    }

    validate_dflash_export_state_dict(allowed)


def test_dflash_body_export_is_logical_and_excludes_target_owned_weights() -> None:
    from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
    from nemo_rl.models.megatron.draft.utils import export_dflash_weights_to_hf

    body = DFlashBody(
        DFlashBodyConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_target_taps=2,
        )
    )

    exported = dict(export_dflash_weights_to_hf(body))

    assert set(exported) == set(body.state_dict())
    assert not any("lm_head" in name for name in exported)
    assert not any("embedding" in name for name in exported)
    assert not any("mask_token" in name for name in exported)
    for name, tensor in body.state_dict().items():
        torch.testing.assert_close(exported[name], tensor)


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


def _logical_dflash_export_state() -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.arange(32, dtype=torch.bfloat16).view(4, 8),
        "hidden_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        "layers.0.self_attn.q_proj.weight": torch.arange(
            32, 48, dtype=torch.bfloat16
        ).view(4, 4),
        "layers.0.self_attn.o_proj.weight": torch.arange(
            48, 64, dtype=torch.bfloat16
        ).view(4, 4),
        "layers.0.mlp.down_proj.weight": torch.arange(64, 88, dtype=torch.float32).view(
            4, 6
        ),
        "norm.weight": torch.arange(4, 8, dtype=torch.float32),
    }


def _local_dflash_export_state(rank: int) -> dict[str, torch.Tensor]:
    logical = _logical_dflash_export_state()
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


def _patch_dflash_export_parallel_state() -> None:
    draft_utils.unwrap_model = lambda wrapped: wrapped
    draft_utils.parallel_state.model_parallel_is_initialized = lambda: True
    draft_utils.parallel_state.get_tensor_model_parallel_group = lambda: (
        dist.group.WORLD
    )
    draft_utils.parallel_state.get_tensor_model_parallel_world_size = lambda: 2


def _run_dflash_tp2_bucket_export(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _patch_dflash_export_parallel_state()
        payload_gather_calls = 0
        real_all_gather = dist.all_gather

        def counted_all_gather(
            gathered: list[torch.Tensor],
            tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal payload_gather_calls
            if tensor.dtype in {torch.bfloat16, torch.float32}:
                payload_gather_calls += 1
            real_all_gather(gathered, tensor, *args, **kwargs)

        dist.all_gather = counted_all_gather
        state = _local_dflash_export_state(rank)
        exported = draft_utils.export_dflash_weights_to_hf(_DFlashExportModel(state))
        reference = _logical_dflash_export_state()

        assert [name for name, _ in exported] == list(state)
        for name, tensor in exported:
            torch.testing.assert_close(tensor, reference[name])
        assert dict(exported)["hidden_norm.weight"] is state["hidden_norm.weight"]
        assert dict(exported)["norm.weight"] is state["norm.weight"]
        assert payload_gather_calls == 2
    finally:
        dist.destroy_process_group()


def _run_dflash_tp2_asymmetric_manifest(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _patch_dflash_export_parallel_state()
        payload_gather_calls = 0
        real_all_gather = dist.all_gather

        def counted_all_gather(
            gathered: list[torch.Tensor],
            tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> None:
            nonlocal payload_gather_calls
            if tensor.dtype in {torch.bfloat16, torch.float32}:
                payload_gather_calls += 1
            real_all_gather(gathered, tensor, *args, **kwargs)

        dist.all_gather = counted_all_gather
        for mismatch in ("missing", "reordered"):
            state = _local_dflash_export_state(rank)
            if rank == 1 and mismatch == "missing":
                del state["norm.weight"]
            elif rank == 1:
                state = {
                    name: state[name]
                    for name in (
                        "hidden_norm.weight",
                        "fc.weight",
                        "layers.0.self_attn.q_proj.weight",
                        "layers.0.self_attn.o_proj.weight",
                        "layers.0.mlp.down_proj.weight",
                        "norm.weight",
                    )
                }

            with pytest.raises(RuntimeError, match="manifest differs across TP ranks"):
                draft_utils.export_dflash_weights_to_hf(_DFlashExportModel(state))
            assert payload_gather_calls == 0
    finally:
        dist.destroy_process_group()


def test_dflash_tp2_export_uses_one_payload_gather_per_dtype_bucket(
    tmp_path: Path,
) -> None:
    mp.start_processes(
        _run_dflash_tp2_bucket_export,
        args=(2, str(tmp_path / "dflash_export_bucket_init")),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_dflash_tp2_export_rejects_asymmetric_manifests_before_payload(
    tmp_path: Path,
) -> None:
    mp.start_processes(
        _run_dflash_tp2_asymmetric_manifest,
        args=(2, str(tmp_path / "dflash_export_manifest_init")),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_dflash_tp1_export_preserves_tensor_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _logical_dflash_export_state()
    monkeypatch.setattr(draft_utils, "unwrap_model", lambda wrapped: wrapped)
    monkeypatch.setattr(
        draft_utils.parallel_state, "model_parallel_is_initialized", lambda: False
    )

    exported = draft_utils.export_dflash_weights_to_hf(_DFlashExportModel(state))

    assert [name for name, _ in exported] == list(state)
    for name, tensor in exported:
        assert tensor is state[name]
