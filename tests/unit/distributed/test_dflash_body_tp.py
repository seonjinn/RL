from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from nemo_rl.models.megatron.draft.block_plan import build_dflash_batch_plan
from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig


pytestmark = pytest.mark.mcore


_COLUMN_SHARDS = (
    "fc.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)
_ROW_SHARDS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def _distributed_world(expected_world_size: int) -> Iterator[None]:
    created_process_group = False
    if not torch.distributed.is_initialized():
        if int(os.environ.get("WORLD_SIZE", "1")) != expected_world_size:
            pytest.skip(f"run with torchrun --nproc-per-node={expected_world_size}")
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        created_process_group = True
    if torch.distributed.get_world_size() != expected_world_size:
        pytest.skip(f"run with torchrun --nproc-per-node={expected_world_size}")
    try:
        yield
    finally:
        if created_process_group:
            torch.distributed.destroy_process_group()


@pytest.fixture
def _tp2_world() -> Iterator[None]:
    yield from _distributed_world(2)


@pytest.fixture
def _pp2_world() -> Iterator[None]:
    yield from _distributed_world(4)


def _config() -> DFlashBodyConfig:
    return DFlashBodyConfig(
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        num_target_taps=2,
        rope_theta=10_000.0,
    )


def _singleton_groups(world_size: int) -> list[torch.distributed.ProcessGroup]:
    return [torch.distributed.new_group([rank]) for rank in range(world_size)]


def _shard_axis(name: str) -> int | None:
    if name == "fc.weight" or any(suffix in name for suffix in _COLUMN_SHARDS[1:]):
        return 0
    if any(suffix in name for suffix in _ROW_SHARDS):
        return 1
    return None


def _gather_global_state(body: DFlashBody) -> dict[str, Tensor]:
    local_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in body.state_dict().items()
    }
    gathered: list[dict[str, Tensor] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_state)
    states = [state for state in gathered if state is not None]
    global_state: dict[str, Tensor] = {}
    for name in local_state:
        axis = _shard_axis(name)
        global_state[name] = (
            states[0][name]
            if axis is None
            else torch.cat([state[name] for state in states], dim=axis)
        )
    return global_state


def _gather_global_gradients(body: DFlashBody) -> dict[str, Tensor]:
    local_gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in body.named_parameters()
        if parameter.grad is not None
    }
    gathered: list[dict[str, Tensor] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_gradients)
    gradients = [gradient for gradient in gathered if gradient is not None]
    global_gradients: dict[str, Tensor] = {}
    for name in local_gradients:
        axis = _shard_axis(name)
        global_gradients[name] = (
            gradients[0][name]
            if axis is None
            else torch.cat([gradient[name] for gradient in gradients], dim=axis)
        )
    return global_gradients


def _inputs() -> tuple[Any, Tensor, Tensor]:
    token_valid = torch.ones((1, 5), dtype=torch.bool)
    plan = build_dflash_batch_plan(
        token_valid,
        torch.tensor([7], dtype=torch.int64),
        anchors_per_sample=1,
        gamma=2,
        optimizer_step=1,
        seed=19,
    )
    generator = torch.Generator().manual_seed(2026)
    target_taps = torch.randn((1, 5, 2, 8), generator=generator)
    block_embeddings = torch.randn((1, 3, 8), generator=generator)
    return plan, target_taps, block_embeddings


def test_tp2_projection_forward_gradient_and_checkpoint_parity(
    tmp_path: Path,
    _tp2_world: None,
) -> None:
    rank = torch.distributed.get_rank()
    tp_group = torch.distributed.group.WORLD
    singleton_groups = _singleton_groups(2)
    dp_group = singleton_groups[rank]

    torch.manual_seed(31)
    body = DFlashBody(_config(), tp_group=tp_group)
    global_state = _gather_global_state(body)
    torch.manual_seed(31)
    reference = DFlashBody(_config(), tp_group=dp_group)
    reference.load_state_dict(global_state, strict=True)

    local_parameter_count = sum(parameter.numel() for parameter in body.parameters())
    reference_parameter_count = sum(
        parameter.numel() for parameter in reference.parameters()
    )
    assert local_parameter_count * 5 < reference_parameter_count * 3

    plan, target, blocks = _inputs()
    target_actual = target.clone().requires_grad_()
    blocks_actual = blocks.clone().requires_grad_()
    target_reference = target.clone().requires_grad_()
    blocks_reference = blocks.clone().requires_grad_()

    actual = body(
        target_taps=target_actual,
        block_embeddings=blocks_actual,
        plan=plan,
    )
    expected = reference(
        target_taps=target_reference,
        block_embeddings=blocks_reference,
        plan=plan,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    weight = torch.linspace(0.25, 1.25, actual.numel()).reshape_as(actual)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.testing.assert_close(target_actual.grad, target_reference.grad)
    torch.testing.assert_close(blocks_actual.grad, blocks_reference.grad)

    actual_gradients = _gather_global_gradients(body)
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(actual_gradients[name], parameter.grad)

    metadata = {"dp_cp_group": dp_group}
    sharded = body.sharded_state_dict(prefix="draft.", metadata=metadata)
    for name, tensor in global_state.items():
        assert tuple(sharded[f"draft.{name}"].global_shape) == tuple(tensor.shape)

    dist_checkpointing = pytest.importorskip("megatron.core.dist_checkpointing")
    checkpoint_paths = [str(tmp_path / "dflash_tp2") if rank == 0 else ""]
    torch.distributed.broadcast_object_list(checkpoint_paths, src=0)
    checkpoint_dir = checkpoint_paths[0]
    if rank == 0:
        Path(checkpoint_dir).mkdir(parents=True)
    torch.distributed.barrier()
    dist_checkpointing.save({"model": sharded}, checkpoint_dir)
    restored = DFlashBody(_config(), tp_group=tp_group)
    template = restored.sharded_state_dict(prefix="draft.", metadata=metadata)
    loaded = dist_checkpointing.load({"model": template}, checkpoint_dir)
    restored.load_state_dict(
        {
            name.removeprefix("draft."): tensor
            for name, tensor in loaded["model"].items()
        },
        strict=True,
    )
    for name, parameter in body.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)


def test_pp2_last_stage_uses_group_local_replica_ids(_pp2_world: None) -> None:
    rank = torch.distributed.get_rank()
    tp_groups = [
        torch.distributed.new_group([0, 1]),
        torch.distributed.new_group([2, 3]),
    ]
    singleton_groups = _singleton_groups(4)
    if rank >= 2:
        tp_group = tp_groups[1]
        body = DFlashBody(_config(), tp_group=tp_group)
        sharded = body.sharded_state_dict(
            prefix="draft.",
            metadata={"dp_cp_group": singleton_groups[rank]},
        )
        replicated = sharded["draft.hidden_norm.weight"]
        q_projection = sharded["draft.layers.0.self_attn.q_proj.weight"]

        assert replicated.replica_id == (0, rank - 2, 0)
        assert q_projection.replica_id == (0, 0, 0)
        assert q_projection.axis_fragmentations == (2, 1)
        assert q_projection.global_offset == ((rank - 2) * 4, 0)
        assert q_projection.global_shape == (8, 8)
    torch.distributed.barrier()
