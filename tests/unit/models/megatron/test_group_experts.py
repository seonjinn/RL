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

"""Tests for train-side expert stacking and native MXFP8 source adaptation."""

import pickle
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


class _FakeNativeBridge:
    def __init__(self) -> None:
        self.outputs: dict[int, tuple[SimpleNamespace, ...] | Exception] = {}
        self.calls: list[object] = []

    def iter_local_native_mxfp8_params(
        self, tasks: list[object]
    ) -> Iterator[SimpleNamespace]:
        assert len(tasks) == 1
        task = tasks[0]
        self.calls.append(task)
        output = self.outputs.get(id(task), ())
        if isinstance(output, Exception):
            raise output
        yield from output


def test_pinned_auto_bridge_exposes_native_mxfp8_public_api() -> None:
    from megatron.bridge import AutoBridge

    assert callable(AutoBridge.get_export_mxfp8_tasks)
    assert callable(AutoBridge.iter_local_native_mxfp8_params)


def _record(
    name: str,
    shape: tuple[int, ...],
    *,
    weight: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=weight
        if weight is not None
        else torch.empty(shape, dtype=torch.float8_e4m3fn),
        weight_scale=weight_scale
        if weight_scale is not None
        else torch.empty((*shape[:-1], shape[-1] // 32), dtype=torch.uint8),
        global_weight_shape=torch.Size(shape),
    )


def _task(
    global_name: str,
    *,
    owned: bool = True,
    broadcast: Callable[[object, str], object] | None = None,
) -> SimpleNamespace:
    mapping = SimpleNamespace()
    if broadcast is not None:
        mapping.broadcast_obj_from_pp_rank = broadcast
        mapping.pp_size = 2
    return SimpleNamespace(
        global_param_name=global_name,
        param_weight=object() if owned else None,
        mapping=mapping,
    )


def _native_worker(
    tasks: list[SimpleNamespace], bridge: _FakeNativeBridge
) -> MegatronPolicyWorkerImpl:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.fp8_cfg = {"enabled": True, "fp8_param": True, "fp8_recipe": "mxfp8"}
    worker.cfg = cast(
        Any,
        {
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"precision": "fp8", "is_mx": True},
            }
        },
    )
    worker.refit_conversion_tasks = tasks
    worker._native_mxfp8_conversion_tasks = tasks
    worker._native_grouped_mxfp8_tasks = []
    worker._misc_conversion_tasks = []
    worker._native_direct_component_specs = {}
    worker.megatron_bridge = SimpleNamespace(
        iter_local_native_mxfp8_params=bridge.iter_local_native_mxfp8_params
    )
    return worker


def _native_components(shape: tuple[int, ...]) -> list[dict[str, object]]:
    return [
        {
            "role": "weight",
            "global_shape": shape,
            "dtype": "torch.float8_e4m3fn",
        },
        {
            "role": "weight_scale",
            "global_shape": (*shape[:-1], shape[-1] // 32),
            "dtype": "torch.uint8",
        },
    ]


def _refit_info(
    params: list[tuple[str, tuple[int, ...], str | None]],
) -> dict[str, Any]:
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "components": _native_components(shape),
                    **(
                        {"grouped_expert_proj": grouped_projection}
                        if grouped_projection is not None
                        else {}
                    ),
                }
                for name, shape, grouped_projection in params
            ]
        },
    }


def _group(
    projection: str,
    grouped_name: str,
    expert_groups: dict[tuple[str, str], list[torch.Tensor]],
) -> torch.Tensor:
    return MegatronPolicyWorkerImpl._group_experts(
        cast(Any, SimpleNamespace()), projection, grouped_name, expert_groups
    )


def test_group_experts_stacks_in_order() -> None:
    prefix = "model.layers.0.mlp.experts"
    experts = [torch.randn(4, 8) for _ in range(3)]

    result = _group(
        "gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): experts}
    )

    assert result.shape == (3, 4, 8)
    assert all(
        torch.equal(result[index], expert) for index, expert in enumerate(experts)
    )


@pytest.mark.parametrize(
    "groups", [{}, {("model.layers.0.mlp.experts", "gate_proj"): []}]
)
def test_group_experts_requires_local_experts(
    groups: dict[tuple[str, str], list[torch.Tensor]],
) -> None:
    with pytest.raises(AssertionError, match="no local experts"):
        _group(
            "gate_proj",
            "model.layers.0.mlp.experts.gate_proj.weight",
            groups,
        )


def test_build_hf_to_local_param_map_train_side() -> None:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(8, 16)
    experts = [torch.randn(4, 16), torch.randn(4, 16)]
    worker._iter_local_hf_param_shards = cast(
        Any,
        lambda: iter(
            [
                ("model.layers.0.mlp.down_proj.weight", direct),
                (f"{prefix}.0.gate_proj.weight", experts[0]),
                (f"{prefix}.1.gate_proj.weight", experts[1]),
            ]
        ),
    )
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {"name": "model.layers.0.mlp.down_proj.weight"},
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "grouped_expert_proj": "gate_proj",
                },
            ]
        },
    }

    param_map = worker.build_hf_to_local_param_map(refit_info)

    direct_spec = param_map.get("model.layers.0.mlp.down_proj.weight")
    assert direct_spec is not None
    assert direct_spec.base is direct
    grouped_spec = param_map.get(f"{prefix}.gate_proj.weight")
    assert grouped_spec is not None and grouped_spec.pre is not None
    assert torch.equal(grouped_spec.pre(grouped_spec.base).buf, torch.stack(experts))


def test_native_mxfp8_task_builder_uses_public_api_and_preserves_order() -> None:
    tasks = [
        _task("decoder.layers.0.mlp.linear_fc1.weight"),
        _task("decoder.layers.0.mlp.experts.linear_fc1.weight"),
        _task("decoder.layers.0.mlp.linear_fc2.weight"),
        _task("decoder.layers.0.mlp.experts.linear_fc2.weight"),
    ]
    calls: list[object] = []

    def get_export_mxfp8_tasks(models: list[object]) -> list[SimpleNamespace]:
        calls.extend(models)
        return tasks

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = object()
    worker.megatron_bridge = SimpleNamespace(
        get_export_mxfp8_tasks=get_export_mxfp8_tasks
    )

    result = worker._build_native_mxfp8_conversion_tasks()

    assert result == tasks
    assert calls == [worker.model]
    assert worker._native_grouped_mxfp8_tasks == [tasks[1], tasks[3]]


def test_native_mxfp8_component_iterator_delegates_each_task_in_order() -> None:
    tasks = [
        _task("decoder.layers.0.mlp.linear_fc1.weight"),
        _task("decoder.layers.0.mlp.linear_fc2.weight"),
    ]
    names = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    bridge = _FakeNativeBridge()
    bridge.outputs[id(tasks[0])] = (
        _record(names[0], (8, 64)),
        _record(names[1], (8, 64)),
    )
    bridge.outputs[id(tasks[1])] = (_record(names[2], (64, 32)),)
    worker = _native_worker(tasks, bridge)

    components = list(worker._iter_local_native_mxfp8_param_components())

    assert bridge.calls == tasks
    assert [(name, role) for name, role, _ in components] == [
        (names[0], "weight"),
        (names[0], "weight_scale"),
        (names[1], "weight"),
        (names[1], "weight_scale"),
        (names[2], "weight"),
        (names[2], "weight_scale"),
    ]


def test_native_mxfp8_partition_preserves_native_grouped_and_misc_order() -> None:
    native = _task("decoder.layers.0.mlp.linear_fc1.weight")
    grouped = _task("decoder.layers.0.mlp.experts.linear_fc2.weight")
    bf16 = _task("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight")
    qkv = _task("decoder.layers.0.self_attention.linear_qkv.weight")
    output = _task("decoder.layers.0.self_attention.linear_proj.weight")
    router = _task("decoder.layers.0.mlp.router.weight")
    embedding = _task("embedding.word_embeddings.weight")
    shared_expert = _task("decoder.layers.0.mlp.shared_expert.linear_fc1.weight")
    mtp = _task("mtp.layers.0.transformer_layer.mlp.linear_fc1.weight")
    tasks = [
        native,
        bf16,
        grouped,
        qkv,
        output,
        router,
        embedding,
        shared_expert,
        mtp,
    ]
    bridge = _FakeNativeBridge()
    bridge.outputs[id(native)] = (
        _record("model.layers.0.mlp.gate_proj.weight", (8, 64)),
    )
    bridge.outputs[id(grouped)] = (
        _record("model.layers.0.mlp.experts.0.down_proj.weight", (64, 32)),
    )
    bridge.outputs[id(qkv)] = (
        _record("model.layers.0.self_attn.q_proj.weight", (128, 64)),
        _record("model.layers.0.self_attn.k_proj.weight", (32, 64)),
        _record("model.layers.0.self_attn.v_proj.weight", (32, 64)),
    )
    bridge.outputs[id(output)] = (
        _record("model.layers.0.self_attn.o_proj.weight", (64, 64)),
    )
    bridge.outputs[id(shared_expert)] = (
        _record("model.layers.0.mlp.shared_expert.gate_proj.weight", (8, 64)),
    )
    bridge.outputs[id(mtp)] = AssertionError("MTP must stay on the misc path")
    worker = _native_worker(tasks, bridge)
    worker._native_grouped_mxfp8_tasks = [grouped]

    native_tasks, grouped_tasks, misc_tasks = (
        worker._partition_native_mxfp8_conversion_tasks(tasks)
    )

    assert native_tasks == [native, grouped, qkv, output]
    assert grouped_tasks == [grouped]
    assert misc_tasks == [bf16, router, embedding, shared_expert, mtp]
    assert bridge.calls == [
        native,
        bf16,
        grouped,
        qkv,
        output,
        router,
        embedding,
        shared_expert,
    ]


def test_native_mxfp8_partition_rejects_mixed_bulk_and_misc_outputs() -> None:
    task = _task("decoder.layers.0.mlp.linear_fc1.weight")
    bridge = _FakeNativeBridge()
    bridge.outputs[id(task)] = (
        _record("model.layers.0.mlp.gate_proj.weight", (8, 64)),
        _record("model.layers.0.mlp.shared_expert.gate_proj.weight", (8, 64)),
    )
    worker = _native_worker([task], bridge)

    with pytest.raises(ValueError, match="mixes bulk and misc"):
        worker._partition_native_mxfp8_conversion_tasks([task])


def test_native_mxfp8_partition_propagates_bridge_validation_error() -> None:
    task = _task("decoder.layers.0.mlp.linear_fc1.weight")
    bridge = _FakeNativeBridge()
    validation_error = ValueError("mapping does not support projection")
    bridge.outputs[id(task)] = validation_error
    worker = _native_worker([task], bridge)

    with pytest.raises(ValueError, match="does not support projection") as raised:
        worker._partition_native_mxfp8_conversion_tasks([task])
    assert raised.value is validation_error


def test_native_mxfp8_storage_error_broadcasts_before_all_pp_ranks_raise() -> None:
    global_name = "decoder.layers.0.self_attention.linear_qkv.weight"
    broadcasted: list[object] = []
    calls: list[tuple[str, object, str]] = []

    def owner_broadcast(value: object, cache_key: str) -> object:
        pickle.dumps(value)
        calls.append(("owner", value, cache_key))
        broadcasted.append(value)
        return value

    def placeholder_broadcast(value: object, cache_key: str) -> object:
        calls.append(("placeholder", value, cache_key))
        assert value is None
        return broadcasted[0]

    owner_task = _task(global_name, broadcast=owner_broadcast)
    placeholder_task = _task(global_name, owned=False, broadcast=placeholder_broadcast)
    validation_error = ValueError("invalid QKV native scale storage")
    owner_bridge = _FakeNativeBridge()
    owner_bridge.outputs[id(owner_task)] = validation_error
    placeholder_bridge = _FakeNativeBridge()
    owner_worker = _native_worker([owner_task], owner_bridge)
    placeholder_worker = _native_worker([placeholder_task], placeholder_bridge)

    with pytest.raises(ValueError) as owner_raised:
        owner_worker._task_uses_native_mxfp8_storage(owner_task)
    with pytest.raises(ValueError) as placeholder_raised:
        placeholder_worker._task_uses_native_mxfp8_storage(placeholder_task)

    assert str(owner_raised.value) == str(placeholder_raised.value)
    assert "storage classification" in str(owner_raised.value)
    assert "ValueError: invalid QKV native scale storage" in str(owner_raised.value)
    assert owner_raised.value.__cause__ is validation_error
    assert placeholder_raised.value.__cause__ is None
    assert calls == [
        (
            "owner",
            ("error", "ValueError", "invalid QKV native scale storage"),
            f"native-mxfp8-storage:{global_name}",
        ),
        ("placeholder", None, f"native-mxfp8-storage:{global_name}"),
    ]
    assert placeholder_bridge.calls == []


def test_native_mxfp8_metadata_uses_bridge_order_and_global_shapes() -> None:
    task = _task("decoder.layers.0.mlp.linear_fc1.weight")
    bridge = _FakeNativeBridge()
    names = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    bridge.outputs[id(task)] = (
        _record(names[0], (32, 64)),
        _record(names[1], (32, 64)),
    )
    worker = _native_worker([task], bridge)

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 8, "ep_size": 1, "pp_size": 1}
    )

    assert list(metadata) == names
    assert metadata[names[0]]["shape"] == [32, 64]
    assert metadata[names[0]]["components"] == [
        {"role": "weight", "shape": [32, 64], "dtype": "torch.float8_e4m3fn"},
        {"role": "weight_scale", "shape": [32, 2], "dtype": "torch.uint8"},
    ]


def test_native_mxfp8_attention_metadata_preserves_shapes_and_placements() -> None:
    from torch.distributed.tensor.placement_types import Shard

    from nemo_rl.weight_sync.nccl_reshard_utils import (
        build_nccl_reshard_refit_info,
    )

    qkv_task = _task("decoder.layers.0.self_attention.linear_qkv.weight")
    output_task = _task("decoder.layers.0.self_attention.linear_proj.weight")
    names_and_shapes = [
        ("model.layers.0.self_attn.q_proj.weight", (128, 64), 0),
        ("model.layers.0.self_attn.k_proj.weight", (32, 64), 0),
        ("model.layers.0.self_attn.v_proj.weight", (32, 64), 0),
        ("model.layers.0.self_attn.o_proj.weight", (64, 128), 1),
    ]
    bridge = _FakeNativeBridge()
    bridge.outputs[id(qkv_task)] = tuple(
        _record(name, shape) for name, shape, _ in names_and_shapes[:3]
    )
    bridge.outputs[id(output_task)] = (
        _record(names_and_shapes[3][0], names_and_shapes[3][1]),
    )
    worker = _native_worker([qkv_task, output_task], bridge)

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 2, "ep_size": 1, "pp_size": 1}
    )
    refit_info = build_nccl_reshard_refit_info(
        metadata,
        train_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
        gen_parallelism={"tp_size": 4, "ep_size": 1, "pp_size": 1},
        train_world_size=2,
        gen_world_size=4,
    )
    params = {
        param["name"]: param
        for layer_name in refit_info["layer_names"]
        for param in refit_info["per_layer_params"][layer_name]
    }

    assert list(metadata) == [name for name, _, _ in names_and_shapes]
    for name, shape, shard_dim in names_and_shapes:
        assert metadata[name]["shape"] == list(shape)
        assert metadata[name]["components"][1]["shape"] == [
            *shape[:-1],
            shape[-1] // 32,
        ]
        assert params[name]["global_shape"] == shape
        for component in params[name]["components"]:
            assert any(
                isinstance(placement, Shard) and placement.dim == shard_dim
                for placement in component["src_placements"]
            )
            assert any(
                isinstance(placement, Shard) and placement.dim == shard_dim
                for placement in component["dst_placements"]
            )


def test_native_mxfp8_metadata_uses_task_broadcast_for_pp_placeholder() -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    broadcasts: list[tuple[object, str]] = []

    def broadcast(value: object, cache_key: str) -> object:
        broadcasts.append((value, cache_key))
        return ("ok", [(name, [64, 32])])

    task = _task(
        "decoder.layers.0.mlp.linear_fc2.weight", owned=False, broadcast=broadcast
    )
    bridge = _FakeNativeBridge()
    worker = _native_worker([task], bridge)

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 1, "ep_size": 1, "pp_size": 2}
    )

    assert list(metadata) == [name]
    assert broadcasts == [
        (None, "native-mxfp8-shape:decoder.layers.0.mlp.linear_fc2.weight")
    ]
    assert bridge.calls == []


def test_native_mxfp8_metadata_error_broadcasts_before_all_pp_ranks_raise() -> None:
    global_name = "decoder.layers.0.self_attention.linear_qkv.weight"
    broadcasted: list[object] = []
    calls: list[tuple[str, object, str]] = []

    def owner_broadcast(value: object, cache_key: str) -> object:
        pickle.dumps(value)
        calls.append(("owner", value, cache_key))
        broadcasted.append(value)
        return value

    def placeholder_broadcast(value: object, cache_key: str) -> object:
        calls.append(("placeholder", value, cache_key))
        assert value is None
        return broadcasted[0]

    owner_task = _task(global_name, broadcast=owner_broadcast)
    placeholder_task = _task(global_name, owned=False, broadcast=placeholder_broadcast)
    validation_error = RuntimeError("invalid QKV native metadata")
    owner_bridge = _FakeNativeBridge()
    owner_bridge.outputs[id(owner_task)] = validation_error
    placeholder_bridge = _FakeNativeBridge()
    owner_worker = _native_worker([owner_task], owner_bridge)
    placeholder_worker = _native_worker([placeholder_task], placeholder_bridge)

    with pytest.raises(ValueError) as owner_raised:
        owner_worker._build_native_mxfp8_shape_metadata(
            {"tp_size": 2, "ep_size": 1, "pp_size": 2}
        )
    with pytest.raises(ValueError) as placeholder_raised:
        placeholder_worker._build_native_mxfp8_shape_metadata(
            {"tp_size": 2, "ep_size": 1, "pp_size": 2}
        )

    assert str(owner_raised.value) == str(placeholder_raised.value)
    assert "shape metadata" in str(owner_raised.value)
    assert "RuntimeError: invalid QKV native metadata" in str(owner_raised.value)
    assert owner_raised.value.__cause__ is validation_error
    assert placeholder_raised.value.__cause__ is None
    assert calls == [
        (
            "owner",
            ("error", "RuntimeError", "invalid QKV native metadata"),
            f"native-mxfp8-shape:{global_name}",
        ),
        ("placeholder", None, f"native-mxfp8-shape:{global_name}"),
    ]
    assert placeholder_bridge.calls == []


def test_native_mxfp8_metadata_expands_experts_in_deterministic_order() -> None:
    prefix = "model.layers.0.mlp.experts"
    task = _task("decoder.layers.0.mlp.experts.linear_fc2.weight")
    bridge = _FakeNativeBridge()
    bridge.outputs[id(task)] = (
        _record(f"{prefix}.2.down_proj.weight", (64, 32)),
        _record(f"{prefix}.3.down_proj.weight", (64, 32)),
    )
    worker = _native_worker([task], bridge)

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 1, "ep_size": 2, "pp_size": 1}
    )

    assert list(metadata) == [
        f"{prefix}.0.down_proj.weight",
        f"{prefix}.1.down_proj.weight",
        f"{prefix}.2.down_proj.weight",
        f"{prefix}.3.down_proj.weight",
    ]


def test_native_mxfp8_source_map_groups_experts_and_refreshes_live_storage() -> None:
    prefix = "model.layers.0.mlp.experts"
    dense_name = "model.layers.0.mlp.down_proj.weight"
    dense_task = _task("decoder.layers.0.mlp.linear_fc2.weight")
    expert_task = _task("decoder.layers.0.mlp.experts.linear_fc1.weight")
    bridge = _FakeNativeBridge()
    initial_dense = torch.full((4, 32), 1, dtype=torch.float8_e4m3fn)
    initial_experts = [
        torch.full((4, 32), marker, dtype=torch.float8_e4m3fn) for marker in (2, 3)
    ]
    bridge.outputs[id(dense_task)] = (
        _record(dense_name, (4, 32), weight=initial_dense),
    )
    bridge.outputs[id(expert_task)] = tuple(
        _record(f"{prefix}.{index}.gate_proj.weight", (4, 32), weight=weight)
        for index, weight in enumerate(initial_experts)
    )
    worker = _native_worker([dense_task, expert_task], bridge)
    refit_info = _refit_info(
        [
            (dense_name, (4, 32), None),
            (f"{prefix}.gate_proj.weight", (2, 4, 32), "gate_proj"),
        ]
    )

    param_map = worker.build_hf_to_local_param_map(refit_info)
    dense_spec = param_map.get(dense_name, role="weight")
    grouped_spec = param_map.get(f"{prefix}.gate_proj.weight", role="weight")
    assert dense_spec is not None
    assert dense_spec.base is initial_dense
    assert grouped_spec is not None and grouped_spec.pre is not None
    assert torch.equal(
        grouped_spec.pre(grouped_spec.base).buf, torch.stack(initial_experts)
    )

    refreshed_dense = torch.full((4, 32), 4, dtype=torch.float8_e4m3fn)
    refreshed_experts = [
        torch.full((4, 32), marker, dtype=torch.float8_e4m3fn) for marker in (5, 6)
    ]
    bridge.outputs[id(dense_task)] = (
        _record(dense_name, (4, 32), weight=refreshed_dense),
    )
    bridge.outputs[id(expert_task)] = tuple(
        _record(f"{prefix}.{index}.gate_proj.weight", (4, 32), weight=weight)
        for index, weight in enumerate(refreshed_experts)
    )

    worker._refresh_local_native_mxfp8_param_components()

    assert dense_spec.base is refreshed_dense
    assert torch.equal(
        grouped_spec.pre(grouped_spec.base).buf, torch.stack(refreshed_experts)
    )


def test_native_mxfp8_refresh_fails_for_changed_bridge_mapping() -> None:
    task = _task("decoder.layers.0.mlp.linear_fc2.weight")
    name = "model.layers.0.mlp.down_proj.weight"
    bridge = _FakeNativeBridge()
    bridge.outputs[id(task)] = (_record(name, (4, 32)),)
    worker = _native_worker([task], bridge)
    worker.build_hf_to_local_param_map(_refit_info([(name, (4, 32), None)]))
    bridge.outputs[id(task)] = (_record("model.layers.0.mlp.other.weight", (4, 32)),)

    with pytest.raises(ValueError, match="Missing native MXFP8 source"):
        worker._refresh_local_native_mxfp8_param_components()


def test_native_mxfp8_refresh_fails_for_missing_bridge_projection() -> None:
    task = _task("decoder.layers.0.mlp.linear_fc1.weight")
    gate_name = "model.layers.0.mlp.gate_proj.weight"
    up_name = "model.layers.0.mlp.up_proj.weight"
    bridge = _FakeNativeBridge()
    bridge.outputs[id(task)] = (
        _record(gate_name, (4, 32)),
        _record(up_name, (4, 32)),
    )
    worker = _native_worker([task], bridge)
    worker.build_hf_to_local_param_map(
        _refit_info([(gate_name, (4, 32), None), (up_name, (4, 32), None)])
    )
    bridge.outputs[id(task)] = (_record(gate_name, (4, 32)),)

    with pytest.raises(ValueError, match="Missing refreshed native MXFP8 source"):
        worker._refresh_local_native_mxfp8_param_components()
