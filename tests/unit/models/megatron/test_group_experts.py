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

"""Unit test for the train-side expert stacking (``_group_experts``).

``_group_experts`` (``MegatronPolicyWorkerImpl``) stacks this rank's local
per-expert tensors for one projection into ``[E_local, ...]``.  It doesn't use
``self`` and operates on plain tensors, so a dummy ``self`` + CPU tensors suffice.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

# megatron_policy_worker imports both megatron.core and megatron.bridge at
# module top, so guard on both: an env can have megatron.core but not
# megatron.bridge, and importing this test module would otherwise raise a
# collection error (not skip) in non-mcore lanes.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


class _FakeMXFP8Tensor:
    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.shape = data.shape
        self._metadata = {
            "rowwise_data": data,
            "rowwise_scale_inv": scale,
            "with_gemm_swizzled_scales": False,
        }

    def get_metadata(self) -> dict[str, object]:
        return self._metadata


def _native_worker(tasks: list[Any]) -> MegatronPolicyWorkerImpl:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.fp8_cfg = {"fp8_param": True, "fp8_recipe": "mxfp8"}
    worker.cfg = {
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {"precision": "fp8", "is_mx": True},
        }
    }
    worker.dtype = torch.bfloat16
    worker.refit_conversion_tasks = tasks
    worker._native_grouped_mxfp8_tasks = []
    return worker


def _native_components() -> list[dict[str, object]]:
    return [
        {"role": "weight", "global_shape": [], "dtype": "torch.float8_e4m3fn"},
        {"role": "weight_scale", "global_shape": [], "dtype": "torch.uint8"},
    ]


def _group(proj, grouped_name, expert_groups):
    # _group_experts ignores self; pass a dummy.
    return MegatronPolicyWorkerImpl._group_experts(
        SimpleNamespace(), proj, grouped_name, expert_groups
    )


def test_group_experts_stacks_in_order():
    prefix = "model.layers.0.mlp.experts"
    e0 = torch.randn(1536, 4096)
    e1 = torch.randn(1536, 4096)
    e2 = torch.randn(1536, 4096)
    groups = {(prefix, "gate_proj"): [e0, e1, e2]}
    out = _group("gate_proj", f"{prefix}.gate_proj.weight", groups)
    assert out.shape == (3, 1536, 4096)
    # Order preserved (expert 0 first).
    assert torch.equal(out[0], e0)
    assert torch.equal(out[1], e1)
    assert torch.equal(out[2], e2)


def test_group_experts_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [torch.randn(8, 8)]}
    with pytest.raises(AssertionError):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_group_experts_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (train/src side) — folds this rank's local
# shards (_iter_local_hf_param_shards) into LocalParamSpecs.  Fake the shard
# iterator; _build_expert_groups / _group_experts run for real.
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_train_side():
    from nemo_rl.weight_sync.nccl_reshard_utils import HFToLocalParamMap

    w = object.__new__(MegatronPolicyWorkerImpl)  # no __init__ / no megatron state
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(8, 16)  # a dense FFN down_proj local shard view
    e0 = torch.randn(128, 16)  # this rank's local expert 0 gate_proj
    e1 = torch.randn(128, 16)  # local expert 1 gate_proj
    w._iter_local_hf_param_shards = lambda: [
        ("model.layers.0.mlp.down_proj.weight", direct),
        (f"{prefix}.0.gate_proj.weight", e0),
        (f"{prefix}.1.gate_proj.weight", e1),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [8, 16],
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 128, 16],
                    "grouped_expert_proj": "gate_proj",
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)

    # Direct: base is the live local view, sent as-is (no hooks).
    d = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert d.base is direct and d.pre is None and d.post is None

    # Grouped expert: pre stacks this rank's per-expert views into [E_local, ...]
    # fresh each refit (base unused — the views are captured in the hook).
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert g.pre is not None
    ctx = g.pre(g.base)
    assert ctx.buf.shape == (2, 128, 16)
    assert torch.equal(ctx.buf[0], e0) and torch.equal(ctx.buf[1], e1)


def test_native_mxfp8_dense_fused_gate_up_exposes_compact_roles():
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping

    fused_data = (
        torch.arange(16 * 64, dtype=torch.int64).to(torch.uint8).reshape(16, 64)
    )
    fused_scale = (
        torch.arange(128 * 4, dtype=torch.int64).to(torch.uint8).reshape(128, 4)
    )
    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    task = SimpleNamespace(
        mapping=mapping,
        param_weight=_FakeMXFP8Tensor(fused_data, fused_scale),
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
    )
    worker = _native_worker([task])
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "components": _native_components(),
                }
                for name in mapping.hf_param.values()
            ]
        },
    }

    source_map = worker.build_hf_to_local_param_map(refit_info)

    gate_name = mapping.hf_param["gate"]
    up_name = mapping.hf_param["up"]
    assert source_map.get(gate_name, role="weight").base.shape == (8, 64)
    assert source_map.get(gate_name, role="weight_scale").base.shape == (8, 2)
    assert torch.equal(
        source_map.get(gate_name, role="weight").base.view(torch.uint8),
        fused_data[:8],
    )
    assert torch.equal(
        source_map.get(up_name, role="weight_scale").base,
        fused_scale[:16, :2].reshape(16, 2)[8:],
    )


def test_native_mxfp8_component_iterator_leaves_misc_parameters_untouched():
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    task = SimpleNamespace(
        mapping=AutoMapping(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ),
        param_weight=torch.empty(64, 64, dtype=torch.bfloat16),
        global_param_name="decoder.layers.0.self_attention.linear_proj.weight",
    )
    worker = _native_worker([task])

    assert list(worker._iter_local_native_mxfp8_param_components()) == []


def test_native_mxfp8_shared_expert_is_filtered_before_extraction(monkeypatch):
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    mapping = GatedMLPMapping(
        "decoder.layers.0.mlp.shared_expert.linear_fc1.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
    )
    task = WeightConversionTask(
        param_name=mapping.megatron_param,
        global_param_name=mapping.megatron_param,
        mapping=mapping,
        param_weight=torch.empty(16, 64, dtype=torch.bfloat16),
    )
    worker = _native_worker([task])
    monkeypatch.setattr(
        worker_module,
        "extract_native_mxfp8_components",
        lambda _tensor: pytest.fail("misc shared expert reached native extraction"),
    )

    assert list(worker._iter_local_native_mxfp8_param_components()) == []


def test_native_mxfp8_pregrouped_gate_up_splits_expert_dimension_independently():
    from megatron.bridge.models.conversion.param_mapping import FusedGatedExpertMapping

    prefix = "model.layers.0.mlp.experts"
    data = torch.arange(2 * 8 * 64, dtype=torch.int64).to(torch.uint8).reshape(2, 8, 64)
    scale = torch.arange(128 * 4, dtype=torch.int64).to(torch.uint8).reshape(128, 4)
    task = SimpleNamespace(
        mapping=FusedGatedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight",
            f"{prefix}.gate_up_proj",
        ),
        param_weight=_FakeMXFP8Tensor(data, scale),
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
    )
    worker = _native_worker([task])
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": f"{prefix}.{proj}.weight",
                    "grouped_expert_proj": proj,
                    "components": _native_components(),
                }
                for proj in ("gate_proj", "up_proj")
            ]
        },
    }

    source_map = worker.build_hf_to_local_param_map(refit_info)

    gate = source_map.get(f"{prefix}.gate_proj.weight", role="weight").base
    up_scale = source_map.get(f"{prefix}.up_proj.weight", role="weight_scale").base
    assert gate.shape == (2, 4, 64)
    assert up_scale.shape == (2, 4, 2)
    assert torch.equal(gate.view(torch.uint8), data[:, :4])
    assert torch.equal(
        up_scale,
        scale[:16, :2].reshape(2, 8, 2)[:, 4:],
    )


def test_native_mxfp8_numbered_experts_group_every_role_in_numeric_order():
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )

    prefix = "model.layers.0.mlp.experts"
    tasks = []
    for expert, marker in ((10, 100), (2, 20)):
        fc1_data = torch.empty(8, 64, dtype=torch.uint8)
        fc1_data[:4].fill_(marker + 1)
        fc1_data[4:].fill_(marker + 2)
        fc1_scale = torch.empty(128, 4, dtype=torch.uint8)
        fc1_scale[:4, :2].fill_(marker + 3)
        fc1_scale[4:8, :2].fill_(marker + 4)
        tasks.append(
            SimpleNamespace(
                mapping=FusedGatedExpertMapping(
                    f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
                    f"{prefix}.gate_up_proj",
                ),
                param_weight=_FakeMXFP8Tensor(fc1_data, fc1_scale),
                global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",
            )
        )
        tasks.append(
            SimpleNamespace(
                mapping=FusedExpertMapping(
                    f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
                    f"{prefix}.down_proj",
                ),
                param_weight=_FakeMXFP8Tensor(
                    torch.full((64, 32), marker + 5, dtype=torch.uint8),
                    torch.full((128, 4), marker + 6, dtype=torch.uint8),
                ),
                global_param_name=f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}",
            )
        )

    worker = _native_worker(tasks)
    params = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        params.append(
            {
                "name": f"{prefix}.{proj}.weight",
                "grouped_expert_proj": proj,
                "components": _native_components(),
            }
        )
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": params},
    }

    source_map = worker.build_hf_to_local_param_map(refit_info)

    expected = {
        "gate_proj": {"weight": (21, 101), "weight_scale": (23, 103)},
        "up_proj": {"weight": (22, 102), "weight_scale": (24, 104)},
        "down_proj": {"weight": (25, 105), "weight_scale": (26, 106)},
    }
    for proj, roles in expected.items():
        name = f"{prefix}.{proj}.weight"
        for role, markers in roles.items():
            spec = source_map.get(name, role=role)
            grouped = spec.pre(spec.base).buf
            assert grouped.shape[0] == 2
            storage = grouped.view(torch.uint8) if role == "weight" else grouped
            assert (
                tuple(int(storage[index].flatten()[0]) for index in range(2)) == markers
            )


def test_native_grouped_task_builder_uses_real_task_contract_and_pp_placeholders(
    monkeypatch,
):
    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        FusedGatedExpertMapping,
    )
    from megatron.core import fp8_utils

    local_name = "decoder.layers.0.mlp.experts.linear_fc1.weight"
    remote_name = "decoder.layers.1.mlp.experts.linear_fc1.weight"
    grouped = torch.nn.Parameter(torch.empty(2, 8, 64), requires_grad=False)

    class _Model:
        config = SimpleNamespace(moe_single_grouped_weight=True)

        def named_parameters(self):
            return [(local_name, grouped)]

    class _Registry:
        def megatron_to_hf_lookup(self, name):
            layer = name.split(".")[2]
            return FusedGatedExpertMapping(
                name,
                f"model.layers.{layer}.mlp.experts.gate_up_proj",
            )

    bridge = SimpleNamespace(
        _unwrap_name=lambda name: name,
        _megatron_global_param_names_all_pp_ranks=lambda _models: [
            local_name,
            remote_name,
        ],
        mapping_registry=lambda: _Registry(),
    )
    worker = _native_worker([])
    worker.model = _Model()
    worker.megatron_bridge = SimpleNamespace(_model_bridge=bridge)
    monkeypatch.setattr(model_bridge, "_megatron_local_name_to_global", lambda *_: _[2])
    monkeypatch.setattr(
        fp8_utils, "is_grouped_mxfp8tensor", lambda param: param is grouped
    )

    tasks = worker._build_native_grouped_mxfp8_tasks()

    assert all(isinstance(task, WeightConversionTask) for task in tasks)
    assert [task.global_param_name for task in tasks] == [local_name, remote_name]
    assert tasks[0].param_name == local_name and tasks[0].param_weight is grouped
    assert tasks[1].param_name == remote_name and tasks[1].param_weight is None


def test_native_mxfp8_single_grouped_weight_refreshes_cached_members(monkeypatch):
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )
    from megatron.core import fp8_utils

    prefix = "model.layers.0.mlp.experts"
    fc1_grouped = torch.nn.Parameter(torch.empty(2, 8, 64), requires_grad=False)
    fc2_grouped = torch.nn.Parameter(torch.empty(2, 64, 32), requires_grad=False)
    fc1_members = []
    fc2_members = []
    for marker in (11, 41):
        data = torch.empty(8, 64, dtype=torch.uint8)
        data[:4].fill_(marker)
        data[4:].fill_(marker + 1)
        scale = torch.empty(128, 4, dtype=torch.uint8)
        scale[:4, :2].fill_(marker + 2)
        scale[4:8, :2].fill_(marker + 3)
        fc1_members.append(_FakeMXFP8Tensor(data, scale))
        fc2_members.append(
            _FakeMXFP8Tensor(
                torch.full((64, 32), marker + 4, dtype=torch.uint8),
                torch.full((128, 4), marker + 5, dtype=torch.uint8),
            )
        )

    member_calls = []
    monkeypatch.setattr(
        fp8_utils,
        "is_grouped_mxfp8tensor",
        lambda param: param is fc1_grouped or param is fc2_grouped,
    )

    def get_members(param, *, create_if_missing):
        member_calls.append((id(param), create_if_missing))
        return fc1_members if param is fc1_grouped else fc2_members

    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", get_members)
    worker = _native_worker([])
    worker._native_grouped_mxfp8_tasks = [
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
            mapping=FusedGatedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                f"{prefix}.gate_up_proj",
            ),
            param_weight=fc1_grouped,
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
        ),
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
            mapping=FusedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc2.weight0",
                f"{prefix}.down_proj",
            ),
            param_weight=fc2_grouped,
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
        ),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": f"{prefix}.{proj}.weight",
                    "grouped_expert_proj": proj,
                    "components": _native_components(),
                }
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
        },
    }

    source_map = worker.build_hf_to_local_param_map(refit_info)

    assert member_calls == []
    expected = {
        "gate_proj": {"weight": (11, 41), "weight_scale": (13, 43)},
        "up_proj": {"weight": (12, 42), "weight_scale": (14, 44)},
        "down_proj": {"weight": (15, 45), "weight_scale": (16, 46)},
    }
    for proj, roles in expected.items():
        name = f"{prefix}.{proj}.weight"
        for role, markers in roles.items():
            spec = source_map.get(name, role=role)
            assert spec.base is None and spec.pre is not None
            grouped = spec.pre(spec.base).buf
            storage = grouped.view(torch.uint8) if role == "weight" else grouped
            assert (
                tuple(int(storage[index].flatten()[0]) for index in range(2)) == markers
            )

    gate_spec = source_map.get(f"{prefix}.gate_proj.weight", role="weight")
    scale_spec = source_map.get(f"{prefix}.gate_proj.weight", role="weight_scale")
    first_gate = gate_spec.pre(gate_spec.base).buf
    first_scale = scale_spec.pre(scale_spec.base).buf
    for index, member in enumerate(fc1_members):
        data = torch.full((8, 64), 71 + index, dtype=torch.uint8)
        scale = torch.full((128, 4), 81 + index, dtype=torch.uint8)
        member._metadata["rowwise_data"] = data
        member._metadata["rowwise_scale_inv"] = scale

    second_gate = gate_spec.pre(gate_spec.base).buf
    second_scale = scale_spec.pre(scale_spec.base).buf

    assert second_gate.data_ptr() != first_gate.data_ptr()
    assert second_scale.data_ptr() != first_scale.data_ptr()
    assert tuple(
        int(second_gate[i].view(torch.uint8).flatten()[0]) for i in range(2)
    ) == (
        71,
        72,
    )
    assert tuple(int(second_scale[i].flatten()[0]) for i in range(2)) == (81, 82)
    assert member_calls[-4:] == [
        (id(fc1_grouped), False),
        (id(fc1_grouped), False),
        (id(fc1_grouped), False),
        (id(fc1_grouped), False),
    ]


def test_native_mxfp8_shape_metadata_uses_task_shapes_and_tp_ep_topology(
    monkeypatch,
):
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
        GatedMLPMapping,
    )
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    prefix = "model.layers.0.mlp.experts"
    tasks = [
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.weight",
            global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
            mapping=GatedMLPMapping(
                "decoder.layers.0.mlp.linear_fc1.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ),
            param_weight=torch.empty(8, 64),
        ),
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
            mapping=FusedGatedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                f"{prefix}.gate_up_proj",
            ),
            param_weight=torch.empty(2, 8, 64),
        ),
        WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
            mapping=FusedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc2.weight0",
                f"{prefix}.down_proj",
            ),
            param_weight=torch.empty(2, 64, 32),
        ),
    ]
    worker = _native_worker(tasks)
    worker.model = SimpleNamespace(config=SimpleNamespace(num_moe_experts=8))
    monkeypatch.setattr(
        worker_module, "broadcast_obj_from_pp_rank", lambda value: value
    )

    metadata = worker._build_native_mxfp8_shape_metadata({"tp_size": 2, "ep_size": 4})

    assert list(metadata) == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
        f"{prefix}.down_proj.weight",
    ]
    assert metadata["model.layers.0.mlp.gate_proj.weight"]["shape"] == [8, 64]
    assert metadata["model.layers.0.mlp.up_proj.weight"]["shape"] == [8, 64]
    assert metadata[f"{prefix}.gate_proj.weight"]["shape"] == [8, 4, 64]
    assert metadata[f"{prefix}.up_proj.weight"]["shape"] == [8, 4, 64]
    assert metadata[f"{prefix}.down_proj.weight"]["shape"] == [8, 64, 32]
    assert metadata[f"{prefix}.down_proj.weight"]["components"][1]["shape"] == [
        8,
        64,
        1,
    ]
