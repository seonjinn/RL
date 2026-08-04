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
                    "components": [{"role": "weight"}],
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 128, 16],
                    "grouped_expert_proj": "gate_proj",
                    "components": [{"role": "weight"}],
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


def test_build_mxfp8_source_specs_quantize_direct_and_grouped_once():
    w = object.__new__(MegatronPolicyWorkerImpl)
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(32, 64, dtype=torch.bfloat16)
    e0 = torch.randn(64, 32, dtype=torch.bfloat16)
    e1 = torch.randn(64, 32, dtype=torch.bfloat16)
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
                    "global_shape": [32, 64],
                    "components": [
                        {"role": "weight"},
                        {"role": "weight_scale"},
                    ],
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 64, 32],
                    "grouped_expert_proj": "gate_proj",
                    "components": [
                        {"role": "weight"},
                        {"role": "weight_scale"},
                    ],
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    direct_ctx = pmap.get("model.layers.0.mlp.down_proj.weight").pre(direct)
    grouped_spec = pmap.get(f"{prefix}.gate_proj.weight")
    grouped_ctx = grouped_spec.pre(grouped_spec.base)

    assert direct_ctx.buf.dtype == torch.float8_e4m3fn
    assert direct_ctx.tensors_for_transfer()[1].shape == (32, 2)
    assert direct_ctx.tensors_for_transfer()[1].dtype == torch.uint8
    assert grouped_ctx.buf.dtype == torch.float8_e4m3fn
    assert grouped_ctx.buf.shape == (2, 64, 32)
    assert grouped_ctx.tensors_for_transfer()[1].shape == (2, 64, 1)
    assert grouped_ctx.tensors_for_transfer()[1].dtype == torch.uint8


def test_refit_ctx_distinguishes_default_from_explicit_empty_transfer_tuple():
    from nemo_rl.weight_sync.nccl_reshard_utils import RefitCtx

    value = torch.empty(1)

    assert RefitCtx(buf=value).tensors_for_transfer() == (value,)
    assert RefitCtx(buf=value, transfer_tensors=()).tensors_for_transfer() == ()


@pytest.mark.parametrize("component_count", [1, 2, 4])
def test_nccl_reshard_refit_transfers_ordered_components(monkeypatch, component_count):
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
        RefitCtx,
    )

    tensors = tuple(torch.empty(2, index + 1) for index in range(component_count))
    src_mesh = object()
    dst_mesh = object()
    component_src_placements = [[object()] for _ in tensors]
    component_dst_placements = [[object()] for _ in tensors]
    group = object()
    stream = object()
    calls = []

    def record_xferdtensor(
        src_tensor,
        actual_src_mesh,
        actual_src_placements,
        dst_tensor,
        actual_dst_mesh,
        actual_dst_placements,
        actual_group,
        actual_stream,
    ):
        calls.append(
            (
                src_tensor._local_tensor,
                tuple(src_tensor.shape),
                actual_src_mesh,
                actual_src_placements,
                dst_tensor,
                actual_dst_mesh,
                actual_dst_placements,
                actual_group,
                actual_stream,
            )
        )

    monkeypatch.setattr(
        "nemo_rl.weight_sync.xferdtensor.xferdtensor", record_xferdtensor
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    name = "model.layers.0.mlp.down_proj.weight"
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.my_pp_stage = 0
    worker.pp_comm_group = group
    components = [
        {
            "role": f"component_{index}",
            "global_shape": tuple(tensor.shape),
            "src_placements": component_src_placements[index],
            "dst_placements": component_dst_placements[index],
        }
        for index, tensor in enumerate(tensors)
    ]
    worker.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "src_mesh_info": src_mesh,
                    "dst_mesh_info": dst_mesh,
                    "components": components,
                }
            ]
        },
    }
    spec = (
        LocalParamSpec(base=tensors[0])
        if component_count == 1
        else LocalParamSpec(
            base=None,
            pre=lambda _: RefitCtx(
                buf=tensors[0],
                transfer_tensors=tensors,
            ),
        )
    )
    worker.hf_to_local_param_map = HFToLocalParamMap(specs={name: spec})
    worker._broadcast_misc_params_packed = lambda kv_scales=None: None

    worker.nccl_reshard_refit()

    assert [call[0] for call in calls] == list(tensors)
    assert [call[1] for call in calls] == [tuple(tensor.shape) for tensor in tensors]
    for index, call in enumerate(calls):
        assert call[2:] == (
            src_mesh,
            component_src_placements[index],
            None,
            dst_mesh,
            component_dst_placements[index],
            group,
            stream,
        )


def test_nccl_reshard_refit_records_every_temporary_component_on_transfer_stream(
    monkeypatch,
):
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
        RefitCtx,
    )

    stream = object()
    transfer_calls = []

    class CudaLikeTensor:
        dtype = torch.float32
        device = torch.device("cuda")
        is_cuda = True

        def __init__(self):
            self.recorded_streams = []

        def record_stream(self, actual_stream):
            self.recorded_streams.append(actual_stream)

    tensors = tuple(CudaLikeTensor() for _ in range(4))
    components = [
        {
            "role": f"component_{index}",
            "global_shape": (2, index + 1),
            "src_placements": [object()],
            "dst_placements": [object()],
        }
        for index in range(len(tensors))
    ]
    name = "model.layers.0.mlp.down_proj.weight"
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.my_pp_stage = 0
    worker.pp_comm_group = object()
    worker.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "src_mesh_info": object(),
                    "dst_mesh_info": object(),
                    "components": components,
                }
            ]
        },
    }
    worker.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            name: LocalParamSpec(
                base=None,
                pre=lambda _: RefitCtx(
                    buf=tensors[0],
                    transfer_tensors=tensors,
                ),
            )
        }
    )
    worker._broadcast_misc_params_packed = lambda kv_scales=None: None

    monkeypatch.setattr(
        "nemo_rl.weight_sync.xferdtensor.xferdtensor",
        lambda src, *_args, **_kwargs: transfer_calls.append(src._local_tensor),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    worker.nccl_reshard_refit()

    assert transfer_calls == list(tensors)
    assert [tensor.recorded_streams for tensor in tensors] == [[stream]] * len(tensors)


@pytest.mark.parametrize(
    ("case", "component_count", "transfer_count", "error"),
    [
        ("explicit_empty", 1, 0, "component count"),
        ("empty_components", 0, 1, "nonempty component list"),
        ("underfilled", 2, 1, "component count"),
        ("overfilled", 1, 2, "component count"),
    ],
)
def test_nccl_reshard_refit_rejects_invalid_component_counts_before_transfer_or_post(
    monkeypatch,
    case,
    component_count,
    transfer_count,
    error,
):
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
        RefitCtx,
    )

    calls = []
    tensors = tuple(torch.empty(2, 2) for _ in range(max(transfer_count, 1)))
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.my_pp_stage = 0
    worker.pp_comm_group = object()
    worker.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "src_mesh_info": object(),
                    "dst_mesh_info": object(),
                    "components": [
                        {
                            "role": f"component_{index}",
                            "global_shape": (2, 2),
                            "src_placements": [object()],
                            "dst_placements": [object()],
                        }
                        for index in range(component_count)
                    ],
                }
            ]
        },
    }
    worker.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            "model.layers.0.mlp.down_proj.weight": LocalParamSpec(
                base=tensors[0],
                pre=lambda _: RefitCtx(
                    buf=tensors[0],
                    transfer_tensors=tensors[:transfer_count],
                ),
                post=lambda _: calls.append("post"),
            )
        }
    )
    worker._broadcast_misc_params_packed = lambda kv_scales=None: None

    monkeypatch.setattr(
        "nemo_rl.weight_sync.xferdtensor.xferdtensor",
        lambda *_args, **_kwargs: calls.append("transfer"),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())

    with pytest.raises(ValueError, match=error):
        worker.nccl_reshard_refit()

    assert calls == [], case
