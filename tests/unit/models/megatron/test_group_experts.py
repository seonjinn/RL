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
                    "refit_transform": "mxfp8",
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 64, 32],
                    "grouped_expert_proj": "gate_proj",
                    "refit_transform": "mxfp8",
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    direct_ctx = pmap.get("model.layers.0.mlp.down_proj.weight").pre(direct)
    grouped_spec = pmap.get(f"{prefix}.gate_proj.weight")
    grouped_ctx = grouped_spec.pre(grouped_spec.base)

    assert direct_ctx.buf.dtype == torch.float8_e4m3fn
    assert direct_ctx.extra["scale_buf"].shape == (32, 2)
    assert direct_ctx.extra["scale_buf"].dtype == torch.uint8
    assert grouped_ctx.buf.dtype == torch.float8_e4m3fn
    assert grouped_ctx.buf.shape == (2, 64, 32)
    assert grouped_ctx.extra["scale_buf"].shape == (2, 64, 1)
    assert grouped_ctx.extra["scale_buf"].dtype == torch.uint8


def test_nccl_reshard_refit_transfers_mxfp8_value_then_scale(monkeypatch):
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
        RefitCtx,
    )

    value = torch.empty(32, 64, dtype=torch.float8_e4m3fn)
    scale = torch.empty(32, 2, dtype=torch.uint8)
    src_mesh = object()
    dst_mesh = object()
    value_src_placements = [object()]
    value_dst_placements = [object()]
    scale_src_placements = [object()]
    scale_dst_placements = [object()]
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
    worker.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": (32, 64),
                    "src_mesh_info": src_mesh,
                    "src_placements": value_src_placements,
                    "dst_mesh_info": dst_mesh,
                    "dst_placements": value_dst_placements,
                    "refit_transform": "mxfp8",
                    "scale_global_shape": (32, 2),
                    "scale_src_placements": scale_src_placements,
                    "scale_dst_placements": scale_dst_placements,
                }
            ]
        },
    }
    worker.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            name: LocalParamSpec(
                base=None,
                pre=lambda _: RefitCtx(
                    buf=value,
                    extra={"scale_buf": scale},
                ),
            )
        }
    )
    worker._broadcast_misc_params_packed = lambda kv_scales=None: None

    worker.nccl_reshard_refit()

    assert calls[0][0] is value
    assert calls[1][0] is scale
    assert [call[1] for call in calls] == [(32, 64), (32, 2)]
    assert calls[0][2:] == (
        src_mesh,
        value_src_placements,
        None,
        dst_mesh,
        value_dst_placements,
        group,
        stream,
    )
    assert calls[1][2:] == (
        src_mesh,
        scale_src_placements,
        None,
        dst_mesh,
        scale_dst_placements,
        group,
        stream,
    )
