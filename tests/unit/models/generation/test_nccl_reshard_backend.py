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

"""Unit tests for the vLLM-side nccl_reshard refit mapping (CPU, no GPU).

Covers the FFN-only bulk path in ``nemo_rl/models/generation/vllm/vllm_backend.py``
(``_build_hf_to_gen_backend_mapping`` + ``build_hf_to_local_param_map``), driven by
a synthetic ``refit_info`` and a fake ``named_parameters()`` (no real vLLM model,
no GPU).

``vllm_backend`` does ``import vllm`` at module top, so these are vllm-marked and
skipped where vllm is unavailable.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")  # module-top `import vllm` in vllm_backend

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (  # noqa: E402
    HFToLocalParamMap,
    MeshInfo,
)
from nemo_rl.weight_sync.refit_transforms import (  # noqa: E402
    RefitTransformPlan,
    TransformComponentSpec,
    build_plan_agreement,
)

pytestmark = pytest.mark.vllm


# --------------------------------------------------------------------------
# _build_hf_to_gen_backend_mapping
# --------------------------------------------------------------------------
def _make_ext(vllm_params):
    """A VllmInternalWorkerExtension whose model exposes ``vllm_params``."""
    ext = VllmInternalWorkerExtension()  # no __init__
    # named_modules() is consulted to detect the FusedMoE backend (w13 layout);
    # an empty module map -> no match -> standard [gate; up] layout (the case
    # these tests assert).  See _build_hf_to_gen_backend_mapping.
    model = SimpleNamespace(
        named_parameters=lambda: list(vllm_params.items()),
        named_modules=lambda: [],
    )
    ext.model_runner = SimpleNamespace(model=model)
    return ext


def _param(*shape):
    return torch.empty(*shape)


def _component(role, shape, dtype, *, src_placements=None, dst_placements=None):
    component = {
        "role": role,
        "global_shape": tuple(shape),
        "dtype": str(dtype),
    }
    if src_placements is not None:
        component["src_placements"] = src_placements
    if dst_placements is not None:
        component["dst_placements"] = dst_placements
    return component


def _identity_components(shape, dtype=torch.float32, **kwargs):
    return [_component("weight", shape, dtype, **kwargs)]


def _mxfp8_components(shape, **kwargs):
    scale_shape = (*shape[:-1], shape[-1] // 32)
    return [
        _component("weight", shape, torch.float8_e4m3fn, **kwargs),
        _component("weight_scale", scale_shape, torch.uint8, **kwargs),
    ]


def test_build_mapping_ffn_only():
    # Downsized bulk path: only FFN gate/up/down reach the resolver.
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 4,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                # Dense MLP: gate/up -> gate_up_proj (merge), down -> direct.
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [256, H],
                    "components": _identity_components((256, H)),
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": [256, H],
                    "components": _identity_components((256, H)),
                },
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                    "components": _identity_components((H, 256)),
                },
                # MoE experts: gate/up -> w13 halves, down -> w2.
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "gate_proj",
                    "components": _identity_components((E, 128, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "up_proj",
                    "components": _identity_components((E, 128, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "grouped_expert_proj": "down_proj",
                    "components": _identity_components((E, H, 128)),
                },
            ]
        },
    }
    gate_up = _param(128, H)  # 256*2/4
    down = _param(H, 64)  # 256/4 (row-parallel local)
    w13 = _param(E, 2 * Pl, H)  # gated: gate||up on intermediate axis (dim 1)
    w2 = _param(E, H, Pl)
    vllm_params = {
        "model.layers.0.mlp.gate_up_proj.weight": gate_up,
        "model.layers.0.mlp.down_proj.weight": down,
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)

    # Dense gate/up -> gate_up_proj (dim-0 sub-slices)
    assert mapping["model.layers.0.mlp.gate_proj.weight"] == (gate_up, (slice(0, 64),))
    assert mapping["model.layers.0.mlp.up_proj.weight"] == (gate_up, (slice(64, 128),))
    # Dense down -> direct 1:1
    assert mapping["model.layers.0.mlp.down_proj.weight"] == (down, None)
    # Grouped expert gate/up -> w13 halves (dim-1 region); down -> w2 direct
    assert mapping["model.layers.0.mlp.experts.gate_proj.weight"] == (
        w13,
        (slice(None), slice(0, Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (
        w13,
        (slice(None), slice(Pl, 2 * Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_non_gated_expert_up_is_direct():
    # Non-gated MoE (no gate_proj present): up_proj maps 1:1 to w13 (no slice).
    H, E = 16, 2
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 64, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 64],
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = _param(E, 64, H)
    w2 = _param(E, H, 64)
    vllm_params = {
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (w13, None)
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_resolves_routed_experts_submodule():
    # vLLM 0.25 hangs the fused-MoE expert weights off a nested
    # ``routed_experts`` submodule (RoutedExperts is an nn.Module assigned as
    # MoERunner.routed_experts), so named_parameters() reports
    # ``...experts.routed_experts.w13_weight``.  The name built from the HF
    # side has no such segment, and an unresolved grouped expert is a hard
    # ValueError -- so without the flattened index this raises and every MoE
    # model fails to refit over nccl_reshard.
    H, E, Pl = 16, 2, 32
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, Pl, H],
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, Pl, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, Pl],
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = _param(E, 2 * Pl, H)
    w2 = _param(E, H, Pl)
    vllm_params = {
        "model.layers.0.mlp.experts.routed_experts.w13_weight": w13,
        "model.layers.0.mlp.experts.routed_experts.w2_weight": w2,
    }
    mapping = _make_ext(vllm_params)._build_hf_to_gen_backend_mapping(refit_info)

    assert mapping["model.layers.0.mlp.experts.gate_proj.weight"] == (
        w13,
        (slice(None), slice(0, Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.up_proj.weight"] == (
        w13,
        (slice(None), slice(Pl, 2 * Pl), slice(None)),
    )
    assert mapping["model.layers.0.mlp.experts.down_proj.weight"] == (w2, None)


def test_build_mapping_unmapped_param_raises():
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.some_unknown_module.weight",
                    "global_shape": [8, 8],
                },
            ]
        },
    }
    ext = _make_ext({"model.embed_tokens.weight": _param(8, 8)})
    with pytest.raises(ValueError):
        ext._build_hf_to_gen_backend_mapping(refit_info)


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (the unified interface) + RefitCtx pre/post
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_specs_and_roundtrip():
    # FFN-only: dense gate/up (merge) + down (direct), MoE experts (w13/w2).
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 4,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [256, H],
                    "components": _identity_components((256, H)),
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": [256, H],
                    "components": _identity_components((256, H)),
                },
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                    "components": _identity_components((H, 256)),
                },
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "gate_proj",
                    "components": _identity_components((E, 128, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "up_proj",
                    "components": _identity_components((E, 128, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "grouped_expert_proj": "down_proj",
                    "components": _identity_components((E, H, 128)),
                },
            ]
        },
    }
    gate_up = _param(128, H)  # dense gate||up, 256*2/4
    down = _param(H, 64)  # dense down (row-parallel local)
    w13 = _param(E, 2 * Pl, H)
    w2 = _param(E, H, Pl)
    ext = _make_ext(
        {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.down_proj.weight": down,
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w2_weight": w2,
        }
    )

    pmap = ext.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)
    assert pmap.get("does.not.exist") is None

    # Direct param: base aliases the live vLLM tensor (.data is a distinct object
    # sharing storage, so compare data_ptr), no hooks (received in place).
    dn = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert dn.base.data_ptr() == down.data_ptr()
    assert dn.pre is None and dn.post is None

    # Grouped expert down_proj -> w2 is also direct.
    edn = pmap.get("model.layers.0.mlp.experts.down_proj.weight")
    assert edn.base.data_ptr() == w2.data_ptr()
    assert edn.pre is None and edn.post is None

    # Merged dense gate_proj: pre allocates a recv buffer for gate's region of
    # gate_up_proj (rows [0:64] at TP=4); post scatters it back.
    g = pmap.get("model.layers.0.mlp.gate_proj.weight")
    assert g.pre is not None and g.post is not None
    ctx = g.pre(g.base)
    assert ctx.buf.shape == gate_up[0:64].shape
    assert ctx.extra["region"].shape == ctx.buf.shape
    ctx.buf.fill_(3.0)
    g.post(ctx)
    assert torch.equal(gate_up[0:64], torch.full_like(gate_up[0:64], 3.0))

    # Grouped expert gate_proj -> w13 gate half (dim-1 region); pre/post round-trip.
    eg = pmap.get("model.layers.0.mlp.experts.gate_proj.weight")
    assert eg.pre is not None and eg.post is not None
    egctx = eg.pre(eg.base)
    assert egctx.buf.shape == w13[:, 0:Pl, :].shape
    egctx.buf.fill_(5.0)
    eg.post(egctx)
    assert torch.equal(w13[:, 0:Pl, :], torch.full_like(w13[:, 0:Pl, :], 5.0))


@pytest.mark.parametrize(
    ("dtype", "shape", "error"),
    [
        (torch.float32, (32, 64), "dtype"),
        (torch.bfloat16, (32, 32), "local shape"),
    ],
)
def test_prepare_refit_rejects_untransformed_destination_metadata_mismatch(
    dtype,
    shape,
    error,
):
    from torch.distributed.tensor.placement_types import Replicate

    name = "model.layers.0.mlp.down_proj.weight"
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": [32, 64],
                    "dtype": "torch.bfloat16",
                    "src_mesh_info": MeshInfo(torch.arange(1)),
                    "src_placements": [Replicate()],
                    "dst_mesh_info": MeshInfo(torch.arange(1)),
                    "dst_placements": [Replicate()],
                    "components": _identity_components(
                        (32, 64),
                        torch.bfloat16,
                        src_placements=[Replicate()],
                        dst_placements=[Replicate()],
                    ),
                }
            ]
        },
    }
    ext = _make_ext({name: torch.empty(shape, dtype=dtype)})

    with pytest.raises(ValueError, match=error):
        ext.prepare_nccl_reshard_refit_info(refit_info)


def test_prepare_refit_returns_agreement_rebuilt_after_destination_mapping():
    from torch.distributed.tensor.placement_types import Replicate

    name = "model.layers.0.mlp.down_proj.weight"
    plan = RefitTransformPlan(
        transform_id="identity",
        components=(TransformComponentSpec("weight", (32, 64), "torch.bfloat16"),),
        finalize_scope="parameter",
    )
    expected = build_plan_agreement({name: plan})
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": [32, 64],
                    "dtype": "torch.bfloat16",
                    "src_mesh_info": MeshInfo(torch.arange(1)),
                    "src_placements": [Replicate()],
                    "dst_mesh_info": MeshInfo(torch.arange(1)),
                    "dst_placements": [Replicate()],
                    "transform_id": plan.transform_id,
                    "finalize_scope": plan.finalize_scope,
                    "components": _identity_components(
                        (32, 64),
                        torch.bfloat16,
                        src_placements=[Replicate()],
                        dst_placements=[Replicate()],
                    ),
                }
            ]
        },
        "refit_protocol_version": 99,
        "refit_component_count": 99,
        "plan_signature": "not-echoed",
    }
    ext = _make_ext({name: torch.empty(32, 64, dtype=torch.bfloat16)})

    assert ext.prepare_nccl_reshard_refit_info(refit_info) == expected


def test_build_mxfp8_map_receives_value_and_scale_into_matching_slices():
    H = 32
    refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": [128, H],
                    "components": _mxfp8_components((128, H)),
                }
            ]
        },
    }
    gate_up = torch.empty(128, H, dtype=torch.float8_e4m3fn)
    gate_up_scale = torch.empty(128, H // 32, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.gate_up_proj.weight_scale_from_checkpoint": (
                gate_up_scale
            ),
        }
    )

    spec = ext.build_hf_to_local_param_map(refit_info).get(
        "model.layers.0.mlp.gate_proj.weight"
    )
    assert spec is not None and spec.pre is not None and spec.post is not None

    ctx = spec.pre(spec.base)
    value_buf, scale_buf = ctx.tensors_for_transfer()
    assert value_buf.shape == (64, H)
    assert value_buf.dtype == torch.float8_e4m3fn
    assert scale_buf.shape == (64, H // 32)
    assert scale_buf.dtype == torch.uint8

    value_buf.fill_(1.0)
    scale_buf.fill_(127)
    spec.post(ctx)
    assert torch.equal(gate_up[:64], torch.ones_like(gate_up[:64]))
    assert torch.equal(
        gate_up_scale[:64],
        torch.full_like(gate_up_scale[:64], 127),
    )


def test_build_mxfp8_moe_map_uses_matching_w13_and_w2_scale_slices():
    E, H, P = 2, 32, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, P, H],
                    "grouped_expert_proj": "gate_proj",
                    "components": _mxfp8_components((E, P, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, P, H],
                    "grouped_expert_proj": "up_proj",
                    "components": _mxfp8_components((E, P, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "grouped_expert_proj": "down_proj",
                    "components": _mxfp8_components((E, H, P)),
                },
            ]
        },
    }
    w13 = torch.empty(E, 2 * P, H, dtype=torch.float8_e4m3fn)
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(E, 2 * P, H // 32, dtype=torch.uint8)
    w2_scale = torch.empty(E, H, P // 32, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.layers.0.mlp.experts.w13_weight_scale_from_checkpoint": (w13_scale),
            "model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint": w2_scale,
        }
    )

    pmap = ext.build_hf_to_local_param_map(refit_info)
    gate = pmap.get("model.layers.0.mlp.experts.gate_proj.weight")
    up = pmap.get("model.layers.0.mlp.experts.up_proj.weight")
    down = pmap.get("model.layers.0.mlp.experts.down_proj.weight")

    gate_ctx = gate.pre(gate.base)
    up_ctx = up.pre(up.base)
    down_ctx = down.pre(down.base)
    assert gate_ctx.tensors_for_transfer()[1].shape == (E, P, H // 32)
    assert up_ctx.tensors_for_transfer()[1].shape == (E, P, H // 32)
    assert down_ctx.tensors_for_transfer()[1].shape == (E, H, P // 32)

    gate_ctx.tensors_for_transfer()[1].fill_(11)
    up_ctx.tensors_for_transfer()[1].fill_(22)
    down_ctx.tensors_for_transfer()[1].fill_(33)
    gate.post(gate_ctx)
    up.post(up_ctx)
    down.post(down_ctx)
    assert torch.equal(w13_scale[:, :P], torch.full_like(w13_scale[:, :P], 11))
    assert torch.equal(w13_scale[:, P:], torch.full_like(w13_scale[:, P:], 22))
    assert torch.equal(w2_scale, torch.full_like(w2_scale, 33))


def test_build_mxfp8_map_rejects_missing_checkpoint_scale_target():
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [32, 64],
                    "components": _mxfp8_components((32, 64)),
                }
            ]
        },
    }
    ext = _make_ext(
        {
            "model.layers.0.mlp.down_proj.weight": torch.empty(
                32, 64, dtype=torch.float8_e4m3fn
            )
        }
    )

    with pytest.raises(ValueError, match="weight_scale_from_checkpoint"):
        ext.build_hf_to_local_param_map(refit_info)


def test_build_mxfp8_map_resolves_routed_expert_scale_from_registered_name():
    E, P, H = 2, 64, 32
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, P, H],
                    "grouped_expert_proj": "gate_proj",
                    "components": _mxfp8_components((E, P, H)),
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, P, H],
                    "grouped_expert_proj": "up_proj",
                    "components": _mxfp8_components((E, P, H)),
                },
            ]
        },
    }
    routed_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    ext = _make_ext(
        {
            routed_name: torch.empty(E, 2 * P, H, dtype=torch.float8_e4m3fn),
            routed_name + "_scale_from_checkpoint": torch.empty(
                E, 2 * P, H // 32, dtype=torch.uint8
            ),
        }
    )

    mapping = ext.build_hf_to_local_param_map(refit_info)

    assert mapping.get("model.layers.0.mlp.experts.gate_proj.weight") is not None
    assert mapping.get("model.layers.0.mlp.experts.up_proj.weight") is not None


@pytest.mark.parametrize(
    "roles",
    [
        ("weight_scale", "weight"),
        ("weight", "weight"),
        ("weight", "weight_scale", "input_scale"),
    ],
    ids=["wrong-order", "duplicate", "extra"],
)
def test_build_hf_to_local_param_map_rejects_unsupported_component_families(roles):
    name = "model.layers.0.mlp.down_proj.weight"
    components = [
        _component(
            role,
            (32, 2) if "scale" in role else (32, 64),
            torch.uint8 if "scale" in role else torch.float8_e4m3fn,
        )
        for role in roles
    ]
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": [32, 64],
                    "components": components,
                }
            ]
        },
    }
    ext = _make_ext({name: torch.empty(32, 64, dtype=torch.float8_e4m3fn)})

    with pytest.raises(ValueError, match="unsupported component family"):
        ext.build_hf_to_local_param_map(refit_info)


def test_mxfp8_receive_transfers_value_then_scale_before_merged_post(
    monkeypatch,
):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.weight_sync import xferdtensor as xferdtensor_module

    src_mesh = object()
    dst_mesh = object()
    value_src_placements = [object()]
    value_dst_placements = [object()]
    scale_src_placements = [object()]
    scale_dst_placements = [object()]
    group = object()
    stage_stream = object()
    call_order = []

    param_info = {
        "name": "model.layers.0.mlp.gate_proj.weight",
        "global_shape": [128, 32],
        "src_mesh_info": src_mesh,
        "dst_mesh_info": dst_mesh,
        "components": [
            _component(
                "weight",
                (128, 32),
                torch.float8_e4m3fn,
                src_placements=value_src_placements,
                dst_placements=value_dst_placements,
            ),
            _component(
                "weight_scale",
                (128, 1),
                torch.uint8,
                src_placements=scale_src_placements,
                dst_placements=scale_dst_placements,
            ),
        ],
    }
    mapping_refit_info = {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                param_info,
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "global_shape": [128, 32],
                    "components": _mxfp8_components((128, 32)),
                },
            ]
        },
    }
    receive_refit_info = {
        **mapping_refit_info,
        "per_layer_params": {"model.layers.0": [param_info]},
    }
    gate_up = torch.zeros(128, 32, dtype=torch.float8_e4m3fn)
    gate_up_scale = torch.zeros(128, 1, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.gate_up_proj.weight": gate_up,
            "model.layers.0.mlp.gate_up_proj.weight_scale_from_checkpoint": (
                gate_up_scale
            ),
        }
    )
    ext.nccl_reshard_refit_info = receive_refit_info
    ext.hf_to_local_param_map = ext.build_hf_to_local_param_map(mapping_refit_info)
    ext.pp_comm_groups = {0: group}
    ext.model_runner.vllm_config = object()
    ext.model_config = object()
    ext.device = object()
    ext._receive_and_load_misc_params = lambda: call_order.append("misc")

    spec = ext.hf_to_local_param_map.get(param_info["name"])
    assert spec is not None and spec.post is not None
    original_post = spec.post

    def recording_post(ctx):
        assert call_order == ["value", "scale"]
        call_order.append("post")
        original_post(ctx)

    spec.post = recording_post

    def recording_xferdtensor(
        src_tensor,
        actual_src_mesh,
        actual_src_placements,
        dst_tensor,
        actual_dst_mesh,
        actual_dst_placements,
        actual_group,
        stream,
    ):
        assert src_tensor is None
        assert actual_src_mesh is src_mesh
        assert actual_dst_mesh is dst_mesh
        assert actual_group is group
        assert stream is stage_stream
        if dst_tensor._local_tensor.dtype == torch.float8_e4m3fn:
            assert tuple(dst_tensor.shape) == (128, 32)
            assert actual_src_placements is value_src_placements
            assert actual_dst_placements is value_dst_placements
            call_order.append("value")
            dst_tensor._local_tensor.fill_(1.0)
        else:
            assert call_order == ["value"]
            assert tuple(dst_tensor.shape) == (128, 1)
            assert actual_src_placements is scale_src_placements
            assert actual_dst_placements is scale_dst_placements
            call_order.append("scale")
            dst_tensor._local_tensor.fill_(127)

    class FakeEvent:
        def record(self):
            return None

    monkeypatch.setattr(xferdtensor_module, "xferdtensor", recording_xferdtensor)
    monkeypatch.setattr(vllm_backend.torch.cuda, "Stream", lambda: stage_stream)
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )
    monkeypatch.setattr(vllm_backend.torch.cuda, "Event", FakeEvent)

    def recording_synchronize():
        if call_order[-1] == "post":
            call_order.append("stage-barrier")
        elif call_order[-1] == "misc":
            call_order.append("misc-barrier")
        else:
            raise AssertionError(f"unexpected synchronize after {call_order!r}")

    monkeypatch.setattr(vllm_backend.torch.cuda, "synchronize", recording_synchronize)
    monkeypatch.setattr(vllm_backend.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vllm_backend.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda *_: call_order.append("process"),
    )

    assert ext.nccl_reshard_refit() is True
    assert call_order == [
        "value",
        "scale",
        "post",
        "stage-barrier",
        "misc",
        "misc-barrier",
        "process",
    ]
    assert torch.equal(gate_up[:64], torch.ones_like(gate_up[:64]))
    assert torch.equal(
        gate_up_scale[:64],
        torch.full_like(gate_up_scale[:64], 127),
    )


@pytest.mark.parametrize(
    ("case", "component_count", "transfer_count", "error"),
    [
        ("explicit_empty", 1, 0, "component count"),
        ("empty_components", 0, 1, "nonempty component list"),
        ("underfilled", 2, 1, "component count"),
        ("overfilled", 1, 2, "component count"),
    ],
)
def test_receive_rejects_invalid_component_counts_before_transfer_or_post(
    monkeypatch,
    case,
    component_count,
    transfer_count,
    error,
):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.weight_sync import xferdtensor as xferdtensor_module
    from nemo_rl.weight_sync.nccl_reshard_utils import LocalParamSpec, RefitCtx

    calls = []
    name = "model.layers.0.mlp.down_proj.weight"
    tensors = tuple(torch.empty(2, 2) for _ in range(max(transfer_count, 1)))
    ext = _make_ext({name: tensors[0]})
    ext.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "src_mesh_info": object(),
                    "dst_mesh_info": object(),
                    "components": [
                        _component(
                            f"component_{index}",
                            (2, 2),
                            torch.float32,
                            src_placements=[object()],
                            dst_placements=[object()],
                        )
                        for index in range(component_count)
                    ],
                }
            ]
        },
    }
    ext.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            name: LocalParamSpec(
                base=tensors[0],
                pre=lambda _: RefitCtx(
                    buf=tensors[0],
                    transfer_tensors=tensors[:transfer_count],
                ),
                post=lambda _: calls.append("post"),
            )
        }
    )
    ext.pp_comm_groups = {0: object()}

    monkeypatch.setattr(
        xferdtensor_module,
        "xferdtensor",
        lambda *_args, **_kwargs: calls.append("transfer"),
    )
    monkeypatch.setattr(vllm_backend.torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )

    with pytest.raises(ValueError, match=error):
        ext.nccl_reshard_refit()

    assert calls == [], case
