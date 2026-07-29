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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")  # module-top `import vllm` in vllm_backend

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (  # noqa: E402
    HFToLocalParamMap,
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
                },
                {"name": "model.layers.0.mlp.up_proj.weight", "global_shape": [256, H]},
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                },
                # MoE experts: gate/up -> w13 halves, down -> w2.
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "grouped_expert_proj": "down_proj",
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
                },
                {"name": "model.layers.0.mlp.up_proj.weight", "global_shape": [256, H]},
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [H, 256],
                },
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, 128, H],
                    "grouped_expert_proj": "up_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, 128],
                    "grouped_expert_proj": "down_proj",
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
                    "refit_transform": "mxfp8",
                    "scale_global_shape": [128, H // 32],
                    "scale_dtype": "torch.uint8",
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
    assert ctx.buf.shape == (64, H)
    assert ctx.buf.dtype == torch.float8_e4m3fn
    assert ctx.extra["scale_buf"].shape == (64, H // 32)
    assert ctx.extra["scale_buf"].dtype == torch.uint8

    ctx.buf.fill_(1.0)
    ctx.extra["scale_buf"].fill_(127)
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
                    "scale_global_shape": [E, P, H // 32],
                    "grouped_expert_proj": "gate_proj",
                    "refit_transform": "mxfp8",
                },
                {
                    "name": "model.layers.0.mlp.experts.up_proj.weight",
                    "global_shape": [E, P, H],
                    "scale_global_shape": [E, P, H // 32],
                    "grouped_expert_proj": "up_proj",
                    "refit_transform": "mxfp8",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "scale_global_shape": [E, H, P // 32],
                    "grouped_expert_proj": "down_proj",
                    "refit_transform": "mxfp8",
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
            "model.layers.0.mlp.experts.w13_weight_scale_from_checkpoint": (
                w13_scale
            ),
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
    assert gate_ctx.extra["scale_buf"].shape == (E, P, H // 32)
    assert up_ctx.extra["scale_buf"].shape == (E, P, H // 32)
    assert down_ctx.extra["scale_buf"].shape == (E, H, P // 32)

    gate_ctx.extra["scale_buf"].fill_(11)
    up_ctx.extra["scale_buf"].fill_(22)
    down_ctx.extra["scale_buf"].fill_(33)
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
                    "scale_global_shape": [32, 2],
                    "refit_transform": "mxfp8",
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
