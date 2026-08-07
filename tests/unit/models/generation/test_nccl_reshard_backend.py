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

The receiver methods are loaded from ``vllm_backend`` with a focused AST harness
so this module does not require a local vLLM installation.
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from torch.distributed._tensor import Shard


@dataclass
class RefitCtx:
    buf: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalParamSpec:
    base: Any
    pre: Callable[[Any], RefitCtx] | None = None
    post: Callable[[RefitCtx], None] | None = None


@dataclass
class HFToLocalParamMap:
    specs: dict[str | tuple[str, str], LocalParamSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.specs = {
            (key, "weight") if isinstance(key, str) else key: spec
            for key, spec in self.specs.items()
        }

    def get(
        self,
        hf_name: str,
        default: LocalParamSpec | None = None,
        *,
        role: str = "weight",
    ) -> LocalParamSpec | None:
        return self.specs.get((hf_name, role), default)


_STR_TO_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "torch.uint8": torch.uint8,
    "uint8": torch.uint8,
}
_LAYER_RE = re.compile(r"^(?:(?P<prefix>.+)\.)?layers\.(?P<index>\d+)(?:\.|$)")


def _extract_layer_prefix(param_name: str) -> str | None:
    match = _LAYER_RE.match(param_name)
    if match is None:
        return None
    return match.group("prefix") or ""


def _load_vllm_extension_class() -> type:
    source_path = (
        Path(__file__).parents[4] / "nemo_rl/models/generation/vllm/vllm_backend.py"
    )
    tree = ast.parse(source_path.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmInternalWorkerExtension"
    )
    method_names = {
        "build_hf_to_local_param_map",
        "_build_hf_to_gen_backend_mapping",
        "nccl_reshard_refit",
    }
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    class_kwargs = {
        "name": "VllmInternalWorkerExtension",
        "bases": [],
        "keywords": [],
        "body": methods,
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_kwargs["type_params"] = []
    test_module = ast.Module(
        body=[ast.ClassDef(**class_kwargs)],
        type_ignores=[],
    )
    ast.fix_missing_locations(test_module)
    namespace = {
        "Any": Any,
        "_STR_TO_DTYPE": _STR_TO_DTYPE,
        "HFToLocalParamMap": HFToLocalParamMap,
        "LocalParamSpec": LocalParamSpec,
        "RefitCtx": RefitCtx,
        "Shard": Shard,
        "_extract_layer_prefix": _extract_layer_prefix,
        "torch": torch,
    }
    exec(compile(test_module, str(source_path), "exec"), namespace)
    return namespace["VllmInternalWorkerExtension"]


VllmInternalWorkerExtension = _load_vllm_extension_class()


# --------------------------------------------------------------------------
# _build_hf_to_gen_backend_mapping
# --------------------------------------------------------------------------
def _make_ext(vllm_params: dict[str, torch.Tensor]) -> Any:
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


def _param(*shape: int) -> torch.Tensor:
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


def test_build_hf_to_local_param_map_quantizes_bf16_for_mxfp8(monkeypatch):
    H, E, Pl = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.gate_proj.weight",
                    "global_shape": [E, Pl, H],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "gate_proj",
                },
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, Pl],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                },
            ]
        },
    }
    w13 = torch.empty(E, 2 * Pl, H, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(E, 2 * Pl, H // 32, dtype=torch.uint8)
    w2 = torch.empty(E, H, Pl, dtype=torch.float8_e4m3fn)
    w2_scale = torch.empty(E, H, Pl // 32, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w13_weight_scale_from_checkpoint": w13_scale,
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint": w2_scale,
        }
    )

    def fake_quantize(weight):
        return (
            torch.full_like(weight, 3, dtype=torch.float8_e4m3fn),
            torch.full(
                (*weight.shape[:-1], weight.shape[-1] // 32), 7, dtype=torch.uint8
            ),
        )

    package_names = [
        "nemo_rl",
        "nemo_rl.models",
        "nemo_rl.models.generation",
        "nemo_rl.models.generation.vllm",
        "nemo_rl.models.generation.vllm.quantization",
    ]
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    fp8_module.quantize_mxfp8_weight = fake_quantize
    monkeypatch.setitem(sys.modules, fp8_module.__name__, fp8_module)

    pmap = ext.build_hf_to_local_param_map(refit_info)

    gate = pmap.get("model.layers.0.mlp.experts.gate_proj.weight")
    assert gate is not None and gate.pre is not None and gate.post is not None
    gate_ctx = gate.pre(gate.base)
    assert gate_ctx.buf.dtype == torch.bfloat16
    assert gate_ctx.buf.shape == w13[:, :Pl, :].shape
    gate.post(gate_ctx)
    assert torch.all(w13[:, :Pl, :].float() == 3)
    assert torch.all(w13_scale[:, :Pl, :] == 7)

    down = pmap.get("model.layers.0.mlp.experts.down_proj.weight")
    assert down is not None and down.pre is not None and down.post is not None
    down_ctx = down.pre(down.base)
    assert down_ctx.buf.dtype == torch.bfloat16
    assert down_ctx.buf.shape == w2.shape
    down.post(down_ctx)
    assert torch.all(w2.float() == 3)
    assert torch.all(w2_scale == 7)


def test_build_hf_to_local_param_map_keeps_matching_blockwise_fp8_storage():
    H, E, P = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "dtype": "torch.float8_e4m3fn",
                    "grouped_expert_proj": "down_proj",
                }
            ]
        },
    }
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    ext = _make_ext({"model.layers.0.mlp.experts.w2_weight": w2})

    spec = ext.build_hf_to_local_param_map(refit_info).get(
        "model.layers.0.mlp.experts.down_proj.weight"
    )

    assert spec is not None
    assert spec.base.data_ptr() == w2.data_ptr()
    assert spec.pre is None and spec.post is None


def test_build_hf_to_local_param_map_rejects_invalid_mxfp8_scale_shape():
    H, E, P = 32, 2, 64
    refit_info = {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.experts.down_proj.weight",
                    "global_shape": [E, H, P],
                    "dtype": "torch.bfloat16",
                    "grouped_expert_proj": "down_proj",
                }
            ]
        },
    }
    w2 = torch.empty(E, H, P, dtype=torch.float8_e4m3fn)
    invalid_scale = torch.empty(E, H, 1, dtype=torch.uint8)
    ext = _make_ext(
        {
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.layers.0.mlp.experts.w2_weight_scale_from_checkpoint": invalid_scale,
        }
    )

    with pytest.raises(ValueError, match="has shape"):
        ext.build_hf_to_local_param_map(refit_info)


def test_native_mxfp8_binds_dense_and_nested_moe_value_scale_regions_at_tp2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, intermediate, experts = 64, 128, 2
    tp_size = 2
    mesh = SimpleNamespace(mesh=torch.arange(tp_size))

    def components(shape: tuple[int, ...], shard_dim: int) -> list[dict[str, Any]]:
        scale_shape = (*shape[:-1], shape[-1] // 32)
        return [
            {
                "role": "weight",
                "global_shape": shape,
                "dtype": "torch.float8_e4m3fn",
                "src_placements": [Shard(shard_dim)],
                "dst_placements": [Shard(shard_dim)],
            },
            {
                "role": "weight_scale",
                "global_shape": scale_shape,
                "dtype": "torch.uint8",
                "src_placements": [Shard(shard_dim)],
                "dst_placements": [Shard(shard_dim)],
            },
        ]

    dense_prefix = "model.layers.0.mlp"
    expert_prefix = f"{dense_prefix}.experts"
    params = [
        {
            "name": f"{dense_prefix}.gate_proj.weight",
            "global_shape": [intermediate, hidden],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "components": components((intermediate, hidden), 0),
        },
        {
            "name": f"{dense_prefix}.up_proj.weight",
            "global_shape": [intermediate, hidden],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "components": components((intermediate, hidden), 0),
        },
        {
            "name": f"{dense_prefix}.down_proj.weight",
            "global_shape": [hidden, intermediate],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "components": components((hidden, intermediate), 1),
        },
        {
            "name": f"{expert_prefix}.gate_proj.weight",
            "global_shape": [experts, intermediate, hidden],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "grouped_expert_proj": "gate_proj",
            "components": components((experts, intermediate, hidden), 1),
        },
        {
            "name": f"{expert_prefix}.up_proj.weight",
            "global_shape": [experts, intermediate, hidden],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "grouped_expert_proj": "up_proj",
            "components": components((experts, intermediate, hidden), 1),
        },
        {
            "name": f"{expert_prefix}.down_proj.weight",
            "global_shape": [experts, hidden, intermediate],
            "dtype": "torch.bfloat16",
            "dst_mesh_info": mesh,
            "grouped_expert_proj": "down_proj",
            "components": components((experts, hidden, intermediate), 2),
        },
    ]
    gate_up = torch.empty(intermediate, hidden, dtype=torch.float8_e4m3fn)
    gate_up_scale = torch.empty(intermediate, hidden // 32, dtype=torch.uint8)
    down = torch.empty(hidden, intermediate // tp_size, dtype=torch.float8_e4m3fn)
    down_scale = torch.empty(hidden, intermediate // tp_size // 32, dtype=torch.uint8)
    w13 = torch.empty(experts, intermediate, hidden, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(experts, intermediate, hidden // 32, dtype=torch.uint8)
    w2 = torch.empty(
        experts, hidden, intermediate // tp_size, dtype=torch.float8_e4m3fn
    )
    w2_scale = torch.empty(
        experts, hidden, intermediate // tp_size // 32, dtype=torch.uint8
    )
    ext = _make_ext(
        {
            f"{dense_prefix}.gate_up_proj.weight": gate_up,
            f"{dense_prefix}.gate_up_proj.weight_scale_from_checkpoint": gate_up_scale,
            f"{dense_prefix}.down_proj.weight": down,
            f"{dense_prefix}.down_proj.weight_scale_from_checkpoint": down_scale,
            f"{expert_prefix}.routed_experts.w13_weight": w13,
            f"{expert_prefix}.routed_experts.w13_weight_scale_from_checkpoint": w13_scale,
            f"{expert_prefix}.routed_experts.w2_weight": w2,
            f"{expert_prefix}.routed_experts.w2_weight_scale_from_checkpoint": w2_scale,
        }
    )

    mapping = ext.build_hf_to_local_param_map(
        {
            "gen_tp_size": tp_size,
            "layer_names": ["model.layers.0"],
            "per_layer_params": {"model.layers.0": params},
        }
    )

    quantize_calls: list[torch.Tensor] = []

    def fail_if_quantized(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantize_calls.append(weight)
        return (
            torch.empty_like(weight, dtype=torch.float8_e4m3fn),
            torch.empty(
                (*weight.shape[:-1], weight.shape[-1] // 32), dtype=torch.uint8
            ),
        )

    package_names = [
        "nemo_rl",
        "nemo_rl.models",
        "nemo_rl.models.generation",
        "nemo_rl.models.generation.vllm",
        "nemo_rl.models.generation.vllm.quantization",
    ]
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    fp8_module = ModuleType("nemo_rl.models.generation.vllm.quantization.fp8")
    fp8_module.quantize_mxfp8_weight = fail_if_quantized
    monkeypatch.setitem(sys.modules, fp8_module.__name__, fp8_module)

    targets = {
        f"{dense_prefix}.gate_proj.weight": (gate_up, gate_up_scale, (64, 64), (64, 2)),
        f"{dense_prefix}.up_proj.weight": (gate_up, gate_up_scale, (64, 64), (64, 2)),
        f"{dense_prefix}.down_proj.weight": (down, down_scale, (64, 64), (64, 2)),
        f"{expert_prefix}.gate_proj.weight": (
            w13,
            w13_scale,
            (2, 64, 64),
            (2, 64, 2),
        ),
        f"{expert_prefix}.up_proj.weight": (
            w13,
            w13_scale,
            (2, 64, 64),
            (2, 64, 2),
        ),
        f"{expert_prefix}.down_proj.weight": (
            w2,
            w2_scale,
            (2, 64, 64),
            (2, 64, 2),
        ),
    }
    for name, (value, scale, value_shape, scale_shape) in targets.items():
        value_spec = mapping.get(name, role="weight")
        scale_spec = mapping.get(name, role="weight_scale")
        assert value_spec is not None
        assert scale_spec is not None
        assert value_spec.base.data_ptr() == value.data_ptr()
        assert scale_spec.base.data_ptr() == scale.data_ptr()
        value_ctx = (
            value_spec.pre(value_spec.base)
            if value_spec.pre is not None
            else RefitCtx(buf=value_spec.base)
        )
        scale_ctx = (
            scale_spec.pre(scale_spec.base)
            if scale_spec.pre is not None
            else RefitCtx(buf=scale_spec.base)
        )
        assert tuple(value_ctx.buf.shape) == value_shape
        assert tuple(scale_ctx.buf.shape) == scale_shape
        assert value_ctx.buf.dtype == torch.float8_e4m3fn
        assert scale_ctx.buf.dtype == torch.uint8
        if value_spec.post is not None:
            value_spec.post(value_ctx)
        if scale_spec.post is not None:
            scale_spec.post(scale_ctx)
    assert quantize_calls == []


def _single_native_down_refit_info(
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    name = "model.layers.0.mlp.down_proj.weight"
    return {
        "gen_tp_size": 1,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": [64, 64],
                    "dtype": "torch.bfloat16",
                    "dst_mesh_info": SimpleNamespace(mesh=torch.arange(1)),
                    "components": components,
                }
            ]
        },
    }


def _native_down_components() -> list[dict[str, Any]]:
    return [
        {
            "role": "weight",
            "global_shape": (64, 64),
            "dtype": "torch.float8_e4m3fn",
            "src_placements": [Shard(1)],
            "dst_placements": [Shard(1)],
        },
        {
            "role": "weight_scale",
            "global_shape": (64, 2),
            "dtype": "torch.uint8",
            "src_placements": [Shard(1)],
            "dst_placements": [Shard(1)],
        },
    ]


def test_native_mxfp8_rejects_incomplete_value_scale_pair() -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    value = torch.empty(64, 64, dtype=torch.float8_e4m3fn)
    scale = torch.empty(64, 2, dtype=torch.uint8)
    refit_info = _single_native_down_refit_info(_native_down_components()[:1])
    ext = _make_ext(
        {
            name: value,
            f"{name}_scale_from_checkpoint": scale,
        }
    )

    with pytest.raises(ValueError, match="requires ordered components"):
        ext.build_hf_to_local_param_map(refit_info)


@pytest.mark.parametrize(
    ("scale_shape", "scale_dtype", "error"),
    [
        ((64, 3), "torch.uint8", "has shape"),
        ((64, 2), "torch.bfloat16", "must use torch.uint8"),
    ],
)
def test_native_mxfp8_rejects_wrong_scale_component_layout(
    scale_shape: tuple[int, ...],
    scale_dtype: str,
    error: str,
) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    components = _native_down_components()
    components[1]["global_shape"] = scale_shape
    components[1]["dtype"] = scale_dtype
    ext = _make_ext(
        {
            name: torch.empty(64, 64, dtype=torch.float8_e4m3fn),
            f"{name}_scale_from_checkpoint": torch.empty(64, 2, dtype=torch.uint8),
        }
    )

    with pytest.raises(ValueError, match=error):
        ext.build_hf_to_local_param_map(_single_native_down_refit_info(components))


@pytest.mark.parametrize(
    ("target_shape", "target_dtype", "error"),
    [
        ((64, 1), torch.uint8, "has shape"),
        ((64, 2), torch.bfloat16, "has dtype"),
    ],
)
def test_native_mxfp8_rejects_wrong_checkpoint_scale_target_layout(
    target_shape: tuple[int, ...],
    target_dtype: torch.dtype,
    error: str,
) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    ext = _make_ext(
        {
            name: torch.empty(64, 64, dtype=torch.float8_e4m3fn),
            f"{name}_scale_from_checkpoint": torch.empty(
                *target_shape, dtype=target_dtype
            ),
        }
    )

    with pytest.raises(ValueError, match=error):
        ext.build_hf_to_local_param_map(
            _single_native_down_refit_info(_native_down_components())
        )


@pytest.mark.parametrize(
    ("target_shape", "target_dtype", "error"),
    [
        ((64, 32), torch.float8_e4m3fn, "has shape"),
        ((64, 64), torch.bfloat16, "has dtype"),
    ],
)
def test_native_mxfp8_rejects_wrong_value_target_layout(
    target_shape: tuple[int, ...],
    target_dtype: torch.dtype,
    error: str,
) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    ext = _make_ext(
        {
            name: torch.empty(*target_shape, dtype=target_dtype),
            f"{name}_scale_from_checkpoint": torch.empty(64, 2, dtype=torch.uint8),
        }
    )

    with pytest.raises(ValueError, match=error):
        ext.build_hf_to_local_param_map(
            _single_native_down_refit_info(_native_down_components())
        )


def test_native_mxfp8_requires_first_post_load_checkpoint_scale() -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    ext = _make_ext({name: torch.empty(64, 64, dtype=torch.float8_e4m3fn)})

    with pytest.raises(ValueError, match="first MXFP8 post-load processing"):
        ext.build_hf_to_local_param_map(
            _single_native_down_refit_info(_native_down_components())
        )


def test_native_mxfp8_receive_loop_uses_component_metadata_and_finalizes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    value = torch.empty(64, 64, dtype=torch.float8_e4m3fn)
    scale = torch.empty(64, 2, dtype=torch.uint8)
    ext = _make_ext({})
    ext.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            (name, "weight"): LocalParamSpec(base=value),
            (name, "weight_scale"): LocalParamSpec(base=scale),
        }
    )
    ext.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "global_shape": (64, 128),
                    "src_mesh_info": "src-mesh",
                    "src_placements": ["parent-src"],
                    "dst_mesh_info": "dst-mesh",
                    "dst_placements": ["parent-dst"],
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": (64, 128),
                            "src_placements": ["weight-src"],
                            "dst_placements": ["weight-dst"],
                        },
                        {
                            "role": "weight_scale",
                            "global_shape": (64, 4),
                            "src_placements": ["scale-src"],
                            "dst_placements": ["scale-dst"],
                        },
                    ],
                }
            ]
        },
    }
    ext.pp_comm_groups = {0: "group"}
    ext.model_runner.vllm_config = "vllm-config"
    ext.model_config = "model-config"
    ext.device = "cpu"
    ext._receive_and_load_misc_params = lambda: None
    ext._maybe_process_fp8_kv_cache = lambda: None

    transfers = []
    finalizer_calls = []

    class FakeDTensorRef:
        def __init__(self, local_tensor: torch.Tensor, global_shape: Any) -> None:
            self._local_tensor = local_tensor
            self.shape = tuple(global_shape)

    def fake_xferdtensor(
        _src_tensor: Any,
        _src_mesh: Any,
        src_placements: Any,
        dst_tensor: FakeDTensorRef,
        _dst_mesh: Any,
        dst_placements: Any,
        _group: Any,
        _stream: Any,
    ) -> None:
        transfers.append(
            (
                dst_tensor._local_tensor.data_ptr(),
                dst_tensor.shape,
                src_placements,
                dst_placements,
            )
        )

    xfer_module = ModuleType("nemo_rl.weight_sync.xferdtensor")
    xfer_module.DTensorRef = FakeDTensorRef
    xfer_module.xferdtensor = fake_xferdtensor
    config_module = ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: nullcontext()
    loader_module = ModuleType("vllm.model_executor.model_loader.utils")
    loader_module.process_weights_after_loading = lambda *args: finalizer_calls.append(
        args
    )
    monkeypatch.setitem(sys.modules, "nemo_rl", ModuleType("nemo_rl"))
    monkeypatch.setitem(
        sys.modules, "nemo_rl.weight_sync", ModuleType("nemo_rl.weight_sync")
    )
    monkeypatch.setitem(sys.modules, xfer_module.__name__, xfer_module)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor", ModuleType("vllm.model_executor")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader",
        ModuleType("vllm.model_executor.model_loader"),
    )
    monkeypatch.setitem(sys.modules, loader_module.__name__, loader_module)

    class FakeEvent:
        def record(self) -> None:
            return None

        def synchronize(self) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    assert ext.nccl_reshard_refit() is True
    assert transfers == [
        (value.data_ptr(), (64, 128), ["weight-src"], ["weight-dst"]),
        (scale.data_ptr(), (64, 4), ["scale-src"], ["scale-dst"]),
    ]
    assert len(finalizer_calls) == 1
