# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pristine MXFP8 scale storage that survives vLLM's layerwise reload.

A refit writes the trainer's freshly quantized weights into a live vLLM engine.
For MXFP8 that means the scales have to be kept twice: ``<name>`` ends up holding
whatever layout the kernel wants, so the checkpoint's own values live in a twin
named ``<name>_from_checkpoint`` that the refit loader targets by name.

Where those twins are *allocated* is the load-bearing part. vLLM snapshots each
layer's parameters in ``initialize_model`` -- right after the model is built, so
after ``create_weights`` and before any weight is loaded -- and
``restore_layer_on_meta`` later deletes everything that snapshot does not
contain. Allocating a twin lazily, on first use in
``process_weights_after_loading``, therefore works at startup and then vanishes
at the first refit, and the loader raises ``AttributeError`` looking it up. The
invariant this module exists to hold:

    every parameter a refit can target must exist by the time create_weights
    returns.

Kept separate from ``fp8.py`` so the same implementation can be shared verbatim
by branches whose ``fp8.py`` have diverged.
"""

import torch

# Marks that a layer's ``*_from_checkpoint`` scales hold the checkpoint values.
# A plain attribute, not a parameter, so vLLM's layerwise reload leaves it alone.
CHECKPOINT_SCALES_SEEDED = "_nrl_checkpoint_scales_seeded"

LINEAR_SCALE_NAMES = ("weight_scale",)
MOE_SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")


def register_checkpoint_scale_params(
    layer: torch.nn.Module,
    scale_names: tuple[str, ...],
    *,
    input_dim: int,
    output_dim: int,
    block_quant_method: bool,
) -> None:
    """Allocate the refit's pristine-scale twins alongside vLLM's own scales.

    Call this from ``create_weights`` and nowhere else; see the module docstring
    for why later is too late.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
    from vllm.model_executor.parameter import ModelWeightParameter
    from vllm.model_executor.utils import set_weight_attrs

    for name in scale_names:
        source = getattr(layer, name, None)
        if source is None:
            raise RuntimeError(
                f"MXFP8 refit expected {type(layer).__name__} to allocate a "
                f"{name!r} parameter, so it could allocate the matching "
                f"{name}_from_checkpoint that refit loads into. vLLM's MXFP8 "
                "weight layout has changed and the refit needs updating."
            )
        # empty_like and not source.data: aliasing the live scale means a later
        # in-place write of the kernel layout goes straight through the twin.
        twin = ModelWeightParameter(
            data=torch.empty_like(source.data),
            input_dim=input_dim,
            output_dim=output_dim,
            weight_loader=source.weight_loader,
        )
        layer.register_parameter(f"{name}_from_checkpoint", twin)
        if block_quant_method:
            set_weight_attrs(
                twin, {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )


def register_linear_checkpoint_scale_params(layer: torch.nn.Module) -> None:
    register_checkpoint_scale_params(
        layer,
        LINEAR_SCALE_NAMES,
        input_dim=1,
        output_dim=0,
        block_quant_method=False,
    )


def register_moe_checkpoint_scale_params(layer: torch.nn.Module) -> None:
    register_checkpoint_scale_params(
        layer,
        MOE_SCALE_NAMES,
        input_dim=2,
        output_dim=1,
        block_quant_method=True,
    )


def seed_checkpoint_scales(
    layer: torch.nn.Module, scale_names: tuple[str, ...]
) -> None:
    """Copy the checkpoint's scales into their ``*_from_checkpoint`` twins once.

    Engine startup and refit deliver the pristine scale to different places: the
    initial checkpoint load writes it into ``<name>``, while a refit writes it
    into ``<name>_from_checkpoint`` because by then ``<name>`` holds the
    kernel's repacked layout. Seeding once at the end of the initial load lets
    everything downstream read the pristine values from one place.
    """
    if getattr(layer, CHECKPOINT_SCALES_SEEDED, False):
        return
    for name in scale_names:
        getattr(layer, f"{name}_from_checkpoint").data.copy_(getattr(layer, name).data)
    setattr(layer, CHECKPOINT_SCALES_SEEDED, True)


def install_processed_tensor(
    layer: torch.nn.Module, name: str, processed: torch.Tensor
) -> None:
    """Write a kernel-layout tensor back into ``name``.

    The repacked layout is usually a different shape from the checkpoint layout
    the parameter was allocated with, and vLLM's layerwise reload restores the
    allocated shape before every refit, so this has to be able to replace the
    parameter and not only copy into it. ``name`` may also be absent entirely:
    the padded execution copies are derived in ``process_weights_after_loading``
    and so are not in the reload snapshot, which puts them back at square one
    after every refit.
    """
    processed = processed.contiguous()
    current = getattr(layer, name, None)
    if (
        current is not None
        and current.shape == processed.shape
        and current.dtype == processed.dtype
    ):
        current.data.copy_(processed)
        return
    replacement = torch.nn.Parameter(processed, requires_grad=False)
    weight_loader = getattr(current, "weight_loader", None)
    if weight_loader is not None:
        replacement.weight_loader = weight_loader
    setattr(layer, name, replacement)


def wrap_create_weights_mxfp8_linear(original):
    """Add the refit's ``weight_scale_from_checkpoint`` to vLLM's allocation.

    Wraps rather than reimplements: how an MXFP8 linear lays its weights out is
    vLLM's to change, and only the extra parameter is ours.
    """

    def create_weights_mxfp8_linear(self, layer, *args, **kwargs):
        original(self, layer, *args, **kwargs)
        register_linear_checkpoint_scale_params(layer)

    return create_weights_mxfp8_linear
