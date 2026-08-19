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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import Tensor


class MarkovRuntimeLayout(str, Enum):
    """Vocabulary layout used by a vLLM DSpark Markov parameter."""

    REPLICATED = "replicated"
    VOCAB_SHARDED = "vocab_sharded"


@dataclass(frozen=True, slots=True)
class PreparedMarkovWeight:
    """A full loader tensor paired with the detected runtime layout."""

    tensor: Tensor
    runtime_layout: MarkovRuntimeLayout
    global_shape: tuple[int, int]


def _target_language_model(target_model: Any) -> Any:
    accessor = getattr(target_model, "get_language_model", None)
    if callable(accessor):
        return accessor()
    return target_model


def _target_lm_head(target_model: Any, target_language_model: Any) -> Any:
    language_model_head = getattr(target_language_model, "lm_head", None)
    if language_model_head is not None:
        return language_model_head
    target_head = getattr(target_model, "lm_head", None)
    if target_head is None:
        raise ValueError("the target model does not expose an lm_head")
    return target_head


def _target_embedding(target_model: Any, target_language_model: Any) -> Any:
    for owner in (target_language_model, target_model):
        inner_model = getattr(owner, "model", None)
        embedding = getattr(inner_model, "embed_tokens", None)
        if embedding is not None:
            return embedding
    raise ValueError("the target model does not expose model.embed_tokens")


def bind_live_target_io(
    *, draft_model: Any, target_model: Any, share_embedding: bool
) -> None:
    """Bind a body-only DFlash/DSpark model to the live target I/O modules.

    vLLM 0.25 exposes the target head on the top-level model, while newer
    conditional wrappers may expose it on the language model. Draft checkpoints
    used for online co-training must not retain stale owned copies of either
    shared component.
    """
    for flag in ("has_own_embed_tokens", "has_own_lm_head"):
        if bool(getattr(draft_model, flag, False)):
            raise ValueError(
                f"body-only draft runtime requires {flag}=False before live binding"
            )

    target_language_model = _target_language_model(target_model)
    target_head = _target_lm_head(target_model, target_language_model)
    draft_model.lm_head = target_head
    if draft_model.lm_head is not target_head:
        raise RuntimeError("failed to bind the live target lm_head")

    if not share_embedding:
        return
    draft_inner_model = getattr(draft_model, "model", None)
    if draft_inner_model is None or not hasattr(draft_inner_model, "embed_tokens"):
        raise ValueError("the draft model does not expose model.embed_tokens")
    target_embedding = _target_embedding(target_model, target_language_model)
    draft_inner_model.embed_tokens = target_embedding
    if draft_inner_model.embed_tokens is not target_embedding:
        raise RuntimeError("failed to bind the live target token embedding")


def prepare_markov_loader_weight(
    *,
    name: str,
    weight: Tensor,
    target_shape: tuple[int, int],
    global_vocab_size: int,
    tp_size: int,
) -> PreparedMarkovWeight:
    """Validate a full DSpark weight for version-dependent vLLM loading.

    The standard vLLM vocabulary loader always consumes a global-vocabulary
    tensor. Older runtimes narrow that tensor into a TP-local parameter inside
    the loader; newer runtimes may keep the parameter replicated. A trainer-
    local tensor therefore requires the component-aware gather supplied by the
    reshard layer rather than a silent per-rank adaptation here.
    """
    if not name.endswith(("markov_w1.weight", "markov_w2.weight")):
        raise ValueError(f"unsupported DSpark Markov weight: {name}")
    if weight.ndim != 2 or len(target_shape) != 2:
        raise ValueError("DSpark Markov weights must be rank-2 tensors")
    if global_vocab_size <= 0:
        raise ValueError("global_vocab_size must be positive")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if weight.shape[0] != global_vocab_size:
        if weight.shape[0] * tp_size == global_vocab_size:
            raise ValueError(
                f"{name} is a TP-local transport tensor; a component-aware gather "
                "is required before the vLLM loader"
            )
        raise ValueError(
            f"{name} has vocab dimension {weight.shape[0]}, expected "
            f"the global size {global_vocab_size}"
        )
    if weight.shape[1] != target_shape[1]:
        raise ValueError(
            f"{name} hidden dimension {weight.shape[1]} does not match runtime "
            f"dimension {target_shape[1]}"
        )

    target_vocab_size = target_shape[0]
    if target_vocab_size == global_vocab_size:
        runtime_layout = MarkovRuntimeLayout.REPLICATED
    elif (
        global_vocab_size % tp_size == 0
        and target_vocab_size == global_vocab_size // tp_size
    ):
        runtime_layout = MarkovRuntimeLayout.VOCAB_SHARDED
    else:
        raise ValueError(
            f"{name} runtime vocab dimension {target_vocab_size} is neither the "
            "global size nor its TP shard"
        )

    return PreparedMarkovWeight(
        tensor=weight,
        runtime_layout=runtime_layout,
        global_shape=(global_vocab_size, weight.shape[1]),
    )
