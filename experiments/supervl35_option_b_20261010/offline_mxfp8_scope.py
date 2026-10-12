from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence


EXPERT_WEIGHT = re.compile(
    r"^language_model\.backbone\.layers\.(?P<layer>\d+)\.mixer\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>up_proj|down_proj)\.weight$"
)


def build_scope(
    names: Iterable[str],
    layer_types: Sequence[str],
    *,
    first_bf16: int,
    last_bf16: int,
) -> tuple[set[str], list[str]]:
    layer_count = len(layer_types)
    if first_bf16 < 0 or last_bf16 < 0 or first_bf16 + last_bf16 >= layer_count:
        raise ValueError("BF16 layer counts must leave at least one middle layer")

    names = set(names)
    expert_weights: dict[tuple[int, int], set[str]] = defaultdict(set)
    targets: set[str] = set()
    ignored = set()
    for name in names:
        match = EXPERT_WEIGHT.fullmatch(name)
        if match is None:
            if name.endswith(".weight"):
                ignored.add(name.removesuffix(".weight"))
            continue

        layer = int(match["layer"])
        expert = int(match["expert"])
        if layer >= layer_count or layer_types[layer] != "moe":
            raise ValueError(f"Expert weight belongs to non-MoE layer: {name}")
        expert_weights[layer, expert].add(match["projection"])
        if first_bf16 <= layer < layer_count - last_bf16:
            targets.add(name)
        else:
            ignored.add(f"*layers.{layer}.mixer.experts")

    for (layer, expert), projections in expert_weights.items():
        for projection in ("up_proj", "down_proj"):
            if projection not in projections:
                raise ValueError(
                    f"Layer {layer} expert {expert} is missing {projection}"
                )
    if not targets:
        raise ValueError("No middle-layer routed expert weights were selected")
    return targets, sorted(ignored)
