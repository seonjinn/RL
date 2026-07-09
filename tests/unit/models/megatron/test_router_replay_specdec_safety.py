# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from nemo_rl.models.megatron.router_replay import (
    _local_layer_numbers_for_model,
    _router_replay_instances_for_model,
)


class _Router(torch.nn.Module):
    def __init__(self, *, layer_number: int, is_mtp_layer: bool) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.router_replay = SimpleNamespace()


def test_router_replay_discovery_excludes_mtp_routers() -> None:
    model = torch.nn.Module()
    model.base_router = _Router(layer_number=3, is_mtp_layer=False)
    model.mtp_router = _Router(layer_number=3, is_mtp_layer=True)

    instances = _router_replay_instances_for_model(model)

    assert instances == [(model.base_router.router_replay, 3)]
    assert _local_layer_numbers_for_model(model) == {3}
