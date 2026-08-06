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

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
    NcclReshardWeightSynchronizer,
)
from nemo_rl.weight_sync.refit_transforms import (
    RefitTransformPlan,
    RefitTransformRequest,
    TransformComponentSpec,
    build_plan_agreement,
)


_PARAM_NAME = "model.layers.0.mlp.down_proj.weight"


def test_nvfp4_nccl_request_is_explicitly_destination_owned() -> None:
    request = RefitTransformRequest(
        parameter_names=(_PARAM_NAME,),
        source_format="bf16",
        target_format="nvfp4_w4a16",
        transform_location="destination",
    )

    assert request.transform_location == "destination"


def _refit_info() -> dict[str, object]:
    plan = RefitTransformPlan(
        transform_id="bf16_to_mxfp8_e4m3_e8m0",
        components=(
            TransformComponentSpec("weight", (64, 64), "torch.float8_e4m3fn"),
            TransformComponentSpec("weight_scale", (64, 2), "torch.uint8"),
        ),
        finalize_scope="parameter",
    )
    agreement = build_plan_agreement({_PARAM_NAME: plan})
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": _PARAM_NAME,
                    "transform_id": plan.transform_id,
                    "finalize_scope": plan.finalize_scope,
                    "components": [
                        {
                            "role": component.role,
                            "global_shape": component.global_shape,
                            "dtype": component.dtype_name,
                        }
                        for component in plan.components
                    ],
                }
            ]
        },
        "refit_protocol_version": agreement["protocol_version"],
        "refit_component_count": agreement["component_count"],
        "plan_signature": agreement["plan_signature"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 2),
        ("component_count", 3),
        ("plan_signature", "destination"),
    ],
)
@patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray.get")
def test_init_communicator_rejects_destination_plan_mismatch_before_refit(
    _mock_ray_get: MagicMock,
    field: str,
    value: object,
) -> None:
    synchronizer = _build_synchronizer([], {})
    request = RefitTransformRequest(
        parameter_names=("model.layers.0.mlp.down_proj.weight",),
        source_format="bf16",
        target_format="mxfp8_e4m3_e8m0",
    )
    synchronizer._generation.prepare_refit_info.side_effect = [[request], None]
    synchronizer._policy.enable_refit_transforms.return_value = {
        "model.layers.0.mlp.down_proj.weight": {"dtype": "torch.float8_e4m3fn"}
    }
    source_info = _refit_info()
    synchronizer._policy.prepare_nccl_reshard_refit_info.side_effect = None
    synchronizer._policy.prepare_nccl_reshard_refit_info.return_value = source_info
    synchronizer._generation.prepare_nccl_reshard_refit_info.side_effect = None
    destination = {
        "protocol_version": source_info["refit_protocol_version"],
        "component_count": source_info["refit_component_count"],
        "plan_signature": source_info["plan_signature"],
    }
    destination[field] = value
    synchronizer._generation.prepare_nccl_reshard_refit_info.return_value = destination

    with pytest.raises(ValueError, match="refit plan agreement mismatch"):
        synchronizer.init_communicator()

    synchronizer._policy.nccl_reshard_refit.assert_not_called()
    synchronizer._generation.nccl_reshard_refit.assert_not_called()


def _build_synchronizer(
    events: list[str], updated_info: dict[str, object] | None
) -> NcclReshardWeightSynchronizer:
    bf16_info = {_PARAM_NAME: {"dtype": "torch.bfloat16"}}
    request = RefitTransformRequest(
        parameter_names=(_PARAM_NAME,),
        source_format="bf16",
        target_format="mxfp8_e4m3_e8m0",
    )
    mxfp8_info = updated_info
    nccl_reshard_info = _refit_info()

    policy = MagicMock()
    policy.cfg = {
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
        "generation": {"vllm_cfg": {}},
    }
    policy.init_collective.return_value = []
    policy.init_nccl_reshard_comm_group.return_value = []

    def policy_prepare_refit_info() -> dict[str, object]:
        events.append("policy.prepare_refit_info")
        return bf16_info

    def policy_enable_refit_transforms(
        *,
        requests: list[RefitTransformRequest],
    ) -> dict[str, object] | None:
        assert requests == [request]
        events.append("policy.enable_refit_transforms")
        return mxfp8_info

    def policy_prepare_nccl_reshard_refit_info(
        *_args: object,
    ) -> dict[str, object]:
        events.append("policy.prepare_nccl_reshard_refit_info")
        return nccl_reshard_info

    policy.prepare_refit_info.side_effect = policy_prepare_refit_info
    policy.enable_refit_transforms.side_effect = policy_enable_refit_transforms
    policy.prepare_nccl_reshard_refit_info.side_effect = (
        policy_prepare_nccl_reshard_refit_info
    )

    generation = MagicMock()
    generation.init_collective.return_value = []
    generation.init_nccl_reshard_comm_group.return_value = []

    def generation_prepare_refit_info(
        state_dict_info: dict[str, object],
    ) -> list[RefitTransformRequest] | None:
        if state_dict_info is bf16_info:
            events.append("generation.prepare_refit_info:bf16")
            return [request]
        assert state_dict_info is mxfp8_info
        events.append("generation.prepare_refit_info:mxfp8")
        return None

    def generation_prepare_nccl_reshard_refit_info(
        refit_info: dict[str, object],
    ) -> dict[str, object]:
        assert refit_info is nccl_reshard_info
        events.append("generation.prepare_nccl_reshard_refit_info")
        return {
            "protocol_version": refit_info["refit_protocol_version"],
            "component_count": refit_info["refit_component_count"],
            "plan_signature": refit_info["plan_signature"],
        }

    generation.prepare_refit_info.side_effect = generation_prepare_refit_info
    generation.prepare_nccl_reshard_refit_info.side_effect = (
        generation_prepare_nccl_reshard_refit_info
    )

    train_cluster = MagicMock()
    train_cluster.world_size.return_value = 1
    train_cluster.num_gpus_per_node = 1
    train_cluster.get_master_address_and_port.return_value = ("127.0.0.1", 29500)
    train_cluster.get_available_address_and_port.return_value = (
        "127.0.0.1",
        29501,
    )

    inference_cluster = MagicMock()
    inference_cluster.world_size.return_value = 1

    return NcclReshardWeightSynchronizer(
        policy,
        generation,
        train_cluster,
        inference_cluster,
    )


@patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray.get")
def test_init_communicator_completes_prequant_handshake_in_order(
    _mock_ray_get: MagicMock,
) -> None:
    events: list[str] = []
    updated_info = {
        "model.layers.0.mlp.down_proj.weight": {"dtype": "torch.float8_e4m3fn"}
    }
    synchronizer = _build_synchronizer(events, updated_info)

    synchronizer.init_communicator()

    assert events == [
        "policy.prepare_refit_info",
        "generation.prepare_refit_info:bf16",
        "policy.enable_refit_transforms",
        "generation.prepare_refit_info:mxfp8",
        "policy.prepare_nccl_reshard_refit_info",
        "generation.prepare_nccl_reshard_refit_info",
    ]


@patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray.get")
def test_init_communicator_rejects_missing_prequant_metadata(
    _mock_ray_get: MagicMock,
) -> None:
    events: list[str] = []
    synchronizer = _build_synchronizer(events, None)

    with pytest.raises(RuntimeError, match="did not return updated metadata"):
        synchronizer.init_communicator()

    assert events == [
        "policy.prepare_refit_info",
        "generation.prepare_refit_info:bf16",
        "policy.enable_refit_transforms",
    ]


@patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray.get")
def test_init_communicator_preserves_untransformed_metadata_path(
    _mock_ray_get: MagicMock,
) -> None:
    events: list[str] = []
    synchronizer = _build_synchronizer(events, {})

    def generation_prepare_refit_info(
        _state_dict_info: dict[str, object],
    ) -> None:
        events.append("generation.prepare_refit_info:untransformed")

    synchronizer._generation.prepare_refit_info.side_effect = (
        generation_prepare_refit_info
    )

    synchronizer.init_communicator()

    assert events == [
        "policy.prepare_refit_info",
        "generation.prepare_refit_info:untransformed",
        "policy.prepare_nccl_reshard_refit_info",
        "generation.prepare_nccl_reshard_refit_info",
    ]
    synchronizer._policy.enable_refit_transforms.assert_not_called()
