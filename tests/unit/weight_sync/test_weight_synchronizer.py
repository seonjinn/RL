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

"""Unit tests for the WeightSynchronizer abstraction and its implementations."""

from unittest.mock import MagicMock, call, patch

import pytest

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.weight_sync.collective_weight_synchronizer import (
    CollectiveWeightSynchronizer,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.http_weight_synchronizer import (
    HTTPWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer, initialize_refit_metadata
from nemo_rl.weight_sync.ipc_weight_synchronizer import (
    IPCWeightSynchronizer,
)
from nemo_rl.weight_sync.refit_transforms import RefitTransformRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_policy(**overrides):
    policy = MagicMock()
    policy.offload_before_refit.return_value = None
    policy.offload_after_refit.return_value = None
    policy.prepare_refit_info.return_value = {"layer_0": {"shape": [4096, 4096]}}
    policy.stream_weights_via_ipc_zmq.return_value = [MagicMock()]
    policy.stream_weights_via_http.return_value = [MagicMock()]
    policy.broadcast_weights_for_collective.return_value = [MagicMock()]
    policy.init_collective.return_value = [MagicMock()]
    policy.get_free_memory_bytes.return_value = 1024**3  # 1 GB
    for k, v in overrides.items():
        setattr(policy, k, v)
    return policy


def _mock_generation(**overrides):
    gen = MagicMock()
    gen.cfg = {}
    gen.prepare_for_generation.return_value = True
    gen.finish_generation.return_value = True
    gen.prepare_refit_info.return_value = None
    gen.update_weights_via_ipc_zmq.return_value = [MagicMock()]
    gen.update_weights_from_collective.return_value = [MagicMock()]
    gen.get_rollout_engine_urls.return_value = ["http://localhost:30000"]
    gen.init_collective.return_value = [MagicMock()]
    for k, v in overrides.items():
        setattr(gen, k, v)
    return gen


def _transform_request(*names: str) -> RefitTransformRequest:
    return RefitTransformRequest(
        parameter_names=tuple(sorted(set(names))),
        source_format="bf16",
        target_format="mxfp8_e4m3_e8m0",
    )


def test_legacy_mxfp8_request_defaults_to_source_transform() -> None:
    assert _transform_request("layer_0").transform_location == "source"


def _mock_cluster(world_size=4, ip="127.0.0.1", port=29500):
    cluster = MagicMock()
    cluster.world_size.return_value = world_size
    cluster.get_master_address_and_port.return_value = (ip, port)
    return cluster


# ---------------------------------------------------------------------------
# WeightSynchronizer ABC contract
# ---------------------------------------------------------------------------


class TestWeightSynchronizerABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            WeightSynchronizer()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        class IncompleteSync(WeightSynchronizer):
            pass

        with pytest.raises(TypeError):
            IncompleteSync()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Shared refit metadata handshake
# ---------------------------------------------------------------------------


class TestInitializeRefitMetadata:
    def test_destination_owned_requests_do_not_require_megatron_or_refresh(self):
        policy = _mock_policy()
        policy.cfg = {
            "dtensor_cfg": {"enabled": True},
            "megatron_cfg": {"enabled": False},
        }
        generation = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value
        request = RefitTransformRequest(
            parameter_names=("layer_0",),
            source_format="bf16",
            target_format="nvfp4_w4a16",
            transform_location="destination",
        )
        generation.prepare_refit_info.return_value = [request]

        initialize_refit_metadata(policy, generation)

        generation.prepare_refit_info.assert_called_once_with(state_dict_info)
        policy.enable_refit_transforms.assert_not_called()

    def test_forwards_structured_transform_requests_and_refreshes_metadata(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        generation = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value
        updated_info = {"layer_0": {"dtype": "torch.float8_e4m3fn"}}
        request = RefitTransformRequest(
            parameter_names=("layer_0",),
            source_format="bf16",
            target_format="mxfp8_e4m3_e8m0",
        )
        generation.prepare_refit_info.side_effect = [[request], None]
        policy.enable_refit_transforms.return_value = updated_info

        initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_called_once_with(requests=[request])
        assert generation.prepare_refit_info.call_args_list == [
            call(state_dict_info),
            call(updated_info),
        ]

    def test_mixed_requests_pass_full_set_to_source_transform(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        generation = _mock_generation()
        updated_info = {"layer_0": {"dtype": "torch.uint8"}}
        requests = [
            RefitTransformRequest(
                parameter_names=("layer_0",),
                source_format="bf16",
                target_format="nvfp4_w4a16",
                transform_location="source",
            ),
            RefitTransformRequest(
                parameter_names=("layer_1",),
                source_format="bf16",
                target_format="nvfp4_w4a16",
                transform_location="destination",
            ),
        ]
        generation.prepare_refit_info.side_effect = [requests, None]
        policy.enable_refit_transforms.return_value = updated_info

        initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_called_once_with(requests=requests)
        assert generation.prepare_refit_info.call_args_list == [
            call(policy.prepare_refit_info.return_value),
            call(updated_info),
        ]

    def test_structured_transform_request_uses_consistent_non_megatron_error(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": False}}
        generation = _mock_generation()
        generation.prepare_refit_info.return_value = [
            RefitTransformRequest(
                parameter_names=("layer_0",),
                source_format="bf16",
                target_format="mxfp8_e4m3_e8m0",
                transform_location="source",
            )
        ]

        with pytest.raises(ValueError, match="requires the Megatron policy backend"):
            initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_not_called()

    def test_rejects_malformed_generation_transform_response(self):
        policy = _mock_policy()
        generation = _mock_generation()
        generation.prepare_refit_info.return_value = [object()]

        with pytest.raises(
            TypeError,
            match="must be RefitTransformRequest instances",
        ):
            initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_not_called()

    def test_returns_when_generation_does_not_request_prequantization(self):
        policy = _mock_policy()
        generation = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value

        initialize_refit_metadata(policy, generation)

        generation.prepare_refit_info.assert_called_once_with(state_dict_info)
        policy.enable_refit_transforms.assert_not_called()

    def test_rejects_prequantization_without_megatron(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": False}}
        generation = _mock_generation()
        generation.prepare_refit_info.return_value = [_transform_request("layer_0")]

        with pytest.raises(ValueError, match="requires the Megatron policy backend"):
            initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_not_called()

    def test_refreshes_generation_metadata_after_prequantization(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        generation = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value
        updated_info = {
            "layer_0": {
                "shape": [4096, 4096],
                "dtype": "float8_e4m3fn",
            },
            "layer_0_scale_from_checkpoint": {
                "shape": [4096, 128],
                "dtype": "uint8",
            },
        }
        request = _transform_request("layer_0")
        generation.prepare_refit_info.side_effect = [[request], None]
        policy.enable_refit_transforms.return_value = updated_info

        initialize_refit_metadata(policy, generation)

        policy.enable_refit_transforms.assert_called_once_with(requests=[request])
        assert generation.prepare_refit_info.call_args_list == [
            call(state_dict_info),
            call(updated_info),
        ]

    def test_rejects_missing_prequantized_metadata(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        policy.enable_refit_transforms.return_value = None
        generation = _mock_generation()
        generation.prepare_refit_info.return_value = [_transform_request("layer_0")]

        with pytest.raises(
            RuntimeError,
            match="did not return updated metadata",
        ):
            initialize_refit_metadata(policy, generation)


# ---------------------------------------------------------------------------
# IPCWeightSynchronizer
# ---------------------------------------------------------------------------


class TestIPCWeightSynchronizer:
    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_calls_full_lifecycle(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.offload_before_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        policy.stream_weights_via_ipc_zmq.assert_called_once()
        gen.update_weights_via_ipc_zmq.assert_called_once()
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_passes_kv_scales(self, mock_ray: MagicMock) -> None:
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        kv_scales = {"layer.0": 0.5}

        sync.sync_weights(kv_scales=kv_scales)

        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        assert call_kwargs.kwargs["kv_scales"] == kv_scales

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_sync_weights_raises_on_failure(self, mock_ray):
        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_fixed_buffer_size(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen, refit_buffer_size_gb=2)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        assert call_kwargs.kwargs["buffer_size_bytes"] == 2 * (1024**3)

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_dynamic_buffer_size(self, mock_ray, monkeypatch):
        monkeypatch.delenv("NRL_REFIT_BUFFER_MEMORY_RATIO", raising=False)
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        policy.get_free_memory_bytes.return_value = 10 * (1024**3)
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_ipc_zmq.call_args
        expected = int(10 * (1024**3) * 0.3)
        assert call_kwargs.kwargs["buffer_size_bytes"] == expected

    def test_mark_stale(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync._stale = False
        assert not sync.is_stale
        sync.mark_stale()
        assert sync.is_stale

    def test_init_communicator(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        sync.init_communicator()
        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()

    def test_init_communicator_completes_prequantization_handshake(self):
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        updated_info = {
            "layer_0": {
                "shape": [4096, 4096],
                "dtype": "float8_e4m3fn",
            }
        }
        policy.enable_refit_transforms.return_value = updated_info
        gen = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value
        request = _transform_request("layer_0")
        gen.prepare_refit_info.side_effect = [[request], None]
        sync = IPCWeightSynchronizer(policy, gen)

        sync.init_communicator()

        policy.enable_refit_transforms.assert_called_once_with(requests=[request])
        assert gen.prepare_refit_info.call_args_list == [
            call(state_dict_info),
            call(updated_info),
        ]

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        """offload_after_refit and kv_cache prep run even when transfer raises."""
        mock_ray.get.side_effect = RuntimeError("IPC transfer exploded")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="IPC transfer exploded"):
            sync.sync_weights()

        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        assert sync.is_stale

    def test_negative_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen, refit_buffer_size_gb=-1)
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_invalid_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "not_a_number")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be a valid float"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
    def test_zero_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = IPCWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be > 0"):
            sync._compute_buffer_size()


# ---------------------------------------------------------------------------
# HTTPWeightSynchronizer
# ---------------------------------------------------------------------------


class TestHTTPWeightSynchronizer:
    def test_sync_weights_rejects_kv_scales(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        with pytest.raises(AssertionError, match="does not support"):
            sync.sync_weights(kv_scales={"layer.0": 0.5})

        policy.offload_before_refit.assert_not_called()
        gen.prepare_for_generation.assert_not_called()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_sync_weights_calls_full_lifecycle(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.offload_before_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["weights"])
        policy.stream_weights_via_http.assert_called_once()
        gen.get_rollout_engine_urls.assert_called_once()
        call_kwargs = policy.stream_weights_via_http.call_args
        assert call_kwargs.kwargs["rollout_engine_urls"] == ["http://localhost:30000"]
        assert call_kwargs.kwargs["buffer_size_bytes"] == int((1024**3) * 0.3)
        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_fixed_buffer_size(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen, refit_buffer_size_gb=2)

        sync.sync_weights()
        call_kwargs = policy.stream_weights_via_http.call_args
        assert call_kwargs.kwargs["rollout_engine_urls"] == ["http://localhost:30000"]
        assert call_kwargs.kwargs["buffer_size_bytes"] == 2 * (1024**3)

    def test_mark_stale(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        sync._stale = False
        assert not sync.is_stale
        sync.mark_stale()
        assert sync.is_stale

    def test_init_communicator(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        sync.init_communicator()
        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_phase_restoration_on_transfer_failure(self, mock_ray):
        """offload_after_refit and kv_cache prep run even when transfer raises."""
        mock_ray.get.side_effect = RuntimeError("HTTP transfer exploded")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)

        with pytest.raises(RuntimeError, match="HTTP transfer exploded"):
            sync.sync_weights()

        policy.offload_after_refit.assert_called_once()
        gen.prepare_for_generation.assert_any_call(tags=["kv_cache"])
        assert sync.is_stale

    def test_negative_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen, refit_buffer_size_gb=-1)
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_invalid_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "not_a_number")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be a valid float"):
            sync._compute_buffer_size()

    @patch("nemo_rl.weight_sync.http_weight_synchronizer.ray")
    def test_zero_env_ratio_raises(self, mock_ray, monkeypatch):
        monkeypatch.setenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0")
        policy = _mock_policy()
        gen = _mock_generation()
        sync = HTTPWeightSynchronizer(policy, gen)
        with pytest.raises(ValueError, match="must be > 0"):
            sync._compute_buffer_size()


# ---------------------------------------------------------------------------
# CollectiveWeightSynchronizer
# ---------------------------------------------------------------------------


class TestCollectiveWeightSynchronizer:
    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_calls_broadcast_and_receive(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        train_cluster = _mock_cluster(world_size=4)
        inference_cluster = _mock_cluster(world_size=2)
        sync = CollectiveWeightSynchronizer(
            policy, gen, train_cluster, inference_cluster
        )

        assert sync.is_stale
        sync.sync_weights()
        assert not sync.is_stale

        policy.broadcast_weights_for_collective.assert_called_once()
        gen.update_weights_from_collective.assert_called_once()

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_passes_kv_scales(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = CollectiveWeightSynchronizer(
            policy, gen, _mock_cluster(), _mock_cluster()
        )
        kv_scales = {"layer.0": 1.0}

        sync.sync_weights(kv_scales=kv_scales)
        call_kwargs = policy.broadcast_weights_for_collective.call_args
        assert call_kwargs.kwargs["kv_scales"] == kv_scales

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_sync_weights_raises_on_failure(self, mock_ray):
        mock_ray.get.side_effect = [
            None,  # futures_train
            [False],  # futures_inference -- update failed
        ]
        policy = _mock_policy()
        gen = _mock_generation()
        sync = CollectiveWeightSynchronizer(
            policy, gen, _mock_cluster(), _mock_cluster()
        )

        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_init_communicator_sets_up_collective(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        gen = _mock_generation()
        train_cluster = _mock_cluster(world_size=4, ip="10.0.0.1", port=29500)
        inference_cluster = _mock_cluster(world_size=2)

        sync = CollectiveWeightSynchronizer(
            policy, gen, train_cluster, inference_cluster
        )
        sync.init_communicator()

        policy.prepare_refit_info.assert_called_once()
        gen.prepare_refit_info.assert_called_once()
        policy.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 6, train_world_size=4
        )
        gen.init_collective.assert_called_once_with(
            "10.0.0.1", 29500, 6, train_world_size=4
        )

    @patch("nemo_rl.weight_sync.collective_weight_synchronizer.ray")
    def test_init_communicator_prequantizes_before_collective_setup(self, mock_ray):
        mock_ray.get.return_value = [True]
        policy = _mock_policy()
        policy.cfg = {"megatron_cfg": {"enabled": True}}
        updated_info = {
            "layer_0": {
                "shape": [4096, 4096],
                "dtype": "float8_e4m3fn",
            }
        }
        policy.enable_refit_transforms.return_value = updated_info
        gen = _mock_generation()
        state_dict_info = policy.prepare_refit_info.return_value
        request = _transform_request("layer_0")
        gen.prepare_refit_info.side_effect = [[request], None]
        sync = CollectiveWeightSynchronizer(
            policy,
            gen,
            _mock_cluster(world_size=4),
            _mock_cluster(world_size=2),
        )

        sync.init_communicator()

        policy.enable_refit_transforms.assert_called_once_with(requests=[request])
        assert gen.prepare_refit_info.call_args_list == [
            call(state_dict_info),
            call(updated_info),
        ]
        policy.init_collective.assert_called_once()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_colocated_vllm_returns_ipc(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=VLLM_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, IPCWeightSynchronizer)

    def test_colocated_sglang_returns_http(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=SGLANG_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, HTTPWeightSynchronizer)

    def test_colocated_megatron_returns_ipc(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=MEGATRON_BACKEND,
            colocated=True,
        )
        assert isinstance(sync, IPCWeightSynchronizer)

    def test_non_colocated_vllm_returns_collective(self):
        policy = _mock_policy()
        gen = _mock_generation()
        sync = create_weight_synchronizer(
            policy=policy,
            generation=gen,
            generation_backend=VLLM_BACKEND,
            colocated=False,
            train_cluster=_mock_cluster(),
            inference_cluster=_mock_cluster(),
        )
        assert isinstance(sync, CollectiveWeightSynchronizer)

    def test_non_colocated_sglang_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(NotImplementedError, match="SGLang"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=SGLANG_BACKEND,
                colocated=False,
            )

    def test_non_colocated_missing_clusters_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="train_cluster"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            )

    def test_unknown_backend_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="Unknown generation backend"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend="vlllm",
                colocated=True,
            )

    def test_negative_refit_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=True,
                refit_buffer_size_gb=-1,
            )

    def test_zero_refit_buffer_size_raises(self):
        policy = _mock_policy()
        gen = _mock_generation()
        with pytest.raises(ValueError, match="refit_buffer_size_gb must be > 0"):
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=True,
                refit_buffer_size_gb=0,
            )
