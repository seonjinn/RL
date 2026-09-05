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

"""Tests for checkpoint-engine weight synchronization and factory routing."""

from unittest.mock import MagicMock, call, patch

import pytest

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer import (
    CheckpointEngineWeightSynchronizer,
    _ordered_generation_metadata,
    _sort_ranked_metadata,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.refit_supervisor import normalize_current_refit_result


def _mock_policy(**overrides):
    policy = MagicMock()
    policy.offload_before_refit.return_value = None
    policy.offload_after_refit.return_value = None
    policy.prepare_refit_info.return_value = {"layer_0": {"shape": [4096, 4096]}}
    policy.stream_weights_via_ipc_zmq.return_value = [MagicMock()]
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
    gen.init_collective.return_value = [MagicMock()]
    for k, v in overrides.items():
        setattr(gen, k, v)
    return gen


def _checkpoint_engine_cfg(
    *,
    release_after_refit=False,
    backend="test_backend",
    bucket_memory_ratio=0.05,
    device="cpu",
):
    return {
        "backend": backend,
        "update_weights_bucket_memory_ratio": bucket_memory_ratio,
        "engine_kwargs": {
            backend: {
                "device": device,
                "release_after_refit": release_after_refit,
            }
        },
    }


def _nixl_refit_cfg(*, release_after_refit=False):
    return {
        "refit_transport": "nixl",
        "refit_cfg": {
            "nixl": {
                "device": "cpu",
                "release_after_refit": release_after_refit,
            }
        },
        "vllm_cfg": {"async_engine": False},
    }


class _CheckpointWorkerGroup:
    def __init__(self, role, *, worker_count, dp_size=1):
        self.role = role
        self.workers = [object() for _ in range(worker_count)]
        self.dp_size = dp_size
        self.calls = []

    def _refs(self, checkpoint_method, count):
        return [
            f"{self.role}-{checkpoint_method}-{participant_index}"
            for participant_index in range(count)
        ]

    def run_all_workers_single_data(self, method_name, **kwargs):
        self.calls.append((method_name, kwargs["checkpoint_method"]))
        participant_count = (
            self.dp_size if kwargs.get("run_rank_0_only_axes") else len(self.workers)
        )
        return self._refs(kwargs["checkpoint_method"], participant_count)

    def run_all_workers_multiple_data(self, method_name, **kwargs):
        self.calls.append(
            (
                method_name,
                kwargs["common_kwargs"]["checkpoint_method"],
                kwargs["method_args"],
            )
        )
        return self._refs(
            kwargs["common_kwargs"]["checkpoint_method"],
            len(kwargs["method_args"]),
        )


def _void_phase_results():
    return [None, None, [None, None], [None, None]]


def _memory_phase_results():
    gibibyte = 1024**3
    return [
        80 * gibibyte,
        96 * gibibyte,
        [64 * gibibyte, 72 * gibibyte],
        [88 * gibibyte, 104 * gibibyte],
    ]


def _metadata_phase_results():
    return [
        {"rank": 1, "id": "policy-1"},
        {"rank": 0, "id": "policy-0"},
        [
            {"rank": 1, "id": "generation-1"},
            {"rank": 0, "id": "generation-0"},
        ],
        [
            {"rank": 1, "id": "generation-3"},
            {"rank": 0, "id": "generation-2"},
        ],
    ]


def _checkpoint_sync(
    mock_ray,
    *,
    async_engine=False,
    release_after_refit=False,
    cycles=1,
    checkpoint_engine_config=None,
    refit_timeout_s=None,
):
    # One return value per ray.get() call, in order:
    #   1. total GPU memory (policy + generation)
    #   2. init_checkpoint_engine (policy + generation)
    #   3. prepare_checkpoint_engine (two policy ranks, then two rollout DP
    #      leaders whose results each contain two model-parallel ranks)
    #   4. init_checkpoint_engine_process_group (policy + generation)
    #   5. finalize_checkpoint_engine (policy + generation)
    mock_ray.get.side_effect = [_memory_phase_results()] + [
        item
        for _ in range(cycles)
        for item in (
            _void_phase_results(),
            _metadata_phase_results(),
            _void_phase_results(),
            _void_phase_results(),
        )
    ]
    policy = _mock_policy()
    policy.worker_group = _CheckpointWorkerGroup("policy", worker_count=2)
    checkpoint_engine_config = checkpoint_engine_config or _checkpoint_engine_cfg(
        release_after_refit=release_after_refit
    )
    gen = _mock_generation(cfg={"vllm_cfg": {"async_engine": async_engine}})
    gen.dp_size = 2
    gen.worker_group = _CheckpointWorkerGroup("generation", worker_count=4, dp_size=2)
    return CheckpointEngineWeightSynchronizer(
        policy,
        gen,
        checkpoint_engine_config,
        refit_timeout_s=refit_timeout_s,
    )


class TestCheckpointEngineWeightSynchronizer:
    @pytest.mark.parametrize(
        ("configured_timeout_s", "expected_timeout_s"),
        [(47.0, 47.0), (None, 300.0)],
        ids=["explicit", "compatibility-default"],
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_refit_timeout_is_finite_and_normalized_at_construction(
        self, mock_ray, configured_timeout_s, expected_timeout_s
    ):
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=configured_timeout_s)

        assert sync._refit_timeout_s == expected_timeout_s
        mock_ray.get.assert_not_called()

    @pytest.mark.parametrize("invalid_timeout_s", [0, -1, float("inf"), True, "3"])
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_invalid_refit_timeout_fails_before_any_rpc(
        self, mock_ray, invalid_timeout_s
    ):
        with pytest.raises(ValueError, match="timeout_s"):
            _checkpoint_sync(mock_ray, refit_timeout_s=invalid_timeout_s)

        mock_ray.get.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_startup_waves_spend_one_monotonic_deadline(self, mock_ray):
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=47.0)

        with patch(
            "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.monotonic",
            side_effect=[float(value) for value in range(100, 111)],
            create=True,
        ):
            sync.init_communicator()

        assert [entry.kwargs["timeout"] for entry in mock_ray.get.call_args_list] == [
            44.0,
            42.0,
            40.0,
            38.0,
        ]

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_refit_metadata_preparation_spends_the_shared_startup_deadline(
        self, mock_ray
    ):
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=47.0)

        with patch(
            "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.monotonic",
            side_effect=[float(value) for value in range(100, 120)],
            create=True,
        ):
            sync.init_communicator()

        policy_timeout_s = sync._policy.prepare_refit_info.call_args.kwargs[
            "refit_timeout_s"
        ]
        generation_timeout_s = sync._generation.prepare_refit_info.call_args.kwargs[
            "refit_timeout_s"
        ]
        assert policy_timeout_s == 46.0
        assert generation_timeout_s == 45.0
        assert mock_ray.get.call_args_list[0].kwargs["timeout"] == 44.0

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_startup_deadline_expiry_stops_before_engine_initialization(self, mock_ray):
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=47.0)

        with patch(
            "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.monotonic",
            side_effect=[100.0, 101.0, 102.0, 148.0],
            create=True,
        ):
            with pytest.raises(TimeoutError, match="bucket-memory-discovery"):
                sync.init_communicator()

        assert sync.is_stale
        assert not sync._checkpoint_engine_ready
        assert (
            "checkpoint_engine_rpc",
            "init_checkpoint_engine",
        ) not in sync._policy.worker_group.calls

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_partial_startup_cleanup_is_bounded_and_preserves_base_exception(
        self, mock_ray
    ):
        class FatalSetupFailure(BaseException):
            pass

        primary_failure = FatalSetupFailure("engine initialization failed")
        cleanup_failure = RuntimeError("finalize also failed")
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=47.0)
        mock_ray.get.side_effect = [
            _memory_phase_results(),
            primary_failure,
            cleanup_failure,
        ]

        with pytest.raises(FatalSetupFailure) as caught:
            sync.init_communicator()

        assert caught.value is primary_failure
        assert sync.is_stale
        assert not sync._checkpoint_engine_ready
        assert mock_ray.get.call_args_list[-1].kwargs["timeout"] <= 10.0
        assert mock_ray.get.call_args_list[-1].kwargs["timeout"] > 0.0
        assert sync._policy.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_startup_rejects_missing_participant_result(self, mock_ray):
        sync = _checkpoint_sync(mock_ray)
        mock_ray.get.side_effect = [
            _memory_phase_results(),
            [None],
            _void_phase_results(),
        ]

        with pytest.raises(RuntimeError, match="engine-initialization.*4.*1"):
            sync.init_communicator()

        assert sync.is_stale
        assert not sync._checkpoint_engine_ready

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_startup_rejects_non_void_generation_ack(self, mock_ray):
        sync = _checkpoint_sync(mock_ray)
        mock_ray.get.side_effect = [
            _memory_phase_results(),
            [None, None, [None, False], [None, None]],
            _void_phase_results(),
        ]

        with pytest.raises(RuntimeError, match="engine-initialization.*exact None"):
            sync.init_communicator()

        assert sync.is_stale
        assert not sync._checkpoint_engine_ready

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_uses_minimum_total_memory_and_is_cached(self, mock_ray, capsys):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=0.125)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)
        mock_ray.get.side_effect = None
        mock_ray.get.return_value = [
            96 * 1024**3,
            72 * 1024**3,
            [64 * 1024**3, 80 * 1024**3],
            [88 * 1024**3, 104 * 1024**3],
        ]

        assert sync._resolve_bucket_size_bytes() == 8192 * 1024**2
        assert sync._resolve_bucket_size_bytes() == 8192 * 1024**2
        mock_ray.get.assert_called_once()
        assert sync._policy.worker_group.calls == [
            ("checkpoint_engine_rpc", "checkpoint_engine_total_memory_bytes")
        ]
        assert sync._generation.worker_group.calls == [
            ("checkpoint_engine_rpc", "checkpoint_engine_total_memory_bytes")
        ]
        assert "8192 MiB per buffer" in capsys.readouterr().out

    @pytest.mark.parametrize("memory_ratio", ["invalid", 0, 1])
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_rejects_invalid_ratio(self, mock_ray, memory_ratio):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=memory_ratio)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)

        with pytest.raises(ValueError, match="update_weights_bucket_memory_ratio"):
            sync._resolve_bucket_size_bytes()
        mock_ray.get.assert_not_called()

    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_bucket_rejects_sub_mibibyte_result(self, mock_ray):
        config = _checkpoint_engine_cfg(bucket_memory_ratio=0.05)
        sync = _checkpoint_sync(mock_ray, checkpoint_engine_config=config)
        mock_ray.get.side_effect = None
        mock_ray.get.return_value = [
            8 * 1024**2,
            8 * 1024**2,
            [8 * 1024**2, 8 * 1024**2],
            [8 * 1024**2, 8 * 1024**2],
        ]

        with pytest.raises(ValueError, match="less than 1 MiB"):
            sync._resolve_bucket_size_bytes()

    def test_sort_ranked_metadata_orders_by_rank(self):
        metadata = [{"rank": 2}, {"rank": 0}, {"rank": 1}]

        assert _sort_ranked_metadata(metadata) == [
            {"rank": 0},
            {"rank": 1},
            {"rank": 2},
        ]

    def test_ordered_generation_metadata_handles_dp_groups_with_colliding_ranks(self):
        # Two vLLM DP groups (engines), each reporting engine-local ranks 0/1 that
        # collide across groups; collective_rpc may return them out of local order.
        # The result must be global rollout-rank order: [g0r0, g0r1, g1r0, g1r1].
        generation_results = [
            [{"rank": 1, "id": "g0r1"}, {"rank": 0, "id": "g0r0"}],
            [{"rank": 1, "id": "g1r1"}, {"rank": 0, "id": "g1r0"}],
        ]

        ordered = _ordered_generation_metadata(generation_results)

        assert [m["id"] for m in ordered] == ["g0r0", "g0r1", "g1r0", "g1r1"]
        # A single global sort over colliding ranks would instead interleave the
        # groups ([g0r0, g1r0, g0r1, g1r1]) and mis-pair policy<->rollout workers.

    def test_ordered_generation_metadata_single_group(self):
        generation_results = [[{"rank": 1, "id": "r1"}, {"rank": 0, "id": "r0"}]]

        ordered = _ordered_generation_metadata(generation_results)

        assert [m["id"] for m in ordered] == ["r0", "r1"]

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sync_weights_runs_checkpoint_engine_lifecycle(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(mock_ray)

        sync.init_communicator()
        sync.sync_weights(kv_scales={"kv": 1.0})

        assert not sync.is_stale
        sync._policy.prepare_refit_info.assert_called_once()
        sync._generation.prepare_refit_info.assert_called_once()
        assert (
            "checkpoint_engine_rpc",
            "send_weights_via_checkpoint_engine",
        ) in sync._policy.worker_group.calls
        assert (
            "checkpoint_engine_rpc",
            "update_weights_from_checkpoint_engine",
        ) in sync._generation.worker_group.calls
        expected_metadata = [
            {"rank": 0, "id": "policy-0"},
            {"rank": 1, "id": "policy-1"},
            {"rank": 0, "id": "generation-0"},
            {"rank": 1, "id": "generation-1"},
            {"rank": 0, "id": "generation-2"},
            {"rank": 1, "id": "generation-3"},
        ]
        assert sync._generation.worker_group.calls[3][2] == [
            (0, 2, 4, expected_metadata),
            (2, 2, 4, expected_metadata),
        ]
        sync.shutdown()
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )
        assert sync._policy.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sync_supervises_producer_and_consumer_as_one_operation(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(mock_ray, refit_timeout_s=47.0)
        sync.init_communicator()

        sync.sync_weights(kv_scales={"kv": 1.0})

        assert mock_supervisor.call_args == call(
            operation="test_backend-checkpoint-engine-weight-sync",
            producer_futures=[
                "policy-send_weights_via_checkpoint_engine-0",
                "policy-send_weights_via_checkpoint_engine-1",
            ],
            consumer_futures=[
                "generation-update_weights_from_checkpoint_engine-0",
                "generation-update_weights_from_checkpoint_engine-1",
            ],
            result_normalizer=normalize_current_refit_result,
            timeout_s=47.0,
        )
        assert not sync.is_stale

    @pytest.mark.parametrize("role", ["policy", "generation"])
    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_transfer_rejects_missing_participant_before_supervision(
        self, mock_ray, mock_supervisor, role
    ):
        sync = _checkpoint_sync(mock_ray)
        sync.init_communicator()
        original_run_policy = sync._run_policy
        original_run_generation = sync._run_generation

        def run_policy(checkpoint_method, **kwargs):
            refs = original_run_policy(checkpoint_method, **kwargs)
            return refs[:1] if role == "policy" else refs

        def run_generation(checkpoint_method, method_args=()):
            refs = original_run_generation(checkpoint_method, method_args)
            return refs[:1] if role == "generation" else refs

        sync._run_policy = run_policy
        sync._run_generation = run_generation

        with pytest.raises(RuntimeError, match=rf"weight-transfer.*2 {role}.*1"):
            sync.sync_weights()

        assert sync.is_stale
        mock_supervisor.assert_not_called()

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_transfer_failure_is_not_delayed_by_release_cleanup(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(mock_ray, release_after_refit=True)
        sync.init_communicator()
        failure = RuntimeError("consumer failed")
        mock_supervisor.side_effect = failure

        with pytest.raises(RuntimeError, match="consumer failed") as caught:
            sync.sync_weights()

        assert caught.value is failure
        assert sync.is_stale
        assert sync._checkpoint_engine_ready
        assert (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        ) not in sync._policy.worker_group.calls

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_success_release_finalization_has_a_separate_small_deadline(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(
            mock_ray,
            release_after_refit=True,
            refit_timeout_s=47.0,
        )
        sync.init_communicator()

        sync.sync_weights()

        finalize_timeout_s = mock_ray.get.call_args_list[-1].kwargs["timeout"]
        assert 0.0 < finalize_timeout_s <= 10.0
        assert finalize_timeout_s < sync._refit_timeout_s
        assert not sync.is_stale
        assert not sync._checkpoint_engine_ready

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_finalize_failure_keeps_sync_stale_and_engine_retryable(
        self, mock_ray, mock_supervisor
    ):
        finalize_failure = RuntimeError("finalize failed")
        sync = _checkpoint_sync(mock_ray, release_after_refit=True)
        mock_ray.get.side_effect = [
            _memory_phase_results(),
            _void_phase_results(),
            _metadata_phase_results(),
            _void_phase_results(),
            finalize_failure,
        ]
        sync.init_communicator()

        with pytest.raises(RuntimeError, match="finalize failed") as caught:
            sync.sync_weights()

        assert caught.value is finalize_failure
        assert sync.is_stale
        assert sync._checkpoint_engine_ready
        assert 0.0 < mock_ray.get.call_args_list[-1].kwargs["timeout"] <= 10.0

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_release_after_refit_reprepares_each_sync(self, mock_ray, mock_supervisor):
        sync = _checkpoint_sync(mock_ray, release_after_refit=True, cycles=2)

        sync.init_communicator()
        sync.sync_weights()
        assert not sync._checkpoint_engine_ready

        sync.sync_weights()
        assert not sync._checkpoint_engine_ready
        assert (
            sync._policy.worker_group.calls.count(
                ("checkpoint_engine_rpc", "prepare_checkpoint_engine")
            )
            == 2
        )
        assert (
            sync._policy.worker_group.calls.count(
                ("checkpoint_engine_rpc", "finalize_checkpoint_engine")
            )
            == 2
        )

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sync_weights_does_not_run_colocated_phase_transitions(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(mock_ray)

        sync.init_communicator()
        sync.sync_weights()

        sync._policy.offload_before_refit.assert_not_called()
        sync._policy.offload_after_refit.assert_not_called()
        sync._policy.prepare_for_training.assert_not_called()
        sync._generation.prepare_for_generation.assert_not_called()

    @patch(
        "nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.supervise_refit_futures"
    )
    @patch("nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer.ray")
    def test_sync_weights_raises_when_generation_update_fails(
        self, mock_ray, mock_supervisor
    ):
        sync = _checkpoint_sync(mock_ray, async_engine=True)
        mock_supervisor.side_effect = RuntimeError("Weight transfer failed")

        sync.init_communicator()
        with pytest.raises(RuntimeError, match="Weight transfer failed"):
            sync.sync_weights()

        assert sync.is_stale
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc_async",
            "update_weights_from_checkpoint_engine",
        )
        sync.shutdown()
        assert sync._generation.worker_group.calls[-1] == (
            "checkpoint_engine_rpc_async",
            "finalize_checkpoint_engine",
        )
        assert (
            sync._generation.worker_group.calls[0][0] == "checkpoint_engine_rpc_async"
        )
        assert sync._policy.worker_group.calls[-1] == (
            "checkpoint_engine_rpc",
            "finalize_checkpoint_engine",
        )


class TestCheckpointEngineFactory:
    def test_factory_forwards_refit_timeout(self):
        sync = create_weight_synchronizer(
            policy=_mock_policy(cfg={}),
            generation=_mock_generation(cfg=_nixl_refit_cfg()),
            generation_backend=VLLM_BACKEND,
            colocated=False,
            refit_timeout_s=19.5,
        )

        assert isinstance(sync, CheckpointEngineWeightSynchronizer)
        assert sync._refit_timeout_s == 19.5

    @pytest.mark.parametrize(
        ("backend", "colocated", "expected"),
        [
            (VLLM_BACKEND, False, CheckpointEngineWeightSynchronizer),
            (VLLM_BACKEND, True, ValueError),
            (SGLANG_BACKEND, False, NotImplementedError),
            (MEGATRON_BACKEND, False, NotImplementedError),
        ],
    )
    def test_checkpoint_engine_factory_routing(self, backend, colocated, expected):
        policy = _mock_policy(cfg={})
        gen = _mock_generation(cfg=_nixl_refit_cfg())
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                create_weight_synchronizer(
                    policy=policy,
                    generation=gen,
                    generation_backend=backend,
                    colocated=colocated,
                )
            return
        assert isinstance(
            create_weight_synchronizer(
                policy=policy,
                generation=gen,
                generation_backend=backend,
                colocated=colocated,
            ),
            expected,
        )

    @pytest.mark.parametrize("cfg", [{"megatron_cfg": {"enabled": False}}, {}])
    def test_checkpoint_engine_accepts_non_megatron_policy(self, cfg):
        gen = _mock_generation(cfg=_nixl_refit_cfg())
        assert isinstance(
            create_weight_synchronizer(
                policy=_mock_policy(cfg=cfg),
                generation=gen,
                generation_backend=VLLM_BACKEND,
                colocated=False,
            ),
            CheckpointEngineWeightSynchronizer,
        )
