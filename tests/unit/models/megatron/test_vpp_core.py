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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.models.policy import MegatronConfig

pytestmark = pytest.mark.mcore


def _megatron_config(**overrides):
    config = {
        "enabled": True,
        "pipeline_model_parallel_size": 1,
        "overlap_moe_expert_parallel_comm": False,
    }
    config.update(overrides)
    return config


def _policy_config(**megatron_overrides):
    return {
        "megatron_cfg": _megatron_config(**megatron_overrides),
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": True},
        "draft": {"enabled": False},
        "router_replay": {"enabled": False},
        "generation": {"backend": "vllm", "colocated": {"enabled": False}},
        "quant_cfg": None,
    }


def test_virtual_pipeline_size_is_optional_public_megatron_config() -> None:
    assert "virtual_pipeline_model_parallel_size" in MegatronConfig.__optional_keys__
    assert "recompute_method" in MegatronConfig.__optional_keys__


@pytest.mark.parametrize("vpp_size", [None, 2])
def test_parallelism_config_maps_optional_virtual_pipeline_size(vpp_size) -> None:
    from nemo_rl.models.megatron.setup import _apply_parallelism_config

    model_cfg = SimpleNamespace()
    megatron_cfg = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "sequence_parallel": False,
        "context_parallel_size": 1,
    }
    if vpp_size is not None:
        megatron_cfg["virtual_pipeline_model_parallel_size"] = vpp_size

    _apply_parallelism_config(
        model_cfg,
        {
            "megatron_cfg": megatron_cfg,
            "sequence_packing": {"enabled": False},
        },
    )

    assert model_cfg.virtual_pipeline_model_parallel_size == vpp_size


@pytest.mark.parametrize("vpp_size", [None, 0, 1, -1, True, "2"])
def test_pp_a2a_overlap_rejects_missing_or_invalid_vpp(vpp_size) -> None:
    import nemo_rl.models.policy as policy_config_module

    validate = getattr(policy_config_module, "validate_virtual_pipeline_config", None)
    assert validate is not None
    config = _policy_config(
        pipeline_model_parallel_size=2,
        overlap_moe_expert_parallel_comm=True,
        virtual_pipeline_model_parallel_size=vpp_size,
    )

    with pytest.raises(ValueError, match="PP>1.*A2A.*VPP"):
        validate(config)


def test_pp_a2a_overlap_accepts_vpp_larger_than_one() -> None:
    import nemo_rl.models.policy as policy_config_module

    validate = getattr(policy_config_module, "validate_virtual_pipeline_config", None)
    assert validate is not None
    validate(
        _policy_config(
            pipeline_model_parallel_size=2,
            overlap_moe_expert_parallel_comm=True,
            virtual_pipeline_model_parallel_size=2,
        )
    )


@pytest.mark.parametrize("vpp_size", [0, 1, -1, True, "2"])
def test_explicit_invalid_vpp_size_fails_without_a2a(vpp_size) -> None:
    from nemo_rl.models.policy import validate_virtual_pipeline_config

    with pytest.raises(ValueError, match="VPP.*integer greater than 1"):
        validate_virtual_pipeline_config(
            _policy_config(
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=vpp_size,
            )
        )


def test_vpp_requires_physical_pipeline_parallelism() -> None:
    from nemo_rl.models.policy import validate_virtual_pipeline_config

    with pytest.raises(ValueError, match="VPP.*pipeline_model_parallel_size.*greater"):
        validate_virtual_pipeline_config(
            _policy_config(
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=2,
            )
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cuda_graph_impl": "full_iteration"}, "full-iteration CUDA Graph"),
        ({"draft": {"enabled": True}}, "draft training"),
        ({"router_replay": {"enabled": True}}, "router replay"),
        ({"generation": {"backend": "megatron"}}, "native Megatron generation"),
        (
            {
                "generation": {
                    "backend": "vllm",
                    "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"},
                }
            },
            "FP8 KV-cache calibration",
        ),
        ({"quant_cfg": "NVFP4_DEFAULT_CFG"}, "ModelOpt"),
    ],
)
def test_unsupported_vpp_policy_combinations_fail_early(overrides, message) -> None:
    import nemo_rl.models.policy as policy_config_module

    validate = getattr(policy_config_module, "validate_virtual_pipeline_config", None)
    assert validate is not None
    config = _policy_config(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    megatron_overrides = {
        key: value for key, value in overrides.items() if key == "cuda_graph_impl"
    }
    config["megatron_cfg"].update(megatron_overrides)
    config.update(
        {key: value for key, value in overrides.items() if key != "cuda_graph_impl"}
    )

    with pytest.raises(ValueError, match=message):
        validate(config)


def test_value_worker_vpp_combination_fails_early() -> None:
    import nemo_rl.models.policy as policy_config_module

    validate = getattr(policy_config_module, "validate_virtual_pipeline_config", None)
    assert validate is not None

    with pytest.raises(ValueError, match="value worker"):
        validate(
            _policy_config(
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
            ),
            component="value worker",
        )


def test_non_vllm_backend_ignores_stale_fp8_vllm_config() -> None:
    from nemo_rl.models.policy import validate_virtual_pipeline_config

    config = _policy_config(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        generation={
            "backend": "sglang",
            "vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"},
        },
    )

    validate_virtual_pipeline_config(config)


def test_vpp_sequence_packing_constraints_use_policy_dp_and_pp() -> None:
    from nemo_rl.models.policy.lm_policy import Policy

    policy = object.__new__(Policy)
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 3
    policy.sequence_packing_args = {}

    configure = getattr(policy, "_configure_vpp_sequence_packing_constraints", None)
    assert configure is not None
    configure(
        _policy_config(
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )
    )

    assert policy.sequence_packing_args["min_bin_count"] == 12
    assert policy.sequence_packing_args["bin_count_multiple"] == 12
    policy.sharding_annotations.get_axis_size.assert_called_once_with("data_parallel")


def test_non_vpp_sequence_packing_does_not_add_bin_constraints() -> None:
    from nemo_rl.models.policy.lm_policy import Policy

    policy = object.__new__(Policy)
    policy.sharding_annotations = MagicMock()
    policy.sequence_packing_args = {}

    configure = getattr(policy, "_configure_vpp_sequence_packing_constraints", None)
    assert configure is not None
    configure(_policy_config())

    assert "min_bin_count" not in policy.sequence_packing_args
    assert "bin_count_multiple" not in policy.sequence_packing_args
    policy.sharding_annotations.get_axis_size.assert_not_called()


def test_interleaved_schedule_receives_independent_iterator_per_chunk() -> None:
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        megatron_forward_backward,
    )

    first = object()
    second = object()
    source_iterator = iter([first, second])
    schedule = MagicMock(return_value=[])

    with patch(
        "nemo_rl.models.megatron.train.get_forward_backward_func",
        return_value=schedule,
    ):
        megatron_forward_backward(
            model=[MagicMock(), MagicMock()],
            data_iterator=source_iterator,
            num_microbatches=2,
            seq_length=8,
            mbs=1,
            post_processing_fn=LossPostProcessor(
                loss_fn=MagicMock(),
                cfg={"sequence_packing": {"enabled": False}},
            ),
        )

    iterators = schedule.call_args.kwargs["data_iterator"]
    assert isinstance(iterators, list)
    assert len(iterators) == 2
    assert iterators[0] is not iterators[1]
    assert list(iterators[0]) == [first, second]
    assert list(iterators[1]) == [first, second]


@pytest.mark.parametrize("model", [MagicMock(), [MagicMock()]])
def test_scalar_and_single_chunk_schedule_keep_scalar_iterator(model) -> None:
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        megatron_forward_backward,
    )

    source_iterator = iter([])
    schedule = MagicMock(return_value=[])

    with patch(
        "nemo_rl.models.megatron.train.get_forward_backward_func",
        return_value=schedule,
    ):
        megatron_forward_backward(
            model=model,
            data_iterator=source_iterator,
            num_microbatches=1,
            seq_length=8,
            mbs=1,
            post_processing_fn=LossPostProcessor(
                loss_fn=MagicMock(),
                cfg={"sequence_packing": {"enabled": False}},
            ),
        )

    assert schedule.call_args.kwargs["data_iterator"] is source_iterator


def test_chunk_prefixed_state_restores_colliding_local_names() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = [
        torch.nn.Linear(2, 1, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    ]
    source_state = {
        "model_chunk_0/weight": torch.tensor([[1.0, 2.0]]),
        "model_chunk_1/weight": torch.tensor([[3.0, 4.0]]),
    }

    with torch.no_grad():
        worker._apply_state_dict_to_model(source_state, raise_if_key_missing=True)

    assert torch.equal(worker.model[0].weight, source_state["model_chunk_0/weight"])
    assert torch.equal(worker.model[1].weight, source_state["model_chunk_1/weight"])


def test_single_chunk_state_restore_accepts_legacy_unprefixed_keys() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = torch.nn.Linear(2, 1, bias=False)
    source_state = {"weight": torch.tensor([[5.0, 6.0]])}

    with torch.no_grad():
        worker._apply_state_dict_to_model(source_state, raise_if_key_missing=True)

    assert torch.equal(worker.model.weight, source_state["weight"])


def test_gpu_info_uses_primary_model_chunk(monkeypatch) -> None:
    from nemo_rl.models.policy import utils as policy_utils
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    chunks = [MagicMock(), MagicMock()]
    get_gpu_info = MagicMock(return_value={"device": "test"})
    monkeypatch.setattr(policy_utils, "get_gpu_info", get_gpu_info)
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = chunks

    assert worker.get_gpu_info() == {"device": "test"}
    get_gpu_info.assert_called_once_with(chunks[0])


def test_qkv_fp8_calibration_rejects_vpp_before_hooks() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = [MagicMock(), MagicMock()]

    with pytest.raises(RuntimeError, match="FP8 KV-cache calibration.*VPP"):
        worker.calibrate_qkv_fp8_scales(data=MagicMock())


def test_forward_pre_hooks_cover_every_model_chunk(monkeypatch) -> None:
    from nemo_rl.models.policy.workers import megatron_policy_worker

    class FakeDDP:
        def __init__(self) -> None:
            self.remove_forward_pre_hook_handles = {}
            self.enable_calls = 0
            self.disable_calls = 0

        def enable_forward_pre_hook(self) -> None:
            self.enable_calls += 1
            self.remove_forward_pre_hook_handles = {0: object()}

        def disable_forward_pre_hook(self, *, param_sync: bool) -> None:
            assert param_sync is False
            self.disable_calls += 1
            self.remove_forward_pre_hook_handles = {}

    monkeypatch.setattr(megatron_policy_worker, "DistributedDataParallel", FakeDDP)
    worker = object.__new__(megatron_policy_worker.MegatronPolicyWorkerImpl)
    worker.model = [FakeDDP(), FakeDDP()]

    worker.enable_forward_pre_hook()
    worker.disable_forward_pre_hook(param_sync=False)

    assert [chunk.enable_calls for chunk in worker.model] == [1, 1]
    assert [chunk.disable_calls for chunk in worker.model] == [1, 1]


def test_move_model_covers_every_model_chunk() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    chunks = [torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]

    moved = worker.move_model(chunks, "cpu")

    assert moved == chunks
    assert all(
        parameter.device.type == "cpu"
        for chunk in moved
        for parameter in chunk.parameters()
    )


def test_refit_export_receives_every_model_chunk() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = [MagicMock(), MagicMock()]
    worker.megatron_bridge = MagicMock()
    worker.megatron_bridge.export_hf_weights.return_value = iter(
        [("weight", torch.ones(1))]
    )
    worker.refit_conversion_tasks = []
    worker.draft_model = None
    worker.cfg = {}

    assert [name for name, _ in worker._iter_params_with_optional_kv_scales()] == [
        "weight"
    ]
    assert worker.megatron_bridge.export_hf_weights.call_args.args[0] is worker.model


def test_checkpoint_receives_every_model_chunk(monkeypatch) -> None:
    from nemo_rl.models.policy.workers import megatron_policy_worker

    class Chunk:
        def __init__(self) -> None:
            self.training = False
            self.eval_calls = 0
            self.train_calls = 0

        def eval(self) -> None:
            self.eval_calls += 1

        def train(self) -> None:
            self.training = True
            self.train_calls += 1

    chunks = [Chunk(), Chunk()]
    checkpoint_cfg = SimpleNamespace(save="old", async_save=False)
    worker = object.__new__(megatron_policy_worker.MegatronPolicyWorkerImpl)
    worker.model = chunks
    worker.mcore_state = SimpleNamespace(
        cfg=SimpleNamespace(checkpoint=checkpoint_cfg),
        train_state=SimpleNamespace(floating_point_operations_so_far=0),
    )
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.checkpointing_context = {}
    worker.should_disable_forward_pre_hook = False
    save = MagicMock()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_policy_worker, "save_checkpoint", save)
    monkeypatch.setattr(
        megatron_policy_worker,
        "maybe_finalize_async_save",
        lambda *args, **kwargs: None,
    )

    worker.save_checkpoint("new")

    assert save.call_args.kwargs["model"] is chunks
    assert [chunk.eval_calls for chunk in chunks] == [1, 1]
    assert [chunk.train_calls for chunk in chunks] == [1, 1]


def test_split_step_training_rejects_vpp_before_cuda_work() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = _policy_config(virtual_pipeline_model_parallel_size=2)
    worker._train_step_state = None

    with pytest.raises(RuntimeError, match="split-step training.*VPP"):
        worker.begin_train_step(MagicMock())
