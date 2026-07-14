from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest
import torch


class _FakeUpstreamFullCudaGraphWrapper:
    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        use_single_mempool=False,
    ):
        self.forward_backward_func = forward_backward_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def __call__(self, *args, **kwargs):
        return self.forward_backward_func(*args, **kwargs)

    def reset_cuda_graph(self, stage=None):
        del stage


@dataclass
class _ProcessedMicrobatch:
    data_dict: dict
    input_ids: torch.Tensor
    input_ids_cp_sharded: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    packed_seq_params: object | None
    cu_seqlens_padded: torch.Tensor | None
    mtp_loss_mask: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    routed_experts_cp_sharded: torch.Tensor | None = None


def _full_cuda_graph_config() -> dict:
    return {
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "generation": {
            "colocated": {"enabled": False},
        },
        "megatron_cfg": {
            "cuda_graph_impl": "full_iteration",
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
        },
    }


def _processed_microbatch(seq_length: int = 4) -> _ProcessedMicrobatch:
    input_ids = torch.arange(seq_length).reshape(1, seq_length)
    return _ProcessedMicrobatch(
        data_dict={
            "input_ids": input_ids,
            "token_mask": torch.ones_like(input_ids),
            "sample_mask": torch.ones(1),
        },
        input_ids=input_ids,
        input_ids_cp_sharded=input_ids,
        attention_mask=torch.ones(1, 1, seq_length, seq_length),
        position_ids=torch.arange(seq_length).reshape(1, seq_length),
        packed_seq_params=None,
        cu_seqlens_padded=None,
    )


def test_aux_loss_scale_buffer_keeps_tensor_and_storage_stable_across_updates() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    buffer = FullCudaGraphAuxLossScaleBuffer()
    assert buffer._value is None

    first = buffer.update(torch.tensor(10))
    first_value = first.item()
    first_storage_pointer = first.untyped_storage().data_ptr()
    second = buffer.update(torch.tensor(5))

    assert first_value == pytest.approx(0.1)
    assert first.item() == pytest.approx(0.2)
    assert second is first
    assert second.untyped_storage().data_ptr() == first_storage_pointer
    assert second.dtype == torch.float32


@pytest.mark.parametrize("valid_token_count", [0, -4])
def test_aux_loss_scale_buffer_clamps_nonpositive_counts_without_reallocation(
    valid_token_count: int,
) -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    buffer = FullCudaGraphAuxLossScaleBuffer()
    first = buffer.update(torch.tensor(10))
    first_storage_pointer = first.untyped_storage().data_ptr()

    clamped = buffer.update(torch.tensor(valid_token_count))

    assert clamped is first
    assert clamped.untyped_storage().data_ptr() == first_storage_pointer
    assert clamped.item() == pytest.approx(1.0)


def test_aux_loss_scale_buffer_rejects_non_scalar_counts() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    with pytest.raises(ValueError, match="scalar global_valid_toks"):
        FullCudaGraphAuxLossScaleBuffer().update(torch.ones(2))


def test_aux_loss_scale_buffer_rejects_device_signature_drift() -> None:
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    buffer = FullCudaGraphAuxLossScaleBuffer()
    buffer.update(torch.tensor(10))

    with pytest.raises(ValueError, match="auxiliary loss scale signature changed"):
        buffer.update(torch.tensor(5, device="meta"))


def test_full_cuda_graph_policy_config_accepts_fixed_shape_noncolocated_training():
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    validate_full_cuda_graph_policy_config(
        _full_cuda_graph_config(), init_optimizer=True
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda cfg: cfg["dynamic_batching"].update(enabled=True), "dynamic batching"),
        (lambda cfg: cfg["sequence_packing"].update(enabled=True), "sequence packing"),
        (
            lambda cfg: cfg["generation"]["colocated"].update(enabled=True),
            "colocated generation/refit",
        ),
        (
            lambda cfg: cfg["generation"].update(backend="megatron"),
            "Megatron generation refit",
        ),
        (
            lambda cfg: cfg["megatron_cfg"].update(context_parallel_size=2),
            "context parallelism",
        ),
    ],
)
def test_full_cuda_graph_policy_config_rejects_unsupported_modes(mutation, match):
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    config = _full_cuda_graph_config()
    mutation(config)

    with pytest.raises(ValueError, match=match):
        validate_full_cuda_graph_policy_config(config, init_optimizer=True)


def test_full_cuda_graph_policy_config_rejects_non_training_worker():
    from nemo_rl.models.megatron.full_cuda_graph import (
        validate_full_cuda_graph_policy_config,
    )

    with pytest.raises(ValueError, match="optimizer-backed PolicyTraining"):
        validate_full_cuda_graph_policy_config(
            _full_cuda_graph_config(), init_optimizer=False
        )


def test_static_microbatch_signature_rejects_shape_drift():
    from nemo_rl.models.megatron.full_cuda_graph import StaticMicrobatchSignature

    expected = StaticMicrobatchSignature.from_microbatch(_processed_microbatch(4))
    actual = StaticMicrobatchSignature.from_microbatch(_processed_microbatch(8))

    with pytest.raises(ValueError, match="static input signature changed"):
        expected.require_match(actual, stage="training", microbatch=0)


def test_static_microbatch_signature_rejects_non_tensor_payload():
    from nemo_rl.models.megatron.full_cuda_graph import StaticMicrobatchSignature

    microbatch = _processed_microbatch()
    microbatch.data_dict["metadata"] = ["not", "graph", "safe"]

    with pytest.raises(TypeError, match="tensor-only"):
        StaticMicrobatchSignature.from_microbatch(microbatch)


def test_static_microbatch_signature_rejects_packed_sequence():
    from nemo_rl.models.megatron.full_cuda_graph import StaticMicrobatchSignature

    microbatch = _processed_microbatch()
    microbatch.packed_seq_params = MagicMock()

    with pytest.raises(ValueError, match="packed sequences"):
        StaticMicrobatchSignature.from_microbatch(microbatch)


def test_static_loader_clones_first_microbatch_into_stable_storage():
    from nemo_rl.models.megatron.full_cuda_graph import (
        ProcessedMicrobatchStaticBufferLoader,
    )

    source = _processed_microbatch()
    loader = ProcessedMicrobatchStaticBufferLoader()

    static = loader(source, "training", 0)

    assert torch.equal(static.input_ids, source.input_ids)
    assert static.input_ids.data_ptr() != source.input_ids.data_ptr()
    assert (
        static.data_dict["input_ids"].data_ptr()
        != source.data_dict["input_ids"].data_ptr()
    )


def test_static_loader_keeps_training_and_validation_storage_independent():
    from nemo_rl.models.megatron.full_cuda_graph import (
        ProcessedMicrobatchStaticBufferLoader,
    )

    loader = ProcessedMicrobatchStaticBufferLoader()

    training = loader(_processed_microbatch(4), "training", 0)
    validation = loader(_processed_microbatch(8), "validation", 0)

    assert training.input_ids.shape == (1, 4)
    assert validation.input_ids.shape == (1, 8)
    assert training.input_ids.data_ptr() != validation.input_ids.data_ptr()
    assert set(loader._signatures) == {("training", 0), ("validation", 0)}


def test_full_cuda_graph_wrapper_tracks_training_and_validation_independently():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphCallSignature,
        NemoRLFullCudaGraphWrapper,
    )

    raw_schedule = MagicMock()
    wrapper = NemoRLFullCudaGraphWrapper(
        raw_schedule,
        cuda_graph_warmup_steps=1,
        use_single_mempool=True,
        upstream_wrapper_cls=_FakeUpstreamFullCudaGraphWrapper,
    )
    training_signature = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        loss_signature="ClippedPGLossFn:v1",
    )
    validation_signature = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=8,
        micro_batch_size=1,
        loss_signature="LogprobsPostProcessor:v1",
    )

    for forward_only, signature, seq_length in (
        (False, training_signature, 4),
        (True, validation_signature, 8),
        (False, training_signature, 4),
        (True, validation_signature, 8),
    ):
        wrapper(
            model=MagicMock(),
            data_iterator=iter([_processed_microbatch(seq_length)]),
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=1,
            forward_only=forward_only,
            nemo_rl_signature=signature,
        )

    assert wrapper.execution_stats(stage="training").warmup_calls == 1
    assert wrapper.execution_stats(stage="training").capture_calls == 1
    assert wrapper.execution_stats(stage="training").replay_calls == 1
    assert wrapper.execution_stats(stage="validation").warmup_calls == 1
    assert wrapper.execution_stats(stage="validation").capture_calls == 1
    assert wrapper.execution_stats(stage="validation").replay_calls == 1

    changed = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=16,
        micro_batch_size=1,
        loss_signature="LogprobsPostProcessor:v1",
    )
    with pytest.raises(ValueError, match="call signature changed"):
        wrapper(
            model=MagicMock(),
            data_iterator=iter([_processed_microbatch(16)]),
            num_microbatches=1,
            seq_length=16,
            micro_batch_size=1,
            forward_only=True,
            nemo_rl_signature=changed,
        )


def test_full_cuda_graph_wrapper_forwards_supported_call_without_private_signature():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphCallSignature,
        NemoRLFullCudaGraphWrapper,
    )

    raw_schedule = MagicMock(return_value="ok")
    wrapper = NemoRLFullCudaGraphWrapper(
        raw_schedule,
        cuda_graph_warmup_steps=3,
        use_single_mempool=True,
        upstream_wrapper_cls=_FakeUpstreamFullCudaGraphWrapper,
    )
    result = wrapper(
        model=MagicMock(),
        data_iterator=iter([_processed_microbatch()]),
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        forward_only=False,
        nemo_rl_signature=FullCudaGraphCallSignature(
            num_microbatches=1,
            seq_length=4,
            micro_batch_size=1,
            loss_signature="ClippedPGLossFn:v1",
        ),
    )

    assert result == "ok"
    assert "nemo_rl_signature" not in raw_schedule.call_args.kwargs


def test_full_cuda_graph_wrapper_reset_clears_call_and_input_signatures(monkeypatch):
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphCallSignature,
        NemoRLFullCudaGraphWrapper,
    )

    wrapper = NemoRLFullCudaGraphWrapper(
        MagicMock(),
        cuda_graph_warmup_steps=3,
        use_single_mempool=True,
        upstream_wrapper_cls=_FakeUpstreamFullCudaGraphWrapper,
    )
    wrapper._expected_call_signatures["training"] = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        loss_signature="ClippedPGLossFn:v1",
    )
    wrapper._expected_call_signatures["validation"] = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=8,
        micro_batch_size=1,
        loss_signature="LogprobsPostProcessor:v1",
    )
    wrapper.static_loader._signatures[("training", 0)] = MagicMock()
    wrapper.static_loader._signatures[("validation", 0)] = MagicMock()
    reset = MagicMock()
    monkeypatch.setattr(_FakeUpstreamFullCudaGraphWrapper, "reset_cuda_graph", reset)

    wrapper.reset_cuda_graph(stage="validation")

    reset.assert_called_once_with(stage="validation")
    assert wrapper._expected_call_signatures["training"] is not None
    assert wrapper._expected_call_signatures["validation"] is None
    assert set(wrapper.static_loader._signatures) == {("training", 0)}


def test_build_full_cuda_graph_schedule_composes_paged_stash_outside_graph():
    from nemo_rl.models.megatron.full_cuda_graph import build_full_cuda_graph_schedule

    raw_schedule = MagicMock()
    model_config = SimpleNamespace(
        cuda_graph_warmup_steps=3,
        cuda_graph_use_single_mempool=True,
        moe_expert_rank_capacity_factor=1.2,
    )
    stash_manager = SimpleNamespace(enabled=True)

    def run_paged_stash(**kwargs):
        assert stash_manager.enabled is (not kwargs["forward_only"])
        return "ok"

    paged_stash_runner = MagicMock(side_effect=run_paged_stash)
    paged_stash_runner.stash_manager = stash_manager
    paged_stash_cls = MagicMock(return_value=paged_stash_runner)

    schedule, graph = build_full_cuda_graph_schedule(
        raw_schedule=raw_schedule,
        model_config=model_config,
        model=[MagicMock()],
        optimizer=MagicMock(),
        copy_main_params=True,
        paged_stash_cls=paged_stash_cls,
        upstream_wrapper_cls=_FakeUpstreamFullCudaGraphWrapper,
    )

    paged_stash_cls.assert_called_once_with(
        model_config,
        True,
        ANY,
        ANY,
        graph,
    )
    assert schedule(forward_only=True) == "ok"
    assert stash_manager.enabled is True
    paged_stash_runner.assert_called_once_with(forward_only=True)
    assert schedule(forward_only=False) == "ok"
    assert stash_manager.enabled is True

    paged_stash_runner.side_effect = RuntimeError("validation failed")
    with pytest.raises(RuntimeError, match="validation failed"):
        schedule(forward_only=True)
    assert stash_manager.enabled is True


def test_paged_stash_adapter_resets_untracked_graph_after_runner_failure():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphExecutionStats,
        _PagedStashValidationStateAdapter,
    )

    class TrackingGraph:
        def __init__(self) -> None:
            self.capture_calls = 0
            self.reset_calls = 0
            self.captured = False

        def execution_stats(self, *, stage: str) -> FullCudaGraphExecutionStats:
            assert stage == "training"
            return FullCudaGraphExecutionStats(
                warmup_calls=1,
                capture_calls=self.capture_calls,
                replay_calls=self.capture_calls,
                reset_calls=self.reset_calls,
            )

        def has_cuda_graph(self, *, stage: str) -> bool:
            assert stage == "training"
            return self.captured

        def reset_cuda_graph(self, *, stage: str) -> None:
            assert stage == "training"
            self.captured = False
            self.reset_calls += 1

    graph = TrackingGraph()
    stash_manager = SimpleNamespace(enabled=True)

    def fail_after_capture(**_kwargs):
        graph.capture_calls = 1
        graph.captured = True
        raise RuntimeError("outer paged-stash failure")

    runner = MagicMock(side_effect=fail_after_capture)
    runner.stash_manager = stash_manager
    runner.forward_backward_func = graph
    adapter = _PagedStashValidationStateAdapter(runner)

    with pytest.raises(RuntimeError, match="outer paged-stash failure"):
        adapter(forward_only=False)

    assert not graph.captured
    assert graph.reset_calls == 1
    assert adapter._training_storage_signature is None
    assert adapter._training_capture_calls == 1
    assert adapter._training_reset_calls == 1


def test_paged_stash_storage_signature_rejects_captured_buffer_reallocation():
    from nemo_rl.models.megatron.full_cuda_graph import _PagedStashStorageSignature

    buffer = SimpleNamespace(
        cuda_buffer=torch.zeros(4),
        host_buffer=None,
        free_list_head=torch.zeros(2, dtype=torch.int64),
    )
    stash_manager = SimpleNamespace(
        stash_buffers={torch.float32: {4: buffer}},
        overflow=torch.zeros(1),
        host_spill=torch.zeros(1),
    )
    signature = _PagedStashStorageSignature.capture(stash_manager)

    signature.require_match(stash_manager)
    buffer.cuda_buffer = torch.zeros(4)

    with pytest.raises(RuntimeError, match="PagedStash storage signature changed"):
        signature.require_match(stash_manager)


def test_full_cuda_graph_runtime_guards_fail_closed():
    from nemo_rl.models.megatron.full_cuda_graph import (
        require_supported_full_cuda_graph_operation,
    )

    for operation in ("logprob", "eval", "split_policy_training", "colocated_refit"):
        with pytest.raises(RuntimeError, match="full-iteration CUDA graph"):
            require_supported_full_cuda_graph_operation(
                enabled=True, operation=operation
            )

    require_supported_full_cuda_graph_operation(enabled=False, operation="logprob")
    require_supported_full_cuda_graph_operation(
        enabled=True, operation="policy_logprob"
    )


def test_full_cuda_graph_stage_evidence_reports_validation_counters():
    from nemo_rl.models.megatron.full_cuda_graph import (
        build_full_cuda_graph_stage_evidence_envelope_consensus,
    )

    envelopes = [
        (True, (0, (3, 1, 4, 0), "a" * 64), (0, (3, 1, 5, 0), "a" * 64)),
        (True, (1, (3, 1, 4, 0), "b" * 64), (1, (3, 1, 5, 0), "b" * 64)),
    ]

    evidence = build_full_cuda_graph_stage_evidence_envelope_consensus(
        envelopes, expected_world_size=2
    )

    assert evidence["full_cuda_graph_capture_calls"] == 1
    assert evidence["full_cuda_graph_replay_calls"] == 4
    assert evidence["full_cuda_graph_validation_capture_calls"] == 1
    assert evidence["full_cuda_graph_validation_replay_calls"] == 5


def test_full_cuda_graph_metric_context_keeps_scalar_tensors_on_device():
    from nemo_rl.algorithms.loss.interfaces import (
        full_cuda_graph_metrics,
        scalar_metric,
    )

    value = torch.tensor(3.0)
    assert scalar_metric(value) == 3.0
    with full_cuda_graph_metrics():
        assert scalar_metric(value) is value


def test_full_cuda_graph_metric_context_has_typed_iterator_return():
    from collections.abc import Iterator
    from typing import get_type_hints

    from nemo_rl.algorithms.loss.interfaces import full_cuda_graph_metrics

    assert get_type_hints(full_cuda_graph_metrics)["return"] == Iterator[None]


def test_clipped_pg_loss_emits_tensor_metrics_in_full_cuda_graph_context():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.loss.interfaces import full_cuda_graph_metrics

    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0))
    data = {
        "token_mask": torch.ones(1, 3),
        "sample_mask": torch.ones(1),
        "advantages": torch.ones(1, 3),
        "prev_logprobs": torch.zeros(1, 3),
        "generation_logprobs": torch.zeros(1, 3),
    }

    with full_cuda_graph_metrics():
        loss, metrics = loss_fn(
            next_token_logprobs=torch.zeros(1, 2, requires_grad=True),
            data=data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(2.0),
        )

    assert loss.requires_grad
    assert metrics
    assert all(isinstance(value, torch.Tensor) for value in metrics.values())


def test_nll_loss_emits_tensor_metrics_in_full_cuda_graph_context():
    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.algorithms.loss.interfaces import full_cuda_graph_metrics

    with full_cuda_graph_metrics():
        _, metrics = NLLLossFn()(
            next_token_logprobs=torch.zeros(1, 2, requires_grad=True),
            data={
                "token_mask": torch.ones(1, 3),
                "sample_mask": torch.ones(1),
            },
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(2.0),
        )

    assert all(isinstance(value, torch.Tensor) for value in metrics.values())


def test_materialize_full_cuda_graph_metrics_restores_python_scalars():
    from nemo_rl.models.megatron.full_cuda_graph import (
        materialize_full_cuda_graph_metrics,
    )

    materialized = materialize_full_cuda_graph_metrics(
        [
            {"loss": torch.tensor(1.25), "name": "first"},
            {"loss": torch.tensor(2.5), "count": torch.tensor(3)},
        ]
    )

    assert materialized == [
        {"loss": 1.25, "name": "first"},
        {"loss": 2.5, "count": 3},
    ]
