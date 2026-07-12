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


def test_aux_loss_scale_buffer_keeps_graph_visible_storage_stable():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    buffer = FullCudaGraphAuxLossScaleBuffer()

    first = buffer.update(torch.tensor(10))
    second = buffer.update(torch.tensor(5))

    assert second is first
    assert second.item() == pytest.approx(0.2)


def test_aux_loss_scale_buffer_rejects_non_scalar_counts():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphAuxLossScaleBuffer,
    )

    with pytest.raises(ValueError, match="scalar global_valid_toks"):
        FullCudaGraphAuxLossScaleBuffer().update(torch.ones(2))


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


def test_full_cuda_graph_wrapper_rejects_forward_only_and_call_signature_drift():
    from nemo_rl.models.megatron.full_cuda_graph import (
        FullCudaGraphCallSignature,
        NemoRLFullCudaGraphWrapper,
    )

    raw_schedule = MagicMock()
    wrapper = NemoRLFullCudaGraphWrapper(
        raw_schedule,
        cuda_graph_warmup_steps=3,
        use_single_mempool=True,
        upstream_wrapper_cls=_FakeUpstreamFullCudaGraphWrapper,
    )
    signature = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        loss_signature="ClippedPGLossFn:v1",
    )

    with pytest.raises(RuntimeError, match="PolicyTraining only"):
        wrapper(
            model=MagicMock(),
            data_iterator=iter([_processed_microbatch()]),
            num_microbatches=1,
            seq_length=4,
            micro_batch_size=1,
            forward_only=True,
            nemo_rl_signature=signature,
        )

    wrapper._expected_call_signature = signature
    changed = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=8,
        micro_batch_size=1,
        loss_signature="ClippedPGLossFn:v1",
    )
    with pytest.raises(ValueError, match="call signature changed"):
        wrapper(
            model=MagicMock(),
            data_iterator=iter([_processed_microbatch(8)]),
            num_microbatches=1,
            seq_length=8,
            micro_batch_size=1,
            forward_only=False,
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
    wrapper._expected_call_signature = FullCudaGraphCallSignature(
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
        loss_signature="ClippedPGLossFn:v1",
    )
    wrapper.static_loader._signatures[("training", 0)] = MagicMock()
    reset = MagicMock()
    monkeypatch.setattr(_FakeUpstreamFullCudaGraphWrapper, "reset_cuda_graph", reset)

    wrapper.reset_cuda_graph(stage="training")

    reset.assert_called_once_with(stage="training")
    assert wrapper._expected_call_signature is None
    assert wrapper.static_loader._signatures == {}


def test_build_full_cuda_graph_schedule_composes_paged_stash_outside_graph():
    from nemo_rl.models.megatron.full_cuda_graph import build_full_cuda_graph_schedule

    raw_schedule = MagicMock()
    model_config = SimpleNamespace(
        cuda_graph_warmup_steps=3,
        cuda_graph_use_single_mempool=True,
        moe_expert_rank_capacity_factor=1.2,
    )
    paged_stash_cls = MagicMock()

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
    assert schedule is paged_stash_cls.return_value


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


def test_full_cuda_graph_metric_context_keeps_scalar_tensors_on_device():
    from nemo_rl.algorithms.loss.interfaces import (
        full_cuda_graph_metrics,
        scalar_metric,
    )

    value = torch.tensor(3.0)
    assert scalar_metric(value) == 3.0
    with full_cuda_graph_metrics():
        assert scalar_metric(value) is value


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
