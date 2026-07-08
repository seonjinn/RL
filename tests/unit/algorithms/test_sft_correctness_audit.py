import ast
import dataclasses
import inspect
import random
import textwrap
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_rl.algorithms.sft_correctness_audit import (
    CorrectnessAuditRecord,
    CorrectnessNextBatchRecord,
    CorrectnessSnapshot,
    SFTCorrectnessAuditor,
    capture_correctness_snapshot,
    compare_correctness_snapshots,
    evaluate_correctness_gate,
    snapshot_digest,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


def _worker_state(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "torch_cuda_rng": {"sha256": f"cuda-{rank}"},
        "mcore_cuda_rng": {"model-parallel-rng": {"sha256": f"mcore-{rank}"}},
        "model": {"parameters": {"weight": {"sum": float(rank)}}},
        "optimizer": {"step_counters": [rank]},
        "training_mode_flags": {"": True},
    }


def _snapshot_fixture(*, worker_order: tuple[int, ...] = (0, 1)) -> CorrectnessSnapshot:
    return CorrectnessSnapshot(
        python_rng_digest="python",
        numpy_rng_digest="numpy",
        torch_cpu_rng_digest="torch-cpu",
        torch_cuda_rng_digests=("torch-cuda-0", "torch-cuda-1"),
        explicit_generator_digest="generator",
        train_loader_digest="loader",
        next_train_batch_digest="next-batch",
        validation_payload_digest="payload",
        validation_sample_ids_digest="sample-ids",
        validation_token_counts_digest="token-counts",
        worker_states={rank: _worker_state(rank) for rank in worker_order},
    )


def test_correctness_snapshot_detects_worker_rng_change() -> None:
    before = _snapshot_fixture()
    after = dataclasses.replace(
        before,
        worker_states={
            0: {**before.worker_states[0], "torch_cuda_rng": "changed"},
            1: before.worker_states[1],
        },
    )

    assert compare_correctness_snapshots(before, after) == [
        "worker_states.0.torch_cuda_rng"
    ]


def test_correctness_snapshot_is_order_independent() -> None:
    assert snapshot_digest(_snapshot_fixture(worker_order=(0, 1))) == snapshot_digest(
        _snapshot_fixture(worker_order=(1, 0))
    )


@pytest.mark.parametrize(
    "state_family",
    [
        "torch_cuda_rng",
        "mcore_cuda_rng",
        "model",
        "optimizer",
        "training_mode_flags",
    ],
)
def test_correctness_gate_rejects_each_worker_state_family(
    state_family: str,
) -> None:
    before = _snapshot_fixture()
    changed_rank = dict(before.worker_states[0])
    changed_rank[state_family] = "changed"
    after = dataclasses.replace(
        before,
        worker_states={0: changed_rank, 1: before.worker_states[1]},
    )

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    assert any(
        difference.startswith(f"worker_states.0.{state_family}")
        for difference in result.differences
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        pytest.param("python_rng_digest", "changed", id="python-rng"),
        pytest.param("numpy_rng_digest", "changed", id="numpy-rng"),
        pytest.param("torch_cpu_rng_digest", "changed", id="torch-cpu-rng"),
        pytest.param(
            "torch_cuda_rng_digests", ("changed", "torch-cuda-1"), id="cuda-rng"
        ),
        pytest.param("explicit_generator_digest", "changed", id="explicit-generator"),
        pytest.param("train_loader_digest", "changed", id="train-loader"),
        pytest.param("next_train_batch_digest", "changed", id="next-train-batch"),
        pytest.param("validation_payload_digest", "changed", id="payload"),
        pytest.param(
            "validation_sample_ids_digest", "changed", id="validation-sample-ids"
        ),
        pytest.param(
            "validation_token_counts_digest", "changed", id="validation-tokens"
        ),
    ],
)
def test_correctness_gate_rejects_each_driver_state_family(
    field_name: str, changed_value: object
) -> None:
    before = _snapshot_fixture()
    after = dataclasses.replace(before, **{field_name: changed_value})

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    assert result.differences == (field_name,)


class _LoaderFixture:
    def __init__(self) -> None:
        self.state = {"position": 3, "sampler": torch.tensor([2, 0, 1])}
        self.iteration_count = 0

    def state_dict(self) -> dict[str, object]:
        return self.state

    def __iter__(self) -> "_LoaderFixture":
        self.iteration_count += 1
        return self

    def __next__(self) -> object:
        raise StopIteration


def _numpy_states_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_capture_correctness_snapshot_reads_without_advancing_driver_state() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    generator = torch.Generator().manual_seed(404)
    loader = _LoaderFixture()
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.return_value = [
        _worker_state(1),
        _worker_state(0),
    ]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator_state = generator.get_state().clone()
    loader_position = loader.state["position"]
    loader_sampler = loader.state["sampler"].clone()  # type: ignore[union-attr]

    with patch.object(torch.cuda, "is_initialized", return_value=False):
        snapshot = capture_correctness_snapshot(
            policy=policy,
            train_loader=loader,
            explicit_generator=generator,
            validation_payload={"input_ids": torch.tensor([[1, 2]])},
            validation_sample_ids=[17],
            validation_token_counts=(2,),
        )

    assert list(snapshot.worker_states) == [0, 1]
    assert random.getstate() == python_state
    assert _numpy_states_equal(
        cast(tuple[Any, ...], np.random.get_state()),
        cast(tuple[Any, ...], numpy_state),
    )
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(generator.get_state(), generator_state)
    assert loader.state["position"] == loader_position
    assert torch.equal(loader.state["sampler"], loader_sampler)  # type: ignore[arg-type]
    assert loader.iteration_count == 0


def test_capture_correctness_snapshot_detects_mutable_canonical_payload() -> None:
    payload = {"input_ids": torch.tensor([[1, 2]])}
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.return_value = [_worker_state(0)]
    kwargs: dict[str, Any] = {
        "policy": policy,
        "train_loader": _LoaderFixture(),
        "explicit_generator": None,
        "validation_payload": payload,
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
    }

    with patch.object(torch.cuda, "is_initialized", return_value=False):
        before = capture_correctness_snapshot(**kwargs)
        payload["input_ids"][0, 0] = -1
        after = capture_correctness_snapshot(**kwargs)

    assert compare_correctness_snapshots(before, after) == ["validation_payload_digest"]


def test_snapshot_matches_after_loader_and_generator_resume_or_restart() -> None:
    first_loader = _LoaderFixture()
    resumed_loader = _LoaderFixture()
    first_generator = torch.Generator().manual_seed(505)
    resumed_generator = torch.Generator()
    resumed_generator.set_state(first_generator.get_state())
    first_policy = MagicMock()
    resumed_policy = MagicMock()
    first_policy.get_correctness_state_fingerprint.return_value = [_worker_state(0)]
    resumed_policy.get_correctness_state_fingerprint.return_value = [_worker_state(0)]
    common: dict[str, Any] = {
        "validation_payload": {"input_ids": torch.tensor([[1, 2]])},
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
        "next_train_batch": {"input_ids": torch.tensor([[9, 10]])},
    }

    with patch.object(torch.cuda, "is_initialized", return_value=False):
        before_restart = capture_correctness_snapshot(
            policy=first_policy,
            train_loader=first_loader,
            explicit_generator=first_generator,
            **common,
        )
        after_restart = capture_correctness_snapshot(
            policy=resumed_policy,
            train_loader=resumed_loader,
            explicit_generator=resumed_generator,
            **common,
        )

    assert snapshot_digest(after_restart) == snapshot_digest(before_restart)


def test_auditor_handles_repeated_validation_boundaries() -> None:
    records: list[CorrectnessAuditRecord | CorrectnessNextBatchRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        validation_payload={"input_ids": torch.tensor([[1, 2]])},
        validation_sample_ids=[17],
        validation_token_counts=(2,),
        record_sink=records.append,
    )
    snapshot = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[snapshot, snapshot, snapshot, snapshot],
    ):
        auditor.audit_validation(step=20, validation=lambda: None)
        auditor.record_next_train_batch({"input_ids": torch.tensor([[3, 4]])})
        auditor.audit_validation(step=40, validation=lambda: None)
        auditor.record_next_train_batch({"input_ids": torch.tensor([[5, 6]])})

    boundary_records = [
        record for record in records if isinstance(record, CorrectnessAuditRecord)
    ]
    next_batch_records = [
        record for record in records if isinstance(record, CorrectnessNextBatchRecord)
    ]
    assert [record.validation_step for record in boundary_records] == [20, 40]
    assert [record.validation_step for record in next_batch_records] == [20, 40]
    assert all(record.gate.ready for record in boundary_records)


def test_auditor_records_next_batch_only_when_caller_supplies_natural_batch() -> None:
    records: list[CorrectnessAuditRecord | CorrectnessNextBatchRecord] = []
    loader = _LoaderFixture()
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=loader,
        explicit_generator=None,
        validation_payload={"input_ids": torch.tensor([[1, 2]])},
        validation_sample_ids=[17],
        validation_token_counts=(2,),
        record_sink=records.append,
    )
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[before, before],
    ):
        assert auditor.audit_validation(step=20, validation=lambda: "result") == (
            "result"
        )

    assert loader.iteration_count == 0
    assert len(records) == 1
    assert isinstance(records[0], CorrectnessAuditRecord)
    auditor.record_next_train_batch({"input_ids": torch.tensor([[1, 2]]), "idx": [17]})
    assert len(records) == 2
    assert isinstance(records[1], CorrectnessNextBatchRecord)
    assert records[1].validation_step == 20
    assert records[1].batch_digest


def test_auditor_gates_failed_validation_before_reraising() -> None:
    records: list[CorrectnessAuditRecord | CorrectnessNextBatchRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        validation_payload={"input_ids": torch.tensor([[1, 2]])},
        validation_sample_ids=[17],
        validation_token_counts=(2,),
        record_sink=records.append,
    )
    snapshot = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)

    def fail_validation() -> None:
        raise RuntimeError("submission failed")

    with (
        patch(
            "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
            side_effect=[snapshot, snapshot],
        ),
        pytest.raises(RuntimeError, match="submission failed"),
    ):
        auditor.audit_validation(step=40, validation=fail_validation)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, CorrectnessAuditRecord)
    assert record.validation_succeeded is False
    assert record.gate.ready is True


def test_policy_correctness_fingerprint_routes_to_every_worker_and_sorts_ranks() -> (
    None
):
    policy = object.__new__(Policy)
    policy.run_all_workers_single_data = MagicMock(  # type: ignore[method-assign]
        return_value=[_worker_state(3), _worker_state(1)]
    )

    result = Policy.get_correctness_state_fingerprint(
        policy,
        content_sample_count=5,
        reduction_chunk_numel=1024,
    )

    assert [record["rank"] for record in result] == [1, 3]
    policy.run_all_workers_single_data.assert_called_once_with(
        "get_correctness_state_fingerprint",
        content_sample_count=5,
        reduction_chunk_numel=1024,
    )


def _worker_fixture() -> MegatronPolicyWorkerImpl:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.rank = 7
    worker._local_coords = lambda: {  # type: ignore[method-assign]
        "pipeline_parallel": 1,
        "tensor_parallel": 2,
        "context_parallel": 0,
    }
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.5))
    model.register_buffer("audit_buffer", torch.tensor([4.0, 5.0]))
    model.train()
    worker.model = model
    parameter = next(model.parameters())
    worker.optimizer = SimpleNamespace(
        param_groups=[{"params": [parameter], "step": 11}],
        state={
            parameter: {
                "exp_avg": torch.arange(parameter.numel(), dtype=torch.float32).reshape(
                    parameter.shape
                ),
                "step": torch.tensor(9),
            }
        },
    )
    return worker


def test_megatron_worker_fingerprint_is_read_only_and_returns_python_records() -> None:
    worker = _worker_fixture()
    torch_rng_state = torch.get_rng_state().clone()
    model_state = {
        name: tensor.detach().clone()
        for name, tensor in worker.model.named_parameters()
    }
    optimizer_state = {
        key: value.detach().clone()
        for key, value in next(iter(worker.optimizer.state.values())).items()
        if torch.is_tensor(value)
    }

    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            torch.cuda,
            "get_rng_state",
            return_value=torch.tensor([1, 2, 3], dtype=torch.uint8),
        ),
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.get_all_rng_states",
            return_value={
                "model-parallel-rng": torch.tensor([4, 5, 6], dtype=torch.uint8)
            },
        ),
    ):
        fingerprint = worker.get_correctness_state_fingerprint(
            content_sample_count=3,
            reduction_chunk_numel=2,
        )

    assert fingerprint["rank"] == 7
    assert fingerprint["device"] == 0
    assert fingerprint["torch_cuda_rng"]["sha256"]
    assert fingerprint["mcore_cuda_rng"]["model-parallel-rng"]["sha256"]
    assert fingerprint["training_mode_flags"][""] is True
    assert fingerprint["model"]["parameters"]
    assert fingerprint["model"]["buffers"]
    assert fingerprint["optimizer"]["step_counters"] == [
        {"owner": [0, 0, 0], "state_key": "step", "value": 9},
        {"group": [0, 0], "state_key": "step", "value": 11},
    ]
    assert _contains_only_python_records(fingerprint)
    assert torch.equal(torch.get_rng_state(), torch_rng_state)
    assert worker.model.training is True
    for name, parameter in worker.model.named_parameters():
        assert torch.equal(parameter, model_state[name])
    for key, value in next(iter(worker.optimizer.state.values())).items():
        if torch.is_tensor(value):
            assert torch.equal(value, optimizer_state[key])


def _contains_only_python_records(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_contains_only_python_records(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _contains_only_python_records(item)
            for key, item in value.items()
        )
    return False


def test_megatron_worker_fingerprint_fails_if_mcore_tracker_is_uninitialized() -> None:
    worker = _worker_fixture()

    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            torch.cuda,
            "get_rng_state",
            return_value=torch.tensor([1], dtype=torch.uint8),
        ),
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.get_all_rng_states",
            side_effect=AssertionError("not initialized"),
        ),
        pytest.raises(RuntimeError, match="MCore CUDA RNG tracker is uninitialized"),
    ):
        worker.get_correctness_state_fingerprint()


def test_megatron_worker_fingerprint_avoids_forbidden_state_paths() -> None:
    source = textwrap.dedent(
        inspect.getsource(MegatronPolicyWorkerImpl.get_correctness_state_fingerprint)
    )
    method = ast.parse(source).body[0]
    assert isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
    call_names = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    direct_calls = {
        node.func.id
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "get_all_rng_states" in direct_calls
    assert call_names.isdisjoint(
        {
            "state_dict",
            "load_state_dict",
            "eval",
            "train",
            "set_rng_state",
            "manual_seed",
            "synchronize",
        }
    )
    assert "get_cuda_rng_tracker" not in source
