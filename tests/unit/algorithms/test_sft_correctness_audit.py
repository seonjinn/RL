import ast
import dataclasses
import inspect
import json
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
    CorrectnessAuditError,
    CorrectnessNextTrainBatchEvidence,
    CorrectnessSnapshot,
    CorrectnessValidationEvidencePair,
    SFTCorrectnessAuditor,
    capture_next_train_batch_evidence,
    capture_correctness_snapshot,
    capture_validation_evidence,
    compare_correctness_snapshots,
    compare_next_train_batch_to_control,
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
    expected_difference = (
        "torch_cuda_rng_digests.0"
        if field_name == "torch_cuda_rng_digests"
        else field_name
    )
    assert result.differences == (expected_difference,)


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
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        record_sink=records.append,
    )
    snapshot = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[snapshot, snapshot, snapshot, snapshot],
    ):
        auditor.audit_validation(
            step=20,
            validation=lambda: None,
            validation_evidence=_validation_evidence_pair,
        )
        auditor.record_next_train_batch({"input_ids": torch.tensor([[3, 4]])})
        auditor.audit_validation(
            step=40,
            validation=lambda: None,
            validation_evidence=_validation_evidence_pair,
        )
        auditor.record_next_train_batch({"input_ids": torch.tensor([[5, 6]])})

    assert [record.validation_step for record in records] == [20, 40]
    assert all(record.gate.ready for record in records)
    assert all(record.next_train_batch is not None for record in records)
    assert all(record.status == "finalized" for record in records)


def test_auditor_records_next_batch_only_when_caller_supplies_natural_batch() -> None:
    records: list[CorrectnessAuditRecord] = []
    loader = _LoaderFixture()
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=loader,
        explicit_generator=None,
        record_sink=records.append,
    )
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[before, before],
    ):
        assert (
            auditor.audit_validation(
                step=20,
                validation=lambda: "result",
                validation_evidence=_validation_evidence_pair,
            )
            == "result"
        )

    assert loader.iteration_count == 0
    assert records == []
    auditor.record_next_train_batch({"input_ids": torch.tensor([[1, 2]]), "idx": [17]})
    assert len(records) == 1
    assert records[0].validation_step == 20
    assert records[0].next_train_batch is not None
    assert records[0].next_train_batch.batch_digest


def test_auditor_record_only_mode_retains_changed_transition() -> None:
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        enforce_unchanged=False,
        record_sink=records.append,
    )
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)
    after = dataclasses.replace(before, torch_cpu_rng_digest="changed")

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[before, after],
    ):
        assert (
            auditor.audit_validation(
                step=20,
                validation=lambda: "result",
                validation_evidence=_validation_evidence_pair,
            )
            == "result"
        )

    assert records == []
    auditor.record_next_train_batch({"input_ids": torch.tensor([[1, 2]])})
    assert len(records) == 1
    assert records[0].gate.ready is False
    assert records[0].gate.differences == ("torch_cpu_rng_digest",)
    assert records[0].status == "finalized_with_state_changes"


def test_auditor_strict_default_rejects_changed_transition() -> None:
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        record_sink=records.append,
    )
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)
    after = dataclasses.replace(before, torch_cpu_rng_digest="changed")

    with (
        patch(
            "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
            side_effect=[before, after],
        ),
        pytest.raises(CorrectnessAuditError, match="torch_cpu_rng_digest"),
    ):
        auditor.audit_validation(
            step=20,
            validation=lambda: None,
            validation_evidence=_validation_evidence_pair,
        )

    assert len(records) == 1
    assert records[0].status == "rejected"


def test_auditor_record_only_mode_preserves_validation_exception() -> None:
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        enforce_unchanged=False,
        record_sink=records.append,
    )
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)
    after = dataclasses.replace(before, torch_cpu_rng_digest="changed")

    def fail_validation() -> None:
        raise RuntimeError("validation submission failed")

    with (
        patch(
            "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
            side_effect=[before, after],
        ),
        pytest.raises(RuntimeError, match="validation submission failed"),
    ):
        auditor.audit_validation(
            step=20,
            validation=fail_validation,
            validation_evidence=_validation_evidence_pair,
        )

    assert len(records) == 1
    assert records[0].status == "validation_failed"


def test_default_record_sink_emits_compact_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        enforce_unchanged=False,
    )
    marker = "must-not-be-serialized-" + "x" * 100_000
    before = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)
    after = dataclasses.replace(
        before,
        worker_states={
            0: {
                **before.worker_states[0],
                "model": {
                    **before.worker_states[0]["model"],
                    "expanded_fingerprint": marker,
                },
            },
            1: before.worker_states[1],
        },
    )

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[before, after],
    ):
        auditor.audit_validation(
            step=20,
            validation=lambda: None,
            validation_evidence=_validation_evidence_pair,
        )
    auditor.record_next_train_batch({"input_ids": torch.tensor([[1, 2]])})

    output = capsys.readouterr().out
    assert output.startswith("SFT_CORRECTNESS_AUDIT_SUMMARY ")
    assert marker not in output
    assert len(output) < 20_000
    payload = json.loads(output.removeprefix("SFT_CORRECTNESS_AUDIT_SUMMARY "))
    assert payload["gate"]["ready"] is False
    assert payload["gate"]["difference_count"] == 1
    assert payload["gate"]["differences_sha256"]
    assert payload["before"]["worker_states"]["0"]["state_digest"]
    assert payload["after"]["worker_states"]["0"]["state_digest"]
    before_categories = payload["before"]["worker_states"]["0"]["category_digests"]
    after_categories = payload["after"]["worker_states"]["0"]["category_digests"]
    assert before_categories["model"] != after_categories["model"]
    assert before_categories["optimizer"] == after_categories["optimizer"]


def test_auditor_gates_failed_validation_before_reraising() -> None:
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
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
        auditor.audit_validation(
            step=40,
            validation=fail_validation,
            validation_evidence=_validation_evidence_pair,
        )

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
    assert fingerprint["optimizer"]["parameters"][0]["owner"] == (0, 0, 0)
    assert fingerprint["optimizer"]["step_counters"] == [
        {"owner": (0, 0, 0), "state_key": "step", "value": 9},
        {"group": (0, 0), "state_key": "step", "value": 11},
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
    if isinstance(value, (list, tuple)):
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


@pytest.mark.parametrize("tracker_states", [{}, [], None, "invalid"])
def test_megatron_worker_fingerprint_rejects_empty_or_nonmapping_tracker_states(
    tracker_states: object,
) -> None:
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
            return_value=tracker_states,
        ),
        pytest.raises(RuntimeError, match="non-empty mapping"),
    ):
        worker.get_correctness_state_fingerprint()


def _capture_worker_fingerprint(
    worker: MegatronPolicyWorkerImpl,
    *,
    torch_cuda_rng: torch.Tensor | None = None,
    mcore_cuda_rng: torch.Tensor | None = None,
) -> dict[str, Any]:
    torch_cuda_rng = (
        torch.tensor([1, 2, 3], dtype=torch.uint8)
        if torch_cuda_rng is None
        else torch_cuda_rng
    )
    mcore_cuda_rng = (
        torch.tensor([4, 5, 6], dtype=torch.uint8)
        if mcore_cuda_rng is None
        else mcore_cuda_rng
    )
    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            torch.cuda,
            "get_rng_state",
            return_value=torch_cuda_rng,
        ),
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.get_all_rng_states",
            return_value={"model-parallel-rng": mcore_cuda_rng},
        ),
    ):
        return worker.get_correctness_state_fingerprint(
            content_sample_count=3,
            reduction_chunk_numel=2,
        )


def test_megatron_worker_fingerprints_optimizer_main_shard_parameter() -> None:
    worker = _worker_fixture()
    model_parameter = next(worker.model.parameters())
    main_shard = torch.nn.Parameter(
        torch.arange(model_parameter.numel(), dtype=torch.float32).reshape(
            model_parameter.shape
        )
        + 100
    )
    worker.optimizer.param_groups[0]["params"] = [main_shard]
    worker.optimizer.state = {main_shard: {"step": torch.tensor(9)}}

    before = _capture_worker_fingerprint(worker)
    with torch.no_grad():
        main_shard.add_(1)
    after = _capture_worker_fingerprint(worker)

    assert before["optimizer"]["parameters"][0]["owner"] == (0, 0, 0)
    assert before["optimizer"]["parameters"][0]["tensor"]["shape"] == list(
        main_shard.shape
    )
    assert before["optimizer"]["parameters"] != after["optimizer"]["parameters"]
    assert before["model"] == after["model"]


@pytest.mark.parametrize(
    "state_family",
    ["python", "numpy", "torch", "generator", "loader"],
)
def test_production_capture_gate_detects_real_driver_state_mutations(
    state_family: str,
) -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    generator = torch.Generator().manual_seed(404)
    loader = _LoaderFixture()
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.return_value = [_worker_state(0)]
    kwargs: dict[str, Any] = {
        "policy": policy,
        "train_loader": loader,
        "explicit_generator": generator,
        "validation_payload": {"input_ids": torch.tensor([[1, 2]])},
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
    }

    with patch.object(torch.cuda, "is_initialized", return_value=False):
        before = capture_correctness_snapshot(**kwargs)
        if state_family == "python":
            random.random()
        elif state_family == "numpy":
            np.random.random()
        elif state_family == "torch":
            torch.rand(1)
        elif state_family == "generator":
            torch.rand(1, generator=generator)
        else:
            loader.state["position"] = 4
        after = capture_correctness_snapshot(**kwargs)

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    expected_field = {
        "python": "python_rng_digest",
        "numpy": "numpy_rng_digest",
        "torch": "torch_cpu_rng_digest",
        "generator": "explicit_generator_digest",
        "loader": "train_loader_digest",
    }[state_family]
    assert expected_field in result.differences


def test_production_capture_gate_detects_driver_torch_cuda_rng_mutation() -> None:
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.return_value = [_worker_state(0)]
    kwargs: dict[str, Any] = {
        "policy": policy,
        "train_loader": _LoaderFixture(),
        "explicit_generator": None,
        "validation_payload": {"input_ids": torch.tensor([[1, 2]])},
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
    }

    with (
        patch.object(torch.cuda, "is_initialized", return_value=True),
        patch.object(
            torch.cuda,
            "get_rng_state_all",
            side_effect=[
                [torch.tensor([1, 2], dtype=torch.uint8)],
                [torch.tensor([1, 3], dtype=torch.uint8)],
            ],
        ),
    ):
        before = capture_correctness_snapshot(**kwargs)
        after = capture_correctness_snapshot(**kwargs)

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    assert result.differences == ("torch_cuda_rng_digests.0",)


@pytest.mark.parametrize(
    ("state_family", "expected_path"),
    [
        pytest.param(
            "torch_cuda_rng",
            "worker_states.7.torch_cuda_rng",
            id="worker-torch-cuda-rng",
        ),
        pytest.param(
            "mcore_cuda_rng",
            "worker_states.7.mcore_cuda_rng",
            id="worker-mcore-cuda-rng",
        ),
    ],
)
def test_production_capture_gate_detects_worker_rng_api_mutations(
    state_family: str,
    expected_path: str,
) -> None:
    worker = _worker_fixture()
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.side_effect = lambda: [
        worker.get_correctness_state_fingerprint(
            content_sample_count=3,
            reduction_chunk_numel=2,
        )
    ]
    torch_states = [
        torch.tensor([1, 2, 3], dtype=torch.uint8),
        torch.tensor([1, 2, 4], dtype=torch.uint8),
    ]
    mcore_states = [
        {"model-parallel-rng": torch.tensor([4, 5, 6], dtype=torch.uint8)},
        {"model-parallel-rng": torch.tensor([4, 5, 7], dtype=torch.uint8)},
    ]
    if state_family == "torch_cuda_rng":
        tracker_side_effect = [mcore_states[0], mcore_states[0]]
    else:
        torch_states[1] = torch_states[0]
        tracker_side_effect = mcore_states
    kwargs: dict[str, Any] = {
        "policy": policy,
        "train_loader": _LoaderFixture(),
        "explicit_generator": None,
        "validation_payload": {"input_ids": torch.tensor([[1, 2]])},
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
    }

    with (
        patch.object(torch.cuda, "is_initialized", return_value=False),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_rng_state", side_effect=torch_states),
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.get_all_rng_states",
            side_effect=tracker_side_effect,
        ),
    ):
        before = capture_correctness_snapshot(**kwargs)
        after = capture_correctness_snapshot(**kwargs)

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    assert any(path.startswith(expected_path) for path in result.differences)


@pytest.mark.parametrize(
    "state_family",
    [
        "model",
        "optimizer_main_shard",
        "optimizer_state_tensor",
        "optimizer_step",
        "training_mode",
    ],
)
def test_production_capture_gate_detects_real_worker_state_mutations(
    state_family: str,
) -> None:
    worker = _worker_fixture()
    model_parameter = next(worker.model.parameters())
    main_shard = torch.nn.Parameter(model_parameter.detach().clone() + 50)
    worker.optimizer.param_groups[0]["params"] = [main_shard]
    worker.optimizer.state = {
        main_shard: {
            "exp_avg": torch.arange(main_shard.numel(), dtype=torch.float32).reshape(
                main_shard.shape
            ),
            "step": torch.tensor(9),
        }
    }
    policy = MagicMock()
    policy.get_correctness_state_fingerprint.side_effect = lambda: [
        _capture_worker_fingerprint(worker)
    ]
    kwargs: dict[str, Any] = {
        "policy": policy,
        "train_loader": _LoaderFixture(),
        "explicit_generator": None,
        "validation_payload": {"input_ids": torch.tensor([[1, 2]])},
        "validation_sample_ids": [17],
        "validation_token_counts": (2,),
    }

    with patch.object(torch.cuda, "is_initialized", return_value=False):
        before = capture_correctness_snapshot(**kwargs)
        with torch.no_grad():
            if state_family == "model":
                model_parameter.add_(1)
            elif state_family == "optimizer_main_shard":
                main_shard.add_(1)
            elif state_family == "optimizer_state_tensor":
                worker.optimizer.state[main_shard]["exp_avg"].add_(1)
            elif state_family == "optimizer_step":
                worker.optimizer.state[main_shard]["step"].add_(1)
            else:
                worker.model.train(False)
        after = capture_correctness_snapshot(**kwargs)

    result = evaluate_correctness_gate(before, after)

    assert result.ready is False
    expected_path = {
        "model": "worker_states.7.model",
        "optimizer_main_shard": "worker_states.7.optimizer.parameters",
        "optimizer_state_tensor": "worker_states.7.optimizer.state_tensors",
        "optimizer_step": "worker_states.7.optimizer.step_counters",
        "training_mode": "worker_states.7.training_mode_flags",
    }[state_family]
    assert any(path.startswith(expected_path) for path in result.differences)


def _validation_evidence_pair() -> CorrectnessValidationEvidencePair:
    evidence = capture_validation_evidence(
        validation_payload={"input_ids": torch.tensor([[11, 12]])},
        validation_sample_ids=torch.tensor([[11, 12]]),
        validation_token_counts=(2,),
    )
    return CorrectnessValidationEvidencePair(before=evidence, after=evidence)


def test_auditor_finalizes_next_batch_and_compares_to_no_validation_control() -> None:
    records: list[CorrectnessAuditRecord] = []
    auditor = SFTCorrectnessAuditor(
        policy=MagicMock(),
        train_loader=_LoaderFixture(),
        explicit_generator=None,
        record_sink=records.append,
    )
    snapshot = dataclasses.replace(_snapshot_fixture(), next_train_batch_digest=None)
    control_batch = {"input_ids": torch.tensor([[3, 4]]), "idx": [19]}

    with patch(
        "nemo_rl.algorithms.sft_correctness_audit.capture_correctness_snapshot",
        side_effect=[snapshot, snapshot],
    ):
        auditor.audit_validation(
            step=20,
            validation=lambda: _validation_evidence_pair(),
            validation_evidence=_validation_evidence_pair,
        )

    assert records == []
    auditor.record_next_train_batch(control_batch)

    assert len(records) == 1
    assert records[0].next_train_batch is not None
    assert records[0].status == "finalized"
    control = capture_next_train_batch_evidence(control_batch)
    assert compare_next_train_batch_to_control(control, records[0]).ready is True

    changed_control = CorrectnessNextTrainBatchEvidence(
        batch_digest="changed",
        sample_ids_digest=control.sample_ids_digest,
        token_counts_digest=control.token_counts_digest,
    )
    assert (
        compare_next_train_batch_to_control(changed_control, records[0]).ready is False
    )
