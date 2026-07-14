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

import ast
from pathlib import Path
from typing import Any

import pytest


_COUNTER_FIELDS = (
    "full_cuda_graph_warmup_calls",
    "full_cuda_graph_capture_calls",
    "full_cuda_graph_replay_calls",
    "full_cuda_graph_reset_calls",
)
_VALIDATION_COUNTER_FIELDS = (
    "full_cuda_graph_validation_warmup_calls",
    "full_cuda_graph_validation_capture_calls",
    "full_cuda_graph_validation_replay_calls",
    "full_cuda_graph_validation_reset_calls",
)
_DIGEST_FIELD = "full_cuda_graph_storage_signature_sha256"


def _evidence(
    *,
    warmup: Any = 1,
    capture: Any = 1,
    replay: Any = 2,
    reset: Any = 0,
    digest: Any = "a" * 64,
) -> dict[str, Any]:
    return {
        _COUNTER_FIELDS[0]: warmup,
        _COUNTER_FIELDS[1]: capture,
        _COUNTER_FIELDS[2]: replay,
        _COUNTER_FIELDS[3]: reset,
        _DIGEST_FIELD: digest,
    }


def test_full_cuda_graph_evidence_tracker_preserves_exact_values_and_deltas() -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    first_metrics: dict[str, Any] = {"loss": 0.5}
    first = _evidence()
    train_results = {
        **first,
        "full_cuda_graph_rank_local_digest": "must-not-leak",
    }

    tracker.preserve(train_results, first_metrics)

    assert {field: first_metrics[field] for field in first} == first
    assert "full_cuda_graph_rank_local_digest" not in first_metrics
    assert {
        f"{field}_delta": first_metrics[f"{field}_delta"] for field in _COUNTER_FIELDS
    } == {
        "full_cuda_graph_warmup_calls_delta": 1,
        "full_cuda_graph_capture_calls_delta": 1,
        "full_cuda_graph_replay_calls_delta": 2,
        "full_cuda_graph_reset_calls_delta": 0,
    }
    assert all(type(first_metrics[field]) is int for field in _COUNTER_FIELDS)
    assert type(first_metrics[_DIGEST_FIELD]) is str

    second_metrics: dict[str, Any] = {"loss": 0.4}
    second = _evidence(warmup=1, capture=1, replay=5, reset=0)
    tracker.preserve(second, second_metrics)

    assert {field: second_metrics[field] for field in second} == second
    assert {
        f"{field}_delta": second_metrics[f"{field}_delta"] for field in _COUNTER_FIELDS
    } == {
        "full_cuda_graph_warmup_calls_delta": 0,
        "full_cuda_graph_capture_calls_delta": 0,
        "full_cuda_graph_replay_calls_delta": 3,
        "full_cuda_graph_reset_calls_delta": 0,
    }


def test_full_cuda_graph_evidence_tracker_preserves_validation_stage_deltas() -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    first = {
        **_evidence(),
        **dict(zip(_VALIDATION_COUNTER_FIELDS, (1, 0, 0, 0))),
    }
    first_metrics: dict[str, Any] = {}
    tracker.preserve(first, first_metrics)

    second = {
        **_evidence(replay=3),
        **dict(zip(_VALIDATION_COUNTER_FIELDS, (2, 1, 1, 0))),
    }
    second_metrics: dict[str, Any] = {}
    tracker.preserve(second, second_metrics)

    assert {
        field: second_metrics[field] for field in _VALIDATION_COUNTER_FIELDS
    } == dict(zip(_VALIDATION_COUNTER_FIELDS, (2, 1, 1, 0)))
    assert {
        f"{field}_delta": second_metrics[f"{field}_delta"]
        for field in _VALIDATION_COUNTER_FIELDS
    } == dict(
        zip(
            (f"{field}_delta" for field in _VALIDATION_COUNTER_FIELDS),
            (1, 1, 1, 0),
        )
    )


def test_full_cuda_graph_evidence_tracker_leaves_disabled_payload_unchanged() -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    metrics = {"loss": 0.5, "logprob_tokens_per_second": 123.0}
    original = metrics.copy()

    tracker.preserve({"loss": 0.5}, metrics)

    assert metrics == original
    assert all("full_cuda_graph" not in field for field in metrics)


@pytest.mark.parametrize(
    "trainer_path",
    ["GRPO", "async GRPO", "TQ/sync GRPO", "PPO"],
)
def test_policy_train_reducers_restore_evidence_after_ordinary_metric_collisions(
    trainer_path: str,
) -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    expected = _evidence(replay=4)
    malicious = {
        **{field: 999 for field in _COUNTER_FIELDS},
        **{f"{field}_delta": 999 for field in _COUNTER_FIELDS},
        _DIGEST_FIELD: "b" * 64,
    }
    metrics: dict[str, Any] = {"trainer_path": trainer_path}

    metrics.update(malicious)
    FullCudaGraphEvidenceTracker().preserve(expected, metrics)

    assert {field: metrics[field] for field in expected} == expected
    assert {
        f"{field}_delta": metrics[f"{field}_delta"] for field in _COUNTER_FIELDS
    } == {
        "full_cuda_graph_warmup_calls_delta": 1,
        "full_cuda_graph_capture_calls_delta": 1,
        "full_cuda_graph_replay_calls_delta": 4,
        "full_cuda_graph_reset_calls_delta": 0,
    }


@pytest.mark.parametrize(
    "train_results",
    [
        {"full_cuda_graph_warmup_calls": 1},
        _evidence(capture=True),
        _evidence(replay=-1),
        _evidence(digest="A" * 64),
    ],
)
def test_full_cuda_graph_evidence_tracker_rejects_partial_or_malformed_values(
    train_results: dict[str, Any],
) -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    with pytest.raises(ValueError, match="full-iteration CUDA graph"):
        FullCudaGraphEvidenceTracker().preserve(train_results, {})


@pytest.mark.parametrize(
    "second",
    [
        _evidence(replay=1),
        _evidence(digest="b" * 64),
        {},
    ],
)
def test_full_cuda_graph_evidence_tracker_rejects_unstable_run_evidence(
    second: dict[str, Any],
) -> None:
    from nemo_rl.algorithms.utils import FullCudaGraphEvidenceTracker

    tracker = FullCudaGraphEvidenceTracker()
    tracker.preserve(_evidence(), {})

    with pytest.raises(ValueError, match="full-iteration CUDA graph"):
        tracker.preserve(second, {})


def test_policy_train_reducers_wire_evidence_after_ordinary_metric_reduction() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_functions = {
        "grpo.py": {
            "grpo_train": {"rollout_metrics", "seq_logprob_error_metrics"},
            "async_grpo_train": {"rollout_metrics", "seq_logprob_error_metrics"},
        },
        "grpo_sync.py": {
            "grpo_train_sync": {"rollout_metrics", "seq_logprob_error_metrics"}
        },
        "sft.py": {"sft_train": set()},
        "ppo.py": {"ppo_train": {"rollout_metrics"}},
    }

    for filename, functions in expected_functions.items():
        source = (repo_root / "nemo_rl" / "algorithms" / filename).read_text()
        tree = ast.parse(source)
        function_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert source.count("FullCudaGraphEvidenceTracker()") == len(functions)

        for function_name, collision_sources in functions.items():
            function = function_nodes[function_name]
            preserve_calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "preserve"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "full_cuda_graph_evidence"
            ]
            assert len(preserve_calls) == 1
            preserve_at = preserve_calls[0].lineno

            ordinary_merges = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "metrics"
                and (
                    (
                        len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id in collision_sources
                    )
                    or (
                        len(node.args) == 1
                        and isinstance(node.args[0], ast.Subscript)
                        and isinstance(node.args[0].value, ast.Name)
                        and node.args[0].value.id == "train_results"
                    )
                )
            ]
            assert len(ordinary_merges) == len(collision_sources) + 1
            ordinary_boundary = max(node.lineno for node in ordinary_merges)
            assert ordinary_boundary < preserve_at

            state_mutations = [
                node.lineno
                for node in ast.walk(function)
                if (
                    (
                        isinstance(node, ast.AugAssign)
                        and isinstance(node.target, ast.Name)
                        and node.target.id in {"total_valid_tokens", "consumed_samples"}
                    )
                    or (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "mark_iteration"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "timeout"
                    )
                )
                and node.lineno > ordinary_boundary
            ]
            assert state_mutations
            assert preserve_at < min(state_mutations)

    combined_source = "".join(
        (repo_root / "nemo_rl" / "algorithms" / filename).read_text()
        for filename in expected_functions
    )
    assert "full_cuda_graph_logprob" not in combined_source
